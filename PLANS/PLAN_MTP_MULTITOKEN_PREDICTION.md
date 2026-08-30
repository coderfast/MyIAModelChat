# Plan: Multi-Token Prediction (MTP) para MyIAModelChat

**Fecha**: 2026-08-30
**Estado**: DONE - Implementado y verificado
**Prioridad**: Alta
**Complejidad**: Media

---

## 1. Concepto

**Multi-Token Prediction** entrena al modelo para predecir los proximos *k* tokens simultaneamente en cada posicion, en lugar de solo el siguiente token (NTP). Esto:

- **Mejora el razonamiento**: El modelo "piensa hacia adelante" (Gloeckle et al., 2024 - Meta FAIR)
- **Acelera inferencia**: Los heads extra permiten *speculative decoding* (hasta 3x mas rapido)
- **Aumenta densidad de gradientes**: Senal de entrenamiento mas rica por posicion
- **Adoptado por**: DeepSeek-V3, Qwen3, Nemotron 3, MiMo-V2-Flash

### Arquitectura Meta/FAIR (la mas adecuada para este proyecto)

```
Shared Transformer Backbone
    ├── lm_head (prediccion token t+1) → PRIMARY
    ├── MTP Head 1 (prediccion token t+2)
    ├── MTP Head 2 (prediccion token t+3)
    └── MTP Head 3 (prediccion token t+4)
```

Cada MTP head: `Linear(hidden → hidden) → GELU → Linear(hidden → vocab)`

### Referencias

| Paper | Ano | Institucion | Relevancia |
|-------|-----|-------------|------------|
| Better & Faster LLMs via Multi-Token Prediction | 2024.04 | Meta FAIR | Fundacional - arquitectura de heads paralelos |
| DeepSeek-V3 Technical Report | 2024.12 | DeepSeek | Produccion - MTP como aux loss + speculative decoding |
| Pre-Training Curriculum for MTP | 2025 | ACL 2025 | Estrategia de entrenamiento curriculum |
| How Transformers Learn to Plan via MTP | 2026.04 | UCLA/Shanghai | Teoria - reverse reasoning por MTP |
| LoopMTP | 2026.08 | Arxiv | Looped transformer + latent MTP |

---

## 2. Viabilidad: ALTA

La arquitectura del proyecto es ideal para MTP:

| Aspecto | Evaluacion |
|---------|------------|
| **Modelo base** | `GPT2LMHeadModel` - simple, flexible. MTP se anade como extension (igual que MoE) |
| ** Patron de extension** | `ChatModelMoE` ya demuestra el patron: subclasear ChatModel, anadir componentes, modificar forward |
| **Perdida** | `_compute_loss()` ya maneja pesos por posicion. MTP requiere offsets de targets (shift +1, +2, +3) |
| **Dataset** | `token_pair_generator()` produce `(t[:-1], t[1:])` - compatible sin cambios. Cada MTP head usa el mismo sequence con offsets diferentes |
| **Checkpoints** | Guardan `architecture` dict - facil anadir campos MTP |
| **Inferencia** | Solo necesita el `lm_head` principal - sin cambios a `model.generate()` |
| **MoE+MTP** | Compatibles: MoE reemplaza FFN, MTP anade output heads. Son ortogonales |

**Riesgo: BAJO** - Sigue exactamente el patron de MoE que ya funciona.

---

## 3. Archivos a Crear

### 3.1. `commons/model/chatmodel_mtp.py` (~120 lineas)

Modelo MTP que extiende ChatModel con heads de prediccion multi-token.

**Componentes:**

- `MTPHead(nn.Module)`: Head individual para predecir el k-esimo token futuro
  - `projection`: `nn.Linear(hidden_size, hidden_size)`
  - `output_head`: `nn.Linear(hidden_size, vocab_size)`
  - Forward: `output_head(gelu(projection(hidden_states)))`

- `ChatModelMTP(ChatModel)`: Modelo principal con MTP
  - `mtp_heads`: `nn.ModuleList` con `(mtp_num_heads - 1)` MTP heads
  - `forward(input_ids)`:
    1. Obtener hidden states del transformer backbone
    2. `primary_logits = model.lm_head(hidden)` (token t+1)
    3. `mtp_logits = [head(hidden) for head in mtp_heads]` (tokens t+2, t+3, ...)
    4. Retornar `(primary_logits, mtp_logits)`
  - `get_mtp_head_accuracies(predictions, targets)`: Metricas por head

**Parametros adicionales estimados** (embed_size=256, vocab_size=8000, mtp_num_heads=4):
- 3 MTP heads x (256x256 + 256x8000) = ~6.3M params
- Overhead: ~15-20% mas de parametros, ~5-10% mas de tiempo de entrenamiento

### 3.2. `tests/test_chatmodel_mtp.py` (~150 lineas)

Tests unitarios:
- Test MTPHead output shape
- Test ChatModelMTP forward retorna tuple (primary, mtp_list)
- Test MTP loss computation con offsets correctos
- Test compatibilidad MoE + MTP
- Test checkpoint save/load con MTP
- Test inference usa solo primary head

---

## 4. Archivos a Modificar

### 4.1. `training/trainer.py`

#### TrainingConfig (lineas 136-283)

Anadir campos despues de los campos MoE:

```python
# MTP (Multi-Token Prediction) fields
mtp_enabled: bool = False
mtp_num_heads: int = 4          # Predecir hasta 4 tokens adelante
mtp_loss_weight: float = 0.3    # Peso de la loss auxiliar MTP
```

Anadir al `_JSON_TO_FIELD` mapping:

```python
'mtp.enabled': 'mtp_enabled',
'mtp.num_heads': 'mtp_num_heads',
'mtp.loss_weight': 'mtp_loss_weight',
```

#### _compute_loss() (lineas 851-1020)

Modificacion principal - detectar output MTP y calcular loss auxiliar:

```python
# Despurs de model(inputs):
outputs = model(inputs)
mtp_logits_list = None
if isinstance(outputs, tuple):
    if len(outputs) == 2 and isinstance(outputs[1], list):
        # MTP model: (primary_logits, [mtp_head_logits...])
        outputs, mtp_logits_list = outputs
    elif len(outputs) == 2:
        # MoE model: (logits, gate_scores)
        outputs, gate_scores = outputs
raw_outputs = outputs

# ... existing loss computation con primary outputs ...

# MTP loss: cada head k predice token en posicion t+k+2
if mtp_logits_list and getattr(self.config, 'mtp_enabled', False):
    mtp_total_loss = torch.tensor(0.0, device=loss.device)
    mtp_head_count = 0
    for k, mtp_logits in enumerate(mtp_logits_list):
        shift = k + 2  # head 0 → t+2, head 1 → t+3, ...
        if targets.size(1) > shift:
            mtp_preds = mtp_logits[:, :-shift].contiguous().view(-1, mtp_logits.size(-1))
            mtp_targets = targets[:, shift:].contiguous().view(-1)
            non_pad = mtp_targets.ne(self.tokenizer.get_pad_index())
            if non_pad.any():
                mtp_loss_k = criterion(mtp_preds[non_pad], mtp_targets[non_pad])
                mtp_total_loss = mtp_total_loss + mtp_loss_k
                mtp_head_count += 1
    if mtp_head_count > 0:
        loss = loss + self.config.mtp_loss_weight * (mtp_total_loss / mtp_head_count)
```

#### performMainTrain() (lineas 2260-2970)

**Instanciar modelo** (lineas ~2417-2475):

Anadir deteccion de MTP en checkpoints:

```python
# Detectar MTP de keys del state_dict
checkpoint_has_mtp = any('mtp_heads' in k for k in sd_keys)
use_mtp = checkpoint_has_moe or self.config.mtp_enabled

if use_moe and use_mtp:
    model = ChatModelMoEMTP(...)  # Combinado
elif use_mtp:
    model = ChatModelMTP(...)
elif use_moe:
    model = ChatModelMoE(...)
else:
    model = ChatModel(...)
```

**Metricas** (lineas ~2629-2647):

Anadir metricas MTP:

```python
mtp_metrics = {}
if self.config.mtp_enabled and hasattr(model, 'get_mtp_head_accuracies'):
    mtp_metrics = model.get_mtp_head_accuracies(predictions, targets)
```

**Checkpoint architecture dict** (lineas ~2690-2713):

Anadir campos MTP:

```python
'architecture': {
    ...existing fields...,
    'mtp_enabled': self.config.mtp_enabled,
    'mtp_num_heads': self.config.mtp_num_heads,
    'mtp_loss_weight': self.config.mtp_loss_weight,
}
```

**CSV metrics** (lineas ~2731-2761):

Anadir columnas:

```python
'mtp_loss': f"{mtp_metrics.get('mtp_loss', 0):.6f}" if mtp_metrics else "",
'mtp_head_0_acc': f"{mtp_metrics.get('mtp_head_0_accuracy', 0):.4f}" if mtp_metrics else "",
'mtp_head_1_acc': f"{mtp_metrics.get('mtp_head_1_accuracy', 0):.4f}" if mtp_metrics else "",
'mtp_head_2_acc': f"{mtp_metrics.get('mtp_head_2_accuracy', 0):.4f}" if mtp_metrics else "",
```

**HTML Report** (~lineas 1492-2195):

Anadir seccion MTP al reporte HTML con graficas:
- MTP Loss por epoch
- MTP Head Accuracy por head

### 4.2. `main.py`

#### CLI Args (lineas ~416-434)

Anadir despues de los args MoE:

```python
# MTP (Multi-Token Prediction) configuration
parser.add_argument("--mtp-enabled", action='store_true',
                    help="Enable Multi-Token Prediction training")
parser.add_argument("--mtp-num-heads", type=int, default=4,
                    help="Number of MTP heads (predict N tokens ahead, default: 4)")
parser.add_argument("--mtp-loss-weight", type=float, default=0.3,
                    help="Weight for MTP auxiliary loss (default: 0.3)")
```

#### TrainingConfig constructor (lineas ~681-713)

Anadir campos MTP en ambas ramas (con config JSON y sin el):

```python
mtp_enabled=getattr(args, 'mtp_enabled', False),
mtp_num_heads=getattr(args, 'mtp_num_heads', 4),
mtp_loss_weight=getattr(args, 'mtp_loss_weight', 0.3),
```

### 4.3. `inference/chat_engine.py`

#### Model loading (lineas ~189-200)

Detectar MTP en checkpoint y instanciar modelo correcto:

```python
# Detect MTP from state_dict keys
checkpoint_has_mtp = any('mtp_heads' in k for k in state_dict.keys())

if arch.get('moe_enabled', False):
    if checkpoint_has_mtp:
        # TODO: ChatModelMoEMTP when implemented
        self.model = ChatModelMoE(...)
    else:
        self.model = ChatModelMoE(...)
elif checkpoint_has_mtp:
    from commons.model.chatmodel_mtp import ChatModelMTP
    self.model = ChatModelMTP(self.tokenizer,
        embed_size=arch.get('embed_size', 256),
        num_layers=arch.get('num_layers', 4),
        mtp_num_heads=arch.get('mtp_num_heads', 4),
        mtp_loss_weight=arch.get('mtp_loss_weight', 0.3))
else:
    self.model = ChatModel(...)
```

**Nota**: Para inferencia, `model.generate()` de HuggingFace usa solo `lm_head`. No se necesitan cambios en la logica de generacion.

### 4.4. `commons/model/__init__.py`

Anadir export:

```python
from .chatmodel_mtp import ChatModelMTP
```

### 4.5. `AGENTS.md`

Anadir documentacion en:
- Seccion "Key Components" → nueva fila en tabla de modulos
- Seccion "Architecture Patterns" → MTP pattern
- Seccion "Training Commands" → ejemplos con `--mtp-enabled`
- Seccion "Recent Improvements" → MTP feature

---

## 5. Diagrama de Dependencias

```
commons/model/chatmodel.py          (BASE - sin cambios)
commons/model/chatmodel_moe.py      (EXISTENTE - sin cambios)
        │
        ▼
commons/model/chatmodel_mtp.py      (NUEVO)
    ├── ChatModelMTP(ChatModel)
    │   ├── MTPHead × (num_heads-1)
    │   └── forward() → (primary_logits, [mtp_logits...])
    │
    ▼
training/trainer.py                 (MODIFICADO)
    ├── TrainingConfig: +mtp_enabled, +mtp_num_heads, +mtp_loss_weight
    ├── _compute_loss(): detectar tuple output, calcular MTP loss
    ├── performMainTrain(): instanciar ChatModelMTP
    └── Metricas CSV + HTML report
    │
    ▼
main.py                             (MODIFICADO)
    └── CLI args: --mtp-enabled, --mtp-num-heads, --mtp-loss-weight
    │
    ▼
inference/chat_engine.py            (MODIFICADO)
    └── Detectar MTP en checkpoint, instanciar ChatModelMTP
```

---

## 6. Orden de Implementacion

| Paso | Archivo | Descripcion | Dependencias | Estado |
|------|---------|-------------|--------------|--------|
| 1 | `commons/model/chatmodel_mtp.py` | Crear modelo MTP | Ninguna | DONE |
| 2 | `tests/test_chatmodel_mtp.py` | Tests unitarios del modelo | Paso 1 | DONE (13/13) |
| 3 | `training/trainer.py` | Config + loss + metricas + instanciacion | Paso 1 | DONE |
| 4 | `main.py` | CLI args | Paso 3 | DONE |
| 5 | `inference/chat_engine.py` | Carga de modelo MTP | Paso 1 | DONE |
| 6 | `commons/model/__init__.py` | Export | Paso 1 | DONE |
| 7 | `AGENTS.md` | Documentacion | Todos | DONE |
| 8 | `commons/model/chatmodel_moe_mtp.py` | MoE+MTP combinado | Paso 1 | DONE |
| 9 | `commons/registry/model_export.py` | Adaptar GGUF/ONNX export | Paso 1 | DONE |
| 10 | `training_config.json` | Seccion MTP | Paso 3 | DONE |

---

## 7. Comandos de Uso

```bash
# Preparar datos
python main.py --prepare-data --aiml

# Entrenar con MTP
python main.py --train --epochs 30 --mtp-enabled --mtp-num-heads 4

# Entrenar con MTP + MoE (combinado)
python main.py --train --epochs 30 --moe-enabled --mtp-enabled

# Entrenar con MTP + validacion + early stopping
python main.py --train --epochs 30 --mtp-enabled --val-split 0.1 --early-stopping-patience 5

# Chat (usa solo lm_head principal, sin cambios)
python main.py --chat --model chat_model

# Tests
pytest tests/test_chatmodel_mtp.py -v
```

---

## 8. Compatibilidad

### Backward Compatibility

- **Checkpoints viejos**: Funcionan sin problemas. `mtp_enabled=False` por defecto.
- **Sin MTP**: El modelo se comporta exactamente igual al actual.
- **MoE existente**: No se afecta. MTP es una feature adicional.

### Forward Compatibility

- **Speculative decoding**: Futura fase para acelerar inferencia usando MTP heads como draft model.
- **Curriculum training**: Empezar con k=1, aumentar gradualmente a k=max.
- **MoE + MTP combinado**: Soportado desde el inicio (features ortogonales).

---

## 9. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|-------------|---------|------------|
| Aumento de memoria | Baja | Medio | MTP heads ligeros (~6M params). Configurable. |
| Mayor tiempo de entrenamiento | Baja | Bajo | ~5-10% overhead. Loss auxiliar simple. |
| Incompatibilidad con checkpoints | Baja | Bajo | Deteccion automatica via state_dict keys. |
| Complejidad de debug | Media | Bajo | Metricas por head. Tests unitarios completos. |
| Overfitting con MTP | Media | Medio | Curriculum training. Validacion por head. |

---

## 10. Metricas Esperadas

### Metricas por Epoch

- `mtp_loss`: Loss auxiliar promedio de todos los MTP heads
- `mtp_head_0_accuracy`: Accuracy del head que predice t+2
- `mtp_head_1_accuracy`: Accuracy del head que predice t+3
- `mtp_head_2_accuracy`: Accuracy del head que predice t+4

### Metricas de Rendimiento

- **Training overhead**: ~5-10% mas tiempo por epoch
- **Memoria**: ~15-20% mas parametros (~6M para embed_size=256)
- **Inference**: Sin cambio (usa solo lm_head principal)

---

## 11. Fases Futuras

### Fase 2: Speculative Decoding

Usar MTP heads como draft model para acelerar inferencia:
1. MTP heads generan tokens candidatos rapidamente
2. Modelo principal verifica y acepta/rechaza
3. Speedup esperado: 2-3x

### Fase 2b: Curriculum Training

Entrenamiento progresivo de MTP heads:
1. Epochs 1-10: Solo lm_head (k=1)
2. Epochs 11-20: Anadir head 0 (k=2)
3. Epochs 21-30: Anadir head 1 (k=3)
4. Epochs 31+: Anadir head 2 (k=4)

### Fase 3: MoE + MTP Combinado

Modelo `ChatModelMoEMTP` que combina:
- MoE en FFN layers (especializacion de expertos)
- MTP heads (prediccion multi-token)
- Arquitectura completa para razonamiento avanzado

---

## 11. Verificacion de Implementacion

### Tests Unitarios
```bash
pytest tests/test_chatmodel_mtp.py -v
# 13/13 tests passing:
# - TestMTPHead: output_shape, parameters_exist
# - TestChatModelMTP: forward_returns_tuple, output_shapes, mtp_num_heads_configurable, gradient_flow, get_mtp_params, mtp_loss_computation
# - TestChatModelMoEMTP: forward_returns_tuple, output_shapes, has_both_moe_and_mtp, gradient_flow, mtp_params_count
```

### Integration Test
```bash
python main.py --train --cpu --mtp-enabled --mtp-num-heads 4 --mtp-loss-weight 0.3 --epochs 2 --checkpoint-name mtp_test
# Successfully completed 2 epochs with MTP loss reported
```

### JSON Config
```bash
python -c "from training.trainer import TrainingConfig; c = TrainingConfig(); c.save_json('test.json')"
# Produces JSON with mtp section: { "mtp": { "enabled": false, "num_heads": 4, "loss_weight": 0.3 } }
```

### Files Modified/Created
- `commons/model/chatmodel_mtp.py` (CREATED) - MTPHead + ChatModelMTP
- `commons/model/chatmodel_moe_mtp.py` (CREATED) - ChatModelMoEMTP
- `tests/test_chatmodel_mtp.py` (CREATED) - 13 unit tests
- `training/trainer.py` (MODIFIED) - Config, loss, metrics, CSV, HTML
- `main.py` (MODIFIED) - CLI args
- `inference/chat_engine.py` (MODIFIED) - MTP/MoE+MTP loading
- `commons/model/__init__.py` (MODIFIED) - Exports
- `commons/registry/model_export.py` (MODIFIED) - GGUF/ONNX MTP support
- `training_config.json` (MODIFIED) - MTP section added
- `AGENTS.md` (MODIFIED) - MTP documentation

---

*Plan creado: 2026-08-30*
*Implementado: 2026-08-30*
*Ultima actualizacion: 2026-08-30*
