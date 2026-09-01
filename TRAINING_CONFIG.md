# Guia de Configuracion de Entrenamiento

## Como usar

```bash
# 1. Generar el archivo de configuracion por defecto
python main.py --generate-config

# 2. Abrir training_config.json con cualquier editor de texto y modificar valores

# 3. Entrenar con la configuracion
python main.py --train --config training_config.json

# 4. (Opcional) Sobreescribir valores desde la linea de comandos
python main.py --train --config training_config.json --epochs 50
```

> **Nota:** Los argumentos de linea de comandos (`--epochs`, `--checkpoint-name`, etc.)
> tienen prioridad sobre los valores del JSON. Si en el JSON pones `"epochs": 30`
> pero ejecutas `--epochs 50`, se usaran 50 epochs.

---

## Archivo `training_config.json`

```json
{
  "training": {
    "epochs": 30,
    "checkpoint_name": "chat_model",
    "dataset_source": "dataset_cache"
  },
  "device": {
    "mode": "auto",
    "gpu_indices": null,
    "use_vulkan": false,
    "num_cores": 0,
    "num_threads": 0,
    "max_ram_fraction": 0.75,
    "max_ram_bytes": null
  },
  "thinking": {
    "enabled": true,
    "loss_weight": 1.0,
    "max_tokens": 64
  },
  "agent": {
    "enabled": false,
    "loss_weight": 1.0,
    "ratio": 0.3
  },
  "moe": {
    "enabled": false,
    "num_experts": 4,
    "top_k": 2,
    "load_balance_weight": 0.01,
    "freeze_attention": false
  },
  "mtp": {
    "enabled": false,
    "num_heads": 4,
    "loss_weight": 0.3
  },
  "draft": {
    "enabled": false,
    "num_layers": 2,
    "embed_size": 128,
    "hidden_size": 256,
    "n_head": 2,
    "kd_enabled": false,
    "kd_temperature": 2.0,
    "kd_loss_weight": 0.5,
    "kd_epochs": 10
  },
  "validation": {
    "split": 0.1,
    "batches": 0,
    "early_stopping_patience": 5
  },
  "scheduler": {
    "type": "cosine",
    "eta_min": 1e-6,
    "step_size": 0,
    "gamma": 0.5,
    "patience": 5,
    "factor": 0.5
  },
  "logging": {
    "metrics_csv": true,
    "statistics": false
  },
  "tensorboard": {
    "enabled": false,
    "log_dir": "runs",
    "comment": "",
    "freq": 1
  }
}
```

---

## Parametros detallados

### Parametros basicos

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `training.epochs` | int | `1` - `1000` | `30` | Numero total de epochs a entrenar. Un epoch = pasar por todo el dataset una vez. |
| `training.checkpoint_name` | string | cualquier texto | `"chat_model"` | Nombre del archivo de checkpoint. Se guardara como `checkpoints/{nombre}.pth`. |
| `training.dataset_source` | string | ruta valida | `"dataset_cache"` | Ruta al dataset preparado. Generalmente no cambiar. |

---

### Dispositivo y hardware

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `device.mode` | string | `"auto"`, `"cpu"`, `"gpu"`, `"cpu+gpu"` | `"auto"` | Donde ejecutar el entrenamiento. `"auto"` detecta automaticamente. |
| `device.gpu_indices` | array/null | `[0]`, `[0,1]`, `null` | `null` | Indices de GPUs a usar. `null` = auto. Para multi-GPU: `[0,1,2]`. |
| `device.use_vulkan` | bool | `true` / `false` | `false` | Usar Vulkan en vez de CUDA. Solo para GPUs AMD. |
| `device.num_cores` | int | `0` - `N` | `0` | Nucleos CPU a usar. `0` = auto (todos los disponibles). |
| `device.num_threads` | int | `0` - `N` | `0` | Hilos de ejecucion. `0` = auto. |
| `device.max_ram_fraction` | float | `0.1` - `1.0` | `0.75` | Maxima fraccion de RAM a usar. `0.75` = 75% de la RAM total. |
| `device.max_ram_bytes` | int/null | `1000000` - N / `null` | `null` | Limite fijo de RAM en bytes. `null` = usar fraccion. |

**Ejemplo:**
```json
{
  "device": {
    "mode": "gpu",
    "gpu_indices": [0],
    "max_ram_fraction": 0.8
  }
}
```

---

### Learning Rate y Scheduler

El learning rate (LR) es CUAN grande es cada paso de aprendizaje. Piensa en el LR como la "velocidad de aprendizaje" del modelo.

| Valor | Significado | Cuando usarlo |
|-------|-------------|---------------|
| `1e-3` (0.001) | Rapido | Datasets grandes (>10k muestras) |
| `1e-4` (0.0001) | Moderado | Default, funciona bien para la mayoria |
| `1e-5` (0.00001) | Lento | Fine-tuning de modelos pre-entrenados |
| `1e-6` (0.000001) | Muy lento | Ajustes minimos, ultimas epochs |

#### Que es un Scheduler?

Un scheduler es como el "freno" del learning rate. Si el LR es la velocidad, el scheduler decide CUANDO y COMO reducir esa velocidad para que el modelo converja mejor.

**Sin scheduler:** El LR se queda fijo todo el entrenamiento. El modelo aprende igual todo el rato, como un coche que va a la misma velocidad en autopista y en ciudad.

**Con scheduler:** El LR cambia a lo largo del entrenamiento. Empieza rapido para aprender lo basico, y luego se frena para afinar los detalles.

#### Tipos de Scheduler

| Tipo | Como funciona | Ventaja | Desventaja |
|------|---------------|---------|------------|
| **cosine** | Curva suave tipo coseno | Funciona bien siempre | No se adapta al progreso |
| **step** | Reduce cada N epochs | Facil de entender | Muy brusco si no se configura bien |
| **exponential** | Reduce multiplicativamente | Suave y predecible | Puede ser muy lento |
| **plateau** | Reduce si no hay mejora | Se adapta al progreso | Necesita validacion |
| **onecycle** | Sube y luego baja | Rapido y efectivo | Requiere saber cuantos epochs |

#### Como funciona cada uno (ejemplos simples)

**Cosine (default):**
```
LR: 0.001 -> 0.0009 -> 0.0007 -> 0.0004 -> 0.0001 -> 0.000001
     rapido --------> lento --------> muy lento
```
Como una campana: empieza alto, baja suavemente, se estabiliza.

**Step:**
```
LR: 0.001 -> 0.001 -> 0.001 -> 0.0005 -> 0.0005 -> 0.0005 -> 0.00025
     epoch 1-5: rapido    epoch 6-10: moderado    epoch 11+: lento
```
Como un escalera: se queda igual un rato, luego baja de golpe.

**Exponential:**
```
LR: 0.001 -> 0.0005 -> 0.00025 -> 0.000125 -> 0.0000625
     cada epoch se reduce a la mitad
```
Como una diagonal: baja siempre a la misma velocidad.

**Plateau:**
```
Val Loss: 3.7 -> 3.6 -> 3.5 -> 3.5 -> 3.5 -> 3.5 (5 epochs sin mejora)
LR:       0.001                            0.0005 (baja!)
```
Como un sensor: si detecta que no hay mejora, reduce el LR.

**OneCycle:**
```
LR: 0.0001 -> 0.0005 -> 0.001 -> 0.0005 -> 0.0001
     sube --------> maximo --------> baja
```
Como un ciclo: sube rapido, llega al maximo, y baja.

#### Parametros del Scheduler

| Parametro | Tipo | Default | Que controla | Ejemplo |
|-----------|------|---------|--------------|---------|
| `scheduler.type` | string | `"cosine"` | Tipo de scheduler | `"cosine"`, `"step"`, `"plateau"` |
| `scheduler.eta_min` | float | `1e-6` | LR minimo (cosine/onecycle) | No bajar de `0.000001` |
| `scheduler.step_size` | int | `0` (auto) | Cada cuantos epochs reduce (step) | `5` = cada 5 epochs |
| `scheduler.gamma` | float | `0.5` | Factor de reduccion (step/exponential) | `0.5` = reducir a la mitad |
| `scheduler.patience` | int | `5` | Epochs sin mejora antes de reducir (plateau) | `5` = esperar 5 epochs |
| `scheduler.factor` | float | `0.5` | Cuanto reducir (plateau) | `0.5` = reducir a la mitad |

#### Ejemplos de configuracion

**Para empezar (default):**
```json
{
  "scheduler": {
    "type": "cosine",
    "eta_min": 1e-6
  }
}
```
Funciona bien para la mayoria de casos. No tocar si no sabes que hacer.

**Para learning rate fijo (sin scheduler):**
No hay opcion "none", pero puedes poner gamma muy alto:
```json
{
  "scheduler": {
    "type": "step",
    "step_size": 1000,
    "gamma": 0.99
  }
}
```
El LR apenas cambia durante el entrenamiento.

**Para reducir solo si no hay mejora:**
```json
{
  "scheduler": {
    "type": "plateau",
    "patience": 5,
    "factor": 0.5
  }
}
```
Ideal si quieres que el modelo se adapte. Si el loss baja, el LR se mantiene. Si se estanca, el LR baja.

**Para entrenamiento rapido:**
```json
{
  "scheduler": {
    "type": "onecycle",
    "eta_min": 1e-6
  }
}
```
Sube el LR al inicio y luego baja. Bueno para entrenamientos cortos.

**Para step suave:**
```json
{
  "scheduler": {
    "type": "step",
    "step_size": 10,
    "gamma": 0.7
  }
}
```
Reduce el 30% cada 10 epochs. Mas suave que el default (gamma=0.5).

#### Que scheduler elegir?

| Situacion | Scheduler | Configuracion |
|-----------|-----------|---------------|
| No se que hacer | cosine | default |
| Quiero que se adapte | plateau | patience=5, factor=0.5 |
| Entrenamiento rapido | onecycle | default |
| Control total | step | step_size=5, gamma=0.5 |
| Fine-tuning | cosine | eta_min=1e-7 |

**Tip:** Si el loss no baja, prueba `plateau`. Si el modelo converge bien pero lento, prueba `onecycle`.

---

### Thinking (razonamiento)

Thinking es el modelo "razona" antes de responder, similar a como un humano piensa antes de hablar.

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `thinking.enabled` | bool | `true` / `false` | `true` | Habilitar datos de thinking en el entrenamiento. |
| `thinking.loss_weight` | float | `0.0` - `1.0` | `1.0` | Peso del loss para tokens de thinking. `0.5` = thinking pesa la mitad que la respuesta. |
| `thinking.max_tokens` | int | `16` - `256` | `64` | Maximo de tokens de thinking por muestra. |

**Que es thinking?**
```
<|problem|>Que es 2+2<|thinking|>Es una suma basica. 2+2 = 4.<|final|>4
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      Esto es "thinking" (razonamiento)
```

| thinking_loss_weight | Efecto |
|---------------------|--------|
| `0.0` | El modelo ignora si el thinking es correcto, solo aprende la respuesta |
| `0.5` | Balance: learning ponderado |
| `1.0` | Thinking y respuesta tienen el mismo peso |

---

### Agent (herramientas)

El sistema agente permite al modelo usar herramientas externas (calculadora, archivos, etc.).

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `agent.enabled` | bool | `true` / `false` | `false` | Habilitar datos de agente en el entrenamiento. |
| `agent.loss_weight` | float | `0.0` - `1.0` | `1.0` | Peso del loss para tokens de agente. |
| `agent.ratio` | float | `0.0` - `1.0` | `0.3` | Porcion del dataset que es agente. `0.3` = 30% agente, 70% normal. |

**Que es agent?**
```
<|problem|>Cual es la raiz cuadrada de 144?<|thinking|>Necesito calcular sqrt(144)<tool_call>
{"tool": "calculator", "input": "sqrt(144)"}</tool_call><|tool_result|>12.0<|final|>12.0
```

| agent_ratio | Efecto |
|-------------|--------|
| `0.0` | Sin datos agente |
| `0.1` - `0.3` | Poco agente (recomendado para empezar) |
| `0.5` | Balance 50/50 |
| `0.7` - `1.0` | Mucho agente (solo si el modelo ya es competente) |

---

### MoE (Mixture of Experts)

MoE usa multiples "expertos" especializados. En vez de un solo modelo grande, tiene varios pequenos y elige los mejores para cada tarea.

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `moe.enabled` | bool | `true` / `false` | `false` | Habilitar MoE. |
| `moe.num_experts` | int | `2` - `8` | `4` | Numero de expertos. Mas expertos = mas capacidad pero mas memoria. |
| `moe.top_k` | int | `1` - `moe.num_experts` | `2` | Cuantos expertos se activan por token. `2` = cada token usa 2 expertos. |
| `moe.load_balance_weight` | float | `0.0` - `1.0` | `0.01` | Peso de la loss de balance. Alto = expertos se usan parejo. |
| `moe.freeze_attention` | bool | `true` / `false` | `false` | Congelar capas de atencion durante MoE. |

**Concepto simple:**
- Imagina que tienes 4 profesores expertos en diferentes materias
- Para cada pregunta, elegir los 2 mejores profesores
- El modelo aprende A QUE profesor preguntar para cada tipo de pregunta

| moe_num_experts | moe_top_k | Efecto |
|-----------------|-----------||
| `4` | `1` | 4 expertos, solo 1 responde (especializacion maxima) |
| `4` | `2` | 4 expertos, 2 responden (balance entre especializacion y generalidad) |
| `4` | `3` | 4 expertos, 3 responden (casi como modelo normal) |
| `2` | `1` | 2 expertos, 1 responde (simple y rapido) |

---

### Draft Model (Speculative Decoding)

El draft model es un modelo pequeno que propone tokens candidatos para **speculative decoding**. El modelo principal verifica todos en un solo forward pass, logrando **1.5x-3x de throughput** sin perdida de calidad (lossless).

**Concepto simple:**
- El draft model "adivina" los proximos N tokens rapidamente
- El modelo target verifica todos de golpe
- Si K de N se aceptan, obtienes K tokens en 1 paso en vez de K pasos
- Si algun token falla, se descarta y se usa la correccion del target

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `draft.enabled` | bool | `true` / `false` | `false` | Habilitar creacion de draft model despues del entrenamiento principal. |
| `draft.num_layers` | int | `1` - `4` | `2` | Capas del draft. Target tiene 4; draft debe ser menor. |
| `draft.embed_size` | int | `64` - `256` | `128` | Tamanio de embedding. Target tiene 256; draft debe ser menor. |
| `draft.hidden_size` | int | `128` - `512` | `256` | Tamanio del hidden layer. Target tiene 512. |
| `draft.n_head` | int | `1` - `4` | `2` | Cabezas de atencion. Target tiene 4. |
| `draft.kd_enabled` | bool | `true` / `false` | `false` | Entrenar con Knowledge Distillation usando el target como teacher. |
| `draft.kd_temperature` | float | `1.0` - `5.0` | `2.0` | Temperature para soft labels del teacher. Mayor = mas suave. |
| `draft.kd_loss_weight` | float | `0.0` - `1.0` | `0.5` | Peso del KD loss vs next-token loss. `0.5` = balance. |
| `draft.kd_epochs` | int | `1` - `100` | `10` | Epochs de entrenamiento del draft (despues del training principal). |

**Relacion de tamanio tipica:**

| Target | Draft | Ratio | Acceptance rate esperado |
|--------|-------|-------|--------------------------|
| 4 capas, embed=256 | 2 capas, embed=128 | ~1/4 | 60-70% |
| 4 capas, embed=256 | 1 capa, embed=64 | ~1/16 | 50-60% |

**Que es Knowledge Distillation (KD)?**
```
Target model:  P("hola" | input) = 0.8,  P("que" | input) = 0.15
                      ↓ soft labels (temperature=2.0)
Draft model:   aprende a imitar esas probabilidades suaves
                      ↓ en vez de solo aprender el token correcto
Resultado:     draft model predice mejor porque entiende la "duda" del target
```

| kd_temperature | Efecto |
|----------------|--------|
| `1.0` | Distribuciones agudas (casi one-hot) |
| `2.0` | Balance (recomendado) |
| `5.0` | Distribuciones muy suaves (draft aprende mas generalidad) |

| kd_loss_weight | Efecto |
|----------------|--------|
| `0.0` | Solo next-token prediction (sin KD) |
| `0.5` | Balance entre KD y hard labels (recomendado) |
| `1.0` | Solo KD (draft solo imita al target) |

**Archivos generados:**
```
checkpoints/
├── chat_model.pth                      # Modelo target
└── chat_model_draft.pth                # Modelo draft (speculative decoding)
```

**Exportar a GGUF:**
```bash
# Exportar target (genera directorio HF + scripts de conversion)
python main.py --export chat_model --formats gguf

# El draft se exporta automaticamente si existe
# Genera: models/exported/chat_model_hf/ y models/exported/chat_model_draft_hf/
```

**Uso con llama.cpp:**
```bash
# Compilar llama.cpp (requiere build 9200+ para soporte MTP)
# Luego ejecutar con draft model:
llama-server \
  -m target.gguf \
  -md draft.gguf \
  --spec-type draft-simple \
  --spec-draft-n-max 5 \
  --port 8080
```

| llama.cpp flag | Descripcion |
|----------------|-------------|
| `-m` / `--model` | Modelo target (principal) |
| `-md` / `--model-draft` | Modelo draft (pequeno) |
| `--spec-type` | Tipo de speculative: `draft-simple`, `draft-mtp`, `ngram-simple` |
| `--spec-draft-n-max` | Maximo de tokens a proponer por paso (default: 3) |
| `--spec-draft-n-min` | Minimo de tokens a proponer (default: 0) |

**Ejemplo con MTP + Draft:**
```bash
# Si el modelo tiene MTP habilitado, tambien puedes usar draft-mtp
llama-server \
  -m target_with_mtp.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --port 8080
# No necesita -md separado porque MTP esta embebido en el target
```

---

### Validacion y Early Stopping

La validacion verifica si el modelo generaliza bien con datos que NO vio durante el entrenamiento.

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `validation.split` | float | `0.0` - `0.5` | `0.1` | Porcion del dataset para validacion. `0.1` = 10%. `0` = sin validacion. |
| `validation.batches` | int | `0` - `N` | `0` | Maximo de batches de validacion por epoch. `0` = todos. |
| `validation.early_stopping_patience` | int | `0` - `N` | `5` | Parar si val_loss no mejora en N epochs. `0` = deshabilitado. |

**Como funciona early stopping:**
```
Epoch 1: val_loss = 3.72 (mejor) -> patience = 0
Epoch 2: val_loss = 3.68 (mejor) -> patience = 0
Epoch 3: val_loss = 3.68 (igual) -> patience = 1
Epoch 4: val_loss = 3.68 (igual) -> patience = 2
Epoch 5: val_loss = 3.68 (igual) -> patience = 3
Epoch 6: val_loss = 3.69 (peor)  -> patience = 4
Epoch 7: val_loss = 3.69 (peor)  -> patience = 5 -> STOP!
```

| val_split | Efecto |
|-----------|--------|
| `0.0` | Sin validacion (no recomendado) |
| `0.05` - `0.1` | Poco datos para val (datasets grandes) |
| `0.2` - `0.3` | Bastante datos para val (datasets pequenos) |
| `0.5` | Mitad y mitad (solo si tienes MUCHOS datos) |

---

### TensorBoard

TensorBoard permite visualizar el progreso del entrenamiento en tiempo real con graficas interactivas.

**Instalar TensorBoard (opcional):**
```bash
pip install tensorboard
```

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `tensorboard.enabled` | bool | `true` / `false` | `false` | Activar logging a TensorBoard |
| `tensorboard.log_dir` | string | ruta valida | `"runs"` | Directorio raiz donde se guardan los logs |
| `tensorboard.comment` | string | cualquier texto | `""` | Sufijo opcional para el nombre del run |
| `tensorboard.freq` | int | `1` - `N` | `1` | Loguear cada N epochs (1 = cada epoch) |

**Metricas registradas:**
- `loss/train`, `loss/val` - Loss de entrenamiento y validacion
- `perplexity/train`, `perplexity/val` - Perplejidad
- `gap` - Diferencia val_loss - train_loss
- `learning_rate` - Learning rate actual
- `grad_norm` - Norma del gradiente
- `tokens_per_sec` - Velocidad de entrenamiento
- `best_loss` - Mejor loss alcanzado
- `thinking/*` - Metricas de thinking (si enabled)
- `agent/*` - Metricas de agente (si enabled)
- `moe/*` - Metricas de MoE (si enabled)
- `mtp/*` - Metricas de MTP (si enabled)

**Ejemplo JSON:**
```json
{
  "tensorboard": {
    "enabled": true,
    "log_dir": "runs",
    "comment": "mi_experimento",
    "freq": 1
  }
}
```

**Ejemplo CLI:**
```bash
python main.py --train --tensorboard-enabled --tensorboard-comment "test_run"

# Ver TensorBoard en el navegador:
tensorboard --logdir=runs
```

**Estructura de archivos generada:**
```
runs/
└── chat_model_20260830_120000_test_run/
    ├── events.out.tfevents.*    # Metricas de TensorBoard
    └── ...
```

---

### Metricas y reporting

| Parametro | Tipo | Valores | Default | Descripcion |
|-----------|------|---------|---------|-------------|
| `log_metrics_csv` | bool | `true` / `false` | `true` | Guardar metricas por epoch en CSV. |

**Archivos generados:**
```
checkpoints/
├── chat_model.pth                      # Modelo actualizado cada epoch
├── chat_model_epoch_N_TIMESTAMP.pth    # Backup del mejor modelo por epoch
├── chat_model_state.json               # Estado del ultimo epoch
├── chat_model_metrics.csv              # Metricas por epoch
└── chat_model_metrics_report.html      # Reporte HTML con graficas
```

---

### Metricas del reporte HTML

El reporte incluye graficas Chart.js con:

| Seccion | Metricas | Que significa |
|---------|----------|---------------|
| **Loss** | train_loss, val_loss | Error del modelo (menor = mejor) |
| **Perplexity** | train_perplexity, val_perplexity | `e^loss` - cuan "seguro" esta el modelo |
| **Gap** | val_loss - train_loss | Negativo = buena generalizacion |
| **Learning Rate** | lr | Tamanio del paso de aprendizaje |
| **Speed** | tokens/segundo | Velocidad de entrenamiento |
| **Thinking** | accuracy, coverage | Que tan bien genera razonamiento |
| **Agent** | tool_call_acc, ratio | Que tan bien usa herramientas |
| **MoE** | entropy, utilization | Balance de expertos |

**Indicadores de estado:**
- **OK** (healthy): Loss bajo >10% o accuracy > umbral
- **WARNING** (overfitting): Loss subio >10% o accuracy < umbral
- **INFORMATION** (stable): Loss cambio <10% (meseta)

El reporte incluye un banner de **Google Translate** en la esquina superior derecha para traducir el reporte a cualquier idioma.

---

## Ejemplos de configuracion

### Entrenamiento basico (recomendado para empezar)
```json
{
  "training": {
    "epochs": 30
  },
  "validation": {
    "split": 0.1,
    "early_stopping_patience": 5
  },
  "thinking": {
    "enabled": true
  },
  "agent": {
    "enabled": false
  },
  "moe": {
    "enabled": false
  }
}
```

### Entrenamiento avanzado con MoE
```json
{
  "training": {
    "epochs": 50
  },
  "validation": {
    "split": 0.15,
    "early_stopping_patience": 10
  },
  "thinking": {
    "enabled": true,
    "loss_weight": 0.5
  },
  "agent": {
    "enabled": true,
    "ratio": 0.3
  },
  "moe": {
    "enabled": true,
    "num_experts": 4,
    "top_k": 2
  }
}
```

### Fine-tuning rapido
```json
{
  "training": {
    "epochs": 10
  },
  "validation": {
    "split": 0.05,
    "early_stopping_patience": 3
  },
  "thinking": {
    "enabled": true,
    "loss_weight": 0.3
  },
  "agent": {
    "enabled": false
  },
  "moe": {
    "enabled": false
  }
}
```

### Entrenamiento con Draft Model (Knowledge Distillation)
```json
{
  "training": {
    "epochs": 30,
    "checkpoint_name": "mi_modelo"
  },
  "validation": {
    "split": 0.1,
    "early_stopping_patience": 5
  },
  "thinking": {
    "enabled": true
  },
  "moe": {
    "enabled": false
  },
  "mtp": {
    "enabled": false
  },
  "draft": {
    "enabled": true,
    "num_layers": 2,
    "embed_size": 128,
    "hidden_size": 256,
    "n_head": 2,
    "kd_enabled": true,
    "kd_temperature": 2.0,
    "kd_loss_weight": 0.5,
    "kd_epochs": 10
  }
}
```

### Entrenamiento con Draft Model (sin KD)
```json
{
  "training": {
    "epochs": 30
  },
  "draft": {
    "enabled": true,
    "num_layers": 2,
    "embed_size": 128,
    "kd_enabled": false,
    "kd_epochs": 15
  }
}
```

---

## Errores comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| Loss no baja | LR muy bajo o muy alto | Probar `1e-3` o `1e-4` |
| Loss oscila mucho | LR muy alto | Bajar LR a `1e-4` o `1e-5` |
| Val loss sube, train baja | Overfitting | Aumentar `val_split`, reducir `early_stopping_patience` |
| Entrenamiento muy lento | Demasiados expertos MoE | Reducir `moe_num_experts` o deshabilitar MoE |
| Memory error | Dataset o modelo muy grande | Reducir `max_ram_fraction`, usar `device_mode: cpu+gpu` |
| Epochs no correlativos | Archivo state corrupto | Verificar `checkpoints/{name}_state.json` |
| Draft loss no baja | KD temperature muy alta | Reducir `kd_temperature` a 1.5-2.0 |
| Draft muy lento | Draft model muy grande | Reducir `draft_num_layers` o `draft_embed_size` |
| Draft no se genera | `draft_enabled=false` | Poner `draft.enabled=true` en JSON o `--draft-enabled` en CLI |
