# IQ Quantization Implementation Roadmap — COMPLETED

**Objetivo**: Implementar cuantización forward para todos los tipos IQ en Python puro.

**Estado actual**: ✅ COMPLETADO — Todos los 9 tipos IQ implementados, testeados y exportando GGUF correctamente.

**Fecha de finalización**: 2026-08-05

---

## Resumen de Implementación

Todos los tipos IQ fueron implementados en `models/exported/gguf_quantizer.py` con las siguientes características:

| Tipo | Estado | Tamaño | Test Export |
|------|--------|--------|-------------|
| IQ4_NL | ✅ | 18 bytes/block | ✅ OK |
| IQ4_XS | ✅ | 136 bytes/block | ✅ OK |
| IQ2_XXS | ✅ | 66 bytes/block | ✅ OK |
| IQ2_XS | ✅ | 74 bytes/block | ✅ OK |
| IQ2_S | ✅ | 82 bytes/block | ✅ OK |
| IQ3_XXS | ✅ | 98 bytes/block | ✅ OK |
| IQ3_S | ✅ | 110 bytes/block | ✅ OK |
| IQ1_S | ✅ | 50 bytes/block | ✅ OK |
| IQ1_M | ✅ | 56 bytes/block | ✅ OK |

**Archivos modificados:**
- `models/exported/gguf_quantizer.py` — Clases de cuantización IQ
- `models/exported/convert_gguf.py` — Mapeo de tipos IQ
- `main.py` — Choices para `--quantization`
- `models/exported/GGUF_QUANTIZATION_README.md` — Documentación actualizada

---

## Orden de Implementación Original

### Fase 1: IQ4 (sin grilla) — 2 horas

#### 1.1 IQ4_NL — El más simple
- **Block**: 32, **Type size**: 18 bytes
- **Grilla**: NO usa grilla. Usa tabla `kvalues` de 16 entradas
- **Fórmula**: `q = kvalues[argmin(|x - kvalues|)]`
- **Algoritmo**:
  1. Dividir en bloques de 32
  2. Encontrar escala `d = max(|block|) / 127`
  3. Para cada elemento: buscar valor más cercano en `kvalues`
  4. Empaquetar 4-bit indices (2 por byte)
- **Tests**: Comparar dequantización round-trip (quantize → dequantize → diff)
- **Código estimado**: 40-50 líneas

#### 1.2 IQ4_XS — Con escalas por super-block
- **Block**: 256, **Type size**: 136 bytes
- **Grilla**: Misma `kvalues` de IQ4_NL
- **Diferencia**: Escalas por sub-block de 32 (8 por super-block)
- **Algoritmo**:
  1. Dividir super-block en 8 sub-blocks de 32
  2. Calcular escala por sub-block (6-bit signed: scales_l | (scales_h << 4) - 32)
  3. Cuantizar cada sub-block con su escala
  4. Empaquetar escalas (16 bytes) + indices (128 bytes)
- **Tests**: Round-trip + verificar tamaño de archivo
- **Código estimado**: 60-70 líneas

---

### Fase 2: IQ2 (con grilla pequeña) — 6 horas

#### 2.1 IQ2_XXS — Grilla base
- **Block**: 256, **Type size**: 66 bytes
- **Grilla**: 256 entradas, 8 valores por entrada (grid_shape=(256,8))
- **Fórmula**: `d * grid[idx] * sign`
- **Algoritmo**:
  1. Calcular escala `d = max(|block|)` normalizada
  2. Para cada grupo de 8 elementos:
     - Encontrar par (signo, índice) que minimize error
     - Buscar en grilla: `idx = argmin(|target - grid|)`
     - Determinar signo: `sign = sign(target - grid[idx])`
  3. Empaquetar: d (2 bytes) + grid_idx (32 bytes) + signs (32 bytes)
- **Dependencia**: Tabla `ksigns` para desempaquetar signos
- **Tests**: Round-trip + verificar que el error es < epsilon
- **Código estimado**: 80-100 líneas

#### 2.2 IQ2_XS — Grilla mediana
- **Block**: 256, **Type size**: 74 bytes
- **Grilla**: 512 entradas (grid_shape=(512,8))
- **Diferencia**: Escalas por sub-block (4-bit) + grilla más grande
- **Algoritmo**:
  1. Dividir en 16 sub-blocks de 16
  2. Calcular escala por sub-block
  3. Cuantizar con grilla de 512
- **Código estimado**: 90-110 líneas

#### 2.3 IQ2_S — Grilla grande
- **Block**: 256, **Type size**: 82 bytes
- **Grilla**: 1024 entradas (grid_shape=(1024,8))
- **Diferencia**: 4 secciones split + qh high bits
- **Algoritmo**: Similar a IQ2_XS pero con 10-bit indices
- **Código estimado**: 120-140 líneas

---

### Fase 3: IQ3 (con grilla media) — 5 horas

#### 3.1 IQ3_XXS — Grilla base
- **Block**: 256, **Type size**: 98 bytes
- **Grilla**: 256 entradas, 4 valores por entrada (grid_shape=(256,4))
- **Fórmula**: `d * grid[idx] * sign`
- **Algoritmo**: Similar a IQ2_XXS pero con 8-bit indices (256 entradas)
- **Dependencia**: Tabla `ksigns` de IQ2_XXS
- **Código estimado**: 70-90 líneas

#### 3.2 IQ3_S — Grilla mediana
- **Block**: 256, **Type size**: 110 bytes
- **Grilla**: 512 entradas (grid_shape=(512,4))
- **Diferencia**: Escalas con fórmula `1 + 2*scales`
- **Código estimado**: 80-100 líneas

---

### Fase 4: IQ1 (extrema compresión) — 8 horas

#### 4.1 IQ1_S — Grilla grande + delta
- **Block**: 256, **Type size**: 50 bytes
- **Grilla**: 2048 entradas, valores {-1, 0, 1} (grid_shape=(2048,8))
- **Fórmula**: `d * (grid[idx] + delta)` — NOTA: suma, no multiplicación
- **Algoritmo**:
  1. Calcular delta: `+0.125` o `-0.125` según bit de control
  2. Para cada grupo de 8:
     - Buscar en grilla de 2048
     - El grid_map usa valores signed (-1, 0, 1)
  3. Empaquetar: d (2 bytes) + qs (32 bytes) + qh (32 bytes)
- **Código estimado**: 100-120 líneas

#### 4.2 IQ1_M — Reutiliza IQ1_S
- **Block**: 256, **Type size**: 56 bytes
- **Grilla**: Misma que IQ1_S
- **Diferencia**: Escalas empaquetadas en nibbles (f16 no está al inicio)
- **Algoritmo**:
  1. Reconstruir f16 desde 4 nibbles empaquetados
  2. Resto igual a IQ1_S
- **Código estimado**: 60-80 líneas

---

## Estructura del Código

### Archivo: `models/exported/gguf_quantizer.py` (extender)

```python
# Agregar después de las clases K-quant

# =============================================================================
# IQ4_NL - Non-linear 4-bit
# =============================================================================
class IQ4_NL_Quantizer:
    block_size = 32
    type_size = 18
    kvalues = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                        1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

    @staticmethod
    def quantize(data):
        # ... algoritmo

# =============================================================================
# IQ4_XS - 4-bit with super-block scales
# =============================================================================
class IQ4_XS_Quantizer:
    block_size = 256
    type_size = 136
    # Reutiliza kvalues de IQ4_NL

    @staticmethod
    def quantize(data):
        # ... algoritmo

# ... etc para cada tipo
```

### Archivo: `models/exported/convert_gguf.py` (extender)

```python
# Agregar IQ types al KQUANT_TYPES
KQUANT_TYPES = {"q2_k", ..., "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs",
                "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"}

# Agregar al KQUANT_ENUM_MAP
KQUANT_ENUM_MAP = {
    ...
    "iq4_nl": GGMLQuantizationType.IQ4_NL,
    "iq4_xs": GGMLQuantizationType.IQ4_XS,
    "iq2_xxs": GGMLQuantizationType.IQ2_XXS,
    "iq2_xs": GGMLQuantizationType.IQ2_XS,
    "iq2_s": GGMLQuantizationType.IQ2_S,
    "iq3_xxs": GGMLQuantizationType.IQ3_XXS,
    "iq3_s": GGMLQuantizationType.IQ3_S,
    "iq1_s": GGMLQuantizationType.IQ1_S,
    "iq1_m": GGMLQuantizationType.IQ1_M,
}
```

### Archivo: `main.py` (extender)

```python
parser.add_argument("--quantization", type=str, default="q8_0",
                    choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
                             "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
                             "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq2_s",
                             "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"],
                    help="GGUF quantization type")
```

---

## Tests por Tipo

Para cada tipo, ejecutar:

```bash
# 1. Test unitario de cuantización
python -c "
from models.exported.quantize import quantize_iq4_nl
import numpy as np
data = np.random.randn(32).astype(np.float32)
q, n, blocks = quantize_iq4_nl(data)
print(f'IQ4_NL: {q.nbytes} bytes')
"

# 2. Test round-trip (quantize → dequantize → diff)
python -c "
from models.exported.quantize import quantize_iq4_nl
from gguf.quants import IQ4_NL
import numpy as np

data = np.random.randn(32).astype(np.float32)
q, n, blocks = quantize_iq4_nl(data)
dequant = IQ4_NL.dequantize(q)
error = np.abs(data - dequant).mean()
print(f'Mean error: {error:.6f}')
"

# 3. Test export completo
python models/exported/convert_gguf.py models/exported/chat_model_hf \
    --outfile models/exported/test_iq4_nl.gguf --outtype iq4_nl
```

---

## Métricas de Éxito

| Tipo | Error max aceptable | Tamaño esperado |
|------|-------------------|-----------------|
| IQ4_NL | < 0.01 | ~1.1x q4_0 |
| IQ4_XS | < 0.01 | ~1.2x q4_0 |
| IQ2_XXS | < 0.05 | ~0.8x q4_0 |
| IQ2_XS | < 0.04 | ~0.9x q4_0 |
| IQ2_S | < 0.03 | ~1.0x q4_0 |
| IQ3_XXS | < 0.03 | ~1.1x q4_0 |
| IQ3_S | < 0.02 | ~1.2x q4_0 |
| IQ1_S | < 0.10 | ~0.6x q4_0 |
| IQ1_M | < 0.08 | ~0.7x q4_0 |

---

## Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Grillas incorrectas | Alto | Copiar hex数据 exactamente de gguf/quants.py |
| Signos empaquetados mal | Alto | Usar misma lógica de dequantización como referencia |
| Performance lenta | Medio | Optimizar con NumPy vectorizado, no loops Python |
| GGUF writer rechaza datos | Bajo | Ya resuelto para K-quants, misma lógica |
| Modelos cargables en llama.cpp | Alto | Verificar que el GGUF generado se carga correctamente |

---

## Dependencias Externas

Ninguna nueva. Solo:
- `numpy` (ya instalado)
- `torch` (ya instalado)
- `gguf` (ya instalado) — solo para constants, no para quantize

---

## Cronograma Estimado

| Fase | Tipo(s) | Horas | Acumulado |
|------|---------|-------|-----------|
| Fase 1 | IQ4_NL, IQ4_XS | 2h | 2h |
| Fase 2 | IQ2_XXS, IQ2_XS, IQ2_S | 6h | 8h |
| Fase 3 | IQ3_XXS, IQ3_S | 5h | 13h |
| Fase 4 | IQ1_S, IQ1_M | 8h | 21h |
| **Total** | **9 tipos** | **21h** | — |

**Nota**: Estimación optimista. Con debugging y tests reales, probablemente **25-30 horas**.
