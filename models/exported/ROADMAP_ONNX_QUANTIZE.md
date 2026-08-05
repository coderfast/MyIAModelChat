# ONNX Quantization Implementation Roadmap

**Objetivo**: Implementar cuantización ONNX completa (estática, dinámica, INT4/INT8/FP8) para exportar modelos GPT-2 a formatos optimizados para inferencia en CPU/GPU.

**Estado actual**: Solo cuantización dinámica INT8/UINT8 básica en `export_to_onnx_quantized()`.

**Estrategia**: Implementar del más usado al más complejo, testeando cada tipo antes de avanzar.

---

## Tipos de Cuantización ONNX Soportados

### Enteros (INT)

| Tipo | Bits | Rango | Uso |
|------|------|-------|-----|
| `QInt8` | 8 | [-128, 127] | Pesos (default) |
| `QUInt8` | 8 | [0, 255] | Activaciones (default) |
| `QInt16` | 16 | [-32768, 32767] | Pesos de alta precisión |
| `QUInt16` | 16 | [0, 65535] | Activaciones de alta precisión |
| `QInt4` | 4 | [-8, 7] | Pesos ultra-comprimidos |
| `QUInt4` | 4 | [0, 15] | Pesos ultra-comprimidos |

### Float8 (FP8)

| Tipo | Bits | Rango | Descripción |
|------|------|-------|-------------|
| `FLOAT8E4M3FN` | 8 | ±448 | 4 exponentes, 3 mantisa, sin infinitos |
| `FLOAT8E4M3FNUZ` | 8 | ±448 | Sin infinitos, sin cero negativo |
| `FLOAT8E5M2` | 8 | ±57344 | 5 exponentes, 2 mantisa, con infinitos |
| `FLOAT8E5M2FNUZ` | 8 | ±57344 | Sin infinitos, sin cero negativo |

### Métodos de Calibración

| Método | Descripción | Precisión |
|--------|-------------|-----------|
| `MinMax` | Rango basado en min/max observado | Rápido, menos preciso |
| `Percentile` | Percentil configurable (default 99.99%) | Balanceado |
| `Entropy` | Minimiza divergencia KL | Lento, mejor precisión |
| `KL` | Minimización de información perdida | Lento, mejor precisión |

### Formatos de Cuantización

| Formato | Descripción | Uso Recomendado |
|---------|-------------|-----------------|
| `QOperator` | Operadores nativos (QLinearConv, MatMulInteger) | Hardware específico |
| `QDQ` | QuantizeLinear + DequantizeLinear insertados | General (default) |

---

## Orden de Implementación

### Fase 1: Cuantización Dinámica Mejorada — 2 horas

#### 1.1 INT8/UINT8 Dinámico
- **Operadores**: MatMul, Conv
- **Tipo**: `quantize_dynamic`
- **Configuración**: `weight_type=QuantType.QInt8` (default)
- **Algoritmo**:
  1. Medir rango de pesos durante inferencia
  2. Cuantizar pesos a INT8/UINT8
  3. Activaciones cuantizadas en runtime
- **Código**: ~30 líneas (extender `export_to_onnx_quantized`)
- **Dependencia**: `onnxruntime` (ya instalado)

#### 1.2 INT4/UINT4 Dinámico
- **Operadores**: MatMul (weight-only), Gather
- **Tipo**: `quantize_dynamic` con `weight_type=QuantType.QInt4`
- **Block size**: 128 (default)
- **Algoritmo**:
  1. Dividir pesos en bloques de 128
  2. Cuantizar por bloque (RTN o HQQ)
  3. Guardar scales por bloque
- **Código**: ~50 líneas
- **Dependencia**: `onnxruntime` >= 1.17

---

### Fase 2: Cuantización Estática INT8 — 4 horas

#### 2.1 Static Quantize INT8 (QDQ)
- **Operadores**: MatMul, Conv, Add
- **Formato**: QDQ (default)
- **Calibración**: MinMax (rápido)
- **Configuración**:
  ```python
  quantize_static(
      model_input="model.onnx",
      model_output="model_quant.onnx",
      calibration_data_reader=reader,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
      per_channel=False,
  )
  ```
- **Algoritmo**:
  1. Ejecutar modelo con datos de calibración
  2. Medir rangos de activaciones
  3. Insertar nodos QuantizeLinear/DequantizeLinear
  4. Optimizar fusiones QDQ
- **Código**: ~80 líneas
- **Dependencia**: `onnxruntime`, `numpy`

#### 2.2 Static Quantize INT8 (QOperator)
- **Operadores**: QLinearConv, MatMulInteger
- **Formato**: QOperator
- **Diferencia**: Operadores nativos en lugar de QDQ
- **Uso**: Hardware que soporta nativamente INT8
- **Código**: ~60 líneas
- **Dependencia**: `onnxruntime`

#### 2.3 Per-Channel Quantization
- **Tipo**: `per_channel=True`
- **Ventaja**: Mejor precisión para modelos con distribuciones variables
- **Operadores**: Conv, MatMul
- **Código**: ~40 líneas (extender 2.1)

---

### Fase 3: Cuantización Estática INT4 — 4 horas

#### 3.1 Static Quantize INT4 (QOperator)
- **Operadores**: MatMulNBits, GatherBlockQuantized
- **Formato**: QOperator
- **Block size**: 128 (default, potencia de 2 >= 16)
- **Algoritmo**:
  1. Dividir pesos en bloques de 128
  2. Cuantizar por bloque (RTN, HQQ, o GPTQ)
  3. Guardar scales por bloque
- **Código**: ~100 líneas
- **Dependencia**: `onnxruntime` >= 1.17, opset >= 21

#### 3.2 Static Quantize INT4 (QDQ)
- **Operadores**: DequantizeLinear -> MatMul
- **Formato**: QDQ
- **Diferencia**: Representación QDQ en lugar de nodo nativo
- **Código**: ~80 líneas

---

### Fase 4: Cuantización Estática FP8 — 6 horas

#### 4.1 FP8 E4M3FN (Pesos)
- **Tipo**: `QuantType.QFLOAT8E4M3FN`
- **Uso**: Pesos (forward pass)
- **Algoritmo**:
  1. Calibración con datos reales
  2. Cuantizar pesos a FP8 E4M3FN
  3. Activaciones en FP16 o FP32
- **Código**: ~70 líneas
- **Dependencia**: `onnxruntime` >= 1.16, GPU con CUDA >= 11.8

#### 4.2 FP8 E5M2 (Activaciones)
- **Tipo**: `QuantType.QFLOAT8E5M2`
- **Uso**: Activaciones y gradientes
- **Diferencia**: Mayor rango dinámico
- **Código**: ~60 líneas

#### 4.3 FP8 E4M3FN + E5M2 Mixto
- **Configuración**: Pesos E4M3FN, activaciones E5M2
- **Ventaja**: Mejor balance precisión/rango
- **Código**: ~50 líneas (extender 4.1/4.2)

#### 4.4 FP8 E4M3FNUZ / E5M2FNUZ
- **Tipos**: Variantes sin cero negativo
- **Uso**: Hardware GraphCore, algunos GPUs
- **Código**: ~40 líneas

---

### Fase 5: Cuantización por Canales — 3 horas

#### 5.1 Per-Channel INT8
- **Tipo**: `per_channel=True`
- **Eje**: Por canal de salida
- **Ventaja**: Mejor precisión para modelos con distribuciones variables
- **Código**: ~50 líneas (extender Fase 2)

#### 5.2 Per-Channel INT4
- **Tipo**: `per_channel=True` con INT4
- **Block size**: 128
- **Código**: ~60 líneas

---

### Fase 6: Mixed Precision — 4 horas

#### 6.1 INT8 Activaciones + INT4 Pesos
- **Configuración**: `activation_type=QUInt8, weight_type=QInt4`
- **Ventaja**: Balance entre compresión y precisión
- **Código**: ~40 líneas

#### 6.2 FP8 Pesos + INT8 Activaciones
- **Configuración**: `weight_type=QFLOAT8E4M3FN, activation_type=QUInt8`
- **Ventaja**: Alta precisión en pesos, eficiente en activaciones
- **Código**: ~40 líneas

#### 6.3 Tensor Quantization Overrides
- **Tipo**: Overrides por tensor específico
- **Uso**: Capas críticas con mayor precisión
- **Código**: ~60 líneas

---

## Estructura del Código

### Archivo: `models/exported/onnx_quantize.py` (nuevo)

```python
"""
ONNX quantization implementation.

Supports:
- Dynamic quantization: INT8, UINT8, INT4, UINT4
- Static quantization: INT8, UINT8, INT4, UINT4, FP8
- Calibration methods: MinMax, Percentile, Entropy, KL
- Formats: QOperator, QDQ
- Per-channel quantization
- Mixed precision
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable
import numpy as np


class QuantFormat(Enum):
    QOPERATOR = "qoperator"
    QDQ = "qdq"


class CalibrationMethod(Enum):
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"
    KL = "kl"


@dataclass
class ONNXQuantConfig:
    """Configuration for ONNX quantization."""
    quant_type: str = "int8"  # int8, uint8, int4, uint4, fp8_e4m3fn, fp8_e5m2
    quant_format: QuantFormat = QuantFormat.QDQ
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    activation_type: Optional[str] = None  # Override activation type
    weight_type: Optional[str] = None  # Override weight type
    per_channel: bool = False
    block_size: int = 128  # For INT4/UINT4
    reduce_range: bool = False  # For non-VNNI x86
    symmetric: bool = False
    op_types_to_quantize: Optional[List[str]] = None
    nodes_to_exclude: Optional[List[str]] = None
    extra_options: Optional[dict] = None


class ONNXQuantizer:
    """Main quantization class for ONNX models."""

    def __init__(self, config: ONNXQuantConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate quantization configuration."""
        valid_types = {
            "int8", "uint8", "int4", "uint4",
            "fp8_e4m3fn", "fp8_e5m2", "fp8_e4m3fnuz", "fp8_e5m2fnuz"
        }
        if self.config.quant_type not in valid_types:
            raise ValueError(f"Invalid quant_type: {self.config.quant_type}")

    def quantize_dynamic(self, model_path: str, output_path: str) -> str:
        """Apply dynamic quantization."""
        # ... implementation
        pass

    def quantize_static(
        self,
        model_path: str,
        output_path: str,
        calibration_data_reader: Callable,
    ) -> str:
        """Apply static quantization."""
        # ... implementation
        pass

    def calibrate(
        self,
        model_path: str,
        calibration_data_reader: Callable,
        num_samples: int = 100,
    ) -> dict:
        """Run calibration to get quantization parameters."""
        # ... implementation
        pass


class CalibrationDataReader:
    """Base class for calibration data readers."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self._index = 0

    def get_next(self) -> Optional[dict]:
        """Get next calibration sample."""
        if self._index >= len(self.data):
            return None
        sample = self.data[self._index]
        self._index += 1
        return {"input_ids": sample}


# Convenience functions
def quantize_onnx_dynamic(
    model_path: str,
    output_path: str,
    quant_type: str = "int8",
    **kwargs,
) -> str:
    """Quick dynamic quantization."""
    config = ONNXQuantConfig(quant_type=quant_type, **kwargs)
    quantizer = ONNXQuantizer(config)
    return quantizer.quantize_dynamic(model_path, output_path)


def quantize_onnx_static(
    model_path: str,
    output_path: str,
    calibration_data: np.ndarray,
    quant_type: str = "int8",
    **kwargs,
) -> str:
    """Quick static quantization."""
    config = ONNXQuantConfig(quant_type=quant_type, **kwargs)
    quantizer = ONNXQuantizer(config)
    reader = CalibrationDataReader(calibration_data)
    return quantizer.quantize_static(model_path, output_path, reader)
```

### Archivo: `models/exported/convert_onnx.py` (nuevo)

```python
#!/usr/bin/env python3
"""
Convert HuggingFace model to ONNX with quantization options.

Usage:
    python convert_onnx.py <hf_dir> --outfile <output.onnx> --quant-type int8
    python convert_onnx.py <hf_dir> --outfile <output.onnx> --quant-type fp8_e4m3fn --static
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert to ONNX with quantization")
    parser.add_argument("hf_dir", help="HuggingFace model directory")
    parser.add_argument("--outfile", required=True, help="Output ONNX file")
    parser.add_argument("--quant-type", default="int8",
                        choices=["int8", "uint8", "int4", "uint4",
                                 "fp8_e4m3fn", "fp8_e5m2", "fp8_e4m3fnuz", "fp8_e5m2fnuz"],
                        help="Quantization type")
    parser.add_argument("--static", action="store_true", help="Use static quantization")
    parser.add_argument("--per-channel", action="store_true", help="Per-channel quantization")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for INT4")
    parser.add_argument("--calibration-samples", type=int, default=100, help="Calibration samples")
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset version")
    args = parser.parse_args()

    # ... implementation
    pass


if __name__ == "__main__":
    main()
```

### Archivo: `main.py` (modificar)

```python
# Agregar argumentos ONNX
parser.add_argument("--onnx-quant-type", type=str, default="int8",
                    choices=["int8", "uint8", "int4", "uint4",
                             "fp8_e4m3fn", "fp8_e5m2", "fp8_e4m3fnuz", "fp8_e5m2fnuz"],
                    help="ONNX quantization type")
parser.add_argument("--onnx-static", action="store_true",
                    help="Use static quantization for ONNX")
parser.add_argument("--onnx-per-channel", action="store_true",
                    help="Use per-channel quantization for ONNX")
parser.add_argument("--onnx-block-size", type=int, default=128,
                    help="Block size for INT4 ONNX quantization")

# En export_model(), agregar formatos:
elif fmt == 'onnx_int4':
    results['onnx_int4'] = export_to_onnx_quantized(pth_path, quant_type='int4')
elif fmt == 'onnx_fp8':
    results['onnx_fp8'] = export_to_onnx_quantized(pth_path, quant_type='fp8_e4m3fn')
```

---

## Tests por Tipo

Para cada tipo, ejecutar:

```bash
# 1. Test dinámico INT8
python -c "
from models.exported.onnx_quantize import quantize_onnx_dynamic
import numpy as np

# Simular modelo ONNX
model_path = 'models/exported/test_model.onnx'
quantize_onnx_dynamic(model_path, 'models/exported/test_int8.onnx', quant_type='int8')
print('Dynamic INT8: OK')
"

# 2. Test estático INT8
python -c "
from models.exported.onnx_quantize import quantize_onnx_static
import numpy as np

model_path = 'models/exported/test_model.onnx'
calib_data = np.random.randn(100, 512).astype(np.float32)
quantize_onnx_static(model_path, 'models/exported/test_int8_static.onnx',
                     calib_data, quant_type='int8')
print('Static INT8: OK')
"

# 3. Test FP8 E4M3FN
python -c "
from models.exported.onnx_quantize import quantize_onnx_static
import numpy as np

model_path = 'models/exported/test_model.onnx'
calib_data = np.random.randn(100, 512).astype(np.float32)
quantize_onnx_static(model_path, 'models/exported/test_fp8.onnx',
                     calib_data, quant_type='fp8_e4m3fn')
print('Static FP8: OK')
"

# 4. Test export completo
python models/exported/convert_onnx.py models/exported/chat_model_hf \
    --outfile models/exported/test_quant.onnx --quant-type int8 --static
```

---

## Métricas de Éxito

| Tipo | Error max aceptable | Tamaño vs FP32 | Velocidad |
|------|-------------------|----------------|-----------|
| INT8 dinámico | < 0.01 | 25-30% | 1.5-2x |
| INT8 estático | < 0.005 | 25-30% | 2-3x |
| INT4 dinámico | < 0.02 | 15-20% | 1.2-1.5x |
| INT4 estático | < 0.01 | 15-20% | 1.5-2x |
| FP8 E4M3FN | < 0.003 | 25-30% | 2-4x (GPU) |
| FP8 E5M2 | < 0.005 | 25-30% | 2-4x (GPU) |

---

## Dependencias

### Requeridas (ya instaladas)
- `onnxruntime` — Quantización dinámica y estática
- `numpy` — Procesamiento de datos
- `torch` — Exportación ONNX

### Opcionales
- `onnxruntime-gpu` — Soporte FP8 en GPU
- `onnx` — Manipulación de grafos ONNX
- `optimum` — HuggingFace Optimum para cuantización avanzada

### Versiones Mínimas
- `onnxruntime >= 1.17` — Soporte INT4
- `onnx >= 1.16` — Tipos INT4/UINT4
- `onnx >= 1.15` — Tipos FP8
- CUDA >= 11.8 — FP8 en GPU

---

## Plataformas y Configuraciones Recomendadas

| Plataforma | Activación | Pesos | Reduce Range | Notas |
|------------|------------|-------|--------------|-------|
| x86 sin VNNI | QUInt8 | QInt8 | True | CPUs pre-Skylake |
| x86 con VNNI | QUInt8 | QInt8 | False | Skylake-SP+, AMD Zen4+ |
| ARM/Apple Silicon | QInt8 | QInt8 | False | NEON/SVE |
| GPU NVIDIA | QInt8 | QInt8 | False | CUDA |
| GPU NVIDIA FP8 | E4M3FN | E4M3FN | False | H100+, Blackwell |
| QNN (Qualcomm) | QUInt8 | QUInt8 | False | Snapdragon |
| QNN INT4 | QUInt4 | QInt4 | False | Snapdragon |

---

## Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Opset insuficiente | Alto | Forzar upgrade a opset >= 21 para INT4/FP8 |
| Calibración insuficiente | Medio | Usar múltiples métodos, validar con datos reales |
| Pérdida de precisión | Medio | Probar mixed precision, per-channel |
| GPU no soporta FP8 | Bajo | Fallback a INT8, detectar hardware |
| onnxruntime no instala | Bajo | Instrucciones claras en requirements |

---

## Cronograma Estimado

| Fase | Tipo(s) | Horas | Acumulado |
|------|---------|-------|-----------|
| Fase 1 | INT8/UINT8/INT4/UINT4 dinámico | 2h | 2h |
| Fase 2 | INT8 estático (QDQ, QOperator, per-channel) | 4h | 6h |
| Fase 3 | INT4 estático (QOperator, QDQ) | 4h | 10h |
| Fase 4 | FP8 (E4M3FN, E5M2, mixto, FNUZ) | 6h | 16h |
| Fase 5 | Per-channel INT8/INT4 | 3h | 19h |
| Fase 6 | Mixed precision | 4h | 23h |
| **Total** | **15+ configuraciones** | **23h** | — |

**Nota**: Estimación optimista. Con debugging y tests reales, probablemente **28-35 horas**.
