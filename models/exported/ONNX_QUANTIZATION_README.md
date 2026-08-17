# ONNX Quantization Reference

All available quantization types for ONNX export (6 phases, 18 types).

## Usage

```bash
# From project root — via main.py
python main.py --export chat_model --formats onnx_int8
python main.py --export chat_model --formats onnx_static_int8
python main.py --export chat_model --formats onnx_fp8

# Direct conversion
python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx --quant-type int8

# Static quantization (needs calibration)
python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx --quant-type static_int8_qdq --static

# FP8 (GPU only)
python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx --quant-type fp8_e4m3fn --static
```

## Standalone Usage

After export, the `_onnx` directory is a self-contained package. No project code needed:

```bash
cd models/exported/chat_model_onnx
pip install -r requirements.txt

# Interactive chat
python inference.py

# Single prompt
python inference.py --prompt "Que es la fotosintesis?"

# With parameters
python inference.py --prompt "Hola" --max-tokens 200 --temperature 0.5

# Use specific model directory
python inference.py --model-dir /path/to/onnx_package
```

## Supported Quantization Types

### Phase 1: Dynamic Quantization

No calibration data required. Fast, good for quick deployment.

| Type | Bits | Size vs FP32 | Speed | Description |
|------|------|-------------|-------|-------------|
| `int8` | 8 | ~28% | 2-3x | Dynamic INT8 weights (default) |
| `uint8` | 8 | ~28% | 2-3x | Dynamic UINT8 weights |
| `int4` | 4 | ~15% | 1.5x | Dynamic INT4 weights (ort >= 1.17) |
| `uint4` | 4 | ~15% | 1.5x | Dynamic UINT4 weights (ort >= 1.17) |

### Phase 2: Static INT8 Quantization

Requires calibration data. Better accuracy than dynamic.

| Type | Format | Size vs FP32 | Speed | Description |
|------|--------|-------------|-------|-------------|
| `static_int8_qdq` | QDQ | ~25% | 3-4x | QDQ format (general purpose) |
| `static_int8_qoperator` | QOperator | ~25% | 3-4x | Native INT8 ops (hardware specific) |

### Phase 3: Static INT4 Quantization

Requires calibration. Best compression for static models.

| Type | Format | Size vs FP32 | Speed | Description |
|------|--------|-------------|-------|-------------|
| `static_int4_qoperator` | QOperator | ~13% | 1.5-2x | Native INT4 ops (ort >= 1.17) |
| `static_int4_qdq` | QDQ | ~13% | 1.5-2x | QDQ format INT4 |

### Phase 4: Static FP8 Quantization

Requires calibration + GPU (H100/Blackwell). Best quality/speed on supported hardware.

| Type | Bits | Size vs FP32 | Speed | Description |
|------|------|-------------|-------|-------------|
| `fp8_e4m3fn` | 8 | ~28% | 3-5x | FP8 E4M3FN weights (H100+) |
| `fp8_e5m2` | 8 | ~28% | 3-5x | FP8 E5M2 activations (wider range) |
| `fp8_mixed` | 8 | ~28% | 3-5x | E4M3FN weights + E5M2 activations |
| `fp8_e4m3fnuz` | 8 | ~28% | 3-5x | E4M3FNUZ (GraphCore, some GPUs) |
| `fp8_e5m2fnuz` | 8 | ~28% | 3-5x | E5M2FNUZ (GraphCore, some GPUs) |

### Phase 5: Per-Channel Quantization

Requires calibration. Better precision for models with variable distributions.

| Type | Size vs FP32 | Speed | Description |
|------|-------------|-------|-------------|
| `per_channel_int8` | ~25% | 3-4x | Per-channel INT8 (axis=output) |
| `per_channel_int4` | ~13% | 1.5-2x | Per-channel INT4 |

### Phase 6: Mixed Precision

Requires calibration. Combines types for optimal balance.

| Type | Size vs FP32 | Speed | Description |
|------|-------------|-------|-------------|
| `mixed_int8_int4` | ~15% | 2-3x | INT8 activations + INT4 weights |
| `mixed_fp8_int8` | ~25% | 3-5x | FP8 weights + INT8 activations |
| `tensor_overrides` | ~25% | 3-4x | Per-layer control (custom ops/nodes) |

## Recommendations

| Use Case | Recommended Type | Why |
|----------|-----------------|-----|
| **Quick start** | `int8` | No calibration, good quality |
| **Best CPU perf** | `static_int8_qdq` | 3-4x speedup, minimal quality loss |
| **Edge devices** | `int4` | Smallest model, low memory |
| **GPU (H100+)** | `fp8_e4m3fn` | Fastest inference on supported hardware |
| **GPU (mixed)** | `fp8_mixed` | Best precision/range balance |
| **Max compression** | `static_int4_qdq` | 13% of FP32, requires calibration |
| **Best precision** | `per_channel_int8` | Per-channel scaling, higher accuracy |
| **Custom layers** | `tensor_overrides` | Exclude sensitive layers from quantization |

## Size Estimates (for ~2.2M parameter model)

| Type | Approximate Size |
|------|-----------------|
| FP32 ONNX | ~9 MB |
| FP16 ONNX | ~4.5 MB |
| `int8` / `uint8` | ~2.5 MB |
| `static_int8_qdq` | ~2.3 MB |
| `int4` / `uint4` | ~1.5 MB |
| `static_int4_*` | ~1.4 MB |
| `fp8_*` | ~2.5 MB |
| `per_channel_int8` | ~2.3 MB |
| `per_channel_int4` | ~1.4 MB |
| `mixed_*` | ~1.5-2.5 MB |

## Examples

```bash
# Export FP32 ONNX baseline
python main.py --export chat_model --formats onnx

# Export with dynamic INT8 (default, no calibration)
python main.py --export chat_model --formats onnx_int8

# Export with static INT8 (better quality, needs calibration)
python main.py --export chat_model --formats onnx_static_int8

# Export with INT4 (smallest)
python main.py --export chat_model --formats onnx_int4

# Export with FP8 (GPU only)
python main.py --export chat_model --formats onnx_fp8

# Export with per-channel quantization
python main.py --export chat_model --formats onnx_per_channel

# Export with mixed precision
python main.py --export chat_model --formats onnx_mixed

# Direct CLI with custom options
python models/exported/convert_onnx.py models/exported/chat_model_hf \
    --outfile model.onnx --quant-type static_int8_qdq --static \
    --calibration-samples 200
```

## Exported Directory Structure

After ONNX export, the `_onnx` directory contains everything needed for independent usage:

```
chat_model_onnx/
  chat_model.onnx              # ONNX model
  chat_model_int8.onnx         # Quantized variant (if applicable)
  config.json                  # Architecture (BOS/EOS/pad IDs)
  sentencepiece.model          # Tokenizer model
  tokenizer.json               # HuggingFace tokenizer
  tokenizer_config.json        # Tokenizer config
  special_tokens_map.json      # Special tokens
  tokenizer_vocab.json         # SentencePiece reference
  chat_model_metadata.json     # ONNX metadata (tokens, arch)
  metadata.json                # Export metadata
  inference.py                 # Standalone inference script
  requirements.txt             # onnxruntime, sentencepiece, numpy
  LICENSE                      # MIT
  NOTICE                       # GPT-2 Modified MIT
  LICENSE_INFO.json            # License metadata
```

The `inference.py` script supports:
- Interactive chat mode (default)
- Single prompt mode (`--prompt`)
- Temperature control (`--temperature`)
- Max tokens control (`--max-tokens`)
- Automatic detection of quantized models

## Requirements

```bash
pip install onnxruntime>=1.17 onnx>=1.16 numpy torch
# Optional: GPU FP8 support
pip install onnxruntime-gpu
```

### Version Requirements

| Feature | Min Version |
|---------|-------------|
| INT8 dynamic/static | onnxruntime >= 1.12 |
| INT4 dynamic/static | onnxruntime >= 1.17 |
| FP8 | onnxruntime-gpu >= 1.16, CUDA >= 11.8 |
| Per-channel | onnxruntime >= 1.12 |

## Technical Details

### Dynamic vs Static

- **Dynamic**: Weights quantized at export time, activations quantized at runtime. No calibration needed.
- **Static**: Both weights and activations quantized at export time. Requires calibration data to measure activation ranges.

### QDQ vs QOperator

- **QDQ** (QuantizeLinear + DequantizeLinear): Inserts quantize/dequantize nodes. General purpose, works with most runtimes.
- **QOperator**: Uses native quantized operators (QLinearConv, MatMulInteger). Faster on hardware with INT8 support.

### Calibration Methods

| Method | Speed | Quality | When to Use |
|--------|-------|---------|-------------|
| MinMax | Fast | Good | Default choice |
| Percentile | Medium | Better | Outlier-heavy data |
| Entropy | Slow | Best | Maximum accuracy needed |

### Per-Channel Quantization

Scales are computed per output channel instead of per tensor. Better for models where different channels have significantly different ranges.

### Mixed Precision

Combines different quantization types for different parts of the model:
- activations: INT8 (range [0, 255])
- weights: INT4 (range [-8, 7])
- Sensitive layers can use higher precision via `tensor_overrides`

---

## See Also

- [GGUF Quantization Reference](GGUF_QUANTIZATION_README.md) — GGUF export types (K-quants, IQ-quants)
- [ONNX Quantization DONE](DONE_ROADMAP_ONNX_QUANTIZE.md) — Completed ONNX quantization implementation
- [IQ Quantization DONE](DONE_ROADMAP_IQ_QUANTIZE.md) — Completed IQ quantization implementation
