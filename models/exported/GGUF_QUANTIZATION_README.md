# GGUF Quantization Reference

All available quantization types for GGUF export.

## Usage

```bash
# From project root
python main.py --export chat_model --formats gguf --quantization q5_0

# Direct conversion
python models/exported/convert_gguf.py <hf_dir> --outfile output.gguf --outtype q5_0
```

## Supported Quantization Types

### Float Types (no quantization)

| Type | Bits | Bytes/elem | Size vs F32 | Description |
|------|------|-----------|-------------|-------------|
| `f32` | 32 | 4 | 100% | Full precision, largest file |
| `f16` | 16 | 2 | 50% | Half precision, good accuracy |

### Standard Quantization (block-wise)

| Type | Bits | Bytes/elem | Size vs F32 | Description |
|------|------|-----------|-------------|-------------|
| `q4_0` | 4 | 0.56 | 14% | 4-bit, smallest, lowest accuracy |
| `q4_1` | 4 | 0.63 | 16% | 4-bit with better accuracy than q4_0 |
| `q5_0` | 5 | 0.69 | 17% | 5-bit, good balance |
| `q5_1` | 5 | 0.75 | 19% | 5-bit with better accuracy than q5_0 |
| `q8_0` | 8 | 1.06 | 27% | 8-bit, recommended default |

### K-Quant Types (improved quality)

K-quants use super-blocks of 256 elements with per-super-block scaling for better quality.

| Type | Bits | Bytes/elem | Size vs F32 | Description |
|------|------|-----------|-------------|-------------|
| `q2_k` | 2 | 0.31 | 8% | 2-bit K-quant, extreme compression |
| `q3_k` | 3 | 0.38 | 10% | 3-bit K-quant |
| `q4_k` | 4 | 0.44 | 11% | 4-bit K-quant, good quality |
| `q5_k` | 5 | 0.53 | 13% | 5-bit K-quant, high quality |
| `q6_k` | 6 | 0.63 | 16% | 6-bit K-quant, very high quality |
| `q8_k` | 8 | 0.88 | 22% | 8-bit K-quant, best quality K-quant |

### IQ-Quant Types (importance matrix)

IQ-quants use importance matrices and grid-based quantization for extreme compression with better quality than standard quantization.

| Type | Bits | Bytes/elem | Size vs F32 | Description |
|------|------|-----------|-------------|-------------|
| `iq4_nl` | 4 | 0.56 | 14% | 4-bit non-linear, 16-value lookup |
| `iq4_xs` | 4 | 0.53 | 13% | 4-bit with super-block scales |
| `iq2_xxs` | 2 | 0.26 | 6% | 2-bit grid (256 entries), extreme compression |
| `iq2_xs` | 2 | 0.29 | 7% | 2-bit grid (512 entries) |
| `iq2_s` | 2 | 0.32 | 8% | 2-bit grid (1024 entries), best IQ2 quality |
| `iq3_xxs` | 3 | 0.38 | 10% | 3-bit grid (256 entries) |
| `iq3_s` | 3 | 0.43 | 11% | 3-bit grid (512 entries) |
| `iq1_s` | 1 | 0.20 | 5% | 1-bit grid (2048 entries), smallest possible |
| `iq1_m` | 1 | 0.22 | 5% | 1-bit with packed f16 scales |

## Recommendations

| Use Case | Recommended Type | Why |
|----------|-----------------|-----|
| **General use** | `q8_0` | Best quality/size balance |
| **Small models (<50M)** | `q5_0` | Compact with good accuracy |
| **Large models (>500M)** | `q5_1` | Better quality at reasonable size |
| **Maximum quality** | `f16` | No quantization loss |
| **Best K-quant quality** | `q8_k` | Best quality among K-quants |
| **Best IQ quality** | `iq4_xs` | Best quality among IQ-quants |
| **Minimum size** | `iq1_s` | Smallest possible (5% of F32) |
| **Balanced K-quant** | `q4_k` | Good quality at 11% of F32 |
| **Balanced IQ** | `iq3_s` | Good quality at 11% of F32 |

## Size Estimates (for ~5M parameter model)

| Type | Approximate Size |
|------|-----------------|
| `f32` | ~20 MB |
| `f16` | ~10 MB |
| `q8_0` | ~5.5 MB |
| `q8_k` | ~4.4 MB |
| `q5_0` | ~2.8 MB |
| `q4_0` | ~2.5 MB |
| `q4_k` | ~2.2 MB |
| `iq4_nl` | ~2.3 MB |
| `iq4_xs` | ~2.1 MB |
| `q2_k` | ~1.6 MB |
| `iq2_xxs` | ~1.3 MB |
| `iq2_xs` | ~1.4 MB |
| `iq2_s` | ~1.6 MB |
| `iq3_xxs` | ~1.9 MB |
| `iq3_s` | ~2.1 MB |
| `iq1_s` | ~1.0 MB |
| `iq1_m` | ~1.1 MB |

## Examples

```bash
# Export with maximum quality (float16)
python main.py --export chat_model --formats gguf --quantization f16

# Export with 5-bit quantization (good balance)
python main.py --export chat_model --formats gguf --quantization q5_0

# Export with 8-bit quantization (default)
python main.py --export chat_model --formats gguf --quantization q8_0

# Export with best K-quant quality
python main.py --export chat_model --formats gguf --quantization q8_k

# Export with K-quant (best quality for 4-bit)
python main.py --export chat_model --formats gguf --quantization q4_k

# Export with IQ-quant (extreme compression)
python main.py --export chat_model --formats gguf --quantization iq4_xs

# Export with smallest possible size
python main.py --export chat_model --formats gguf --quantization iq1_s
```

## Requirements

```bash
pip install gguf torch numpy
```

## Technical Details

### K-Quant Block Structure

K-quants use a hierarchical block structure:
- **Super-block**: 256 elements
- **Sub-blocks**: 16-32 elements each
- Each sub-block has its own scale and minimum value
- Better quality than standard quantization at same bit width

### IQ-Quant Block Structure

IQ-quants use importance matrices and grid-based quantization:
- **Grid**: Pre-computed set of possible quantized values
- **Importance matrix**: Determines which elements get more precision
- **Signs**: Packed separately for efficient storage
- **Scales**: Per-sub-block or global scaling
- Better quality than K-quants at same bit width

### Implementation Notes

- Standard types (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`) use the `gguf` package's built-in quantize function
- K-quant types (`q2_k` through `q6_k`) use custom implementation in `gguf_quantizer.py`
- IQ-quant types (`iq4_nl` through `iq1_m`) use custom implementation in `gguf_quantizer.py`
- All quantization is pure Python/NumPy (no external dependencies required)
- The model must be exported to HuggingFace format first before GGUF conversion

---

## See Also

- [ONNX Quantization Reference](ONNX_QUANTIZATION_README.md) — INT8/INT4/FP8 quantization for ONNX export
- [IQ Quantization DONE](DONE_ROADMAP_IQ_QUANTIZE.md) — Completed IQ quantization implementation
