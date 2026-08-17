# Exported Models Directory

This directory contains exported models and conversion tools for GGUF and ONNX formats.

## Contents

| File | Description |
|------|-------------|
| `convert_gguf.py` | HuggingFace to GGUF converter (with `--name` for Ollama) |
| `gguf_quantizer.py` | GGUF K-quant + IQ-quant implementations (15 types) |
| `convert_onnx.py` | HuggingFace to ONNX converter (with quantization) |
| `onnx_quantizer.py` | ONNX quantization engine (6 phases, 18 types) |
| `requirements.txt` | Python dependencies |

## Documentation

| File | Description |
|------|-------------|
| `GGUF_QUANTIZATION_README.md` | GGUF quantization reference (26 types) + Ollama integration |
| `ONNX_QUANTIZATION_README.md` | ONNX quantization reference (18 types) + standalone usage |
| `DONE_ROADMAP_IQ_QUANTIZE.md` | Completed IQ quantization roadmap |
| `DONE_ROADMAP_ONNX_QUANTIZE.md` | Completed ONNX quantization roadmap |

## Export Commands (from project root)

### GGUF

```bash
# Basic export
python main.py --export chat_model --formats gguf --quantization q8_0

# With custom name (appears in GGUF metadata and Ollama)
python main.py --export chat_model --formats gguf --quantization q4_k

# Multiple quantizations
python main.py --export chat_model --formats gguf --quantization f16
python main.py --export chat_model --formats gguf --quantization iq4_xs
```

### ONNX

```bash
# FP32 baseline
python main.py --export chat_model --formats onnx

# Dynamic quantization (no calibration)
python main.py --export chat_model --formats onnx_int8
python main.py --export chat_model --formats onnx_int4

# Static quantization (needs calibration)
python main.py --export chat_model --formats onnx_static_int8
python main.py --export chat_model --formats onnx_static_int4

# FP8 (GPU only)
python main.py --export chat_model --formats onnx_fp8

# Mixed/Per-Channel
python main.py --export chat_model --formats onnx_per_channel
python main.py --export chat_model --formats onnx_mixed

# Multiple formats at once
python main.py --export chat_model --formats gguf,onnx,onnx_int8
```

## Ollama Integration (GGUF)

After GGUF export, the `_hf` directory includes a `Modelfile`:

```bash
# 1. Export
python main.py --export chat_model --formats gguf --quantization q8_0

# 2. Convert to GGUF
cd models/exported/chat_model_hf
convert_to_gguf.bat   # Windows

# 3. Create Ollama model
ollama create chat_model -f Modelfile

# 4. Run
ollama run chat_model
```

## Standalone Inference (ONNX)

After ONNX export, the `_onnx` directory is self-contained:

```bash
cd models/exported/chat_model_onnx
pip install -r requirements.txt

# Interactive chat
python inference.py

# Single prompt
python inference.py --prompt "Hello"
```

## Direct CLI

```bash
# GGUF
python models/exported/convert_gguf.py <hf_dir> --outfile output.gguf --outtype q8_0 --name my_model

# ONNX
python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx --quant-type int8
```
