Exported Models Directory
=========================

This directory contains exported models and conversion tools for GGUF and ONNX formats.

Contents:
  convert_gguf.py       - HuggingFace to GGUF converter
  gguf_quantizer.py     - GGUF K-quant + IQ-quant implementations (15 types)
  convert_onnx.py       - HuggingFace to ONNX converter (with quantization)
  onnx_quantizer.py     - ONNX quantization engine (6 phases, 18 types)
  requirements.txt      - Python dependencies
  chat_model_hf/        - Example exported model (HuggingFace format)

Documentation:
  GGUF_QUANTIZATION_README.md   - GGUF quantization reference (26 types)
  ONNX_QUANTIZATION_README.md   - ONNX quantization reference (18 types)
  DONE_ROADMAP_IQ_QUANTIZE.md   - Completed IQ quantization roadmap
  DONE_ROADMAP_ONNX_QUANTIZE.md - Completed ONNX quantization roadmap

Export commands (from project root):
  # GGUF
  python main.py --export chat_model --formats gguf --quantization q8_0
  python main.py --export chat_model --formats gguf --quantization q4_k

  # ONNX (FP32)
  python main.py --export chat_model --formats onnx

  # ONNX Dynamic
  python main.py --export chat_model --formats onnx_int8
  python main.py --export chat_model --formats onnx_int4

  # ONNX Static (needs calibration)
  python main.py --export chat_model --formats onnx_static_int8
  python main.py --export chat_model --formats onnx_static_int4

  # ONNX FP8 (GPU only)
  python main.py --export chat_model --formats onnx_fp8

  # ONNX Mixed/Per-Channel
  python main.py --export chat_model --formats onnx_per_channel
  python main.py --export chat_model --formats onnx_mixed

  # Multiple formats
  python main.py --export chat_model --formats gguf,onnx,onnx_int8

Direct CLI:
  # GGUF
  python models/exported/convert_gguf.py <hf_dir> --outfile output.gguf --outtype q8_0

  # ONNX
  python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx
  python models/exported/convert_onnx.py <hf_dir> --outfile output.onnx --quant-type int8

After GGUF export, run convert_to_gguf.bat (Windows) or bash convert_to_gguf.sh (Linux/Mac)
to convert the HuggingFace folder to a .gguf file usable with llama.cpp or Ollama.
