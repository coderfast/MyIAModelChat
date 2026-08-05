Exported Models Directory
=========================

This directory contains exported models and conversion tools.

Contents:
  convert.py          - HuggingFace to GGUF converter (from llama.cpp)
  requirements.txt    - Python dependencies for conversion
  chat_model_hf/      - Example exported model (HuggingFace format)

Export commands (from project root):
  python main.py --export chat_model --formats gguf
  python main.py --export chat_model --formats onnx
  python main.py --export chat_model --formats onnx_int8
  python main.py --export chat_model --formats gguf,onnx,onnx_int8

After GGUF export, run convert_to_gguf.bat (Windows) or bash convert_to_gguf.sh (Linux/Mac)
to convert the HuggingFace folder to a .gguf file usable with llama.cpp or Ollama.
