# APP_TECHNICALSTACK - MyIAModelChat

## Technology Stack Overview

```
+======================================================+
|                   APPLICATION                         |
|  MyIAModelChat - Advanced Conversational AI System   |
+======================================================+
        |               |               |
        v               v               v
+---------------+ +-----------+ +---------------+
|   Languages   | | Frameworks| |   Libraries   |
+---------------+ +-----------+ +---------------+
| Python 3.12+  | | PyTorch   | | transformers  |
|               | | FastAPI   | | sentencepiece |
|               | | Uvicorn   | | datasets      |
+---------------+ +-----------+ +---------------+
```

---

## 1. Programming Languages

| Language | Version | Usage |
|----------|---------|-------|
| **Python** | 3.12+ (tested with 3.14) | All source code, CLI, API, training, data processing |
| **Shell Script** | Bash/PowerShell | Export converter scripts (GGUF via llama.cpp) |

---

## 2. Core Frameworks

### 2.1 Deep Learning

| Framework | Version | Purpose | Key Components Used |
|-----------|---------|---------|---------------------|
| **PyTorch** | 2.0+ | Neural network engine | `nn.Module`, `DataLoader`, `autograd`, `AMP`, `torch.onnx` |
| **transformers** (HuggingFace) | latest | Pre-built model architectures | `GPT2LMHeadModel`, `GPT2Config`, `pipeline` (sentiment/intent) |
| **sentencepiece** | latest | Subword tokenization | BPE model training, encoding, decoding |
| **llama-cpp-python** | latest | GGUF inference | `Llama` class for GGUF model loading/generation |

### 2.2 Web/API

| Framework | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | latest | REST API server (both inference servers) |
| **Uvicorn** | latest | ASGI server for FastAPI |
| **Pydantic** | latest | Request/response validation, data models |
| **Starlette** | (bundled) | ASGI foundation (via FastAPI) |

---

## 3. Data Processing & Loading

### 3.1 Dataset Management

| Library | Purpose | Used In |
|---------|---------|---------|
| **datasets** (HuggingFace) | Dataset loading, caching, and saving | `dataset_preparer/data_preparer.py`, `training/trainer.py`, `dataset_preparer/aiml/loader.py` |
| **pypdf** | PDF text extraction | `dataset_preparer/data_preparer.py` |
| **ebooklib** | EPUB e-book parsing | `dataset_preparer/data_preparer.py` |
| **beautifulsoup4** | HTML parsing (EPUB/web) | `dataset_preparer/data_preparer.py`, `dataset_preparer/web/scraper.py` |
| **trafilatura** | Web content extraction | `dataset_preparer/web/scraper.py` |
| **requests** | HTTP fetching | `dataset_preparer/web/scraper.py` |
| **csv** (stdlib) | CSV file parsing | `dataset_preparer/data_preparer.py` |

### 3.2 Text Processing

| Library | Type | Purpose |
|---------|------|---------|
| **spaCy** | Optional | Professional sentence tokenization (es/en) |
| **langdetect** | Optional | Language detection (50+ languages) |
| **datasketch** | Optional | MinHash LSH for near-duplicate detection |
| **re** (stdlib) | Built-in | Regex pattern matching, text cleaning |
| **unicodedata** (stdlib) | Built-in | NFKC Unicode normalization |

---

## 4. Storage & Persistence

### 4.1 Model Storage

| Format | Library | Extension | Purpose |
|--------|---------|-----------|---------|
| PyTorch checkpoint | torch | `.pth` | Trained model weights + optimizer state |
| ONNX | onnx + torch.onnx | `.onnx` | Cross-platform model export |
| ONNX Int8 | onnxruntime | `.onnx` | Quantized model for edge deployment |
| GGUF | llama.cpp scripts | `.gguf` | Optimized for Ollama/llama.cpp |
| HuggingFace format | transformers | `pytorch_model.bin` + `config.json` | Intermediate for GGUF conversion |

### 4.2 Dataset Cache

| Format | Library | Location |
|--------|---------|----------|
| HuggingFace Dataset on disk | datasets | `dataset_cache/prepared_dataset/` |
| SentencePiece model | sentencepiece | `dataset_cache/sentencepiece.model` |
| Pickle metadata | pickle | `dataset_cache/dataset_stats.pkl`, `cache_metadata.pkl` |
| Tokenizer vocabulary | json | `checkpoints/tokenizer_vocab.json` |

### 4.3 Source Data

| Type | Location | Format |
|------|----------|--------|
| AIML files | `datasets_source/aiml/` | `.aiml` (XML) + `.datasets` (pickle) |
| CSV datasets | `datasets_source/csv/` | `.csv` |
| PDF documents | `datasets_source/pdf/` | `.pdf` |
| EPUB e-books | `datasets_source/epub/` | `.epub` |
| Web scrape config | `datasets_source/web/` | `.txt` (URL list) |

---

## 5. Python Standard Library Usage

| Module | Purpose |
|--------|---------|
| `argparse` | CLI argument parsing |
| `threading` | Training interrupt handling, daemon threads |
| `multiprocessing` | Parallel data processing |
| `msvcrt` | Windows console keyboard input |
| `csv` | CSV data reading |
| `json` | Metadata serialization |
| `pickle` | Cache serialization |
| `psutil` | System memory monitoring |
| `re` | Text pattern matching |
| `unicodedata` | Unicode normalization |
| `os` / `pathlib` | File system operations |
| `typing` | Type hints |

---

## 6. Hardware & Platform Support

### 6.1 Compute

| Backend | Support | Details |
|---------|---------|---------|
| **CPU** | Full | Multi-threaded (OMP/MKL), Intel/AMD |
| **CUDA GPU** | Full | Mixed precision (AMP via `GradScaler`), Tesla K80 supported |
| **MPS (Apple)** | Partial | Via `torch.backends.mps` |

### 6.2 Operating Systems

| OS | Support |
|----|---------|
| Windows | Full (msvcrt console input, NT paths) |
| Linux | Full (alternative input handling) |
| macOS | Partial (MPS support, no msvcrt) |

### 6.3 Memory Management

| Mechanism | Description |
|-----------|-------------|
| `max-ram-fraction` | Limits RAM usage (default: 75%) |
| `psutil` monitoring | Adjusts workers, warns at limits |
| Gradient checkpointing | Reduces memory during training |
| Memory cleanup interval | Periodic cache clearing (every 10 batches) |

---

## 7. API Protocols

### 7.1 REST API (Both Servers)

| Aspect | Detail |
|--------|---------|
| Protocol | HTTP/1.1 |
| Serialization | JSON |
| Streaming | Server-Sent Events (SSE) |
| Port | 11434 (default) |
| Auth | None (local-only) |

### 7.2 Endpoints Summary

**inference/chat_engine.py server (PyTorch GPT-2):**
| Endpoint | Method | OpenAI Compatible |
|----------|--------|-------------------|
| `/v1/health` | GET | - |
| `/v1/models` | GET | Yes |
| `/v1/version` | GET | - |
| `/v1/chat/completions` | POST | Yes |
| `/api/chat/completions` | POST | Yes (alias) |
| `/api/chat` | POST | No (legacy) |
| `/v1/embeddings` | POST | Yes |
| `/api/generate` | POST | No |

**ServerFastAPI server (GGUF/llama.cpp):**
| Endpoint | Method | OpenAI Compatible |
|----------|--------|-------------------|
| `/health` | GET | - |
| `/api/version` | GET | - |
| `/api/models` / `/api/tags` | GET | - |
| `/api/generate` / `/api/chat` | POST | No |
| `/api/chat/completions` | POST | Yes |
| `/v1/health` | GET | - |
| `/v1/models` | GET | Yes |
| `/v1/completions` | POST | Yes |
| `/v1/chat/completions` | POST | Yes |

---

## 8. Development & Testing Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **pytest** | Unit and integration testing | `pytest tests/` |
| **Ollama** | GGUF model serving | `ollama create <name> -f Modelfile` |
| **llama.cpp** | GGUF conversion | `python convert.py <hf_dir> --outfile <model>.gguf` |

---

## 9. Deployment

### 9.1 Inference Server Deployment

```bash
# Start PyTorch GPT-2 server
python -m ServerFastAPI.app --port 11434

# Start GGUF server
cd ServerFastAPI
# Set MODEL_GGUF_PATH in .env
uvicorn app:app --host 0.0.0.0 --port 11435
```

### 9.2 Ollama Integration

```bash
# Export model to GGUF
python main.py --export ciencias_naturales --formats gguf

# Create Ollama model
ollama create mi-model -f Modelfile.ciencias_naturales

# Run with Ollama
ollama run mi-model
```

---

## 10. Dependencies Matrix

| Package | Required? | Training | Inference | Data Prep | Export |
|---------|-----------|----------|-----------|-----------|--------|
| `torch` | Yes | X | X | - | X |
| `transformers` | Yes | X | X | - | X |
| `sentencepiece` | Yes | X | X | X | - |
| `datasets` | Yes | X | - | X | - |
| `numpy` | Yes | X | X | X | X |
| `fastapi` | No* | - | X | - | - |
| `uvicorn` | No* | - | X | - | - |
| `pydantic` | No* | - | X | - | - |
| `pypdf` | No | - | - | X | - |
| `ebooklib` | No | - | - | X | - |
| `beautifulsoup4` | No | - | - | X | - |
| `trafilatura` | No | - | - | X | - |
| `requests` | No | - | - | X | - |
| `llama-cpp-python` | No | - | X | - | - |
| `huggingface_hub` | No | - | X | X | - |
| `onnx` / `onnxruntime` | No | - | - | - | X |
| `spacy` | No | - | - | X | - |
| `langdetect` | No | - | - | X | - |
| `datasketch` | No | - | - | X | - |
| `python-aiml` | Yes | - | - | X | - |
| `psutil` | Yes | X | X | X | - |
| `keyboard` | No | X | - | - | - |

*Required for API server mode only.

---

## 11. Version Compatibility

| Component | Min Version | Recommended | Notes |
|-----------|-------------|-------------|-------|
| Python | 3.12 | 3.14 | Type hints, dict ordering |
| PyTorch | 2.0 | 2.1+ | AMP, torch.onnx improvements |
| sentencepiece | 0.1.99 | 0.2+ | BPE training stability |
| transformers | 4.30 | 4.35+ | GPT2Config, pipeline API |
| CUDA | 11.0 | 11.8+ | Mixed precision support |
| ONNX Runtime | 1.14 | 1.16+ | Quantization APIs |

---

*Generated from codebase analysis - reflects the current state of the project.*
