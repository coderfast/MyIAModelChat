# MyIAModelChat

Advanced Conversational AI with Multi-Source Data Support, Chain-of-Thought Reasoning, and Bilingual Capabilities.

> **Note:** This is an experimental/educational project. Not intended for production use.

---

## Features

### Core Capabilities
- **Neural Chat Model**: GPT-2 Transformer architecture for natural language generation
- **Bilingual Support**: English/Spanish tokenization with accent handling
- **Intent Recognition**: BERT-based intent classification for user queries
- **Sentiment Analysis**: BERT-based star rating detection (1-5 stars)
- **Context Management**: Maintains conversation history and coherence
- **Persona Modeling**: Customizable AI personality traits
- **Chain-of-Thought Reasoning**: Optional `<think>` reasoning in training and inference
- **Model Library**: Train, combine, and export multiple independent models

### Data Sources & Processing
- **AIML Pattern Matching**: Traditional rule-based responses from AIML files
- **PDF Text Extraction**: Automatic content extraction from PDF documents
- **EPUB E-Book Support**: Full e-book parsing and integration
- **Hugging Face Datasets**: Integration with pre-trained datasets
- **Web Scraping**: Documentation crawling with trafilatura + BeautifulSoup
- **CSV Datasets**: Curated QA pairs from CSV files
- **Dataset Caching**: 12x faster training with intelligent caching system

### Advanced Text Processing
- **Professional Sentence Tokenization**: spaCy-based sentence splitting (replaces naive `split('.')`)
- **Unicode Normalization**: NFKC normalization for consistent text
- **Text Chunking**: Split long documents into overlapping token windows
- **Deduplication**: MinHash LSH for removing duplicate/near-duplicate texts
- **Quality Filtering**: Filter low-quality texts by length, alpha ratio, spam detection
- **Metadata Preservation**: Extract and preserve document metadata (title, author, etc.)
- **Language Detection**: Filter texts by detected language (supports 50+ languages)

### Generation & Export
- **Dynamic N-gram Penalization**: Prevents repetitive responses
- **Min-Length Enforcement**: Ensures meaningful response lengths
- **Autoregressive Generation**: High-quality text generation with sampling controls
- **GGUF/ONNX Export**: Export models to Ollama, llama.cpp, ONNX Runtime
- **Model Merging**: Combine multiple trained models by weight averaging

---

## Quick Start

### 1. Prepare Your Data
```bash
python main.py --prepare-data --aiml --pdf --epub
```

### 2. Train the Model
```bash
python main.py --train --use-cache --epochs 30
```

### 3. Start Chatting
```bash
python main.py --chat
```

---

## Model Library

Train multiple independent models and use them individually or combined.

### Train a model
```bash
python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --aiml --hf --epochs 30
python main.py --train --dataset datasets_source/programacion/ --checkpoint-name programacion --aiml --hf --epochs 30
```

### List available models
```bash
python main.py --list-models
```

### Chat with a specific model
```bash
python main.py --chat --model ciencias_naturales
```

### Combine models (merge)
```bash
python main.py --chat --model ciencias_naturales+programacion
```

### Export models
```bash
# Export to GGUF for Ollama/llama.cpp
python main.py --export ciencias_naturales --formats gguf

# Export to ONNX
python main.py --export ciencias_naturales --formats onnx,onnx_int8

# Export merged model
python main.py --export ciencias_naturales+programacion --formats gguf,onnx
```

### Use with Ollama
```bash
ollama create mi-ciencias -f Modelfile.ciencias_naturales
ollama run mi-ciencias
```

---

## API Server

MyIAModelChat provides an Ollama-compatible REST API server built with FastAPI.

### PyTorch GPT-2 Server (main_chat.py)

```bash
python main_chat.py --port 11434
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/version` | GET | Version info |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/api/chat/completions` | POST | Alias for above |
| `/api/chat` | POST | Legacy chat endpoint |
| `/v1/embeddings` | POST | Text embeddings (mean pooling) |
| `/api/generate` | POST | Text generation |

**Example request:**
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ciencias_naturales",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**Streaming:**
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ciencias_naturales",
    "prompt": "Hola",
    "stream": true
  }'
```

### GGUF Server (envAIModels/)

Separate inference server for GGUF models using llama-cpp-python.

```bash
cd envAIModels
# Set MODEL_GGUF_PATH in .env or environment variable
uvicorn app:app --host 0.0.0.0 --port 11435
```

**Configuration:**
- `MODEL_GGUF_PATH` — Path to `.gguf` model file (fallback: `models/Qwen2.5-1.5B-Instruct-Q4_0.gguf`)
- Uses lazy-loading: model loads on first request, not at startup

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/version` | GET | Version info |
| `/api/models` | GET | List models |
| `/api/tags` | GET | Model tags |
| `/api/generate` | POST | Text generation (streaming) |
| `/api/chat` | POST | Chat (streaming) |
| `/api/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | Models with file sizes |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |

---

## Examples

### Basic Conversation
```
User: Hello, how are you?
AI: Hello! I'm doing well, thank you for asking. How can I help you today?

User: Tell me about machine learning
AI: Machine learning is a fascinating field of artificial intelligence that focuses on creating algorithms that can learn from data...
```

### Bilingual Support
```
User: Hola, ¿cómo estás?
AI: ¡Hola! Estoy bien, gracias por preguntar. ¿En qué puedo ayudarte?

User: ¿Qué es el aprendizaje automático?
AI: El aprendizaje automático es un campo fascinante de la inteligencia artificial...
```

### Chain-of-Thought Reasoning
```
User: What is 2+2?

<think>
The user is asking a simple arithmetic question.
I need to add the numbers 2 and 2.
2 + 2 = 4
</think>

4
```

---

## Project Structure

```
MyIAModelChat/
├── main.py                 # Main entry point with all CLI commands
├── main_train.py           # Training pipeline
├── main_chat.py            # Chat interface + FastAPI server
├── dialogmanager.py        # Response generation with intent/sentiment
├── data_preparer.py        # Multi-source data loading (AIML, PDF, EPUB, HF, Web, CSV)
├── bpe_tokenizer.py        # SentencePiece BPE tokenizer (multilingual)
├── chatmodel.py            # GPT-2 model architecture
├── chatdataset.py          # PyTorch Dataset loader (legacy)
├── aimlloder.py            # AIML file processing
├── model_downloader.py     # HuggingFace model downloader
├── model_registry.py       # Model discovery, listing, validation
├── model_merge.py          # Model merging by weight averaging
├── model_export.py         # Export to GGUF, ONNX, ONNX quantized
├── generate_thinking_data.py  # Chain-of-thought data generation
├── web_scraper.py          # Web crawling and scraping
├── manual_test.py          # Manual testing utilities
├── requirements.txt        # Python dependencies
├── APP_ARCHITECTURE.md     # System architecture documentation
├── APP_TECHNICALSTACK.md   # Technology stack documentation
├── ROADMAP.md              # Thinking implementation roadmap
├── checkpoints/            # Model checkpoints (legacy)
├── models/                 # Trained model checkpoints (.pth)
│   └── exported/           # Exported models (.gguf, .onnx)
├── datasets_source/        # User-prepared datasets (aiml/, csv/, pdf/, epub/, web/)
├── dataset_cache/          # Cached datasets (tokenized HF Dataset + BPE model)
├── aiml_dev/               # AIML development files
├── envAIModels/            # GGUF inference server (FastAPI + llama-cpp-python)
│   ├── app.py              # FastAPI application
│   ├── routers_api.py      # /api routes
│   ├── routers_v1.py       # /v1 routes
│   ├── schemas.py          # Pydantic models
│   ├── model.py            # GGUF model loading (lazy)
│   └── utils.py            # Prompt formatting, streaming, response building
├── tests/                  # Test suite
└── .mimocode/              # MiMoCode configuration
```

---

## Data Sources

### AIML Files
- Traditional chatbot patterns from A.L.I.C.E. project
- Rule-based responses for common queries
- Place `.aiml` files in `datasets_source/aiml/`

### PDF Documents
- Automatic text extraction using PyPDF2
- Supports complex layouts and formatting
- Place PDFs in `datasets_source/pdf/`

### EPUB E-books
- Full e-book parsing with ebooklib
- Chapter-by-chapter content extraction
- Supports metadata and structure
- Place EPUBs in `datasets_source/epub/`

### Web Scraping
- Crawls documentation sites with configurable depth and rate limit
- Uses trafilatura for main content extraction, BeautifulSoup as fallback
- Create `datasets_source/web/urls.txt` with one URL per line
- Scraped output saved to `datasets_source/web/`

### CSV Datasets
- Curated QA pairs with `input`/`output` columns
- Automatically oversampled 20x during training
- Place CSVs in `datasets_source/csv/`

### Hugging Face Datasets
- Integration with any HF dataset via `load_dataset()`
- Default: wikitext, first 1000 samples

---

## Configuration

### Training Parameters (`main_train.py`)

```python
TRAINING_CONFIG = {
    'batch_size': 4,                # Micro batch size
    'accumulation_steps': 8,        # Gradient accumulation (effective batch = 32)
    'learning_rate': 1e-3,          # Adam learning rate
    'embed_size': 256,              # Embedding dimension
    'hidden_size': 512,             # Hidden state dimension
    'num_layers': 4,                # Transformer layers
    'n_head': 4,                    # Attention heads
    'n_positions': 512,             # Max sequence length
    'grad_clip_norm': 1.0,          # Gradient clipping
    'warm_up': True,                # Learning rate warm-up
    'warm_up_ratio': 0.1,           # Warm-up fraction
    'warm_up_steps': 100,           # Warm-up steps
}
```

### Generation Settings (`dialogmanager.py`)

```python
DEFAULT_CONFIG = {
    'top_k': 50,                    # Top-k sampling
    'top_p': 0.9,                   # Nucleus sampling
    'temperature': 0.7,             # Sampling temperature
    'max_len': 128,                 # Max generation tokens
    'min_length': 3,                # Min generated tokens
    'no_repeat_ngram_size': 3,      # N-gram repetition penalty
}
```

### BPE Tokenizer Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--bpe-vocab-size` | BPE vocabulary size | `8000` |
| `--refresh-cache` | Rebuild cache from scratch | (flag) |
| `--use-cache` | Load cached dataset if exists | (flag) |

### System Resources (`main.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `--num_cores` | CPU cores to use | 50% of available |
| `--num_threads` | Training threads | Auto |
| `--max-ram-fraction` | Max RAM usage (0.0-1.0) | `0.75` |

---

## Chain-of-Thought Reasoning (`<think>`)

The model supports **chain-of-thought reasoning** using the `<think>` tag. The model learns to generate its reasoning process before the final response.

### Output Format

```
<think>
Reasoning process here...
</think>
Final clean response
```

### Generate Thinking Data

```bash
# From CSV and AIML sources
python generate_thinking_data.py --source all

# Output: datasets/thinking/thinking_data.csv
```

### Train with Thinking

```bash
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
python main.py --train --use-cache --epochs 30
```

### Inference with Thinking

**Console:**
```bash
python main.py --chat --show-thinking    # With thinking visible
python main.py --chat                     # Response only
```

**API:**
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hola"}],
    "include_thinking": true
  }'
```

See [THINKING_GUIDE.md](THINKING_GUIDE.md) for full documentation.

---

## Advanced Text Processing

MyIAModelChat includes professional text processing features for high-quality dataset preparation.

| Feature | Description | Flag |
|---------|-------------|------|
| **Sentence Tokenization** | spaCy-based sentence splitting | Automatic |
| **Unicode Normalization** | NFKC normalization for consistency | Automatic |
| **Text Chunking** | Split long texts into overlapping windows | `--enable-chunking` |
| **Deduplication** | Remove duplicate texts with MinHash LSH | `--enable-dedup` |
| **Quality Filtering** | Filter low-quality texts | `--enable-quality-filter` |
| **Metadata Preservation** | Extract document metadata | `--preserve-metadata` |
| **Language Detection** | Filter by detected language | `--enable-lang-filter` |

### Text Processing Pipeline

```
Raw Text -> Unicode Normalization -> Sentence Tokenization -> Quality Filtering
    -> Deduplication -> Language Filtering -> Chunking (optional) -> Dataset
```

### Dependencies (Optional)

```bash
# Professional sentence tokenization
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

# Language detection
pip install langdetect

# MinHash deduplication
pip install datasketch
```

---

## Performance

- **Dataset Caching**: 12x faster training iterations
- **Memory Efficient**: Optimized for CPU training with configurable RAM limits
- **Multi-threaded**: Parallel data processing
- **GPU Support**: CUDA acceleration with mixed precision (AMP)
- **Lazy Loading**: GGUF server loads models on first request

---

## Documentation

- [APP_ARCHITECTURE.md](APP_ARCHITECTURE.md) - Complete system architecture
- [APP_TECHNICALSTACK.md](APP_TECHNICALSTACK.md) - Technology stack and dependencies
- [ROADMAP.md](ROADMAP.md) - Thinking implementation roadmap and task plan
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training setup
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Neural network architecture
- [THINKING_GUIDE.md](THINKING_GUIDE.md) - Chain-of-thought reasoning
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Troubleshooting

---

## Requirements

### Core
- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- sentencepiece
- datasets
- numpy

### API Server
- fastapi
- uvicorn
- pydantic

### Data Processing
- PyPDF2
- ebooklib
- beautifulsoup4
- trafilatura
- requests

### Optional
- spacy (professional sentence tokenization)
- langdetect (language detection)
- datasketch (MinHash deduplication)
- llama-cpp-python (GGUF inference)
- onnxruntime (ONNX quantization)

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd MyIAModelChat

# Create virtual environment
python -m venv envMyIAModelChat
envMyIAModelChat\Scripts\activate  # Windows
# source envMyIAModelChat/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## Security

- Never commit `.env` files or API keys
- Model weights in `checkpoints/` and `models/` are large; use `.gitignore`
- Virtual environments (`envMyIAModelChat/`) are excluded from version control
- API servers are local-only by default (no auth)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- A.L.I.C.E. AIML project for pattern data
- Hugging Face for transformers and datasets
- PyTorch community for neural network framework
- llama.cpp project for GGUF inference
