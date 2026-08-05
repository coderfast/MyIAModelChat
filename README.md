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
- **Chain-of-Thought Reasoning**: Optional `<thinking>` reasoning in training and inference
- **Model Library**: Train, combine, and export multiple independent models

### Data Sources & Processing
- **AIML 2.0 Smart Parser**: Resolves `<srai>`, `<random>`, wildcards, `<thinking>`, HTML tags
- **PDF Text Extraction**: Automatic content extraction from PDF documents
- **EPUB E-Book Support**: Full e-book parsing and integration
- **Hugging Face Datasets**: Integration with pre-trained datasets
- **Web Scraping**: Documentation crawling with trafilatura + BeautifulSoup
- **CSV Datasets**: Curated QA pairs from CSV files
- **Dataset Caching**: 12x faster training with intelligent caching system
- **Contamination Filtering**: 6-phase pipeline (noise, quality, dedup, balance, leakage, language)

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
python main.py --train --epochs 30
```

### 3. Start Chatting
```bash
python main.py --chat
```

---

## Project Structure

```
MyIAModelChat/
├── main.py                              # Primary entry point (CLI) - ONLY file that processes args
│
├── commons/                             # Shared code (reusable across projects)
│   ├── __init__.py
│   ├── model/                           # Model architecture
│   │   ├── __init__.py
│   │   └── chatmodel.py                 # GPT-2 Transformer
│   ├── tokenizer/                       # Tokenization
│   │   ├── __init__.py
│   │   └── bpe_tokenizer.py             # SentencePiece BPE
│   ├── dialogue/                        # Dialogue management
│   │   ├── __init__.py
│   │   └── dialogmanager.py             # DialogueManager with intent/sentiment
│   ├── dataset/                         # PyTorch datasets
│   │   ├── __init__.py
│   │   └── chatdataset.py               # ChatDataset
│   └── registry/                        # Model management
│       ├── __init__.py
│       ├── model_registry.py            # Model discovery, listing
│       ├── model_merge.py               # Model merging
│       ├── model_export.py              # Export to GGUF/ONNX
│       └── model_downloader.py          # HuggingFace downloader
│
├── dataset_preparer/                    # Data preparation
│   ├── __init__.py
│   ├── data_preparer.py                 # Main data preparer
│   ├── source_validators.py             # Data quality validators
│   ├── thinking_generators.py           # Thinking generation base
│   ├── thinking_quality.py              # Quality validation
│   ├── generate_thinking_data.py        # Thinking data generation
│   ├── aiml/
│   │   ├── __init__.py
│   │   ├── parser.py                      # AIML Parser (resolve elements, wildcards, quality)
│   │   ├── loader.py                      # AIML file processing
│   │   └── thinking.py                    # AIML thinking
│   ├── contamination/                     # Contamination filtering pipeline
│   │   ├── __init__.py
│   │   ├── filters.py                     # Noise + Quality filters
│   │   ├── dedup.py                       # Cross-source deduplication
│   │   ├── balance.py                     # Source balance control
│   │   ├── leakage.py                     # Train/test leakage detection
│   │   └── audit.py                       # Audit reporting
│   ├── pdf/
│   │   ├── __init__.py
│   │   └── thinking.py                  # PDF thinking
│   ├── epub/
│   │   ├── __init__.py
│   │   └── thinking.py                  # EPUB thinking
│   ├── csv/
│   │   ├── __init__.py
│   │   └── thinking.py                  # CSV thinking
│   ├── hf/
│   │   ├── __init__.py
│   │   └── thinking.py                  # HuggingFace thinking
│   └── web/
│       ├── __init__.py
│       ├── scraper.py                   # Web crawling
│       └── thinking.py                  # Web thinking
│
├── training/                            # Training
│   ├── __init__.py
│   └── trainer.py                       # Trainer class + TrainingConfig
│
├── inference/                           # Inference
│   ├── __init__.py
│   └── chat_engine.py                   # ChatEngine class + ChatConfig
│
├── envAIModels/                         # FastAPI server (unchanged)
│   ├── __init__.py
│   ├── app.py
│   ├── routers_api.py
│   ├── routers_v1.py
│   ├── schemas.py
│   ├── utils.py
│   ├── model.py
│   └── model_metadata.py
│
├── manual_test.py                       # Manual testing
├── tests/                               # Test suite
├── checkpoints/                         # Model checkpoints
├── models/                              # Trained models
│   └── exported/                        # Exported models
├── dataset_cache/                       # Cached datasets
└── datasets_source/                     # Data sources
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

### PyTorch GPT-2 Server (envAIModels/)

```bash
cd envAIModels
python -m uvicorn app:app --host 127.0.0.1 --port 11434
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

---

## Key Modules Reference

### commons/model/chatmodel.py
GPT-2 Transformer architecture for conversational AI.

```python
from commons.model.chatmodel import ChatModel

model = ChatModel(tokenizer, embed_size=256, num_layers=4)
```

### commons/tokenizer/bpe_tokenizer.py
SentencePiece BPE tokenizer wrapper (multilingual).

```python
from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper

tokenizer = SentencePieceTokenizerWrapper(model_path)
encoded = tokenizer.encode("Hello, world!")
```

### commons/dialogue/dialogmanager.py
Dialogue management with intent/sentiment analysis.

```python
from commons.dialogue.dialogmanager import DialogueManager

dm = DialogueManager(model, device, tokenizer, intent_classifier, sentiment_analyzer)
response = dm.generate_response("Hello!")
```

### commons/registry/model_registry.py
Model discovery, listing, and validation.

```python
from commons.registry.model_registry import scan_models, list_models_cli

models = scan_models()  # Returns dict of available models
list_models_cli()       # Prints models to console
```

### commons/registry/model_merge.py
Model merging by weighted averaging.

```python
from commons.registry.model_merge import merge_models, merge_from_names

merged_path = merge_from_names(["model1", "model2"], [0.5, 0.5])
```

### commons/registry/model_export.py
Export models to GGUF, ONNX formats.

```python
from commons.registry.model_export import export_to_onnx, export_to_gguf

export_to_onnx("model.pth")
export_to_gguf("model.pth")
```

### dataset_preparer/data_preparer.py
Multi-source data loading and preparation.

```python
from dataset_preparer.data_preparer import DataPreparer, prepare_datasets_for_training

dataset, stats = prepare_datasets_for_training(args)
```

### training/trainer.py
Model training with checkpointing.

```python
from training.trainer import Trainer, TrainingConfig

config = TrainingConfig(aiml=True, hf=True, epochs=30)
trainer = Trainer(config)
trainer.performMainTrain()
```

### inference/chat_engine.py
Chat interface and inference.

```python
from inference.chat_engine import ChatEngine, ChatConfig

config = ChatConfig(model_name="my_model")
engine = ChatEngine(config)
response = engine.generate_response("Hello!")
engine.start_chat_loop()
```

---

## Configuration

### TrainingConfig (training/trainer.py)

```python
@dataclass
class TrainingConfig:
    aiml: bool = False
    hf: bool = False
    pdf: bool = False
    epub: bool = False
    web: bool = False
    csv: bool = False
    epochs: int = 1
    checkpoint_name: str = 'chat_model'
    dataset_source: str = 'dataset_cache'
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] or None=auto
    use_vulkan: bool = False
    num_cores: int = 0
    num_threads: int = 0
    max_ram_fraction: float = 0.75
    thinking_loss_weight: float = 0.5
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64
    bpe_vocab_size: int = 8000
    onlytokenize: bool = False
    enable_chunking: bool = False
    chunk_max_tokens: int = 512
    chunk_overlap: int = 50
    enable_dedup: bool = False
    dedup_threshold: float = 0.8
    enable_quality_filter: bool = False
    min_words: int = 5
    max_words: int = 1000
    preserve_metadata: bool = False
    enable_lang_filter: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ['es', 'en'])
```

### ChatConfig (inference/chat_engine.py)

```python
@dataclass
class ChatConfig:
    model_name: Optional[str] = None
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] or None=auto
    use_vulkan: bool = False
    show_thinking: bool = False
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64
```

---

## Command Line Reference

### Operations

| Flag | Description |
|------|-------------|
| `--train` | Train the neural network model |
| `--chat` | Run interactive chat interface |
| `--prepare-data` | Prepare and validate datasets only |
| `--clear-cache` | Clear cached datasets and exit |
| `--list-models` | List available trained models |
| `--model-info NAME` | Show detailed layer info for a model |
| `--gpu-enum` | Enumerate available GPUs and exit |
| `--export NAME` | Export model to GGUF/ONNX |

### Data Sources (for --prepare-data only)

| Flag | Description |
|------|-------------|
| `--aiml` | Include AIML data from datasets_source/aiml/ |
| `--hf` | Include Hugging Face datasets |
| `--pdf` | Include PDF data from datasets_source/pdf/ |
| `--epub` | Include EPUB data from datasets_source/epub/ |
| `--web` | Include web documentation data |
| `--csv` | Include CSV data from datasets_source/csv/ |

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs NUM` | Number of training epochs | `1` |
| `--refresh-cache` | Rebuild cache from scratch | (flag) |
| `--onlytokenize` | Build vocabulary only | (flag) |
| `--bpe-vocab-size` | BPE vocabulary size | `8000` |

### Device Selection

| Flag | Description | Default |
|------|-------------|---------|
| `--cpu` | Force CPU-only execution | (flag) |
| `--gpu [N,N,...]` | Use GPU (no arg=auto, or indices) | auto |
| `--vulkan` | Force Vulkan backend | (flag) |
| `--cpu+gpu N,N,...` | CPU+GPU hybrid (split by VRAM) | - |

### System Resources

| Flag | Description | Default |
|------|-------------|---------|
| `--num_cores` | CPU cores to use | 50% of available |
| `--num_threads` | Training threads | 50% of available |
| `--max-ram-fraction` | Max RAM usage (0.0-1.0) | `0.75` |

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

<thinking>
The user is asking a simple arithmetic question.
I need to add the numbers 2 and 2.
2 + 2 = 4
</thinking>

4
```

---

## Architecture Overview

### Data Flow
```
User Input → Tokenizer → Model → Logits → Decoding → Response
                ↓
        Intent Classifier (BERT)
                ↓
        Sentiment Analyzer (BERT)
                ↓
        Temperature Adjustment + Prompt Enrichment
                ↓
        GPT-2 Transformer Generation
```

### Module Dependencies
```
main.py
├── commons/
│   ├── model/chatmodel.py
│   ├── tokenizer/bpe_tokenizer.py
│   ├── dialogue/dialogmanager.py
│   ├── dataset/chatdataset.py
│   └── registry/
│       ├── model_registry.py
│       ├── model_merge.py
│       ├── model_export.py
│       └── model_downloader.py
├── dataset_preparer/
│   ├── data_preparer.py
│   ├── aiml/
│   │   ├── loader.py
│   │   └── thinking.py
│   ├── pdf/thinking.py
│   ├── epub/thinking.py
│   ├── csv/thinking.py
│   ├── hf/thinking.py
│   ├── web/
│   │   ├── scraper.py
│   │   └── thinking.py
│   ├── thinking_generators.py
│   └── generate_thinking_data.py
├── training/
│   └── trainer.py
└── inference/
    └── chat_engine.py
```

---

## Documentation

- [APP_ARCHITECTURE.md](APP_ARCHITECTURE.md) - Complete system architecture
- [APP_TECHNICALSTACK.md](APP_TECHNICALSTACK.md) - Technology stack and dependencies
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training setup
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Neural network architecture
- [THINKING_GUIDE.md](THINKING_GUIDE.md) - Chain-of-thought reasoning
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Troubleshooting
- [AGENTS.md](AGENTS.md) - Project agents and roles

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
