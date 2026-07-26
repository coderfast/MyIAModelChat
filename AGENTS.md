# AGENTS.md - MyIAModelChat Project Agents & Roles

## Project Overview

MyIAModelChat is an advanced conversational AI system built with PyTorch, featuring GPT-2 Transformer architecture, bilingual support (English/Spanish), intent recognition, sentiment analysis, chain-of-thought reasoning, and multi-source data processing (AIML, PDF, EPUB, HuggingFace, Web, CSV).

---

## Agent Roles

### Primary Agents

#### Build Agent (Default)
- **Purpose**: Full tool access for implementation tasks
- **Permissions**: Read, write, edit, bash, search, orchestration
- **Use Case**: Writing code, fixing bugs, adding features, running tests
- **When to Use**: Most development tasks

#### Plan Agent
- **Purpose**: Read-only design mode for planning
- **Permissions**: Read-only (except plan files in `.mimocode/plans/`)
- **Use Case**: Architecture decisions, multi-file refactoring, design specifications
- **When to Use**: Non-trivial implementation work requiring planning first

### Subagents

#### Explore Agent
- **Purpose**: Fast, read-only codebase exploration
- **Permissions**: grep, glob, list, bash (read-only), webfetch, read
- **Use Case**: Finding files, searching code patterns, answering codebase questions
- **When to Use**: Search tasks requiring more than 3 queries

#### General Agent
- **Purpose**: General-purpose multi-step worker
- **Permissions**: Full tool access within project scope
- **Use Case**: Complex delegated tasks, parallel work
- **When to Use**: Heavy lifting that should be isolated from main context

---

## Project Structure

```
MyIAModelChat/
├── main.py                 # Primary entry point (CLI)
├── main_train.py           # Training pipeline
├── main_chat.py            # Chat interface + FastAPI server
├── chatmodel.py            # GPT-2 Transformer model architecture
├── dialogmanager.py        # Dialogue management with intent/sentiment
├── chatdataset.py          # PyTorch Dataset loader
├── bpe_tokenizer.py        # SentencePiece BPE tokenizer (multilingual)
├── data_preparer.py        # Multi-source data loading (AIML, PDF, EPUB, HF, Web, CSV)
├── aimlloder.py            # AIML file processing
├── model_downloader.py     # HuggingFace model downloader
├── generate_thinking_data.py  # Chain-of-thought data generation
├── model_registry.py      # Model discovery, listing, validation
├── model_merge.py         # Model merging by weight averaging
├── model_export.py        # Export to GGUF, ONNX, ONNX quantized
├── web_scraper.py         # Web crawling and scraping
├── manual_test.py         # Manual testing utilities
├── requirements.txt       # Python dependencies
├── APP_ARCHITECTURE.md    # System architecture documentation
├── APP_TECHNICALSTACK.md  # Technology stack documentation
├── checkpoints/           # Model checkpoints (legacy)
├── models/                # Trained model checkpoints (.pth)
│   └── exported/          # Exported models (.gguf, .onnx)
├── datasets_source/       # User-prepared datasets (aiml/, csv/, pdf/, epub/, web/)
├── dataset_cache/         # Cached datasets (tokenized HF Dataset + BPE model)
├── aiml_dev/              # AIML development files
├── envAIModels/           # FastAPI server (GGUF/llama_cpp)
│   ├── app.py             # FastAPI application
│   ├── routers_api.py     # API routes (/api)
│   ├── routers_v1.py      # v1 routes (/v1)
│   ├── schemas.py         # Pydantic models
│   ├── utils.py           # Utility functions
│   └── model.py           # GGUF model loading (llama-cpp-python)
├── tests/                 # Test suite
└── .mimocode/             # MiMoCode configuration
```

---

## Key Components

### Core Python Modules

| Module | Purpose |
|--------|---------|
| `chatmodel.py` | GPT-2 Transformer architecture (embedding → Transformer → logits) |
| `dialogmanager.py` | Intent classification (BERT), sentiment analysis (BERT), temperature adjustment |
| `chatdataset.py` | PyTorch Dataset for loading tokenized chat data |
| `bpe_tokenizer.py` | SentencePiece BPE tokenizer wrapper (multilingual) |
| `data_preparer.py` | Multi-source data loading (AIML, PDF, EPUB, HuggingFace) |
| `aimlloder.py` | AIML file parsing and pattern matching |
| `main_train.py` | Training pipeline with dataset handling and checkpointing |
| `main_chat.py` | Chat interface + FastAPI API server |
| `model_registry.py` | Model discovery, listing, validation |
| `model_merge.py` | Model merging by weight averaging |
| `model_export.py` | Export to GGUF, ONNX, ONNX quantized |

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

---

## Development Guidelines

### Code Style
- Python 3.12 compatible
- Type hints recommended for new functions
- Docstrings for public functions (brief, one-line)
- No emojis in code unless explicitly requested
- Prefer editing existing files over creating new ones

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_chatmodel.py

# Manual testing
python manual_test.py
```

### Training Commands

```bash
# Prepare data with BPE tokenizer
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000

# Train model (default checkpoint name)
python main.py --train --use-cache --epochs 30

# Train with custom name and dataset
python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --aiml --hf --epochs 30

# List available models
python main.py --list-models

# Chat with specific model
python main.py --chat --model ciencias_naturales

# Chat with merged models
python main.py --chat --model ciencias_naturales+programacion

# Export model
python main.py --export ciencias_naturales --formats gguf,onnx
```

### API Server

```bash
# Start FastAPI server
python main_chat.py

# API endpoints
POST /v1/chat/completions  # Chat completion
GET  /v1/models            # List available models
```

---

## Task Management

### Task Lifecycle
```
open → in_progress → done
           ↓
        blocked → open
           ↓
        abandoned
```

### Task Best Practices
- Mark task `start` before working
- Mark task `done` immediately after completion
- Keep one task `in_progress` when working solo
- Use `block` when waiting on external dependency

### When to Create Tasks
- Multi-step work (3+ steps)
- Spans multiple turns
- Will be referenced again
- Needs to be visible in session

---

## Memory System

### Project Memory
- **Location**: `~/.mimocode/memory/projects/<project-id>/MEMORY.md`
- **Purpose**: Persistent cross-session knowledge
- **Content**: Architecture decisions, rules, durable facts

### Session Checkpoint
- **Location**: `~/.mimocode/memory/sessions/<session-id>/checkpoint.md`
- **Purpose**: Current session state
- **Content**: Active intent, task tree, current work, files, learnings

### Global Memory
- **Location**: `~/.mimocode/memory/global/MEMORY.md`
- **Purpose**: User preferences across all projects

---

## Common Tasks

### Adding New AIML Patterns
1. Add `.aiml` files to `datasets_source/aiml/` directory
2. Use standard AIML XML structure
3. Run `python main.py --prepare-data --aiml` to regenerate cache

### Modifying Model Architecture
1. Edit `chatmodel.py` ChatModel class
2. Update `chatdataset.py` if input format changes
3. Retrain with `python main.py --train`

### Extending Dialogue Context
1. Modify prompt enrichment in `dialogmanager.py`
2. Add context window for longer conversations
3. Retrain model with new context length

### Adding New Data Source
1. Add loader in `data_preparer.py`
2. Add CLI flag for new source
3. Integrate with training pipeline

---

## Dependencies

### Core
- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- sentencepiece (BPE tokenizer)

### Optional
- spacy (professional sentence tokenization)
- langdetect (language detection)
- datasketch (MinHash deduplication)
- PyPDF2 (PDF extraction)
- ebooklib (EPUB support)

---

## Security Notes

- Never commit `.env` files or API keys
- Model weights in `checkpoints/` are large; use `.gitignore`
- Virtual environments (`envMyIAModelChat/`) excluded from version control

---

## Recent Improvements

- Model library system with merge and export capabilities
- GGUF/ONNX export for Ollama, llama.cpp, ONNX Runtime
- SentencePiece BPE multilingual tokenizer
- Chain-of-thought reasoning support
- Dynamic n-gram penalization

---

*Last updated: Auto-generated by MiMoCode*
