# AGENTS.md - MyIAModelChat Project Agents & Roles

## Project Overview

MyIAModelChat is an advanced conversational AI system built with PyTorch, featuring GPT-2 Transformer architecture, bilingual support (English/Spanish), intent recognition, sentiment analysis, chain-of-thought reasoning (`<thinking>`), and multi-source data processing (AIML, PDF, EPUB, HuggingFace, Web, CSV).

---

## Project Structure

`
MyIAModelChat/
├── main.py                              # Primary entry point (CLI) - ONLY file that processes args
├── config.py                            # Centralized config (OLLAMA_MODEL, OLLAMA_URL)
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
│   │   ├── loader.py                    # AIML file processing
│   │   └── thinking.py                  # AIML thinking
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
`

---

## Key Components

### Core Python Modules

| Module | Purpose |
|--------|---------|
| main.py | CLI entry point - parses args, orchestrates operations |
| commons/model/chatmodel.py | GPT-2 Transformer architecture |
| commons/dialogue/dialogmanager.py | Intent/sentiment analysis, temperature adjustment |
| commons/dataset/chatdataset.py | PyTorch Dataset for tokenized chat data |
| commons/tokenizer/bpe_tokenizer.py | SentencePiece BPE tokenizer (multilingual) |
| commons/registry/model_registry.py | Model discovery and listing |
| commons/registry/model_merge.py | Model merging by weight averaging |
| commons/registry/model_export.py | Export to GGUF, ONNX |
| dataset_preparer/data_preparer.py | Multi-source data loading |
| dataset_preparer/aiml/loader.py | AIML file parsing |
| 	raining/trainer.py | Training pipeline with checkpointing |
| inference/chat_engine.py | Chat interface and inference |

### Data Flow

`
User Input → Tokenizer → Model → Logits → Decoding → Response
                ↓
        Intent Classifier (BERT)
                ↓
        Sentiment Analyzer (BERT)
                ↓
        Temperature Adjustment + Prompt Enrichment
                ↓
        GPT-2 Transformer Generation
`

---

## Development Guidelines

### Code Style
- Python 3.12+ compatible (tested with 3.14)
- Type hints recommended for new functions
- Docstrings for public functions (brief, one-line)
- No emojis in code unless explicitly requested
- Prefer editing existing files over creating new ones

### Testing

`ash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_chatmodel.py

# Manual testing
python manual_test.py
`

### Training Commands

`ash
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
`

### API Server

`ash
# Start FastAPI server
python main_chat.py

# API endpoints
POST /v1/chat/completions  # Chat completion
GET  /v1/models            # List available models
`

---

## Architecture Patterns

### Configuration Dataclasses
- TrainingConfig: All training parameters (no CLI args)
- ChatConfig: All inference parameters (no CLI args)

### Class Inheritance
- ThinkingGenerator base class with source-specific implementations
- ChatEngine encapsulates all inference logic
- Trainer encapsulates all training logic

### Module Organization
- commons/: Reusable code (model, tokenizer, dialogue, registry)
- dataset_preparer/: Data loading and preprocessing
- 	raining/: Training pipeline
- inference/: Inference and chat
- envAIModels/: FastAPI server (separate concern)

---

## Task Management

### Task Lifecycle
`
open → in_progress → done
           ↓
        blocked → open
           ↓
        abandoned
`

### Task Best Practices
- Mark task start before working
- Mark task done immediately after completion
- Keep one task in_progress when working solo
- Use lock when waiting on external dependency

---

## Memory System

### Project Memory
- **Location**: ~/.mimocode/memory/projects/<project-id>/MEMORY.md
- **Purpose**: Persistent cross-session knowledge
- **Content**: Architecture decisions, rules, durable facts

### Session Checkpoint
- **Location**: ~/.mimocode/memory/sessions/<session-id>/checkpoint.md
- **Purpose**: Current session state
- **Content**: Active intent, task tree, current work, files, learnings

---

## Common Tasks

### Adding New Data Source
1. Create subfolder in `dataset_preparer/` (e.g., `dataset_preparer/newsource/`)
2. Add loader in `dataset_preparer/newsource/loader.py`
3. Add thinking generator in `dataset_preparer/newsource/thinking.py`
4. Add CLI flag in `main.py`
5. Integrate with training pipeline

### Modifying Model Architecture
1. Edit commons/model/chatmodel.py
2. Update commons/dataset/chatdataset.py if input format changes
3. Retrain with python main.py --train

### Extending Dialogue Context
1. Modify prompt enrichment in commons/dialogue/dialogmanager.py
2. Add context window for longer conversations
3. Retrain model with new context length

---

## Dependencies

### Core
- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- sentencepiece (BPE tokenizer)
- datasets (Hugging Face datasets)
- dill / multiprocess (serialization)

### Optional
- spacy (professional sentence tokenization)
- langdetect (language detection)
- datasketch (MinHash deduplication)
- PyPDF2 (PDF extraction)
- ebooklib (EPUB support)
- ollama (external teacher for thinking generation)

---

## Security Notes

- Never commit .env files or API keys
- Model weights in checkpoints/ are large; use .gitignore
- Virtual environments (envMyIAModelChat/) excluded from version control

---

## Recent Improvements

- Restructured codebase into modular packages (commons, dataset_preparer, training, inference)
- Dataclass-based configuration (TrainingConfig, ChatConfig)
- Removed CLI dependency from training/inference modules
- Improved code organization for better maintainability

---

*Last updated: 2026-07-28*
