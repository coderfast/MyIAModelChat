# AGENTS.md - MyIAModelChat Project Agents & Roles

## Project Overview

MyIAModelChat is an advanced conversational AI system built with PyTorch, featuring GPT-2 Transformer architecture, multilingual support (English/Spanish and all EU and european languages), intent recognition, sentiment analysis, chain-of-thought reasoning (`<thinking>`), and multi-source data processing (AIML, PDF, EPUB, HuggingFace, Web, CSV).

**GPT-2 Standard Tokens (consolidated):**
- `<|problem|>` - Question/problem prefix
- `<|thinking|>` - Reasoning prefix (opens thinking block)
- `<|final|>` - Answer prefix (also closes thinking block)
- `<|user|>` - User prefix (agentic)
- `<|assistant|>` - Assistant prefix (agentic)
- `<|system|>` - System instructions prefix
- `<|end|>` - End of message/turn
- `<|sep|>` - Intra-message separator
- `<tool_call>` - Tool call start
- `</tool_call>` - Tool call end
- `<|tool_result|>` - Tool result prefix (no closing tag; runs until `<|end|>`)

**Legacy tokens (deprecated, consolidated):** `<|context|>`→`<|problem|>`,
`<|answer|>`→`<|final|>`, `<thinking>`→`<|thinking|>`, `</thinking>`→`<|final|>`,
`<observation>`/`</observation>`→`<|tool_result|>`. See `PLAN_SPECIAL_TOKENS.md`.

---

## Project Structure

```
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
│   ├── registry/                        # Model management
│   │   ├── __init__.py
│   │   ├── model_registry.py            # Model discovery, listing
│   │   ├── model_merge.py               # Model merging
│   │   ├── model_export.py              # Export to GGUF/ONNX
│   │   └── model_downloader.py          # HuggingFace downloader
│   ├── tools/                           # Agentic tool system
│   │   ├── __init__.py
│   │   ├── tool_registry.py             # Tool registration and platform detection
│   │   ├── tool_executor.py             # Tool execution engine
│   │   ├── permission_manager.py        # Tool permission management
│   │   └── platform_detector.py         # Platform-specific tool filtering
│   └── utils/                           # Shared utilities
│       ├── __init__.py
│       └── device_utils.py              # GPU detection, device resolution
│
├── dataset_preparer/                    # Data preparation
│   ├── __init__.py
│   ├── data_preparer.py                 # Main data preparer
│   ├── source_validators.py             # Data quality validators
│   ├── thinking_generators.py           # Thinking generation base
│   ├── thinking_quality.py              # Quality validation
│   ├── thinking_engine.py               # NLP-based chain-of-thought reasoning
│   ├── generate_thinking_data.py        # Thinking data generation
│   ├── contamination/                   # Data contamination filtering
│   │   ├── __init__.py
│   │   ├── filters.py                   # Noise/quality filters
│   │   ├── dedup.py                     # Deduplication
│   │   ├── balance.py                   # Class balancing
│   │   ├── audit.py                     # Data audit
│   │   └── leakage.py                   # Leakage detection
│   ├── agent/                           # Agentic data generation
│   │   ├── __init__.py
│   │   ├── moe_data.py                  # MoE (Mixture of Experts) data
│   │   ├── quality.py                   # Agent data quality validation
│   │   └── thinking.py                  # Agent thinking generation
│   ├── aiml/
│   │   ├── __init__.py
│   │   ├── parser.py                    # AIML Parser (resolve elements, wildcards, quality)
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
├── ServerFastAPI/                         # FastAPI server
│   ├── __init__.py
│   ├── app.py
│   ├── routers_api.py
│   ├── routers_v1.py
│   ├── schemas.py
│   ├── utils.py
│   ├── model.py
│   └── model_metadata.py
│
├── models/                              # Trained models
│   ├── intent/                          # BERT intent classifier
│   ├── sentiment/                       # BERT sentiment analyzer
│   └── exported/                        # Exported models
├── dataset_cache/                       # Cached datasets
├── datasets_source/                     # Data sources
├── tests/                               # Test suite (21 test files)
├── requirements.txt                     # Python dependencies
├── APP_CACHE_VIEWER/                    # PyQt5 dataset cache viewer
├── DOCS/                                # Documentation
└── ROADMAPS/                            # Project roadmaps
```

---

## Key Components

### Core Python Modules

| Module | Purpose |
|--------|---------|
| main.py | CLI entry point - parses args, orchestrates operations |
| config.py | Centralized config (OLLAMA_MODEL, OLLAMA_URL) |
| commons/model/chatmodel.py | GPT-2 Transformer architecture |
| commons/dialogue/dialogmanager.py | Intent/sentiment analysis, temperature adjustment |
| commons/dataset/chatdataset.py | PyTorch Dataset for tokenized chat data |
| commons/tokenizer/bpe_tokenizer.py | SentencePiece BPE tokenizer (multilingual) |
| commons/registry/model_registry.py | Model discovery and listing |
| commons/registry/model_merge.py | Model merging by weight averaging |
| commons/registry/model_export.py | Export to GGUF, ONNX |
| commons/utils/device_utils.py | GPU detection, device resolution |
| commons/tools/tool_registry.py | Tool registration and platform detection |
| commons/tools/tool_executor.py | Tool execution engine |
| commons/tools/permission_manager.py | Tool permission management |
| commons/tools/platform_detector.py | Platform-specific tool filtering |
| dataset_preparer/data_preparer.py | Multi-source data loading |
| dataset_preparer/thinking_engine.py | NLP-based chain-of-thought reasoning |
| dataset_preparer/thinking_quality.py | Thinking quality validation |
| dataset_preparer/agent/moe_data.py | MoE (Mixture of Experts) data generation |
| dataset_preparer/agent/quality.py | Agent data quality validation |
| dataset_preparer/agent/thinking.py | Agent thinking generation |
| dataset_preparer/aiml/parser.py | AIML Parser (resolve elements, wildcards, quality) |
| dataset_preparer/aiml/loader.py | AIML file processing |
| dataset_preparer/contamination/ | Data contamination filtering pipeline |
| training/trainer.py | Training pipeline with checkpointing |
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

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_chatmodel.py

# Run thinking tests
pytest tests/test_thinking.py tests/test_thinking_engine.py tests/test_thinking_quality_v2.py -v

# Run agent/tool tests
pytest tests/test_agent_data.py tests/test_tool_registry.py tests/test_tool_executor.py tests/test_permission_system.py -v

# Run MoE tests
pytest tests/test_moe.py -v

# Run tokenizer tests
pytest tests/test_bpe_tokenizer.py -v

# Run AIML parser tests
pytest tests/test_aiml_parser.py -v
```

### Training Commands

```bash
# Prepare data with BPE tokenizer
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000

# Train model (default checkpoint name)
python main.py --train --epochs 30

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

# Generate thinking data with NLP engine
python main.py --prepare-data --aiml --generate-thinking --thinking-mode nlp

# Generate agentic data with tool calls
python main.py --prepare-data --aiml --generate-agent-data --agent-ratio 0.3

# Enable Mixture of Experts (MoE) training
python main.py --train --moe-enabled --moe-num-experts 4 --moe-top-k 2

# Apply contamination filtering
python main.py --prepare-data --aiml --filter-noise --filter-contamination --filter-dedup --filter-balance

# Validate data sources
python main.py --prepare-data --aiml --validate-sources

# Text chunking for large documents
python main.py --prepare-data --pdf --enable-chunking --chunk-max-tokens 512

# PDF/EPUB samples are whole paragraphs (or whole EPUB chapters when they fit
# the context window). Repeated headers/footers and standalone page numbers are
# removed automatically via cross-page detection (clean_page_artifacts).
```

### API Server

```bash
# Start FastAPI server (from ServerFastAPI/)
python -m ServerFastAPI.app

# API endpoints
POST /v1/chat/completions  # Chat completion
GET  /v1/models            # List available models
```

### Agentic Tool System

The project includes an agentic tool system for extending model capabilities:

```python
# Tool registration and execution
from commons.tools.tool_registry import ToolRegistry, register_default_tools
from commons.tools.tool_executor import execute_tool_call
from commons.tools.permission_manager import PermissionManager
from commons.tools.platform_detector import detect_platform

# Default tools: calculator, current_date, word_count, get_platform, read_file, list_directory, web_search
```

**Key Features:**
- Platform-aware tool filtering (Windows/Linux/macOS)
- Permission management for shell/file operations
- Tool call parsing from model output (`<tool_call>` tokens)
- Extensible registry for custom tools

### MoE (Mixture of Experts) Architecture

Support for Mixture of Experts training for improved model capacity:

```bash
# Enable MoE during training
python main.py --train --moe-enabled --moe-num-experts 4 --moe-top-k 2

# MoE data generation
python main.py --prepare-data --aiml --generate-agent-data --agent-ratio 0.3
```

**MoE Configuration:**
- `--moe-enabled`: Enable MoE architecture
- `--moe-num-experts`: Number of experts per layer (default: 4)
- `--moe-top-k`: Experts to route each token to (default: 2)
- `--moe-load-balance-weight`: Load balancing loss weight (default: 0.01)
- `--moe-freeze-attention`: Freeze attention layers during MoE training

## Architecture Patterns

### Configuration Dataclasses
- TrainingConfig: All training parameters (no CLI args)
- ChatConfig: All inference parameters (no CLI args)

### Class Inheritance
- ThinkingGenerator base class with source-specific implementations
- ChatEngine encapsulates all inference logic
- Trainer encapsulates all training logic

### Module Organization
- commons/: Reusable code (model, tokenizer, dialogue, registry, utils)
- dataset_preparer/: Data loading, preprocessing, and contamination filtering
- training/: Training pipeline
- inference/: Inference and chat
- ServerFastAPI/: FastAPI server (separate concern)

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
- Python 3.12+ (tested with 3.14)
- PyTorch 2.0+ (torch, torchvision, torchaudio, torchtext)
- transformers (Hugging Face)
- sentencepiece (BPE tokenizer)
- datasets (Hugging Face datasets)
- numpy
- onnx (model export)
- huggingface_hub (model download)

### API Server
- fastapi
- uvicorn
- pydantic

### Data Processing
- python-aiml (AIML parsing)
- pypdf (PDF extraction)
- ebooklib (EPUB support)
- trafilatura (web scraping)
- beautifulsoup4 (HTML parsing)
- requests (HTTP client)

### Optional
- spacy (professional sentence tokenization)
- langdetect (language detection)
- datasketch (MinHash deduplication)
- keyboard (hotkey support)
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
- AIML 2.0 Smart Parser: resolves `<srai>`, `<random>`, wildcards, HTML tags
- Special Tokens Consolidation: unified legacy tokens to GPT-2 standard set, added `<|system|>`/`<|end|>`/`<|sep|>`, `<|tool_result|>` prefix-only format, dataset migrator (`dataset_preparer/migrator.py`)
- Contamination Filtering Pipeline: 6 phases (noise, quality, dedup, balance, leakage, language)
- Source Validators: improved detection of URLs, emails, phone numbers, code, boilerplate
- Agentic Tool System: platform-aware tool registry, permission management, tool call parsing
- MoE (Mixture of Experts) architecture support for training
- Enhanced testing suite with 21 test files covering tools, agents, MoE, and thinking
- PDF/EPUB Whole-Paragraph Extraction: PDF samples are whole paragraphs; EPUB uses native chapters (whole chapter when it fits, else paragraphs via BeautifulSoup); repeated headers/footers and standalone page numbers removed via cross-page detection (`clean_page_artifacts`)

---

*Last updated: 2026-08-17*
