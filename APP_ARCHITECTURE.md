# APP_ARCHITECTURE - MyIAModelChat

## System Overview

MyIAModelChat is a conversational AI system built on a **custom GPT-2 Transformer** architecture using PyTorch. It supports bilingual operation (Spanish/English), chain-of-thought reasoning via `<think>` tags, intent classification and sentiment analysis via BERT, and multi-source data ingestion (AIML, PDF, EPUB, HuggingFace, Web scraping, CSV). It provides both an interactive CLI and an Ollama-compatible REST API server (FastAPI), plus a secondary GGUF/llama.cpp inference server.

---

## 1. High-Level Architecture

```
+------------------------------------------------------------------+
|                        CLI Layer (main.py)                        |
|  Argument parsing, dispatch to submodules, interrupt handling     |
+------------------------------------------------------------------+
          |                    |                    |
          v                    v                    v
+------------------+  +------------------+  +------------------+
|  Data Pipeline   |  |  Training Engine |  |  Inference       |
| (data_preparer)  |->| (main_train)     |  | (main_chat)      |
|                  |  |                  |  |                  |
| AIML  HF  PDF    |  | GPT-2 Transformer|  | BERT Intent/Sent |
| EPUB  Web  CSV   |  | GPU/CPU          |  | Top-K/P Sampling |
| Chunking  Dedup  |  | Mixed Precision  |  | Streaming API    |
| Quality Filter   |  | Gradient Accum.  |  | Ollama-compat    |
+------------------+  +------------------+  +------------------+
                              |                    |
                              v                    v
                     +------------------+  +------------------+
                     |  Model Library   |  |  GGUF Server     |
                     |  (models/*.pth)  |  | (envAIModels/)   |
                     |                  |  |                  |
                     | Merge | Export   |  | llama.cpp        |
                     | ONNX | GGUF      |  | Ollama-API       |
                     +------------------+  +------------------+
```

---

## 2. Component Architecture

### 2.1 CLI Entry Point (`main.py`)

Central orchestrator that parses CLI arguments and dispatches to submodules.

**Dispatch Logic:**
```
main.py
  ├── --list-models  -> model_registry.list_models_cli()
  ├── --export       -> model_export.export_cli()
  ├── --clear-cache  -> DataPreparer._clear_cache()
  ├── --prepare-data -> data_preparer.prepare_datasets_for_training()
  ├── --train        -> MainTrain (daemon thread + ESC/Ctrl+C interrupt)
  └── --chat         -> MainChat (interactive loop or FastAPI server)
```

**Key CLI Flags:**
| Category | Flags |
|----------|-------|
| Operations | `--train`, `--chat`, `--prepare-data`, `--clear-cache`, `--list-models`, `--export` |
| Data Sources | `--aiml`, `--hf`, `--pdf`, `--epub`, `--web`, `--web-url`, `--web-max-pages` |
| Training | `--epochs`, `--use-cache`, `--refresh-cache`, `--bpe-vocab-size`, `--cuda-device` |
| Text Processing | `--enable-chunking`, `--enable-dedup`, `--enable-quality-filter`, `--enable-lang-filter` |
| Model | `--model`, `--checkpoint-name`, `--dataset`, `--formats` |
| CPU/RAM | `--num_cores`, `--num_threads`, `--max-ram-fraction` |

### 2.2 Data Pipeline (`data_preparer.py`)

Multi-source data ingestion, processing, and caching pipeline.

**Data Sources:**
```
AIML Files (.aiml)
  -> aimlloder.py -> .datasets pickles (HuggingFace Dataset)

HuggingFace Datasets
  -> load_dataset() -> first 1000 samples

PDF Documents
  -> PyPDF2 -> sentence splitting -> optional chunking

EPUB E-Books
  -> ebooklib -> HTML stripping -> text

Web Scraping
  -> WebDocScraper (trafilatura + BeautifulSoup) -> sentences

CSV Files
  -> csv.reader() -> oversampled 20x
```

**Processing Pipeline:**
```
Raw Text
  -> Unicode NFKC Normalization (clean_text)
  -> Sentence Splitting (spaCy sentencizer / regex fallback)
  -> Optional: Quality Filtering (word count, alpha ratio, diversity)
  -> Optional: Deduplication (MinHash LSH / exact match)
  -> Optional: Language Filtering (langdetect / heuristic)
  -> Optional: Text Chunking (overlapping token windows)
  -> Train SentencePiece BPE Tokenizer
  -> Tokenize all text -> token_ids column
  -> Save HuggingFace Dataset to disk cache
```

**Cache Structure:**
```
dataset_cache/
  prepared_dataset/          # HuggingFace Dataset on disk
  dataset_stats.pkl          # Per-source statistics
  cache_metadata.pkl         # BPE model path, vocab size
  sentencepiece.model        # Trained BPE model
  sentencepiece.vocab        # SentencePiece vocabulary
```

### 2.3 Tokenizer (`bpe_tokenizer.py`)

**Class:** `SentencePieceTokenizerWrapper`

Wraps a SentencePiece BPE model with special token support.

**Special Tokens:**
| Token | ID (default) | Purpose |
|-------|-------------|---------|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown |
| `<s>` | 2 | Start of sequence |
| `</s>` | 3 | End of sequence |
| `<think>` | varies | Chain-of-thought start |
| `</think>` | varies | Chain-of-thought end |

**Key Methods:**
- `encode(text)` / `decode(indices)` - Basic tokenization
- `batch_encode(texts)` - Batch encoding
- `get_thinking_index()` / `get_thinking_end_index()` - CoT token IDs
- `has_thinking(text)` / `split_thinking(text)` / `extract_response(text)` - Thinking utilities
- `encode_with_thinking(text)` - Thinking-aware encoding
- `save_vocabulary(filepath)` - Persist vocabulary metadata

**Vocab Size:** Configurable via `--bpe-vocab-size` (default: 8000)

### 2.4 Training Engine (`main_train.py`)

**Class:** `MainTrain`

**Training Flow:**
```
Cached Dataset (HuggingFace)
  -> Load tokenizer + metadata
  -> Build TokenPairIterableDataset (input[:-1], output[1:])
  -> DataLoader (collate_fn: pad/truncate to 512)
  -> Device setup (GPU AMP / CPU)
  -> Initialize ChatModel (GPT2LMHeadModel)
  -> Warm-up phase (10% data, max 100 batches)
  -> Training loop:
       for each epoch:
         for each batch:
           forward -> CrossEntropy loss (ignore pad)
           backward (gradient accumulation x8)
           clip_grad_norm (1.0)
           optimizer.step() + scheduler.step()
           compute thinking metrics
         save checkpoint if loss improved
  -> CSV fine-tuning (optional)
  -> Save final checkpoint
```

**Training Configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 4 | Micro batch size |
| `accumulation_steps` | 8 | Gradient accumulation (effective: 32) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `embed_size` | 256 | Embedding dimension |
| `hidden_size` | 512 | Hidden state dimension |
| `num_layers` | 4 | Transformer layers |
| `n_head` | 4 | Attention heads |
| `n_positions` | 512 | Max sequence length |
| `grad_clip_norm` | 1.0 | Gradient clipping |
| `warm_up_ratio` | 0.1 | LR warm-up fraction |

**Checkpoint Format:**
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'loss': float,
    'tokenizer': dict,          # Vocabulary metadata
    'model_name': str,
    'architecture': {           # embed_size, hidden_size, num_layers, n_head, n_positions, vocab_size
        ...
    },
    'dataset_source': str,
}
```

### 2.5 Model Architecture (`chatmodel.py`)

**Class:** `ChatModel(nn.Module)`

Wraps HuggingFace `GPT2LMHeadModel` with custom configuration.

**Architecture:**
```
Input Tokens (vocab_size)
  -> Token Embedding (embed_size=256)
  -> Position Embedding (n_positions=512)
  -> x4 GPT-2 Transformer Layers:
       -> LayerNorm
       -> Multi-Head Self-Attention (n_head=4)
       -> Residual + LayerNorm
       -> Feed-Forward (hidden_size=512)
       -> Residual
  -> LayerNorm
  -> LM Head (vocab_size) -> Logits
```

**Forward:** `input_ids -> logits` (batch, seq_len, vocab_size)

### 2.6 Dialogue Management (`dialogmanager.py`)

**Class:** `DialogueManager`

Orchestrates response generation with context-aware controls.

**Generation Pipeline:**
```
User Input
  -> BERT Intent Classification (sentiment/intent models)
  -> Sentiment stars (1-5) mapped to temperature adjustment
  -> Prompt enrichment: "Pregunta: ...\nContexto: ...\nRespuesta:"
  -> Autoregressive generation:
       token-by-token with:
         - top-k filtering (k=50)
         - top-p nucleus sampling (p=0.9)
         - temperature scaling (0.7)
         - n-gram repetition penalty (size=3)
         - min-length enforcement (3 tokens)
         - special-token abort detection
  -> Output validation (repeated n-gram check, min word count)
  -> Response
```

**Sampling Strategy:**
```python
def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    # top-k: keep only top k logits
    # top-p: keep cumulative probability mass p
    # Combine both filters

def sample_next_token(logits, banned_tokens, temperature=0.7):
    # Temperature-scaled multinomial sampling
    # Fallback to argmax if sampling fails
```

**Persona (configurable):**
```python
{
    "name": "Eduardo Piñera Aznárez",
    "age": 52,
    "occupation": "AI assistant",
    "interests": ["IT technology", "MS Office", "Libre Office", "Games", "Humanity simulation"]
}
```

### 2.7 Inference Server (`main_chat.py`)

**Class:** `MainChat`

Dual-mode: interactive CLI chat or Ollama-compatible FastAPI HTTP server.

**Singleton Pattern for API Mode:**
```python
_main_chat_instance: Optional[MainChat] = None

def get_main_chat_instance(args) -> MainChat:
    if _main_chat_instance is None:
        _main_chat_instance = MainChat(args)
    return _main_chat_instance
```

**API Endpoints (Port 11434):**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/version` | GET | Version string |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/chat/completions` | POST | Alias |
| `/api/chat` | POST | Legacy chat |
| `/v1/embeddings` | POST | Text embeddings (mean pooling) |
| `/api/generate` | POST | Text generation |

**Chat Request Format:**
```json
{
    "model": "ciencias_naturales",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false,
    "stop": [],
    "include_thinking": false
}
```

**Streaming:** SSE-based streaming with 64-char chunks.

**Model Loading:**
1. Resolve model path (searches `checkpoints/`, `models/`, root)
2. Load tokenizer from `tokenizer_vocab.json` or checkpoint
3. Load model state dict (with `weights_only=True` safety)
4. Initialize BERT pipelines for intent and sentiment

### 2.8 Model Library

#### Registry (`model_registry.py`)
- Scans `models/` for `.pth` files
- Loads metadata without full weights
- Validates architecture compatibility
- CLI pretty-print

#### Merging (`model_merge.py`)
- Weighted state dict averaging
- Architecture compatibility validation
- On-the-fly merge (model name with `+` syntax)
- Parser for `model_a:0.6+model_b:0.4`

#### Export (`model_export.py`)
| Format | Method | Output |
|--------|--------|--------|
| ONNX | `export_to_onnx()` | `.onnx` with dynamic axes |
| ONNX Int8 | `export_to_onnx_quantized()` | Quantized `.onnx` |
| GGUF | `export_to_gguf()` | HF format + converter script |

### 2.9 GGUF Inference Server (`envAIModels/`)

Parallel inference server for GGUF models via `llama-cpp-python`.

**Singleton:** `_LazyModel` proxy that lazily loads `llama_cpp.Llama` on first access.

**Configuration:** From `.env` file or environment variables (`MODEL_GGUF_PATH`, fallback `MODEL_PATH`).

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/version` | GET | Version |
| `/api/models` | GET | Model list |
| `/api/tags` | GET | Model tags |
| `/api/generate` | POST | Generation (streaming) |
| `/api/chat` | POST | Chat (streaming) |
| `/api/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | Models with file size |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |

### 2.10 Chain-of-Thought (`generate_thinking_data.py`)

Generates synthetic `<think>` reasoning blocks for QA pairs.

**Categories:** identity, greeting, question, farewell, default

**Output Format:** CSV with columns: `input`, `output`, `thinking`, `thinking_text`, `category`

**Training Integration:**
- `MainTrain._detect_thinking_data()` samples first 100 items for `<think>` tags
- `_compute_thinking_metrics()` tracks `<think>` and `</think>` token prediction accuracy

---

## 3. Data Flow Diagrams

### 3.1 Training Data Flow

```
[Source Files]                 [DataPreparer]                [Cache]
    .aiml --------->+                                     dataset_cache/
    .pdf ---------->|                                      prepared_dataset/
    .epub --------->+-> load -> standardize -> filter ----> sentencepiece.model
    HF datasets --->|         (input_ids)     chunk        dataset_stats.pkl
    .csv ---------->+                         dedup
    web scrape ---->+                         quality
                                               lang
                             |
                             v
                       [Tokenizer Train]
                       SentencePiece BPE
                             |
                             v
                       [Tokenize]
                       text -> token_ids
                             |
                             v
                       [Cache Save]
```

### 3.2 Training Flow

```
[Cached Dataset] -> [TokenPairIterableDataset] -> [DataLoader]
                                                       |
                                                       v
                                              [ChatModel (GPT-2)]
                                                       |
                                                       v
                                              [CrossEntropy Loss]
                                                       |
                                                       v
                                              [Backward + Accumulate]
                                                       |
                                                       v
                                              [Optimizer Step]
                                                       |
                                                       v
                                              [Checkpoint Save]
```

### 3.3 Inference Flow

```
[User Input]
     |
     v
[DialogueManager]
     |
     +-> [BERT Intent Classifier] -> intent label
     +-> [BERT Sentiment Analyzer] -> star rating (1-5)
     |                                       |
     |                                       v
     |                               temperature adjustment
     |
     +-> [Prompt Enrichment]
     |       Pregunta: {input}
     |       Contexto: {intent, sentiment}
     |       Respuesta:
     |
     v
[ChatModel (GPT-2)]
     |
     v
[Autoregressive Generation]
     top-k/top-p filtering
     temperature scaling
     n-gram penalty
     min-length enforcement
     |
     v
[Response]
```

---

## 4. Module Dependency Graph

```
main.py
  ├── data_preparer.py
  │     ├── aimlloder.py
  │     ├── web_scraper.py
  │     ├── bpe_tokenizer.py
  │     ├── PyPDF2
  │     ├── ebooklib
  │     ├── spacy (optional)
  │     ├── langdetect (optional)
  │     └── datasketch (optional)
  │
  ├── main_train.py
  │     ├── chatmodel.py
  │     │     └── transformers.GPT2LMHeadModel
  │     └── bpe_tokenizer.py
  │
  ├── main_chat.py
  │     ├── chatmodel.py
  │     ├── dialogmanager.py
  │     ├── bpe_tokenizer.py
  │     ├── model_downloader.py (BERT models)
  │     ├── model_merge.py
  │     │     └── model_registry.py
  │     └── FastAPI / uvicorn
  │
  ├── model_registry.py
  ├── model_merge.py
  ├── model_export.py
  │     ├── onnxruntime
  │     └── transformers (HF format)
  ├── generate_thinking_data.py
  └── manual_test.py

envAIModels/
  ├── app.py
  │     ├── routers_api.py
  │     ├── routers_v1.py
  │     ├── schemas.py
  │     ├── model.py (llama-cpp-python)
  │     └── utils.py
  └── FastAPI / uvicorn
```

---

## 5. Configuration Reference

### System Configuration (`main.py`)
```python
SYSTEM_CONFIG = {
    'default_cores_fraction': 0.5,
    'min_cores': 1,
    'min_threads': 1,
    'max_ram_fraction': 0.75,
}
```

### Training Configuration (`main_train.py`)
```python
TRAINING_CONFIG = {
    'batch_size': 4,
    'accumulation_steps': 8,
    'learning_rate': 1e-3,
    'embed_size': 256,
    'hidden_size': 512,
    'grad_clip_norm': 1.0,
    'memory_cleanup_interval': 10,
    'warm_up': True,
    'warm_up_ratio': 0.1,
    'warm_up_steps': 100,
}
```

### Generation Configuration (`dialogmanager.py`)
```python
DEFAULT_CONFIG = {
    'top_k': 50,
    'top_p': 0.9,
    'temperature': 0.7,
    'max_len': 128,
    'min_length': 3,
    'no_repeat_ngram_size': 3,
}
```

---

## 6. Error Handling & Safety

| Area | Mechanism |
|------|-----------|
| Checkpoint Loading | `weights_only=True` first, fallback to safe_globals |
| OOM Prevention | RAM monitoring, worker count adjustment |
| Training Interrupt | `threading.Event` + daemon thread, graceful stop |
| Generation Loop | Max token limit, min length enforcement, n-gram repetition guard |
| API Errors | FastAPI exception handlers, `try/except` with fallback |
| Invalid Output | Fallback to default response if generation fails |

---

## 7. Performance Characteristics

| Aspect | Detail |
|--------|--------|
| Dataset Caching | 12x faster training iterations |
| Memory | Optimized for CPU training (consumer hardware) |
| Parallelism | Multi-threaded data loading, gradient accumulation |
| GPU | CUDA with mixed precision (AMP), Tesla K80 supported |
| Tokenization | Pre-computed token_ids in cache, no re-tokenization |
| Generation | 128 max tokens, ~50ms-200ms per response (CPU) |

---

## 8. Security Notes

- Model weights in `checkpoints/` and `models/` excluded from version control
- Virtual environments (`envMyIAModelChat/`) excluded from version control
- No `.env` files or API keys committed
- `weights_only=True` for safe `torch.load()`
- Input validation on API endpoints via Pydantic schemas

---

*Generated from codebase analysis - reflects the current state of the project.*
