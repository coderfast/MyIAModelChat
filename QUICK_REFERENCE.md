# Quick Reference - MyIAModelChat

## File Structure Quick Map

| File | Purpose |
|------|---------|
| `chatmodel.py` | GPT-2 model architecture (HuggingFace) |
| `chatdataset.py` | PyTorch Dataset loader |
| `dialogmanager.py` | Dialogue flow, intent/sentiment, persona modeling |
| `bpe_tokenizer.py` | SentencePiece BPE tokenizer (multilingual) |
| `aimlloder.py` | Loads AIML files for training |
| `data_preparer.py` | Multi-source data loading (AIML, PDF, EPUB, HF) |
| `main_train.py` | Training pipeline (MainTrain class) |
| `main_chat.py` | Chat interface + FastAPI server |
| `main.py` | Primary entry point with argument parsing |
| `model_registry.py` | Model discovery, listing, validation |
| `model_merge.py` | Model merging by weight averaging |
| `model_export.py` | Export to GGUF, ONNX, ONNX quantized |
| `model_downloader.py` | HuggingFace model downloader |
| `generate_thinking_data.py` | Chain-of-thought data generation |

## Command Cheat Sheet

```bash
# Activate environment
.\envMyIAModelChat\Scripts\Activate.ps1

# === DATA PREPARATION ===
# Prepare data from multiple sources
python main.py --prepare-data --aiml --pdf --epub
# Prepare and build a BPE-tokenized cache
python main.py --prepare-data --aiml --pdf --epub --bpe-vocab-size 8000

# === TRAINING ===
# Train with cached data (fast)
python main.py --train --use-cache --epochs 30

# Train with custom checkpoint name and dataset
python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --aiml --hf --epochs 30

# Train without cache (slower)
python main.py --train --epochs 10 --aiml --pdf --epub

# Refresh cache after adding new data
python main.py --prepare-data --aiml --epub --refresh-cache

# Clear old cache
python main.py --clear-cache

# Tokenize only (no training)
python main.py --train --onlytokenize

# === MODEL LIBRARY ===
# List available trained models
python main.py --list-models

# Chat with a specific model
python main.py --chat --model ciencias_naturales

# Chat with merged models
python main.py --chat --model ciencias_naturales+programacion

# Export model to GGUF/ONNX
python main.py --export ciencias_naturales --formats gguf,onnx,onnx_int8

# Export merged model
python main.py --export ciencias_naturales+programacion --formats gguf

# === CHAT / SERVER ===
# Run chat interface
python main.py --chat

# Chat with CPU optimization
python main.py --chat --use-cpuonly --num_cores 4 --num_threads 4

# Inspect cache metadata
python -c "import pickle; print(pickle.load(open('dataset_cache/cache_metadata.pkl','rb')))"
```

**BPE note:** SentencePiece BPE is used automatically during `--prepare-data`. If `sentencepiece` is not installed the prepare step will skip BPE and create a non-tokenized cache (a warning is emitted).

## Model Config Essentials

```python
# chatmodel.py - Key parameters (GPT-2 architecture)
vocab_size = 8000           # Vocabulary size (BPE, default)
embed_size = 256            # Embedding dimension
hidden_size = 512           # Hidden state size
num_layers = 4              # Number of transformer layers
n_head = 4                  # Attention heads
n_positions = 512           # Max sequence length

# dialogmanager.py - Key settings (from main_chat.py)
intent_model = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
top_k = 50                 # Sampling parameter
top_p = 0.9                # Nucleus sampling
temperature = 0.7          # Generation temperature
min_length = 3             # Minimum response length
no_repeat_ngram_size = 3   # N-gram repetition prevention

# data_preparer.py - Feature flags
enable_pdf = True          # Enable PDF processing
enable_epub = True         # Enable EPUB processing
enable_caching = True      # Enable dataset caching

# main_train.py - Training params
batch_size = 4             # Micro batch size
accumulation_steps = 8     # Gradient accumulation (effective batch = 32)
learning_rate = 0.001
epochs = 1                 # Default (use --epochs to increase)
num_workers = 0            # DataLoader workers
use_cache = False          # Default (use --use-cache to enable)
```

## Model Architecture Flow

```
User Input (text)
    ↓
Intent/Sentiment Analysis (BERT)
    ↓
SentencePieceTokenizerWrapper.encode() → Token IDs
    ↓
ChatModel forward pass:
    Token IDs → GPT-2 Transformer → Logits
    ↓
Dynamic N-gram Penalization + Top-K/Top-P Sampling
    ↓
Output Token IDs
    ↓
SentencePieceTokenizerWrapper.decode() → Response text
```

## Key Classes & Methods

### ChatModel
```python
from chatmodel import ChatModel

model = ChatModel(tokenizer, embed_size=256, hidden_size=512)
output = model(input_ids)  # Forward pass
model.to(device)  # Move to GPU/CPU
```

### DialogueManager
```python
from dialogmanager import DialogueManager

dialog = DialogueManager(model, device, tokenizer,
                         intent_classifier, sentiment_analyzer,
                         persona, top_k=50, top_p=0.9,
                         temperature=0.7, min_length=3, no_repeat_ngram_size=3)
response = dialog.generate_response(user_input)
```

### SentencePieceTokenizerWrapper
```python
from bpe_tokenizer import SentencePieceTokenizerWrapper

tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
token_ids = tokenizer.encode("hello world")
text = tokenizer.decode(token_ids)
# Multilingual: works with any language
```

### DataPreparer
```python
from data_preparer import DataPreparer

preparer = DataPreparer(enable_pdf=True, enable_epub=True, enable_caching=True)
datasets = preparer.load_all_data()
cached_dataset = preparer.cache_dataset(datasets)
```

### MainTrain
```python
from main_train import MainTrain

trainer = MainTrain(args)
trainer.prepare_datasets()  # Load AIML + PDF + EPUB + HF datasets
trainer.train()
```

## Dataset Sources

1. **AIML** (`datasets_source/aiml/` folder): ~60 AIML pattern files from A.L.I.C.E.
2. **PDF Documents** (`pdfs/` folder): Automatic text extraction via PyPDF2
3. **EPUB E-books** (`epub/` folder): Full e-book parsing via ebooklib
4. **Hugging Face**: Dialogue datasets (wikitext, bookcorpus, etc.)
5. **Custom**: User-provided data in `datasets/` folder

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `models/<name>.pth` | ~37MB | Trained model checkpoint (with metadata) |
| `models/<a>+<b>.pth` | ~37MB | Merged model (weight average) |
| `models/exported/<name>.gguf` | ~20MB | GGUF for Ollama/llama.cpp |
| `models/exported/<name>.onnx` | ~15MB | ONNX for ONNX Runtime |
| `models/exported/<name>_int8.onnx` | ~10MB | Quantized ONNX |
| `checkpoints/tokenizer_vocab.json` | ~1KB | Tokenizer metadata |
| `dataset_cache/` | ~5GB | Cached processed datasets |

## Caching System

```bash
# Enable caching for 12x faster training
python main.py --train --use-cache --epochs 10

# Refresh cache after adding new PDFs/EPUBs
python main.py --prepare-data --pdf --epub --refresh-cache

# Clear cache to free space
python main.py --clear-cache

# Check cache metadata
python -c "import pickle; print(pickle.load(open('dataset_cache/cache_metadata.pkl','rb')))"
```

## Common Fixes

| Issue | Fix |
|-------|-----|
| Import errors | Activate venv: `.\envMyIAModelChat\Scripts\Activate.ps1` |
| CUDA out of memory | Use `--use-cpuonly` or reduce batch_size |
| Gibberish responses | Ensure `model.eval()` and check tokenizer match |
| Tokenizer mismatch | Use same checkpoint for train & inference |
| No data loaded | Run `--prepare-data` first, check source folders |
| Slow training | Use `--use-cache`, increase `num_workers` |
| PDF/EPUB not loading | Verify PyPDF2/ebooklib installed, files in correct folders |
| Repetitive responses | Adjust `no_repeat_ngram_size`, `temperature` |
| Short responses | Increase `min_length` parameter |

## Debugging Commands

```python
# Check model structure
print(model)

# Debug tokenizer
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Sample tokens: {tokenizer.encode('Hello world')}")

# Check dataset loading
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")

# Debug generation
response = dialog.generate_response("debug")
print(f"Debug response: {response}")

# Check cache status
import os
print(f"Cache exists: {os.path.exists('dataset_cache')}")

# Check tensor shapes
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")

# Check device
print(f"Model on: {next(model.parameters()).device}")

# Check AIML loading
import os
print(f"AIML files: {len(os.listdir('datasets_source/aiml/'))}")

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Training Progress Monitoring

Watch these metrics:
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should follow training loss (overfitting if diverges)
- **Intent Accuracy**: % of correctly detected intents
- **Sentiment Accuracy**: % of correctly detected sentiments
- **Speed**: Tokens/sec (improve with more workers)

## Best Practices

✅ **Do**:
- Always activate virtual environment
- Use `torch.no_grad()` during inference
- Call `model.eval()` before generating responses
- Save checkpoints periodically
- Keep tokenizer consistent
- Log important diagnostics

❌ **Don't**:
- Train on GPU without checking memory
- Mix different tokenizers
- Forget to load model weights
- Leave model in training mode during inference
- Ignore validation loss divergence
- Commit model files to git (add to .gitignore)

## Performance Targets

- Training speed: 100-500 samples/sec (depends on hardware)
- Inference speed: <1 second per response
- Model accuracy: ~80-90% intent detection
- Sentiment accuracy: ~85-95%

## Resource Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for training
- **GPU**: NVIDIA with CUDA support (optional but recommended)
- **Disk**: 5GB+ for models and data

## Getting Help

1. Check [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for debugging
2. See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for architecture questions
3. Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training issues
4. Review [BERT_NLP_TRANSFORMERS.md](BERT_NLP_TRANSFORMERS.md) for BERT-related questions
5. Study [AIML_USAGE.md](AIML_USAGE.md) for pattern questions
