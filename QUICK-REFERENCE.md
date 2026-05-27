# Quick Reference - MyIAModelChat

## File Structure Quick Map

| File | Purpose |
|------|---------|
| `chatmodel.py` | PyTorch LSTM model (Embedding → LSTM → FC) |
| `chatdataset.py` | PyTorch Dataset loader |
| `dialogmanager.py` | Dialogue flow, intent/sentiment, persona modeling |
| `simpletokenizer.py` | Custom tokenization (encode/decode) |
| `bilingual_tokenizer.py` | Bilingual EN/ES tokenization with accent handling |
| `aimlloder.py` | Loads AIML files for training |
| `data_preparer.py` | Multi-source data loading (AIML, PDF, EPUB) |
| `main_train.py` | Training pipeline (MainTrain class) |
| `main_chat.py` | Chat inference interface |
| `main.py` | Primary entry point with argument parsing |

## Command Cheat Sheet

```bash
# Activate environment
.\envMyIAModelChat\Scripts\Activate.ps1

# Prepare data from multiple sources
python main.py --prepare-data --aiml --pdf --epub
# Prepare and build a BPE-tokenized cache (requires sentencepiece)
python main.py --prepare-data --aiml --pdf --epub --use-bpe --bpe-vocab-size 8000

# Train with cached data (fast)
python main.py --train --use-cache --epochs 30

# Train without cache (slower)
python main.py --train --epochs 10 --aiml --pdf --epub

# Use cache for faster iterations
python main.py --train --use-cache --epochs 5

# Refresh cache after adding new data
python main.py --prepare-data --aiml --epub --refresh-cache

# Clear old cache
python main.py --clear-cache

# Prepare data and build BPE-tokenized cache
python main.py --prepare-data --aiml --hf --use-bpe --bpe-vocab-size 8000

# Train with AIML only
python main.py --train --epochs 10 --aiml

# Tokenize only (no training)
python main.py --train --onlytokenize

# Run chat interface
python main.py --chat

# Chat with CPU optimization
python main.py --chat --use-cpuonly --num_cores 4 --num_threads 4

# Just preprocess data
python main.py --prepare-data

# Inspect cache metadata (quick check)
python - <<'PY'
import pickle
md = pickle.load(open('dataset_cache/cache_metadata.pkl','rb'))
print(md)
PY
```

**BPE note:** Use `--use-bpe` to build a SentencePiece BPE-tokenized cache from textual sources (AIML, extracted PDF/EPUB text, Hugging Face datasets). If `sentencepiece` is not installed the prepare step will skip BPE and create a non-tokenized cache (a warning is emitted).

## Model Config Essentials

```python
# chatmodel.py - Key parameters
vocab_size = 50000        # Vocabulary size (increased)
embedding_dim = 256       # Embedding dimension
hidden_size = 512         # LSTM hidden state size
output_size = vocab_size  # Output vocabulary size

# dialogmanager.py - Key settings
max_history = 5           # Conversation history window
intent_model = "facebook/bart-large-mnli"
sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
top_k = 50               # Sampling parameter
top_p = 0.9              # Nucleus sampling
temperature = 0.8        # Generation temperature
min_length = 5           # Minimum response length
no_repeat_ngram_size = 3 # N-gram repetition prevention

# data_preparer.py - Feature flags
enable_pdf = True        # Enable PDF processing
enable_epub = True       # Enable EPUB processing
enable_caching = True    # Enable dataset caching

# main_train.py - Training params
batch_size = 32
learning_rate = 0.001
epochs = 30             # Default increased
num_workers = 4
use_cache = True        # Use cached datasets
```

## Model Architecture Flow

```
User Input (text)
    ↓
Language Detection (EN/ES)
    ↓
Intent Classifier (BERT)
    ↓
Sentiment Analyzer (BERT)
    ↓
BilingualTokenizer.encode() → Token IDs
    ↓
ChatModel forward pass:
    Token IDs → Embedding → LSTM → FC → Logits
    ↓
Dynamic N-gram Penalization + Top-K/Top-P Sampling
    ↓
Output Token IDs
    ↓
BilingualTokenizer.decode() → Response text
```

## Key Classes & Methods

### ChatModel
```python
from chatmodel import ChatModel

model = ChatModel(vocab_size, embedding_dim, hidden_size)
output = model(input_ids)  # Forward pass
model.to(device)  # Move to GPU/CPU
model.save_pretrained()  # Save weights
```

### DialogueManager
```python
from dialogmanager import DialogueManager

dialog = DialogueManager(model, device, tokenizer,
                         intent_classifier, sentiment_analyzer,
                         persona, max_history=5, top_k=50, top_p=0.9,
                         temperature=0.8, min_length=5, no_repeat_ngram_size=3)
response = dialog.generate_response(user_input)
intent = dialog.detected_intent
sentiment = dialog.detected_sentiment
```

### BilingualTokenizer
```python
from bilingual_tokenizer import BilingualTokenizer

tokenizer = BilingualTokenizer()
tokenizer.train(raw_text)
token_ids = tokenizer.encode("hello world")
text = tokenizer.decode(token_ids)
# Supports accent handling: "español" → proper tokenization
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

1. **AIML** (`aiml/` folder): ~60 AIML pattern files from A.L.I.C.E.
2. **PDF Documents** (`pdfs/` folder): Automatic text extraction via PyPDF2
3. **EPUB E-books** (`epub/` folder): Full e-book parsing via ebooklib
4. **Hugging Face**: Dialogue datasets (wikitext, bookcorpus, etc.)
5. **Custom**: User-provided data in `datasets/` folder

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `checkpoints/chat_model_best.pth` | ~2GB | Best trained model weights |
| `checkpoints/tokenizer.pkl` | ~20MB | Bilingual tokenizer with vocab |
| `dataset_cache/` | ~5GB | Cached processed datasets |
| `pretrained_embeddings.pth` | ~100MB | Pre-trained embeddings (optional) |

## Caching System

```bash
# Enable caching for 12x faster training
python main.py --train --use-cache --epochs 10

# Refresh cache after adding new PDFs/EPUBs
python main.py --prepare-data --pdf --epub --refresh-cache

# Clear cache to free space
python main.py --clear-cache

# Check cache status
python main.py --cache-status
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
```

# Check tensor shapes
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")

# Check device
print(f"Model on: {next(model.parameters()).device}")

# Check AIML loading
import os
print(f"AIML files: {len(os.listdir('aiml/'))}")

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

1. Check [DEVELOPMENT-GUIDE.md](DEVELOPMENT-GUIDE.md) for debugging
2. See [MODEL-ARCHITECTURE.md](MODEL-ARCHITECTURE.md) for architecture questions
3. Read [TRAINING-GUIDE.md](TRAINING-GUIDE.md) for training issues
4. Review [NLP-TRANSFORMERS.md](NLP-TRANSFORMERS.md) for BERT-related questions
5. Study [AIML-USAGE.md](AIML-USAGE.md) for pattern questions
