# Training & Dataset Management Guide

## Dataset Structure

### Supported Dataset Types

1. **AIML-based Datasets**
   - Source: `aiml/` directory (~60 .aiml files)
   - Processing: `aimlloder.py` converts patterns to (input, output) pairs
   - Format: XML patterns converted to dialogue examples

2. **PDF Document Datasets**
   - Source: `pdfs/` directory
   - Processing: `data_preparer.py` uses PyPDF2 for text extraction
   - Format: Automatic page-by-page text extraction
   - Flag: `--pdf` enables PDF processing

3. **EPUB E-book Datasets**
   - Source: `epub/` directory
   - Processing: `data_preparer.py` uses ebooklib for parsing
   - Format: Chapter-by-chapter content extraction with metadata
   - Flag: `--epub` enables EPUB processing

4. **Hugging Face Datasets**
   - Loaded via `transformers.datasets`
   - Multiple public dialogue datasets available
   - Examples: wikitext, bookcorpus, common_voice, opus_100
   - Flag: `--hf` in training enables this

5. **Custom Local Datasets**
   - Place data in `datasets/` directory
   - Can be .json, .txt, .csv format
   - Must be tokenized before training

### Dataset Caching System

The system includes intelligent caching for 12x faster training iterations:

```bash
# First time: Prepare and cache datasets
python main.py --prepare-data --aiml --pdf --epub

# Subsequent training: Use cached data
python main.py --train --use-cache --epochs 30

# Refresh cache after adding new data
python main.py --prepare-data --pdf --epub --refresh-cache

# Clear cache to free space
python main.py --clear-cache

## Build a BPE-tokenized cache (optional)

```bash
python main.py --prepare-data --aiml --pdf --epub --use-bpe --bpe-vocab-size 8000
```

When using `--use-bpe`, SentencePiece (BPE) is trained on extracted text and the prepared cache will include a `token_ids` column for each sample as well as `dataset_cache/sentencepiece.model` and `dataset_cache/cache_metadata.pkl` containing tokenizer metadata. Note: BPE is applied to textual sources (AIML/PDF/EPUB/HF). If `sentencepiece` is not installed the pipeline will skip BPE and continue preparing a non-tokenized cache (a warning is emitted).

**Caching Benefits:**
- 12x faster dataset loading
- Consistent preprocessing across runs
- Automatic schema alignment for concatenation
- Memory-efficient storage

### ChatDataset Class (chatdataset.py)

```python
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns (input_ids, target_ids) pair
        # Input: dialogue context
        # Target: next token(s) to predict
```

### Dataset Combination Strategy

```python
# Load multiple datasets with caching
from data_preparer import DataPreparer

preparer = DataPreparer(enable_pdf=True, enable_epub=True, enable_caching=True)

# Load from cache if available
if use_cache and os.path.exists('dataset_cache'):
    final_dataset = preparer.load_cached_dataset()
else:
    # Load fresh data
    aiml_dataset = preparer._load_aiml_data()
    pdf_dataset = preparer._load_pdf_data()
    epub_dataset = preparer._load_epub_data()
    hf_dataset = load_dataset('wikitext')

    # Combine all datasets (with schema alignment)
    final_dataset = concatenate_datasets([aiml_dataset, pdf_dataset, epub_dataset, hf_dataset])

    # Cache for future use
    preparer.cache_dataset(final_dataset)

# Create DataLoader with multiprocessing
loader = DataLoader(final_dataset, batch_size=32, num_workers=4)
```

## Training Configuration

### Command Line Arguments (main.py --train)

```bash
python main.py --train \
    --epochs 30 \
    --use-cache \
    --aiml \
    --pdf \
    --epub \
    --hf \
    --use-cpuonly \
    --num-cores 4
```

| Argument | Purpose | Default |
|----------|---------|---------|
| `--epochs` | Number of training epochs | 30 |
| `--use-cache` | Use cached datasets for speed | False |
| `--aiml` | Include AIML datasets | False |
| `--pdf` | Include PDF document datasets | False |
| `--epub` | Include EPUB e-book datasets | False |
| `--hf` | Include Hugging Face datasets | False |
| `--use-cpuonly` | Force CPU training | False |
| `--num-cores` | CPU cores for DataLoader workers | 4 |
| `--num-threads` | Additional threads per worker | 2 |
| `--onlytokenize` | Run tokenization only, skip training | False |

### Hyperparameter Tuning

**Learning Rate**
- Start with 0.001 for Adam optimizer
- Reduce if loss oscillates: 0.0001
- Increase if training is too slow: 0.01
- Use learning rate scheduling after epoch 5

**Batch Size**
- Small (8-16): More gradient updates, slower training
- Medium (32-64): Balanced (recommended for this project)
- Large (128+): Faster training, needs more memory

**Embedding Dimension**
- 256-512: Recommended for bilingual support
- 300+: Better for large vocabulary (>50k tokens)
- <256: Faster inference, less expressive

**LSTM Hidden Size**
- 512-1024: Standard range for complex data
- 256: Faster, lighter model
- 2048+: More capacity, slower training

## Tokenization

### BilingualTokenizer Workflow

The system now uses BilingualTokenizer for enhanced EN/ES support:

```python
# Training phase
from bilingual_tokenizer import BilingualTokenizer

tokenizer = BilingualTokenizer()
tokenizer.train(raw_text_data)  # Learns vocabulary with language detection
tokenizer.save('checkpoints/tokenizer.pkl')

# Inference phase
tokenizer.load('checkpoints/tokenizer.pkl')
token_ids = tokenizer.encode("Hello world")  # English
token_ids_es = tokenizer.encode("Hola mundo")  # Spanish with accents
text = tokenizer.decode(token_ids)
```

**Bilingual Features:**
- Automatic language detection (EN/ES)
- Accent handling for Spanish text
- Efficient encoding/decoding
- Vocabulary size up to 50k tokens

### SimpleTokenizer (Legacy)

For basic tokenization:

```python
from simpletokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.train(raw_text_data)
tokenizer.save('tokenizer.pth')
token_ids = tokenizer.encode("Hello world")
text = tokenizer.decode(token_ids)
```

### Vocabulary Management

- Vocabulary size affects model parameters
- Larger vocab = more flexibility, more memory
- Common sizes: 25k, 50k, 100k tokens
- Special tokens: [PAD], [UNK], [CLS], [SEP], [EOS]

### Issues & Solutions

**Out-of-vocabulary (OOV) tokens**
- Problem: Unseen words during inference
- Solution: Use special [UNK] token
- Prevention: Larger vocabulary or subword tokenization

**Tokenization mismatch**
- Problem: Different tokenization between train/inference
- Solution: Always use same tokenizer from checkpoints
- Verify: Check token IDs for same input across runs

**Language detection issues**
- Problem: Incorrect EN/ES classification
- Solution: BilingualTokenizer handles mixed content automatically
- Prevention: Ensure training data includes both languages

## Training Best Practices

### Data Preprocessing with Multiple Sources
1. Clean text (remove special chars if needed)
2. Normalize case (lowercase or mixed)
3. Remove duplicates across all sources
4. Balance dataset categories if using classification
5. Split into train/validation/test (70/15/15)
6. Cache processed datasets for speed

### Training Monitoring with Caching
```python
# Track metrics per epoch
for epoch in range(epochs):
    if use_cache:
        # Load cached dataset (fast)
        dataset = load_cached_dataset()
    else:
        # Load fresh data (slow)
        dataset = load_all_sources()

    train_loss = run_training_epoch(dataset)
    val_loss = run_validation_epoch(dataset)

    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'loss': val_loss
        }, 'checkpoints/chat_model_best.pth')
```

### Common Training Issues

**Out of Memory (OOM)**
- Reduce batch size
- Use `--use-cpuonly` for CPU training
- Reduce sequence length (max_len parameter)
- Use gradient accumulation

**Loss not decreasing**
- Learning rate too high/low
- Model capacity insufficient
- Bad data quality from PDF/EPUB extraction
- Try different initialization seed

**Overfitting**
- Loss decreases but validation worsens
- Add dropout layers (currently 0.1)
- Use early stopping
- Increase regularization (L2)
- Add more diverse training data (PDFs/EPUBs)

**Slow training**
- Use `--use-cache` for 12x speedup
- Increase `num_workers` in DataLoader
- Use GPU instead of CPU
- Profile code to find bottlenecks

**PDF/EPUB parsing errors**
- Verify PyPDF2 and ebooklib are installed
- Check file formats are valid
- Use `--prepare-data` to test loading
- Check extracted text quality

## Dataset Splitting

### Train/Validation/Test Split

```python
from torch.utils.data import random_split

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size]
)
```

## Checkpointing & Recovery

### Saving Training State with Tokenizer

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'tokenizer': tokenizer,  # Now saves tokenizer in checkpoint
    'config': {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size
    }
}
torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')
```

### Resuming Training

```python
checkpoint = torch.load('checkpoints/checkpoint_epoch_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
tokenizer = checkpoint['tokenizer']  # Load tokenizer from checkpoint
start_epoch = checkpoint['epoch'] + 1
```

## Performance Profiling

### Identifying Bottlenecks

```python
import torch.profiler as profiler

with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
    # Your training code
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Common Bottlenecks
1. Data loading (use `--use-cache` for 12x speedup)
2. PDF/EPUB parsing (cache extracted text)
3. GPU transfer (use pinned memory)
4. Tokenization (use batched encoding with BilingualTokenizer)
5. AIML parsing (cache parsed files)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, 
    [train_size, val_size, test_size]
)
```

## Checkpointing & Recovery

### Saving Training State

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'tokenizer': tokenizer
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

### Resuming Training

```python
checkpoint = torch.load('checkpoint_epoch_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Performance Profiling

### Identifying Bottlenecks

```python
import torch.profiler as profiler

with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
    # Your training code
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Common Bottlenecks
1. Data loading (use more workers, prefetch)
2. GPU transfer (use pinned memory)
3. AIML parsing (cache parsed files)
4. Tokenization (use batched encoding)
