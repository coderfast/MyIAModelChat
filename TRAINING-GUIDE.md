# Training & Dataset Management Guide

## Dataset Structure

### Supported Dataset Types

1. **AIML-based Datasets**
   - Source: `aiml/` directory (~60 .aiml files)
   - Processing: `aimlloder.py` converts patterns to (input, output) pairs
   - Format: XML patterns converted to dialogue examples

2. **Hugging Face Datasets**
   - Loaded via `transformers.datasets`
   - Multiple public dialogue datasets available
   - Examples: wikitext, common_voice, opus_100
   - Flag `--hf` in training enables this

3. **Custom Local Datasets**
   - Place data in `datasets/` directory
   - Can be .json, .txt, .csv format
   - Must be tokenized before training

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
# Load multiple datasets
aiml_dataset = load_aiml_data()          # From AIML files
hf_dataset = load_dataset('wikitext')    # From Hugging Face
custom_dataset = load_custom_data()      # From local files

# Combine all datasets
final_dataset = concatenate_datasets([aiml_dataset, hf_dataset, custom_dataset])

# Create DataLoader with multiprocessing
loader = DataLoader(final_dataset, batch_size=32, num_workers=4)
```

## Training Configuration

### Command Line Arguments (main_train.py)

```bash
python main_train.py \
    --epochs 10 \
    --num-cores 4 \
    --num-threads 2 \
    --aiml \
    --hf \
    --onlytokenize
```

| Argument | Purpose | Default |
|----------|---------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--num-cores` | CPU cores for DataLoader workers | 4 |
| `--num-threads` | Additional threads per worker | 2 |
| `--aiml` | Include AIML datasets | False |
| `--hf` | Include Hugging Face datasets | False |
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
- 100-300: Typical for chat models
- 300+: Better for large vocabulary (>50k tokens)
- <100: Faster inference, less expressive

**LSTM Hidden Size**
- 256-512: Standard range
- 128: Faster, lighter model
- 1024+: More capacity, slower training

## Tokenization

### SimpleTokenizer Workflow

```python
# Training phase
tokenizer = SimpleTokenizer()
tokenizer.train(raw_text_data)  # Learns vocabulary
tokenizer.save('tokenizer.pth')

# Inference phase
tokenizer.load('tokenizer.pth')
token_ids = tokenizer.encode("Hello world")
text = tokenizer.decode(token_ids)
```

### Vocabulary Management

- Vocabulary size affects model parameters
- Larger vocab = more flexibility, more memory
- Common sizes: 10k, 25k, 50k, 100k tokens
- Special tokens: [PAD], [UNK], [CLS], [SEP]

### Issues & Solutions

**Out-of-vocabulary (OOV) tokens**
- Problem: Unseen words during inference
- Solution: Use special [UNK] token
- Prevention: Larger vocabulary or subword tokenization

**Tokenization mismatch**
- Problem: Different tokenization between train/inference
- Solution: Always use same tokenizer.pth
- Verify: Check token IDs for same input across runs

## Training Best Practices

### Data Preprocessing
1. Clean text (remove special chars if needed)
2. Normalize case (lowercase or mixed)
3. Remove duplicates
4. Balance dataset categories if using classification
5. Split into train/validation/test (70/15/15)

### Training Monitoring
```python
# Track metrics per epoch
for epoch in range(epochs):
    train_loss = run_training_epoch()
    val_loss = run_validation_epoch()
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping if val loss increases
    if val_loss > best_val_loss:
        patience_counter += 1
        if patience_counter > 3:
            break
    else:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
```

### Common Training Issues

**Out of Memory (OOM)**
- Reduce batch size
- Reduce sequence length
- Use gradient accumulation
- Use mixed precision training

**Loss not decreasing**
- Learning rate too high/low
- Model capacity insufficient
- Bad data quality
- Try different initialization seed

**Overfitting**
- Loss decreases but validation worsens
- Add dropout layers
- Use early stopping
- Increase regularization (L2)
- Add more training data

**Slow training**
- Increase batch size
- Increase num_workers in DataLoader
- Use GPU instead of CPU
- Profile code to find bottlenecks

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
