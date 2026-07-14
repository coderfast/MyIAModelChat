# Multilingual BPE Tokenizer - Usage Guide

## Overview

The project uses **SentencePiece BPE** (Byte-Pair Encoding) as the sole tokenization system. SentencePiece is natively multilingual — it supports any language (English, Spanish, French, German, etc.) without additional configuration.

**Note**: This guide replaces the old `BilingualTokenizer` documentation. The project has migrated from word-level tokenization to BPE for better multilingual support.

## Key Features

- **Multilingual Support** - Works with any language natively
- **Subword Tokenization** - Handles out-of-vocabulary words via subword splits
- **Consistent Tokenization** - Same model for training and inference
- **Efficient Encoding/Decoding** - O(1) lookups via vocabulary dictionaries
- **Thinking Support** - Special tokens for chain-of-thought reasoning

## Installation

```bash
pip install sentencepiece
```

## Basic Usage

### 1. Initialize the Tokenizer

```python
from bpe_tokenizer import SentencePieceTokenizerWrapper

# Load trained BPE model
tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
```

### 2. Encode Text

```python
# English
english_tokens = tokenizer.encode("Hello, how are you?")
print(f"English tokens: {english_tokens}")

# Spanish
spanish_tokens = tokenizer.encode("Hola, ¿cómo estás?")
print(f"Spanish tokens: {spanish_tokens}")

# French
french_tokens = tokenizer.encode("Bonjour, comment allez-vous?")
print(f"French tokens: {french_tokens}")

# German
german_tokens = tokenizer.encode("Hallo, wie geht es Ihnen?")
print(f"German tokens: {german_tokens}")
```

### 3. Decode Tokens

```python
# Decode back to text
decoded_text = tokenizer.decode(english_tokens)
print(f"Decoded: {decoded_text}")
```

### 4. Batch Processing

```python
# Batch encode multiple sentences
texts = [
    "Good morning, how are you?",
    "Buenos días, ¿cómo estás?",
    "Bonjour, comment allez-vous?",
    "Hallo, wie geht es Ihnen?"
]

batch_tokens = tokenizer.batch_encode(texts)
for i, tokens in enumerate(batch_tokens):
    print(f"{i+1}. Tokens: {tokens}")
```

## Advanced Usage

### Special Tokens

```python
# Get special token indices
pad_idx = tokenizer.get_pad_index()
unk_idx = tokenizer.get_unk_index()
eos_idx = tokenizer.get_eos_index()

print(f"PAD index: {pad_idx}")
print(f"UNK index: {unk_idx}")
print(f"EOS index: {eos_idx}")
```

### Thinking Tokens (Chain-of-Thought)

```python
# Check if thinking tokens exist
if tokenizer.has_thinking():
    print("Thinking tokens available")
    
    # Split thinking from response
    text_with_thinking = "<think>Analizando...</think>La respuesta es..."
    thinking, response = tokenizer.split_thinking(text_with_thinking)
    print(f"Thinking: {thinking}")
    print(f"Response: {response}")
```

### Vocabulary Information

```python
# Get vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Get vocabulary dictionary
vocab = tokenizer.vocab
print(f"First 10 tokens: {list(vocab.items())[:10]}")

# Get inverse vocabulary
idx2word = tokenizer.idx2word
print(f"Index 0: {idx2word.get(0, 'N/A')}")
```

## Training the BPE Model

### During Data Preparation

```bash
# Prepare data with BPE tokenization
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000

# This creates:
# - dataset_cache/sentencepiece.model (trained BPE model)
# - dataset_cache/cache_metadata.pkl (tokenizer metadata)
# - dataset_cache/prepared_dataset/ (tokenized dataset with token_ids)
```

### Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--bpe-vocab-size` | Vocabulary size for BPE | `8000` |
| `--refresh-cache` | Rebuild cache from scratch | N/A |
| `--use-cache` | Load cached dataset if exists | N/A |

### Vocabulary Size Guidelines

| Size | Use Case | Trade-off |
|------|----------|-----------|
| 2000 | Quick testing | Fast, but low coverage |
| 8000 | Standard training | Good balance |
| 16000 | Multilingual | Better coverage, more memory |
| 32000 | Large datasets | Best coverage, highest memory |

## Integration Examples

### Training Pipeline

```python
from bpe_tokenizer import SentencePieceTokenizerWrapper
from chatmodel import ChatModel

# Load tokenizer
tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')

# Create model
model = ChatModel(tokenizer, embed_size=256, hidden_size=512)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids']  # Already tokenized
    output = model(input_ids)
    # ... training code
```

### Inference Pipeline

```python
from bpe_tokenizer import SentencePieceTokenizerWrapper
from chatmodel import ChatModel

# Load tokenizer and model
tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
model = ChatModel(tokenizer, embed_size=256, hidden_size=512)
model.load_state_dict(torch.load('chat_model.pth'))
model.eval()

# Inference
user_input = "Hola, ¿cómo estás?"
input_ids = tokenizer.encode(user_input)
# ... model inference
response = tokenizer.decode(output_ids)
```

## Multilingual Examples

### English
```python
tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
# Result: [token1, token2, token3, ...]
```

### Spanish
```python
tokens = tokenizer.encode("El rápido zorro marrón salta sobre el perro perezoso")
# Result: [token1, token2, token3, ...]
```

### French
```python
tokens = tokenizer.encode("Le renard brun rapide saute par-dessus le chien paresseux")
# Result: [token1, token2, token3, ...]
```

### German
```python
tokens = tokenizer.encode("Der schnelle braune Fuchs springt über den faulen Hund")
# Result: [token1, token2, token3, ...]
```

### Mixed Languages
```python
tokens = tokenizer.encode("Hello world, hola mundo, bonjour le monde")
# Result: [token1, token2, token3, ...]
```

## File Locations

| File | Description |
|------|-------------|
| `bpe_tokenizer.py` | SentencePiece tokenizer wrapper |
| `dataset_cache/sentencepiece.model` | Trained BPE model |
| `dataset_cache/cache_metadata.pkl` | Tokenizer metadata |
| `checkpoints/tokenizer_vocab.json` | Vocabulary JSON |

## Troubleshooting

### Issue: "No SentencePiece model found"
**Solution**: Run data preparation first:
```bash
python main.py --prepare-data --aiml --hf --bpe-vocab-size 8000 --refresh-cache
```

### Issue: Low vocabulary coverage
**Solution**: Increase vocabulary size:
```bash
python main.py --prepare-data --aiml --hf --bpe-vocab-size 16000 --refresh-cache
```

### Issue: Tokens appear as `⁇`
**Solution**: Rebuild cache with thinking tokens:
```bash
python main.py --prepare-data --aiml --bpe-vocab-size 8000 --refresh-cache
```

### Issue: Inconsistent tokenization
**Solution**: Ensure same BPE model is used for training and inference. Check `cache_metadata.pkl` for the model path.

## Migration from Word-Level Tokenizers

If you were using `BilingualTokenizer` or `WordTokenizer`:

1. **Remove old imports**:
   ```python
   # OLD
   from word_tokenizer import BilingualTokenizer
   
   # NEW
   from bpe_tokenizer import SentencePieceTokenizerWrapper
   ```

2. **Update initialization**:
   ```python
   # OLD
   tokenizer = BilingualTokenizer(max_vocab_size=65536)
   tokenizer.fit(texts)
   
   # NEW
   tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
   ```

3. **Update encode/decode**:
   ```python
   # OLD
   tokens = tokenizer.encode("Hello world")
   text = tokenizer.decode(tokens)
   
   # NEW (same API!)
   tokens = tokenizer.encode("Hello world")
   text = tokenizer.decode(tokens)
   ```

## Performance Notes

- **Encoding**: ~0.1ms per text
- **Decoding**: ~0.1ms per text
- **Batch encoding**: O(batch_size * text_length)
- **Memory**: ~50MB for 8k vocabulary model

## Next Steps

1. Train BPE model with your data
2. Integrate with training pipeline
3. Test multilingual tokenization
4. Monitor vocabulary coverage
5. Fine-tune vocabulary size if needed
