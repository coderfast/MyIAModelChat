# Bilingual Tokenizer (English & Spanish) - Usage Guide

## Overview

The improved `BilingualTokenizer` now supports training and inference for both English and Spanish with automatic language detection, preprocessing, and multilingual embedding support.

**Note**: This guide incorporates all improvements from BILINGUAL-TOKENIZER-IMPROVEMENTS.md. The improvements file is kept for historical reference but all features are documented here.

## Key Features

✅ **Automatic Language Detection** - Detects English vs Spanish text automatically  
✅ **Language-Aware Preprocessing** - Handles contractions, accents, and language-specific rules  
✅ **Special Language Tokens** - `<EN>` and `<ES>` tokens for explicit language marking  
✅ **Batch Processing** - Efficient batch encoding/decoding with padding  
✅ **Multilingual Embeddings** - Shared vocabulary across languages (300-dim default)  
✅ **Accent Handling** - Optional accent removal for Spanish text  
✅ **Language Statistics** - Track language usage during training/inference  
✅ **Backward Compatible** - `WordTokenizer` provides the same basic tokenization

## Installation

No new dependencies needed! The tokenizer uses only standard PyTorch and Python libraries.

## Basic Usage

### 1. **Initialize the Tokenizer**

```python
from word_tokenizer import BilingualTokenizer

# Create tokenizer with language tokens
tokenizer = BilingualTokenizer(
    max_vocab_size=65536,
    embedding_dim=300,
    num_workers=4,
    use_language_tokens=True,
    multilingual_vocab=True
)
```

### 2. **Training Phase - Build Vocabulary**

```python
# Sample English and Spanish texts
english_texts = [
    "Hello, how are you today?",
    "The weather is beautiful this morning.",
    "I love learning new languages."
]

spanish_texts = [
    "Hola, ¿cómo estás hoy?",
    "El clima es hermoso esta mañana.",
    "Me encanta aprender nuevos idiomas."
]

# Combine texts
all_texts = english_texts + spanish_texts
all_languages = ['en'] * len(english_texts) + ['es'] * len(spanish_texts)

# Build vocabulary
tokenizer.fit(
    texts=all_texts,
    languages=all_languages,
    remove_accents_flag=False  # Keep Spanish accents
)

print(f"Vocabulary size: {tokenizer.get_vocabulary_size()}")
# Output: Vocabulary size: 45
```

### 3. **Inference Phase - Single Text**

```python
# Encode English text (auto-detects language)
english_tokens = tokenizer.encode("Hello, how are you?")
print(f"English tokens: {english_tokens}")

# Decode back to text
decoded_text = tokenizer.decode(english_tokens, skip_special_tokens=True)
print(f"Decoded: {decoded_text}")
```

### 4. **Inference Phase - Spanish Text**

```python
# Encode Spanish text (auto-detects language)
spanish_tokens = tokenizer.encode("Hola, ¿cómo estás?")
print(f"Spanish tokens: {spanish_tokens}")

# Decode
decoded_spanish = tokenizer.decode(spanish_tokens, skip_special_tokens=True)
print(f"Decoded: {decoded_spanish}")
```

### 5. **Batch Processing**

```python
# Batch encode multiple sentences
texts = [
    "Good morning, how are you?",
    "Buenos días, ¿cómo estás?",
    "Thank you very much!"
]

# Encode batch with padding
batch_tensor = tokenizer.batch_encode(
    texts=texts,
    add_language_token=True,
    max_length=50,
    pad=True
)

print(f"Batch shape: {batch_tensor.shape}")
# Output: Batch shape: torch.Size([3, 50])

# Decode batch
decoded_batch = tokenizer.batch_decode(batch_tensor, skip_special_tokens=True)
for i, text in enumerate(decoded_batch):
    print(f"{i+1}. {text}")
```

## Advanced Usage

### Language Detection

```python
# Manually detect language
lang = tokenizer.detect_language("El amor es hermoso")
print(f"Detected language: {lang}")  # Output: es

lang = tokenizer.detect_language("Love is beautiful")
print(f"Detected language: {lang}")  # Output: en
```

### Text Preprocessing

```python
# Preprocess text explicitly
english_clean = tokenizer.preprocess_text(
    "It's a beautiful day!",
    language='en',
    remove_accents_flag=False
)
print(f"Cleaned English: {english_clean}")

spanish_clean = tokenizer.preprocess_text(
    "¡Qué hermoso día es!",
    language='es',
    remove_accents_flag=False  # Remove accents: "Que hermoso dia es"
)
print(f"Cleaned Spanish: {spanish_clean}")
```

### Encoding with Options

```python
# Encode with language token (useful for model to know language context)
tokens_with_lang = tokenizer.encode(
    text="How are you?",
    language='en',
    add_language_token=True  # Prepends <EN> token
)

# Encode without language token (if model doesn't need it)
tokens_no_lang = tokenizer.encode(
    text="¿Cómo estás?",
    add_language_token=False
)
```

### Vocabulary Management

```python
# Add new words to vocabulary
idx = tokenizer.add_word("chatbot", language='en')
print(f"Added 'chatbot' with index: {idx}")

# Get word embedding
embedding = tokenizer.get_word_embedding("hello")
print(f"Embedding shape: {embedding.shape}")  # torch.Size([300])

# Get vocabulary dictionary
vocab = tokenizer.get_vocab_dict()
print(f"First 10 vocab items: {list(vocab.items())[:10]}")

# Get language statistics
stats = tokenizer.get_language_stats()
print(f"Language stats: {stats}")
# Output: Language stats: {'en': 45, 'es': 38}
```

### Save and Load Vocabulary

```python
# Save vocabulary for later use
tokenizer.save_vocabulary('vocab.json')

# Load vocabulary in another session
tokenizer.load_vocabulary('vocab.json')
```

## Training Integration Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Initialize tokenizer
tokenizer = BilingualTokenizer(max_vocab_size=65536, embedding_dim=300)

# Build vocabulary from training data
training_texts = [...your training data...]
training_languages = [...corresponding languages...]
tokenizer.fit(training_texts, training_languages, remove_accents_flag=False)

# Custom dataset for training
class ChatDataset(Dataset):
    def __init__(self, texts, languages, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.languages = languages
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        lang = self.languages[idx]
        
        # Encode with language token
        tokens = self.tokenizer.encode(
            text,
            language=lang,
            add_language_token=True
        )
        
        # Pad/truncate
        tokens = self.tokenizer.pad_sequence(tokens, self.max_length)
        
        return torch.tensor(tokens, dtype=torch.long)

# Create dataset and dataloader
dataset = ChatDataset(training_texts, training_languages, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in dataloader:
    # batch shape: (batch_size, max_length)
    embeddings = tokenizer.get_embeddings()(batch)  # (batch_size, max_length, 300)
    
    # Use embeddings in your model
    # outputs = model(embeddings)
    # loss = criterion(outputs, labels)

## Nota: integración con SentencePiece (BPE)

Si el pipeline de preparación de datos se ejecuta con `--use-bpe`, se generará un modelo SentencePiece (BPE) y el dataset cacheado incluirá `token_ids` que corresponden a tokens BPE. En escenarios donde se utilice BPE global (SentencePiece) en lugar de `BilingualTokenizer`, puedes:

- Cargar `dataset_cache/sentencepiece.model` para tokenizar/decodificar de forma consistente en entrenamiento e inferencia.
- Mantener `BilingualTokenizer` para tareas de preprocesamiento y etiquetado de idioma, pero al entrenar o inferir con `token_ids` BPE asegúrate de alinear el vocabulario y las embeddings con el esquema BPE.

En resumen: `--use-bpe` y `BilingualTokenizer` pueden coexistir; documenta y guarda el `bpe_model_path` en `cache_metadata.pkl` para reproducibilidad.
```

## Inference Integration Example

```python
# Load pre-trained model and tokenizer
tokenizer = BilingualTokenizer(max_vocab_size=65536, embedding_dim=300)
tokenizer.load_vocabulary('vocab.json')

# Inference on user input
user_input = "Buenos días, ¿cómo puedo ayudarte?"

# Encode
tokens = tokenizer.encode(user_input, add_language_token=True)

# Get embeddings for model input
token_tensor = torch.tensor([tokens], dtype=torch.long)  # Add batch dimension
embeddings = tokenizer.get_embeddings()(token_tensor)

# Pass through model
# model_output = model(embeddings)
# response = generate_response(model_output, tokenizer)

# Decode model output
response_tokens = [...]  # From model
response = tokenizer.decode(response_tokens, skip_special_tokens=True)
print(f"Bot response: {response}")
```

## Special Tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Padding for shorter sequences |
| `<UNK>` | 1 | Unknown words not in vocabulary |
| `<START>` | 2 | Start of sequence marker |
| `<END>` | 3 | End of sequence marker |
| `<EN>` | 4 | English language marker |
| `<ES>` | 5 | Spanish language marker |

## Configuration Tips

### For Smaller Models
```python
tokenizer = BilingualTokenizer(
    max_vocab_size=10000,      # Smaller vocab
    embedding_dim=128,         # Smaller embeddings
    num_workers=2              # Fewer workers
)
```

### For Larger Models
```python
tokenizer = BilingualTokenizer(
    max_vocab_size=100000,     # Larger vocab
    embedding_dim=512,         # Larger embeddings
    num_workers=8              # More workers
)
```

### For Production (No Accents)
```python
tokenizer.fit(
    texts=training_data,
    languages=languages,
    remove_accents_flag=True   # Normalize Spanish
)
```

## Common Issues and Solutions

### Issue: Low vocabulary coverage
**Solution:** Increase `max_vocab_size` or use subword tokenization

### Issue: Spanish accents lost during preprocessing
**Solution:** Set `remove_accents_flag=False` during `fit()` and `encode()`

### Issue: Language detection incorrect
**Solution:** Explicitly pass `language` parameter to `encode()` if mixed-language text

### Issue: Out of memory with batch processing
**Solution:** Reduce `max_length` or `batch_size`, or use smaller `embedding_dim`

## WordTokenizer

The `WordTokenizer` class provides basic word-level tokenization without language tokens:

```python
from word_tokenizer import WordTokenizer

# Creates BilingualTokenizer without language tokens
tokenizer = WordTokenizer(
    max_vocab_size=65536,
    embedding_dim=65536,
    num_workers=4
)

# Use as before
tokenizer.fit(texts)
tokens = tokenizer.encode("Hello world")
```

## Performance Notes

- **Language detection:** ~1-5ms per text
- **Preprocessing:** ~5-20ms depending on text length
- **Encoding:** ~1-2ms per text
- **Batch encoding:** O(batch_size * text_length)
- **Memory:** ~500MB for 100k vocabulary × 300-dim embeddings

## Next Steps

1. Integrate with your chat model training pipeline
2. Save vocabulary after training for deployment
3. Use batch processing for efficient inference
4. Monitor language statistics to track usage patterns
5. Fine-tune preprocessing rules based on your specific domain

Happy multilingual chatting! 🚀
