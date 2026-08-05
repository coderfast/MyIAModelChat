# Model Architecture Guide

## PyTorch ChatModel Architecture

### Layer Structure

```
ChatModel (nn.Module)
└── GPT-2 Transformer (HuggingFace)
    ├── Token Embedding
    │   ├── Vocabulary Size → Embedding Dimension
    │   └── Converts token IDs to dense vectors
    ├── Positional Embedding
    │   └── Encodes position information
    ├── Transformer Blocks (x num_layers)
    │   ├── Multi-Head Self-Attention
    │   ├── Feed-Forward Network
    │   └── Layer Normalization
    └── Language Model Head
        ├── Hidden dim → Vocabulary size
        └── Outputs logits for next token prediction
```

### Forward Pass Flow

```
Input Tokens (LongTensor)
    ↓
Token Embedding + Positional Embedding
    ↓ (token_ids → embedding_vectors)
Transformer Blocks (x num_layers)
    ↓ (self-attention + feed-forward)
Hidden States
    ↓
Language Model Head
    ↓ (projects to vocab dimension)
Output Logits
```

### Key Parameters (from commons/model/chatmodel.py)

- **vocab_size**: Number of unique tokens in vocabulary (default: 8000 via BPE)
- **embed_size**: Dimension of embedding vectors (256), maps to `n_embd` in GPT2Config
- **num_layers**: Number of transformer layers (default: 2, trained with 4)
- **n_head**: Number of attention heads (4)
- **n_positions**: Maximum sequence length (512)

### Input/Output Specifications

**Input**:
- Shape: `(batch_size, sequence_length)`
- Type: `torch.LongTensor` (token IDs)
- Device: GPU or CPU (specified in training)

**Output**:
- Shape: `(batch_size, sequence_length, vocab_size)`
- Type: `torch.Tensor` (logits)
- Represents: Probability distribution over vocabulary for each position

## Training Pipeline (training/trainer.py)

### DataLoading with Multiple Sources
1. Load AIML data via `dataset_preparer/aiml/loader.py`
2. Load PDF documents via `dataset_preparer/data_preparer.py` (PyPDF2)
3. Load EPUB e-books via `dataset_preparer/data_preparer.py` (ebooklib)
4. Load Hugging Face datasets
5. Combine datasets using `concatenate_datasets()` with schema alignment
6. Cache processed datasets for 12x faster loading
7. Create ChatDataset instances
8. Wrap with DataLoader (supports multiprocessing)

### Training Loop with Best Model Saving
```python
from training.trainer import Trainer, TrainingConfig

config = TrainingConfig(
    epochs=30,
    dataset_path='datasets_source/ciencias/',
    checkpoint_name='ciencias_naturales',
    use_cache=True,
    aiml=True,
    hf=True
)

trainer = Trainer(config)
trainer.run()
```

### Key Training Classes

**Trainer** (training/trainer.py)
- Initializes training pipeline
- Manages SentencePieceTokenizerWrapper and datasets
- Handles multiprocessing for data loading
- Supports AIML, PDF, EPUB, and Hugging Face datasets
- Implements best model checkpointing

**DataPreparer** (dataset_preparer/data_preparer.py)
- Multi-source data loading and processing
- PDF text extraction with PyPDF2
- EPUB parsing with ebooklib
- Dataset caching system (12x speedup)
- Schema alignment for concatenation
- Advanced text processing (chunking, dedup, quality filtering)

**ChatDataset** (commons/dataset/chatdataset.py)
- PyTorch Dataset subclass
- Loads chat dialogue data from multiple sources
- Handles SentencePieceTokenizerWrapper
- Returns (input_ids, target_ids) pairs

**SentencePieceTokenizerWrapper** (commons/tokenizer/bpe_tokenizer.py)
- BPE tokenization (multilingual, language-agnostic)
- Automatic subword tokenization
- Supports any language via SentencePiece
- Efficient encoding/decoding with O(1) lookups

## Dialogue Pipeline (DialogueManager)

### Processing Flow
```
User Input String
    ↓
Intent/Sentiment Analysis (BERT pipeline)
    ↓ (1-5 stars sentiment, intent classification)
Temperature Adjustment
    ↓ (angry → lower temp, happy → higher temp)
Prompt Enrichment
    ↓ (adds context like "El usuario está satisfecho")
SentencePieceTokenization
    ↓ (string → token IDs, multilingual)
Model Inference
    ↓ (autoregressive generation with GPT-2)
Dynamic N-gram Penalization (no_repeat_ngram_size=3)
    ↓
Min-Length Enforcement (min_length=3)
    ↓
Top-K/Top-P Sampling (k=50, p=0.9)
    ↓
Token ID Decoding
    ↓ (token IDs → string)
Response String
```

### Intent/Sentiment Analysis
- Uses `transformers.pipeline('text-classification')`
- Pre-trained BERT model (nlptown/bert-base-multilingual-uncased-sentiment)
- Detects sentiment from 1-5 stars
- Used for temperature adjustment (angry → lower temp, happy → higher temp)

### Generation Improvements
- **Autoregressive Generation**: Proper token-by-token generation
- **Dynamic N-gram Penalization**: Prevents repetitive 3-grams with logit penalties
- **Min-Length Enforcement**: Ensures responses aren't truncated
- **Sampling Controls**: Top-K and Top-P filtering with temperature

## Recent Architecture Improvements

### Tokenizer Migration
- **Migrated**: From word-level tokenizers to SentencePiece BPE
- **Multilingual**: Supports any language without configuration
- **Subword**: Handles out-of-vocabulary words via BPE

### Generation Quality
- **Repetition Prevention**: Dynamic penalization instead of hard banning
- **Length Control**: Min-length enforcement during generation
- **Sampling**: Configurable top-k, top-p, temperature parameters

## Saved Models

### Checkpoint Files
- **models/<name>.pth**: Complete model checkpoint (state_dict + metadata)
- **models/<a>+<b>.pth**: Merged model (weight averaging)
- **checkpoints/tokenizer_vocab.json**: SentencePiece tokenizer vocabulary
- **dataset_cache/sentencepiece.model**: Trained BPE model
- **dataset_cache/**: Cached processed datasets for fast loading
- **models/**: Downloaded ML models (sentiment, intent)

### Loading Models
```python
from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
from commons.model.chatmodel import ChatModel

tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
model = ChatModel(tokenizer, embed_size=256)
model.load_state_dict(torch.load('chat_model.pth'))
model.eval()  # Set to evaluation mode
```

## Performance Considerations

- **Batch Size**: Larger batches = faster training but more memory
- **Sequence Length**: Longer sequences = more context but slower processing
- **Embedding Dimension**: Higher = more expressive but more parameters
- **Multiprocessing**: Use multiple workers in DataLoader for faster data loading

## Extending the Architecture

### Adding More Layers
- Increase `num_layers` in GPT2Config
- More layers = more capacity but slower training

### Increasing Embedding Dimension
- Increase `embed_size` parameter (maps to `n_embd` in GPT2Config)
- Larger embedding dimension = more expressive model

### Adding Dropout
- Reduces overfitting
- Add after embedding and transformer layers
- Typical dropout rate: 0.1-0.3

### Ensemble Approaches
- Train multiple models with different seeds
- Average predictions for robustness
- Useful for production deployments

---

*Updated: 2026-07-28 - Reflects new modular project structure*
