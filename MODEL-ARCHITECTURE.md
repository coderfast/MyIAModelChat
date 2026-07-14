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

### Key Parameters (from chatmodel.py)

- **vocab_size**: Number of unique tokens in vocabulary (up to 50k)
- **embed_size**: Dimension of embedding vectors (256-512 recommended)
- **hidden_size**: Hidden state dimension (512 for standard models)
- **num_layers**: Number of transformer layers (2-4 recommended)
- **n_head**: Number of attention heads (4-8 recommended)
- **n_positions**: Maximum sequence length (512 default)

### Input/Output Specifications

**Input**:
- Shape: `(batch_size, sequence_length)`
- Type: `torch.LongTensor` (token IDs)
- Device: GPU or CPU (specified in training)

**Output**:
- Shape: `(batch_size, sequence_length, vocab_size)`
- Type: `torch.Tensor` (logits)
- Represents: Probability distribution over vocabulary for each position

## Training Pipeline (main_train.py)

### DataLoading with Multiple Sources
1. Load AIML data via `aimlloder.py`
2. Load PDF documents via `data_preparer.py` (PyPDF2)
3. Load EPUB e-books via `data_preparer.py` (ebooklib)
4. Load Hugging Face datasets
5. Combine datasets using `concatenate_datasets()` with schema alignment
6. Cache processed datasets for 12x faster loading
7. Create ChatDataset instances
8. Wrap with DataLoader (supports multiprocessing)

### Training Loop with Best Model Saving
```python
best_val_loss = float('inf')
for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_ids)

        # Compute loss (CrossEntropyLoss typical)
        loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation and checkpointing
    val_loss = validate(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'loss': val_loss
        }, 'checkpoints/chat_model_best.pth')
```

### Key Training Classes

**MainTrain**
- Initializes training pipeline
- Manages SentencePieceTokenizerWrapper and datasets
- Handles multiprocessing for data loading
- Supports AIML, PDF, EPUB, and Hugging Face datasets
- Implements best model checkpointing

**DataPreparer**
- Multi-source data loading and processing
- PDF text extraction with PyPDF2
- EPUB parsing with ebooklib
- Dataset caching system (12x speedup)
- Schema alignment for concatenation
- Advanced text processing (chunking, dedup, quality filtering)

**ChatDataset**
- PyTorch Dataset subclass
- Loads chat dialogue data from multiple sources
- Handles SentencePieceTokenizerWrapper
- Returns (input_ids, target_ids) pairs

**SentencePieceTokenizerWrapper**
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
Context Management
    ↓ (deque maintains last 5 turns)
Persona Modeling
    ↓ (adjusts response based on personality)
SentencePieceTokenization
    ↓ (string → token IDs, multilingual)
Model Inference
    ↓ (autoregressive generation with GPT-2)
Dynamic N-gram Penalization (no_repeat_ngram_size=3)
    ↓
Min-Length Enforcement (min_length=5)
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

### Context Window
- Maintains last 5 user inputs/bot responses (configurable)
- Combined into single context string
- Used to condition model predictions
- Prevents loss of conversation coherence

### Generation Improvements
- **Autoregressive Generation**: Proper token-by-token generation
- **Dynamic N-gram Penalization**: Prevents repetitive 3-grams with logit penalties
- **Min-Length Enforcement**: Ensures responses aren't truncated
- **Sampling Controls**: Top-K and Top-P filtering with temperature

## Recent Architecture Improvements

### Forward Pass Fixes
- **Issue**: LSTM was incorrectly handling batch dimensions
- **Fix**: Proper tensor reshaping and sequence processing
- **Impact**: Correct gradient flow and training convergence

### Tokenizer Enhancements
- **Issue**: Race conditions in SimpleTokenizer.fit()
- **Fix**: Thread-safe vocabulary building
- **Addition**: BilingualTokenizer with EN/ES support and accent handling
- **Performance**: O(1) decoding with idx2word mapping

### Language Support
- **Automatic Detection**: EN/ES language classification
- **Accent Handling**: Proper tokenization of Spanish text (español, México, etc.)
- **Mixed Content**: Handles bilingual conversations seamlessly

### Generation Quality
- **Repetition Prevention**: Dynamic penalization instead of hard banning
- **Length Control**: Min-length enforcement during generation
- **Sampling**: Configurable top-k, top-p, temperature parameters

## Saved Models

### Checkpoint Files
- **chat_model.pth**: Complete ChatModel state_dict
- **checkpoints/tokenizer_vocab.json**: SentencePiece tokenizer vocabulary
- **dataset_cache/sentencepiece.model**: Trained BPE model
- **dataset_cache/**: Cached processed datasets for fast loading
- **models/**: Downloaded ML models (sentiment, intent)

### Loading Models
```python
from bpe_tokenizer import SentencePieceTokenizerWrapper
from chatmodel import ChatModel

tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')
model = ChatModel(tokenizer, embed_size=256, hidden_size=512)
model.load_state_dict(torch.load('chat_model.pth'))
model.eval()  # Set to evaluation mode
```

## Performance Considerations

- **Batch Size**: Larger batches = faster training but more memory
- **Sequence Length**: Longer sequences = more context but slower processing
- **Embedding Dimension**: Higher = more expressive but more parameters
- **Hidden Size**: Larger LSTM = more capacity but slower inference
- **Multiprocessing**: Use multiple workers in DataLoader for faster data loading

## Extending the Architecture

### Adding Attention Mechanism
- Replace LSTM with Transformer encoder
- Add multi-head attention layers
- Improves long-range dependency modeling

### Adding Bidirectional Processing
- Set `bidirectional=True` in LSTM
- Doubles hidden_size output
- Requires adjusting FC layer input dimension

### Adding Dropout
- Reduces overfitting
- Add after embedding and LSTM layers
- Typical dropout rate: 0.3-0.5

### Ensemble Approaches
- Train multiple models with different seeds
- Average predictions for robustness
- Useful for production deployments
