# Model Architecture Guide

## PyTorch ChatModel Architecture

### Layer Structure

```
ChatModel (nn.Module)
├── Embedding Layer
│   ├── Vocabulary Size → Embedding Dimension
│   └── Converts token IDs to dense vectors
├── LSTM Layer
│   ├── Input: embedded sequences
│   ├── Output: hidden states (bidirectional optional)
│   └── Captures sequential dependencies
└── Fully Connected Layer
    ├── LSTM hidden dim → Vocabulary size
    └── Outputs logits for next token prediction
```

### Forward Pass Flow

```
Input Tokens (LongTensor)
    ↓
Embedding Layer
    ↓ (token_ids → embedding_vectors)
Dense Embeddings
    ↓
LSTM Layer
    ↓ (processes sequences with memory)
LSTM Output (hidden states)
    ↓
Fully Connected Layer
    ↓ (projects to vocab dimension)
Output Logits
```

**NOTE: Fixed in recent update** - The forward pass now properly handles batch input dimensions and LSTM sequence processing. Previous version had issues with tensor shapes that caused incorrect gradient flow.

### Key Parameters (from chatmodel.py)

- **vocab_size**: Number of unique tokens in vocabulary (up to 50k)
- **embedding_dim**: Dimension of embedding vectors (256-512 recommended)
- **hidden_size**: LSTM hidden state dimension (512-1024 for complex data)
- **output_size**: Vocabulary size (same as vocab_size)

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
- Manages BilingualTokenizer and datasets
- Handles multiprocessing for data loading
- Supports AIML, PDF, EPUB, and Hugging Face datasets
- Implements best model checkpointing

**DataPreparer**
- Multi-source data loading and processing
- PDF text extraction with PyPDF2
- EPUB parsing with ebooklib
- Dataset caching system (12x speedup)
- Schema alignment for concatenation

**ChatDataset**
- PyTorch Dataset subclass
- Loads chat dialogue data from multiple sources
- Handles BilingualTokenizer
- Returns (input_ids, target_ids) pairs

**BilingualTokenizer**
- Enhanced tokenization for EN/ES support
- Automatic language detection
- Accent handling for Spanish text
- Efficient encoding/decoding with O(1) lookups

## Dialogue Pipeline (DialogueManager)

### Processing Flow with Language Support
```
User Input String
    ↓
Language Detection (EN/ES automatic)
    ↓
Intent Classification (BERT pipeline)
    ↓ (e.g., "greeting", "question", "statement")
Sentiment Analysis (BERT pipeline)
    ↓ (e.g., "positive", "negative", "neutral")
Context Management
    ↓ (deque maintains last 5 turns)
Persona Modeling
    ↓ (adjusts response based on personality)
Bilingual Tokenization
    ↓ (string → token IDs with accent support)
Model Inference
    ↓ (autoregressive generation with penalization)
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

### Intent Recognition
- Uses `transformers.pipeline('zero-shot-classification')`
- Pre-trained BERT model (facebook/bart-large-mnli)
- Detects intent from user input
- Example outputs: greeting, question, statement, request

### Sentiment Analysis
- Uses `transformers.pipeline('sentiment-analysis')`
- Pre-trained BERT model (distilbert-base-uncased-finetuned-sst-2-english)
- Detects emotional tone: positive, negative, neutral
- Used for persona modeling adjustment

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
- **checkpoints/chat_model_best.pth**: Complete ChatModel state_dict with best validation loss
- **checkpoints/tokenizer.pkl**: BilingualTokenizer state (vocabulary, mappings, language detection)
- **dataset_cache/**: Cached processed datasets for fast loading
- **pretrained_embeddings.pth**: Pre-trained embedding weights (optional)

### Loading Models
```python
model = ChatModel(vocab_size, embedding_dim, hidden_size)
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
