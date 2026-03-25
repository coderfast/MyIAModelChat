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

### Key Parameters (from chatmodel.py)

- **vocab_size**: Number of unique tokens in vocabulary
- **embedding_dim**: Dimension of embedding vectors
- **hidden_size**: LSTM hidden state dimension
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

### DataLoading
1. Load AIML data via `aimlloder.py`
2. Load Hugging Face datasets
3. Combine datasets using `concatenate_datasets()`
4. Create ChatDataset instances
5. Wrap with DataLoader (supports multiprocessing)

### Training Loop
```python
for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input_ids)
        
        # Compute loss (CrossEntropyLoss typical)
        loss = criterion(output, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Key Training Classes

**MainTrain**
- Initializes training pipeline
- Manages tokenizer and datasets
- Handles multiprocessing for data loading
- Supports both AIML and Hugging Face datasets

**ChatDataset**
- PyTorch Dataset subclass
- Loads chat dialogue data
- Handles tokenization
- Returns (input_ids, target_ids) pairs

**SimpleTokenizer**
- Custom tokenization implementation
- `encode()`: converts text → token IDs
- `decode()`: converts token IDs → text
- Uses learned vocabulary

## Dialogue Pipeline (DialogueManager)

### Processing Flow
```
User Input String
    ↓
Intent Classification (BERT pipeline)
    ↓ (e.g., "greeting", "question", "statement")
Sentiment Analysis (BERT pipeline)
    ↓ (e.g., "positive", "negative", "neutral")
Context Management
    ↓ (deque maintains last 5 turns)
Persona Modeling
    ↓ (adjusts response based on personality)
Tokenization
    ↓ (string → token IDs)
Model Inference
    ↓ (forward pass on GPU/CPU)
Token ID Decoding
    ↓ (token IDs → string)
Response String
```

### Intent Recognition
- Uses `transformers.pipeline('zero-shot-classification')`
- Pre-trained BERT model
- Detects intent from user input
- Example outputs: greeting, question, statement, request

### Sentiment Analysis
- Uses `transformers.pipeline('sentiment-analysis')`
- Pre-trained BERT model
- Detects emotional tone: positive, negative, neutral
- Used for persona modeling adjustment

### Context Window
- Maintains last 5 user inputs/bot responses (configurable)
- Combined into single context string
- Used to condition model predictions
- Prevents loss of conversation coherence

## Saved Models

### Checkpoint Files
- **chat_model.pth**: Complete ChatModel state_dict
- **tokenizer.pth**: Tokenizer state (vocabulary, mappings)
- **pretrained_embeddings.pth**: Pre-trained embedding weights

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
