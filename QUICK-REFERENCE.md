# Quick Reference - MyIAModelChat

## File Structure Quick Map

| File | Purpose |
|------|---------|
| `chatmodel.py` | PyTorch LSTM model (Embedding → LSTM → FC) |
| `chatdataset.py` | PyTorch Dataset loader |
| `dialogmanager.py` | Dialogue flow, intent/sentiment, persona modeling |
| `simpletokenizer.py` | Custom tokenization (encode/decode) |
| `aimlloder.py` | Loads AIML files for training |
| `main_train.py` | Training pipeline (MainTrain class) |
| `main_chat.py` | Chat inference interface |
| `main.py` | Primary entry point |

## Command Cheat Sheet

```bash
# Activate environment
.\envMyIAModelChat\Scripts\Activate.ps1

# Train model (all data)
python main_train.py --epochs 10 --num-cores 4 --aiml --hf

# Train with AIML only
python main_train.py --epochs 10 --aiml

# Tokenize only (no training)
python main_train.py --onlytokenize

# Run chat interface
python main_chat.py

# Just preprocess data
python main.py --prepare-data
```

## Model Config Essentials

```python
# chatmodel.py - Key parameters
vocab_size = 10000        # Vocabulary size
embedding_dim = 100       # Embedding dimension
hidden_size = 256         # LSTM hidden state size
output_size = vocab_size  # Output vocabulary size

# dialogmanager.py - Key settings
max_history = 5           # Conversation history window
intent_model = "facebook/bart-large-mnli"
sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"

# main_train.py - Training params
batch_size = 32
learning_rate = 0.001
epochs = 10
num_workers = 4
```

## Model Architecture Flow

```
User Input (text)
    ↓
Intent Classifier (BERT)
    ↓
Sentiment Analyzer (BERT)
    ↓
Tokenizer.encode() → Token IDs
    ↓
ChatModel forward pass:
    Token IDs → Embedding → LSTM → FC → Logits
    ↓
argmax(logits) → Output Token IDs
    ↓
Tokenizer.decode() → Response text
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
                         persona, max_history=5)
response = dialog.generate_response(user_input)
intent = dialog.detected_intent
sentiment = dialog.detected_sentiment
```

### SimpleTokenizer
```python
from simpletokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.train(raw_text)
token_ids = tokenizer.encode("hello")
text = tokenizer.decode(token_ids)
```

### MainTrain
```python
from main_train import MainTrain

trainer = MainTrain(args)
trainer.prepare_datasets()  # Load AIML + HF datasets
trainer.train()
```

## Dataset Sources

1. **AIML** (`aiml/` folder): ~60 AIML pattern files
2. **Hugging Face**: Dialogue datasets (wikitext, etc.)
3. **Custom**: User-provided data in `datasets/` folder

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `chat_model.pth` | ~1GB | Trained model weights |
| `tokenizer.pth` | ~10MB | Vocabulary & token mappings |
| `pretrained_embeddings.pth` | ~50MB | Pre-trained embeddings (optional) |

## Common Fixes

| Issue | Fix |
|-------|-----|
| Import errors | Activate venv: `.\envMyIAModelChat\Scripts\Activate.ps1` |
| CUDA out of memory | Reduce batch size or use CPU |
| Gibberish responses | Ensure `model.eval()` is called |
| Tokenizer mismatch | Use same `tokenizer.pth` for train & inference |
| No AIML data | Verify `aiml/` folder exists and `--aiml` flag used |
| Slow training | Increase `num_workers`, use GPU |

## Debugging Commands

```python
# Check model structure
print(model)

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
