# Development & Debugging Guide

## Common Issues & Solutions

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Activate virtual environment
.\envMyIAModelChat\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

**Prevention**:
- Always activate environment before running scripts
- Check virtual environment is active in terminal

---

### Issue: CUDA/GPU Memory Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Option 1: Force CPU usage
device = torch.device('cpu')

# Option 2: Reduce batch size in DataLoader
loader = DataLoader(dataset, batch_size=8)  # Was 32

# Option 3: Clear cache
torch.cuda.empty_cache()

# Option 4: Use gradient checkpointing
model.gradient_checkpointing_enable()
```

**Prevention**:
- Start with small batch sizes
- Monitor GPU memory usage: `nvidia-smi`
- Use mixed precision training

---

### Issue: AIML Files Not Loading

**Problem**: AIML data shows as empty in training

**Solution**:
```python
# Debug aimlloder.py
from aimlloder import *

# Check if AIML files are found
aiml_files = os.listdir('aiml/')
print(f"Found {len(aiml_files)} AIML files")

# Manually load one file
from python-aiml import Kernel
k = Kernel()
k.learn("aiml/ai.aiml")
print(f"Patterns loaded: {len(k._brain._nodes)}")
```

**Prevention**:
- Verify `aiml/` directory exists and contains .aiml files
- Check file permissions
- Test with `--aiml` flag after confirming files present

---

### Issue: Tokenizer Mismatch Between Train & Inference

**Problem**: Model works in training but fails during chat

**Solution**:
```python
# Ensure same tokenizer is used everywhere
# main_train.py
tokenizer.save('tokenizer.pth')

# main_chat.py
tokenizer = SimpleTokenizer()
tokenizer.load('tokenizer.pth')

# Verify they're identical
test_text = "Hello world"
train_ids = train_tokenizer.encode(test_text)
inference_ids = tokenizer.encode(test_text)
assert train_ids == inference_ids, "Tokenizers don't match!"
```

**Prevention**:
- Save tokenizer only once during training
- Use same `tokenizer.pth` file for all inference
- Version control tokenizer files

---

### Issue: Tokenizer Issues

**Problem**: Slow decoding, vocabulary corruption, or inconsistent tokenization

**Solution**:
```python
# For SentencePiece BPE tokenizer:
from bpe_tokenizer import SentencePieceTokenizerWrapper

tokenizer = SentencePieceTokenizerWrapper('dataset_cache/sentencepiece.model')

# Test multilingual support
test_texts = [
    "Hello world",      # English
    "Hola mundo",       # Spanish
    "Bonjour le monde", # French
    "Hallo Welt",       # German
]

for text in test_texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Original: {text}")
    print(f"Decoded: {decoded}")
    print(f"Tokens: {ids}")
    print()
```

**Recent Fixes Applied**:
- **Migration to BPE**: Replaced word-level tokenizers with SentencePiece BPE
- **Multilingual support**: BPE handles any language natively
- **Eliminated race conditions**: BPE training is single-threaded and atomic
- **Consistent tokenization**: Same model used for training and inference

**Prevention**:
- Use `SentencePieceTokenizerWrapper` for all tokenization
- Save BPE model to `dataset_cache/sentencepiece.model`
- Test tokenization consistency across runs
- Monitor decoding performance (should be near-instant)

---

## Debugging Tools & Techniques

### Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug(f"Model device: {device}")
logger.debug(f"Input shape: {input_tensor.shape}")
```

### Print Debugging

```python
# Add strategic print statements
print(f"[DEBUG] Input text: {repr(user_input)}")
print(f"[DEBUG] Intent: {intent}")
print(f"[DEBUG] Sentiment: {sentiment}")
print(f"[DEBUG] Context tokens: {input_ids}")
print(f"[DEBUG] Model output shape: {output.shape}")
print(f"[DEBUG] Response tokens: {output_ids}")
print(f"[DEBUG] Response text: {repr(response_text)}")
```

### Interactive Debugging (pdb)

```python
import pdb

# In your code, add breakpoint
pdb.set_trace()

# Commands:
# n (next)
# s (step into)
# c (continue)
# p variable (print variable)
# l (list code)
# h (help)
```

### PyTorch Debugging

```python
# Check tensor properties
print(f"Dtype: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Shape: {tensor.shape}")
print(f"Requires grad: {tensor.requires_grad}")
print(f"Min/Max: {tensor.min()}, {tensor.max()}")

# Check for NaNs
assert not torch.isnan(output).any(), "Output contains NaN"
assert not torch.isinf(output).any(), "Output contains Inf"

# Monitor gradients
for name, param in model.named_parameters():
    print(f"Gradient norm for {name}: {param.grad.norm()}")
```

---

## Performance Analysis

### Profiling Training

```python
import cProfile
import pstats
from io import StringIO

pr = cProfile.Profile()
pr.enable()

# Your training loop here
train_one_epoch()

pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(10)  # Top 10 functions
print(s.getvalue())
```

### Timing Code Blocks

```python
import time

start = time.time()
# Your code here
elapsed = time.time() - start
print(f"Execution time: {elapsed:.2f} seconds")

# With timing decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def expensive_function():
    pass
```

---

## Code Quality

### Linting & Formatting

```bash
# Install tools
pip install pylint black flake8

# Check code style
pylint main_train.py
flake8 chatmodel.py

# Auto-format code
black chatmodel.py
```

### Type Checking

```bash
pip install mypy

# Check type hints
mypy main_train.py
```

### Testing

```python
# Create test_chatmodel.py
import unittest
from chatmodel import ChatModel

class TestChatModel(unittest.TestCase):
    def setUp(self):
        self.model = ChatModel(vocab_size=1000, embedding_dim=100, hidden_size=256)
    
    def test_forward_shape(self):
        input_tensor = torch.LongTensor([[1, 2, 3, 4]])
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 4, 1000))
    
    def test_device_transfer(self):
        self.model.to('cpu')
        self.assertTrue(next(self.model.parameters()).is_cpu)

if __name__ == '__main__':
    unittest.main()
```

---

## Common Development Patterns

### Safe File Operations

```python
import os
from pathlib import Path

# Check file exists
if not os.path.exists('chat_model.pth'):
    print("Model file not found!")

# Create directories if needed
Path('checkpoints').mkdir(parents=True, exist_ok=True)

# Safe file writing
try:
    torch.save(model.state_dict(), 'chat_model.pth')
except IOError as e:
    print(f"Failed to save model: {e}")
```

### Configuration Management

```python
# config.py
class Config:
    vocab_size = 10000
    embedding_dim = 100
    hidden_size = 256
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Usage
cfg = Config()
model = ChatModel(cfg.vocab_size, cfg.embedding_dim, cfg.hidden_size)
```

### Error Recovery

```python
def robust_inference(model, input_tensor, max_retries=3):
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                return model(input_tensor)
        except RuntimeError as e:
            if attempt < max_retries - 1:
                torch.cuda.empty_cache()
                print(f"Attempt {attempt+1} failed, retrying...")
            else:
                raise e
```

---

## VS Code Integration

### Launch Configuration (.vscode/launch.json)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main_train.py",
            "console": "integratedTerminal",
            "args": ["--epochs", "10", "--aiml", "--hf"]
        },
        {
            "name": "Chat Interface",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main_chat.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### Task Configuration (.vscode/tasks.json)

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Train Model",
            "type": "shell",
            "command": "python",
            "args": ["main_train.py", "--epochs", "10"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

---

## Git & Version Control

### Useful Commands

```bash
# Check modified files
git status

# Stage changes
git add main_train.py

# Commit with descriptive message
git commit -m "Fix intent detection in DialogueManager"

# View recent changes
git log --oneline -10

# Diff before committing
git diff main_train.py

# Revert changes
git checkout main_train.py
```

### .gitignore Best Practices

```
# Already in your .gitignore
__pycache__/
*.pth
envMyIAModelChat/

# Add model checkpoints
checkpoints/

# Add large datasets
datasets/large_data/

# IDE files
.vscode/
```
