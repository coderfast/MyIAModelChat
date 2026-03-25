# NLP & Transformers Integration Guide

## Transformers Models Used

### Intent Classification

```python
from transformers import pipeline

intent_classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli'
)

# Usage
result = intent_classifier(
    "What time is it?",
    candidate_labels=['greeting', 'question', 'statement', 'request']
)
# Output: {'sequence': 'What time is it?', 
#          'labels': ['question', 'statement', ...],
#          'scores': [0.95, 0.04, ...]}
```

### Sentiment Analysis

```python
from transformers import pipeline

sentiment_analyzer = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english'
)

# Usage
result = sentiment_analyzer("I love this!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Pipeline Stages in DialogueManager

### Stage 1: Intent Detection

**Purpose**: Understand what the user is trying to do

**Candidate Intents**:
- greeting: "Hi", "Hello", "Hey"
- question: "What", "How", "Why", "Where", "When"
- statement: "I think", "In my opinion"
- request: "Can you", "Could you", "Please"
- sentiment_inquiry: "How are you", "Are you okay"
- knowledge: "Tell me about", "Explain"

**Usage in DialogueManager**:
```python
intent = self.intent_classifier(user_input)[0]['label']
print(f"Detected intent: {intent}")
# Persona response is then adjusted based on intent
```

### Stage 2: Sentiment Analysis

**Purpose**: Understand emotional tone

**Output Labels**:
- POSITIVE: user is happy, satisfied, excited
- NEGATIVE: user is angry, frustrated, sad
- NEUTRAL: user is matter-of-fact

**Usage in DialogueManager**:
```python
sentiment = self.sentiment_analyzer(user_input)[0]['label']
print(f"Detected sentiment: {sentiment}")
# Adjust response tone accordingly
```

## Advanced NLP Techniques

### Text Preprocessing for Intent/Sentiment

```python
import re
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove special chars but keep punctuation
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Before sending to transformers
cleaned_input = preprocess_text(user_input)
intent = intent_classifier(cleaned_input)
```

### Multi-label Intent Detection

For inputs that express multiple intents:

```python
def detect_multi_intent(text, num_intents=3):
    result = intent_classifier(
        text,
        candidate_labels=['greeting', 'question', 'request', 'statement'],
        multi_class=True  # Allow multiple labels
    )
    return result['labels'][:num_intents]

# Usage
intents = detect_multi_intent("Hi, can you tell me about Python?")
# Returns: ['greeting', 'question', 'request']
```

## Model Selection & Alternatives

### For Intent Classification

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| facebook/bart-large-mnli | Medium | High | 1.6GB | Current (recommended) |
| distilbert-base-uncased | Fast | Good | 268MB | Resource-constrained |
| roberta-large-mnli | Slow | Very High | 1.4GB | High accuracy needed |
| xlm-roberta-base | Medium | Good | 564MB | Multilingual |

### For Sentiment Analysis

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| distilbert-fst-sst-2 | Fast | Good | Current (recommended) |
| bert-base-uncased-fst | Medium | Better | More accurate |
| distilbert-multilingual | Medium | Good | Multiple languages |
| roberta-large-sst2 | Slow | Excellent | Maximum accuracy |

### Switching Models

```python
from transformers import pipeline

# Change intent classifier model
intent_classifier = pipeline(
    'zero-shot-classification',
    model='distilbert-base-uncased'  # Smaller, faster
)

# Change sentiment analyzer model
sentiment_analyzer = pipeline(
    'sentiment-analysis',
    model='bert-base-uncased-finetuned-sst-2-english'  # More accurate
)
```

## Custom Fine-tuning

### Fine-tune Intent Classifier for Domain

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load your custom intent dataset
dataset = load_dataset('json', data_files='intent_data.json')

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    'facebook/bart-large-mnli',
    num_labels=5
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./intent_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Fine-tune
trainer.train()

# Save and use
model.save_pretrained('./intent_model')
```

## Context-Aware Intent Detection

### Incorporating Conversation History

```python
def detect_intent_with_context(current_input, history):
    # Concatenate history with current input
    context = " ".join(history[-3:]) + " " + current_input
    
    # Detect intent on full context
    result = intent_classifier(
        context,
        candidate_labels=['greeting', 'question', 'statement', 'request']
    )
    
    return result['labels'][0]

# Usage in DialogueManager
context_intent = detect_intent_with_context(
    user_input, 
    list(self.history)
)
```

## Error Handling & Robustness

### Fallback Strategies

```python
def safe_intent_detection(text, fallback_intent='statement'):
    try:
        result = intent_classifier(text)
        return result['labels'][0]
    except Exception as e:
        print(f"Intent detection error: {e}")
        return fallback_intent

def safe_sentiment_analysis(text, fallback_sentiment='NEUTRAL'):
    try:
        result = sentiment_analyzer(text)
        return result[0]['label']
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return fallback_sentiment
```

### Handling Edge Cases

```python
def preprocess_for_nlp(text):
    # Empty input
    if not text or len(text.strip()) == 0:
        return "[No input]"
    
    # Very long input
    if len(text) > 512:
        text = text[:512]
        print("Warning: Input truncated to 512 characters")
    
    # Special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    return text.strip()
```

## Performance Optimization

### Model Caching

```python
from transformers import pipeline

# Cache models to avoid redownloading
intent_classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=0  # Use GPU if available
)

# Models cached in ~/.cache/huggingface/
```

### Batch Processing

```python
# Process multiple inputs at once for efficiency
texts = [
    "Hello there",
    "What time is it?",
    "I need help"
]

intents = intent_classifier(
    texts,
    candidate_labels=['greeting', 'question', 'request'],
    batch_size=3
)
```

### Mixed Precision

```python
import torch
from torch.cuda.amp import autocast

# Faster inference with lower precision
with autocast():
    intent = intent_classifier(user_input)
```

## Monitoring & Logging

### Track Intent/Sentiment Distribution

```python
from collections import defaultdict

intents_seen = defaultdict(int)
sentiments_seen = defaultdict(int)

for user_input in conversation:
    intent = intent_classifier(user_input)[0]['label']
    sentiment = sentiment_analyzer(user_input)[0]['label']
    
    intents_seen[intent] += 1
    sentiments_seen[sentiment] += 1

print(f"Intent distribution: {dict(intents_seen)}")
print(f"Sentiment distribution: {dict(sentiments_seen)}")
```
