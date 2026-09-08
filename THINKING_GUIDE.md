# Chain-of-Thought Reasoning

## Overview

MyIAModelChat supports **chain-of-thought reasoning** using GPT-2 standard tokens. The model learns to generate structured reasoning before answering, improving response quality through explicit thinking steps.

---

## Token System

### GPT-2 Standard Tokens

| Token | Role |
|-------|------|
| `<\|problem\|>` | Prefix marking the question/problem |
| `<\|thinking\|>` | Prefix marking the reasoning section |
| `<\|final\|>` | Prefix marking the final answer |

### Sample Formats

#### THINKING sample (with chain-of-thought)

```
<|problem|>question<|thinking|>reasoning<|final|>answer
```

Breakdown:
```
<|problem|>         ← TOKEN: marks the question
question            ← User's question (plain text)
<|thinking|>        ← TOKEN: marks start of reasoning
reasoning           ← Chain-of-thought text (NLP analysis)
<|final|>           ← TOKEN: marks start of final answer
answer              ← The actual answer
```

#### TEXT sample (direct Q&A, no reasoning)

```
<|problem|>question<|final|>answer
```

### Why These Tokens?

- **`<|problem|>`** clearly marks the input question
- **`<|thinking|>`** enables chain-of-thought reasoning
- **`<|final|>`** separates reasoning from the answer
- Follows GPT-2 standard format for compatibility

---

## Dataset Format

Each training sample is stored with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | list[int] | Full token sequence (model input) |
| `token_ids` | list[int] | Full token sequence (teacher-forced target) |
| `question` | string | Raw question text |
| `answer` | string | Raw answer text |
| `type` | string | `CONTEXT` or `THINKING` |
| `thinking` | string | Reasoning text (empty for CONTEXT samples) |

### Example Dataset Row

```json
{
  "question": "what is Python",
  "answer": "Python is a high-level programming language",
  "type": "THINKING",
  "thinking": "The user is asking about Python. Python is a versatile programming language created by Guido van Rossum in 1991.",
  "token_ids": [5, 123, 456, ..., 8, 789, 101, ..., 6, 321, 654, ...]
}
```

---

## Loss Weighting

The trainer applies **differentiated loss** based on token position and sample type:

### TEXT samples (no thinking)

```
<|problem|>question<|final|>answer
───────────────────────── ────────────
  weight = 0.0              weight = 1.0
```

### THINKING samples

```
<|problem|>question<thinking|>reasoning<|final|>answer
────────────────────────── ────────────────────  ──────────────
  weight = 0.0                weight = 0.5            weight = 1.0
  (preamble)                  (reasoning)             (answer)
```

| Segment | Weight | Rationale |
|---------|--------|-----------|
| Preamble (before `<|thinking|>`) | 0.0 | Don't penalize prefix learning |
| `<|thinking|>` delimiter | 1.0 | Must learn delimiter |
| Reasoning content | 0.5 | Learn structure, allow variation |
| `<|final|>` delimiter | 1.0 | Must learn delimiter |
| Answer tokens | 1.0 | Full weight on response |

### Configuration

```python
@dataclass
class TrainingConfig:
    thinking_loss_weight: float = 0.5  # Weight for reasoning tokens
```

---

## Flujo completo

### 1. Preparar datos con thinking

```bash
# NLP-based thinking (no external dependencies)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --refresh-cache

# Ollama teacher (higher quality, requires Ollama)
python main.py --prepare-data --aiml --hf --thinking-mode ollama --refresh-cache

# Multilingual support (30 languages)
python main.py --prepare-data --aiml --hf --thinking-mode nlp --allowed-languages es,en,fr,de,it,pt,ca,gl,eu,ro,sv,no,da,fi,nl,pl,cs,sk,hu,bg,hr,sr,sl,bs,mk,sq,el,et,lv,lt --refresh-cache
```

### 2. Entrenar el modelo

```bash
# Standard training (random init)
python main.py --train --epochs 30

# CPU-only mode
python main.py --train --cpu --epochs 30

# With specific checkpoint
python main.py --train --checkpoint-name my_model --epochs 30
```

### 3. Inferencia

```bash
# Chat with thinking visible
python main.py --chat --show-thinking

# Chat with specific model
python main.py --chat --model my_model --show-thinking
```

---

## Special Tokens in BPE

All thinking tokens are registered as `user_defined_symbols` during SentencePiece BPE training:

```
--user_defined_symbols=<|problem|>,<|thinking|>,<|final|>,<|user|>,<|assistant|>,<tool_call>,</tool_call>,<|tool_result|>
```

This ensures they are **never fragmented** into sub-tokens (always whole tokens).

### Token IDs

| Token | Accessor Method |
|-------|----------------|
| `<pad>` | `get_pad_index()` |
| `<unk>` | `get_unk_index()` |
| `<s>` | `get_bos_index()` |
| `</s>` | `get_eos_index()` |
| `<\|problem\|>` | `get_problem_index()` |
| `<\|thinking\|>` | `get_thinking_index()` |
| `<\|final\|>` | `get_final_index()` |
| `<\|user\|>` | `get_user_index()` |
| `<\|assistant\|>` | `get_assistant_index()` |
| `<tool_call>` | `get_tool_call_index()` |
| `<\|tool_result\|>` | `get_tool_result_index()` |

---

## Data Preparation

### Sources with Thinking Support

| Source | Thinking Generator | Quality |
|--------|-------------------|---------|
| AIML | AIML thinking module | High |
| PDF | PDF thinking module | Medium |
| EPUB | EPUB thinking module | Medium |
| CSV | CSV thinking module | High |
| HuggingFace | HF thinking module | Variable |
| Web | Web thinking module | Medium |

### What Data Preparation Does

1. Generates thinking data for each source (AIML, PDF, HF, etc.)
2. Creates TWO samples per entry (CONTEXT and THINKING)
3. Trains BPE model with all 5 special tokens
4. Tokenizes all data including thinking
5. Stores in `dataset_cache/`

### Generated Files

```
dataset_cache/
├── prepared_dataset/          # Dataset with token_ids
├── sentencepiece.model        # BPE model (includes all special tokens)
├── cache_metadata.pkl         # Includes has_thinking_tokens: true
└── dataset_stats.pkl          # Statistics
```

---

## Training

### What Training Does

- Detects thinking data via mode tokens (`<|thinking|>`, `<|context|>`) or content tags
- Applies differentiated loss weighting based on token position
- Tracks thinking metrics (delimiter accuracy, reasoning coverage)
- Registers metrics per epoch

### Expected Output

```
Training batch 50/inf in progress...
  Thinking Metrics Summary:
    thinking_token_accuracy: 0.8234
    thinking_open_accuracy: 0.8189
    thinking_close_accuracy: 0.8278
    thinking_coverage: 0.6543
    response_token_accuracy: 0.7891
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| `thinking_token_accuracy` | Combined accuracy for `<\|thinking\|>` and `<\|final\|>` delimiters |
| `thinking_open_accuracy` | Accuracy for `<\|thinking\|>` opening token |
| `thinking_close_accuracy` | Accuracy for `<\|final\|>` closing token |
| `thinking_coverage` | Ratio of thinking tokens to total tokens |
| `thinking_token_accuracy_full` | Accuracy on all thinking content tokens |
| `response_token_accuracy` | Accuracy on answer tokens |

---

## Inference

### Console

```bash
# With thinking visible
python main.py --chat --show-thinking
# You: What is Python?
# Bot [thinking]: The user is asking about Python. Python is a programming language.
# Bot: Python is a high-level programming language.

# Without thinking (clean response)
python main.py --chat
# You: What is Python?
# Bot: Python is a high-level programming language.
```

### Response Parsing

The dialogue manager uses GPT-2 standard tokens for inference:

```python
# Builds prompt as:
<|problem|>question<|thinking|>

# Parses response by splitting on <|final|>
# Extracts thinking portion and clean response
```

### Token Reference

| Token | Purpose |
|-------|---------|
| `<\|problem\|>` | Marks the question |
| `<\|thinking\|>` | Opens reasoning block |
| `<\|final\|>` | Opens final answer |
| `<\|user\|>` | User message (agentic) |
| `<\|assistant\|>` | Assistant response (agentic) |

### API

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is Python?"}],
    "include_thinking": true
  }'
```

Response with thinking:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Python is a high-level programming language",
      "reasoning": "The user is asking about Python. Python is a programming language."
    }
  }]
}
```

---

## Multilingual Support

The thinking engine supports 30 languages:

### EU Languages
English, Spanish, French, German, Italian, Portuguese, Catalan, Galician, Basque, Irish, Dutch, Danish, Swedish, Finnish, Polish, Czech, Slovak, Hungarian, Romanian, Bulgarian, Croatian, Slovenian, Greek, Estonian, Latvian, Lithuanian, Maltese

### Eastern European
Serbian, Bosnian, Macedonian, Albanian

### How It Works

1. Language detected from input text
2. Appropriate reasoning template selected
3. Language-specific patterns applied (greetings, question starters, etc.)
4. Thinking generated in the same language as input

---

## Files Reference

### Core Files

| File | Purpose |
|------|---------|
| `commons/tokenizer/bpe_tokenizer.py` | Special token definitions and helpers |
| `dataset_preparer/thinking_generators.py` | `_format_thinking_sample()` with mode tokens |
| `dataset_preparer/thinking_engine.py` | NLP reasoning generation (30 languages) |
| `dataset_preparer/thinking_quality.py` | Quality validation with multilingual patterns |
| `training/trainer.py` | Mode-aware loss computation and metrics |
| `commons/dialogue/dialogmanager.py` | Inference with mode token parsing |
| `inference/chat_engine.py` | Chat interface with thinking display |
| `dataset_preparer/data_preparer.py` | BPE training with special tokens |

### Legacy Files

| File | Purpose |
|------|---------|
| `dataset_preparer/generate_thinking_data.py` | Older format without mode tokens |

---

## Troubleshooting

### Model doesn't generate `<|thinking|>`

- Verify dataset contains thinking data: check `type` column for THINKING samples
- Verify BPE was trained with all special tokens
- Train more epochs (model needs time to learn structure)

### Thinking appears as `⁇`

- BPE model was trained without the tokens
- Re-run: `python main.py --prepare-data --aiml --hf --thinking-mode nlp --bpe-vocab-size 8000 --refresh-cache`
- Re-train the model

### Thinking not visible in chat

- Use `--show-thinking` flag
- Verify model was trained with thinking data
- Check that response actually contains `<|final|>` delimiter

### Validation checklist

- [ ] `<|problem|>` prefix is present at start of THINKING samples
- [ ] `<|thinking|>` delimiter opens reasoning block
- [ ] `<|final|>` delimiter separates reasoning from answer
- [ ] Reasoning logically leads to the answer
- [ ] Language matches between question, thinking, and answer
- [ ] Appropriate length ratio (1:1 to 3:1 thinking:answer)

---

*Updated: 2026-08-09 - Reflects new mode token system, 30-language support, and differentiated loss weighting*
