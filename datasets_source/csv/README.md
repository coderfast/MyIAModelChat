# CSV Data Sources

This directory contains CSV files for training data. All CSV files should use GPT-2 standard tokens format.

## GPT-2 Standard Tokens

| Token | Purpose |
|-------|---------|
| `<|problem|>` | Question/problem prefix |
| `<|thinking|>` | Reasoning prefix |
| `<|final|>` | Answer prefix |

## CSV Formats

### 1. Simple Q&A (special_facts.csv)

```csv
input,output
"<|problem|>What is Python?<|final|>","Python is a programming language."
"<|problem|>Who created AIML?<|final|>","Dr. Richard S. Wallace created AIML."
```

### 2. With Thinking (thinking_data.csv)

```csv
input,output,thinking,thinking_text,category
"<|problem|>What is AI?<|thinking|>","<|final|>Artificial intelligence is...","Short reasoning","<|problem|>What is AI?<|thinking|>Short reasoning<|final|>Artificial intelligence is...",default
```

## Columns

| Column | Required | Description |
|--------|----------|-------------|
| `input` | Yes | GPT-2 format input with `<|problem|>` and optionally `<|thinking|>` |
| `output` | Yes | GPT-2 format output with `<|final|>` |
| `thinking` | No | Short reasoning text (without tokens) |
| `thinking_text` | No | Full thinking text with GPT-2 tokens |
| `category` | No | Category label (identity, default, etc.) |

## Example Files

- `special_facts.csv` - Simple Q&A pairs
- `thinking_data.csv` - Q&A with chain-of-thought reasoning

## Adding New Data

1. Create a new `.csv` file in this directory
2. Use the formats above with GPT-2 standard tokens
3. Run `python main.py --prepare-data --csv` to include in training
