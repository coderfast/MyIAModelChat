# Recommended HuggingFace Datasets

## Primary: lightblue/tagengo-gpt4

- **URL**: https://huggingface.co/datasets/lightblue/tagengo-gpt4
- **Size**: 78K conversations
- **Languages**: English (15,771), Spanish (8,318), + 72 more
- **Format**: Human-GPT-4 conversations
- **Use**: Casual conversational training data

### How to use

```python
from datasets import load_dataset

ds = load_dataset("lightblue/tagengo-gpt4", split="train")

# Filter EN and ES
ds = ds.filter(lambda x: x["language"] in ("en", "es"))

# Transform to GPT-2 format
def format_sample(example):
    text = example["conversation"][0]["content"]
    response = example["response"][0]
    return {
        "input_ids": f"<|problem|>{text}<|final|>{response}",
        "source": "hf",
        "language": example["language"]
    }

ds = ds.map(format_sample)
```

## Alternatives

- **tucnguyen/ShareChat**: https://huggingface.co/datasets/tucnguyen/ShareChat
- **lindazeng979/bilingual-babyLM**: https://huggingface.co/datasets/lindazeng979/bilingual-babyLM
