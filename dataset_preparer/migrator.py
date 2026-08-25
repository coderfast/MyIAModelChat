"""Migrator for datasets using legacy special tokens to the consolidated GPT-2 format.

Converts old token names to the canonical consolidated set:
- <|context|>   -> <|problem|>
- <|answer|>    -> <|final|>
- <thinking>    -> <|thinking|>
- </thinking>   -> <|final|>
- <observation> -> <|tool_result|>
- </observation> -> <|tool_result|>
"""

import logging
import re

logger = logging.getLogger(__name__)

LEGACY_TO_CANONICAL = [
    ('<|context|>', '<|problem|>'),
    ('<thinking>', '<|thinking|>'),
    ('</thinking><|answer|>', '<|final|>'),  # combined: avoids double <|final|>
    ('</thinking>', '<|final|>'),             # standalone close
    ('<|answer|>', '<|final|>'),              # standalone answer
]

# <observation> is a prefix-only token: <observation>X</observation> -> <|tool_result|>X<|end|>
_OBSERVATION_RE = re.compile(r'<observation>(.*?)</observation>', re.DOTALL)


def migrate_text(text: str) -> str:
    """Migrate a single text string from legacy tokens to the canonical format."""
    if not text:
        return text
    result = text
    for legacy, canonical in LEGACY_TO_CANONICAL:
        result = result.replace(legacy, canonical)
    # Rebuild prefix-only observation blocks BEFORE collapsing stray markers
    result = _OBSERVATION_RE.sub(lambda m: f"<|tool_result|>{m.group(1).strip()}<|end|>", result)
    # Collapse any remaining stray observation markers
    result = result.replace('<observation>', '').replace('</observation>', '')
    return result


def migrate_sample(sample: dict) -> dict:
    """Migrate a single dataset sample dict in-place compatible fields."""
    if not isinstance(sample, dict):
        return sample
    for key in ('input_ids', 'text', 'bpe_text', 'thinking_text', 'question', 'answer', 'input', 'output', 'thinking'):
        value = sample.get(key)
        if isinstance(value, str):
            sample[key] = migrate_text(value)
    return sample


def migrate_dataset_texts(dataset):
    """Migrate all text fields of a HuggingFace Dataset or list of dicts."""
    try:
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            text_columns = [c for c in dataset.column_names if c in (
                'input_ids', 'text', 'bpe_text', 'thinking_text',
                'question', 'answer', 'input', 'output', 'thinking',
            )]
            if not text_columns:
                logger.info("  No text columns to migrate")
                return dataset

            def _map(example):
                for col in text_columns:
                    v = example.get(col)
                    if isinstance(v, str):
                        example[col] = migrate_text(v)
                return example

            try:
                migrated = dataset.map(_map, batched=False, num_proc=1)
            except Exception as e:
                logger.warning(f"  Migration map failed: {e}")
                return dataset
            migrated_count = sum(
                1 for i in range(min(len(migrated), 1000))
                if any('<|' in str(migrated[i].get(c, '')) or '<thinking>' in str(migrated[i].get(c, '')) for c in text_columns)
            )
            logger.info(f"  Migrated {migrated_count}+ samples to canonical token format")
            return migrated

    except ImportError:
        pass

    if isinstance(dataset, list):
        return [migrate_sample(dict(s)) for s in dataset]

    return dataset


def migration_report(dataset) -> dict:
    """Return a report of how many samples contained legacy tokens before migration."""
    legacy_tokens = [legacy for legacy, _ in LEGACY_TO_CANONICAL]
    count = 0
    sample_count = 0
    try:
        for i in range(min(len(dataset), 1000)):
            sample = dataset[i]
            sample_count += 1
            for key in ('input_ids', 'text', 'bpe_text'):
                v = sample.get(key)
                if isinstance(v, str) and any(tok in v for tok in legacy_tokens):
                    count += 1
                    break
    except Exception:
        pass
    return {'legacy_samples': count, 'sampled': sample_count}