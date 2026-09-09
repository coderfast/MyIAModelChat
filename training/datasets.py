"""Dataset helpers: IterableDataset wrappers and tokenization mixin."""
import os
import logging

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import Dataset

from training.config import CACHE_TOKENIZED_DATASET_DIR

logger = logging.getLogger(__name__)


class TokenPairIterableDataset(IterableDataset):
    """Iterable dataset that yields input-output token pairs without materializing all in memory."""
    def __init__(self, sequence_generator, length=None, rank=0, world_size=1):
        self.sequence_generator = sequence_generator
        self._length = length
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        if self._world_size > 1:
            return self._sharded_iter()
        return iter(self.sequence_generator())

    def _sharded_iter(self):
        for i, item in enumerate(self.sequence_generator()):
            if i % self._world_size == self._rank:
                yield item

    def __len__(self):
        if self._length is not None:
            if self._world_size > 1:
                return self._length // self._world_size
            return self._length
        raise TypeError("TokenPairIterableDataset length not set")


class _RangedIterableDataset(IterableDataset):
    """Wraps a generator factory to yield only items in [offset, offset+limit)."""
    def __init__(self, base_factory, offset, limit, length=None):
        self._base_factory = base_factory
        self._offset = offset
        self._limit = limit
        self._length = length

    def __iter__(self):
        count = 0
        for i, item in enumerate(self._base_factory()):
            if i < self._offset:
                continue
            yield item
            count += 1
            if self._limit > 0 and count >= self._limit:
                break

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("_RangedIterableDataset length not set")


class DatasetMixin:
    """Mixin providing dataset loading, tokenization, and DataLoader helpers.

    Expects the host class to define: self.loaded_dataset, self.tokenizer,
    self.raw_dataset, self.rank, self.world_size, self.use_gpu,
    self.max_ram_bytes, and self._get_num_proc().
    """

    def _sample_generator(self):
        """Yield text or tokenized sequences from the loaded dataset in streaming mode."""
        for item in self.loaded_dataset:
            value = item.get('token_ids', item.get('input_ids', None))
            if value is None:
                continue

            if isinstance(value, str):
                raw = value.strip()
                if raw:
                    yield raw, None
            elif isinstance(value, dict):
                input_text = value.get('input', '').strip()
                output_text = value.get('output', '').strip()
                merged = f"{input_text} {output_text}".strip()
                if merged:
                    yield merged, None
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue

                if all(isinstance(v, int) for v in value):
                    yield None, list(value)
                else:
                    merged = " ".join(str(v).strip() for v in value if isinstance(v, str) and str(v).strip())
                    if merged:
                        yield merged, None
            elif hasattr(value, 'tolist'):
                seq = list(value.tolist())
                if seq and all(isinstance(v, int) for v in seq):
                    yield None, seq
                else:
                    text_tokens = " ".join(str(v).strip() for v in seq if str(v).strip())
                    if text_tokens:
                        yield text_tokens, None

    def collate_fn(self, batch):
        """Collate function for DataLoader with proper padding and truncation."""
        if not batch:
            return torch.LongTensor([]), torch.LongTensor([])

        input_sequences, output_sequences = zip(*batch)

        max_positions = getattr(self.config, 'n_positions', 512) if hasattr(self, 'config') else 512

        # Truncate sequences that exceed max_positions
        input_sequences = [seq[:max_positions] for seq in input_sequences]
        output_sequences = [seq[:max_positions] for seq in output_sequences]

        # Find max length across all sequences
        max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))

        # Pad sequences to max length
        pad_idx = self.tokenizer.get_pad_index()
        padded_inputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in input_sequences]
        padded_outputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in output_sequences]

        # Convert to tensors
        input_tensor = torch.LongTensor(padded_inputs)
        output_tensor = torch.LongTensor(padded_outputs)

        return input_tensor, output_tensor

    def _get_csv_dataloader(self, batch_size):
        """Build a dataloader from CSV data for fine-tuning."""
        if getattr(self, 'raw_dataset', None) is None:
            return None

        def csv_pair_generator():
            for item in self.raw_dataset:
                value = item.get('input_ids', None)
                if not isinstance(value, str):
                    continue
                if not value.startswith('Pregunta:'):
                    continue
                token_seq = self.tokenizer.encode(value)
                if not token_seq or len(token_seq) <= 1:
                    continue
                yield token_seq[:-1], token_seq[1:]

        csv_length = len(self.raw_dataset) if hasattr(self.raw_dataset, '__len__') else None
        return DataLoader(
            TokenPairIterableDataset(csv_pair_generator, length=csv_length, rank=self.rank, world_size=self.world_size),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.use_gpu,
            num_workers=0
        )

    def _tokenize_dataset_item(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            if all(isinstance(v, int) for v in value):
                return value
            return self.tokenizer.encode(" ".join(str(v) for v in value))
        if isinstance(value, dict):
            input_text = value.get('input', '').strip()
            output_text = value.get('output', '').strip()
            merged = f"{input_text} {output_text}".strip()
            return self.tokenizer.encode(merged) if merged else []
        if isinstance(value, str):
            return self.tokenizer.encode(value)
        if hasattr(value, 'tolist'):
            seq = list(value.tolist())
            if seq and all(isinstance(v, int) for v in seq):
                return seq
            return self.tokenizer.encode(" ".join(str(v) for v in seq))
        return []

    def _tokenize_batch(self, batch):
        if 'token_ids' in batch:
            return {'token_ids': batch['token_ids']}

        if 'input_ids' in batch:
            if all(isinstance(value, str) for value in batch['input_ids']):
                tokenized = self.tokenizer.batch_encode(
                    batch['input_ids'],
                    add_language_token=False,
                    remove_accents_flag=False,
                    pad=False,
                    return_tensors=False
                )
            else:
                tokenized = [self._tokenize_dataset_item(value) for value in batch['input_ids']]
            return {'token_ids': tokenized}

        if 'input' in batch and 'output' in batch:
            texts = [
                f"{inp.strip()} {out.strip()}".strip()
                for inp, out in zip(batch['input'], batch['output'])
            ]
        elif 'text' in batch:
            texts = batch['text']
        elif 'sentence' in batch:
            texts = batch['sentence']
        else:
            columns = [k for k in batch.keys() if k not in ('__index_level_0__', 'token_ids')]
            length = len(batch[next(iter(batch))]) if batch else 0
            texts = []
            for i in range(length):
                pieces = []
                for key in columns:
                    value = batch[key][i]
                    if value is None:
                        continue
                    pieces.append(str(value))
                texts.append(' '.join(pieces).strip())

        tokenized = self.tokenizer.batch_encode(
            texts,
            add_language_token=False,
            remove_accents_flag=False,
            pad=False,
            return_tensors=False
        )
        return {'token_ids': tokenized}

    def _get_tokenized_dataset(self):
        if os.path.exists(CACHE_TOKENIZED_DATASET_DIR):
            logger.info(f"Loading tokenized dataset from cache: {CACHE_TOKENIZED_DATASET_DIR}")
            tokenized_ds = Dataset.load_from_disk(CACHE_TOKENIZED_DATASET_DIR)
            logger.info(f" Loaded tokenized dataset with {len(tokenized_ds)} samples")
            return tokenized_ds

        if 'token_ids' in self.loaded_dataset.column_names:
            logger.info("Dataset already contains token_ids; skipping tokenization")
            return self.loaded_dataset

        logger.info("Tokenizing cached dataset for faster training...")

        PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'question', 'answer', 'type', 'thinking', 'has_tool_call')
        columns_to_remove = [c for c in self.loaded_dataset.column_names if c not in PRESERVED_COLUMNS]
        num_proc = self._get_num_proc()
        try:
            tokenized_ds = self.loaded_dataset.map(
                self._tokenize_batch,
                batched=True,
                batch_size=512,
                num_proc=num_proc,
                remove_columns=columns_to_remove
            )
        except Exception as e:
            logger.warning(f"Could not tokenize dataset with num_proc={num_proc}: {e}. Falling back to num_proc=1")
            tokenized_ds = self.loaded_dataset.map(
                self._tokenize_batch,
                batched=True,
                batch_size=512,
                num_proc=1,
                remove_columns=columns_to_remove
            )

        os.makedirs(CACHE_TOKENIZED_DATASET_DIR, exist_ok=True)
        tokenized_ds.save_to_disk(CACHE_TOKENIZED_DATASET_DIR)
        logger.info(f" Saved tokenized dataset to {CACHE_TOKENIZED_DATASET_DIR}")
        return tokenized_ds
