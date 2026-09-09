"""PyTorch Dataset for chatbot input-output pairs.

Provides a tokenized dataset compatible with the Trainer's collate_fn,
which expects tuples of (input_token_ids, output_token_ids).

Usage:
    dataset = ChatDataset(tokenizer, samples, max_length=512)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=trainer.collate_fn)
"""

from torch.utils.data import Dataset


class ChatDataset(Dataset):
    """Tokenized chat dataset returning (input_ids, output_ids) pairs.

    Args:
        tokenizer: Tokenizer with encode() returning list[int].
        samples: Iterable of dicts with 'input' and 'output' string keys.
        max_length: Truncate sequences to this length (0 = no truncation).
    """

    def __init__(self, tokenizer, samples=None, max_length=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if samples is not None:
            self._load_samples(samples)

    def _load_samples(self, samples):
        """Tokenize and store input-output pairs."""
        for sample in samples:
            input_text = sample.get('input', '')
            output_text = sample.get('output', '')

            if not input_text or not output_text:
                continue

            input_ids = self.tokenizer.encode(input_text)
            output_ids = self.tokenizer.encode(output_text)

            if not input_ids or not output_ids:
                continue

            if self.max_length > 0:
                input_ids = input_ids[:self.max_length]
                output_ids = output_ids[:self.max_length]

            self.data.append((input_ids, output_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
