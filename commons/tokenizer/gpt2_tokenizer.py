"""GPT-2 tokenizer wrapper — provides same interface as SentencePieceTokenizerWrapper.

Uses HuggingFace's GPT2Tokenizer for tokenization when no custom BPE model exists.
Allows the model to chat immediately with GPT-2's vocabulary before fine-tuning.
"""

import logging
import torch

try:
    from transformers import GPT2Tokenizer
except Exception:
    GPT2Tokenizer = None

logger = logging.getLogger(__name__)


class GPT2TokenizerWrapper:
    """Wrapper around HuggingFace GPT2Tokenizer — same interface as SentencePieceTokenizerWrapper.

    Supports: encode, decode, batch_encode, get_pad_index, get_unk_index,
    get_eos_index, convert_ids_to_tokens, fit (no-op), save_vocabulary.
    Also exposes vocab and idx2word dicts for DialogueManager compatibility.
    """

    def __init__(self, model_name: str = 'gpt2', add_special_tokens: bool = False):
        if GPT2Tokenizer is None:
            raise ImportError("transformers package is required for GPT-2 tokenization")
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optionally add GPT-2 standard special tokens (only for custom training)
        if add_special_tokens:
            special_tokens = [
                '<|problem|>', '<|thinking|>', '<|final|>',
                '<|user|>', '<|assistant|>',
                '<tool_call>', '</tool_call>', '<|tool_result|>',
                '<thinking>', '</thinking>',
                '<|context|>', '<|answer|>',
                '<observation>', '</observation>',
            ]
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens to GPT-2 tokenizer")

        self._initialize_special_token_ids()
        self._build_vocab_dicts()
        logger.info(f"GPT-2 tokenizer loaded: vocab_size={self.vocab_size}")

    def _initialize_special_token_ids(self):
        def _get_id(token_str):
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return token_id
            return -1

        self._pad_id = self.tokenizer.pad_token_id or 0
        self._unk_id = self.tokenizer.unk_token_id or 1
        self._bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else -1
        self._eos_id = self.tokenizer.eos_token_id or 1

        # GPT-2 standard tokens — try to find them, fallback to -1
        self._problem_id = _get_id('<|problem|>')
        self._thinking_id = _get_id('<thinking>')
        self._thinking_end_id = _get_id('</thinking>')
        self._thinking_mode_id = _get_id('<|thinking|>')
        self._final_id = _get_id('<|final|>')
        self._user_id = _get_id('<|user|>')
        self._assistant_id = _get_id('<|assistant|>')
        self._tool_call_id = _get_id('<tool_call>')
        self._tool_call_end_id = _get_id('</tool_call>')
        self._tool_result_id = _get_id('<|tool_result|>')

        # Legacy tokens
        self._context_id = _get_id('<|context|>')
        self._answer_id = _get_id('<|answer|>')
        self._observation_id = _get_id('<observation>')
        self._observation_end_id = _get_id('</observation>')

        # Map legacy to GPT-2 standard
        if self._context_id < 0:
            self._context_id = self._problem_id
        if self._answer_id < 0:
            self._answer_id = self._final_id
        if self._observation_id < 0:
            self._observation_id = self._tool_result_id
        if self._observation_end_id < 0:
            self._observation_end_id = self._tool_result_id

        for name, tid in [
            ('pad', self._pad_id), ('unk', self._unk_id),
            ('bos', self._bos_id), ('eos', self._eos_id),
            ('<|problem|>', self._problem_id), ('<|thinking|>', self._thinking_mode_id),
            ('<thinking>', self._thinking_id), ('</thinking>', self._thinking_end_id),
            ('<|final|>', self._final_id),
            ('<|user|>', self._user_id), ('<|assistant|>', self._assistant_id),
            ('<tool_call>', self._tool_call_id), ('</tool_call>', self._tool_call_end_id),
            ('<|tool_result|>', self._tool_result_id),
        ]:
            status = 'OK' if tid >= 0 else 'NOT FOUND'
            logger.info(f"  Token {name} -> id={tid} ({status})")

    def _build_vocab_dicts(self):
        self._vocab = self.tokenizer.get_vocab()
        self._idx2word = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str, *args, **kwargs):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def batch_encode(self, texts, *args, **kwargs):
        return [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]

    def get_pad_index(self):
        return self._pad_id

    def get_unk_index(self):
        return self._unk_id

    def get_eos_index(self):
        return self._eos_id

    def decode(self, indices, skip_special_tokens=True):
        if indices is None:
            return ""
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        if isinstance(indices, int):
            indices = [indices]
        if not isinstance(indices, (list, tuple)):
            try:
                indices = list(indices)
            except TypeError:
                indices = [indices]

        token_ids = [int(i) for i in indices if i is not None]
        if not token_ids:
            return ""

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(self, ids):
        if ids is None:
            return []
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        return [self.tokenizer.convert_ids_to_tokens(int(i)) for i in ids]

    def fit(self, texts):
        pass

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        return self._vocab

    @property
    def idx2word(self):
        return self._idx2word

    def save_vocabulary(self, filepath: str):
        import json
        data = {'gpt2_model': self.model_name, 'vocab_size': self.vocab_size}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── GPT-2 Standard token helpers ───────────────────────────────

    def get_problem_index(self) -> int:
        return self._problem_id

    def get_final_index(self) -> int:
        return self._final_id

    def get_user_index(self) -> int:
        return self._user_id

    def get_assistant_index(self) -> int:
        return self._assistant_id

    def get_tool_result_index(self) -> int:
        return self._tool_result_id

    # ── Thinking helpers ──────────────────────────────────────────────

    def get_thinking_index(self) -> int:
        return self._thinking_id

    def get_thinking_end_index(self) -> int:
        return self._thinking_end_id

    def get_thinking_mode_index(self) -> int:
        return self._thinking_mode_id

    def has_thinking(self, text: str) -> bool:
        return '<thinking>' in text and '</thinking>' in text

    def split_thinking(self, text: str):
        if not self.has_thinking(text):
            return ('', text)
        try:
            thinking = text.split('<thinking>')[1].split('</thinking>')[0]
            response = text.split('</thinking>')[1].strip()
            return (thinking, response)
        except (IndexError, ValueError):
            return ('', text)

    def extract_response(self, text: str) -> str:
        _, response = self.split_thinking(text)
        return response

    # ── Legacy mode token helpers (backward compat) ───────────────

    def get_context_index(self) -> int:
        return self._context_id

    def get_answer_index(self) -> int:
        return self._answer_id

    def has_mode_tokens(self, text: str) -> bool:
        return '<|problem|>' in text or '<|thinking|>' in text or '<|context|>' in text or '<|answer|>' in text

    def split_mode(self, text: str):
        if text.startswith('<|thinking|>'):
            return ('thinking', text[len('<|thinking|>'):].strip())
        elif text.startswith('<|problem|>'):
            return ('problem', text[len('<|problem|>'):].strip())
        elif text.startswith('<|context|>'):
            return ('problem', text[len('<|context|>'):].strip())
        return ('problem', text)

    # ── Agentic token helpers ──────────────────────────────────────

    def get_tool_call_index(self) -> int:
        return self._tool_call_id

    def get_tool_call_end_index(self) -> int:
        return self._tool_call_end_id

    def get_observation_index(self) -> int:
        return self._observation_id

    def get_observation_end_index(self) -> int:
        return self._observation_end_id

    def has_tool_call(self, text: str) -> bool:
        return '<tool_call>' in text and '</tool_call>' in text

    def split_tool_call(self, text: str):
        if not self.has_tool_call(text):
            return (text, '', '')
        try:
            before = text.split('<tool_call>')[0]
            tool_call_json = text.split('<tool_call>')[1].split('</tool_call>')[0]
            after = text.split('</tool_call>')[1]
            return (before, tool_call_json, after)
        except (IndexError, ValueError):
            return (text, '', '')

    def extract_tool_response(self, text: str) -> str:
        _, _, after = self.split_tool_call(text)
        return after.strip()
