"""SentencePiece BPE tokenizer wrapper — multilingual, language-agnostic."""

import torch

try:
    import sentencepiece as spm
except Exception:
    spm = None


class SentencePieceTokenizerWrapper:
    """Wrapper around SentencePiece BPE model — multilingual, language-agnostic.

    Supports: encode, decode, batch_encode, get_pad_index, get_unk_index,
    get_eos_index, convert_ids_to_tokens, fit (no-op), save_vocabulary.
    Also exposes vocab and idx2word dicts for DialogueManager compatibility.
    """
    def __init__(self, model_path: str):
        if spm is None:
            raise ImportError("sentencepiece package is required for BPE tokenization")
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._initialize_special_token_ids()
        self._build_vocab_dicts()

    def _initialize_special_token_ids(self):
        def _resolve_id(*candidates):
            for candidate in candidates:
                try:
                    token_id = self.sp.piece_to_id(candidate)
                    if token_id is not None and token_id >= 0:
                        return token_id
                except Exception:
                    continue
            return -1

        pad_id = _resolve_id('<pad>', '<PAD>')
        unk_id = _resolve_id('<unk>', '<UNK>')
        bos_id = _resolve_id('<s>', '<BOS>', '<bos>')
        eos_id = _resolve_id('</s>', '<eos>', '<EOS>')
        thinking_id = _resolve_id('<thinking>')
        thinking_end_id = _resolve_id('</thinking>')

        if pad_id < 0:
            pad_id = 0
        if unk_id < 0:
            unk_id = 1
        if bos_id < 0:
            bos_id = -1
        if eos_id < 0:
            eos_id = -1

        self._pad_id = pad_id
        self._unk_id = unk_id
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._thinking_id = thinking_id
        self._thinking_end_id = thinking_end_id

    def _build_vocab_dicts(self):
        self._idx2word = {}
        for i in range(self.sp.get_piece_size()):
            self._idx2word[i] = self.sp.id_to_piece(i)
        self._word2idx = {v: k for k, v in self._idx2word.items()}

    def _ensure_special_token_ids(self):
        if not hasattr(self, '_pad_id') or not hasattr(self, '_unk_id') or not hasattr(self, '_eos_id'):
            self._initialize_special_token_ids()

    def encode(self, text: str, *args, **kwargs):
        return list(self.sp.encode(text, out_type=int))

    def batch_encode(self, texts, *args, **kwargs):
        return [list(self.sp.encode(t, out_type=int)) for t in texts]

    def get_pad_index(self):
        self._ensure_special_token_ids()
        return self._pad_id

    def get_unk_index(self):
        self._ensure_special_token_ids()
        return self._unk_id

    def get_eos_index(self):
        self._ensure_special_token_ids()
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

        self._ensure_special_token_ids()
        token_ids = [int(i) for i in indices if i is not None]
        if skip_special_tokens:
            special_ids = {id_ for id_ in (self._pad_id, self._unk_id, self._bos_id, self._eos_id) if id_ >= 0}
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        if not token_ids:
            return ""

        if hasattr(self.sp, "decode_ids"):
            return self.sp.decode_ids(token_ids)
        if hasattr(self.sp, "decode"):
            return self.sp.decode(token_ids)
        return " ".join(str(t) for t in token_ids)

    def convert_ids_to_tokens(self, ids):
        if ids is None:
            return []
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        return [self.sp.id_to_piece(int(i)) for i in ids]

    def fit(self, texts):
        # No-op: model already trained
        return

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def vocab(self):
        """Word-to-index mapping dict for DialogueManager compatibility."""
        return self._word2idx

    @property
    def idx2word(self):
        """Index-to-word mapping dict for DialogueManager compatibility."""
        return self._idx2word

    def save_vocabulary(self, filepath: str):
        # Save small metadata pointing to sentencepiece model
        import json
        data = {'sentencepiece_model': self.model_path, 'vocab_size': self.vocab_size}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Thinking helpers ──────────────────────────────────────────────

    def get_thinking_index(self) -> int:
        """Return the token ID for <thinking>."""
        self._ensure_special_token_ids()
        return self._thinking_id

    def get_thinking_end_index(self) -> int:
        """Return the token ID for </thinking>."""
        self._ensure_special_token_ids()
        return self._thinking_end_id

    def has_thinking(self, text: str) -> bool:
        """Check if text contains <thinking> tags."""
        return '<thinking>' in text and '</thinking>' in text

    def split_thinking(self, text: str):
        """Split text into (thinking, response) parts.

        Returns:
            Tuple[str, str]: (thinking_content, response_text)
            If no thinking tags found, returns ('', text).
        """
        if not self.has_thinking(text):
            return ('', text)
        try:
            thinking = text.split('<thinking>')[1].split('</thinking>')[0]
            response = text.split('</thinking>')[1].strip()
            return (thinking, response)
        except (IndexError, ValueError):
            return ('', text)

    def extract_response(self, text: str) -> str:
        """Extract only the response part, removing <thinking> blocks."""
        _, response = self.split_thinking(text)
        return response

    def encode_with_thinking(self, text: str):
        """Encode text, returning (thinking_ids, response_ids, all_ids).

        Useful for training where you want to separate thinking from response.
        """
        thinking, response = self.split_thinking(text)
        all_ids = self.encode(text)
        thinking_ids = self.encode(thinking) if thinking else []
        response_ids = self.encode(response) if response else []
        return (thinking_ids, response_ids, all_ids)
