"""SentencePiece BPE tokenizer wrapper — multilingual, language-agnostic.

GPT-2 standard special tokens:
  <|problem|>  - question/problem prefix
  <|thinking|> - reasoning prefix
  <|final|>    - answer prefix
  <|user|>     - user prefix (agentic)
  <|assistant|> - assistant prefix (agentic)
  <tool_call> - tool call start
  </tool_call> - tool call end
  <|tool_result|> - tool result prefix
"""

import logging
import torch

try:
    import sentencepiece as spm
except Exception:
    spm = None

logger = logging.getLogger(__name__)


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
            unk = self.sp.unk_id()
            for candidate in candidates:
                try:
                    token_id = self.sp.piece_to_id(candidate)
                    if token_id is not None and token_id >= 0 and token_id != unk:
                        return token_id
                except Exception:
                    continue
            return -1

        pad_id = _resolve_id('<pad>', '<PAD>')
        unk_id = _resolve_id('<unk>', '<UNK>')
        bos_id = _resolve_id('<s>', '<BOS>', '<bos>')
        eos_id = _resolve_id('</s>', '<eos>', '<EOS>')

        # GPT-2 standard tokens
        problem_id = _resolve_id('<|problem|>')
        thinking_mode_id = _resolve_id('<|thinking|>')
        final_id = _resolve_id('<|final|>')
        user_id = _resolve_id('<|user|>')
        assistant_id = _resolve_id('<|assistant|>')
        system_id = _resolve_id('<|system|>')
        end_id = _resolve_id('<|end|>')
        sep_id = _resolve_id('<|sep|>')
        tool_call_id = _resolve_id('<tool_call>')
        tool_call_end_id = _resolve_id('</tool_call>')
        tool_result_id = _resolve_id('<|tool_result|>')

        # Legacy tokens (backward compatibility, deprecated)
        legacy_thinking_id = _resolve_id('<thinking>')
        legacy_thinking_end_id = _resolve_id('</thinking>')
        legacy_context_id = _resolve_id('<|context|>')
        legacy_answer_id = _resolve_id('<|answer|>')
        legacy_observation_id = _resolve_id('<observation>')
        legacy_observation_end_id = _resolve_id('</observation>')

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

        # GPT-2 standard
        self._problem_id = problem_id
        self._thinking_id = thinking_mode_id if thinking_mode_id >= 0 else legacy_thinking_id
        self._thinking_end_id = final_id if final_id >= 0 else legacy_thinking_end_id
        self._thinking_mode_id = thinking_mode_id
        self._final_id = final_id
        self._user_id = user_id
        self._assistant_id = assistant_id
        self._system_id = system_id
        self._end_id = end_id
        self._sep_id = sep_id
        self._tool_call_id = tool_call_id
        self._tool_call_end_id = tool_call_end_id
        self._tool_result_id = tool_result_id

        # Legacy raw IDs kept for decode/skip_special_tokens and backwards compat
        self._legacy_thinking_id = legacy_thinking_id
        self._legacy_thinking_end_id = legacy_thinking_end_id
        self._legacy_context_id = legacy_context_id
        self._legacy_answer_id = legacy_answer_id
        self._legacy_observation_id = legacy_observation_id
        self._legacy_observation_end_id = legacy_observation_end_id

        # Legacy (mapped to GPT-2 standard IDs)
        self._context_id = problem_id if problem_id >= 0 else legacy_context_id
        self._answer_id = final_id if final_id >= 0 else legacy_answer_id
        self._observation_id = tool_result_id if tool_result_id >= 0 else legacy_observation_id
        self._observation_end_id = tool_result_id if tool_result_id >= 0 else legacy_observation_end_id

        # Log real token IDs for debugging
        for name, tid in [
            ('pad', pad_id), ('unk', unk_id), ('bos', bos_id), ('eos', eos_id),
            ('<|problem|>', problem_id), ('<|thinking|>', thinking_mode_id),
            ('<thinking>', legacy_thinking_id), ('</thinking>', legacy_thinking_end_id),
            ('<|final|>', final_id),
            ('<|user|>', user_id), ('<|assistant|>', assistant_id),
            ('<|system|>', system_id), ('<|end|>', end_id), ('<|sep|>', sep_id),
            ('<tool_call>', tool_call_id), ('</tool_call>', tool_call_end_id),
            ('<|tool_result|>', tool_result_id),
        ]:
            status = 'OK' if tid >= 0 else 'NOT FOUND'
            logger.info(f"  Token {name} -> id={tid} ({status})")

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
            special_ids = {id_ for id_ in (
                self._pad_id, self._unk_id, self._bos_id, self._eos_id,
                self._problem_id, self._thinking_id, self._thinking_end_id,
                self._thinking_mode_id, self._final_id,
                self._user_id, self._assistant_id, self._system_id,
                self._end_id, self._sep_id,
                self._tool_call_id, self._tool_call_end_id, self._tool_result_id,
                self._context_id, self._answer_id,
                self._observation_id, self._observation_end_id,
                self._legacy_thinking_id, self._legacy_thinking_end_id,
                self._legacy_context_id, self._legacy_answer_id,
                self._legacy_observation_id, self._legacy_observation_end_id,
            ) if id_ >= 0}
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
        import json
        data = {'sentencepiece_model': self.model_path, 'vocab_size': self.vocab_size}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_as_huggingface(self, output_dir: str):
        """Generate HuggingFace-compatible tokenizer files.

        Creates tokenizer.json, tokenizer_config.json, and special_tokens_map.json
        so the model can be loaded with AutoTokenizer from HuggingFace Transformers.
        """
        import json
        import shutil
        os.makedirs(output_dir, exist_ok=True)

        # Build vocab for HF tokenizer.json
        vocab = {}
        merges = []
        for i in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
            # BPE merges: piece pairs from SentencePiece
            if i >= self.sp.bos_id() + 1 and piece.startswith('▁') is False:
                pass  # SentencePiece handles merges internally

        # Get special token IDs
        pad_id = self.get_pad_index()
        unk_id = self.get_unk_index()
        eos_id = self.get_eos_index()
        bos_id = getattr(self, '_bos_id', -1)

        # Build special_tokens_map.json
        special_tokens = {}
        if eos_id >= 0:
            special_tokens['eos_token'] = {'content': '</s>', 'lstrip': False, 'rstrip': False, 'single_word': False, 'id': eos_id}
        if bos_id >= 0:
            special_tokens['bos_token'] = {'content': '<s>', 'lstrip': False, 'rstrip': False, 'single_word': False, 'id': bos_id}
        if pad_id >= 0:
            special_tokens['pad_token'] = {'content': '<pad>', 'lstrip': False, 'rstrip': False, 'single_word': False, 'id': pad_id}

        # Map agentic tokens
        agentic_map = {
            '<|system|>': self.get_system_index(),
            '<|user|>': self.get_user_index(),
            '<|assistant|>': self.get_assistant_index(),
            '<|end|>': self.get_end_index(),
            '<|sep|>': self.get_sep_index(),
            '<|problem|>': self.get_problem_index(),
            '<|thinking|>': self.get_thinking_mode_index(),
            '<|final|>': self.get_final_index(),
            '<tool_call>': self.get_tool_call_index(),
            '</tool_call>': self.get_tool_call_end_index(),
            '<|tool_result|>': self.get_tool_result_index(),
            '<thinking>': self.get_thinking_index(),
            '</thinking>': self.get_thinking_end_index(),
        }
        for token_str, token_id in agentic_map.items():
            if token_id >= 0:
                safe_name = token_str.strip('<>').replace('|', '').replace('/', '_')
                special_tokens[f'{safe_name}_token'] = {
                    'content': token_str,
                    'lstrip': False,
                    'rstrip': False,
                    'single_word': False,
                    'id': token_id
                }

        with open(os.path.join(output_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
            json.dump(special_tokens, f, indent=2, ensure_ascii=False)

        # Build tokenizer_config.json
        tokenizer_config = {
            'bos_token': '<s>' if bos_id >= 0 else None,
            'eos_token': '</s>' if eos_id >= 0 else None,
            'pad_token': '<pad>' if pad_id >= 0 else '<unk>',
            'unk_token': '<unk>',
            'model_max_length': 512,
            'tokenizer_class': 'PreTrainedTokenizerFast',
            'special_tokens': list(special_tokens.keys()),
        }
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        # Build tokenizer.json (simplified HF format)
        tokenizer_json = {
            'version': '1.0',
            'truncation': None,
            'padding': None,
            'added_tokens': [
                {
                    'id': tid,
                    'content': tname,
                    'single_word': False,
                    'lstrip': False,
                    'rstrip': False,
                    'special': True
                }
                for tname, info in special_tokens.items()
                if 'id' in info and info['id'] >= 0
            ],
            'model': {
                'type': 'BPE',
                'vocab': vocab,
                'merges': merges
            }
        }
        with open(os.path.join(output_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)

        # Copy the SentencePiece model file
        if self.model_path and os.path.exists(self.model_path):
            shutil.copy2(self.model_path, os.path.join(output_dir, 'sentencepiece.model'))

        logger.info(f"HuggingFace tokenizer files saved to: {output_dir}")

    # ── GPT-2 Standard token helpers ───────────────────────────────

    def get_problem_index(self) -> int:
        """Return the token ID for <|problem|>."""
        self._ensure_special_token_ids()
        return self._problem_id

    def get_final_index(self) -> int:
        """Return the token ID for <|final|>."""
        self._ensure_special_token_ids()
        return self._final_id

    def get_user_index(self) -> int:
        """Return the token ID for <|user|>."""
        self._ensure_special_token_ids()
        return self._user_id

    def get_assistant_index(self) -> int:
        """Return the token ID for <|assistant|>."""
        self._ensure_special_token_ids()
        return self._assistant_id

    def get_system_index(self) -> int:
        """Return the token ID for <|system|>."""
        self._ensure_special_token_ids()
        return self._system_id

    def get_end_index(self) -> int:
        """Return the token ID for <|end|>."""
        self._ensure_special_token_ids()
        return self._end_id

    def get_sep_index(self) -> int:
        """Return the token ID for <|sep|>."""
        self._ensure_special_token_ids()
        return self._sep_id

    def get_tool_result_index(self) -> int:
        """Return the token ID for <|tool_result|>."""
        self._ensure_special_token_ids()
        return self._tool_result_id

    # ── Thinking helpers ──────────────────────────────────────────────

    def get_thinking_index(self) -> int:
        """Return the token ID for <thinking>."""
        self._ensure_special_token_ids()
        return self._thinking_id

    def get_thinking_end_index(self) -> int:
        """Return the token ID for </thinking>."""
        self._ensure_special_token_ids()
        return self._thinking_end_id

    def get_thinking_mode_index(self) -> int:
        """Return the token ID for <|thinking|>."""
        self._ensure_special_token_ids()
        return self._thinking_mode_id

    def has_thinking(self, text: str) -> bool:
        """Check if text contains thinking tags (new <|thinking|> or legacy <thinking>)."""
        return ('<|thinking|>' in text and '<|final|>' in text) or ('<thinking>' in text and '</thinking>' in text)

    def split_thinking(self, text: str):
        """Split text into (thinking, response) parts.

        Supports new format (<|thinking|>...</|final|>) and legacy (<thinking>...</thinking>).

        Returns:
            Tuple[str, str]: (thinking_content, response_text)
            If no thinking tags found, returns ('', text).
        """
        if not self.has_thinking(text):
            return ('', text)
        try:
            if '<|thinking|>' in text:
                thinking = text.split('<|thinking|>')[1].split('<|final|>')[0]
                response = text.split('<|final|>')[1].strip()
            else:
                thinking = text.split('<thinking>')[1].split('</thinking>')[0]
                response = text.split('</thinking>')[1].strip()
            return (thinking, response)
        except (IndexError, ValueError):
            return ('', text)

    def extract_response(self, text: str) -> str:
        """Extract only the response part, removing thinking blocks."""
        _, response = self.split_thinking(text)
        return response

    # ── Legacy mode token helpers (backward compat) ───────────────

    def get_context_index(self) -> int:
        """Return the token ID for <|context|> (legacy, maps to <|problem|>)."""
        self._ensure_special_token_ids()
        return self._context_id

    def get_answer_index(self) -> int:
        """Return the token ID for <|answer|> (legacy, maps to <|final|>)."""
        self._ensure_special_token_ids()
        return self._answer_id

    def has_mode_tokens(self, text: str) -> bool:
        """Check if text contains mode tokens."""
        return '<|problem|>' in text or '<|thinking|>' in text or '<|context|>' in text or '<|answer|>' in text

    def split_mode(self, text: str):
        """Split text into (mode, content) where mode is 'problem' or 'thinking'."""
        if text.startswith('<|thinking|>'):
            return ('thinking', text[len('<|thinking|>'):].strip())
        elif text.startswith('<|problem|>'):
            return ('problem', text[len('<|problem|>'):].strip())
        elif text.startswith('<|context|>'):
            return ('problem', text[len('<|context|>'):].strip())
        return ('problem', text)

    # ── Agentic token helpers ──────────────────────────────────────

    def get_tool_call_index(self) -> int:
        """Return the token ID for <tool_call>."""
        self._ensure_special_token_ids()
        return self._tool_call_id

    def get_tool_call_end_index(self) -> int:
        """Return the token ID for </tool_call>."""
        self._ensure_special_token_ids()
        return self._tool_call_end_id

    def get_observation_index(self) -> int:
        """Return the token ID for <observation> (legacy, maps to <|tool_result|>)."""
        self._ensure_special_token_ids()
        return self._observation_id

    def get_observation_end_index(self) -> int:
        """Return the token ID for </observation> (legacy, maps to <|tool_result|>)."""
        self._ensure_special_token_ids()
        return self._observation_end_id

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains <tool_call> tags."""
        return '<tool_call>' in text and '</tool_call>' in text

    def split_tool_call(self, text: str):
        """Split text into (before_tool_call, tool_call_json, after_tool_call).

        Returns:
            Tuple[str, str, str]: (prefix, tool_call_json, suffix)
            If no tool_call found, returns (text, '', '').
        """
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
        """Extract only the tool response (after </tool_call>)."""
        _, _, after = self.split_tool_call(text)
        return after.strip()
