"""Tests for SentencePiece BPE tokenizer wrapper."""

import os
import pytest

SP_MODEL_PATH = os.path.join('dataset_cache', 'sentencepiece.model')


@pytest.fixture(scope="module")
def sp_tokenizer():
    """Load SentencePiece tokenizer if model exists."""
    if not os.path.exists(SP_MODEL_PATH):
        pytest.skip("sentencepiece.model not found; run --prepare-data first")
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
    return SentencePieceTokenizerWrapper(SP_MODEL_PATH)


class TestSentencePieceTokenizerWrapper:
    def test_load_model(self, sp_tokenizer):
        assert sp_tokenizer.sp is not None
        assert sp_tokenizer.vocab_size > 0

    def test_expected_vocab_size(self, sp_tokenizer):
        assert sp_tokenizer.vocab_size == 8000

    def test_encode_returns_list_of_int(self, sp_tokenizer):
        result = sp_tokenizer.encode("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)
        assert len(result) > 0

    def test_encode_non_english(self, sp_tokenizer):
        result = sp_tokenizer.encode("Hola, como estas?")  # Spanish — any language works
        assert isinstance(result, list)
        assert len(result) > 0

    def test_decode_returns_string(self, sp_tokenizer):
        ids = sp_tokenizer.encode("Hello world")
        text = sp_tokenizer.decode(ids)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_encode_decode_roundtrip(self, sp_tokenizer):
        original = "hello world"
        ids = sp_tokenizer.encode(original)
        decoded = sp_tokenizer.decode(ids)
        assert decoded == original

    def test_encode_decode_roundtrip_non_english(self, sp_tokenizer):
        original = "hola mundo"  # Spanish — any language works
        ids = sp_tokenizer.encode(original)
        decoded = sp_tokenizer.decode(ids)
        assert decoded == original

    def test_batch_encode(self, sp_tokenizer):
        texts = ["Hello", "Hola mundo", "test"]
        results = sp_tokenizer.batch_encode(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_get_pad_index(self, sp_tokenizer):
        pad_id = sp_tokenizer.get_pad_index()
        assert isinstance(pad_id, int)
        assert pad_id >= 0

    def test_get_unk_index(self, sp_tokenizer):
        unk_id = sp_tokenizer.get_unk_index()
        assert isinstance(unk_id, int)
        assert unk_id >= 0

    def test_get_eos_index(self, sp_tokenizer):
        eos_id = sp_tokenizer.get_eos_index()
        assert isinstance(eos_id, int)

    def test_convert_ids_to_tokens(self, sp_tokenizer):
        ids = sp_tokenizer.encode("Hello")
        tokens = sp_tokenizer.convert_ids_to_tokens(ids)
        assert isinstance(tokens, list)
        assert len(tokens) == len(ids)
        assert all(isinstance(t, str) for t in tokens)

    def test_vocab_dict(self, sp_tokenizer):
        assert hasattr(sp_tokenizer, 'vocab')
        assert isinstance(sp_tokenizer.vocab, dict)
        assert len(sp_tokenizer.vocab) > 0

    def test_idx2word_dict(self, sp_tokenizer):
        assert hasattr(sp_tokenizer, 'idx2word')
        assert isinstance(sp_tokenizer.idx2word, dict)
        assert len(sp_tokenizer.idx2word) > 0

    def test_save_vocabulary(self, sp_tokenizer, tmp_path):
        outfile = str(tmp_path / "test_vocab.json")
        sp_tokenizer.save_vocabulary(outfile)
        assert os.path.exists(outfile)
        import json
        with open(outfile, 'r') as f:
            data = json.load(f)
        assert 'sentencepiece_model' in data
        assert 'vocab_size' in data
        assert data['vocab_size'] == sp_tokenizer.vocab_size

    # ── Thinking token tests ──────────────────────────────────────────

    def test_has_thinking(self, sp_tokenizer):
        assert sp_tokenizer.has_thinking("<think>pensamiento</think>respuesta") is True
        assert sp_tokenizer.has_thinking("respuesta sin thinking") is False
        assert sp_tokenizer.has_thinking("<think>solo apertura") is False

    def test_split_thinking(self, sp_tokenizer):
        thinking, response = sp_tokenizer.split_thinking("<think>razón</think>respuesta final")
        assert thinking == "razón"
        assert response == "respuesta final"

    def test_split_thinking_no_tags(self, sp_tokenizer):
        thinking, response = sp_tokenizer.split_thinking("respuesta normal")
        assert thinking == ""
        assert response == "respuesta normal"

    def test_split_thinking_multiline(self, sp_tokenizer):
        text = "<think>paso 1\npaso 2</think>la respuesta"
        thinking, response = sp_tokenizer.split_thinking(text)
        assert "paso 1" in thinking
        assert "paso 2" in thinking
        assert response == "la respuesta"

    def test_extract_response(self, sp_tokenizer):
        result = sp_tokenizer.extract_response("<think>pensamiento</think>respuesta limpia")
        assert result == "respuesta limpia"
        assert "pensamiento" not in result

    def test_extract_response_no_thinking(self, sp_tokenizer):
        result = sp_tokenizer.extract_response("respuesta directa")
        assert result == "respuesta directa"

    def test_encode_decode_with_thinking(self, sp_tokenizer):
        text = "<think>pensamiento breve</think>hola mundo"
        ids = sp_tokenizer.encode(text)
        assert isinstance(ids, list)
        decoded = sp_tokenizer.decode(ids, skip_special_tokens=False)
        assert "<think>" in decoded
        assert "</think>" in decoded
        assert "hola mundo" in decoded

    def test_get_thinking_index(self, sp_tokenizer):
        thinking_id = sp_tokenizer.get_thinking_index()
        assert isinstance(thinking_id, int)
        # Should be >= 0 if the token exists in vocab
        if thinking_id >= 0:
            token = sp_tokenizer.idx2word.get(thinking_id, '')
            assert 'think' in token.lower() or '<think>' in token

    def test_get_thinking_end_index(self, sp_tokenizer):
        thinking_end_id = sp_tokenizer.get_thinking_end_index()
        assert isinstance(thinking_end_id, int)
        if thinking_end_id >= 0:
            token = sp_tokenizer.idx2word.get(thinking_end_id, '')
            assert 'think' in token.lower() or '</think>' in token
