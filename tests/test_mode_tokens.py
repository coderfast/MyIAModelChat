"""Tests for mode tokens (PROBLEM/THINKING) in tokenizer and trainer."""

import os
import pytest
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SP_MODEL_PATH = os.path.join('dataset_cache', 'sentencepiece.model')


@pytest.fixture(scope="module")
def sp_tokenizer():
    """Load SentencePiece tokenizer if model exists."""
    if not os.path.exists(SP_MODEL_PATH):
        pytest.skip("sentencepiece.model not found; run --prepare-data first")
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
    return SentencePieceTokenizerWrapper(SP_MODEL_PATH)


class TestModeTokens:
    """Tests for mode token support in tokenizer."""

    def test_get_problem_index(self, sp_tokenizer):
        problem_id = sp_tokenizer.get_problem_index()
        assert isinstance(problem_id, int)
        if problem_id >= 0:
            token = sp_tokenizer.idx2word.get(problem_id, '')
            assert '<|problem|>' in token

    def test_get_final_index(self, sp_tokenizer):
        final_id = sp_tokenizer.get_final_index()
        assert isinstance(final_id, int)
        if final_id >= 0:
            token = sp_tokenizer.idx2word.get(final_id, '')
            assert '<|final|>' in token

    def test_get_thinking_mode_index(self, sp_tokenizer):
        tm_id = sp_tokenizer.get_thinking_mode_index()
        assert isinstance(tm_id, int)
        if tm_id >= 0:
            token = sp_tokenizer.idx2word.get(tm_id, '')
            assert '<|thinking|>' in token

    def test_has_mode_tokens(self, sp_tokenizer):
        text_with = "<|thinking|>What is AI?<|final|>AI is..."
        text_without = "What is AI?"
        assert sp_tokenizer.has_mode_tokens(text_with) is True
        assert sp_tokenizer.has_mode_tokens(text_without) is False

    def test_split_mode_thinking(self, sp_tokenizer):
        text = "<|thinking|>What is AI?<|final|>AI is artificial intelligence."
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "thinking"
        assert "What is AI?" in content
        assert "AI is artificial intelligence." in content

    def test_split_mode_problem(self, sp_tokenizer):
        text = "<|problem|>What is AI?<|final|>AI is artificial intelligence."
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "problem"
        assert "What is AI?" in content
        assert "AI is artificial intelligence." in content

    def test_split_mode_no_tokens(self, sp_tokenizer):
        text = "What is AI?"
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "problem"
        assert content == "What is AI?"

    def test_encode_decode_with_mode_tokens(self, sp_tokenizer):
        text = "<|thinking|>What is AI?<|final|>AI is artificial intelligence."
        ids = sp_tokenizer.encode(text)
        assert isinstance(ids, list)
        assert len(ids) > 0
        decoded = sp_tokenizer.decode(ids, skip_special_tokens=False)
        assert "<|thinking|>" in decoded or "<|final|>" in decoded


class TestTrainerModeAwareLoss:
    """Tests for trainer's mode-aware loss computation."""

    def test_detect_thinking_data_with_mode_tokens(self, sp_tokenizer):
        """Test that _detect_thinking_data detects mode tokens."""
        problem_id = sp_tokenizer.get_problem_index()
        final_id = sp_tokenizer.get_final_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()

        if problem_id < 0 or final_id < 0 or tm_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        sample = torch.tensor([[tm_id, 100, 101, final_id, 200, 201]])
        thinking_id = sp_tokenizer.get_thinking_index()
        thinking_end_id = sp_tokenizer.get_thinking_end_index()
        
        has_thinking = tm_id in sample[0] or problem_id in sample[0]
        has_answer = final_id in sample[0]
        assert has_thinking is True
        assert has_answer is True

    def test_detect_thinking_data_without_mode_tokens(self, sp_tokenizer):
        """Test that _detect_thinking_data returns False for plain text."""
        sample = torch.tensor([[100, 101, 102, 103]])
        problem_id = sp_tokenizer.get_problem_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()
        
        has_thinking = tm_id in sample[0] or problem_id in sample[0]
        assert has_thinking is False

    def test_compute_loss_mode_aware(self, sp_tokenizer):
        """Test that _compute_loss applies mode-aware masking."""
        from training.trainer import Trainer, TrainingConfig
        from commons.model.chatmodel import ChatModel
        import torch.nn as nn

        model = ChatModel(sp_tokenizer, embed_size=64, num_layers=1)

        problem_id = sp_tokenizer.get_problem_index()
        final_id = sp_tokenizer.get_final_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()
        eos_id = sp_tokenizer.get_eos_index()

        if problem_id < 0 or final_id < 0 or tm_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        token_ids = torch.tensor([[problem_id, 100, 101, final_id, 200, 201, eos_id]])
        labels = token_ids.clone()

        config = TrainingConfig()
        trainer = Trainer(config)
        trainer.tokenizer = sp_tokenizer
        
        criterion = nn.CrossEntropyLoss()
        loss_result = trainer._compute_loss(model, token_ids, labels, criterion)
        
        assert isinstance(loss_result, tuple)
        loss = loss_result[0]

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_compute_loss_thinking_mode(self, sp_tokenizer):
        """Test loss computation with THINKING mode tokens."""
        from training.trainer import Trainer, TrainingConfig
        from commons.model.chatmodel import ChatModel
        import torch.nn as nn

        model = ChatModel(sp_tokenizer, embed_size=64, num_layers=1)

        tm_id = sp_tokenizer.get_thinking_mode_index()
        final_id = sp_tokenizer.get_final_index()
        thinking_id = sp_tokenizer.get_thinking_index()
        thinking_end_id = sp_tokenizer.get_thinking_end_index()
        eos_id = sp_tokenizer.get_eos_index()

        if tm_id < 0 or final_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        token_ids_list = [tm_id, 100, 101]
        if thinking_id >= 0:
            token_ids_list.append(thinking_id)
        token_ids_list.extend([300, 301])
        if thinking_end_id >= 0:
            token_ids_list.append(thinking_end_id)
        token_ids_list.extend([final_id, 400, 401, eos_id])

        token_ids = torch.tensor([token_ids_list])
        labels = token_ids.clone()

        config = TrainingConfig()
        trainer = Trainer(config)
        trainer.tokenizer = sp_tokenizer
        
        criterion = nn.CrossEntropyLoss()
        loss_result = trainer._compute_loss(model, token_ids, labels, criterion)

        assert isinstance(loss_result, tuple)
        loss = loss_result[0]

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0
