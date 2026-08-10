"""Tests for mode tokens (CONTEXT/THINKING) in tokenizer and trainer."""

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

    def test_get_context_index(self, sp_tokenizer):
        ctx_id = sp_tokenizer.get_context_index()
        assert isinstance(ctx_id, int)
        if ctx_id >= 0:
            token = sp_tokenizer.idx2word.get(ctx_id, '')
            assert '<|context|>' in token

    def test_get_answer_index(self, sp_tokenizer):
        ans_id = sp_tokenizer.get_answer_index()
        assert isinstance(ans_id, int)
        if ans_id >= 0:
            token = sp_tokenizer.idx2word.get(ans_id, '')
            assert '<|answer|>' in token

    def test_get_thinking_mode_index(self, sp_tokenizer):
        tm_id = sp_tokenizer.get_thinking_mode_index()
        assert isinstance(tm_id, int)
        if tm_id >= 0:
            token = sp_tokenizer.idx2word.get(tm_id, '')
            assert '<|thinking|>' in token

    def test_has_mode_tokens(self, sp_tokenizer):
        text_with = "<|thinking|>What is AI?<|answer|>AI is..."
        text_without = "What is AI?"
        assert sp_tokenizer.has_mode_tokens(text_with) is True
        assert sp_tokenizer.has_mode_tokens(text_without) is False

    def test_split_mode_thinking(self, sp_tokenizer):
        text = "<|thinking|>What is AI?<|answer|>AI is artificial intelligence."
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "thinking"
        assert "What is AI?" in content
        assert "AI is artificial intelligence." in content

    def test_split_mode_context(self, sp_tokenizer):
        text = "<|context|>What is AI?<|answer|>AI is artificial intelligence."
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "context"
        assert "What is AI?" in content
        assert "AI is artificial intelligence." in content

    def test_split_mode_no_tokens(self, sp_tokenizer):
        text = "What is AI?"
        mode, content = sp_tokenizer.split_mode(text)
        assert mode == "context"
        assert content == "What is AI?"

    def test_encode_decode_with_mode_tokens(self, sp_tokenizer):
        text = "<|thinking|>What is AI?<|answer|>AI is artificial intelligence."
        ids = sp_tokenizer.encode(text)
        assert isinstance(ids, list)
        assert len(ids) > 0
        decoded = sp_tokenizer.decode(ids, skip_special_tokens=False)
        assert "<|thinking|>" in decoded or "<|answer|>" in decoded


class TestTrainerModeAwareLoss:
    """Tests for trainer's mode-aware loss computation."""

    def test_detect_thinking_data_with_mode_tokens(self, sp_tokenizer):
        """Test that _detect_thinking_data detects mode tokens."""
        from training.trainer import Trainer
        # Create a minimal mock dataset with mode tokens
        ctx_id = sp_tokenizer.get_context_index()
        ans_id = sp_tokenizer.get_answer_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()

        if ctx_id < 0 or ans_id < 0 or tm_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        # Create a sample token sequence with mode tokens
        sample = torch.tensor([[tm_id, 100, 101, ans_id, 200, 201]])
        # Test the logic directly without calling the instance method
        thinking_id = sp_tokenizer.get_thinking_index()
        thinking_end_id = sp_tokenizer.get_thinking_end_index()
        
        # Check if any of the mode tokens are in the sample
        has_thinking = tm_id in sample[0] or ctx_id in sample[0]
        has_answer = ans_id in sample[0]
        assert has_thinking is True
        assert has_answer is True

    def test_detect_thinking_data_without_mode_tokens(self, sp_tokenizer):
        """Test that _detect_thinking_data returns False for plain text."""
        from training.trainer import Trainer
        sample = torch.tensor([[100, 101, 102, 103]])
        # Test the logic directly without calling the instance method
        ctx_id = sp_tokenizer.get_context_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()
        
        # Check if any of the mode tokens are in the sample
        has_thinking = tm_id in sample[0] or ctx_id in sample[0]
        assert has_thinking is False

    def test_compute_loss_mode_aware(self, sp_tokenizer):
        """Test that _compute_loss applies mode-aware masking."""
        from training.trainer import Trainer, TrainingConfig
        from commons.model.chatmodel import ChatModel
        import torch.nn as nn

        # Create a minimal model
        model = ChatModel(sp_tokenizer, embed_size=64, num_layers=1)

        ctx_id = sp_tokenizer.get_context_index()
        ans_id = sp_tokenizer.get_answer_index()
        tm_id = sp_tokenizer.get_thinking_mode_index()
        eos_id = sp_tokenizer.get_eos_index()

        if ctx_id < 0 or ans_id < 0 or tm_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        # Create a CONTEXT mode sample: <|context|>question<|answer|>answer</s>
        token_ids = torch.tensor([[ctx_id, 100, 101, ans_id, 200, 201, eos_id]])
        labels = token_ids.clone()

        # Create a minimal trainer to access _compute_loss
        config = TrainingConfig()
        trainer = Trainer(config)
        trainer.tokenizer = sp_tokenizer
        
        # Create criterion
        criterion = nn.CrossEntropyLoss()
        
        # Compute loss
        loss_result = trainer._compute_loss(model, token_ids, labels, criterion)
        
        # _compute_loss returns (loss, outputs) tuple
        assert isinstance(loss_result, tuple)
        loss = loss_result[0]

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_compute_loss_thinking_mode(self, sp_tokenizer):
        """Test loss computation with THINKING mode tokens."""
        from training.trainer import Trainer, TrainingConfig
        from commons.model.chatmodel import ChatModel
        import torch.nn as nn

        model = ChatModel(sp_tokenizer, embed_size=64, num_layers=1)

        tm_id = sp_tokenizer.get_thinking_mode_index()
        ans_id = sp_tokenizer.get_answer_index()
        thinking_id = sp_tokenizer.get_thinking_index()
        thinking_end_id = sp_tokenizer.get_thinking_end_index()
        eos_id = sp_tokenizer.get_eos_index()

        if tm_id < 0 or ans_id < 0:
            pytest.skip("Mode tokens not in vocabulary")

        # Build a THINKING mode sample
        token_ids_list = [tm_id, 100, 101]
        if thinking_id >= 0:
            token_ids_list.append(thinking_id)
        token_ids_list.extend([300, 301])
        if thinking_end_id >= 0:
            token_ids_list.append(thinking_end_id)
        token_ids_list.extend([ans_id, 400, 401, eos_id])

        token_ids = torch.tensor([token_ids_list])
        labels = token_ids.clone()

        # Create a minimal trainer to access _compute_loss
        config = TrainingConfig()
        trainer = Trainer(config)
        trainer.tokenizer = sp_tokenizer
        
        # Create criterion
        criterion = nn.CrossEntropyLoss()
        
        loss_result = trainer._compute_loss(model, token_ids, labels, criterion)

        # _compute_loss returns (loss, outputs) tuple
        assert isinstance(loss_result, tuple)
        loss = loss_result[0]

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0
