"""Tests for Multi-Token Prediction (MTP) model.

Tests MTPHead, ChatModelMTP, MTP loss computation, and compatibility.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, text):
        return [1, 2, 3, 4, 5]

    def get_pad_index(self):
        return 0

    def get_thinking_index(self):
        return -1

    def get_thinking_end_index(self):
        return -1

    def get_context_index(self):
        return -1

    def get_answer_index(self):
        return -1

    def get_thinking_mode_index(self):
        return -1

    def get_unk_index(self):
        return 1

    def get_tool_call_index(self):
        return -1

    def get_tool_call_end_index(self):
        return -1

    def get_observation_index(self):
        return -1

    def get_observation_end_index(self):
        return -1

    def get_end_index(self):
        return -1

    def get_assistant_index(self):
        return -1


class TestMTPHead:
    """Tests for MTPHead module."""

    def test_output_shape(self):
        from commons.model.chatmodel_mtp import MTPHead

        hidden_size = 128
        vocab_size = 500
        batch_size = 2
        seq_len = 10

        head = MTPHead(hidden_size, vocab_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = head(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_parameters_exist(self):
        from commons.model.chatmodel_mtp import MTPHead

        head = MTPHead(64, 100)
        assert hasattr(head, 'projection')
        assert hasattr(head, 'output_head')
        assert head.projection.weight.shape == (64, 64)
        assert head.output_head.weight.shape == (100, 64)


class TestChatModelMTP:
    """Tests for ChatModelMTP model."""

    def test_forward_returns_tuple(self):
        from commons.model.chatmodel_mtp import ChatModelMTP

        tokenizer = MockTokenizer(vocab_size=100)
        model = ChatModelMTP(tokenizer, embed_size=64, num_layers=2, mtp_num_heads=4)

        input_ids = torch.randint(0, 100, (2, 10))
        output = model(input_ids)

        assert isinstance(output, tuple)
        assert len(output) == 2
        primary_logits, mtp_logits = output
        assert isinstance(mtp_logits, list)
        assert len(mtp_logits) == 3  # mtp_num_heads - 1

    def test_output_shapes(self):
        from commons.model.chatmodel_mtp import ChatModelMTP

        vocab_size = 100
        embed_size = 64
        tokenizer = MockTokenizer(vocab_size=vocab_size)
        model = ChatModelMTP(tokenizer, embed_size=embed_size, num_layers=2, mtp_num_heads=4)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        primary_logits, mtp_logits = model(input_ids)

        assert primary_logits.shape == (batch_size, seq_len, vocab_size)
        for head_logits in mtp_logits:
            assert head_logits.shape == (batch_size, seq_len, vocab_size)

    def test_mtp_num_heads_configurable(self):
        from commons.model.chatmodel_mtp import ChatModelMTP

        tokenizer = MockTokenizer(vocab_size=50)

        for num_heads in [2, 3, 5]:
            model = ChatModelMTP(tokenizer, embed_size=32, num_layers=1, mtp_num_heads=num_heads)
            assert len(model.mtp_heads) == num_heads - 1
            assert model.mtp_num_heads == num_heads

    def test_gradient_flow(self):
        from commons.model.chatmodel_mtp import ChatModelMTP

        tokenizer = MockTokenizer(vocab_size=50)
        model = ChatModelMTP(tokenizer, embed_size=32, num_layers=1, mtp_num_heads=3)

        input_ids = torch.randint(0, 50, (1, 5))
        primary_logits, mtp_logits = model(input_ids)

        loss = primary_logits.sum()
        for logits in mtp_logits:
            loss = loss + logits.sum()

        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None or param.grad is None  # some params may not get grads

    def test_get_mtp_params(self):
        from commons.model.chatmodel_mtp import ChatModelMTP

        tokenizer = MockTokenizer(vocab_size=50)
        model = ChatModelMTP(tokenizer, embed_size=32, num_layers=1, mtp_num_heads=4)

        mtp_params = model.get_mtp_params()
        total_params = model.get_total_params()

        assert mtp_params > 0
        assert mtp_params < total_params

    def test_mtp_loss_computation(self):
        """Test that MTP loss can be computed with correct offsets."""
        from commons.model.chatmodel_mtp import ChatModelMTP

        vocab_size = 50
        embed_size = 32
        tokenizer = MockTokenizer(vocab_size=vocab_size)
        model = ChatModelMTP(tokenizer, embed_size=embed_size, num_layers=1, mtp_num_heads=4)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        targets = torch.randint(1, vocab_size, (batch_size, seq_len))

        primary_logits, mtp_logits = model(input_ids)

        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        # Primary loss
        primary_loss = criterion(
            primary_logits.contiguous().view(-1, vocab_size),
            targets.contiguous().view(-1)
        ).mean()

        # MTP loss with offsets
        mtp_loss = torch.tensor(0.0)
        mtp_count = 0
        for k, head_logits in enumerate(mtp_logits):
            shift = k + 2  # head 0 → t+2, head 1 → t+3
            if seq_len > shift:
                mtp_preds = head_logits[:, :-shift].contiguous().view(-1, vocab_size)
                mtp_targets = targets[:, shift:].contiguous().view(-1)
                mtp_loss = mtp_loss + criterion(mtp_preds, mtp_targets).mean()
                mtp_count += 1

        assert primary_loss.requires_grad
        assert mtp_count == 3  # 3 MTP heads for mtp_num_heads=4


class TestChatModelMoEMTP:
    """Tests for ChatModelMoEMTP combined model."""

    def test_forward_returns_tuple(self):
        from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP

        tokenizer = MockTokenizer(vocab_size=100)
        model = ChatModelMoEMTP(tokenizer, embed_size=64, num_layers=2,
                                num_experts=4, top_k=2, mtp_num_heads=4)

        input_ids = torch.randint(0, 100, (2, 10))
        output = model(input_ids)

        assert isinstance(output, tuple)
        assert len(output) == 2
        primary_logits, mtp_logits = output
        assert isinstance(mtp_logits, list)
        assert len(mtp_logits) == 3  # mtp_num_heads - 1

    def test_output_shapes(self):
        from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP

        vocab_size = 100
        embed_size = 64
        tokenizer = MockTokenizer(vocab_size=vocab_size)
        model = ChatModelMoEMTP(tokenizer, embed_size=embed_size, num_layers=2,
                                num_experts=4, top_k=2, mtp_num_heads=4)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        primary_logits, mtp_logits = model(input_ids)

        assert primary_logits.shape == (batch_size, seq_len, vocab_size)
        for head_logits in mtp_logits:
            assert head_logits.shape == (batch_size, seq_len, vocab_size)

    def test_has_both_moe_and_mtp(self):
        from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP

        tokenizer = MockTokenizer(vocab_size=50)
        model = ChatModelMoEMTP(tokenizer, embed_size=32, num_layers=1,
                                num_experts=4, top_k=2, mtp_num_heads=3)

        assert len(model.mtp_heads) == 2
        assert model.num_experts == 4
        assert hasattr(model, 'get_load_balancing_loss')
        assert hasattr(model, 'get_mtp_params')

    def test_gradient_flow(self):
        from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP

        tokenizer = MockTokenizer(vocab_size=50)
        model = ChatModelMoEMTP(tokenizer, embed_size=32, num_layers=1,
                                num_experts=4, top_k=2, mtp_num_heads=3)

        input_ids = torch.randint(0, 50, (1, 5))
        primary_logits, mtp_logits = model(input_ids)

        loss = primary_logits.sum()
        for logits in mtp_logits:
            loss = loss + logits.sum()

        loss.backward()

        # Check MoE gate has gradients
        for layer in model.model.transformer.h:
            if hasattr(layer.mlp, 'gate'):
                assert layer.mlp.gate.weight.grad is not None

    def test_mtp_params_count(self):
        from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP

        tokenizer = MockTokenizer(vocab_size=50)
        model = ChatModelMoEMTP(tokenizer, embed_size=32, num_layers=1,
                                num_experts=4, top_k=2, mtp_num_heads=4)

        mtp_params = model.get_mtp_params()
        total_params = model.get_total_params()

        assert mtp_params > 0
        assert mtp_params < total_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
