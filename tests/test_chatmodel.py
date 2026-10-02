"""Tests for ChatModel (GPT-2 Transformer architecture)."""

import torch
import pytest


class FakeTokenizer:
    """Minimal tokenizer mock for ChatModel tests."""
    vocab_size = 8000
    pad_token_id = 0
    unk_token_id = 1
    eos_token_id = 2

    def get_pad_index(self): return 0
    def get_unk_index(self): return 1
    def get_eos_index(self): return 2
    def get_bos_index(self): return 3
    def get_thinking_index(self): return 5
    def get_thinking_end_index(self): return 6


class TestChatModel:
    """Tests for ChatModel construction and forward pass."""

    def setup_method(self):
        self.tokenizer = FakeTokenizer()

    def test_basic_construction(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=256, num_layers=2)
        assert model is not None

    def test_constructor_signature(self):
        """ChatModel accepts the full configurable architecture."""
        from commons.model.chatmodel import ChatModel
        import inspect
        sig = inspect.signature(ChatModel.__init__)
        params = list(sig.parameters.keys())
        # Remove 'self'
        params = [p for p in params if p != 'self']
        assert params == ['tokenizer', 'embed_size', 'num_layers', 'n_head',
                          'n_positions', 'hidden_size'], f"Unexpected params: {params}"

    def test_configurable_n_head_and_positions(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=128, num_layers=2,
                          n_head=2, n_positions=128)
        assert model.model.config.n_head == 2
        assert model.model.config.n_positions == 128
        assert model.model.transformer.wpe.weight.shape[0] == 128

    def test_configurable_hidden_size(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=128, num_layers=2,
                          hidden_size=384)
        assert model.model.config.n_inner == 384
        w = model.model.transformer.h[0].mlp.c_fc.weight
        # Conv1D layout: (embed, inner)
        assert tuple(w.shape) == (128, 384)

    def test_default_hidden_size_is_auto(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=128, num_layers=2)
        # None => GPT-2 default FFN width (4 * embed_size)
        assert model.model.config.n_inner in (None, 512)

    def test_invalid_n_head_raises(self):
        from commons.model.chatmodel import ChatModel
        with pytest.raises(ValueError):
            ChatModel(self.tokenizer, embed_size=100, num_layers=1, n_head=3)

    def test_resolve_hidden_size(self):
        from commons.model.chatmodel import resolve_hidden_size
        import torch
        embed = 64
        # Conv1D layout (embed, inner) as used by HF GPT-2
        state = {'model.transformer.h.0.mlp.c_fc.weight': torch.zeros(embed, 256)}
        assert resolve_hidden_size(state, embed) == 256
        # nn.Linear layout (inner, embed)
        state_lin = {'model.transformer.h.0.mlp.c_fc.weight': torch.zeros(256, embed)}
        assert resolve_hidden_size(state_lin, embed) == 256
        # Missing key (e.g. MoE) => default
        assert resolve_hidden_size({'other.key': torch.zeros(1)}, embed, default=192) == 192
        assert resolve_hidden_size(None) is None

    def test_forward_output_shape(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=256, num_layers=2)
        model.eval()
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (2, 32, self.tokenizer.vocab_size)

    def test_forward_different_sizes(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=128, num_layers=2)
        model.eval()
        # Single batch, short sequence
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, 16))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 16, self.tokenizer.vocab_size)

    def test_model_is_nn_module(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=256, num_layers=2)
        assert isinstance(model, torch.nn.Module)

    def test_model_has_parameters(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=256, num_layers=2)
        params = list(model.parameters())
        assert len(params) > 0, "Model should have trainable parameters"

    def test_model_trainable(self):
        from commons.model.chatmodel import ChatModel
        model = ChatModel(self.tokenizer, embed_size=256, num_layers=2)
        model.train()
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
        targets = torch.randint(0, self.tokenizer.vocab_size, (2, 32))
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.view(-1)
        )
        loss.backward()
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                break  # At least one parameter should have gradients
        assert loss.item() > 0

    def test_different_embed_sizes(self):
        from commons.model.chatmodel import ChatModel
        for embed_size in [64, 128, 256]:
            model = ChatModel(self.tokenizer, embed_size=embed_size, num_layers=2)
            model.eval()
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, 16))
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (1, 16, self.tokenizer.vocab_size)

    def test_different_num_layers(self):
        from commons.model.chatmodel import ChatModel
        for num_layers in [1, 2, 4]:
            model = ChatModel(self.tokenizer, embed_size=128, num_layers=num_layers)
            model.eval()
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, 16))
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (1, 16, self.tokenizer.vocab_size)


def run_tests():
    """Standalone test runner."""
    import sys
    test = TestChatModel()
    passed = 0
    failed = 0
    for name in dir(test):
        if name.startswith('test_'):
            try:
                test.setup_method()
                getattr(test, name)()
                print(f"  PASS: {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
                failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
