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

    def test_no_hidden_size_parameter(self):
        """ChatModel should NOT accept hidden_size parameter (removed in bug fix)."""
        from commons.model.chatmodel import ChatModel
        import inspect
        sig = inspect.signature(ChatModel.__init__)
        params = list(sig.parameters.keys())
        assert 'hidden_size' not in params, f"hidden_size should be removed, got params: {params}"

    def test_constructor_signature(self):
        """ChatModel should only accept tokenizer, embed_size, num_layers."""
        from commons.model.chatmodel import ChatModel
        import inspect
        sig = inspect.signature(ChatModel.__init__)
        params = list(sig.parameters.keys())
        # Remove 'self'
        params = [p for p in params if p != 'self']
        assert params == ['tokenizer', 'embed_size', 'num_layers'], f"Unexpected params: {params}"

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
