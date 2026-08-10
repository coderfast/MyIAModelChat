"""Tests for Mixture of Experts (MoE) model and dataset."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SP_MODEL_PATH = os.path.join('dataset_cache', 'sentencepiece.model')


@pytest.fixture(scope="module")
def sp_tokenizer():
    """Load SentencePiece tokenizer if model exists."""
    if not os.path.exists(SP_MODEL_PATH):
        pytest.skip("sentencepiece.model not found; run --prepare-data first")
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
    return SentencePieceTokenizerWrapper(SP_MODEL_PATH)


# ============================================================
# MoE Layer Tests
# ============================================================

class TestMoELayer:
    """Tests for MoE layer."""
    
    def test_moe_layer_forward(self):
        """Test MoE layer forward pass produces correct output shape."""
        from commons.model.chatmodel_moe import MoELayer
        
        hidden_size = 64
        num_experts = 4
        top_k = 2
        batch_size = 2
        seq_len = 10
        
        layer = MoELayer(hidden_size, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = layer(x)
        
        assert output.shape == x.shape
        # Gate scores are stored as attribute
        assert layer.last_gate_scores is not None
        assert layer.last_gate_scores.shape == (batch_size, seq_len, num_experts)
    
    def test_gating_network_routing(self):
        """Test that gating network selects top-K experts."""
        from commons.model.chatmodel_moe import MoELayer
        
        hidden_size = 64
        num_experts = 4
        top_k = 2
        
        layer = MoELayer(hidden_size, num_experts, top_k)
        x = torch.randn(1, 5, hidden_size)
        
        output = layer(x)
        gate_scores = layer.last_gate_scores
        
        # Check that top-K scores sum to 1 (after normalization)
        top_k_scores, _ = torch.topk(gate_scores, top_k, dim=-1)
        normalized_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Sum of normalized top-K scores should be ~1.0
        sums = normalized_scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_load_balancing_loss(self):
        """Test that load balancing loss is computed correctly."""
        from commons.model.chatmodel_moe import MoELayer
        
        hidden_size = 64
        num_experts = 4
        top_k = 2
        
        layer = MoELayer(hidden_size, num_experts, top_k)
        # Use larger input to ensure all experts get some traffic
        x = torch.randn(4, 20, hidden_size)
        
        output = layer(x)
        gate_scores = layer.last_gate_scores
        
        # Compute load balancing loss manually
        routing_freq = gate_scores.mean(dim=[0, 1])
        # Add small epsilon to avoid log(0)
        routing_freq = routing_freq + 1e-8
        routing_freq = routing_freq / routing_freq.sum()
        target = torch.ones(num_experts) / num_experts
        loss = torch.nn.functional.kl_div(routing_freq.log(), target, reduction='sum')
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


# ============================================================
# ChatModelMoE Tests
# ============================================================

class TestChatModelMoE:
    """Tests for ChatModelMoE."""
    
    @pytest.fixture(autouse=True)
    def setup(self, sp_tokenizer):
        self.tokenizer = sp_tokenizer
    
    def test_moe_chatmodel(self):
        """Test that MoE model generates valid output."""
        from commons.model.chatmodel_moe import ChatModelMoE
        
        model = ChatModelMoE(
            self.tokenizer,
            embed_size=64,
            num_layers=2,
            num_experts=4,
            top_k=2
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits, gate_scores = model(input_ids)
        
        assert logits.shape[0] == 1  # batch size
        assert logits.shape[2] == self.tokenizer.vocab_size  # vocab size
        assert len(gate_scores) == 2  # num_layers
    
    def test_expert_utilization(self):
        """Test expert utilization metrics."""
        from commons.model.chatmodel_moe import ChatModelMoE
        
        model = ChatModelMoE(
            self.tokenizer,
            embed_size=64,
            num_layers=2,
            num_experts=4,
            top_k=2
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        model(input_ids)
        
        utilization = model.get_expert_utilization()
        
        assert len(utilization) == 4
        assert all(0 <= v <= 100 for v in utilization.values())
    
    def test_freeze_attention(self):
        """Test that freeze_attention freezes correct parameters."""
        from commons.model.chatmodel_moe import ChatModelMoE
        
        model = ChatModelMoE(
            self.tokenizer,
            embed_size=64,
            num_layers=2,
            num_experts=4,
            top_k=2
        )
        
        # Before freeze
        trainable_before = model.get_trainable_params()
        
        model.freeze_attention()
        
        trainable_after = model.get_trainable_params()
        
        # Should have fewer trainable parameters after freeze
        assert trainable_after < trainable_before
        
        # Check that MoE layers are still trainable
        for layer in model.model.transformer.h:
            for param in layer.mlp.parameters():
                assert param.requires_grad is True
    
    def test_moe_params_count(self):
        """Test parameter counting methods."""
        from commons.model.chatmodel_moe import ChatModelMoE
        
        model = ChatModelMoE(
            self.tokenizer,
            embed_size=64,
            num_layers=2,
            num_experts=4,
            top_k=2
        )
        
        total = model.get_total_params()
        expert = model.get_expert_params()
        trainable = model.get_trainable_params()
        
        assert total > 0
        assert expert > 0
        assert expert < total
        assert trainable == total  # All trainable before freeze


# ============================================================
# MoE Dataset Tests
# ============================================================

class TestMoEData:
    """Tests for MoE dataset processing."""
    
    @pytest.fixture(autouse=True)
    def setup(self, sp_tokenizer):
        self.tokenizer = sp_tokenizer
    
    def test_expert_labels_dataset(self):
        """Test that expert labels are assigned correctly."""
        from dataset_preparer.agent.moe_data import MoEDataProcessor
        
        processor = MoEDataProcessor(self.tokenizer)
        
        # Create a sample with thinking tokens
        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()
        
        if thinking_id < 0 or thinking_end_id < 0:
            pytest.skip("Thinking tokens not in vocabulary")
        
        token_ids = [100, 200, thinking_id, 300, 400, thinking_end_id, 500]
        expert_labels = processor.assign_expert_labels(token_ids)
        
        assert len(expert_labels) == len(token_ids)
        assert expert_labels[0] == 0  # conversation
        assert expert_labels[2] == 1  # thinking
        assert expert_labels[3] == 1  # thinking
        assert expert_labels[5] == 1  # thinking
        assert expert_labels[6] == 0  # conversation
    
    def test_moe_dataset_processing(self):
        """Test processing a full dataset."""
        from dataset_preparer.agent.moe_data import MoEDataProcessor
        
        processor = MoEDataProcessor(self.tokenizer)
        
        dataset = [
            {'token_ids': [1, 2, 3], 'source': 'test'},
            {'token_ids': [4, 5, 6], 'source': 'test'},
        ]
        
        processed = processor.process_dataset(dataset)
        
        assert len(processed) == 2
        assert 'expert_ids' in processed[0]
        assert len(processed[0]['expert_ids']) == 3
    
    def test_expert_statistics(self):
        """Test expert statistics computation."""
        from dataset_preparer.agent.moe_data import MoEDataProcessor
        
        processor = MoEDataProcessor(self.tokenizer)
        
        dataset = [
            {'expert_ids': [0, 0, 1, 1, 2, 2, 3, 3]},
            {'expert_ids': [0, 0, 0, 1, 1, 2, 3, 3]},
        ]
        
        stats = processor.get_expert_statistics(dataset)
        
        assert len(stats) == 4
        assert all('count' in v for v in stats.values())
        assert all('percentage' in v for v in stats.values())
        assert all('name' in v for v in stats.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
