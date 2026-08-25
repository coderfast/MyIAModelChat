"""Mixture of Experts (MoE) Transformer model for conversational AI.

Extends ChatModel with MoE architecture where each transformer block has
multiple expert FFN layers with a gating network for routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .chatmodel import ChatModel


@dataclass
class MoEConfig:
    """Configuration for MoE layer."""
    num_experts: int = 4
    top_k: int = 2
    load_balance_weight: float = 0.01
    expert_capacity_factor: float = 1.25


class MoELayer(nn.Module):
    """Mixture of Experts layer with gating network.
    
    Each MoE layer replaces the standard FFN in a transformer block with
    multiple expert FFNs and a gating network that routes tokens to the
    top-K experts.
    
    Args:
        hidden_size: Dimension of input/output vectors
        num_experts: Number of expert FFN networks
        top_k: Number of experts to route each token to
    """
    
    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Expert FFN networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize gate with small weights for balanced routing
        nn.init.normal_(self.gate.weight, std=0.01)
        
        # Store gate scores for loss computation (not part of output)
        self.last_gate_scores = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            output: Combined output from selected experts (same shape as input)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (batch, seq, num_experts)
        
        # Store gate scores for loss computation
        self.last_gate_scores = gate_scores
        
        # Select top-K experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize top-K scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (batch, seq)
            expert_weight = top_k_scores[:, :, k]  # (batch, seq)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output
        
        return output


class ChatModelMoE(ChatModel):
    """GPT-2 Transformer model with Mixture of Experts.
    
    Extends ChatModel by replacing the FFN in each transformer block with
    a MoE layer. This allows the model to specialize different experts for
    different types of tasks (conversation, reasoning, tool use, etc.).
    
    Note: This model uses the parent ChatModel's forward method which handles
    the transformer forward pass automatically. The MoE layers are injected
    into the transformer blocks.
    
    Args:
        tokenizer: Tokenizer instance for vocabulary size
        embed_size: Embedding dimension
        num_layers: Number of transformer layers
        num_experts: Number of experts per MoE layer
        top_k: Number of experts to route each token to
        load_balance_weight: Weight for load balancing loss
    """
    
    def __init__(self, tokenizer, embed_size: int, num_layers: int = 2,
                 num_experts: int = 4, top_k: int = 2,
                 load_balance_weight: float = 0.01):
        # Initialize parent ChatModel
        super().__init__(tokenizer, embed_size, num_layers)
        
        # Store MoE configuration
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.embed_size = embed_size
        
        # Store gate scores for loss computation
        self._all_gate_scores: List[torch.Tensor] = []
        
        # Replace FFN in each transformer block with MoE
        self._replace_ffn_with_moe()
        
        # Routing statistics
        self.routing_stats: Dict[str, List[float]] = {
            'gate_scores_entropy': [],
            'expert_utilization': [],
            'load_balance_loss': [],
        }
    
    def _replace_ffn_with_moe(self):
        """Replace FFN layers with MoE layers in all transformer blocks."""
        for layer in self.model.transformer.h:
            # Replace with MoE layer
            layer.mlp = MoELayer(self.embed_size, self.num_experts, self.top_k)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass with MoE routing.
        
        This method uses the parent ChatModel's forward method but captures
        gate scores from the MoE layers.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            gate_scores: List of gate scores from each MoE layer (for loss)
        """
        self._all_gate_scores.clear()
        
        # Use parent's forward method
        logits = super().forward(input_ids)
        
        # Collect gate scores from MoE layers
        for layer in self.model.transformer.h:
            if isinstance(layer.mlp, MoELayer) and layer.mlp.last_gate_scores is not None:
                self._all_gate_scores.append(layer.mlp.last_gate_scores)
        
        return logits, self._all_gate_scores if self._all_gate_scores else None
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """Get the utilization statistics for each expert.
        
        Returns:
            Dictionary mapping expert index to utilization percentage
        """
        if not self._all_gate_scores:
            return {i: 0.0 for i in range(self.num_experts)}
        
        # Stack all gate scores and average
        all_scores = torch.stack([s.detach() if s.requires_grad else s for s in self._all_gate_scores])
        all_scores = all_scores.to(next(self.parameters()).device)
        avg_scores = all_scores.mean(dim=[0, 1, 2])  # Average across batch, seq, layers
        
        utilization = {}
        for i in range(self.num_experts):
            utilization[i] = avg_scores[i].item() * 100
        
        return utilization
    
    def get_load_balancing_loss(self, gate_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage.
        
        Args:
            gate_scores: Optional pre-computed gate scores. If None, uses stored scores.
            
        Returns:
            Scalar loss tensor
        """
        if gate_scores is None:
            if not self._all_gate_scores:
                return torch.tensor(0.0)
            gate_scores = torch.stack([s.detach() if s.requires_grad else s for s in self._all_gate_scores])
            gate_scores = gate_scores.to(next(self.parameters()).device)
        
        # Average gate scores across batch and sequence
        # gate_scores shape: (num_layers, batch, seq, num_experts)
        routing_freq = gate_scores.mean(dim=[0, 1, 2])  # (num_experts,)
        
        # Target: uniform distribution
        target = torch.ones_like(routing_freq) / self.num_experts
        
        # KL divergence loss
        loss = F.kl_div(
            routing_freq.log(), 
            target, 
            reduction='sum'
        )
        
        return loss
    
    def freeze_attention(self):
        """Freeze attention layers, leaving only MoE experts trainable."""
        # Freeze embeddings
        for param in self.model.transformer.wte.parameters():
            param.requires_grad = False
        for param in self.model.transformer.wpe.parameters():
            param.requires_grad = False
        
        # Freeze attention in each layer
        for layer in self.model.transformer.h:
            for param in layer.attn.parameters():
                param.requires_grad = False
        
        # Unfreeze MoE layers
        for layer in self.model.transformer.h:
            for param in layer.mlp.parameters():
                param.requires_grad = True
        
        # Unfreeze final layers
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_expert_params(self) -> int:
        """Count parameters in MoE experts only."""
        expert_params = 0
        for layer in self.model.transformer.h:
            expert_params += sum(p.numel() for p in layer.mlp.parameters())
        return expert_params
