"""Mixture of Experts (MoE) + Multi-Token Prediction (MTP) combined model.

Extends ChatModelMoE with MTP heads. MoE replaces FFN layers for expert routing,
while MTP adds auxiliary prediction heads for future tokens.

Architecture:
    MoE Transformer Backbone (expert FFNs)
        ├── lm_head (predicts token t+1) → PRIMARY
        ├── MTP Head 1 (predicts token t+2)
        ├── MTP Head 2 (predicts token t+3)
        └── MTP Head 3 (predicts token t+4)

Combines:
    - MoE: Specialized expert routing for different task types
    - MTP: Multi-token prediction for better reasoning and planning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from .chatmodel_moe import ChatModelMoE
from .chatmodel_mtp import MTPHead


class ChatModelMoEMTP(ChatModelMoE):
    """GPT-2 Transformer with both MoE and Multi-Token Prediction.

    Combines MoE expert routing (replacing FFN layers) with MTP heads
    that predict multiple future tokens simultaneously.

    Args:
        tokenizer: Tokenizer instance for vocabulary size
        embed_size: Embedding dimension
        num_layers: Number of transformer layers
        num_experts: Number of experts per MoE layer
        top_k: Number of experts to route each token to
        load_balance_weight: Weight for load balancing loss
        mtp_num_heads: Total prediction heads including primary lm_head
        mtp_loss_weight: Weight for MTP auxiliary loss
    """

    def __init__(self, tokenizer, embed_size: int, num_layers: int = 2,
                 num_experts: int = 4, top_k: int = 2,
                 load_balance_weight: float = 0.01,
                 mtp_num_heads: int = 4, mtp_loss_weight: float = 0.3):
        # Initialize ChatModelMoE (handles MoE FFN replacement)
        super().__init__(tokenizer, embed_size, num_layers,
                         num_experts=num_experts, top_k=top_k,
                         load_balance_weight=load_balance_weight)

        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight

        # Create MTP heads (excluding primary lm_head)
        self.mtp_heads = nn.ModuleList([
            MTPHead(embed_size, tokenizer.vocab_size)
            for _ in range(mtp_num_heads - 1)
        ])

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass with MoE routing and MTP predictions.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            primary_logits: Output logits for next token (batch, seq_len, vocab_size)
            gate_scores: List of gate score tensors from MoE layers (for load balancing)
        """
        # Clear gate scores
        self._all_gate_scores.clear()

        # Get hidden states from MoE transformer backbone
        hidden_states = self.model.transformer(input_ids).last_hidden_state

        # Collect gate scores from MoE layers
        for layer in self.model.transformer.h:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'last_gate_scores') and layer.mlp.last_gate_scores is not None:
                self._all_gate_scores.append(layer.mlp.last_gate_scores)

        # Primary prediction (next token)
        primary_logits = self.model.lm_head(hidden_states)

        # MTP predictions for future tokens (stored as side channel for loss computation)
        self._mtp_logits = [head(hidden_states) for head in self.mtp_heads]

        return primary_logits, self._all_gate_scores if self._all_gate_scores else None

    def get_mtp_head_accuracies(self, predictions: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, float]:
        """Compute per-head accuracy for MTP predictions.

        Args:
            predictions: Primary model predictions (batch, seq_len)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Dictionary mapping head names to accuracy values
        """
        metrics = {}
        non_pad = targets.ne(0)
        if non_pad.any():
            correct = (predictions.argmax(dim=-1) == targets) & non_pad
            metrics['primary_accuracy'] = correct.float().sum().item() / non_pad.float().sum().item()
        return metrics

    def get_mtp_params(self) -> int:
        """Count parameters in MTP heads only."""
        mtp_params = 0
        for head in self.mtp_heads:
            mtp_params += sum(p.numel() for p in head.parameters())
        return mtp_params
