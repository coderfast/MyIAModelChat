"""Multi-Token Prediction (MTP) Transformer model for conversational AI.

Extends ChatModel with MTP heads that predict multiple future tokens simultaneously.
Based on Gloeckle et al. (2024) - Meta FAIR: "Better & Faster Large Language Models
via Multi-Token Prediction".

Architecture:
    Shared Transformer Backbone
        ├── lm_head (predicts token t+1) → PRIMARY
        ├── MTP Head 1 (predicts token t+2)
        ├── MTP Head 2 (predicts token t+3)
        └── MTP Head 3 (predicts token t+4)

Each MTP head: Linear(hidden → hidden) → GELU → Linear(hidden → vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from .chatmodel import ChatModel


class MTPHead(nn.Module):
    """Single MTP head for predicting the k-th future token.

    Args:
        hidden_size: Dimension of input hidden states
        vocab_size: Size of vocabulary for output logits
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states and produce vocab logits.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size)

        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        projected = F.gelu(self.projection(hidden_states))
        return self.output_head(projected)


class ChatModelMTP(ChatModel):
    """GPT-2 Transformer model with Multi-Token Prediction.

    Extends ChatModel by adding independent MTP heads that predict future
    tokens at positions t+2, t+3, ..., t+k. The primary lm_head still
    predicts t+1. During inference, only the primary head is used.

    Args:
        tokenizer: Tokenizer instance for vocabulary size
        embed_size: Embedding dimension
        num_layers: Number of transformer layers
        mtp_num_heads: Total prediction heads including primary lm_head (default: 4)
        mtp_loss_weight: Weight for MTP auxiliary loss (default: 0.3)
    """

    def __init__(self, tokenizer, embed_size: int, num_layers: int = 2,
                 mtp_num_heads: int = 4, mtp_loss_weight: float = 0.3):
        super().__init__(tokenizer, embed_size, num_layers)

        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.embed_size = embed_size

        # Create MTP heads (excluding primary lm_head which is head 0)
        self.mtp_heads = nn.ModuleList([
            MTPHead(embed_size, tokenizer.vocab_size)
            for _ in range(mtp_num_heads - 1)
        ])

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass with Multi-Token Prediction.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            primary_logits: Output logits for next token (batch, seq_len, vocab_size)
            mtp_logits: List of logit tensors from MTP heads, each (batch, seq_len, vocab_size)
        """
        # Get hidden states from the transformer backbone
        hidden_states = self.model.transformer(input_ids).last_hidden_state

        # Primary prediction (next token) - uses existing lm_head
        primary_logits = self.model.lm_head(hidden_states)

        # MTP predictions for future tokens
        mtp_logits = [head(hidden_states) for head in self.mtp_heads]

        return primary_logits, mtp_logits

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

        # Primary head accuracy (token t+1)
        non_pad = targets.ne(0)  # assuming pad_idx=0
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

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
