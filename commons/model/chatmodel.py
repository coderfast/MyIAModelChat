"""
GPT-2 Transformer model for conversational AI.

Uses HuggingFace's GPT2LMHeadModel with configurable architecture.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class ChatModel(nn.Module):
    def __init__(self, tokenizer, embed_size, hidden_size, num_layers=2):
        super(ChatModel, self).__init__()
        self.tokenizer = tokenizer
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=embed_size,
            n_head=4,
            n_layer=num_layers,
            n_positions=512
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids):
        """
        Forward pass through the model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) with token IDs

        Returns:
            logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        return self.model(input_ids).logits
