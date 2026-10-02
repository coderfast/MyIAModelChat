"""
GPT-2 Transformer model for conversational AI.

Uses HuggingFace's GPT2LMHeadModel with configurable architecture.

All architecture parameters (n_head, n_positions, hidden_size, ...) are
constructor arguments sourced from training_config.json via TrainingConfig.
"""

import logging
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)

# State-dict key of the first standard FFN layer (absent in MoE models,
# where the FFN is replaced by the experts module).
_FIRST_C_FC_KEY = 'model.transformer.h.0.mlp.c_fc.weight'


def resolve_hidden_size(state_dict, embed_size=None, default=None):
    """Recover the FFN inner dimension (GPT-2 `n_inner`) from a state dict.

    HF GPT-2 stores c_fc as Conv1D with weight (nx=embed, nf=inner); an
    nn.Linear layout would be (inner, embed) and is disambiguated with
    `embed_size`. Returns `default` when the key is absent (e.g. MoE models,
    where the FFN was replaced by the experts module).
    """
    if not state_dict:
        return default
    weight = state_dict.get(_FIRST_C_FC_KEY)
    if weight is None:
        return default
    nx, nf = int(weight.shape[0]), int(weight.shape[1])
    if embed_size is not None and nx != embed_size and nf == embed_size:
        return nx  # nn.Linear layout (inner, embed)
    return nf      # Conv1D layout (embed, inner)


class ChatModel(nn.Module):
    def __init__(self, tokenizer, embed_size, num_layers=2,
                 n_head=4, n_positions=512, hidden_size=None):
        super(ChatModel, self).__init__()
        if n_head < 1 or embed_size % n_head != 0:
            raise ValueError(
                f"embed_size ({embed_size}) must be a positive multiple of n_head ({n_head})"
            )
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=embed_size,
            n_head=n_head,
            n_layer=num_layers,
            n_positions=n_positions,
            n_inner=hidden_size
        )

        self.model = GPT2LMHeadModel(config)
        self.embed_size = embed_size
        self.n_head = n_head
        self.n_positions = n_positions

    def forward(self, input_ids):
        """
        Forward pass through the model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) with token IDs

        Returns:
            logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        return self.model(input_ids).logits
