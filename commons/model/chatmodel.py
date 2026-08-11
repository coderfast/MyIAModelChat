"""
GPT-2 Transformer model for conversational AI.

Uses HuggingFace's GPT2LMHeadModel with configurable architecture.
Can optionally load pre-trained GPT-2 weights for better initialization.
"""

import os
import logging
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pretrained', 'gpt2')


class ChatModel(nn.Module):
    def __init__(self, tokenizer, embed_size, num_layers=2, pretrained=False):
        super(ChatModel, self).__init__()
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=embed_size,
            n_head=4,
            n_layer=num_layers,
            n_positions=512
        )

        if pretrained:
            self._init_from_pretrained(config)
        else:
            self.model = GPT2LMHeadModel(config)

    def _init_from_pretrained(self, config):
        """Load pre-trained GPT-2 weights (first N layers + embeddings) from local folder or HuggingFace."""
        pretrained_model = None
        source = None

        # Try local folder first
        if os.path.isdir(PRETRAINED_DIR) and os.path.exists(os.path.join(PRETRAINED_DIR, 'pytorch_model.bin')):
            try:
                logger.info(f"Loading pre-trained GPT-2 from local: {PRETRAINED_DIR}")
                pretrained_model = GPT2LMHeadModel.from_pretrained(PRETRAINED_DIR)
                source = "local"
            except Exception as e:
                logger.warning(f"Could not load local GPT-2: {e}")

        # Fallback to HuggingFace Hub
        if pretrained_model is None:
            try:
                logger.info("Loading pre-trained GPT-2 from HuggingFace Hub...")
                pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
                source = "HuggingFace"
            except Exception as e:
                logger.warning(f"Could not load GPT-2 from HuggingFace: {e}. Using random init.")
                self.model = GPT2LMHeadModel(config)
                return

        pretrained_config = pretrained_model.config
        self.model = GPT2LMHeadModel(config)

        with torch.no_grad():
            src_vocab = pretrained_config.vocab_size
            dst_vocab = config.vocab_size
            copy_vocab = min(src_vocab, dst_vocab)
            self.model.transformer.wte.weight[:copy_vocab] = pretrained_model.transformer.wte.weight[:copy_vocab]

            copy_pos = min(config.n_positions, pretrained_config.n_positions)
            self.model.transformer.wpe.weight[:copy_pos] = pretrained_model.transformer.wpe.weight[:copy_pos]

            src_layers = pretrained_model.transformer.h
            dst_layers = self.model.transformer.h
            for i in range(min(len(dst_layers), len(src_layers))):
                dst_layers[i].load_state_dict(src_layers[i].state_dict())

            self.model.transformer.ln_f.load_state_dict(pretrained_model.transformer.ln_f.state_dict())

        del pretrained_model
        logger.info(f"GPT-2 pre-trained ({source}): {copy_vocab} tokens, {min(len(dst_layers), len(src_layers))} layers copied")

    def forward(self, input_ids):
        """
        Forward pass through the model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) with token IDs

        Returns:
            logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        return self.model(input_ids).logits
