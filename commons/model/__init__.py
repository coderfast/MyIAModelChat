"""Model package for conversational AI.

This package provides:
- ChatModel: Standard GPT-2 Transformer model
- ChatModelMoE: GPT-2 with Mixture of Experts
"""

from .chatmodel import ChatModel
from .chatmodel_moe import ChatModelMoE, MoELayer, MoEConfig

__all__ = [
    'ChatModel',
    'ChatModelMoE',
    'MoELayer',
    'MoEConfig',
]
