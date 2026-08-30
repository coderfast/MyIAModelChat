"""Model package for conversational AI.

This package provides:
- ChatModel: Standard GPT-2 Transformer model
- ChatModelMoE: GPT-2 with Mixture of Experts
- ChatModelMTP: GPT-2 with Multi-Token Prediction
- ChatModelMoEMTP: GPT-2 with MoE + MTP combined
"""

from .chatmodel import ChatModel
from .chatmodel_moe import ChatModelMoE, MoELayer, MoEConfig
from .chatmodel_mtp import ChatModelMTP, MTPHead
from .chatmodel_moe_mtp import ChatModelMoEMTP

__all__ = [
    'ChatModel',
    'ChatModelMoE',
    'MoELayer',
    'MoEConfig',
    'ChatModelMTP',
    'MTPHead',
    'ChatModelMoEMTP',
]
