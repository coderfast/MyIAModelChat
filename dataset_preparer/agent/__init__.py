"""Agent data preparation package.

This package provides:
- MoEDataProcessor: Assigns expert labels to tokens for MoE training
"""

from .moe_data import MoEDataProcessor, MoELabelConfig, assign_expert_labels_to_dataset

__all__ = [
    'MoEDataProcessor',
    'MoELabelConfig',
    'assign_expert_labels_to_dataset',
]
