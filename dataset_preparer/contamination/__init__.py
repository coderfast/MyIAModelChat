"""
Contamination detection and filtering module for dataset_preparer.

Provides cross-source deduplication, quality filtering, balance control,
leakage detection, and audit reporting for training data preparation.
"""

from dataset_preparer.contamination.filters import NoiseFilter, QualityFilter
from dataset_preparer.contamination.dedup import CrossSourceDeduplicator
from dataset_preparer.contamination.balance import SourceBalancer
from dataset_preparer.contamination.leakage import LeakageDetector
from dataset_preparer.contamination.audit import AuditReport

__all__ = [
    'NoiseFilter',
    'QualityFilter',
    'CrossSourceDeduplicator',
    'SourceBalancer',
    'LeakageDetector',
    'AuditReport',
]
