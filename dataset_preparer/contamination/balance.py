"""
Source balance controller.

Ensures no single data source dominates the combined dataset
by applying ratio limits and intelligent undersampling.
"""
import random
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BalanceResult:
    """Result of balance operation."""
    kept_indices: List[int] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    original_counts: Dict[str, int] = field(default_factory=dict)
    removed_per_source: Dict[str, int] = field(default_factory=dict)
    total_removed: int = 0

    @property
    def total_kept(self) -> int:
        return len(self.kept_indices)

    @property
    def ratios(self) -> Dict[str, float]:
        total = sum(self.source_counts.values())
        if total == 0:
            return {}
        return {s: c / total for s, c in self.source_counts.items()}

    def summary(self) -> str:
        parts = [f"Balance: {self.total_kept} samples"]
        for source, ratio in sorted(self.ratios.items()):
            count = self.source_counts[source]
            parts.append(f"  {source}: {count} ({ratio:.1%})")
        if self.total_removed > 0:
            parts.append(f"  Removed: {self.total_removed}")
        return " | ".join(parts)


class SourceBalancer:
    """Controls the balance of samples across data sources."""

    def __init__(self, max_ratio: float = 0.3, seed: int = 42):
        """
        Initialize source balancer.

        Args:
            max_ratio: Maximum fraction any single source can have (0-1)
            seed: Random seed for reproducibility
        """
        self.max_ratio = max_ratio
        self.seed = seed

    def balance(
        self,
        indices: List[int],
        sources: List[str],
        max_total: Optional[int] = None
    ) -> BalanceResult:
        """
        Balance samples across sources.

        Args:
            indices: List of sample indices
            sources: List of source names (same length as indices)
            max_total: Optional maximum total samples after balancing

        Returns:
            BalanceResult with kept indices and statistics
        """
        result = BalanceResult()

        if not indices or not sources:
            return result

        source_to_indices: Dict[str, List[int]] = {}
        for idx, source in zip(indices, sources):
            if source not in source_to_indices:
                source_to_indices[source] = []
            source_to_indices[source].append(idx)

        for source, idxs in source_to_indices.items():
            result.original_counts[source] = len(idxs)

        total_samples = len(indices)
        max_per_source = int(total_samples * self.max_ratio)

        kept_indices = []
        removed_per_source = {}

        for source, idxs in source_to_indices.items():
            if len(idxs) <= max_per_source:
                kept_indices.extend(idxs)
                removed_per_source[source] = 0
            else:
                rng = random.Random(self.seed + hash(source))
                selected = rng.sample(idxs, max_per_source)
                kept_indices.extend(selected)
                removed_per_source[source] = len(idxs) - max_per_source

        if max_total and len(kept_indices) > max_total:
            rng = random.Random(self.seed)
            kept_indices = rng.sample(kept_indices, max_total)

        result.kept_indices = kept_indices
        result.total_removed = sum(removed_per_source.values())
        result.removed_per_source = removed_per_source

        source_counter = Counter()
        for idx in kept_indices:
            idx_pos = indices.index(idx) if idx in indices else -1
            if idx_pos >= 0 and idx_pos < len(sources):
                source_counter[sources[idx_pos]] += 1

        result.source_counts = dict(source_counter)

        return result


def balance_sources(
    indices: List[int],
    sources: List[str],
    max_ratio: float = 0.3,
    max_total: Optional[int] = None
) -> Tuple[List[int], BalanceResult]:
    """
    Convenience function for source balancing.

    Args:
        indices: Sample indices
        sources: Source names
        max_ratio: Maximum ratio per source
        max_total: Optional total sample cap

    Returns:
        Tuple of (kept_indices, BalanceResult)
    """
    balancer = SourceBalancer(max_ratio=max_ratio)
    result = balancer.balance(indices, sources, max_total)
    return result.kept_indices, result
