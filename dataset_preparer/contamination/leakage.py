"""
Data leakage detection module.

Detects n-gram overlap between samples, repeated fragments
across the dataset, and potential train/test contamination.
"""
import re
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LeakageResult:
    """Result of leakage detection."""
    flagged_indices: List[int] = field(default_factory=list)
    overlap_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    repeated_fragments: List[Dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def total_flagged(self) -> int:
        return len(self.flagged_indices)

    def summary(self) -> str:
        return (
            f"Leakage: {len(self.flagged_indices)} flagged samples | "
            f"Overlap pairs: {len(self.overlap_pairs)} | "
            f"Repeated fragments: {len(self.repeated_fragments)}"
        )


class LeakageDetector:
    """Detects data leakage via n-gram overlap and repeated fragments."""

    def __init__(
        self,
        ngram_size: int = 5,
        overlap_threshold: float = 0.5,
        min_fragment_count: int = 3
    ):
        """
        Initialize leakage detector.

        Args:
            ngram_size: Size of n-grams for overlap detection
            overlap_threshold: Threshold for flagging overlap (0-1)
            min_fragment_count: Minimum occurrences to flag a fragment
        """
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        self.min_fragment_count = min_fragment_count

    def detect(self, texts: List[str]) -> LeakageResult:
        """
        Detect leakage in a list of texts.

        Args:
            texts: List of text strings

        Returns:
            LeakageResult with flagged samples and details
        """
        result = LeakageResult()

        if not texts or len(texts) < 2:
            return result

        # Safety limit: for large datasets, sample to avoid O(n^2) memory/time
        MAX_PAIRWISE = 5000
        if len(texts) > MAX_PAIRWISE:
            import random
            sampled_indices = sorted(random.sample(range(len(texts)), MAX_PAIRWISE))
            sampled_texts = [texts[i] for i in sampled_indices]
            logger.info(f"  Leakage: sampling {MAX_PAIRWISE}/{len(texts)} texts for pairwise comparison")
        else:
            sampled_indices = list(range(len(texts)))
            sampled_texts = texts

        ngram_sets = []
        for text in sampled_texts:
            words = text.lower().split()
            ngrams = set()
            for i in range(max(0, len(words) - self.ngram_size + 1)):
                ngram = tuple(words[i:i + self.ngram_size])
                ngrams.add(ngram)
            ngram_sets.append(ngrams)

        overlap_pairs = []
        flagged = set()

        for i in range(len(sampled_texts)):
            for j in range(i + 1, len(sampled_texts)):
                if not ngram_sets[i] or not ngram_sets[j]:
                    continue
                intersection = ngram_sets[i] & ngram_sets[j]
                union = ngram_sets[i] | ngram_sets[j]
                if not union:
                    continue
                jaccard = len(intersection) / len(union)
                if jaccard >= self.overlap_threshold:
                    overlap_pairs.append((sampled_indices[i], sampled_indices[j], jaccard))
                    flagged.add(sampled_indices[i])
                    flagged.add(sampled_indices[j])

        result.overlap_pairs = overlap_pairs
        result.flagged_indices = sorted(flagged)

        fragment_counts = self._find_repeated_fragments(texts)
        result.repeated_fragments = fragment_counts

        for frag in fragment_counts:
            for idx in frag.get('indices', []):
                flagged.add(idx)

        result.flagged_indices = sorted(flagged)
        result.stats['ngram_pairs_checked'] = len(texts) * (len(texts) - 1) // 2
        result.stats['overlapping_pairs'] = len(overlap_pairs)
        result.stats['repeated_fragments'] = len(fragment_counts)

        return result

    def _find_repeated_fragments(self, texts: List[str]) -> List[Dict]:
        """Find text fragments that appear in multiple samples."""
        fragment_indices: Dict[str, Set[int]] = defaultdict(set)

        for idx, text in enumerate(texts):
            words = text.lower().split()
            for frag_len in [5, 8, 10]:
                for i in range(len(words) - frag_len + 1):
                    fragment = ' '.join(words[i:i + frag_len])
                    fragment_indices[fragment].add(idx)

        repeated = []
        for fragment, indices in fragment_indices.items():
            if len(indices) >= self.min_fragment_count:
                repeated.append({
                    'fragment': fragment[:100],
                    'count': len(indices),
                    'indices': sorted(indices)
                })

        repeated.sort(key=lambda x: x['count'], reverse=True)
        return repeated[:100]


def detect_leakage(
    texts: List[str],
    ngram_size: int = 5,
    overlap_threshold: float = 0.5,
    min_fragment_count: int = 3
) -> Tuple[List[int], LeakageResult]:
    """
    Convenience function for leakage detection.

    Args:
        texts: List of text strings
        ngram_size: N-gram size
        overlap_threshold: Overlap threshold
        min_fragment_count: Min fragment occurrences

    Returns:
        Tuple of (flagged_indices, LeakageResult)
    """
    detector = LeakageDetector(
        ngram_size=ngram_size,
        overlap_threshold=overlap_threshold,
        min_fragment_count=min_fragment_count
    )
    result = detector.detect(texts)
    return result.flagged_indices, result
