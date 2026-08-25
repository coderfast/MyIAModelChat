"""
Cross-source deduplication module.

Provides exact, near-duplicate (MinHash LSH), and cross-source
deduplication with source preservation.
"""
import re
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of deduplication."""
    unique_texts: List[str] = field(default_factory=list)
    unique_with_source: List[Dict] = field(default_factory=list)
    duplicates_removed: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    cross_source_duplicates: int = 0
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def total_input(self) -> int:
        return len(self.unique_texts) + self.duplicates_removed

    @property
    def retention_rate(self) -> float:
        total = self.total_input
        return len(self.unique_texts) / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Dedup: {len(self.unique_texts)}/{self.total_input} unique "
            f"({self.retention_rate:.1%}) | Exact: {self.exact_duplicates} | "
            f"Near: {self.near_duplicates} | Cross-source: {self.cross_source_duplicates}"
        )


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


class CrossSourceDeduplicator:
    """Deduplication with source tracking."""

    def __init__(self, mode: str = 'all', near_threshold: float = 0.8):
        """
        Initialize deduplicator.

        Args:
            mode: 'exact', 'near', 'cross', or 'all'
            near_threshold: Similarity threshold for near-dedup (0-1)
        """
        self.mode = mode
        self.near_threshold = near_threshold

    def deduplicate(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None
    ) -> DedupResult:
        """
        Deduplicate texts with optional source tracking.

        Args:
            texts: List of text strings
            sources: Optional list of source names (same length as texts)

        Returns:
            DedupResult with deduplication results
        """
        result = DedupResult()

        if not texts:
            return result

        has_sources = sources is not None and len(sources) == len(texts)

        if self.mode in ('exact', 'all'):
            result = self._exact_dedup(texts, sources if has_sources else None)
        else:
            result.unique_texts = list(texts)
            if has_sources:
                result.unique_with_source = [
                    {'text': t, 'source': s} for t, s in zip(texts, sources)
                ]

        if self.mode in ('near', 'cross', 'all') and len(result.unique_texts) > 1:
            result = self._near_dedup(result, sources if has_sources else None)

        if self.mode in ('cross', 'all') and has_sources:
            result = self._cross_source_dedup(result)

        return result

    def _exact_dedup(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None
    ) -> DedupResult:
        """Remove exact duplicates (normalized)."""
        result = DedupResult()
        seen: Set[str] = set()

        for i, text in enumerate(texts):
            normalized = _normalize_text(text)
            if normalized not in seen:
                seen.add(normalized)
                result.unique_texts.append(text)
                if sources:
                    result.unique_with_source.append({
                        'text': text,
                        'source': sources[i]
                    })
            else:
                result.exact_duplicates += 1
                result.duplicates_removed += 1

        return result

    def _near_dedup(
        self,
        input_result: DedupResult,
        sources: Optional[List[str]] = None
    ) -> DedupResult:
        """Remove near-duplicates using MinHash LSH."""
        texts = input_result.unique_texts
        result = DedupResult()

        try:
            from datasketch import MinHash, MinHashLSH
            has_datasketch = True
        except ImportError:
            has_datasketch = False

        if not has_datasketch:
            logger.warning("datasketch not installed, using exact dedup only")
            result.unique_texts = texts
            result.unique_with_source = input_result.unique_with_source
            result.exact_duplicates = input_result.exact_duplicates
            result.duplicates_removed = input_result.duplicates_removed
            return result

        lsh = MinHashLSH(threshold=self.near_threshold, num_perm=128)
        minhashes = {}

        for i, text in enumerate(texts):
            m = MinHash(num_perm=128)
            words = text.split()
            if len(words) < 3:
                minhashes[i] = None
                continue
            for idx in range(len(words) - 2):
                shingle = ' '.join(words[idx:idx + 3])
                m.update(shingle.encode('utf-8'))
            minhashes[i] = m
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass

        duplicates: Set[int] = set()
        for i, m in minhashes.items():
            if i in duplicates or m is None:
                continue
            result_set = lsh.query(m)
            for j in result_set:
                j_int = int(j)
                if j_int != i:
                    duplicates.add(j_int)

        for i in range(len(texts)):
            if i not in duplicates:
                result.unique_texts.append(texts[i])
                if input_result.unique_with_source and i < len(input_result.unique_with_source):
                    result.unique_with_source.append(input_result.unique_with_source[i])

        result.near_duplicates = len(duplicates)
        result.duplicates_removed = input_result.duplicates_removed + len(duplicates)
        result.exact_duplicates = input_result.exact_duplicates

        return result

    def _cross_source_dedup(self, input_result: DedupResult) -> DedupResult:
        """Remove duplicates that appear across different sources."""
        result = DedupResult()
        text_to_sources: Dict[str, Set[str]] = defaultdict(set)
        text_to_original: Dict[str, str] = {}

        for item in input_result.unique_with_source:
            normalized = _normalize_text(item['text'])
            text_to_sources[normalized].add(item['source'])
            if normalized not in text_to_original:
                text_to_original[normalized] = item['text']

        cross_count = 0
        for normalized, source_set in text_to_sources.items():
            if len(source_set) > 1:
                cross_count += 1

        seen: Set[str] = set()
        for item in input_result.unique_with_source:
            normalized = _normalize_text(item['text'])
            if normalized not in seen:
                seen.add(normalized)
                result.unique_texts.append(item['text'])
                result.unique_with_source.append(item)

        result.cross_source_duplicates = cross_count
        result.duplicates_removed = input_result.duplicates_removed + cross_count
        result.exact_duplicates = input_result.exact_duplicates
        result.near_duplicates = input_result.near_duplicates

        return result


def deduplicate_texts(
    texts: List[str],
    sources: Optional[List[str]] = None,
    mode: str = 'all',
    threshold: float = 0.8
) -> Tuple[List[str], DedupResult]:
    """
    Convenience function for deduplication.

    Args:
        texts: List of text strings
        sources: Optional source names
        mode: 'exact', 'near', 'cross', or 'all'
        threshold: Near-duplicate threshold

    Returns:
        Tuple of (deduplicated_texts, DedupResult)
    """
    dedup = CrossSourceDeduplicator(mode=mode, near_threshold=threshold)
    result = dedup.deduplicate(texts, sources)
    return result.unique_texts, result
