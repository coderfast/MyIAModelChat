"""
Noise and quality filters for training data.

Detects and removes: URLs, emails, phone numbers, code blocks,
boilerplate text, corrupted formatting, excessive repetition,
and anomalous text lengths.
"""
import re
import unicodedata
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying filters to a batch of texts."""
    kept: List[str] = field(default_factory=list)
    discarded: List[Dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def total_input(self) -> int:
        return len(self.kept) + len(self.discarded)

    @property
    def retention_rate(self) -> float:
        total = self.total_input
        return len(self.kept) / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Filter result: {len(self.kept)}/{self.total_input} kept "
            f"({self.retention_rate:.1%}) | Discarded: {len(self.discarded)}"
        )


# Compiled regex patterns for noise detection
URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\'\)]+|'
    r'www\.[^\s<>\"\'\)]+|'
    r'\b[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|io|co)\b',
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
)

PHONE_PATTERN = re.compile(
    r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
)

SYSTEM_PATH_PATTERN = re.compile(
    r'([A-Z]:\\|/home/|/usr/|/var/|/etc/|~/|C:\\Users\\)',
    re.IGNORECASE
)

CODE_BLOCK_PATTERN = re.compile(
    r'("""|\'\'\'|```|def\s+\w+\s*\(|class\s+\w+|import\s+\w+|'
    r'function\s*\(|var\s+\w+\s*=|document\.|window\.|addEventListener|'
    r'<\?php|<html|<div|<script)',
    re.IGNORECASE
)

# Boilerplate patterns (navigation, footer, cookie banners)
BOILERPLATE_PATTERNS = [
    re.compile(r'(cookie|cookies)\s*(policy|notice|consent|accept)', re.IGNORECASE),
    re.compile(r'(privacy|terms)\s*(policy|of use|and conditions)', re.IGNORECASE),
    re.compile(r'all rights reserved', re.IGNORECASE),
    re.compile(r'copyright\s*©?\s*\d{4}', re.IGNORECASE),
    re.compile(r'(subscribe|unsubscribe)\s*(to|our|newsletter)', re.IGNORECASE),
    re.compile(r'(sign up|log in|sign in|login|register)\s*(now|here|account)', re.IGNORECASE),
    re.compile(r'(click here|read more|learn more|find out more)', re.IGNORECASE),
    re.compile(r'(follow us|share|tweet|pin|like)\s*(on|us)', re.IGNORECASE),
    re.compile(r'(advertisement|sponsored|promoted|paid partnership)', re.IGNORECASE),
    re.compile(r'(loading\.\.\.|please wait|page not found|404|500)', re.IGNORECASE),
]

# Encoding corruption patterns
CORRUPTION_PATTERNS = [
    re.compile(r'[\ufffd]{2,}'),  # Multiple replacement characters
    re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]'),  # Control characters
    re.compile(r'Ã[£Â§Ã]|Ã©|Ã¡|Ã±|Ã­|Ã³'),  # Common mojibake
]


class NoiseFilter:
    """Detects and removes noisy/contaminated text samples."""

    def __init__(self, categories: List[str] = None):
        """
        Initialize noise filter.

        Args:
            categories: List of categories to filter.
                        Options: urls, emails, phones, paths, code, boilerplate, corruption
                        Default: all categories
        """
        self.all_categories = {'urls', 'emails', 'phones', 'paths', 'code', 'boilerplate', 'corruption'}
        if categories is None:
            self.categories = self.all_categories
        else:
            self.categories = set(categories) & self.all_categories

    def filter_batch(self, texts: List[str]) -> FilterResult:
        """
        Filter a batch of texts for noise.

        Args:
            texts: List of text strings to filter

        Returns:
            FilterResult with kept texts and discarded reasons
        """
        result = FilterResult()

        for text in texts:
            if not text or not isinstance(text, str):
                result.discarded.append({'text': text or '', 'reason': 'empty'})
                result.stats['empty'] = result.stats.get('empty', 0) + 1
                continue

            reason = self._detect_noise(text)
            if reason:
                result.discarded.append({'text': text[:100], 'reason': reason})
                result.stats[reason] = result.stats.get(reason, 0) + 1
            else:
                result.kept.append(text)

        return result

    def _detect_noise(self, text: str) -> str:
        """Detect the primary noise type in text. Returns category or empty string."""
        text_normalized = unicodedata.normalize('NFKC', text)

        if 'urls' in self.categories and URL_PATTERN.search(text_normalized):
            url_matches = URL_PATTERN.findall(text_normalized)
            if len(url_matches) > 2 or len(text_normalized) < 50:
                return 'urls'

        if 'emails' in self.categories and EMAIL_PATTERN.search(text_normalized):
            return 'emails'

        if 'phones' in self.categories and PHONE_PATTERN.search(text_normalized):
            phone_matches = PHONE_PATTERN.findall(text_normalized)
            if len(phone_matches) > 1:
                return 'phones'

        if 'paths' in self.categories and SYSTEM_PATH_PATTERN.search(text_normalized):
            path_matches = SYSTEM_PATH_PATTERN.findall(text_normalized)
            if len(path_matches) > 1 or len(text_normalized) < 80:
                return 'paths'

        if 'code' in self.categories and CODE_BLOCK_PATTERN.search(text_normalized):
            code_matches = CODE_BLOCK_PATTERN.findall(text_normalized)
            if len(code_matches) > 2:
                return 'code'

        if 'boilerplate' in self.categories:
            boilerplate_count = sum(1 for p in BOILERPLATE_PATTERNS if p.search(text_normalized))
            if boilerplate_count >= 2:
                return 'boilerplate'

        if 'corruption' in self.categories:
            for pattern in CORRUPTION_PATTERNS:
                if pattern.search(text_normalized):
                    return 'corruption'

        return ''


class QualityFilter:
    """Extended quality filtering beyond basic length checks."""

    MIN_ALPHA_RATIO = 0.5
    MAX_REPEAT_RATIO = 0.3
    MIN_WORDS = 3
    MAX_WORDS = 1500
    MAX_PERIOD_RATIO = 0.5

    def filter_batch(self, texts: List[str]) -> FilterResult:
        """
        Filter a batch of texts for quality.

        Args:
            texts: List of text strings to filter

        Returns:
            FilterResult with kept texts and discarded reasons
        """
        result = FilterResult()

        for text in texts:
            if not text or not isinstance(text, str):
                result.discarded.append({'text': text or '', 'reason': 'empty'})
                continue

            reason = self._check_quality(text)
            if reason:
                result.discarded.append({'text': text[:100], 'reason': reason})
                result.stats[reason] = result.stats.get(reason, 0) + 1
            else:
                result.kept.append(text)

        return result

    def _check_quality(self, text: str) -> str:
        """Check text quality. Returns reason for discard or empty string."""
        words = text.split()
        word_count = len(words)

        if word_count < self.MIN_WORDS:
            return 'too_few_words'

        if word_count > self.MAX_WORDS:
            return 'too_many_words'

        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / len(text) if len(text) > 0 else 0
        if alpha_ratio < self.MIN_ALPHA_RATIO:
            return 'low_alpha_ratio'

        if text.count('.') > word_count * self.MAX_PERIOD_RATIO:
            return 'excessive_periods'

        unique_words = set(w.lower() for w in words)
        if len(unique_words) < word_count * 0.3:
            return 'low_diversity'

        if len(set(text)) < len(text) * 0.05:
            return 'repeated_chars'

        repeated_pattern = self._detect_repeated_phrases(text)
        if repeated_pattern:
            return 'repeated_phrases'

        return ''

    def _detect_repeated_phrases(self, text: str) -> bool:
        """Detect if text contains excessively repeated phrases."""
        words = text.lower().split()
        if len(words) < 6:
            return False

        for phrase_len in [3, 4, 5]:
            phrases = {}
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                phrases[phrase] = phrases.get(phrase, 0) + 1

            for phrase, count in phrases.items():
                if count > 2 and count / len(words) > self.MAX_REPEAT_RATIO:
                    return True

        return False


def apply_noise_filter(texts: List[str], categories: List[str] = None) -> Tuple[List[str], FilterResult]:
    """Apply noise filter to texts. Returns (filtered_texts, result)."""
    f = NoiseFilter(categories=categories)
    result = f.filter_batch(texts)
    return result.kept, result


def apply_quality_filter(texts: List[str]) -> Tuple[List[str], FilterResult]:
    """Apply quality filter to texts. Returns (filtered_texts, result)."""
    f = QualityFilter()
    result = f.filter_batch(texts)
    return result.kept, result
