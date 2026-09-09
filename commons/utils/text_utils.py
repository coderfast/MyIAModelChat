"""
Shared text processing utilities for data preparation.

Provides clean_text, split_sentences, split_paragraphs, chunk_text_by_tokens,
clean_page_artifacts, detect_language, filter_by_language, and mojibake repair.
Used by data_preparer.py and all source-to-markdown scripts.
"""

import re
import unicodedata
import logging
from typing import List
from collections import Counter

logger = logging.getLogger(__name__)

# ============================================================================
# Unicode and Mojibake Repair
# ============================================================================

# UTF-8 multi-byte sequences (above ASCII 0x7F) appear as Latin-1 chars
# when mis-decoded. Detect any char in the 0x80-0xFF range (Latin-1 extended).
_MOJIBAKE_DETECT = re.compile(r'[\x80-\xff]')

# Encodings to try, ordered by likelihood for European web content.
_MOJIBAKE_ENCODINGS = [
    'latin-1', 'cp1252',
    'cp1250', 'cp1251', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258',
    'iso8859-2', 'iso8859-3', 'iso8859-4', 'iso8859-5', 'iso8859-6', 'iso8859-7',
    'iso8859-8', 'iso8859-9', 'iso8859-10', 'iso8859-11', 'iso8859-13', 'iso8859-14',
    'iso8859-15', 'iso8859-16',
    'koi8-r', 'koi8-u',
    'macroman', 'maccyrillic', 'macgreek', 'macturkish', 'maciceland', 'maccentraleurope',
]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFKC form for consistency."""
    return unicodedata.normalize('NFKC', text)


def _decode_mojibake_buf(buf):
    """Try to decode a buffer of high-byte chars as mojibake."""
    s = ''.join(buf)
    for encoding in _MOJIBAKE_ENCODINGS:
        try:
            return s.encode(encoding, errors='strict').decode('utf-8', errors='strict')
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return s


def _repair_mojibake(text: str) -> str:
    """Repair mojibake caused by UTF-8 bytes decoded as a single-byte encoding."""
    if not text or not isinstance(text, str):
        return text
    if not _MOJIBAKE_DETECT.search(text):
        return text

    for encoding in _MOJIBAKE_ENCODINGS:
        try:
            repaired = text.encode(encoding, errors='strict').decode('utf-8', errors='strict')
            return repaired
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue

    result = []
    buf = []
    for ch in text:
        if ord(ch) > 0x7F:
            buf.append(ch)
        else:
            if buf:
                result.append(_decode_mojibake_buf(buf))
                buf = []
            result.append(ch)
    if buf:
        result.append(_decode_mojibake_buf(buf))
    return ''.join(result)


# ============================================================================
# Text Cleaning
# ============================================================================

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace, repairing encoding, and normalizing."""
    if not text or not isinstance(text, str):
        return ""
    text = _repair_mojibake(text)
    text = normalize_unicode(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# ============================================================================
# Sentence and Paragraph Splitting
# ============================================================================

# spaCy model singleton cache
_spacy_nlp = None
SPACY_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy if available, fallback to regex."""
    if not text or not isinstance(text, str):
        return []
    text = clean_text(text)
    if not text:
        return []

    if SPACY_AVAILABLE:
        try:
            global _spacy_nlp
            if _spacy_nlp is None:
                try:
                    _spacy_nlp = spacy.load('es_core_news_sm')
                except OSError:
                    try:
                        _spacy_nlp = spacy.load('en_core_web_sm')
                    except OSError:
                        _spacy_nlp = spacy.blank("en")
                        _spacy_nlp.add_pipe("sentencizer")
            doc = _spacy_nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences
        except Exception:
            pass

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C1\u00C9\u00CD\u00D3\u00DA\u00D1])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """Split text into paragraphs by blank lines, falling back to grouped sentences."""
    if not text or not isinstance(text, str):
        return []
    text = normalize_unicode(text)
    if not text.strip():
        return []

    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [clean_text(p) for p in raw_paragraphs]
    paragraphs = [p for p in paragraphs if p]

    if len(paragraphs) <= 1:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            paragraphs = sentences if sentences else []
        else:
            paragraphs = []
            for i in range(0, len(sentences), 3):
                grouped = ' '.join(sentences[i:i + 3])
                if grouped.strip():
                    paragraphs.append(grouped)

    return [p for p in paragraphs if len(p) >= min_length]


# ============================================================================
# Text Chunking
# ============================================================================

def chunk_text_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count (approximation of tokens)."""
    if not text or not isinstance(text, str):
        return []
    words = text.split()
    if len(words) <= max_tokens:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        step = max(1, max_tokens - overlap_tokens)
        start += step
        if start >= len(words):
            break
    return chunks


# ============================================================================
# Page Artifact Removal (PDF/EPUB)
# ============================================================================

PAGE_NUMBER_ONLY = re.compile(r'^\s*\d{1,4}\s*$')
PAGE_REF_LINE = re.compile(
    r'^\s*(?:p(?:agina|ag)?\.?\s*\d+|page\s*\d+|'
    r'p\.\s*\d+\s*(?:de|of|/)\s*\d+|\d+\s*[-–/]\s*\d+)\s*$',
    re.IGNORECASE
)


def clean_page_artifacts(page_texts: List[str], min_repeat_pages: int = 2) -> List[str]:
    """Remove headers, footers and page numbers from extracted page texts."""
    if not page_texts:
        return page_texts

    header_candidates = Counter()
    footer_candidates = Counter()

    for page_text in page_texts:
        lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
        if not lines:
            continue
        for ln in lines[:3]:
            header_candidates[ln] += 1
        for ln in lines[-3:]:
            footer_candidates[ln] += 1

    repeated_headers = {ln for ln, count in header_candidates.items()
                        if count >= min_repeat_pages and len(ln) <= 60}
    repeated_footers = {ln for ln, count in footer_candidates.items()
                        if count >= min_repeat_pages and len(ln) <= 60}

    cleaned = []
    for page_text in page_texts:
        lines = page_text.split('\n')
        out_lines = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                out_lines.append('')
                continue
            if PAGE_NUMBER_ONLY.match(stripped) or PAGE_REF_LINE.match(stripped):
                continue
            if stripped in repeated_headers or stripped in repeated_footers:
                continue
            out_lines.append(stripped)
        cleaned.append('\n'.join(out_lines))

    return cleaned


# ============================================================================
# Quality Filtering
# ============================================================================

def filter_by_quality(texts: List[str], min_words: int = 5, max_words: int = 1000,
                      min_alpha_ratio: float = 0.5) -> List[str]:
    """Filter texts by quality metrics."""
    if not texts:
        return []
    filtered = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        words = text.split()
        if len(words) < min_words or len(words) > max_words:
            continue
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / len(text) if len(text) > 0 else 0
        if alpha_ratio < min_alpha_ratio:
            continue
        if len(set(text)) < len(text) * 0.1:
            continue
        if text.count('.') > len(words) * 0.5:
            continue
        filtered.append(text)
    return filtered


# ============================================================================
# Language Detection and Filtering
# ============================================================================

def detect_language(text: str) -> str:
    """Detect the language of a text using commons.language_utils."""
    try:
        from commons.language_utils import detect_language as _detect
        return _detect(text)
    except ImportError:
        return 'unknown'


def filter_by_language(texts: List[str], allowed_languages: List[str]) -> List[str]:
    """Filter texts by language."""
    if not texts or not allowed_languages:
        return texts
    filtered = []
    for text in texts:
        lang = detect_language(text)
        if lang in allowed_languages or lang == 'unknown':
            filtered.append(text)
    return filtered


# ============================================================================
# Content Extraction for Contamination Pipeline
# ============================================================================

def extract_content_for_contamination(text: str) -> str:
    """Extract raw text from special-tagged markdown for contamination filtering.

    Removes all GPT-2 special tokens and thinking blocks, keeping only
    the actual content. Used by contamination filters to avoid false positives.

    Args:
        text: Markdown text with GPT-2 special tags

    Returns:
        Clean text without any special tags
    """
    if not text or not isinstance(text, str):
        return ''
    clean = re.sub(r'<\|problem\|>', '', text)
    clean = re.sub(r'<\|thinking\|>', '', clean)
    clean = re.sub(r'<\|final\|>', ' ', clean)
    clean = re.sub(r'<\|user\|>', '', clean)
    clean = re.sub(r'<\|assistant\|>', '', clean)
    clean = re.sub(r'<\|end\|>', ' ', clean)
    clean = re.sub(r'<\|sep\|>', ' ', clean)
    clean = re.sub(r'<\|system\|>', '', clean)
    clean = re.sub(r'<tool_call>.*?</tool_call>', '', clean, flags=re.DOTALL)
    clean = re.sub(r'<\|tool_result\|>', '', clean)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


# ============================================================================
# MinHash Deduplication (simple, for backward compatibility)
# ============================================================================

def deduplicate_texts_simple(texts: List[str], threshold: float = 0.8) -> List[str]:
    """Remove duplicate or near-duplicate texts using MinHash LSH or exact match fallback."""
    if not texts:
        return []

    try:
        from datasketch import MinHash, MinHashLSH
        HAS_DATASKETCH = True
    except ImportError:
        HAS_DATASKETCH = False

    if not HAS_DATASKETCH:
        seen = set()
        unique_texts = []
        for text in texts:
            text_normalized = text.strip().lower()
            if text_normalized not in seen:
                seen.add(text_normalized)
                unique_texts.append(text)
        return unique_texts

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}

    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        words = text.split()
        for idx in range(len(words) - 2):
            shingle = ' '.join(words[idx:idx+3])
            m.update(shingle.encode('utf-8'))
        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass

    duplicates = set()
    for i, m in minhashes.items():
        if i in duplicates:
            continue
        result = lsh.query(m)
        for j in result:
            j_int = int(j)
            if j_int != i:
                duplicates.add(j_int)

    unique_texts = [texts[i] for i in range(len(texts)) if i not in duplicates]
    return unique_texts
