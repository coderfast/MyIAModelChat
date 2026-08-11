"""
Data Preparation Module for MyIAModelChat

Handles loading, validating, and preparing datasets from multiple sources:
- AIML files
- Hugging Face datasets
- Custom local datasets

Provides statistics, validation, and dataset caching for efficient reuse.
"""

import os
import pickle
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataset_preparer.aiml.loader import AIMLLoader
from datasets import load_dataset, concatenate_datasets, Dataset
import sys
import multiprocessing as mp
from config import OLLAMA_MODEL

try:
    import psutil
except ImportError:
    psutil = None
try:
    import sentencepiece as spm
except Exception:
    spm = None
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None
try:
    from dataset_preparer.web.scraper import scrape_web_docs, load_urls_from_file, WEB_SCRAPER_DIR
except ImportError:
    scrape_web_docs = None
    load_urls_from_file = None
    WEB_SCRAPER_DIR = os.path.join('datasets_source', 'web')

# Contamination filtering module
try:
    from dataset_preparer.contamination.filters import NoiseFilter, QualityFilter, apply_noise_filter, apply_quality_filter
    from dataset_preparer.contamination.dedup import CrossSourceDeduplicator, deduplicate_texts as cross_deduplicate
    from dataset_preparer.contamination.balance import SourceBalancer, balance_sources
    from dataset_preparer.contamination.leakage import LeakageDetector, detect_leakage
    from dataset_preparer.contamination.audit import AuditReporter, SourceReport
    CONTAMINATION_AVAILABLE = True
except ImportError:
    CONTAMINATION_AVAILABLE = False

# NLP libraries for text processing
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import unicodedata
import re

# spaCy model singleton cache
_spacy_nlp = None


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFKC form for consistency."""
    return unicodedata.normalize('NFKC', text)


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing."""
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize Unicode first
    text = normalize_unicode(text)
    
    # Remove multiple whitespace characters
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy if available, fallback to regex-based splitting.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []

    # Clean text first
    text = clean_text(text)

    if not text:
        return []

    # Try spaCy first (most accurate)
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

    # Fallback: improved regex-based sentence splitting
    # Split on sentence-ending punctuation followed by space or end of string
    # Handles: Dr. Smith, 3.14, URLs, etc.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z¡¿])', text)

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def chunk_text_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by token count.
    Useful for models with fixed context windows.

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (default 512)
        overlap_tokens: Number of overlapping tokens between chunks (default 50)

    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        return []

    # Simple word-based tokenization (approximation)
    # For production, use actual tokenizer
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

        # Move start forward, accounting for overlap
        start += max_tokens - overlap_tokens

        # Safety check to avoid infinite loop
        if start >= len(words):
            break

    return chunks


def deduplicate_texts(texts: List[str], threshold: float = 0.8) -> List[str]:
    """
    Remove duplicate or near-duplicate texts using MinHash LSH.

    Args:
        texts: List of input texts
        threshold: Similarity threshold for deduplication (0-1, default 0.8)

    Returns:
        List of deduplicated texts
    """
    if not texts:
        return []

    try:
        from datasketch import MinHash, MinHashLSH
        HAS_DATASKETCH = True
    except ImportError:
        HAS_DATASKETCH = False

    if not HAS_DATASKETCH:
        # Fallback: simple exact match deduplication
        seen = set()
        unique_texts = []
        for text in texts:
            text_normalized = text.strip().lower()
            if text_normalized not in seen:
                seen.add(text_normalized)
                unique_texts.append(text)
        return unique_texts

    # MinHash LSH deduplication
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}

    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        # Create shingles (3-grams)
        words = text.split()
        for idx in range(len(words) - 2):
            shingle = ' '.join(words[idx:idx+3])
            m.update(shingle.encode('utf-8'))

        minhashes[i] = m
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # Duplicate signature, skip

    # Query for duplicates
    duplicates = set()
    for i, m in minhashes.items():
        if i in duplicates:
            continue
        result = lsh.query(m)
        for j in result:
            j_int = int(j)
            if j_int != i:
                duplicates.add(j_int)

    # Keep non-duplicate texts
    unique_texts = [texts[i] for i in range(len(texts)) if i not in duplicates]
    return unique_texts


def filter_by_quality(texts: List[str], min_words: int = 5, max_words: int = 1000,
                      min_alpha_ratio: float = 0.5) -> List[str]:
    """
    Filter texts by quality metrics.

    Args:
        texts: List of input texts
        min_words: Minimum number of words (default 5)
        max_words: Maximum number of words (default 1000)
        min_alpha_ratio: Minimum ratio of alphabetic characters (default 0.5)

    Returns:
        List of filtered texts
    """
    if not texts:
        return []

    filtered = []

    for text in texts:
        if not text or not isinstance(text, str):
            continue

        text = text.strip()

        # Check word count
        words = text.split()
        if len(words) < min_words or len(words) > max_words:
            continue

        # Check alpha ratio (filter out texts with too many numbers/symbols)
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / len(text) if len(text) > 0 else 0
        if alpha_ratio < min_alpha_ratio:
            continue

        # Check for repeated characters (spam/noise detection)
        if len(set(text)) < len(text) * 0.1:
            continue

        # Check for reasonable sentence structure
        if text.count('.') > len(words) * 0.5:  # Too many periods
            continue

        filtered.append(text)

    return filtered


def extract_pdf_metadata(pdf_path: str) -> Dict:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with metadata (title, author, pages, etc.)
    """
    metadata = {
        'source_type': 'pdf',
        'source_file': os.path.basename(pdf_path),
        'title': '',
        'author': '',
        'pages': 0,
        'language': 'unknown'
    }

    if PdfReader is None:
        return metadata

    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)

            # Get page count
            metadata['pages'] = len(pdf_reader.pages)

            # Get document info
            if pdf_reader.metadata:
                metadata['title'] = pdf_reader.metadata.title or ''
                metadata['author'] = pdf_reader.metadata.author or ''

    except Exception:
        pass

    return metadata


def extract_epub_metadata(epub_path: str) -> Dict:
    """
    Extract metadata from an EPUB file.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        Dictionary with metadata (title, author, language, chapters, etc.)
    """
    metadata = {
        'source_type': 'epub',
        'source_file': os.path.basename(epub_path),
        'title': '',
        'author': '',
        'language': 'unknown',
        'chapters': 0
    }

    if epub is None:
        return metadata

    try:
        book = epub.read_epub(epub_path)

        # Get metadata
        try:
            metadata['title'] = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else ''
        except Exception:
            pass

        try:
            metadata['author'] = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else ''
        except Exception:
            pass

        try:
            metadata['language'] = book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else 'unknown'
        except Exception:
            pass

        # Count chapters
        chapter_count = 0
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_count += 1
        metadata['chapters'] = chapter_count

    except Exception:
        pass

    return metadata


def detect_language(text: str) -> str:
    """
    Detect the language of a text.

    Args:
        text: Input text

    Returns:
        Language code (e.g., 'es', 'en', 'fr') or 'unknown'
    """
    if not text or not isinstance(text, str):
        return 'unknown'

    try:
        from langdetect import detect, LangDetectException
        HAS_LANGDETECT = True
    except ImportError:
        HAS_LANGDETECT = False

    if not HAS_LANGDETECT:
        # Simple heuristic-based language detection
        text_lower = text.lower()

        # Spanish indicators
        spanish_indicators = [' el ', ' la ', ' los ', ' las ', ' de ', ' del ', ' en ', ' un ', ' una ']
        spanish_count = sum(1 for ind in spanish_indicators if ind in text_lower)

        # English indicators
        english_indicators = [' the ', ' is ', ' are ', ' was ', ' were ', ' have ', ' has ', ' had ']
        english_count = sum(1 for ind in english_indicators if ind in text_lower)

        if spanish_count > english_count:
            return 'es'
        elif english_count > spanish_count:
            return 'en'
        else:
            return 'unknown'

    try:
        # Use first 1000 chars for detection (faster and more accurate)
        sample = text[:1000]
        lang = detect(sample)
        return lang
    except LangDetectException:
        return 'unknown'
    except Exception:
        return 'unknown'


def filter_by_language(texts: List[str], allowed_languages: List[str]) -> List[str]:
    """
    Filter texts by language.

    Args:
        texts: List of input texts
        allowed_languages: List of allowed language codes (e.g., ['es', 'en'])

    Returns:
        List of texts in allowed languages
    """
    if not texts or not allowed_languages:
        return texts

    filtered = []
    for text in texts:
        lang = detect_language(text)
        if lang in allowed_languages or lang == 'unknown':
            filtered.append(text)

    return filtered

# Setup logging (configured by main.py)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.pkl')
BPE_MODEL_PATH = os.path.join(CACHE_DIR, 'sentencepiece.model')



class DataPreparer:
    """Prepare and validate datasets for training with caching support."""
    
    def __init__(self, args):
        """
        Initialize DataPreparer.
        
        Args:
            args: Command line arguments with aiml, hf, pdf, epub, use_cache, refresh_cache flags
        """
        self.args = args
        self.aiml_data = None
        self.hf_data = None
        self.pdf_data = None
        self.epub_data = None
        self.web_data = None
        self.csv_data = None
        self.combined_data = None
        self.statistics = {}

        self.max_ram_fraction = getattr(args, 'max_ram_fraction', 0.75)
        self.max_ram_bytes = self._get_memory_limit_bytes(self.max_ram_fraction)
        self._ensure_cache_dir()
        
    def _get_memory_limit_bytes(self, max_ram_fraction: float):
        if psutil is None:
            logger.warning("psutil is not installed. Cannot enforce memory limit in data preparer.")
            return None
        total_bytes = psutil.virtual_memory().total
        limit_bytes = int(total_bytes * max_ram_fraction)
        logger.info(f"DataPreparer applying memory limit: {max_ram_fraction*100:.0f}% of {total_bytes/(1024**3):.2f} GB = {limit_bytes/(1024**3):.2f} GB")
        return limit_bytes

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            logger.debug(f"Created cache directory: {CACHE_DIR}")

    def _get_num_proc(self) -> int:
        """Calculate safe number of processes for Dataset.map().
        
        Returns:
            int: Number of processes to use (min 1, max 4)
        """
        if sys.version_info >= (3, 14):
            return 0  # dill incompatible
        else:
            try:
                cpus = mp.cpu_count()
                if cpus <= 2:
                    return 1
                return min(4, max(1, cpus // 2))
            except Exception:
                return 0

    def _cache_exists(self) -> bool:
        """Check if cached dataset exists."""
        return (os.path.exists(CACHE_DATASET_FILE) and 
                os.path.exists(CACHE_STATS_FILE))
    
    def _load_from_cache(self) -> Optional[Tuple[Dataset, Dict]]:
        """
        Load dataset from cache.
        
        Returns:
            Tuple of (dataset, statistics) or None if cache doesn't exist
        """
        try:
            if not self._cache_exists():
                return None
            
            logger.info("💾 Loading cached dataset...")
            
            # Load dataset
            self.combined_data = Dataset.load_from_disk(CACHE_DATASET_FILE)
            logger.info(f"  ✓ Loaded cached dataset: {len(self.combined_data)} samples")
            
            # Load statistics
            with open(CACHE_STATS_FILE, 'rb') as f:
                logger.warning("Loading pickle cache (untrusted data risk): %s", CACHE_STATS_FILE)
                self.statistics = pickle.load(f)
            logger.info(f"  ✓ Loaded cached statistics")
            
            logger.info("✅ DATASET LOADED FROM CACHE")
            return self.combined_data, self.statistics
            
        except Exception as e:
            logger.warning(f"  Could not load from cache: {e}")
            logger.info("  Will prepare fresh dataset...")
            return None
    
    def _save_to_cache(self):
        """Save prepared dataset to cache."""
        try:
            if self.combined_data is None:
                return
            
            logger.info("Saving dataset to cache...")
            
            # Save dataset
            self.combined_data.save_to_disk(CACHE_DATASET_FILE)
            logger.info(f"  Saved dataset: {CACHE_DATASET_FILE}")
            
            # Save statistics
            with open(CACHE_STATS_FILE, 'wb') as f:
                pickle.dump(self.statistics, f)
            logger.info(f"  Saved statistics: {CACHE_STATS_FILE}")
            # Save metadata (e.g., tokenizer info)
            metadata = getattr(self, 'cache_metadata', {})
            with open(CACHE_METADATA_FILE, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"  Saved cache metadata: {CACHE_METADATA_FILE}")

            # Export JSONL (GPT-2 standard format)
            self._export_jsonl()
            
            logger.info("  Cache ready for future runs")
            
        except Exception as e:
            logger.warning(f"  Could not save to cache: {e}")

    def _export_jsonl(self):
        """Export dataset as JSONL files (GPT-2 standard format)."""
        try:
            jsonl_dir = os.path.join(CACHE_DIR, 'jsonl')
            os.makedirs(jsonl_dir, exist_ok=True)

            if self.combined_data is None or len(self.combined_data) == 0:
                logger.warning("  No data to export as JSONL")
                return

            # Split into train/val/test (80/10/10)
            total = len(self.combined_data)
            train_end = int(total * 0.8)
            val_end = int(total * 0.9)

            splits = {
                'train': range(0, train_end),
                'val': range(train_end, val_end),
                'test': range(val_end, total),
            }

            for split_name, indices in splits.items():
                jsonl_path = os.path.join(jsonl_dir, f'{split_name}.jsonl')
                count = 0
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for i in indices:
                        sample = self.combined_data[i]
                        # GPT-2 standard: {"text": "<|problem|>...<|thinking|>...<|final|>..."}
                        text = sample.get('bpe_text', sample.get('input_ids', ''))
                        if isinstance(text, str) and text.strip():
                            import json
                            line = json.dumps({"text": text}, ensure_ascii=False)
                            f.write(line + '\n')
                            count += 1

                logger.info(f"  Exported {split_name}.jsonl: {count} samples")

        except Exception as e:
            logger.warning(f"  Could not export JSONL: {e}")
    
    def _clear_cache(self):
        """Clear cached dataset, preserving README.txt."""
        try:
            if os.path.exists(CACHE_DIR):
                for item in os.listdir(CACHE_DIR):
                    if item == 'README.txt':
                        continue
                    item_path = os.path.join(CACHE_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

                logger.info("Cache cleared (README.txt preserved)")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
    
    def prepare(self) -> Dict:
        """
        Prepare all datasets and collect statistics.
        
        Returns:
            Dictionary with preparation status and statistics
        """
        logger.info("=" * 80 + " DATASET PREPARATION STARTING " + "=" * 80)

        if self.max_ram_bytes is not None and psutil is not None:
            used = psutil.virtual_memory().used
            if used > self.max_ram_bytes:
                logger.warning(
                    f"Current memory usage ({used/(1024**3):.2f} GB) exceeds max allowed ({self.max_ram_bytes/(1024**3):.2f} GB). "
                    "Please reduce system load before preparation."
                )

        try:
            # Check for refresh cache flag
            if hasattr(self.args, 'refresh_cache') and self.args.refresh_cache:
                logger.info("🔄 Refreshing cache (clearing old cache)...")
                self._clear_cache()
            
            # Check for use cache flag
            if hasattr(self.args, 'use_cache') and self.args.use_cache and not (hasattr(self.args, 'refresh_cache') and self.args.refresh_cache):
                result = self._load_from_cache()
                if result:
                    self.combined_data, self.statistics = result
                    self._display_summary()
                    logger.info("=" * 80)
                    logger.info("[OK] DATASET READY (from cache)")
                    logger.info("=" * 80)
                    return {
                        'status': 'success',
                        'combined_dataset': self.combined_data,
                        'statistics': self.statistics,
                        'from_cache': True
                    }
            
            # Prepare fresh dataset
            logger.info("[1/7] Loading AIML data..." if (self.args.aiml or self.args.hf) else "[1/7] No data sources selected...")
            
            # Load AIML data
            if self.args.aiml:
                self.aiml_data = self._load_aiml_data()
            
            # Load Hugging Face data
            if self.args.hf:
                logger.info("[2/7] Loading Hugging Face datasets...")
                self.hf_data = self._load_hf_data()
            
            # Load PDF data
            if hasattr(self.args, 'pdf') and self.args.pdf:
                logger.info("[3/7] Loading PDF files...")
                self.pdf_data = self._load_pdf_data()
            
            # Load EPUB data
            if hasattr(self.args, 'epub') and self.args.epub:
                logger.info("[4/7] Loading EPUB files...")
                self.epub_data = self._load_epub_data()

            # Load Web documentation data
            if hasattr(self.args, 'web') and self.args.web:
                logger.info("[4.5/7] Scraping web documentation...")
                self.web_data = self._load_web_data()

            # Load CSV data (curated supplementary dataset)
            if hasattr(self.args, 'csv') and self.args.csv:
                logger.info("[5/7] Loading CSV data...")
                self.csv_data = self._load_csv()

            # Combine datasets
            logger.info("[6/7] Combining datasets...")
            self.combined_data = self._combine_datasets()
            self._standardize_combined_dataset()

            # Initialize audit reporter if enabled
            audit = None
            if CONTAMINATION_AVAILABLE and getattr(self.args, 'audit_report', False):
                audit_dir = getattr(self.args, 'audit_dir', None)
                audit = AuditReporter(report_dir=audit_dir, enabled=True)
                audit.start_timer()
                logger.info("  Audit reporting enabled")

            # Validate and clean sources (ANTES de thinking)
            validate_sources = getattr(self.args, 'validate_sources', False)
            if validate_sources:
                logger.info("[6.4/7] Validating and cleaning sources...")
                self._validate_and_clean_sources()

            # Apply noise filter (NEW)
            filter_noise = getattr(self.args, 'filter_noise', False)
            if filter_noise:
                logger.info("[6.42/7] Applying noise filter...")
                self._apply_noise_filter(audit=audit)

            # Apply extended quality filter (NEW)
            filter_quality = getattr(self.args, 'filter_contamination', False)
            if filter_quality:
                logger.info("[6.43/7] Applying quality filter...")
                self._apply_quality_filter_contamination(audit=audit)

            # Apply cross-source deduplication (NEW)
            filter_dedup = getattr(self.args, 'filter_dedup', False)
            if filter_dedup:
                logger.info("[6.44/7] Applying cross-source deduplication...")
                self._apply_cross_source_dedup(audit=audit)

            # Apply balance control (NEW)
            filter_balance = getattr(self.args, 'filter_balance', False)
            if filter_balance:
                logger.info("[6.45/7] Applying balance control...")
                self._apply_balance_control(audit=audit)

            # Generate thinking data for each source (DESPUES de validacion)
            generate_thinking = getattr(self.args, 'generate_thinking', False)
            if generate_thinking:
                logger.info("[6.5/7] Generating real thinking data...")
                self._generate_thinking_for_sources()
                # Recombine after thinking generation

            # Generate agentic data with tool calls (DESPUES de thinking)
            generate_agent = getattr(self.args, 'generate_agent_data', False)
            if generate_agent:
                logger.info("[6.55/7] Generating agentic data with tool calls...")
                self._generate_agent_data()


            # Apply deduplication if enabled
            enable_dedup = getattr(self.args, 'enable_dedup', False)
            if enable_dedup:
                logger.info("[6.6/7] Applying deduplication...")
                self._apply_deduplication()

            # Apply quality filtering if enabled
            enable_quality = getattr(self.args, 'enable_quality_filter', False)
            if enable_quality:
                logger.info("[6.7a/7] Applying quality filtering...")
                self._apply_quality_filter()

            # Apply language filtering if enabled
            enable_lang_filter = getattr(self.args, 'enable_lang_filter', False)
            if enable_lang_filter:
                logger.info("[6.7b/7] Applying language filtering...")
                self._apply_language_filter()

            # Apply leakage detection (NEW)
            filter_leakage = getattr(self.args, 'filter_leakage', False)
            if filter_leakage:
                logger.info("[6.72/7] Applying leakage detection...")
                self._apply_leakage_detection(audit=audit)

            # Collect statistics
            logger.info("Gathering statistics...")
            self.statistics = self._collect_statistics()

            # Save audit report if enabled
            if audit is not None:
                total_original = sum(
                    audit.report.per_source.get(s, SourceReport()).original_count
                    for s in audit.report.per_source
                )
                audit.set_final(
                    count=len(self.combined_data) if self.combined_data else 0,
                    discarded=total_original - (len(self.combined_data) if self.combined_data else 0)
                )
                audit.save()
                audit.print_summary()

            # Always train BPE tokenizer and tokenize data before caching
            bpe_vocab_size = getattr(self.args, 'bpe_vocab_size', 8000)
            try:
                self._prepare_bpe_tokenizer_and_tokenize(bpe_vocab_size)
            except Exception as e:
                logger.warning(f"  ⚠ BPE tokenization failed: {e}")
            
            # Save to cache
            self._save_to_cache()
            
            # Display summary
            self._display_summary()
            
            logger.info("=" * 80)
            logger.info("[OK] DATASET PREPARATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'combined_dataset': self.combined_data,
                'statistics': self.statistics,
                'from_cache': False
            }
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            logger.error("=" * 80)
            return {
                'status': 'failed',
                'error': str(e),
                'combined_dataset': None,
                'statistics': {},
                'from_cache': False
            }
    
    def _safe_load_dataset_pickle(self, file_path: str):
        """
        Load a pickled dataset in a backward-compatible way, handling old DatasetInfo fields such as task_templates.
        """
        try:
            from datasets import DatasetInfo
        except Exception:
            DatasetInfo = None

        orig_init = None
        if DatasetInfo is not None:
            orig_init = DatasetInfo.__init__

            def patched_init(self, *args, **kwargs):
                kwargs.pop('task_templates', None)
                return orig_init(self, *args, **kwargs)

            DatasetInfo.__init__ = patched_init

        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # If it is still failing with task_templates and still DatasetInfo is present, retry once more.
            if DatasetInfo is not None and 'task_templates' in str(e):
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
            raise
        finally:
            if DatasetInfo is not None and orig_init is not None:
                DatasetInfo.__init__ = orig_init

    def _load_aiml_data(self) -> Dataset:
        """
        Load AIML files from datasets_source/aiml directory.
        Uses the parser to process AIML files directly (ignores cached .datasets files).

        Returns:
            Hugging Face Dataset with AIML data
        """
        data_dir = os.path.join('datasets_source', 'aiml')
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        logger.info(f"  Loading AIML files from: {data_dir}")
        
        # Use the parser to process AIML files directly
        try:
            aiml_loader = AIMLLoader(data_dir)
            dataset = aiml_loader.load_aiml_files()
            
            if dataset is not None and len(dataset) > 0:
                logger.info(f"  ✓ Loaded {len(dataset)} AIML samples from parser")
                return dataset
            else:
                logger.warning(f"  ⚠ No samples extracted from AIML files")
                return Dataset.from_list([])
                
        except Exception as e:
            logger.warning(f"  ⚠ Error loading AIML files: {e}")
            return Dataset.from_list([])
    
    def _load_hf_data(self) -> Dataset:
        """
        Load Hugging Face datasets.
        
        Returns:
            Hugging Face Dataset with HF data
        """
        hf_datasets = []
        
        # Example HF datasets to load
        hf_dataset_names = [
            'wikitext',  # Wikipedia text (multilingual via different configs)
            # 'opus_100',  # Multi-language translation data (100+ languages)
            # 'common_voice',  # Speech data (multiple languages)
        ]
        
        for dataset_name in hf_dataset_names:
            try:
                logger.info(f"  Loading Hugging Face dataset: {dataset_name}...")
                
                if dataset_name == 'wikitext':
                    ds = load_dataset('wikitext', 'wikitext-2', split='train')
                    # Take subset for faster preparation
                    ds = ds.select(range(min(1000, len(ds))))
                elif dataset_name == 'common_voice':
                    ds = load_dataset('common_voice', '2024-08', split='train[:1000]', languages=["en", "es", "fr", "de", "pt", "it", "nl", "ru", "zh", "ja"])
                elif dataset_name == 'opus_100':
                    ds = load_dataset('opus_100', split='train[:1000]')
                else:
                    ds = load_dataset(dataset_name, split='train[:1000]')
                
                # Convert to input_ids format for consistency
                if 'text' in ds.column_names:
                    ds = ds.map(lambda x: {'input_ids': x['text']})
                elif 'sentence' in ds.column_names:
                    ds = ds.map(lambda x: {'input_ids': x['sentence']})
                
                # Keep only input_ids column
                ds = ds.select_columns(['input_ids'])
                
                hf_datasets.append(ds)
                logger.info(f"  ✓ Loaded: {dataset_name} ({len(ds)} samples)")
                
            except Exception as e:
                logger.warning(f"  ⚠ Error loading {dataset_name}: {e}")
                continue
        
        if hf_datasets:
            combined = concatenate_datasets(hf_datasets)
            logger.info(f"Total HuggingFace datasets loaded: {len(hf_datasets)}")
            logger.info(f"Total HuggingFace samples: {len(combined)}")
            return combined
        else:
            logger.warning("  ⚠ No HuggingFace datasets loaded successfully")
            return Dataset.from_list([])

    def _load_csv(self) -> Dataset:
        """
        Load curated supplemental data from all CSV files in datasets_source/csv/ directory.
        This keeps special knowledge in data, not hardcoded code paths.

        CSV Format:
            input,output
            "question","answer"

        Returns:
            Hugging Face Dataset with input_ids column
        """
        csv_dir = os.path.join('datasets_source', 'csv')

        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            logger.info(f"  Created directory: {csv_dir}")
            logger.info(f"  ℹ Place CSV files in '{csv_dir}' directory to load them")
            return Dataset.from_list([])

        csv_items = []
        csv_count = 0

        try:
            import csv as csv_mod

            for filename in os.listdir(csv_dir):
                if not filename.lower().endswith('.csv'):
                    continue

                csv_path = os.path.join(csv_dir, filename)
                try:
                    logger.info(f"  Reading CSV: {filename}...")

                    with open(csv_path, 'r', encoding='utf-8') as f:
                        # Detect if file has header
                        first_line = f.readline().strip()
                        f.seek(0)

                        has_header = first_line.lower().startswith('input')

                        reader = csv_mod.DictReader(f) if has_header else csv_mod.reader(f)

                        file_count = 0
                        for row in reader:
                            if has_header:
                                input_text = row.get('input', '').strip()
                                output_text = row.get('output', '').strip()
                            else:
                                if len(row) < 2:
                                    continue
                                input_text = row[0].strip().strip('"')
                                output_text = row[1].strip().strip('"')

                            # Validate and clean
                            input_text = clean_text(input_text)
                            output_text = clean_text(output_text)

                            if not input_text or not output_text:
                                continue

                            # Standardized format: question/answer pair
                            prompt_text = f"{input_text} {output_text}"

                            # Oversample with moderate repetition (5x)
                            for _ in range(5):
                                csv_items.append({'input_ids': prompt_text})
                                file_count += 1

                        csv_count += 1
                        logger.info(f"  ✓ Loaded: {filename} ({file_count} samples)")

                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")

            if csv_items:
                ds = Dataset.from_list(csv_items)
                logger.info(f"Total CSV files processed: {csv_count}")
                logger.info(f"Total CSV samples: {len(csv_items)}")
                return ds
            else:
                logger.warning(f"  ⚠ No CSV files found in '{csv_dir}' directory")
                return Dataset.from_list([])

        except Exception as e:
            logger.warning(f"  ⚠ Error reading CSV files: {e}")
            return Dataset.from_list([])

    def _load_web_data(self) -> Dataset:
        """
        Load text data by scraping documentation from web URLs.

        Reads URLs from --web-url flag or datasets_source/web/urls.txt.
        Saves scraped text to datasets_source/web/.

        Returns:
            Hugging Face Dataset with web-scraped text data
        """
        if scrape_web_docs is None:
            logger.warning("  ⚠ Web scraper not available. Install: pip install trafilatura beautifulsoup4 requests")
            return Dataset.from_list([])

        web_url = getattr(self.args, 'web_url', None)
        urls_file = os.path.join(WEB_SCRAPER_DIR, 'urls.txt')

        # Determine URLs to scrape
        urls_to_scrape = []
        if web_url:
            urls_to_scrape.append(web_url)
        elif os.path.exists(urls_file):
            urls_to_scrape = load_urls_from_file(urls_file)
            if urls_to_scrape:
                logger.info(f"  Loaded {len(urls_to_scrape)} URLs from {urls_file}")
        else:
            logger.warning("  ⚠ No --web-url specified and no datasets_source/web/urls.txt found")
            return Dataset.from_list([])

        max_pages = getattr(self.args, 'web_max_pages', 50)
        max_depth = getattr(self.args, 'web_max_depth', 3)

        all_texts = []
        for seed_url in urls_to_scrape:
            logger.info(f"  Scraping: {seed_url}")
            try:
                texts = scrape_web_docs(
                    url=seed_url,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    rate_limit=1.0,
                )
                all_texts.extend(texts)
            except ImportError as e:
                logger.warning(f"  ⚠ Web scraping dependencies missing: {e}")
                logger.warning("    Install with: pip install trafilatura beautifulsoup4 requests")
                return Dataset.from_list([])
            except Exception as e:
                logger.warning(f"  ⚠ Failed to scrape {seed_url}: {e}")
                continue

        if not all_texts:
            logger.warning(f"  ⚠ No text extracted from web URLs")
            return Dataset.from_list([])

        # Get chunking options
        enable_chunking = getattr(self.args, 'enable_chunking', False)
        max_tokens = getattr(self.args, 'chunk_max_tokens', 512)
        overlap_tokens = getattr(self.args, 'chunk_overlap', 50)

        # Process each scraped text into training samples
        web_items = []
        for text in all_texts:
            sentences = split_sentences(text)
            for sentence in sentences:
                sentence = clean_text(sentence)
                if len(sentence) > 10:
                    if enable_chunking and len(sentence.split()) > max_tokens:
                        chunks = chunk_text_by_tokens(sentence, max_tokens, overlap_tokens)
                        for chunk in chunks:
                            web_items.append({'input_ids': chunk})
                    else:
                        web_items.append({'input_ids': sentence})

        dataset = Dataset.from_list(web_items)
        logger.info(f"  Total web pages scraped: {len(all_texts)}")
        logger.info(f"  Total web samples: {len(dataset)}")

        # Save scraped text to output directory for inspection
        os.makedirs(WEB_SCRAPER_DIR, exist_ok=True)
        output_file = os.path.join(WEB_SCRAPER_DIR, 'scraped_text.txt')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in all_texts:
                    f.write(text + '\n\n---\n\n')
            logger.info(f"  Saved scraped text to {output_file}")
        except Exception as e:
            logger.warning(f"  ⚠ Could not save scraped text: {e}")

        return dataset

    def _load_pdf_data(self) -> Dataset:
        """
        Load text data from PDF files in datasets_source/pdf directory.

        Returns:
            Hugging Face Dataset with PDF text data
        """
        if PdfReader is None:
            logger.warning("  ⚠ PyPDF2 not installed. Install with: pip install PyPDF2")
            return Dataset.from_list([])

        pdf_dir = os.path.join('datasets_source', 'pdf')

        # Create pdfs directory if it doesn't exist
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            logger.info(f"  Created directory: {pdf_dir}")
            logger.info(f"  ℹ Place PDF files in '{pdf_dir}' directory to load them")
            return Dataset.from_list([])

        # Get options from args
        enable_chunking = getattr(self.args, 'enable_chunking', False)
        max_tokens = getattr(self.args, 'chunk_max_tokens', 512)
        overlap_tokens = getattr(self.args, 'chunk_overlap', 50)
        preserve_metadata = getattr(self.args, 'preserve_metadata', False)

        pdf_texts = []
        pdf_count = 0

        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                try:
                    logger.info(f"  Reading PDF: {filename}...")

                    # Extract metadata if enabled
                    metadata = {}
                    if preserve_metadata:
                        metadata = extract_pdf_metadata(file_path)

                    with open(file_path, 'rb') as f:
                        pdf_reader = PdfReader(f)
                        text_parts = []

                        # Extract text from all pages
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    text_parts.append(page_text)
                            except Exception as e:
                                logger.warning(f"    ⚠ Error extracting page {page_num} from {filename}: {e}")

                        text = " ".join(text_parts)

                        # Split text by sentences/paragraphs into samples
                        if text.strip():
                            # Use professional sentence tokenization
                            sentences = split_sentences(text)
                            sentence_count = 0

                            for sentence in sentences:
                                # Additional cleaning
                                sentence = clean_text(sentence)
                                if len(sentence) > 10:  # Skip very short sentences
                                    sample = {'input_ids': sentence}

                                    # Add metadata if enabled
                                    if preserve_metadata and metadata:
                                        sample['metadata'] = metadata

                                    # Apply chunking if enabled
                                    if enable_chunking and len(sentence.split()) > max_tokens:
                                        chunks = chunk_text_by_tokens(sentence, max_tokens, overlap_tokens)
                                        for chunk in chunks:
                                            chunk_sample = {'input_ids': chunk}
                                            if preserve_metadata and metadata:
                                                chunk_sample['metadata'] = metadata
                                            pdf_texts.append(chunk_sample)
                                            sentence_count += 1
                                    else:
                                        pdf_texts.append(sample)
                                        sentence_count += 1

                            pdf_count += 1
                            logger.info(f"  ✓ Loaded: {filename} ({sentence_count} samples)")
                        else:
                            logger.warning(f"  ⚠ No text extracted from {filename}")

                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")

        if pdf_texts:
            dataset = Dataset.from_list(pdf_texts)
            logger.info(f"Total PDF files processed: {pdf_count}")
            logger.info(f"Total PDF samples: {len(dataset)}")
            return dataset
        else:
            logger.warning(f"  ⚠ No PDF files found in '{pdf_dir}' directory")
            return Dataset.from_list([])
    
    def _load_epub_data(self) -> Dataset:
        """
        Load text data from EPUB files (e-books) in 'epub' directory.

        Returns:
            Hugging Face Dataset with EPUB text data
        """
        if epub is None:
            logger.warning("  ⚠ ebooklib not installed. Install with: pip install ebooklib")
            return Dataset.from_list([])

        epub_dir = os.path.join('datasets_source', 'epub')

        # Create epub directory if it doesn't exist
        if not os.path.exists(epub_dir):
            os.makedirs(epub_dir)
            logger.info(f"  Created directory: {epub_dir}")
            logger.info(f"  ℹ Place EPUB files in '{epub_dir}' directory to load them")
            return Dataset.from_list([])

        # Get options from args
        enable_chunking = getattr(self.args, 'enable_chunking', False)
        max_tokens = getattr(self.args, 'chunk_max_tokens', 512)
        overlap_tokens = getattr(self.args, 'chunk_overlap', 50)
        preserve_metadata = getattr(self.args, 'preserve_metadata', False)

        epub_texts = []
        epub_count = 0

        for filename in os.listdir(epub_dir):
            if filename.lower().endswith('.epub'):
                file_path = os.path.join(epub_dir, filename)
                try:
                    logger.info(f"  Reading EPUB: {filename}...")

                    # Extract metadata if enabled
                    metadata = {}
                    if preserve_metadata:
                        metadata = extract_epub_metadata(file_path)

                    # Open and parse EPUB
                    book = epub.read_epub(file_path)
                    text_parts = []

                    # Extract text from all chapters
                    for item in book.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            try:
                                # Get chapter content
                                content = item.get_content().decode('utf-8', errors='ignore')

                                # Preserve <thinking> tags before removing HTML
                                content = re.sub(r'<thinking>', '§THINKING_START§', content)
                                content = re.sub(r'</thinking>', '§THINKING_END§', content)

                                # Remove HTML tags (simple regex approach)
                                content = re.sub(r'<[^>]+>', '', content)

                                # Restore <thinking> tags
                                content = re.sub(r'§THINKING_START§', '<thinking>', content)
                                content = re.sub(r'§THINKING_END§', '</thinking>', content)

                                # Clean up whitespace
                                content = re.sub(r'\s+', ' ', content)
                                if content.strip():
                                    text_parts.append(content)
                            except Exception as e:
                                logger.warning(f"    ⚠ Error extracting chapter from {filename}: {e}")

                    text = " ".join(text_parts)

                    # Split text into samples
                    if text.strip():
                        # Use professional sentence tokenization
                        sentences = split_sentences(text)
                        sentence_count = 0

                        for sentence in sentences:
                            # Additional cleaning
                            sentence = clean_text(sentence)
                            if len(sentence) > 10:  # Skip very short sentences
                                sample = {'input_ids': sentence}

                                # Add metadata if enabled
                                if preserve_metadata and metadata:
                                    sample['metadata'] = metadata

                                # Apply chunking if enabled
                                if enable_chunking and len(sentence.split()) > max_tokens:
                                    chunks = chunk_text_by_tokens(sentence, max_tokens, overlap_tokens)
                                    for chunk in chunks:
                                        chunk_sample = {'input_ids': chunk}
                                        if preserve_metadata and metadata:
                                            chunk_sample['metadata'] = metadata
                                        epub_texts.append(chunk_sample)
                                        sentence_count += 1
                                else:
                                    epub_texts.append(sample)
                                    sentence_count += 1

                        epub_count += 1
                        logger.info(f"  ✓ Loaded: {filename} ({sentence_count} samples)")
                    else:
                        logger.warning(f"  ⚠ No text extracted from {filename}")

                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")

        if epub_texts:
            dataset = Dataset.from_list(epub_texts)
            logger.info(f"Total EPUB files processed: {epub_count}")
            logger.info(f"Total EPUB samples: {len(dataset)}")
            return dataset
        else:
            logger.warning(f"  ⚠ No EPUB files found in '{epub_dir}' directory")
            return Dataset.from_list([])

    def _prepare_bpe_tokenizer_and_tokenize(self, vocab_size: int = 8000):
        """
        Train or load a SentencePiece (BPE) tokenizer and apply it to the combined dataset.
        Adds a `token_ids` column with list[int] per sample and stores tokenizer model in cache.
        """
        if spm is None:
            logger.warning("sentencepiece not installed. Install with: pip install sentencepiece\nSkipping BPE tokenization.")
            return

        # Ensure we have a textual field to train BPE on (avoid numeric token ids)
        logger.info("[7/7] Preparing BPE tokenizer (SentencePiece)...")
        
        # Check if dataset contains thinking tokens
        try:
            sample = self.combined_data[0]
            input_text = sample.get('input_ids', '')
            if isinstance(input_text, str) and '<thinking>' in input_text:
                logger.info("  ✓ Dataset contains <thinking> tokens in input_ids - will use for BPE training")
            else:
                logger.info("  ℹ No thinking tokens detected in input_ids - using for BPE training")
        except Exception:
            pass

        sample = None
        try:
            sample = self.combined_data[0]
        except Exception:
            sample = None

        text_column = None
        if sample is not None:
            # Prefer existing textual 'input_ids' if it's a string
            val = sample.get('input_ids')
            if isinstance(val, str):
                text_column = 'input_ids'
            else:
                # Find any other string column to use
                for col in self.combined_data.column_names:
                    if col == 'input_ids':
                        continue
                    v = sample.get(col)
                    if isinstance(v, str):
                        text_column = col
                        break

        if text_column is None:
            logger.warning("  ⚠ No textual field found for BPE training (dataset may already be pre-tokenized). Skipping BPE.")
            return

        # Create a 'bpe_text' column copying the chosen text column to avoid overwriting
        if 'bpe_text' not in self.combined_data.column_names:
            def extract_text(example):
                return {'bpe_text': example.get(text_column, '') if example.get(text_column) is not None else ''}

            num_proc = self._get_num_proc()
            try:
                self.combined_data = self.combined_data.map(extract_text, batched=False, num_proc=num_proc)
            except Exception:
                self.combined_data = self.combined_data.map(extract_text, batched=False, num_proc=1)

        # Collect texts for tokenizer training (subset if too large)
        texts = []
        max_train_samples = 50000
        try:
            for i, item in enumerate(self.combined_data):
                text = item.get('bpe_text')
                if isinstance(text, str) and text.strip():
                    texts.append(text)
                if len(texts) >= max_train_samples:
                    break
        except Exception as e:
            logger.warning(f"  ⚠ Error collecting texts for tokenizer training: {e}")

        if not texts:
            logger.warning("  ⚠ No texts available for BPE tokenizer training after extraction. Skipping.")
            return

        # Prepare temporary file for sentencepiece training
        tmp_corpus = os.path.join(CACHE_DIR, 'sp_corpus.txt')
        with open(tmp_corpus, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t.replace('\n', ' ') + "\n")

        model_prefix = os.path.join(CACHE_DIR, 'sentencepiece')
        # GPT-2 standard special tokens
        user_symbols = '--user_defined_symbols=<|problem|>,<|thinking|>,<|final|>,<|user|>,<|assistant|>,<tool_call>,</tool_call>,<|tool_result|>'
        
        # Retry with decreasing vocab_size if training fails (e.g., corpus too small)
        current_vocab = vocab_size
        max_retries = 3
        for attempt in range(max_retries):
            spm_cmd = f"--input={tmp_corpus} --model_prefix={model_prefix} --vocab_size={current_vocab} --model_type=bpe --character_coverage=0.9995 {user_symbols}"
            logger.info(f"  Training SentencePiece BPE model (vocab_size={current_vocab})... this may take a while")
            try:
                spm.SentencePieceTrainer.Train(spm_cmd)
                break
            except Exception as sp_err:
                err_msg = str(sp_err)
                if 'Vocabulary size too high' in err_msg or 'vocab' in err_msg.lower():
                    current_vocab = max(256, current_vocab // 2)
                    logger.warning(f"  vocab_size too large for this corpus. Retrying with vocab_size={current_vocab}...")
                else:
                    raise

        model_file = model_prefix + '.model'
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SentencePiece model not found after training: {model_file}")

        # Load the trained model
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)

        # Tokenize dataset to token ids
        def tokenize_example(example):
            # Prefer bpe_text (raw text); fall back to input_ids if textual
            text = example.get('bpe_text') or example.get('input_ids', '')
            if not isinstance(text, str):
                # If input_ids is numeric tokens, we cannot reliably re-tokenize into BPE
                return {'token_ids': []}
            ids = sp.encode(text, out_type=int)
            return {'token_ids': ids}

        num_proc = self._get_num_proc()
        try:
            logger.info(f"  Applying BPE tokenizer to dataset with num_proc={num_proc}...")
            self.combined_data = self.combined_data.map(
                tokenize_example,
                batched=False,
                num_proc=num_proc,
            )
        except Exception as e:
            logger.warning(f"  Could not tokenize with num_proc={num_proc}: {e}. Falling back to num_proc=1...")
            self.combined_data = self.combined_data.map(
                tokenize_example,
                batched=False,
                num_proc=1,
            )

        # Save metadata about tokenizer
        self.cache_metadata = getattr(self, 'cache_metadata', {})
        self.cache_metadata['bpe_vocab_size'] = vocab_size
        self.cache_metadata['bpe_model'] = os.path.basename(model_file)
        # Move model to canonical path
        try:
            shutil.copyfile(model_file, BPE_MODEL_PATH)
            self.cache_metadata['bpe_model_path'] = BPE_MODEL_PATH
        except Exception:
            self.cache_metadata['bpe_model_path'] = model_file

        # Detect thinking tokens in the dataset
        has_thinking = False
        if self.combined_data is not None and len(self.combined_data) > 0:
            sample_size = min(100, len(self.combined_data))
            for i in range(sample_size):
                sample = self.combined_data[i]
                text = sample.get('input_ids', '') if isinstance(sample.get('input_ids'), str) else str(sample.get('input_ids', ''))
                if '<|thinking|>' in text and '<|final|>' in text:
                    has_thinking = True
                    break
        self.cache_metadata['has_thinking_tokens'] = has_thinking
        if has_thinking:
            logger.info("  ✓ Detected GPT-2 standard thinking tokens (<|thinking|>/<|final|>) in dataset")

        # Cleanup temporary corpus
        try:
            os.remove(tmp_corpus)
        except Exception:
            pass
    
    def _add_source_column(self, dataset: Dataset, source_name: str) -> Dataset:
        """
        Add a 'source' column to a dataset with the given source name.
        
        Args:
            dataset: Input dataset
            source_name: Name of the source (e.g., 'aiml', 'hf', 'pdf', etc.)
            
        Returns:
            Dataset with 'source' column added
        """
        if dataset is None or len(dataset) == 0:
            return dataset
        
        # Check if source column already exists
        if 'source' in dataset.column_names:
            return dataset
        
        # Add source column
        return dataset.add_column('source', [source_name] * len(dataset))
    
    def _combine_datasets(self) -> Dataset:
        """
        Combine AIML, HF, PDF, EPUB, Web, and CSV datasets.
        Adds a 'source' column to track the origin of each sample.
        
        Returns:
            Combined dataset with 'source' column
        """
        datasets_to_combine = []
        total_samples = 0
        
        if self.aiml_data is not None and len(self.aiml_data) > 0:
            aiml_with_source = self._add_source_column(self.aiml_data, 'AIML')
            datasets_to_combine.append(aiml_with_source)
            total_samples += len(self.aiml_data)
            logger.info(f"  Adding AIML data: {len(self.aiml_data)} samples")
        
        if self.hf_data is not None and len(self.hf_data) > 0:
            hf_with_source = self._add_source_column(self.hf_data, 'HuggingFace')
            datasets_to_combine.append(hf_with_source)
            total_samples += len(self.hf_data)
            logger.info(f"  Adding HuggingFace data: {len(self.hf_data)} samples")
        
        if self.pdf_data is not None and len(self.pdf_data) > 0:
            pdf_with_source = self._add_source_column(self.pdf_data, 'PDF')
            datasets_to_combine.append(pdf_with_source)
            total_samples += len(self.pdf_data)
            logger.info(f"  Adding PDF data: {len(self.pdf_data)} samples")
        
        if self.epub_data is not None and len(self.epub_data) > 0:
            epub_with_source = self._add_source_column(self.epub_data, 'EPUB')
            datasets_to_combine.append(epub_with_source)
            total_samples += len(self.epub_data)
            logger.info(f"  Adding EPUB data: {len(self.epub_data)} samples")

        if self.web_data is not None and len(self.web_data) > 0:
            web_with_source = self._add_source_column(self.web_data, 'Web')
            datasets_to_combine.append(web_with_source)
            total_samples += len(self.web_data)
            logger.info(f"  Adding Web data: {len(self.web_data)} samples")

        if self.csv_data is not None and len(self.csv_data) > 0:
            csv_with_source = self._add_source_column(self.csv_data, 'CSV')
            datasets_to_combine.append(csv_with_source)
            total_samples += len(self.csv_data)
            logger.info(f"  Adding CSV data: {len(self.csv_data)} samples")

        if not datasets_to_combine:
            logger.warning("  ⚠ No datasets to combine!")
            return Dataset.from_list([])
        
        if len(datasets_to_combine) == 1:
            combined = datasets_to_combine[0]
        else:
            combined = concatenate_datasets(datasets_to_combine)
        
            logger.info(f"  Combined dataset total: {total_samples} samples")
        return combined
    
    def _standardize_combined_dataset(self):
        """Ensure the combined dataset exposes a consistent input_ids text field using parallel processing."""
        if self.combined_data is None:
            return
        if 'input_ids' in self.combined_data.column_names:
            return

        def build_input_ids(example):
            if 'input' in example and 'output' in example:
                text = f"{example.get('input', '').strip()} {example.get('output', '').strip()}".strip()
                return {'input_ids': text}
            if 'text' in example:
                return {'input_ids': example.get('text', '').strip()}
            if 'sentence' in example:
                return {'input_ids': example.get('sentence', '').strip()}

            text_fields = [k for k in example.keys() if k in ('input', 'output', 'text', 'sentence')]
            if text_fields:
                merged = ' '.join(str(example.get(k, '')).strip() for k in text_fields if example.get(k))
                return {'input_ids': merged.strip()}

            # Fallback: join all string fields
            merged = ' '.join(str(v).strip() for v in example.values() if isinstance(v, str) and v.strip())
            return {'input_ids': merged.strip()}

        columns_to_remove = [c for c in self.combined_data.column_names if c != 'input_ids']
        num_proc = self._get_num_proc()
        
        try:
            logger.info(f"  Standardizing dataset with num_proc={num_proc}...")
            self.combined_data = self.combined_data.map(
                build_input_ids,
                batched=False,
                num_proc=num_proc,
                remove_columns=columns_to_remove
            )
        except Exception as e:
            logger.warning(f"Could not standardize with num_proc={num_proc}: {e}. Falling back to num_proc=1...")
            self.combined_data = self.combined_data.map(
                build_input_ids,
                batched=False,
                num_proc=1,
                remove_columns=columns_to_remove
            )

    def _validate_and_clean_sources(self):
        """Validate and clean each source dataset using source-specific validators."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        try:
            from dataset_preparer.source_validators import get_validator, QualityReport
        except ImportError:
            logger.warning("  ⚠ source_validators module not found, skipping validation")
            return

        original_count = len(self.combined_data)
        reports = {}

        source_datasets = {
            'aiml': self.aiml_data,
            'hf': self.hf_data,
            'pdf': self.pdf_data,
            'epub': self.epub_data,
            'web': self.web_data,
            'csv': self.csv_data,
        }

        cleaned_datasets = []
        for source_name, dataset in source_datasets.items():
            if dataset is None or len(dataset) == 0:
                continue

            try:
                validator = get_validator(source_name)
                samples = [dict(item) for item in dataset]
                cleaned_samples, report = validator.validate_batch(samples)
                reports[source_name] = report

                if cleaned_samples:
                    from datasets import Dataset
                    cleaned_ds = Dataset.from_list(cleaned_samples)
                    cleaned_datasets.append(cleaned_ds)
                    # Sync source attribute so downstream generators use cleaned data
                    if source_name == 'aiml':
                        self.aiml_data = cleaned_ds
                    elif source_name == 'hf':
                        self.hf_data = cleaned_ds
                    elif source_name == 'pdf':
                        self.pdf_data = cleaned_ds
                    elif source_name == 'epub':
                        self.epub_data = cleaned_ds
                    elif source_name == 'web':
                        self.web_data = cleaned_ds
                    elif source_name == 'csv':
                        self.csv_data = cleaned_ds
            except Exception as e:
                logger.warning(f"  ⚠ Validation failed for {source_name}: {e}")
                cleaned_datasets.append(dataset)
                reports[source_name] = QualityReport(
                    source=source_name,
                    total_samples=len(dataset),
                    good=len(dataset)
                )

        if not cleaned_datasets:
            logger.warning("  ⚠ No source datasets to validate, keeping combined_data as-is")
            return

        from datasets import concatenate_datasets
        self.combined_data = concatenate_datasets(cleaned_datasets)

        logger.info("--- Source Validation Reports ---")
        for source_name, report in reports.items():
            logger.info(f"  {report.summary()}")
        logger.info(f"  Total: {original_count} -> {len(self.combined_data)} samples")

    def _generate_thinking_for_sources(self):
        """Generate real thinking data for each source using ThinkingEngine."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        try:
            from dataset_preparer.thinking_engine import ThinkingEngine
            from dataset_preparer.aiml.thinking import AIMLThinkingGenerator
            from dataset_preparer.csv.thinking import CSVThinkingGenerator
            from dataset_preparer.pdf.thinking import PDFThinkingGenerator
            from dataset_preparer.epub.thinking import EPUBThinkingGenerator
            from dataset_preparer.web.thinking import WebThinkingGenerator
            from dataset_preparer.hf.thinking import HFThinkingGenerator
            from dataset_preparer.thinking_quality import validate_thinking
        except ImportError as e:
            logger.warning(f"  ⚠ Thinking generator modules not found: {e}")
            return

        # Initialize ThinkingEngine (always available, no external dependency)
        thinking_depth = getattr(self.args, 'thinking_depth', 'adaptive')
        engine = ThinkingEngine(depth=thinking_depth)
        logger.info(f"  ✓ ThinkingEngine initialized (depth={thinking_depth})")

        # Optionally try Ollama teacher for enhanced thinking
        teacher = None
        use_ollama = getattr(self.args, 'thinking_ollama', False) or getattr(self.args, 'thinking_mode', 'nlp') == 'ollama'
        if use_ollama:
            try:
                from dataset_preparer.thinking_generators import OllamaTeacher
                thinking_model = getattr(self.args, 'thinking_model', OLLAMA_MODEL)
                teacher = OllamaTeacher(model=thinking_model)
                if not teacher.is_model_available():
                    logger.info(f"  ℹ Ollama not available, using ThinkingEngine only")
                    teacher = None
                else:
                    logger.info(f"  ✓ Ollama teacher available for enhanced thinking")
            except Exception:
                pass

        generators = {
            'aiml': AIMLThinkingGenerator(engine, teacher, depth=thinking_depth),
            'csv': CSVThinkingGenerator(engine, teacher, depth=thinking_depth),
            'pdf': PDFThinkingGenerator(engine, teacher, depth=thinking_depth),
            'epub': EPUBThinkingGenerator(engine, teacher, depth=thinking_depth),
            'web': WebThinkingGenerator(engine, teacher, depth=thinking_depth),
            'hf': HFThinkingGenerator(engine, teacher, depth=thinking_depth),
        }

        source_datasets = {
            'aiml': self.aiml_data,
            'hf': self.hf_data,
            'pdf': self.pdf_data,
            'epub': self.epub_data,
            'web': self.web_data,
            'csv': self.csv_data,
        }

        thinking_datasets = []
        total_thinking = 0
        total_samples = 0

        for source_name, dataset in source_datasets.items():
            if dataset is None or len(dataset) == 0:
                continue

            generator = generators.get(source_name)
            if not generator:
                continue

            logger.info(f"  Generating thinking for {source_name} ({len(dataset)} samples)...")

            try:
                samples = [dict(item) for item in dataset]
                enriched = []
                source_thinking = 0
                for sample in samples:
                    result = generator.generate(sample)
                    if result.get('thinking'):
                        validation = validate_thinking(result['thinking'], result.get('output', ''))
                        if validation.valid:
                            source_thinking += 1
                            # Original sample: ensure input_ids exists
                            original = dict(sample)
                            if 'input_ids' not in original:
                                inp = original.get('input', '')
                                out = original.get('output', '')
                                original['input_ids'] = f"{inp} {out}".strip() if inp and out else out or inp
                            enriched.append(original)    # Original sample (without thinking)
                            enriched.append(result)      # Thinking sample (with thinking)
                        else:
                            enriched.append(sample)      # Original only (validation failed)
                    else:
                        enriched.append(sample)          # Original only (no thinking generated)
                    total_samples += 1

                total_thinking += source_thinking
                from datasets import Dataset
                ds = Dataset.from_list(enriched)
                ds = self._add_source_column(ds, source_name)
                thinking_datasets.append(ds)
                logger.info(f"    {source_name}: {source_thinking}/{len(samples)} samples with valid thinking ({len(ds)} total)")
            except Exception as e:
                logger.warning(f"    ⚠ Thinking generation failed for {source_name}: {e}")
                thinking_datasets.append(self._add_source_column(dataset, source_name))

        if thinking_datasets:
            from datasets import concatenate_datasets
            self.combined_data = concatenate_datasets(thinking_datasets)

            logger.info(f"  Thinking generation complete: {total_thinking}/{total_samples} samples enriched")

    def _generate_agent_data(self):
        """Generate agentic data with tool calls for training."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        try:
            from dataset_preparer.agent.thinking import AgentThinkingGenerator
            from dataset_preparer.agent.quality import validate_agent_sample, filter_low_quality_agent
            from dataset_preparer.thinking_generators import OllamaTeacher
        except ImportError as e:
            logger.warning(f"  Agent modules not found: {e}")
            return

        agent_ratio = getattr(self.args, 'agent_ratio', 0.3)
        thinking_depth = getattr(self.args, 'thinking_depth', 'adaptive')

        # Initialize agent generator
        teacher = None
        use_ollama = getattr(self.args, 'thinking_ollama', False)
        if use_ollama:
            try:
                from config import OLLAMA_MODEL
                teacher = OllamaTeacher(model=getattr(self.args, 'thinking_model', OLLAMA_MODEL))
                if not teacher.is_model_available():
                    teacher = None
            except Exception:
                pass

        generator = AgentThinkingGenerator(teacher, depth=thinking_depth)

        # Process all samples
        samples = [dict(item) for item in self.combined_data]
        total = len(samples)
        agent_count = int(total * agent_ratio)

        logger.info(f"  Processing {total} samples for agentic data (target: {agent_count} tool-call samples)...")

        enriched = []
        tool_count = 0
        for i, sample in enumerate(samples):
            if tool_count < agent_count:
                # Force tool call for first N samples
                result = generator.generate(sample)
                if result.get('has_tool_call'):
                    validation = validate_agent_sample(result)
                    if validation.valid:
                        tool_count += 1
                        enriched.append(result)
                        continue
            # Normal sample (no tool call)
            enriched.append(sample)

        # Validate and filter
        valid_enriched = filter_low_quality_agent(enriched, min_score=0.3)

        # Convert to dataset
        from datasets import Dataset
        self.combined_data = Dataset.from_list(valid_enriched)

        stats = generator.get_stats()
        logger.info(f"  Agent data generation complete: {stats['tool_call_samples']} tool-call samples, "
                    f"{stats['normal_samples']} normal samples (total: {len(valid_enriched)})")

    def _apply_deduplication(self):
        """Apply deduplication to the combined dataset using MinHash LSH."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        original_count = len(self.combined_data)
        dedup_threshold = getattr(self.args, 'dedup_threshold', 0.8)

        try:
            # Extract all texts
            texts = []
            for item in self.combined_data:
                text = item.get('input_ids', '')
                if isinstance(text, str) and text.strip():
                    texts.append(text)

            if not texts:
                logger.warning("  ⚠ No texts found for deduplication")
                return

            # Apply deduplication
            unique_texts = deduplicate_texts(texts, threshold=dedup_threshold)

            # Recreate dataset with unique texts
            unique_data = [{'input_ids': text} for text in unique_texts]
            self.combined_data = Dataset.from_list(unique_data)

            removed_count = original_count - len(self.combined_data)
            logger.info(f"  ✓ Deduplication complete: {original_count} → {len(self.combined_data)} samples ({removed_count} removed)")

        except Exception as e:
            logger.warning(f"  ⚠ Deduplication failed: {e}. Continuing with original dataset.")

    def _apply_quality_filter(self):
        """Apply quality filtering to the combined dataset."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        original_count = len(self.combined_data)
        min_words = getattr(self.args, 'min_words', 5)
        max_words = getattr(self.args, 'max_words', 1000)

        try:
            # Extract all texts
            texts = []
            for item in self.combined_data:
                text = item.get('input_ids', '')
                if isinstance(text, str) and text.strip():
                    texts.append(text)

            if not texts:
                logger.warning("  ⚠ No texts found for quality filtering")
                return

            # Apply quality filtering
            filtered_texts = filter_by_quality(texts, min_words=min_words, max_words=max_words)

            # Recreate dataset with filtered texts
            filtered_data = [{'input_ids': text} for text in filtered_texts]
            self.combined_data = Dataset.from_list(filtered_data)

            removed_count = original_count - len(self.combined_data)
            logger.info(f"  ✓ Quality filtering complete: {original_count} → {len(self.combined_data)} samples ({removed_count} removed)")

        except Exception as e:
            logger.warning(f"  ⚠ Quality filtering failed: {e}. Continuing with original dataset.")

    def _apply_language_filter(self):
        """Apply language filtering to the combined dataset."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        original_count = len(self.combined_data)
        allowed_languages = getattr(self.args, 'allowed_languages', ['es', 'en'])

        try:
            # Extract all texts
            texts = []
            for item in self.combined_data:
                text = item.get('input_ids', '')
                if isinstance(text, str) and text.strip():
                    texts.append(text)

            if not texts:
                logger.warning("  ⚠ No texts found for language filtering")
                return

            # Apply language filtering
            filtered_texts = filter_by_language(texts, allowed_languages=allowed_languages)

            # Recreate dataset with filtered texts
            filtered_data = [{'input_ids': text} for text in filtered_texts]
            self.combined_data = Dataset.from_list(filtered_data)

            removed_count = original_count - len(self.combined_data)
            logger.info(f"  ✓ Language filtering complete: {original_count} → {len(self.combined_data)} samples ({removed_count} removed)")
            logger.info(f"    Allowed languages: {allowed_languages}")

        except Exception as e:
            logger.warning(f"  ⚠ Language filtering failed: {e}. Continuing with original dataset.")

    # ============================================================
    # Contamination filtering pipeline methods
    # ============================================================

    def _apply_noise_filter(self, audit=None):
        """Apply noise filter to remove URLs, emails, code, boilerplate, etc."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        if not CONTAMINATION_AVAILABLE:
            logger.warning("  Contamination module not available, skipping noise filter")
            return

        original_count = len(self.combined_data)
        categories = getattr(self.args, 'noise_categories', None)
        if categories and isinstance(categories, str):
            categories = [c.strip() for c in categories.split(',')]

        texts = [item.get('input_ids', '') for item in self.combined_data]
        sources = [item.get('source', 'unknown') for item in self.combined_data]

        nf = NoiseFilter(categories=categories)
        result = nf.filter_batch(texts)

        if result.discarded:
            kept_data = []
            for i, text in enumerate(result.kept):
                if i < len(sources):
                    kept_data.append({'input_ids': text, 'source': sources[i]})
                else:
                    kept_data.append({'input_ids': text})
            self.combined_data = Dataset.from_list(kept_data)
        else:
            kept_data = [{'input_ids': t, 'source': s} for t, s in zip(result.kept, sources)]
            self.combined_data = Dataset.from_list(kept_data)

        removed = original_count - len(self.combined_data)
        logger.info(f"  Noise filter: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        for reason, count in result.stats.items():
            logger.info(f"    {reason}: {count}")

        if audit:
            audit.set_filter('noise')
            per_source_discarded = {}
            for d in result.discarded:
                reason = d.get('reason', 'unknown')
                per_source_discarded[reason] = per_source_discarded.get(reason, 0) + 1
            for source_name in set(sources):
                source_discards = {r: c for r, c in per_source_discarded.items()}
                audit.update_source(source_name, after_noise=original_count - removed, discarded=source_discards)

    def _apply_quality_filter_contamination(self, audit=None):
        """Apply extended quality filter using contamination module."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        if not CONTAMINATION_AVAILABLE:
            logger.warning("  Contamination module not available, skipping quality filter")
            return

        original_count = len(self.combined_data)
        texts = [item.get('input_ids', '') for item in self.combined_data]

        qf = QualityFilter()
        result = qf.filter_batch(texts)

        kept_data = [{'input_ids': t} for t in result.kept]
        self.combined_data = Dataset.from_list(kept_data)

        removed = original_count - len(self.combined_data)
        logger.info(f"  Quality filter: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        for reason, count in result.stats.items():
            logger.info(f"    {reason}: {count}")

        if audit:
            audit.set_filter('quality')
            for source_name in self._get_source_names():
                audit.update_source(source_name, after_quality=len(self.combined_data))

    def _apply_cross_source_dedup(self, audit=None):
        """Apply cross-source deduplication."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        if not CONTAMINATION_AVAILABLE:
            logger.warning("  Contamination module not available, skipping cross-source dedup")
            return

        original_count = len(self.combined_data)
        dedup_mode = getattr(self.args, 'dedup_mode', 'all')
        dedup_threshold = getattr(self.args, 'dedup_threshold', 0.8)

        texts = [item.get('input_ids', '') for item in self.combined_data]
        sources = [item.get('source', 'unknown') for item in self.combined_data]

        dedup = CrossSourceDeduplicator(mode=dedup_mode, near_threshold=dedup_threshold)
        result = dedup.deduplicate(texts, sources)

        kept_data = []
        for item in result.unique_with_source:
            entry = {'input_ids': item['text']}
            if 'source' in item:
                entry['source'] = item['source']
            kept_data.append(entry)

        self.combined_data = Dataset.from_list(kept_data)

        removed = original_count - len(self.combined_data)
        logger.info(f"  Cross-source dedup: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        logger.info(f"    Exact: {result.exact_duplicates} | Near: {result.near_duplicates} | Cross-source: {result.cross_source_duplicates}")

        if audit:
            audit.set_filter('dedup')
            audit.set_cross_source_duplicates(result.cross_source_duplicates)
            for source_name in self._get_source_names():
                audit.update_source(source_name, after_dedup=len(self.combined_data))

    def _apply_balance_control(self, audit=None):
        """Apply source balance control."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        if not CONTAMINATION_AVAILABLE:
            logger.warning("  Contamination module not available, skipping balance control")
            return

        original_count = len(self.combined_data)
        max_ratio = getattr(self.args, 'max_source_ratio', 0.3)

        indices = list(range(len(self.combined_data)))
        sources = [item.get('source', 'unknown') for item in self.combined_data]

        balancer = SourceBalancer(max_ratio=max_ratio)
        result = balancer.balance(indices, sources)

        kept_data = []
        for idx in result.kept_indices:
            item = self.combined_data[idx]
            kept_data.append({'input_ids': item.get('input_ids', ''), 'source': item.get('source', 'unknown')})

        self.combined_data = Dataset.from_list(kept_data)

        removed = original_count - len(self.combined_data)
        logger.info(f"  Balance control: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        for source, ratio in sorted(result.ratios.items()):
            logger.info(f"    {source}: {result.source_counts[source]} ({ratio:.1%})")

        if audit:
            audit.set_filter('balance')
            audit.set_balance(result.ratios)
            for source_name, count in result.source_counts.items():
                audit.update_source(source_name, after_balance=count)

    def _apply_leakage_detection(self, audit=None):
        """Apply data leakage detection."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        if not CONTAMINATION_AVAILABLE:
            logger.warning("  Contamination module not available, skipping leakage detection")
            return

        original_count = len(self.combined_data)
        threshold = getattr(self.args, 'leakage_threshold', 0.5)

        texts = [item.get('input_ids', '') for item in self.combined_data]

        detector = LeakageDetector(overlap_threshold=threshold)
        result = detector.detect(texts)

        if result.flagged_indices:
            flag_set = set(result.flagged_indices)
            kept_data = [item for i, item in enumerate(self.combined_data) if i not in flag_set]
            self.combined_data = Dataset.from_list(kept_data)
        else:
            kept_data = [item for item in self.combined_data]

        removed = original_count - len(self.combined_data)
        logger.info(f"  Leakage detection: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        logger.info(f"    Overlap pairs: {len(result.overlap_pairs)} | Repeated fragments: {len(result.repeated_fragments)}")

        if audit:
            audit.set_filter('leakage')
            for source_name in self._get_source_names():
                audit.update_source(source_name, after_leakage=len(self.combined_data))

    def _apply_language_filter_contamination(self, audit=None):
        """Apply language filter using contamination module."""
        if self.combined_data is None or len(self.combined_data) == 0:
            return

        original_count = len(self.combined_data)
        allowed_languages = getattr(self.args, 'allowed_languages', ['es', 'en'])

        texts = [item.get('input_ids', '') for item in self.combined_data]
        sources = [item.get('source', 'unknown') for item in self.combined_data]

        filtered_texts = filter_by_language(texts, allowed_languages=allowed_languages)

        kept_data = [{'input_ids': t, 'source': s} for t, s in zip(filtered_texts, sources) if t]
        self.combined_data = Dataset.from_list(kept_data)

        removed = original_count - len(self.combined_data)
        logger.info(f"  Language filter: {original_count} -> {len(self.combined_data)} ({removed} removed)")
        logger.info(f"    Allowed: {allowed_languages}")

        if audit:
            audit.set_filter('language')
            for source_name in self._get_source_names():
                audit.update_source(source_name, after_language=len(self.combined_data))

    def _get_source_names(self) -> set:
        """Get unique source names from combined dataset."""
        if self.combined_data is None:
            return set()
        sources = set()
        for item in self.combined_data:
            s = item.get('source', 'unknown')
            if s:
                sources.add(s)
        return sources

    def _collect_statistics(self) -> Dict:
        """
        Collect statistics about prepared datasets.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': 0,
            'aiml_samples': 0,
            'hf_samples': 0,
            'pdf_samples': 0,
            'epub_samples': 0,
            'web_samples': 0,
            'csv_samples': 0,
            'avg_text_length': 0,
            'min_text_length': 0,
            'max_text_length': 0,
            'source_breakdown': {}
        }
        
        if self.aiml_data:
            aiml_count = len(self.aiml_data)
            stats['aiml_samples'] = aiml_count
            stats['source_breakdown']['AIML'] = aiml_count
        
        if self.hf_data:
            hf_count = len(self.hf_data)
            stats['hf_samples'] = hf_count
            stats['source_breakdown']['HuggingFace'] = hf_count
        
        if self.pdf_data:
            pdf_count = len(self.pdf_data)
            stats['pdf_samples'] = pdf_count
            stats['source_breakdown']['PDF'] = pdf_count
        
        if self.epub_data:
            epub_count = len(self.epub_data)
            stats['epub_samples'] = epub_count
            stats['source_breakdown']['EPUB'] = epub_count

        if self.web_data:
            web_count = len(self.web_data)
            stats['web_samples'] = web_count
            stats['source_breakdown']['Web'] = web_count

        if self.csv_data:
            csv_count = len(self.csv_data)
            stats['csv_samples'] = csv_count
            stats['source_breakdown']['CSV'] = csv_count

        if self.combined_data and len(self.combined_data) > 0:
            stats['total_samples'] = len(self.combined_data)
            
            # Calculate text length statistics in single pass
            try:
                total_length = 0
                min_length = float('inf')
                max_length = 0
                count = 0
                for item in self.combined_data:
                    if isinstance(item['input_ids'], str):
                        length = len(item['input_ids'].split())
                    else:
                        length = len(item['input_ids'])
                    total_length += length
                    min_length = min(min_length, length)
                    max_length = max(max_length, length)
                    count += 1
                
                if count > 0:
                    stats['avg_text_length'] = total_length / count
                    stats['min_text_length'] = min_length
                    stats['max_text_length'] = max_length
            except Exception as e:
                logger.warning(f"  Could not calculate length statistics: {e}")
        
        return stats
    
    def _display_summary(self):
        """Display preparation summary and statistics."""
        logger.info("=" * 80)
        logger.info("DATASET PREPARATION SUMMARY")
        logger.info("=" * 80)
        
        stats = self.statistics
        
        logger.info("")
        logger.info("Data Sources:")
        for source, count in stats.get('source_breakdown', {}).items():
            logger.info(f"   {source:20} {count:>10,} samples")
        
        logger.info("")
        logger.info("Combined Statistics:")
        logger.info(f"   Total Samples:       {stats.get('total_samples', 0):>10,}")
        logger.info(f"   Average Text Length: {stats.get('avg_text_length', 0):>10.1f} words")
        logger.info(f"   Min Text Length:     {stats.get('min_text_length', 0):>10} words")
        logger.info(f"   Max Text Length:     {stats.get('max_text_length', 0):>10} words")
        
        logger.info("=" * 80)


class DataValidator:
    """Validate datasets for quality and completeness."""
    
    @staticmethod
    def validate_dataset(dataset: Dataset) -> Dict:
        """
        Validate dataset structure and content.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if dataset is None or len(dataset) == 0:
            results['valid'] = False
            results['errors'].append("Dataset is empty")
            return results
        
        # Check for required columns
        try:
            first_item = dataset[0]
            if 'input_ids' not in first_item:
                results['errors'].append("Missing 'input_ids' column")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Error accessing dataset: {e}")
            results['valid'] = False
        
        # Check for null values
        null_count = 0
        for item in dataset:
            if item['input_ids'] is None or (isinstance(item['input_ids'], str) and len(item['input_ids'].strip()) == 0):
                null_count += 1
        
        if null_count > 0:
            results['warnings'].append(f"Found {null_count} null/empty values in input_ids")
        
        return results


def prepare_datasets_for_training(args) -> Tuple[Dataset, Dict]:
    """
    Main function to prepare datasets.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (combined_dataset, statistics)
    """
    preparer = DataPreparer(args)
    result = preparer.prepare()
    
    if result['status'] == 'success':
        # Validate combined dataset
        validator = DataValidator()
        validation = validator.validate_dataset(result['combined_dataset'])
        
        if not validation['valid']:
            logger.error("Dataset validation failed:")
            for error in validation['errors']:
                    logger.error(f"  {error}")
            return None, {}
        
        for warning in validation['warnings']:
            logger.warning(f"  ⚠ {warning}")
        
        return result['combined_dataset'], result['statistics']
    else:
        logger.error(f"Dataset preparation failed: {result.get('error')}")
        return None, {}
