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
from aimlloder import AIMLLoader
from datasets import load_dataset, concatenate_datasets, Dataset
import sys

try:
    import psutil
except ImportError:
    psutil = None
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.pkl')


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
        self.special_facts_data = None
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
            
            logger.info("\n💾 Loading cached dataset...")
            
            # Load dataset
            self.combined_data = Dataset.load_from_disk(CACHE_DATASET_FILE)
            logger.info(f"  ✓ Loaded cached dataset: {len(self.combined_data)} samples")
            
            # Load statistics
            with open(CACHE_STATS_FILE, 'rb') as f:
                self.statistics = pickle.load(f)
            logger.info(f"  ✓ Loaded cached statistics")
            
            logger.info("\n✅ DATASET LOADED FROM CACHE")
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
            
            logger.info("\n💾 Saving dataset to cache...")
            
            # Save dataset
            self.combined_data.save_to_disk(CACHE_DATASET_FILE)
            logger.info(f"  ✓ Saved dataset: {CACHE_DATASET_FILE}")
            
            # Save statistics
            with open(CACHE_STATS_FILE, 'wb') as f:
                pickle.dump(self.statistics, f)
            logger.info(f"  ✓ Saved statistics: {CACHE_STATS_FILE}")
            
            logger.info("  ✓ Cache ready for future runs")
            
        except Exception as e:
            logger.warning(f"  Could not save to cache: {e}")
    
    def _clear_cache(self):
        """Clear cached dataset."""
        try:
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                self._ensure_cache_dir()
                logger.info("✅ Cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
    
    def prepare(self) -> Dict:
        """
        Prepare all datasets and collect statistics.
        
        Returns:
            Dictionary with preparation status and statistics
        """
        logger.info("=" * 80)
        logger.info("DATASET PREPARATION STARTING")
        logger.info("=" * 80)

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
                logger.info("\n🔄 Refreshing cache (clearing old cache)...")
                self._clear_cache()
            
            # Check for use cache flag
            if hasattr(self.args, 'use_cache') and self.args.use_cache and not (hasattr(self.args, 'refresh_cache') and self.args.refresh_cache):
                result = self._load_from_cache()
                if result:
                    self.combined_data, self.statistics = result
                    self._display_summary()
                    logger.info("\n" + "=" * 80)
                    logger.info("✅ DATASET READY (from cache)")
                    logger.info("=" * 80)
                    return {
                        'status': 'success',
                        'combined_dataset': self.combined_data,
                        'statistics': self.statistics,
                        'from_cache': True
                    }
            
            # Prepare fresh dataset
            logger.info("\n[1/3] Loading AIML data..." if (self.args.aiml or self.args.hf) else "\n[1/3] No data sources selected...")
            
            # Load AIML data
            if self.args.aiml:
                self.aiml_data = self._load_aiml_data()
            
            # Load Hugging Face data
            if self.args.hf:
                logger.info("\n[2/3] Loading Hugging Face datasets...")
                self.hf_data = self._load_hf_data()
            
            # Load PDF data
            if hasattr(self.args, 'pdf') and self.args.pdf:
                logger.info("\n[3/3] Loading PDF files...")
                self.pdf_data = self._load_pdf_data()
            
            # Load EPUB data
            if hasattr(self.args, 'epub') and self.args.epub:
                logger.info("\n[4/3] Loading EPUB files...")
                self.epub_data = self._load_epub_data()

            # Load special facts (curated supplementary dataset)
            logger.info("\n[5/3] Loading special facts (CSV)...")
            self.special_facts_data = self._load_special_facts()
            
            # Combine datasets
            logger.info("\n[6/3] Combining datasets...")
            self.combined_data = self._combine_datasets()
            
            # Collect statistics
            logger.info("\nGathering statistics...")
            self.statistics = self._collect_statistics()
            
            # Save to cache
            self._save_to_cache()
            
            # Display summary
            self._display_summary()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ DATASET PREPARATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'combined_dataset': self.combined_data,
                'statistics': self.statistics,
                'from_cache': False
            }
            
        except Exception as e:
            logger.error(f"\n❌ Dataset preparation failed: {e}")
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
        Load AIML files from aiml_dev directory.
        
        Returns:
            Hugging Face Dataset with AIML data
        """
        data_dir = 'aiml_dev'
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        logger.info(f"  Loading AIML files from: {data_dir}")
        
        # Load AIML files
        try:
            aiml_loader = AIMLLoader(data_dir)
            logger.info(f"  ✓ AIML loader initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Error initializing AIML loader: {e}")
            return Dataset.from_list([])
        
        # Load .datasets files
        aiml_datasets = []
        datasets_count = 0
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.datasets'):
                file_path = os.path.join(data_dir, filename)
                try:
                    tokenized_data = self._safe_load_dataset_pickle(file_path)

                    # AIML dataset pickles are usually HuggingFace Datasets with input/output fields.
                    if isinstance(tokenized_data, Dataset):
                        ds = tokenized_data

                        if 'input' in ds.column_names and 'output' in ds.column_names:
                            ds = ds.map(
                                lambda x: {
                                    'input_ids': (x['input'] + ' ' + x['output']).strip()
                                    if x.get('output') else x['input']
                                },
                                batched=False
                            )
                        elif 'input_ids' in ds.column_names:
                            # Already normalized
                            ds = ds
                        else:
                            # Fallback, try to concatenate any available text fields
                            text_fields = [c for c in ds.column_names if c in ('input', 'text', 'sentence')]
                            if text_fields:
                                ds = ds.map(
                                    lambda x: {'input_ids': ' '.join(str(x[c]) for c in text_fields if x.get(c))},
                                    batched=False
                                )
                            else:
                                raise ValueError("AIML dataset missing both 'input'/'output' and 'input_ids' fields")

                        if 'input_ids' in ds.column_names:
                            ds = ds.select_columns(['input_ids'])

                        aiml_datasets.append(ds)
                        datasets_count += len(ds)
                        logger.info(f"  ✓ Loaded: {filename} ({len(ds)} samples)")

                    elif isinstance(tokenized_data, list):
                        # Fallback: tokenized_data is list of sequences
                        data = [{'input_ids': ' '.join(map(str, token_ids))} for token_ids in tokenized_data]
                        aiml_datasets.append(Dataset.from_list(data))
                        datasets_count += len(data)
                        logger.info(f"  ✓ Loaded: {filename} ({len(data)} samples)")

                    else:
                        raise ValueError("Unexpected AIML .datasets content type: %s" % type(tokenized_data))

                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")
        
        if aiml_datasets:
            combined = concatenate_datasets(aiml_datasets)
            logger.info(f"\n  Total AIML files processed: {datasets_count}")
            logger.info(f"  Total AIML samples: {len(combined)}")
            return combined
        else:
            logger.warning("  ⚠ No .datasets files found in aiml_dev/")
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
            'wikitext',  # Wikipedia text
            # 'common_voice',  # Speech data (uncomment for more data)
            # 'opus_100',  # Multi-language data (uncomment for multilingual)
        ]
        
        for dataset_name in hf_dataset_names:
            try:
                logger.info(f"  Loading Hugging Face dataset: {dataset_name}...")
                
                if dataset_name == 'wikitext':
                    ds = load_dataset('wikitext', 'wikitext-2', split='train')
                    # Take subset for faster preparation
                    ds = ds.select(range(min(1000, len(ds))))
                elif dataset_name == 'common_voice':
                    ds = load_dataset('common_voice', '2024-08', split='train[:1000]', languages=["en"])
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
            logger.info(f"\nTotal HuggingFace datasets loaded: {len(hf_datasets)}")
            logger.info(f"Total HuggingFace samples: {len(combined)}")
            return combined
        else:
            logger.warning("  ⚠ No HuggingFace datasets loaded successfully")
            return Dataset.from_list([])

    def _load_special_facts(self) -> Dataset:
        """
        Load curated supplemental facts/conversational pairs from datasets/special_facts.csv.
        This keeps special knowledge in data, not hardcoded code paths.
        Returns:
            Hugging Face Dataset with input_ids column
        """
        csv_path = os.path.join('datasets', 'special_facts.csv')

        if not os.path.exists(csv_path):
            logger.warning(f"  ⚠ Special facts CSV not found: {csv_path}")
            return Dataset.from_list([])

        special_items = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip header if present
                header = f.readline().strip().split(',')
                has_header = 'input' in header and 'output' in header
                if not has_header:
                    f.seek(0)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',', 1)
                    if len(parts) != 2:
                        continue

                    input_text = parts[0].strip().strip('"')
                    output_text = parts[1].strip().strip('"')
                    if input_text and output_text:
                        special_items.append({'input_ids': f"{input_text} {output_text}"})

            if special_items:
                ds = Dataset.from_list(special_items)
                logger.info(f"  ✓ Loaded special facts: {len(ds)} samples")
                return ds
            else:
                logger.warning("  ⚠ No special facts found in CSV")
                return Dataset.from_list([])

        except Exception as e:
            logger.warning(f"  ⚠ Error reading special facts CSV: {e}")
            return Dataset.from_list([])

    def _load_pdf_data(self) -> Dataset:
        """
        Load text data from PDF files in 'pdfs' directory.
        
        Returns:
            Hugging Face Dataset with PDF text data
        """
        if PdfReader is None:
            logger.warning("  ⚠ PyPDF2 not installed. Install with: pip install PyPDF2")
            return Dataset.from_list([])
        
        pdf_dir = 'pdfs'
        
        # Create pdfs directory if it doesn't exist
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            logger.info(f"  Created directory: {pdf_dir}")
            logger.info(f"  ℹ Place PDF files in '{pdf_dir}' directory to load them")
            return Dataset.from_list([])
        
        pdf_texts = []
        pdf_count = 0
        
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                try:
                    logger.info(f"  Reading PDF: {filename}...")
                    
                    with open(file_path, 'rb') as f:
                        pdf_reader = PdfReader(f)
                        text = ""
                        
                        # Extract text from all pages
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                text += page.extract_text() + " "
                            except Exception as e:
                                logger.warning(f"    ⚠ Error extracting page {page_num} from {filename}: {e}")
                        
                        # Split text by sentences/paragraphs into samples
                        if text.strip():
                            # Split by periods, keeping reasonable chunk sizes
                            sentences = text.split('.')
                            sentence_count = 0
                            
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if len(sentence) > 10:  # Skip very short sentences
                                    pdf_texts.append({'input_ids': sentence})
                                    sentence_count += 1
                            
                            pdf_count += 1
                            logger.info(f"  ✓ Loaded: {filename} ({sentence_count} samples)")
                        else:
                            logger.warning(f"  ⚠ No text extracted from {filename}")
                
                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")
        
        if pdf_texts:
            dataset = Dataset.from_list(pdf_texts)
            logger.info(f"\nTotal PDF files processed: {pdf_count}")
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
        
        epub_dir = 'epub'
        
        # Create epub directory if it doesn't exist
        if not os.path.exists(epub_dir):
            os.makedirs(epub_dir)
            logger.info(f"  Created directory: {epub_dir}")
            logger.info(f"  ℹ Place EPUB files in '{epub_dir}' directory to load them")
            return Dataset.from_list([])
        
        epub_texts = []
        epub_count = 0
        
        for filename in os.listdir(epub_dir):
            if filename.lower().endswith('.epub'):
                file_path = os.path.join(epub_dir, filename)
                try:
                    logger.info(f"  Reading EPUB: {filename}...")
                    
                    # Open and parse EPUB
                    book = epub.read_epub(file_path)
                    text = ""
                    
                    # Extract text from all chapters
                    for item in book.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            try:
                                # Get chapter content
                                content = item.get_content().decode('utf-8', errors='ignore')
                                
                                # Remove HTML tags (simple regex approach)
                                import re
                                content = re.sub(r'<[^>]+>', '', content)
                                
                                # Clean up whitespace
                                content = re.sub(r'\s+', ' ', content)
                                text += content + " "
                            except Exception as e:
                                logger.warning(f"    ⚠ Error extracting chapter from {filename}: {e}")
                    
                    # Split text into samples
                    if text.strip():
                        # Split by periods, keeping reasonable chunk sizes
                        sentences = text.split('.')
                        sentence_count = 0
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if len(sentence) > 10:  # Skip very short sentences
                                epub_texts.append({'input_ids': sentence})
                                sentence_count += 1
                        
                        epub_count += 1
                        logger.info(f"  ✓ Loaded: {filename} ({sentence_count} samples)")
                    else:
                        logger.warning(f"  ⚠ No text extracted from {filename}")
                
                except Exception as e:
                    logger.warning(f"  ⚠ Error loading {filename}: {e}")
        
        if epub_texts:
            dataset = Dataset.from_list(epub_texts)
            logger.info(f"\nTotal EPUB files processed: {epub_count}")
            logger.info(f"Total EPUB samples: {len(dataset)}")
            return dataset
        else:
            logger.warning(f"  ⚠ No EPUB files found in '{epub_dir}' directory")
            return Dataset.from_list([])
    
    def _combine_datasets(self) -> Dataset:
        """
        Combine AIML, HF, PDF, and EPUB datasets.
        
        Returns:
            Combined dataset
        """
        datasets_to_combine = []
        total_samples = 0
        
        if self.aiml_data is not None and len(self.aiml_data) > 0:
            datasets_to_combine.append(self.aiml_data)
            total_samples += len(self.aiml_data)
            logger.info(f"  Adding AIML data: {len(self.aiml_data)} samples")
        
        if self.hf_data is not None and len(self.hf_data) > 0:
            datasets_to_combine.append(self.hf_data)
            total_samples += len(self.hf_data)
            logger.info(f"  Adding HuggingFace data: {len(self.hf_data)} samples")
        
        if self.pdf_data is not None and len(self.pdf_data) > 0:
            datasets_to_combine.append(self.pdf_data)
            total_samples += len(self.pdf_data)
            logger.info(f"  Adding PDF data: {len(self.pdf_data)} samples")
        
        if self.epub_data is not None and len(self.epub_data) > 0:
            datasets_to_combine.append(self.epub_data)
            total_samples += len(self.epub_data)
            logger.info(f"  Adding EPUB data: {len(self.epub_data)} samples")

        if self.special_facts_data is not None and len(self.special_facts_data) > 0:
            datasets_to_combine.append(self.special_facts_data)
            total_samples += len(self.special_facts_data)
            logger.info(f"  Adding special facts data: {len(self.special_facts_data)} samples")
        
        if not datasets_to_combine:
            logger.warning("  ⚠ No datasets to combine!")
            return Dataset.from_list([])
        
        if len(datasets_to_combine) == 1:
            combined = datasets_to_combine[0]
        else:
            combined = concatenate_datasets(datasets_to_combine)
        
        logger.info(f"\n  ✓ Combined dataset total: {total_samples} samples")
        return combined
    
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
            'special_facts_samples': 0,
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

        if self.special_facts_data:
            special_count = len(self.special_facts_data)
            stats['special_facts_samples'] = special_count
            stats['source_breakdown']['SpecialFacts'] = special_count
        
        if self.combined_data and len(self.combined_data) > 0:
            stats['total_samples'] = len(self.combined_data)
            
            # Calculate text length statistics
            try:
                lengths = []
                for item in self.combined_data:
                    if isinstance(item['input_ids'], str):
                        length = len(item['input_ids'].split())
                    else:
                        length = len(item['input_ids'])
                    lengths.append(length)
                
                if lengths:
                    stats['avg_text_length'] = sum(lengths) / len(lengths)
                    stats['min_text_length'] = min(lengths)
                    stats['max_text_length'] = max(lengths)
            except Exception as e:
                logger.warning(f"  Could not calculate length statistics: {e}")
        
        return stats
    
    def _display_summary(self):
        """Display preparation summary and statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("DATASET PREPARATION SUMMARY")
        logger.info("=" * 80)
        
        stats = self.statistics
        
        logger.info(f"\n📊 Data Sources:")
        for source, count in stats.get('source_breakdown', {}).items():
            logger.info(f"   {source:20} {count:>10,} samples")
        
        logger.info(f"\n📈 Combined Statistics:")
        logger.info(f"   Total Samples:       {stats.get('total_samples', 0):>10,}")
        logger.info(f"   Average Text Length: {stats.get('avg_text_length', 0):>10.1f} words")
        logger.info(f"   Min Text Length:     {stats.get('min_text_length', 0):>10} words")
        logger.info(f"   Max Text Length:     {stats.get('max_text_length', 0):>10} words")
        
        logger.info("\n" + "=" * 80)


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
                logger.error(f"  ❌ {error}")
            return None, {}
        
        for warning in validation['warnings']:
            logger.warning(f"  ⚠ {warning}")
        
        return result['combined_dataset'], result['statistics']
    else:
        logger.error(f"Dataset preparation failed: {result.get('error')}")
        return None, {}
