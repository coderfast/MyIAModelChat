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
            args: Command line arguments with aiml, hf, use_cache, refresh_cache flags
        """
        self.args = args
        self.aiml_data = None
        self.hf_data = None
        self.combined_data = None
        self.statistics = {}
        self._ensure_cache_dir()
        
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
            
            # Combine datasets
            logger.info("\n[3/3] Combining datasets...")
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
                    with open(file_path, 'rb') as f:
                        tokenized_data = pickle.load(f)
                        data = [{'input_ids': token_ids} for token_ids in tokenized_data]
                        aiml_datasets.append(Dataset.from_list(data))
                        datasets_count += 1
                        logger.info(f"  ✓ Loaded: {filename} ({len(data)} samples)")
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
    
    def _combine_datasets(self) -> Dataset:
        """
        Combine AIML and HF datasets.
        
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
