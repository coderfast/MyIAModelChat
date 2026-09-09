"""
AIML Loader - loads AIML files and creates training datasets.
Uses the AIMLParser for full element resolution and wildcard expansion.
"""
import os
import logging

from datasets import Dataset

from dataset_preparer.aiml.parser import AIMLParser, parse_aiml_file, parse_aiml_directory

logger = logging.getLogger(__name__)


class AIMLLoader:
    """Load AIML files and create training datasets."""

    def __init__(self, aiml_dir: str):
        self.aiml_dir = aiml_dir
        self.final_dataset = None
        self._stats = {}

    def load_aiml_files(self) -> Dataset:
        """
        Load all AIML files using the full parser.

        Returns:
            HuggingFace Dataset with normalized training samples
        """
        logger.info(f"Loading AIML files from {self.aiml_dir}")

        samples, stats = parse_aiml_directory(self.aiml_dir)
        self._stats = stats

        if not samples:
            logger.warning("No samples extracted from AIML files")
            self.final_dataset = Dataset.from_list([])
            return self.final_dataset

        self.final_dataset = Dataset.from_list(samples)
        logger.info(
            f"Created dataset: {len(samples)} samples from {stats['files_processed']} files"
        )

        return self.final_dataset

    def create_hf_dataset(self) -> Dataset:
        """
        Create HuggingFace Dataset from AIML files.
        Uses the full parser for element resolution.

        Returns:
            HuggingFace Dataset
        """
        return self.load_aiml_files()

    def get_response(self, input_text: str) -> str:
        """
        Get response for input text (for testing).
        Uses the parser to find matching patterns.
        """
        # Simple pattern matching for testing
        input_upper = input_text.upper().strip()

        for sample in self.final_dataset or []:
            pattern = sample.get('original_pattern', '').upper()
            if pattern == input_upper or pattern in input_upper:
                return sample.get('original_template', '')

        return "I don't have a response for that."

    @property
    def stats(self) -> dict:
        return dict(self._stats)