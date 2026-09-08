"""
AIML to Markdown converter.

Reads AIML files from datasets_source/aiml/ and generates .md files
in datasets_processed/markdowns/aiml/ with GPT-2 standard tokens.

Usage:
    python -m dataset_preparer.aiml.aiml_to_md
    python -m dataset_preparer.aiml.aiml_to_md --config path/to/config.json
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load AIML-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aiml_to_md(config: dict) -> int:
    """Convert AIML files to markdown format.

    Args:
        config: Configuration dict with source_dir, output_dir, format, quality

    Returns:
        Number of .md files generated
    """
    from dataset_preparer.aiml.parser import parse_aiml_directory
    from commons.utils.text_utils import clean_text, filter_by_quality
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    source_dir = config.get('source_dir', 'datasets_source/aiml')
    output_dir = config.get('output_dir', 'datasets_processed/markdowns/aiml')
    fmt = config.get('format', 'problem_final')
    quality_cfg = config.get('quality', {})
    min_words = quality_cfg.get('min_words', 3)
    max_words = quality_cfg.get('max_words', 500)
    min_alpha = quality_cfg.get('min_alpha_ratio', 0.5)

    manifest = load_manifest()
    lang_code = get_language_for_file(source_dir, 'aiml', manifest)
    lang_token = get_language_token(lang_code)

    if not os.path.exists(source_dir):
        logger.warning(f"AIML source directory not found: {source_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading AIML files from: {source_dir}")
    samples, stats = parse_aiml_directory(source_dir)

    if not samples:
        logger.warning("No samples extracted from AIML files")
        return 0

    logger.info(f"Extracted {len(samples)} samples from {stats.get('files_processed', 0)} files")

    generated = 0
    for i, sample in enumerate(samples):
        pattern = sample.get('original_pattern', sample.get('input', sample.get('pattern', ''))).strip()
        template = sample.get('original_template', sample.get('output', sample.get('template', ''))).strip()

        if not pattern or not template:
            continue

        pattern = clean_text(pattern)
        template = clean_text(template)

        if not pattern or not template:
            continue

        source_file = sample.get('source_file', 'aiml')
        file_output_dir = os.path.join(output_dir, source_file)
        os.makedirs(file_output_dir, exist_ok=True)

        combined_text = f"{pattern} {template}"
        words = combined_text.split()
        if len(words) < min_words or len(words) > max_words:
            continue

        alpha_count = sum(1 for c in combined_text if c.isalpha())
        alpha_ratio = alpha_count / len(combined_text) if len(combined_text) > 0 else 0
        if alpha_ratio < min_alpha:
            continue

        if fmt == 'chat':
            md_content = f"{lang_token}<|user|>{pattern}<|end|><|assistant|>{template}<|end|>"
        else:
            md_content = f"{lang_token}<|problem|>{pattern}<|final|>"

        source_file = sample.get('source_file', 'aiml')
        safe_name = f"aiml_{source_file}_{i:06d}.md"
        output_path = os.path.join(file_output_dir, safe_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        generated += 1

    logger.info(f"Generated {generated} markdown files in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert AIML files to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'aiml_config.json'),
                        help='Path to AIML config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = aiml_to_md(config)
    logger.info(f"AIML conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
