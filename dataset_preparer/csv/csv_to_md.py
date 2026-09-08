"""
CSV to Markdown converter.

Reads CSV files from datasets_source/csv/ and generates .md files
in datasets_processed/markdowns/csv/ with GPT-2 standard tokens.

CSV format: input,output
    "question","answer"

Usage:
    python -m dataset_preparer.csv.csv_to_md
    python -m dataset_preparer.csv.csv_to_md --config path/to/config.json
"""
import os
import sys
import csv
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load CSV-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def csv_to_md(config: dict) -> int:
    """Convert CSV files to markdown format.

    Args:
        config: Configuration dict with source_dir, output_dir, format, oversample_factor

    Returns:
        Number of .md files generated
    """
    from commons.utils.text_utils import clean_text
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    source_dir = config.get('source_dir', 'datasets_source/csv')
    output_dir = config.get('output_dir', 'datasets_processed/markdowns/csv')
    fmt = config.get('format', 'chat')
    oversample_factor = config.get('oversample_factor', 5)

    manifest = load_manifest()

    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        logger.info(f"Created CSV source directory: {source_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    file_count = 0

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.csv'):
            continue
        csv_path = os.path.join(source_dir, filename)
        file_stem = Path(filename).stem
        file_output_dir = os.path.join(output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)
        try:
            logger.info(f"Processing CSV: {filename}...")
            file_generated = 0

            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)
                has_header = first_line.lower().startswith('input')

                reader = csv.DictReader(f) if has_header else csv.reader(f)

                for row in reader:
                    if has_header:
                        input_text = row.get('input', '').strip()
                        output_text = row.get('output', '').strip()
                    else:
                        if len(row) < 2:
                            continue
                        input_text = row[0].strip().strip('"')
                        output_text = row[1].strip().strip('"')

                    input_text = clean_text(input_text)
                    output_text = clean_text(output_text)

                    if not input_text or not output_text:
                        continue

                    combined_text = f"{input_text} {output_text}"
                    lang_code = get_language_for_file(csv_path, 'csv', manifest, combined_text)
                    lang_token = get_language_token(lang_code)

                    if fmt == 'chat':
                        md_content = f"{lang_token}<|user|>{input_text}<|end|><|assistant|>{output_text}<|end|>"
                    else:
                        md_content = f"{lang_token}<|problem|>{input_text}<|final|>"

                    for _ in range(oversample_factor):
                        safe_name = f"csv_{file_stem}_{generated:06d}.md"
                        with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as out_f:
                            out_f.write(md_content)
                        generated += 1
                        file_generated += 1

            file_count += 1
            logger.info(f"  Processed: {filename} ({file_generated} samples)")

        except Exception as e:
            logger.warning(f"  Error processing {filename}: {e}")

    logger.info(f"Generated {generated} markdown files from {file_count} CSVs in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'csv_config.json'),
                        help='Path to CSV config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = csv_to_md(config)
    logger.info(f"CSV conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
