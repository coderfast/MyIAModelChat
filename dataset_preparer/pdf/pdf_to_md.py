"""
PDF to Markdown converter.

Reads PDF files from datasets_source/pdf/ and generates .md files
in datasets_processed/markdowns/pdf/ with GPT-2 standard tokens.

Usage:
    python -m dataset_preparer.pdf.pdf_to_md
    python -m dataset_preparer.pdf.pdf_to_md --config path/to/config.json
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load PDF-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pdf_to_md(config: dict) -> int:
    """Convert PDF files to markdown format.

    Args:
        config: Configuration dict with source_dir, output_dir, format, chunking, quality

    Returns:
        Number of .md files generated
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        return 0

    from commons.utils.text_utils import (
        clean_text, split_paragraphs, chunk_text_by_tokens,
        clean_page_artifacts, detect_language
    )
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    source_dir = config.get('source_dir', 'datasets_source/pdf')
    output_dir = config.get('output_dir', 'datasets_processed/markdowns/pdf')
    fmt = config.get('format', 'problem_final')
    chunking_cfg = config.get('chunking', {})
    enable_chunking = chunking_cfg.get('enabled', False)
    max_tokens = chunking_cfg.get('max_tokens', 512)
    overlap_tokens = chunking_cfg.get('overlap_tokens', 50)
    quality_cfg = config.get('quality', {})
    min_paragraph_length = quality_cfg.get('min_paragraph_length', 20)

    manifest = load_manifest()

    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        logger.info(f"Created PDF source directory: {source_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    file_count = 0

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.pdf'):
            continue
        file_path = os.path.join(source_dir, filename)
        file_stem = Path(filename).stem
        file_output_dir = os.path.join(output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)
        try:
            logger.info(f"Processing PDF: {filename}...")
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"  Error extracting page {page_num}: {e}")

                text_parts = clean_page_artifacts(text_parts)
                text = "\n\n".join(text_parts)

                if not text.strip():
                    logger.warning(f"  No text extracted from {filename}")
                    continue

                lang_code = get_language_for_file(file_path, 'pdf', manifest, text)
                lang_token = get_language_token(lang_code)

                paragraphs = split_paragraphs(text, min_length=min_paragraph_length)

                for para in paragraphs:
                    para = clean_text(para)
                    if len(para) < min_paragraph_length:
                        continue

                    if enable_chunking and len(para.split()) > max_tokens:
                        chunks = chunk_text_by_tokens(para, max_tokens, overlap_tokens)
                        for chunk in chunks:
                            chunk = clean_text(chunk)
                            if chunk:
                                md_content = f"{lang_token}<|problem|>{chunk}<|final|>"
                                safe_name = f"pdf_{file_stem}_{generated:06d}.md"
                                with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                                    f.write(md_content)
                                generated += 1
                    else:
                        md_content = f"{lang_token}<|problem|>{para}<|final|>"
                        safe_name = f"pdf_{file_stem}_{generated:06d}.md"
                        with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        generated += 1

                file_count += 1
                logger.info(f"  Processed: {filename} ({len(paragraphs)} paragraphs)")

        except Exception as e:
            logger.warning(f"  Error processing {filename}: {e}")

    logger.info(f"Generated {generated} markdown files from {file_count} PDFs in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert PDF files to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'pdf_config.json'),
                        help='Path to PDF config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = pdf_to_md(config)
    logger.info(f"PDF conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
