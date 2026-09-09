"""
Markdown to Markdown converter.

Reads .md files from datasets_source/markdown/ and generates processed .md files
in datasets_processed/markdowns/markdown/ with GPT-2 standard tokens.

Extracts code blocks as separate samples, keeps headings, cleans images/HTML.

Usage:
    python -m dataset_preparer.markdown.markdown_to_md
    python -m dataset_preparer.markdown.markdown_to_md --config path/to/config.json
"""
import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load Markdown-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def markdown_to_md(config: dict) -> int:
    """Convert Markdown files to processed markdown format.

    Extracts code blocks as separate training samples, keeps headings,
    cleans images and HTML tags, wraps each with GPT-2 standard tokens.

    Args:
        config: Configuration dict with source_dir, output_dir, format, chunking, quality

    Returns:
        Number of .md files generated
    """
    from commons.utils.text_utils import (
        clean_text, split_paragraphs, chunk_text_by_tokens,
    )
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    source_dir = config.get('source_dir', 'datasets_source/markdown')
    output_dir = config.get('output_dir', 'datasets_processed/markdowns/markdown')
    fmt = config.get('format', 'problem_final')
    chunking_cfg = config.get('chunking', {})
    enable_chunking = chunking_cfg.get('enabled', False)
    max_tokens = chunking_cfg.get('max_tokens', 512)
    overlap_tokens = chunking_cfg.get('overlap_tokens', 50)
    quality_cfg = config.get('quality', {})
    min_paragraph_length = quality_cfg.get('min_paragraph_length', 5)

    manifest = load_manifest()

    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        logger.info(f"Created Markdown source directory: {source_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    file_count = 0

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.md'):
            continue
        file_path = os.path.join(source_dir, filename)
        file_stem = Path(filename).stem
        file_output_dir = os.path.join(output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)
        try:
            logger.info(f"Processing Markdown: {filename}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content or not content.strip():
                logger.warning(f"  Empty file: {filename}")
                continue

            lang_code = get_language_for_file(file_path, 'markdown', manifest, content)
            lang_token = get_language_token(lang_code)

            # Remove image references (![alt](path))
            content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

            # Extract code blocks as separate samples (they are valuable content)
            code_blocks = re.findall(r'```[\s\S]*?```', content)
            for block in code_blocks:
                code_text = re.sub(r'^```\w*\n?', '', block)
                code_text = re.sub(r'```\s*$', '', code_text)
                code_text = code_text.strip()
                if code_text and len(code_text.split()) >= 3:
                    if enable_chunking and len(code_text.split()) > max_tokens:
                        chunks = chunk_text_by_tokens(code_text, max_tokens, overlap_tokens)
                        for chunk in chunks:
                            chunk = clean_text(chunk)
                            if chunk:
                                md_content = f"{lang_token}<|problem|>{chunk}<|final|>"
                                safe_name = f"md_{file_stem}_{generated:06d}.md"
                                with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                                    f.write(md_content)
                                generated += 1
                    else:
                        md_content = f"{lang_token}<|problem|>{code_text}<|final|>"
                        safe_name = f"md_{file_stem}_{generated:06d}.md"
                        with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        generated += 1

            # Remove code blocks from main content (already extracted)
            content = re.sub(r'```[\s\S]*?```', '', content)

            # Remove inline code backticks but keep the code text
            content = re.sub(r'`([^`]+)`', r'\1', content)

            # Remove HTML tags
            content = re.sub(r'<[^>]+>', '', content)

            paragraphs = split_paragraphs(content, min_length=min_paragraph_length)

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
                            safe_name = f"md_{file_stem}_{generated:06d}.md"
                            with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                                f.write(md_content)
                            generated += 1
                else:
                    md_content = f"{lang_token}<|problem|>{para}<|final|>"
                    safe_name = f"md_{file_stem}_{generated:06d}.md"
                    with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    generated += 1

            file_count += 1
            logger.info(f"  Processed: {filename} ({generated} samples)")

        except Exception as e:
            logger.warning(f"  Error processing {filename}: {e}")

    logger.info(f"Generated {generated} markdown files from {file_count} Markdown files in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert Markdown files to processed Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'markdown_config.json'),
                        help='Path to Markdown config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = markdown_to_md(config)
    logger.info(f"Markdown conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
