"""
EPUB to Markdown converter.

Reads EPUB files from datasets_source/epub/ and generates .md files
in datasets_processed/markdowns/epub/ with GPT-2 standard tokens.

Usage:
    python -m dataset_preparer.epub.epub_to_md
    python -m dataset_preparer.epub.epub_to_md --config path/to/config.json
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
    """Load EPUB-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def epub_to_md(config: dict) -> int:
    """Convert EPUB files to markdown format.

    Args:
        config: Configuration dict with source_dir, output_dir, format, chunking, quality

    Returns:
        Number of .md files generated
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        logger.error("ebooklib not installed. Install with: pip install ebooklib")
        return 0

    try:
        from bs4 import BeautifulSoup
        BS4_AVAILABLE = True
    except ImportError:
        BS4_AVAILABLE = False

    from commons.utils.text_utils import (
        clean_text, split_paragraphs, chunk_text_by_tokens,
        clean_page_artifacts, detect_language
    )
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    source_dir = config.get('source_dir', 'datasets_source/epub')
    output_dir = config.get('output_dir', 'datasets_processed/markdowns/epub')
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
        logger.info(f"Created EPUB source directory: {source_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    file_count = 0

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.epub'):
            continue
        file_path = os.path.join(source_dir, filename)
        file_stem = Path(filename).stem
        file_output_dir = os.path.join(output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)
        try:
            logger.info(f"Processing EPUB: {filename}...")
            book = epub.read_epub(file_path)
            chapters = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        content = item.get_content().decode('utf-8', errors='ignore')
                        # Remove AIML <thinking> tags completely (legacy)
                        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

                        chapter_paragraphs = []
                        if BS4_AVAILABLE:
                            try:
                                soup = BeautifulSoup(content, 'html.parser')
                                for p in soup.find_all(['p', 'div', 'li', 'blockquote', 'pre']):
                                    para = p.get_text(separator=' ', strip=True)
                                    if para:
                                        chapter_paragraphs.append(para)
                                if not chapter_paragraphs:
                                    chapter_paragraphs = [soup.get_text(separator=' ', strip=True)]
                            except Exception:
                                chapter_paragraphs = []

                        if not chapter_paragraphs:
                            content_clean = re.sub(r'<[^>]+>', '', content)
                            if content_clean.strip():
                                chapter_paragraphs = [content_clean]

                        restored = chapter_paragraphs

                        chapter_text = '\n\n'.join(
                            re.sub(r'\s+', ' ', p).strip() for p in restored if p.strip()
                        )
                        if chapter_text.strip():
                            chapters.append(chapter_text)
                    except Exception as e:
                        logger.warning(f"  Error extracting chapter: {e}")

            chapters = clean_page_artifacts(chapters)

            all_chapters_text = ' '.join(chapters)
            lang_code = get_language_for_file(file_path, 'epub', manifest, all_chapters_text)
            lang_token = get_language_token(lang_code)

            for chapter in chapters:
                chapter = clean_text(chapter)
                if len(chapter) < min_paragraph_length:
                    continue

                if not enable_chunking or len(chapter.split()) <= max_tokens:
                    if fmt == 'chat':
                        md_content = f"{lang_token}<|problem|>{chapter}<|final|>"
                    else:
                        md_content = f"{lang_token}<|problem|>{chapter}<|final|>"
                    safe_name = f"epub_{file_stem}_{generated:06d}.md"
                    with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    generated += 1
                    continue

                paragraphs = split_paragraphs(chapter, min_length=min_paragraph_length)
                for paragraph in paragraphs:
                    if len(paragraph) < min_paragraph_length:
                        continue
                    if enable_chunking and len(paragraph.split()) > max_tokens:
                        chunks = chunk_text_by_tokens(paragraph, max_tokens, overlap_tokens)
                        for chunk in chunks:
                            chunk = clean_text(chunk)
                            if chunk:
                                if fmt == 'chat':
                                    md_content = f"{lang_token}<|problem|>{chunk}<|final|>"
                                else:
                                    md_content = f"{lang_token}<|problem|>{chunk}<|final|>"
                                safe_name = f"epub_{file_stem}_{generated:06d}.md"
                                with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                                    f.write(md_content)
                                generated += 1
                    else:
                        if fmt == 'chat':
                            md_content = f"{lang_token}<|problem|>{paragraph}<|final|>"
                        else:
                            md_content = f"{lang_token}<|problem|>{paragraph}<|final|>"
                        safe_name = f"epub_{file_stem}_{generated:06d}.md"
                        with open(os.path.join(file_output_dir, safe_name), 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        generated += 1

            file_count += 1
            logger.info(f"  Processed: {filename} ({len(chapters)} chapters)")

        except Exception as e:
            logger.warning(f"  Error processing {filename}: {e}")

    logger.info(f"Generated {generated} markdown files from {file_count} EPUBs in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert EPUB files to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'epub_config.json'),
                        help='Path to EPUB config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = epub_to_md(config)
    logger.info(f"EPUB conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
