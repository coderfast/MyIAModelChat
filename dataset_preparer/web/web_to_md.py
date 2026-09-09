"""
Web documentation to Markdown converter.

Scrapes web documentation from URLs and generates .md files
in datasets_processed/markdowns/web/ with GPT-2 standard tokens.

Usage:
    python -m dataset_preparer.web.web_to_md
    python -m dataset_preparer.web.web_to_md --config path/to/config.json
    python -m dataset_preparer.web.web_to_md --url https://example.com/docs
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load web-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def web_to_md(config: dict, cli_url: str = None, cli_max_pages: int = None,
              cli_max_depth: int = None) -> int:
    """Convert web documentation to markdown format.

    Args:
        config: Configuration dict with output_dir, urls_file, format, chunking, defaults
        cli_url: Optional single URL override from CLI
        cli_max_pages: Optional max pages override from CLI
        cli_max_depth: Optional max depth override from CLI

    Returns:
        Number of .md files generated
    """
    try:
        from dataset_preparer.web.scraper import scrape_web_docs, load_urls_from_json
    except ImportError:
        logger.error("Web scraper not available. Install: pip install trafilatura beautifulsoup4 requests")
        return 0

    from commons.utils.text_utils import clean_text, split_paragraphs, chunk_text_by_tokens
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    output_dir = config.get('output_dir', 'datasets_processed/markdowns/web')
    fmt = config.get('format', 'problem_final')
    chunking_cfg = config.get('chunking', {})
    enable_chunking = chunking_cfg.get('enabled', False)
    max_tokens = chunking_cfg.get('max_tokens', 512)
    overlap_tokens = chunking_cfg.get('overlap_tokens', 50)
    defaults = config.get('defaults', {})

    manifest = load_manifest()

    os.makedirs(output_dir, exist_ok=True)

    url_configs = []
    if cli_url:
        url_configs.append({
            'url': cli_url,
            'max_pages': cli_max_pages or defaults.get('max_pages', 50),
            'max_depth': cli_max_depth or defaults.get('max_depth', 3),
            'rate_limit': defaults.get('rate_limit', 1.0),
        })
    else:
        urls_file = config.get('urls_file', 'datasets_source/web/urls_to_process.json')
        if os.path.exists(urls_file):
            url_configs = load_urls_from_json(urls_file)
            logger.info(f"Loaded {len(url_configs)} URL configs from {urls_file}")
        else:
            logger.warning(f"No URLs file found: {urls_file}")
            return 0

    if not url_configs:
        logger.warning("No URLs to scrape")
        return 0

    all_texts = []
    url_stems = {}
    for url_cfg in url_configs:
        seed_url = url_cfg['url']
        url_stem = seed_url.split('//')[-1].replace('/', '_').replace('.', '_')[:50]
        url_stems[seed_url] = url_stem
        url_output_dir = os.path.join(output_dir, url_stem)
        os.makedirs(url_output_dir, exist_ok=True)
        logger.info(f"Scraping: {seed_url}")
        try:
            texts = scrape_web_docs(
                url=seed_url,
                max_pages=url_cfg.get('max_pages', defaults.get('max_pages', 50)),
                max_depth=url_cfg.get('max_depth', defaults.get('max_depth', 3)),
                rate_limit=url_cfg.get('rate_limit', defaults.get('rate_limit', 1.0)),
            )
            for t in texts:
                all_texts.append((t, url_stem))
        except Exception as e:
            logger.warning(f"  Failed to scrape {seed_url}: {e}")
            continue

    if not all_texts:
        logger.warning("No text extracted from web URLs")
        return 0

    generated = 0
    for text, url_stem in all_texts:
        lang_code = get_language_for_file('', 'web', manifest, text)
        lang_token = get_language_token(lang_code)
        url_output_dir = os.path.join(output_dir, url_stem)

        paragraphs = split_paragraphs(text)
        for paragraph in paragraphs:
            paragraph = clean_text(paragraph)
            if len(paragraph) < 20:
                continue

            if enable_chunking and len(paragraph.split()) > max_tokens:
                chunks = chunk_text_by_tokens(paragraph, max_tokens, overlap_tokens)
                for chunk in chunks:
                    chunk = clean_text(chunk)
                    if chunk:
                        md_content = f"{lang_token}<|problem|>{chunk}<|final|>"
                        safe_name = f"web_{url_stem}_{generated:06d}.md"
                        with open(os.path.join(url_output_dir, safe_name), 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        generated += 1
            else:
                md_content = f"{lang_token}<|problem|>{paragraph}<|final|>"
                safe_name = f"web_{url_stem}_{generated:06d}.md"
                with open(os.path.join(url_output_dir, safe_name), 'w', encoding='utf-8') as f:
                    f.write(md_content)
                generated += 1

    logger.info(f"Generated {generated} markdown files from {len(all_texts)} web pages in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert web documentation to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'web_config.json'),
                        help='Path to web config JSON')
    parser.add_argument('--url', type=str, default=None,
                        help='Single URL to scrape (overrides config)')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Max pages per URL')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Max link-following depth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = web_to_md(config, cli_url=args.url, cli_max_pages=args.max_pages,
                      cli_max_depth=args.max_depth)
    logger.info(f"Web conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
