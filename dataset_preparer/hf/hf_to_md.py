"""
HuggingFace dataset to Markdown converter.

Downloads HuggingFace datasets and generates .md files
in datasets_processed/markdowns/hf/ with GPT-2 standard tokens.

Usage:
    python -m dataset_preparer.hf.hf_to_md
    python -m dataset_preparer.hf.hf_to_md --config path/to/config.json
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load HF-to-markdown configuration from JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def hf_to_md(config: dict) -> int:
    """Convert HuggingFace datasets to markdown format.

    Args:
        config: Configuration dict with output_dir, datasets list, defaults

    Returns:
        Number of .md files generated
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Install with: pip install datasets")
        return 0

    from commons.utils.text_utils import clean_text
    from dataset_preparer.language_utils import get_language_for_file, get_language_token, load_manifest

    output_dir = config.get('output_dir', 'datasets_processed/markdowns/hf')
    datasets_config = config.get('datasets', [])
    defaults = config.get('defaults', {})

    manifest = load_manifest()

    if not datasets_config:
        logger.warning("No datasets defined in config")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    generated = 0

    for ds_cfg in datasets_config:
        try:
            url = ds_cfg.get('url', '')
            dataset_name = url.rstrip('/').split('/datasets/')[-1]
            cfg = ds_cfg.get('config')
            split = ds_cfg.get('split', defaults.get('split', 'train'))
            fmt = ds_cfg.get('format', defaults.get('format', 'plain_text'))
            languages = ds_cfg.get('languages', defaults.get('languages', ['en']))
            limit = ds_cfg.get('limit', defaults.get('limit'))
            language_field = ds_cfg.get('language_field')
            language_values_map = ds_cfg.get('language_values_map')

            logger.info(f"Loading HuggingFace dataset: {dataset_name}...")

            load_kwargs = {'split': split}
            if cfg:
                load_kwargs['name'] = cfg

            ds = load_dataset(dataset_name, **load_kwargs)

            if language_field and languages:
                if language_values_map:
                    allowed_values = [k for k, v in language_values_map.items() if v in languages]
                    ds = ds.filter(lambda x, lf=language_field, av=allowed_values: x.get(lf) in av)
                else:
                    ds = ds.filter(lambda x, lf=language_field, langs=languages: x.get(lf) in langs)
                logger.info(f"  Filtered to {languages}: {len(ds)} samples")

            if limit:
                ds = ds.select(range(min(limit, len(ds))))

            ds_generated = 0
            for example in ds:
                if fmt == 'chat':
                    conversations = example.get('conversations', [])
                    if len(conversations) < 2:
                        continue
                    human = conversations[0].get('value', '')
                    gpt = conversations[1].get('value', '')
                    if not human or not gpt:
                        continue
                    human = clean_text(human)
                    gpt = clean_text(gpt)
                    if not human or not gpt:
                        continue
                    combined_text = f"{human} {gpt}"
                    lang_code = get_language_for_file(dataset_name, 'hf', manifest, combined_text)
                    lang_token = get_language_token(lang_code)
                    md_content = f"{lang_token}<|user|>{human}<|end|><|assistant|>{gpt}<|end|>"
                else:
                    text = example.get('text', example.get('sentence', ''))
                    if not text or not isinstance(text, str):
                        continue
                    text = clean_text(text)
                    if not text:
                        continue
                    lang_code = get_language_for_file(dataset_name, 'hf', manifest, text)
                    lang_token = get_language_token(lang_code)
                    md_content = f"{lang_token}<|problem|>{text}<|final|>"

                safe_name = f"hf_{dataset_name.replace('/', '_')}_{ds_generated:06d}.md"
                safe_name = safe_name.replace(' ', '_')
                with open(os.path.join(output_dir, safe_name), 'w', encoding='utf-8') as f:
                    f.write(md_content)
                ds_generated += 1
                generated += 1

            logger.info(f"  Processed: {dataset_name} ({ds_generated} samples)")

        except Exception as e:
            logger.warning(f"  Error processing dataset: {e}")
            continue

    logger.info(f"Generated {generated} markdown files in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace datasets to Markdown')
    parser.add_argument('--config', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'hf_config.json'),
                        help='Path to HF config JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    count = hf_to_md(config)
    logger.info(f"HuggingFace conversion complete: {count} files generated")


if __name__ == '__main__':
    main()
