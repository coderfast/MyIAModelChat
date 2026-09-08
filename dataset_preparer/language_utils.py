"""Language utilities for markdown generation.

Provides language detection for *_to_md.py scripts using:
1. Language manifest (datasets_source/language_manifest.json)
2. Automatic detection from text content
"""
import os
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = os.path.join('datasets_source', 'language_manifest.json')


def load_manifest(manifest_path: Optional[str] = None) -> dict:
    """Load language manifest from JSON file."""
    if manifest_path is None:
        manifest_path = DEFAULT_MANIFEST_PATH
    if not os.path.exists(manifest_path):
        return {'files': {}, 'sources': {}, 'default': 'unknown'}
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning('Could not load language manifest: %s', e)
        return {'files': {}, 'sources': {}, 'default': 'unknown'}


def get_language_for_file(
    file_path: str,
    source_key: str,
    manifest: Optional[dict] = None,
    text_content: Optional[str] = None
) -> str:
    """Get language code for a file using manifest or detection.

    Priority:
    1. manifest['files'][file_path] - exact file match
    2. manifest['sources'][source_key] - source-level default
    3. detect_language(text_content) - automatic detection
    4. manifest['default'] - fallback
    """
    if manifest is None:
        manifest = load_manifest()

    mf = manifest.get('files', {})
    ms = manifest.get('sources', {})
    md = manifest.get('default', 'unknown')

    if file_path and mf:
        if file_path in mf:
            return mf[file_path]
        norm = file_path.replace(os.sep, '/').lstrip('./')
        for fp, lang in mf.items():
            nfp = fp.replace(os.sep, '/').lstrip('./')
            if norm.endswith(nfp) or nfp.endswith(norm):
                return lang

    if source_key and source_key in ms:
        return ms[source_key]

    if text_content:
        try:
            from commons.language_utils import detect_language
            lang = detect_language(text_content)
            if lang and lang != 'unknown':
                return lang
        except ImportError:
            pass

    return md


def get_language_token(lang_code: str) -> str:
    """Convert language code to GPT-2 token format."""
    if not lang_code or lang_code == 'unknown':
        return ''
    return '<|%s|>' % lang_code


def extract_language_from_text(text: str) -> tuple:
    """Extract language token from beginning of text.

    Returns (language_code, remaining_text).
    If no token found, returns ('', text).
    """
    match = re.match(r'^<\|([a-z]{2,3})\|>', text)
    if match:
        return match.group(1), text[match.end():]
    return '', text
