"""Insert <|thinking|>...<|final|> reasoning blocks before body paragraphs of the
MD books in this folder, for later AI model training.

Scope (plan THINKING_IN_PROGRESS.md v2):
  - IN : ids 1-120, 125, 151  (122 clean files)
  - OUT: ids 121-124, 126-150 (29 template-filler files, excluded)

Usage (from this folder):
  python add_thinking_to_md.py --dry-run --limit 1        # preview, no writes
  python add_thinking_to_md.py --limit 1                  # process 001 only
  python add_thinking_to_md.py                            # process all 122
  python add_thinking_to_md.py --files 001 075            # explicit subset

Guarantees:
  - Backup of every original to backup/ BEFORE first overwrite (pristine copy)
  - Idempotent: paragraphs already preceded by <|thinking|> are left untouched
  - Headers (#, ##, ###) never modified
  - No external LLM: ThinkingEngine (spaCy/regex NLP) only
"""
import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]  # .../MyIAModelChat
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_preparer.thinking_engine import ThinkingEngine  # noqa: E402

FOLDER_TAG = 'english - thinking'
# Full template family used by the filler generator (verified: 0 hits in IN corpus).
# 125_aquatic_ecosystems is 100% template (1813/1813) and excluded from scope.
FILLER_PHRASES = (
    'represents an extremely broad and fascinating area of study',
    'In the current context, characterized by rapid digital',
    'It is fundamental to highlight that knowledge of',
    'extends to the international arena, where cooperation between nations',
    'Looking to the future, the prospects for',
    'is a dynamic and constantly evolving field',
    'This book presents a',
    'In this chapter, we will analyze in depth the topic of',
)
MIN_WORDS = 10
MAX_PARAGRAPH_WORDS = 400
SPLIT_SENTENCE = re.compile(r'(?<=[.!?])\s+')

logger = logging.getLogger('add_thinking')


def in_scope(topic_id: int) -> bool:
    """Scope IN (plan v2 + revision 2026-09-29): 001-120 + 151 = 121 files.
    125 excluded: verified 100% template filler (1813/1813 blocks)."""
    return topic_id <= 120 or topic_id == 151


def load_topics(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_blocks(text: str):
    """Split MD into ordered ('header'|'para', text) blocks.

    Handles headers immediately followed by body text without a blank line.
    """
    blocks = []
    current = []

    def flush():
        if current:
            blocks.append(('para', '\n'.join(current).strip()))
            current.clear()

    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            flush()
            blocks.append(('header', line.rstrip()))
        elif not line.strip():
            flush()
        else:
            current.append(line.rstrip())
    flush()
    return blocks


def split_long_paragraph(text: str, max_words: int = MAX_PARAGRAPH_WORDS):
    """Split a very long paragraph at sentence boundaries (<= max_words each)."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    current = []
    current_words = 0
    for sentence in SPLIT_SENTENCE.split(text):
        sw = len(sentence.split())
        if current and current_words + sw > max_words:
            chunks.append(' '.join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += sw
    if current:
        chunks.append(' '.join(current))
    return chunks


def should_skip(text: str) -> str:
    """Return a reason to skip thinking for this paragraph, or ''."""
    if len(text.split()) < MIN_WORDS:
        return 'too_short'
    if any(phrase in text for phrase in FILLER_PHRASES):
        return 'boilerplate'
    return ''


def process_file(md_path: Path, title: str, lang: str, engine: ThinkingEngine,
                 dry_run: bool, backup_dir: Path) -> dict:
    stats = {'paragraphs': 0, 'thinking': 0, 'skipped_short': 0,
             'skipped_boilerplate': 0, 'skipped_existing': 0, 'split': 0}
    original = md_path.read_text(encoding='utf-8')
    blocks = parse_blocks(original)
    out_parts = []
    prev_was_thinking = False

    for kind, content in blocks:
        if kind == 'header':
            out_parts.append(content)
            prev_was_thinking = False
            continue
        if content.startswith('<|thinking|>'):
            out_parts.append(content)
            prev_was_thinking = True
            continue
        if prev_was_thinking:
            out_parts.append(content)
            stats['skipped_existing'] += 1
            prev_was_thinking = False
            continue

        reason = should_skip(content)
        if reason:
            stats[f'skipped_{reason}' if reason != 'too_short' else 'skipped_short'] += 1
            out_parts.append(content)
            continue

        chunks = split_long_paragraph(content)
        if len(chunks) > 1:
            stats['split'] += 1
        for chunk in chunks:
            stats['paragraphs'] += 1
            thinking = engine.generate_thinking(chunk, context={'title': title, 'lang': lang})
            if thinking:
                out_parts.append(f'<|thinking|>{thinking}<|final|>')
                out_parts.append(chunk)
                stats['thinking'] += 1
            else:
                out_parts.append(chunk)

    rebuilt = '\n\n'.join(out_parts) + '\n'
    if dry_run:
        return stats

    if rebuilt != original:
        backup_dir.mkdir(exist_ok=True)
        backup_file = backup_dir / md_path.name
        if not backup_file.exists():
            shutil.copy2(md_path, backup_file)
        md_path.write_text(rebuilt, encoding='utf-8')
    return stats


def main():
    parser = argparse.ArgumentParser(description='Add thinking blocks to MD books')
    parser.add_argument('--dry-run', action='store_true', help='print stats only, no writes')
    parser.add_argument('--show', type=int, default=0, metavar='N',
                        help='print N sample thinkings during dry-run')
    parser.add_argument('--limit', type=int, default=0, help='process only the first N files')
    parser.add_argument('--files', nargs='*', default=None,
                        help='explicit file ids (e.g. 001 002) or filenames')
    parser.add_argument('--depth', default='adaptive', choices=['basic', 'adaptive', 'detailed'])
    parser.add_argument('--no-backup', action='store_true', help='DANGEROUS: skip backup')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s',
    )

    topics = load_topics(HERE / 'topics.json')
    lang = topics.get('language', 'en')
    entries = [t for t in topics['topics'] if in_scope(t['id'])]

    if args.files:
        wanted = set(args.files)
        entries = [t for t in entries
                   if str(t['id']).zfill(3) in wanted or t['filename'] in wanted
                   or f"{t['id']:03d}" in wanted]
    if args.limit > 0:
        entries = entries[:args.limit]

    if not entries:
        logger.error('No files selected (check --files / scope IN list)')
        return 1

    engine = ThinkingEngine(depth=args.depth)
    backup_dir = HERE / 'backup'
    totals = {}
    t0 = time.time()
    logger.info(f'[{FOLDER_TAG}] processing {len(entries)} files | lang={lang} '
                f'| depth={args.depth} | dry_run={args.dry_run}')

    for i, entry in enumerate(entries, 1):
        md_path = HERE / entry['filename']
        if not md_path.exists():
            logger.warning(f'  [{i}/{len(entries)}] MISSING {entry["filename"]}')
            continue
        stats = process_file(md_path, entry['title'], lang, engine,
                             args.dry_run, backup_dir)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        logger.info(
            f'  [{i}/{len(entries)}] {entry["filename"]}: '
            f'{stats["thinking"]} thinking / {stats["paragraphs"]} paras '
            f'(skip: {stats["skipped_short"]} short, '
            f'{stats["skipped_boilerplate"]} boilerplate, '
            f'{stats["skipped_existing"]} existing)'
        )
        if args.dry_run and args.show and i == 1:
            _show_samples(md_path, entry['title'], lang, engine, args.show)

    elapsed = time.time() - t0
    mode = 'DRY-RUN (no files written)' if args.dry_run else 'WRITTEN (backups in backup/)'
    logger.info(f'\n=== {mode} ===')
    logger.info(f"  paragraphs processed : {totals.get('paragraphs', 0)}")
    logger.info(f"  thinking inserted    : {totals.get('thinking', 0)}")
    logger.info(f"  skipped too short    : {totals.get('skipped_short', 0)}")
    logger.info(f"  skipped boilerplate  : {totals.get('skipped_boilerplate', 0)}")
    logger.info(f"  skipped existing     : {totals.get('skipped_existing', 0)}")
    logger.info(f"  paragraphs split     : {totals.get('split', 0)} (> {MAX_PARAGRAPH_WORDS} words)")
    logger.info(f"  time                 : {elapsed:.1f}s")
    return 0


def _show_samples(md_path: Path, title: str, lang: str, engine: ThinkingEngine, n: int):
    blocks = parse_blocks(md_path.read_text(encoding='utf-8'))
    shown = 0
    logger.info(f'\n--- SAMPLE THINKINGS from {md_path.name} ---')
    for kind, content in blocks:
        if kind == 'header' or content.startswith('<|thinking|>'):
            continue
        if should_skip(content):
            continue
        thinking = engine.generate_thinking(content, context={'title': title, 'lang': lang})
        if thinking:
            logger.info(f'\nPARAGRAPH: {content[:100]}...')
            logger.info(f'THINKING : {thinking}')
            shown += 1
            if shown >= n:
                break


if __name__ == '__main__':
    sys.exit(main())
