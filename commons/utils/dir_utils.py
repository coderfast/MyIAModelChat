"""
Project directory bootstrap.

Ensures all required directories exist at startup so the rest of the
codebase never has to check or create them.
"""
import os
import logging

logger = logging.getLogger(__name__)

# ── All directories the project expects, grouped by root ──────────────
_DIRS = {
    'datasets_source': [
        'aiml',
        'csv',
        'epub',
        'hf',
        'markdown',
        'pdf',
        'web',
    ],
    'datasets_processed': [
        'markdowns/aiml',
        'markdowns/csv',
        'markdowns/epub',
        'markdowns/hf',
        'markdowns/markdown',
        'markdowns/pdf',
        'markdowns/web',
    ],
    'dataset_cache': [
        'jsonl',
        'prepared_dataset',
    ],
    'checkpoints': [],
    'models': [
        'intent',
        'sentiment',
        'exported',
    ],
}


def ensure_project_dirs(root: str = '.') -> None:
    """Create every directory the project needs.

    Args:
        root: Project root folder (default: current working directory).
    """
    created = 0
    for base, subs in _DIRS.items():
        base_path = os.path.join(root, base)
        if not os.path.isdir(base_path):
            os.makedirs(base_path, exist_ok=True)
            created += 1
            logger.info(f"  Created: {base}/")
        for sub in subs:
            full = os.path.join(base_path, sub)
            if not os.path.isdir(full):
                os.makedirs(full, exist_ok=True)
                created += 1
                logger.info(f"  Created: {base}/{sub}/")
    if created:
        logger.info(f"  {created} directories created")
    else:
        logger.info("  All project directories already exist")
