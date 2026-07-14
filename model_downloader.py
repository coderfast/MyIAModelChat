"""Download HuggingFace models to local directory if not already present."""

import os
import logging

logger = logging.getLogger(__name__)


def ensure_model_local(model_id: str, local_dir: str) -> str:
    """
    Download a HuggingFace model to local_dir if not already present.
    Returns the local path to the model directory.

    Args:
        model_id: HuggingFace model ID (e.g. 'nlptown/bert-base-multilingual-uncased-sentiment')
        local_dir: Local directory to store the model (e.g. 'models/sentiment')

    Returns:
        str: Local path to the model directory
    """
    local_path = os.path.abspath(local_dir)

    # Check if model already exists locally
    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"Model found locally: {local_path}")
        return local_path

    # Download from HuggingFace
    logger.info(f"Downloading model '{model_id}' to '{local_path}'...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        logger.info(f"Model downloaded successfully to: {local_path}")
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model download. "
            "Install it: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model '{model_id}': {e}. "
            "Check your internet connection or download manually."
        )

    return local_path
