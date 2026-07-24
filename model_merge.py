"""
Model merging: combines multiple trained model checkpoints by weight averaging.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

import torch

from model_registry import load_model_metadata, validate_compatibility, MODELS_DIR

logger = logging.getLogger(__name__)


def merge_models(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Merge multiple model checkpoints by weighted average of their state dicts.

    Args:
        model_paths: List of .pth checkpoint paths to merge.
        weights: Optional weights for each model. If None, uniform average is used.
        output_path: Where to save the merged model. If None, auto-generates in models/.

    Returns:
        Path to the merged checkpoint file.
    """
    if len(model_paths) < 2:
        raise ValueError("Need at least 2 models to merge")

    # Validate compatibility
    for i in range(len(model_paths) - 1):
        compatible, errors = validate_compatibility(model_paths[i], model_paths[i + 1])
        if not compatible:
            raise ValueError(
                f"Models are not compatible:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # Default uniform weights
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        if len(weights) != len(model_paths):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_paths)})")
        total = sum(weights)
        weights = [w / total for w in weights]

    logger.info(f"Merging {len(model_paths)} models with weights: {weights}")

    # Load all state dicts
    state_dicts = []
    ref_metadata = None
    for path in model_paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get('model_state_dict', ckpt)
            if ref_metadata is None:
                ref_metadata = {
                    'epoch': ckpt.get('epoch'),
                    'loss': ckpt.get('loss'),
                    'tokenizer': ckpt.get('tokenizer'),
                }
        else:
            sd = ckpt
        state_dicts.append(sd)

    # Merge by weighted average
    merged = {}
    keys = state_dicts[0].keys()
    for key in keys:
        merged[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            merged[key] += sd[key].float() * w

    # Convert back to original dtypes
    for key in keys:
        merged[key] = merged[key].to(state_dicts[0][key].dtype)

    # Determine output path
    if output_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        names = [Path(p).stem for p in model_paths]
        output_path = os.path.join(MODELS_DIR, '+'.join(names) + '.pth')

    # Load metadata from first model for architecture info
    meta = load_model_metadata(model_paths[0])

    # Build dataset_source string
    dataset_sources = []
    for p in model_paths:
        m = load_model_metadata(p)
        ds = m.get('dataset_source', Path(p).stem)
        dataset_sources.append(ds)

    torch.save({
        'model_state_dict': merged,
        'model_name': '+'.join(Path(p).stem for p in model_paths),
        'architecture': meta.get('architecture', {}),
        'dataset_source': '+'.join(dataset_sources),
        'merged': True,
        'merge_weights': weights,
        'merge_sources': [str(p) for p in model_paths],
        'tokenizer': ref_metadata.get('tokenizer') if ref_metadata else None,
    }, output_path)

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    logger.info(f"Merged model saved to {output_path} ({size_mb} MB)")
    return output_path


def merge_from_names(
    model_names: List[str],
    weights: Optional[List[float]] = None,
    models_dir: str = MODELS_DIR,
) -> str:
    """Merge models by name, resolving paths from the models directory."""
    paths = []
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        paths.append(path)
    return merge_models(paths, weights)


def parse_merge_spec(spec: str) -> tuple:
    """Parse a merge specification like 'model_a:0.6+model_b:0.4' or 'model_a+model_b'.

    Returns (names: list[str], weights: list[float] or None)
    """
    parts = spec.split('+')
    names = []
    weights = []
    has_weights = False

    for part in parts:
        part = part.strip()
        if ':' in part:
            name, w = part.rsplit(':', 1)
            names.append(name.strip())
            weights.append(float(w))
            has_weights = True
        else:
            names.append(part)
            weights.append(1.0)

    return names, weights if has_weights else None
