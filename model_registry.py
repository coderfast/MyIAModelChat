"""
Model registry: discovers, lists, validates, and manages trained model checkpoints.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = 'models'
EXPORTED_DIR = os.path.join(MODELS_DIR, 'exported')


def scan_models(directory: str = MODELS_DIR) -> Dict[str, dict]:
    """Scan directory for .pth model files and return {name: {path, metadata, size_mb}}."""
    models = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        return models

    for pth_file in sorted(dir_path.glob('*.pth')):
        name = pth_file.stem
        size_mb = round(pth_file.stat().st_size / (1024 * 1024), 2)
        meta = load_model_metadata(str(pth_file))
        models[name] = {
            'path': str(pth_file),
            'size_mb': size_mb,
            'metadata': meta,
        }
    return models


def load_model_metadata(path: str) -> dict:
    """Load metadata from a checkpoint without loading weights."""
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            return {
                'model_name': ckpt.get('model_name', Path(path).stem),
                'architecture': ckpt.get('architecture', {}),
                'dataset_source': ckpt.get('dataset_source', 'unknown'),
                'epoch': ckpt.get('epoch', None),
                'loss': ckpt.get('loss', None),
            }
    except Exception as e:
        logger.warning(f"Could not load metadata from {path}: {e}")
    return {}


def validate_compatibility(model_a_path: str, model_b_path: str) -> tuple:
    """Compare architecture and tokenizer of two checkpoints.
    
    Returns (is_compatible: bool, errors: list[str])
    """
    errors = []
    meta_a = load_model_metadata(model_a_path)
    meta_b = load_model_metadata(model_b_path)

    arch_a = meta_a.get('architecture', {})
    arch_b = meta_b.get('architecture', {})

    if not arch_a or not arch_b:
        if not arch_a:
            errors.append(f"{Path(model_a_path).stem}: no architecture metadata found")
        if not arch_b:
            errors.append(f"{Path(model_b_path).stem}: no architecture metadata found")
        return False, errors

    # Compare architecture fields
    fields = ['embed_size', 'hidden_size', 'num_layers', 'n_head', 'n_positions', 'vocab_size']
    for field in fields:
        val_a = arch_a.get(field)
        val_b = arch_b.get(field)
        if val_a is not None and val_b is not None and val_a != val_b:
            errors.append(
                f"Architecture mismatch: {field} = {val_a} vs {val_b}"
            )

    # Compare state_dict keys and shapes
    try:
        import torch
        ckpt_a = torch.load(model_a_path, map_location='cpu', weights_only=False)
        ckpt_b = torch.load(model_b_path, map_location='cpu', weights_only=False)

        sd_a = ckpt_a.get('model_state_dict', ckpt_a) if isinstance(ckpt_a, dict) else ckpt_a
        sd_b = ckpt_b.get('model_state_dict', ckpt_b) if isinstance(ckpt_b, dict) else ckpt_b

        keys_a = set(sd_a.keys())
        keys_b = set(sd_b.keys())

        missing_in_b = keys_a - keys_b
        missing_in_a = keys_b - keys_a

        if missing_in_b:
            errors.append(f"Keys in A but not in B: {missing_in_b}")
        if missing_in_a:
            errors.append(f"Keys in B but not in A: {missing_in_a}")

        for key in keys_a & keys_b:
            if sd_a[key].shape != sd_b[key].shape:
                errors.append(f"Shape mismatch for '{key}': {sd_a[key].shape} vs {sd_b[key].shape}")
    except Exception as e:
        errors.append(f"Could not compare state_dicts: {e}")

    return len(errors) == 0, errors


def list_models_cli():
    """Print available models to console."""
    models = scan_models()
    if not models:
        print("No models found in models/")
        return

    print(f"\nAvailable models ({len(models)}):")
    print("-" * 60)
    for name, info in models.items():
        meta = info.get('metadata', {})
        arch = meta.get('architecture', {})
        dataset = meta.get('dataset_source', 'unknown')
        epoch = meta.get('epoch', '?')
        loss = meta.get('loss')
        loss_str = f"{loss:.4f}" if loss is not None else '?'
        arch_str = f"embed={arch.get('embed_size', '?')} hidden={arch.get('hidden_size', '?')} layers={arch.get('num_layers', '?')}"
        print(f"  {name}")
        print(f"    File: {info['path']} ({info['size_mb']} MB)")
        print(f"    Dataset: {dataset} | Epoch: {epoch} | Loss: {loss_str}")
        print(f"    Architecture: {arch_str}")
        print()
