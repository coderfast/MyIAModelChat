"""
Shared device detection and resolution utilities.

Provides GPU enumeration, device resolution, Vulkan checks,
and CPU+GPU layer distribution logic.
"""

import os
import logging
from typing import List, Dict, Optional, Any

import torch

logger = logging.getLogger(__name__)

# GPU reference database for AI recommendations
GPU_REFERENCE_DB: Dict[str, Dict[str, Any]] = {
    "NVIDIA GeForce RTX 5050": {
        "vram_gb": 8,
        "compute_capability": (10, 0),
        "ai_recommended": True,
        "ai_level": "GOOD",
        "max_model_config": "embed_size=512, num_layers=8",
        "notes": "Blackwell arch, good for small-medium models",
    },
    "NVIDIA GeForce RTX 3080": {
        "vram_gb": 10,
        "compute_capability": (8, 6),
        "ai_recommended": True,
        "ai_level": "EXCELLENT",
        "max_model_config": "embed_size=512, num_layers=12",
        "notes": "Ampere arch, solid performance",
    },
    "NVIDIA GeForce RTX 4090": {
        "vram_gb": 24,
        "compute_capability": (8, 9),
        "ai_recommended": True,
        "ai_level": "EXCELLENT",
        "max_model_config": "embed_size=1024, num_layers=24",
        "notes": "Ada Lovelace, excellent for large models",
    },
    "NVIDIA Tesla K80": {
        "vram_gb": 24,
        "compute_capability": (3, 7),
        "ai_recommended": True,
        "ai_level": "EXCELLENT",
        "max_model_config": "embed_size=1024, num_layers=24 (slow compute)",
        "notes": "Kepler arch, slower compute, needs memory fraction limit",
    },
    "NVIDIA A100": {
        "vram_gb": 80,
        "compute_capability": (8, 0),
        "ai_recommended": True,
        "ai_level": "OPTIMAL",
        "max_model_config": "embed_size=2048, num_layers=48+",
        "notes": "Datacenter GPU, ideal for large-scale training",
    },
}


def _get_ai_level(vram_gb: float) -> str:
    """Return AI recommendation level based on VRAM size."""
    if vram_gb < 3.5:
        return "NOT_RECOMMENDED"
    if vram_gb < 7.5:
        return "BASIC"
    if vram_gb < 15.5:
        return "GOOD"
    if vram_gb < 23.5:
        return "EXCELLENT"
    return "OPTIMAL"


def _get_max_model_config(vram_gb: float) -> str:
    """Return approximate max model config for given VRAM."""
    if vram_gb < 3.5:
        return "embed_size<=128, num_layers<=2"
    if vram_gb < 7.5:
        return "embed_size<=256, num_layers<=4"
    if vram_gb < 15.5:
        return "embed_size<=512, num_layers<=8"
    if vram_gb < 23.5:
        return "embed_size<=512, num_layers<=12"
    return "embed_size<=1024, num_layers<=24+"


def enumerate_gpus() -> List[Dict[str, Any]]:
    """Enumerate available GPUs with detailed info and AI recommendations.

    Returns a list of dicts, one per detected GPU, with:
    - index, name, vram_total_gb, vram_free_gb
    - compute_capability, status
    - ai_recommended, ai_level, max_model_config, notes
    """
    gpus: List[Dict[str, Any]] = []

    if not torch.cuda.is_available():
        return gpus

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(i)
            name = props.name
            vram_total = props.total_memory / (1024 ** 3)
            cc = (props.major, props.minor)

            # Try to get free memory
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                vram_free = free_mem / (1024 ** 3)
            except Exception:
                vram_free = vram_total

            # Check reference DB
            ref = GPU_REFERENCE_DB.get(name, {})
            ai_level = _get_ai_level(vram_total)
            ai_recommended = ai_level != "NOT_RECOMMENDED"
            max_config = ref.get("max_model_config", _get_max_model_config(vram_total))
            notes = ref.get("notes", "")

            status = "Available"

            gpus.append({
                "index": i,
                "name": name,
                "vram_total_gb": round(vram_total, 2),
                "vram_free_gb": round(vram_free, 2),
                "compute_capability": cc,
                "status": status,
                "ai_recommended": ai_recommended,
                "ai_level": ai_level,
                "max_model_config": max_config,
                "notes": notes,
            })
        except Exception as e:
            logger.warning(f"Could not query GPU {i}: {e}")

    return gpus


def check_vulkan_available() -> bool:
    """Check if Vulkan backend is available in PyTorch."""
    try:
        device = torch.device("vulkan")
        return device.type == "vulkan"
    except Exception:
        return False


def resolve_device(
    device_mode: str = "auto",
    gpu_indices: Optional[List[int]] = None,
    use_vulkan: bool = False,
) -> torch.device:
    """Resolve the torch.device based on configuration.

    Args:
        device_mode: 'cpu', 'gpu', 'cpu+gpu', or 'auto'
        gpu_indices: List of GPU indices to use, or None for auto
        use_vulkan: Force Vulkan backend if available

    Returns:
        Resolved torch.device
    """
    if use_vulkan:
        if check_vulkan_available():
            logger.info("Using Vulkan backend")
            return torch.device("vulkan")
        raise RuntimeError("Vulkan requested but not available on this system")

    if device_mode == "cpu":
        logger.info("CPU-only mode")
        return torch.device("cpu")

    if device_mode in ("gpu", "cpu+gpu"):
        if torch.cuda.is_available():
            idx = gpu_indices[0] if gpu_indices else 0
            device = torch.device(f"cuda:{idx}")
            logger.info(f"Using CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
            return device
        if device_mode == "gpu":
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        # cpu+gpu without CUDA is just CPU
        logger.warning("CUDA not available for cpu+gpu mode, using CPU only")
        return torch.device("cpu")

    # auto mode: CUDA > Vulkan > MPS > CPU
    if torch.cuda.is_available():
        idx = gpu_indices[0] if gpu_indices else 0
        device = torch.device(f"cuda:{idx}")
        logger.info(f"Auto-detected CUDA: {torch.cuda.get_device_name(idx)}")
        return device
    if check_vulkan_available():
        logger.info("Auto-detected Vulkan")
        return torch.device("vulkan")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected MPS (Apple Silicon)")
        return torch.device("mps")
    logger.info("No accelerator found, using CPU")
    return torch.device("cpu")


def calculate_layers_for_vram(
    model: torch.nn.Module,
    gpu_device: torch.device,
    vram_fraction: float = 0.80,
    bytes_per_param: int = 4,
) -> int:
    """Calculate how many transformer layers fit in GPU VRAM.

    Args:
        model: The full model (on CPU)
        gpu_device: Target GPU device
        vram_fraction: Fraction of VRAM to use (default 80%)
        bytes_per_param: Bytes per parameter (4=FP32, 2=FP16, 1=INT8)

    Returns:
        Number of transformer layers that can be placed on GPU
    """
    if not torch.cuda.is_available():
        return 0
    gpu_props = torch.cuda.get_device_properties(gpu_device)
    vram_total = gpu_props.total_memory / (1024 ** 3)
    vram_budget = vram_total * vram_fraction

    # Get transformer layers
    transformer_layers = None
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        transformer_layers = model.model.transformer.h
    if transformer_layers is None or len(transformer_layers) == 0:
        return 0

    num_layers = len(transformer_layers)

    # Estimate memory per layer
    total_layer_params = sum(p.numel() for p in transformer_layers.parameters())
    params_per_layer = total_layer_params / num_layers
    mem_per_layer_gb = (params_per_layer * bytes_per_param) / (1024 ** 3)

    # Estimate overhead (embeddings + lm_head)
    overhead_params = 0
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        if hasattr(model.model.transformer, "wte"):
            overhead_params += sum(p.numel() for p in model.model.transformer.wte.parameters())
        if hasattr(model.model.transformer, "wpe"):
            overhead_params += sum(p.numel() for p in model.model.transformer.wpe.parameters())
    if hasattr(model, "lm_head"):
        overhead_params += sum(p.numel() for p in model.lm_head.parameters())
    overhead_gb = (overhead_params * bytes_per_param) / (1024 ** 3)

    available_for_layers = vram_budget - overhead_gb
    if available_for_layers <= 0:
        return 0

    layers_on_gpu = min(num_layers, max(1, int(available_for_layers / mem_per_layer_gb)))
    return layers_on_gpu
