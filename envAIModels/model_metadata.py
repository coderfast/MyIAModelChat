import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional


def _format_value(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return f"<{type(value).__name__}>"
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _format_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_format_value(v, depth + 1) for v in value]
    if hasattr(value, "to_dict") and not isinstance(value, str):
        try:
            return _format_value(value.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _format_value(v, depth + 1) for k, v in vars(value).items() if not k.startswith("_")}
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _extract_config(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return _format_value(obj)
    if hasattr(obj, "to_dict"):
        try:
            return _format_value(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _format_value(vars(obj))
    return _format_value(obj)


def infer_model_layer_count(model: Any) -> Optional[int]:
    def _get_int(value: Any) -> Optional[int]:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                return int(value)
        try:
            iv = int(value)
            return iv if iv > 0 else None
        except Exception:
            return None

    def search_for_layer_count(root: Any) -> Optional[int]:
        if root is None:
            return None
        if isinstance(root, Mapping):
            for key, value in root.items():
                if any(k in key.lower() for k in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count")):
                    if (count := _get_int(value)) is not None:
                        return count
            for key in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count", "qwen2.block_count"):
                if key in root:
                    if (count := _get_int(root[key])) is not None:
                        return count
            return None
        for key in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count"):
            if hasattr(root, key):
                try:
                    if (count := _get_int(getattr(root, key))) is not None:
                        return count
                except Exception:
                    continue
        for key in dir(root):
            if any(k in key.lower() for k in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count")):
                try:
                    if (count := _get_int(getattr(root, key))) is not None:
                        return count
                except Exception:
                    continue
        return None

    for root in (
        getattr(model, "metadata", None),
        getattr(model, "model_params", None),
        getattr(model, "config", None),
        getattr(model, "context_params", None),
        getattr(model, "_model", None),
    ):
        if (count := search_for_layer_count(root)) is not None:
            return count

    names = []
    if hasattr(model, "state_dict"):
        try:
            state = model.state_dict() if callable(model.state_dict) else model.state_dict
            if isinstance(state, Mapping):
                names = list(state.keys())
            elif isinstance(state, (list, tuple)):
                names = [name for name, _ in state if isinstance(name, str)]
        except Exception:
            pass

    if not names and hasattr(model, "tensors"):
        try:
            tensors = model.tensors
            if isinstance(tensors, (list, tuple)):
                names = [getattr(t, "name", str(t)) for t in tensors]
        except Exception:
            pass

    if names:
        import re
        layer_indices = set()
        patterns = [
            r"\bblk\.(\d+)\b",
            r"\blayer\.(\d+)\b",
            r"\btransformer\.h\.(\d+)\b",
            r"\bencoder\.layers\.(\d+)\b",
            r"\bdecoder\.layers\.(\d+)\b",
            r"\bblocks\.(\d+)\b",
            r"\bblock\.(\d+)\b",
        ]
        for name in names:
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer_indices.add(int(match.group(1)))
        if layer_indices:
            return max(layer_indices) + 1

        prefix_counts = {}
        for name in names:
            for prefix in ("blk.", "layer.", "transformer.h.", "encoder.layers.", "decoder.layers.", "block."):
                if prefix in name:
                    try:
                        part = name.split(prefix, 1)[1]
                        idx = int(part.split(".", 1)[0])
                        prefix_counts[prefix] = max(prefix_counts.get(prefix, -1), idx)
                    except Exception:
                        continue
        if prefix_counts:
            return max(prefix_counts.values()) + 1

    return None


def _find_attribute_names(model: Any, keywords: List[str]) -> List[str]:
    return sorted({
        name for name in dir(model)
        if not name.startswith("_") and any(keyword in name.lower() for keyword in keywords)
    })


def gather_model_metadata(model: Any, max_weights: int = 50, max_tensors: int = 20) -> Dict[str, Any]:
    metadata = {
        "model_name": getattr(model, "model_path", None) or None,
        "layer_count": infer_model_layer_count(model),
        "model_params": _extract_config(getattr(model, "model_params", None)),
        "context_params": _extract_config(getattr(model, "context_params", None)),
        "metadata": _extract_config(getattr(model, "metadata", None)),
        "config": _extract_config(getattr(model, "config", None)),
        "available_attributes": _find_attribute_names(model, ["model", "config", "context", "layers", "blocks", "tensor", "weight", "embed", "attention"]),
    }

    tensors = []
    if hasattr(model, "tensors"):
        try:
            for t in getattr(model, "tensors")[:max_tensors]:
                tensors.append(str(t))
        except Exception:
            pass
    metadata["tensors"] = tensors

    weights = []
    if hasattr(model, "state_dict"):
        try:
            state = model.state_dict() if callable(model.state_dict) else model.state_dict
            if isinstance(state, Mapping):
                for i, name in enumerate(state.keys()):
                    if i >= max_weights:
                        break
                    weights.append(str(name))
        except Exception:
            pass
    metadata["weights"] = weights

    return metadata


def print_model_metadata(model: Any) -> None:
    print(json.dumps(gather_model_metadata(model), ensure_ascii=False, indent=2))
