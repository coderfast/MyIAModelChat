import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, List

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    with open(ENV_PATH, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

_DEFAULT_GGUF = "models/Qwen2.5-1.5B-Instruct-Q4_0.gguf"
MODEL_GGUF_PATH = os.environ.get("MODEL_GGUF_PATH", "")
MODEL_PATH = str(BASE_DIR / (MODEL_GGUF_PATH if MODEL_GGUF_PATH else os.environ.get("MODEL_PATH", _DEFAULT_GGUF)))
MODEL_NAME = os.environ.get("MODEL_NAME", "local-Qwen2.5-1.5B-Instruct-Q4_0")
OLLAMA_VERSION = os.environ.get("OLLAMA_VERSION", "0.6.4")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "64"))
EXPORTED_DIR = os.environ.get("EXPORTED_DIR", str(BASE_DIR.parent / "models" / "exported"))

logger = logging.getLogger("uvicorn.error")


def _resolve_model_name(path: str) -> str:
    """Derive a human-readable model name from the GGUF filename."""
    stem = Path(path).stem
    for suffix in ("-Q4_0", "-Q4_K_M", "-Q5_0", "-Q5_K_M", "-Q8_0", "-F16"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _get_model_file_size(path: str) -> Optional[int]:
    """Return file size in bytes, or None if inaccessible."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def list_available_models() -> List[Dict]:
    """Scan EXPORTED_DIR for .gguf files and return metadata."""
    models = []
    exported_path = Path(EXPORTED_DIR)
    if not exported_path.exists():
        return models
    for gguf_file in exported_path.glob("*.gguf"):
        file_size = _get_model_file_size(str(gguf_file))
        models.append({
            "name": gguf_file.stem,
            "id": gguf_file.stem,
            "path": str(gguf_file),
            "file_size_mb": round(file_size / (1024 * 1024), 1) if file_size else None,
            "family": "gguf",
        })
    return models


class _LazyModel:
    """Proxy object that loads the real Llama instance on first attribute access.

    This avoids attempting to load a large model at import time during tests or
    when only metadata endpoints are needed.
    """

    def __init__(self):
        self._real: Optional[object] = None
        self._lock = threading.Lock()
        self._current_path: Optional[str] = None

    def _load(self, model_path: Optional[str] = None):
        target = model_path or MODEL_PATH
        with self._lock:
            if self._real is not None and self._current_path == target:
                return
            if Llama is None:
                raise RuntimeError("llama_cpp.Llama is not available in this environment")
            if not os.path.exists(target):
                raise FileNotFoundError(
                    f"Model file not found: {target}\n"
                    f"Set MODEL_GGUF_PATH or MODEL_PATH to a valid .gguf file."
                )
            try:
                logger.info(f"Loading model: {target}")
                self._real = Llama(model_path=target)
                self._current_path = target
                logger.info(f"Model loaded successfully: {_resolve_model_name(target)}")
            except Exception:
                logger.exception("failed loading model")
                raise

    def reload(self, model_path: Optional[str] = None) -> str:
        """Reload the model from disk. Returns the path of the loaded model."""
        target = model_path or MODEL_PATH
        with self._lock:
            if self._real is not None:
                try:
                    del self._real
                except Exception:
                    pass
                self._real = None
                self._current_path = None
        self._load(target)
        return target

    def __getattr__(self, name):
        self._load()
        return getattr(self._real, name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._real(*args, **kwargs)


model = _LazyModel()
