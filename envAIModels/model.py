import logging
import os
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # allow tests/usage without the native library present

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

# Model path: MODEL_GGUF_PATH takes priority, falls back to MODEL_PATH
_DEFAULT_GGUF = "models/Qwen2.5-1.5B-Instruct-Q4_0.gguf"
MODEL_GGUF_PATH = os.environ.get("MODEL_GGUF_PATH", "")
MODEL_PATH = str(BASE_DIR / (MODEL_GGUF_PATH if MODEL_GGUF_PATH else os.environ.get("MODEL_PATH", _DEFAULT_GGUF)))
MODEL_NAME = os.environ.get("MODEL_NAME", "local-Qwen2.5-1.5B-Instruct-Q4_0")
OLLAMA_VERSION = os.environ.get("OLLAMA_VERSION", "0.6.4")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "64"))

logger = logging.getLogger("uvicorn.error")


def _resolve_model_name(path: str) -> str:
    """Derive a human-readable model name from the GGUF filename."""
    stem = Path(path).stem
    # Strip common quantization suffixes for a cleaner name
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


class _LazyModel:
    """Proxy object that loads the real Llama instance on first attribute access.

    This avoids attempting to load a large model at import time during tests or
    when only metadata endpoints are needed.
    """

    def __init__(self):
        self._real: Optional[object] = None

    def _load(self):
        if self._real is None:
            if Llama is None:
                raise RuntimeError("llama_cpp.Llama is not available in this environment")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file not found: {MODEL_PATH}\n"
                    f"Set MODEL_GGUF_PATH or MODEL_PATH to a valid .gguf file."
                )
            try:
                logger.info(f"Loading model: {MODEL_PATH}")
                self._real = Llama(model_path=MODEL_PATH)
                logger.info(f"Model loaded successfully: {MODEL_NAME}")
            except Exception:
                logger.exception("failed loading model")
                raise

    def __getattr__(self, name):
        self._load()
        return getattr(self._real, name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._real(*args, **kwargs)


# Expose a `model` object that will lazily instantiate the real model when used.
model = _LazyModel()
