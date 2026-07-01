import logging
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # allow tests/usage without the native library present

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "Qwen2.5-1.5B-Instruct-Q4_0.gguf")
MODEL_NAME = "local-Qwen2.5-1.5B-Instruct-Q4_0"
OLLAMA_VERSION = "0.6.4"
CHUNK_SIZE = 64

logger = logging.getLogger("uvicorn.error")


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
            try:
                self._real = Llama(model_path=MODEL_PATH)
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
