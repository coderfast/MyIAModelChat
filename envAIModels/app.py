from fastapi import FastAPI
import logging

from .routers_api import router as api_router
from .routers_v1 import router as v1_router
from .model import MODEL_NAME, OLLAMA_VERSION

app = FastAPI(title="envAIModels", version=OLLAMA_VERSION)
logger = logging.getLogger("uvicorn.error")

app.include_router(api_router)
app.include_router(v1_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}
