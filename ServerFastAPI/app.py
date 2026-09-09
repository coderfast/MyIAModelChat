from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routers_api import router as api_router
from .routers_v1 import router as v1_router
from .model import MODEL_NAME, MODEL_PATH, OLLAMA_VERSION

app = FastAPI(title="ServerFastAPI", version=OLLAMA_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

if __name__ == "__main__":
    logger.info(f"Starting ServerFastAPI server")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Path:  {MODEL_PATH}")

app.include_router(api_router)
app.include_router(v1_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "model_path": MODEL_PATH}
