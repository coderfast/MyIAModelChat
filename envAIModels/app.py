from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from llama_cpp import Llama
import json
import time

MODEL_PATH = "models/Hunyuan-PythonGOD-0.5B.Q4_K_M.gguf"
MODEL_NAME = "local-hunyuan"
OLLAMA_VERSION = "0.6.4"

app = FastAPI()
model = Llama(model_path=MODEL_PATH)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

@app.get("/api/version")
def version():
    return {"name": MODEL_NAME, "version": OLLAMA_VERSION}

@app.get("/api/models")
def models():
    return [
        {
            "name": MODEL_NAME,
            "description": "Hunyuan-PythonGOD-0.5B (gguf, quantized)",
            "id": MODEL_NAME,
            "size": "0.5B",
            "family": "hunyuan"
        }
    ]

@app.get("/api/tags")
def tags():
    return {"tags": ["local", "llama_cpp", "gguf", "chat"], "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok"}

def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role.lower() in ("system", "assistant", "user"):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"user: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

def llama_create_stream(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]):
    out = model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    text = out["choices"][0]["text"]
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    chunk_size = 64
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]
        time.sleep(0.01)

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.stream:
        def gen():
            for chunk in llama_create_stream(req.prompt, req.max_tokens or 256, req.temperature or 0.0, req.top_p or 1.0, req.stop):
                yield json.dumps({"id": None, "object": "text.chunk", "text": chunk}) + "\n"
            yield json.dumps({"id": None, "object": "text.complete"}) + "\n"
        return StreamingResponse(gen(), media_type="application/json")
    out = model.create(prompt=req.prompt, max_tokens=req.max_tokens or 256, temperature=req.temperature or 0.0)
    text = out["choices"][0]["text"]
    if req.stop:
        for s in req.stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    return {"id": None, "model": MODEL_NAME, "text": text, "raw": out}

# proxy /api/chat to your internal chat handler (uses llama_cpp locally)
@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    # expected shape: {"model": "...", "messages": [{"role":"user","content":"..."}], ...}
    messages = data.get("messages", [])
    prompt = build_prompt_from_messages(messages)
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stop = data.get("stop")
    if stream:
        def gen():
            for chunk in llama_create_stream(prompt, max_tokens, temperature, top_p, stop):
                payload = {"id": None, "object": "chat.completion.chunk", "delta": {"role": "assistant", "content": chunk}}
                yield json.dumps(payload) + "\n"
            yield json.dumps({"id": None, "object": "chat.completion.complete"}) + "\n"
        return StreamingResponse(gen(), media_type="application/json")
    out = model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    text = out["choices"][0]["text"]
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    response = {
        "id": None,
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ]
    }
    return JSONResponse(response)

@app.post("/api/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = build_prompt_from_messages([m.dict() for m in req.messages])
    if req.stream:
        def gen():
            for chunk in llama_create_stream(prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0, req.stop):
                yield json.dumps({"id": None, "object": "chat.completion.chunk", "delta": {"role": "assistant", "content": chunk}}) + "\n"
            yield json.dumps({"id": None, "object": "chat.completion.complete"}) + "\n"
        return StreamingResponse(gen(), media_type="application/json")
    out = model.create(prompt=prompt, max_tokens=req.max_tokens or 512, temperature=req.temperature or 0.0)
    text = out["choices"][0]["text"]
    if req.stop:
        for s in req.stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    response = {
        "id": None,
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }
        ]
    }
    return JSONResponse(response)

@app.get("/v1/health")
def v1_health():
    return {"status": "ok"}

@app.get("/v1/models")
def v1_models():
    return models()

@app.get("/v1/version")
def v1_version():
    return version()
