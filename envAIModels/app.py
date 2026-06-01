from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from collections.abc import Iterable, Mapping, AsyncIterable
from llama_cpp import Llama
import json
import time
import logging
import asyncio
from functools import partial
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "Qwen2.5-1.5B-Instruct-Q4_0.gguf")
MODEL_NAME = "local-Qwen2.5-1.5B-Instruct-Q4_0"
OLLAMA_VERSION = "0.6.4"
CHUNK_SIZE = 64  # tamaño de chunk para streaming emulado si es necesario

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Inicializa el modelo con manejo de errores
try:
    model = Llama(model_path=MODEL_PATH)
except Exception as e:
    logger.exception("failed loading model")
    raise

# Pydantic models
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

# Utilities
def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content", "")
        if role in ("system", "assistant", "user"):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"user: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

def safe_get_text_from_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, (bytes, bytearray)):
        try:
            return item.decode(errors="ignore")
        except Exception:
            return ""
    if isinstance(item, dict):
        for key in ("text", "content", "token", "output"):
            val = item.get(key)
            if isinstance(val, (str, bytes)):
                return val if isinstance(val, str) else val.decode(errors="ignore")
        if "choices" in item and isinstance(item["choices"], list) and item["choices"]:
            return safe_get_text_from_item(item["choices"][0])
        if "message" in item:
            return safe_get_text_from_item(item["message"])
        if "generations" in item:
            return safe_get_text_from_item(item["generations"])
        try:
            return json.dumps(item, ensure_ascii=False)
        except Exception:
            return str(item)
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return safe_get_text_from_item(item[0])
    for attr in ("text", "token", "output", "content"):
        val = getattr(item, attr, None)
        if isinstance(val, (str, bytes)):
            return val if isinstance(val, str) else val.decode(errors="ignore")
    try:
        return str(item)
    except Exception:
        return ""

def is_sync_streamable(obj: Any) -> bool:
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, Mapping))

def is_async_streamable(obj: Any) -> bool:
    return isinstance(obj, AsyncIterable)

def call_model_with_signatures(prompt: str, max_tokens: int, temperature: float, top_p: float, stream: bool = False):
    """
    Try several possible model completion signatures and return whatever the model returns.
    Synchronous function suitable to be run inside run_in_executor.
    """
    attempts = [
        lambda: model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream),
        lambda: model.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream),
        lambda: model.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream),
        lambda: model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream),
        lambda: model.generate(prompt, stream=stream),
        lambda: model.generate(prompt),
    ]
    last_exc = None
    for fn in attempts:
        try:
            resp = fn()
            return resp
        except TypeError as te:
            last_exc = te
            continue
        except Exception:
            logger.exception("model.generate/create_completion failed on attempt")
            raise
    raise TypeError("no compatible call signature found for model.generate/create_completion") from last_exc

def consume_generator(resp):
    text = ""
    if isinstance(resp, Iterable) and not isinstance(resp, (str, bytes, Mapping)):
        for item in resp:
            text += safe_get_text_from_item(item)
        return text
    return safe_get_text_from_item(resp)


def model_create_compat_sync(prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0, stop: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Synchronous compatibility wrapper for various model.generate return shapes.
    - Calls call_model_with_signatures synchronously (suitable to run inside run_in_executor).
    - If the response is an iterable generator, consumes it and concatenates text parts.
    - Handles resp.generations, OpenAI-style dicts with choices/messages, plain strings, bytes.
    - Applies optional stop token truncation to the final text.
    Returns an OpenAI-like dict: {"choices":[{"text": text}]}
    """
    try:
        resp = call_model_with_signatures(prompt, max_tokens, temperature, top_p, False)
    except Exception as e:
        logger.exception("call_model_with_signatures failed")
        return {"choices": [{"text": f"ERROR_CALLING_MODEL: {e}"}]}

    text = ""

    # If resp is a sync iterable stream (generator) -> consume it
    try:
        if is_sync_streamable(resp):
            parts = []
            for item in resp:
                part = safe_get_text_from_item(item)
                if part:
                    parts.append(part)
            text = "".join(parts)
    except Exception:
        logger.exception("failed iterating model response (generator)")

    # If nothing collected, try common attributes (.generations, .text, dict choices, etc.)
    if not text:
        # .generations attribute (llama-cpp may use this)
        try:
            gens = getattr(resp, "generations", None)
            if gens:
                if isinstance(gens, list):
                    collected = []
                    for g in gens:
                        if isinstance(g, list) and len(g) > 0:
                            collected.append(safe_get_text_from_item(g[0]))
                        else:
                            collected.append(safe_get_text_from_item(g))
                    text = "".join(collected)
                else:
                    text = safe_get_text_from_item(gens)
        except Exception:
            logger.exception("failed reading resp.generations")

    # OpenAI-like dict with choices
    if not text and isinstance(resp, dict):
        try:
            if "choices" in resp and isinstance(resp["choices"], list) and len(resp["choices"]) > 0:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    text = c.get("text") or (c.get("message") or {}).get("content") or ""
                else:
                    text = safe_get_text_from_item(c)
            else:
                # fallback: try to stringify dict
                text = safe_get_text_from_item(resp)
        except Exception:
            text = safe_get_text_from_item(resp)

    # Plain string or bytes
    if not text and isinstance(resp, (str, bytes, bytearray)):
        text = safe_get_text_from_item(resp)

    # Last resort: generic extraction
    if not text:
        try:
            text = safe_get_text_from_item(resp) or str(resp)
        except Exception:
            text = f"<unserializable response of type {type(resp).__name__}>"

    # Apply stop tokens truncation if provided
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break

    return {"choices": [{"text": text}]}

def llama_create_stream_sync(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None):
    """
    Synchronous generator that either consumes a streaming model.generate or falls back
    to chunking a full generation. Designed to run in executor.
    """
    gen = None
    for attempt in (
        lambda: model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True),
        lambda: model.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True),
        lambda: model.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True),
        lambda: model.generate(prompt, stream=True),
    ):
        try:
            candidate = attempt()
            if is_sync_streamable(candidate):
                gen = candidate
                break
            if is_async_streamable(candidate):
                logger.warning("model returned async iterable for streaming; falling back to non-stream path")
                gen = None
                break
        except TypeError:
            continue
        except Exception:
            logger.exception("streaming attempt failed")
            gen = None

    if gen is not None:
        buffer = ""
        try:
            for item in gen:
                chunk = safe_get_text_from_item(item)
                buffer += chunk
                if stop:
                    for s in stop:
                        idx = buffer.find(s)
                        if idx != -1:
                            yield buffer[:idx]
                            return
                if chunk:
                    yield chunk
            return
        except Exception:
            logger.exception("error while iterating streaming generator; falling back")

    # Fallback: call full generate and chunk the result
    out = model_create_compat_sync(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    text = out["choices"][0].get("text", "") or ""
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    for i in range(0, len(text), CHUNK_SIZE):
        yield text[i:i+CHUNK_SIZE]
        time.sleep(0.01)

# Async wrapper to stream results from a sync generator running in executor
async def stream_generator_wrapper_sync_to_async(sync_gen_func, *args, media_type="application/x-ndjson"):
    loop = asyncio.get_running_loop()
    # Create the sync generator in executor
    gen = await loop.run_in_executor(None, partial(sync_gen_func, *args))
    # Iterate over it by repeatedly calling next() in executor
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if chunk is None:
            break
        yield chunk

# Metadata endpoints
@app.get("/api/version")
def version():
    return {"name": MODEL_NAME, "version": OLLAMA_VERSION}

@app.get("/api/models")
def models():
    return [
        {
            "name": MODEL_NAME,
            "description": "Qwen2.5-1.5B Instruct (gguf, quantized)",
            "id": MODEL_NAME,
            "size": "1.5B",
            "family": "qwen"
        }
    ]

@app.get("/api/tags")
def tags():
    return {"tags": ["local", "llama_cpp", "gguf", "chat"], "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok"}

# /api/generate
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.prompt:
        return JSONResponse({"error": "no prompt provided"}, status_code=400)

    if req.stream:
        async def event_stream():
            async for chunk in stream_generator_wrapper_sync_to_async(llama_create_stream_sync, req.prompt, req.max_tokens or 256, req.temperature or 0.0, req.top_p or 1.0, req.stop):
                yield json.dumps({"id": None, "object": "text.chunk", "text": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"id": None, "object": "text.complete"}, ensure_ascii=False) + "\n"
        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    # non-streaming: run generation in executor
    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, req.prompt, req.max_tokens or 256, req.temperature or 0.0, req.top_p or 1.0))
    except Exception as e:
        logger.exception("generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    text = out["choices"][0].get("text", "") or ""
    if req.stop:
        for s in req.stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    return {"id": None, "object": "text.completion", "model": MODEL_NAME, "choices": [{"text": text, "index": 0}], "raw": out}

# /api/chat (proxy)
@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "no messages provided"}, status_code=400)
    prompt = build_prompt_from_messages(messages)
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stop = data.get("stop")

    if stream:
        async def event_stream():
            first_sent = True
            async for chunk in stream_generator_wrapper_sync_to_async(llama_create_stream_sync, prompt, max_tokens, temperature, top_p, stop):
                payload = {"id": None, "object": "chat.completion.chunk", "delta": {"role": "assistant" if first_sent else None, "content": chunk}}
                if payload["delta"]["role"] is None:
                    del payload["delta"]["role"]
                yield json.dumps(payload, ensure_ascii=False) + "\n"
                first_sent = False
            yield json.dumps({"id": None, "object": "chat.completion.complete"}, ensure_ascii=False) + "\n"
        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, max_tokens, temperature, top_p))
    except Exception as e:
        logger.exception("chat generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    text = out["choices"][0].get("text", "") or ""
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

# /api/chat/completions (ollama-style)
@app.post("/api/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages:
        return JSONResponse({"error": "no messages provided"}, status_code=400)
    prompt = build_prompt_from_messages([m.dict() for m in req.messages])
    if req.stream:
        async def event_stream():
            first_sent = True
            async for chunk in stream_generator_wrapper_sync_to_async(llama_create_stream_sync, prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0, req.stop):
                payload = {"id": None, "object": "chat.completion.chunk", "delta": {"role": "assistant" if first_sent else None, "content": chunk}}
                if payload["delta"]["role"] is None:
                    del payload["delta"]["role"]
                yield json.dumps(payload, ensure_ascii=False) + "\n"
                first_sent = False
            yield json.dumps({"id": None, "object": "chat.completion.complete"}, ensure_ascii=False) + "\n"
        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0))
    except Exception as e:
        logger.exception("chat completions failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    text = out["choices"][0].get("text", "") or ""
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

# v1 endpoints for compatibility
@app.get("/v1/health")
def v1_health():
    return {"status": "ok"}

@app.get("/v1/models")
def v1_models():
    return models()

@app.get("/v1/version")
def v1_version():
    return version()
