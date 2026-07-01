import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, Response
from functools import partial

from .utils import (
    build_prompt_from_messages,
    normalize_stop,
    truncate_text_by_stop,
    extract_v1_prompt,
    stream_generator_wrapper_sync_to_async,
    llama_create_stream_sync,
    model_create_compat_sync,
    build_text_completion_response,
    build_chat_completion_response,
)
from .model import MODEL_NAME, OLLAMA_VERSION

router = APIRouter(prefix="/v1")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/models")
def models():
    return [
        {
            "name": MODEL_NAME,
            "description": "Qwen2.5-1.5B Instruct (gguf, quantized)",
            "id": MODEL_NAME,
            "size": "1.5B",
            "family": "qwen",
        }
    ]


@router.get("/version")
def version():
    return {"name": MODEL_NAME, "version": OLLAMA_VERSION}


@router.post("/completions")
async def v1_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return Response(content=json.dumps({"error": "invalid JSON body"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    prompt = extract_v1_prompt(data)
    if not prompt:
        return Response(content=json.dumps({"error": "no prompt provided"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stop = normalize_stop(data.get("stop"))
    stream = data.get("stream", False)

    if stream:
        async def event_stream():
            async for chunk in stream_generator_wrapper_sync_to_async(
                llama_create_stream_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop,
            ):
                yield json.dumps({"id": None, "object": "text.chunk", "text": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"id": None, "object": "text.complete"}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, max_tokens, temperature, top_p, stop))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)

    text = out["choices"][0].get("text", "") or ""
    text = truncate_text_by_stop(text, stop)
    return {"id": None, "object": "text.completion", "model": MODEL_NAME, "choices": [{"text": text, "index": 0}], "raw": out}


@router.post("/chat/completions")
async def v1_chat_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return Response(content=json.dumps({"error": "invalid JSON body"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    messages = data.get("messages", [])
    if not messages:
        return Response(content=json.dumps({"error": "no messages provided"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    prompt = build_prompt_from_messages(messages)
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stop = normalize_stop(data.get("stop"))
    if stop is None:
        stop = ["\nuser:", "\nassistant:"]

    if stream:
        async def event_stream():
            first_sent = True
            async for chunk in stream_generator_wrapper_sync_to_async(
                llama_create_stream_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop,
            ):
                payload = {
                    "id": None,
                    "object": "chat.completion.chunk",
                    "delta": {"role": "assistant" if first_sent else None, "content": chunk},
                }
                if payload["delta"].get("role") is None:
                    del payload["delta"]["role"]
                yield json.dumps(payload, ensure_ascii=False) + "\n"
                first_sent = False
            yield json.dumps({"id": None, "object": "chat.completion.complete"}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, max_tokens, temperature, top_p, stop))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)

    text = out["choices"][0].get("text", "") or ""
    text = truncate_text_by_stop(text, stop)
    response = build_chat_completion_response(text)
    return response
