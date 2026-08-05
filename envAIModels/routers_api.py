import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, Response
from functools import partial

from .schemas import GenerateRequest, ChatRequest
from .utils import (
    build_prompt_from_messages,
    normalize_stop,
    truncate_text_by_stop,
    remove_excessive_repetition,
    stream_generator_wrapper_sync_to_async,
    llama_create_stream_sync,
    model_create_compat_sync,
    build_text_completion_response,
    build_chat_completion_response,
    parse_thinking_response,
)
from .model import MODEL_NAME, OLLAMA_VERSION, list_available_models

router = APIRouter(prefix="/api")


@router.get("/version")
def version():
    return {"name": MODEL_NAME, "version": OLLAMA_VERSION}


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


@router.get("/tags")
def tags():
    return {"tags": ["local", "llama_cpp", "gguf", "chat"], "model": MODEL_NAME}


@router.get("/models/available")
def api_models_available():
    """List all .gguf files available in the exported models directory."""
    return list_available_models()


@router.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt:
        return Response(content=json.dumps({"error": "no prompt provided"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    if req.stream:
        async def event_stream():
            async for chunk in stream_generator_wrapper_sync_to_async(
                llama_create_stream_sync,
                req.prompt,
                req.max_tokens or 256,
                req.temperature or 0.0,
                req.top_p or 1.0,
                req.stop,
            ):
                yield json.dumps({"id": None, "object": "text.chunk", "text": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"id": None, "object": "text.complete"}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(
            None,
            partial(
                model_create_compat_sync,
                req.prompt,
                req.max_tokens or 256,
                req.temperature or 0.0,
                req.top_p or 1.0,
                req.stop,
            ),
        )
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)

    text = out["choices"][0].get("text", "") or ""
    text = truncate_text_by_stop(text, req.stop)

    parsed = parse_thinking_response(text)
    result = {"id": None, "object": "text.completion", "model": MODEL_NAME, "choices": [{"text": parsed['response'] if not req.include_thinking else text, "index": 0}], "raw": out}
    if req.include_thinking and parsed['thinking']:
        result['reasoning'] = parsed['thinking']
    return result


@router.post("/chat")
async def api_chat(request: Request):
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
    include_thinking = data.get("include_thinking", False)
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
    text = remove_excessive_repetition(text)
    response = build_chat_completion_response(text, include_thinking=include_thinking)
    return response


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages:
        return Response(content=json.dumps({"error": "no messages provided"}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=400)

    prompt = build_prompt_from_messages([m.dict() for m in req.messages])
    stop = normalize_stop(req.stop)
    if stop is None:
        stop = ["\nuser:", "\nassistant:"]

    if req.stream:
        async def event_stream():
            first_sent = True
            async for chunk in stream_generator_wrapper_sync_to_async(
                llama_create_stream_sync,
                prompt,
                req.max_tokens or 512,
                req.temperature or 0.0,
                req.top_p or 1.0,
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
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0, stop))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)

    text = out["choices"][0].get("text", "") or ""
    text = truncate_text_by_stop(text, stop)
    text = remove_excessive_repetition(text)
    response = build_chat_completion_response(text, include_thinking=req.include_thinking)
    return response
