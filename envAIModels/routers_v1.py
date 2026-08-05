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
    parse_thinking_response,
)
from .model import (
    model, MODEL_NAME, MODEL_PATH, OLLAMA_VERSION,
    _resolve_model_name, _get_model_file_size,
    list_available_models, EXPORTED_DIR,
)

router = APIRouter(prefix="/v1")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/models")
def models():
    file_size = _get_model_file_size(MODEL_PATH)
    file_size_mb = round(file_size / (1024 * 1024), 1) if file_size else None
    return [
        {
            "name": MODEL_NAME,
            "id": MODEL_NAME,
            "description": f"{_resolve_model_name(MODEL_PATH)} (gguf, quantized)",
            "model_file": MODEL_PATH,
            "file_size_mb": file_size_mb,
            "family": "gguf",
            "source": "envAIModels",
        }
    ]


@router.get("/models/available")
def models_available():
    """List all .gguf files available in the exported models directory."""
    return list_available_models()


@router.post("/models/reload")
async def models_reload(request: Request):
    """Reload the current model or switch to a different model.

    Body (optional):
        model_path: str  # Path to .gguf file to load
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    model_path = data.get("model_path") if data else None

    try:
        loaded_path = model.reload(model_path)
        file_size = _get_model_file_size(loaded_path)
        file_size_mb = round(file_size / (1024 * 1024), 1) if file_size else None
        return {
            "status": "ok",
            "model": _resolve_model_name(loaded_path),
            "model_file": loaded_path,
            "file_size_mb": file_size_mb,
        }
    except FileNotFoundError as e:
        return Response(
            content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            status_code=404,
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": f"Failed to reload model: {str(e)}"}, ensure_ascii=False).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            status_code=500,
        )


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
    include_thinking = data.get("include_thinking", False)

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

    parsed = parse_thinking_response(text)
    result = {"id": None, "object": "text.completion", "model": MODEL_NAME, "choices": [{"text": parsed['response'] if not include_thinking else text, "index": 0}], "raw": out}
    if include_thinking and parsed['thinking']:
        result['reasoning'] = parsed['thinking']
    return result


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
    response = build_chat_completion_response(text, include_thinking=include_thinking)
    return response
