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
            "source": "ServerFastAPI",
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
        api_key: str     # API key for authentication
    """
    import os as _os
    api_key = _os.environ.get("MODEL_RELOAD_API_KEY", "")
    if not api_key:
        return Response(
            content=json.dumps({"error": "Model reload is disabled (set MODEL_RELOAD_API_KEY to enable)"}).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            status_code=403,
        )

    try:
        data = await request.json()
    except Exception:
        data = {}

    provided_key = (data or {}).get("api_key", "")
    import hmac
    if not hmac.compare_digest(provided_key, api_key):
        return Response(
            content=json.dumps({"error": "Invalid or missing api_key"}).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            status_code=403,
        )

    model_path = data.get("model_path") if data else None

    # Validate model_path is within allowed directories
    if model_path:
        allowed_dirs = [str(MODEL_PATH.parent), str(EXPORTED_DIR)] if EXPORTED_DIR else [str(MODEL_PATH.parent)]
        import pathlib
        resolved = pathlib.Path(model_path).resolve()
        if not any(resolved.is_relative_to(pathlib.Path(d)) for d in allowed_dirs):
            return Response(
                content=json.dumps({"error": "model_path must be within the models or exported directory"}).encode("utf-8"),
                media_type="application/json; charset=utf-8",
                status_code=403,
            )

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
            content=json.dumps({"error": "Model file not found"}).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            status_code=404,
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": "Failed to reload model"}).encode("utf-8"),
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
            yield json.dumps({"id": None, "object": "text.chunk", "text": "", "finish_reason": "stop"}) + "\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

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
            yield json.dumps({'id': None, 'object': 'chat.completion.chunk', 'delta': {}, 'finish_reason': 'stop'}) + '\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    try:
        out = await loop.run_in_executor(None, partial(model_create_compat_sync, prompt, max_tokens, temperature, top_p, stop))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)

    text = out["choices"][0].get("text", "") or ""
    text = truncate_text_by_stop(text, stop)
    response = build_chat_completion_response(text, include_thinking=include_thinking)
    return response


@router.post("/chat/completions/agentic")
async def v1_chat_completions_agentic(request: Request):
    """Agentic chat completions endpoint with tool call support.
    
    Supports:
    - tool_calls: List of tool calls in the response
    - observations: List of observation results
    - thinking: Chain-of-thought reasoning
    - agent_enabled: Enable agentic capabilities
    """
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
    include_thinking = data.get("include_thinking", True)
    agent_enabled = data.get("agent_enabled", True)
    
    if stop is None:
        stop = ["\nuser:", "\nassistant:"]

    # Use agentic streaming if agent is enabled
    if stream and agent_enabled:
        async def agentic_event_stream():
            import asyncio
            from inference.chat_engine import get_chat_engine_instance, stream_chat_agentic
            
            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(None, get_chat_engine_instance)
            response = await loop.run_in_executor(None, lambda: engine.generate_response(prompt, full=True))
            
            if isinstance(response, dict):
                for chunk in stream_chat_agentic(response):
                    yield chunk
            else:
                # Fallback to regular streaming
                yield json.dumps({
                    "id": None,
                    "object": "chat.completion.chunk",
                    "type": "response",
                    "delta": {"content": response},
                    "finish_reason": None
                }) + '\n'
                yield json.dumps({
                    "id": None,
                    "object": "chat.completion.chunk",
                    "delta": {},
                    "finish_reason": "stop"
                }) + '\n'
                yield 'data: [DONE]\n\n'

        return StreamingResponse(agentic_event_stream(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    try:
        from inference.chat_engine import get_chat_engine_instance
        
        engine = await loop.run_in_executor(None, get_chat_engine_instance)
        response = await loop.run_in_executor(None, lambda: engine.generate_response(prompt, full=True))
        
        # Build agentic response
        result = {
            "id": None,
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.get("response", "") if isinstance(response, dict) else response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Add agentic fields if present
        if isinstance(response, dict):
            if response.get("thinking"):
                result["choices"][0]["message"]["reasoning"] = response["thinking"]
            if response.get("tool_calls"):
                result["choices"][0]["message"]["tool_calls"] = response["tool_calls"]
            if response.get("observations"):
                result["choices"][0]["message"]["observations"] = response["observations"]
        
        return result
        
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), media_type="application/json; charset=utf-8", status_code=500)
