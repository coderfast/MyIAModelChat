import json
import time
import asyncio
import inspect
import logging
from functools import partial
from typing import Any, Dict, List, Optional
from collections.abc import Iterable, Mapping, AsyncIterable

try:
    from .model import model, MODEL_NAME, CHUNK_SIZE
except ImportError:
    from model import model, MODEL_NAME, CHUNK_SIZE

logger = logging.getLogger("uvicorn.error")

# Application helpers

def make_json_response(obj: Any, status_code: int = 200):
    try:
        body = json.dumps(obj, ensure_ascii=False)
    except Exception:
        try:
            body = str(obj)
        except Exception:
            body = "{}"
    return {
        "content": body.encode("utf-8"),
        "media_type": "application/json; charset=utf-8",
        "status_code": status_code,
    }


def normalize_stop(stop: Optional[Any]) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(item) for item in stop if item is not None]
    return [str(stop)]


def parse_thinking_response(text: str) -> Dict[str, Optional[str]]:
    """Parse <think> tags from response text.

    Returns:
        Dict with 'thinking' and 'response' keys.
        If no thinking tags found, thinking is None and response is the full text.
    """
    if '<think>' not in text or '</think>' not in text:
        return {'thinking': None, 'response': text}
    try:
        thinking = text.split('<think>')[1].split('</think>')[0]
        response = text.split('</think>')[1].strip()
        return {'thinking': thinking, 'response': response}
    except (IndexError, ValueError):
        return {'thinking': None, 'response': text}


def truncate_text_by_stop(text: str, stop: Optional[List[str]]) -> str:
    if not text or not stop:
        return text
    for token in stop:
        if token and token in text:
            idx = text.find(token)
            if idx != -1:
                return text[:idx]
    return text


def remove_excessive_repetition(text: str, max_repetitions: int = 1) -> str:
    if not text:
        return text
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen_sentences = {}
    filtered = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        normalized = re.sub(r'\s+', ' ', sentence.lower())
        seen_sentences[normalized] = seen_sentences.get(normalized, 0) + 1
        if seen_sentences[normalized] <= max_repetitions:
            filtered.append(sentence)
    return ' '.join(filtered).strip()


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
    if isinstance(item, (list, tuple)) and item:
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


def call_model_with_signatures(prompt: str, max_tokens: int, temperature: float, top_p: float, stream: bool = False, stop: Optional[List[str]] = None):
    attempts = [
        lambda: model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=stream),
        lambda: model.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=stream),
        lambda: model.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=stream),
        lambda: model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=stream),
        lambda: model.generate(prompt, stop=stop, stream=stream),
        lambda: model.generate(prompt),
    ]
    last_exc = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as te:
            last_exc = te
            continue
        except Exception:
            logger.exception("model.generate/create_completion failed on attempt")
            raise
    raise TypeError("no compatible call signature found for model.generate/create_completion") from last_exc


def consume_generator(resp: Any) -> str:
    if is_sync_streamable(resp):
        text = ""
        for item in resp:
            text += safe_get_text_from_item(item)
        return text
    return safe_get_text_from_item(resp)


def model_create_compat_sync(prompt: str, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0, stop: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        resp = call_model_with_signatures(prompt, max_tokens, temperature, top_p, False, stop=stop)
    except Exception as e:
        logger.exception("call_model_with_signatures failed")
        return {"choices": [{"text": f"ERROR_CALLING_MODEL: {e}"}]}

    text = ""
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

    if not text:
        try:
            gens = getattr(resp, "generations", None)
            if gens:
                if isinstance(gens, list):
                    collected = []
                    for g in gens:
                        if isinstance(g, list) and g:
                            collected.append(safe_get_text_from_item(g[0]))
                        else:
                            collected.append(safe_get_text_from_item(g))
                    text = "".join(collected)
                else:
                    text = safe_get_text_from_item(gens)
        except Exception:
            logger.exception("failed reading resp.generations")

    if not text and isinstance(resp, dict):
        try:
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    text = c.get("text") or (c.get("message") or {}).get("content") or ""
                else:
                    text = safe_get_text_from_item(c)
            else:
                text = safe_get_text_from_item(resp)
        except Exception:
            text = safe_get_text_from_item(resp)

    if not text and isinstance(resp, (str, bytes, bytearray)):
        text = safe_get_text_from_item(resp)

    if not text:
        try:
            text = safe_get_text_from_item(resp) or str(resp)
        except Exception:
            text = f"<unserializable response of type {type(resp).__name__}>"

    if stop:
        text = truncate_text_by_stop(text, stop)

    return {"choices": [{"text": text}]}


def llama_create_stream_sync(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None):
    gen = None
    for attempt in (
        lambda: model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=True),
        lambda: model.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=True),
        lambda: model.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=True),
        lambda: model.generate(prompt, stop=stop, stream=True),
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
                        if s and s in buffer:
                            yield buffer[: buffer.find(s)]
                            return
                if chunk:
                    yield chunk
            return
        except Exception:
            logger.exception("error while iterating streaming generator; falling back")

    out = model_create_compat_sync(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    text = out["choices"][0].get("text", "") or ""
    if stop:
        text = truncate_text_by_stop(text, stop)
    for i in range(0, len(text), CHUNK_SIZE):
        yield text[i : i + CHUNK_SIZE]
        time.sleep(0.01)


async def stream_generator_wrapper_sync_to_async(sync_gen_func, *args, media_type: str = "application/x-ndjson"):
    loop = asyncio.get_running_loop()
    gen = await loop.run_in_executor(None, partial(sync_gen_func, *args))
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if chunk is None:
            break
        yield chunk


def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content", "")
        if role in ("system", "assistant", "user"):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"user: {content}")
    parts.append("assistant: ")
    return "\n".join(parts)


def build_text_completion_response(text: str, model_name: str = MODEL_NAME) -> Dict[str, Any]:
    return {"id": None, "object": "text.completion", "model": model_name, "choices": [{"text": text, "index": 0}]}


def build_chat_completion_response(text: str, model_name: str = MODEL_NAME, include_thinking: bool = False) -> Dict[str, Any]:
    parsed = parse_thinking_response(text)
    message = {"role": "assistant", "content": parsed['response']}
    if include_thinking and parsed['thinking']:
        message['reasoning'] = parsed['thinking']
    return {
        "id": None,
        "object": "chat.completion",
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def extract_v1_prompt(data: dict) -> Optional[str]:
    prompt = data.get("prompt")
    if isinstance(prompt, str):
        return prompt
    raw_input = data.get("input")
    if isinstance(raw_input, str):
        return raw_input
    return None
