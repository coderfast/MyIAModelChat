from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from collections.abc import Iterable, Mapping, AsyncIterable
from llama_cpp import Llama
import json
import time
import logging
import asyncio
import inspect
from functools import partial
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "Qwen2.5-1.5B-Instruct-Q4_0.gguf")
MODEL_NAME = "local-Qwen2.5-1.5B-Instruct-Q4_0"
OLLAMA_VERSION = "0.6.4"
CHUNK_SIZE = 64  # tamaño de chunk para streaming emulado si es necesario
DEFAULT_STOP = ["\nuser:", "\nassistant:"]

app = FastAPI()
logger = logging.getLogger("uvicorn.error")


# Helper to create UTF-8 JSON responses (preserve non-ASCII chars)
def make_json_response(obj, status_code: int = 200):
    try:
        body = json.dumps(obj, ensure_ascii=False)
    except Exception:
        try:
            body = str(obj)
        except Exception:
            body = "{}"
    return Response(content=body.encode("utf-8"), media_type="application/json; charset=utf-8", status_code=status_code)

# Inicializa el modelo con manejo de errores
try:
    model = Llama(model_path=MODEL_PATH)

    def call_no_args(value):
        if not callable(value):
            return value
        try:
            sig = inspect.signature(value)
            params = [p for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty]
            if len(params) == 0:
                return value()
        except Exception:
            pass
        return value

    def format_value(value, depth=0):
        if depth > 1:
            return f"<{type(value).__name__}>"
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, dict):
            return {k: format_value(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [format_value(v, depth + 1) for v in value]
        if hasattr(value, 'to_dict') and not isinstance(value, str):
            try:
                return format_value(value.to_dict(), depth + 1)
            except Exception:
                pass
        if hasattr(value, '__dict__'):
            return {k: format_value(v, depth + 1) for k, v in vars(value).items() if not k.startswith("_")}
        return f"<{type(value).__name__}>"

    def print_object_config(prefix, cfg_obj):
        if isinstance(cfg_obj, dict):
            for k, v in cfg_obj.items():
                print(f"{prefix}{k}: {format_value(v)}")
            return
        if hasattr(cfg_obj, 'to_dict'):
            try:
                cfg_dict = cfg_obj.to_dict()
                for k, v in cfg_dict.items():
                    print(f"{prefix}{k}: {format_value(v)}")
                return
            except Exception:
                pass
        if hasattr(cfg_obj, '__dict__'):
            for k, v in vars(cfg_obj).items():
                if not k.startswith("_"):
                    print(f"{prefix}{k}: {format_value(v)}")
            return
        if hasattr(cfg_obj, '__slots__'):
            for k in cfg_obj.__slots__:
                if k.startswith("_"):
                    continue
                try:
                    value = getattr(cfg_obj, k)
                    print(f"{prefix}{k}: {format_value(value)}")
                except Exception:
                    print(f"{prefix}{k}: <unreadable>")
            return
        attrs = [a for a in dir(cfg_obj) if not a.startswith("_")]
        for name in attrs:
            try:
                value = getattr(cfg_obj, name)
                if callable(value):
                    continue
                print(f"{prefix}{name}: {format_value(value)}")
            except Exception:
                print(f"{prefix}{name}: <unreadable>")

    def infer_model_layer_count(model: Any) -> Optional[int]:
        """Try to infer the number of transformer layers from a loaded model.

        This uses several heuristics:
        1. Common config/model params attributes like n_layer, n_layers, num_layers.
        2. Model metadata (e.g. gguf metadata keys such as qwen2.block_count).
        3. Tensor/state_dict names with layer/block prefixes.
        """
        def _get_int(value: Any) -> Optional[int]:
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str):
                value = value.strip()
                if value.isdigit():
                    return int(value)
            try:
                iv = int(value)
                return iv if iv > 0 else None
            except Exception:
                return None

        def search_for_layer_count(root: Any) -> Optional[int]:
            if root is None:
                return None

            if isinstance(root, Mapping):
                for key, value in root.items():
                    if any(k in key.lower() for k in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count")):
                        if (count := _get_int(value)) is not None:
                            return count
                for key in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count", "qwen2.block_count"):
                    if key in root:
                        if (count := _get_int(root[key])) is not None:
                            return count
                return None

            for key in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count"):
                if hasattr(root, key):
                    try:
                        if (count := _get_int(getattr(root, key))) is not None:
                            return count
                    except Exception:
                        continue

            for key in dir(root):
                if any(k in key.lower() for k in ("n_layer", "n_layers", "num_layers", "block_count", "layer_count")):
                    try:
                        if (count := _get_int(getattr(root, key))) is not None:
                            return count
                    except Exception:
                        continue
            return None

        for root in (
            getattr(model, "metadata", None),
            getattr(model, "model_params", None),
            getattr(model, "config", None),
            getattr(model, "context_params", None),
            getattr(model, "_model", None),
        ):
            if (count := search_for_layer_count(root)) is not None:
                return count

        names = []
        if hasattr(model, "state_dict"):
            try:
                state = model.state_dict() if callable(model.state_dict) else model.state_dict
                if isinstance(state, Mapping):
                    names = list(state.keys())
                elif isinstance(state, (list, tuple)):
                    names = [name for name, _ in state if isinstance(name, str)]
            except Exception:
                pass

        if not names and hasattr(model, "tensors"):
            try:
                tensors = model.tensors
                if isinstance(tensors, (list, tuple)):
                    names = [getattr(t, "name", str(t)) for t in tensors]
            except Exception:
                pass

        if names:
            import re

            layer_indices = set()
            patterns = [
                r"\bblk\.(\d+)\b",
                r"\blayer\.(\d+)\b",
                r"\btransformer\.h\.(\d+)\b",
                r"\bencoder\.layers\.(\d+)\b",
                r"\bdecoder\.layers\.(\d+)\b",
                r"\bblocks\.(\d+)\b",
                r"\bblock\.(\d+)\b",
            ]
            for name in names:
                for pattern in patterns:
                    match = re.search(pattern, name)
                    if match:
                        layer_indices.add(int(match.group(1)))
            if layer_indices:
                return max(layer_indices) + 1

            prefix_counts = {}
            for name in names:
                for prefix in ("blk.", "layer.", "transformer.h.", "encoder.layers.", "decoder.layers.", "block."):
                    if prefix in name:
                        try:
                            part = name.split(prefix, 1)[1]
                            idx = int(part.split(".", 1)[0])
                            prefix_counts[prefix] = max(prefix_counts.get(prefix, -1), idx)
                        except Exception:
                            continue
            if prefix_counts:
                return max(prefix_counts.values()) + 1

        return None

    # Enumerar parámetros del modelo
    print("\n" + "="*60)
    print("AI PARAMETERS")
    print("-"*60)
    try:
        param_names = [a for a in dir(model) if not a.startswith("_")]
        for name in param_names:
            try:
                value = getattr(model, name)
                if callable(value):
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    print(f"Param: {name} -> {value}")
                elif isinstance(value, (list, tuple, dict)):
                    print(f"Param: {name} -> {type(value).__name__} (len={len(value)})")
                else:
                    print(f"Param: {name} -> {type(value).__name__}")
            except Exception:
                print(f"Param: {name} -> <unreadable>")

    except Exception as e:
        logger.exception(f"Error enumerating parameters: {e}")

    # Total estimated de capas
    print("\n" + "="*60)
    print("AI TOTAL LAYERS")
    print("-"*60)
    try:
        layer_count = infer_model_layer_count(model)
        print(f"Inferred total layers: {layer_count}")
    except Exception as e:
        logger.exception(f"Error inferring total layer count: {e}")

    # Enumerar capas del modelo
    print("\n" + "="*60)
    print("AI LAYERS")
    print("-"*60)
    try:
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                print(f"Capa {i}: {layer}")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                print(f"Capa {i}: {layer}")
        else:
            print("No se encontraron capas accesibles en el modelo")
            print("Atributos del modelo:", [a for a in dir(model) if 'layer' in a.lower()])
    except Exception as e:
        logger.exception(f"Error al enumerar capas: {e}")

    # Enumerar bloques del modelo
    print("\n" + "="*60)
    print("AI BLOCKS")
    print("-"*60)
    try:
        if hasattr(model, 'blocks'):
            for i, block in enumerate(model.blocks):
                print(f"Bloque {i}: {block}")
        elif hasattr(model, 'model') and hasattr(model.model, 'blocks'):
            for i, block in enumerate(model.model.blocks):
                print(f"Bloque {i}: {block}")
        else:
            print("No se encontraron bloques accesibles en el modelo")
            print("Atributos del modelo:", [a for a in dir(model) if 'block' in a.lower()])
    except Exception as e:
        logger.exception(f"Error al enumerar bloques: {e}")

    # Enumerar tensores del modelo
    print("\n" + "="*60)
    print("AI TENSORS")
    print("-"*60)
    try:
        tensor_count = 0
        if hasattr(model, 'tensors'):
            for i, tensor in enumerate(model.tensors):
                print(f"Tensor {i}: {tensor}")
                tensor_count += 1
        elif hasattr(model, 'state_dict'):
            state = model.state_dict() if callable(model.state_dict) else model.state_dict
            for i, (name, tensor) in enumerate(state.items()):
                print(f"Tensor {i}: {name} - Shape: {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")
                tensor_count += 1
                if tensor_count >= 20:  # Limitar a los primeros 20 tensores
                    print(f"... ({len(state) - 20} tensores más)")
                    break
        else:
            print("No se encontraron tensores accesibles en el modelo")
            print("Atributos del modelo:", [a for a in dir(model) if 'tensor' in a.lower() or 'state' in a.lower()])
    except Exception as e:
        logger.exception(f"Error al enumerar tensores: {e}")

    # Enumerar submodules del modelo
    print("\n" + "="*60)
    print("AI SUBMODULES")
    print("-"*60)
    try:
        submodule_names = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('model', 'transformer', 'encoder', 'decoder', 'body', 'layers')) and not a.startswith("_")]
        if submodule_names:
            for name in sorted(set(submodule_names)):
                try:
                    sub = getattr(model, name)
                    print(f"Submodule: {name} -> {type(sub).__name__}")
                except Exception:
                    print(f"Submodule: {name} -> <unreadable>")
        else:
            print("No se encontraron submodules detectables en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar submodules: {e}")

    # Enumerar model params del modelo
    print("\n" + "="*60)
    print("AI MODEL PARAMS")
    print("-"*60)
    try:
        if hasattr(model, 'model_params'):
            mp = model.model_params
            print(f"Model Params: {type(mp).__name__}")
            print_object_config("  ", mp)
        else:
            print("No model_params detectable on the model")
    except Exception as e:
        logger.exception(f"Error enumerating model_params: {e}")

    # Enumerar context params del modelo
    print("\n" + "="*60)
    print("AI CONTEXT PARAMETERS")
    print("-"*60)
    try:
        if hasattr(model, 'context_params'):
            cp = model.context_params
            print(f"Context Params: {type(cp).__name__}")
            print_object_config("  ", cp)
        else:
            print("No context_params detectable on the model")
    except Exception as e:
        logger.exception(f"Error enumerating context_params: {e}")

    # Enumerar pesos y bias del modelo
    print("\n" + "="*60)
    print("AI WEIGHTS")
    print("-"*60)
    try:
        if hasattr(model, 'state_dict'):
            state = model.state_dict() if callable(model.state_dict) else model.state_dict
            for i, name in enumerate(state.keys()):
                print(f"Weight {i}: {name}")
                if i >= 49:
                    print("... (displaying first 50 weights)")
                    break
        else:
            print("No se encontró state_dict en el modelo")
            print("Atributos del modelo:", [a for a in dir(model) if 'state' in a.lower() or 'weight' in a.lower() or 'bias' in a.lower()])
    except Exception as e:
        logger.exception(f"Error al enumerar weights: {e}")

    # Enumerar embeddings del modelo
    print("\n" + "="*60)
    print("AI EMBEDDINGS")
    print("-"*60)
    try:
        embedding_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('embed', 'token_emb', 'word_embeddings', 'position_embeddings')) and not a.startswith("_")]
        if embedding_attrs:
            for name in sorted(set(embedding_attrs)):
                try:
                    emb = getattr(model, name)
                    print(f"Embedding: {name} -> {type(emb).__name__}")
                except Exception:
                    print(f"Embedding: {name} -> <unreadable>")
        else:
            print("No se encontraron embeddings accesibles en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar embeddings: {e}")

    # Enumerar output layers del modelo
    print("\n" + "="*60)
    print("AI OUTPUT LAYERS")
    print("-"*60)
    try:
        output_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('lm_head', 'head', 'output', 'proj')) and not a.startswith("_")]
        if output_attrs:
            for name in sorted(set(output_attrs)):
                try:
                    out_layer = getattr(model, name)
                    print(f"Output Layer: {name} -> {type(out_layer).__name__}")
                except Exception:
                    print(f"Output Layer: {name} -> <unreadable>")
        else:
            print("No se encontraron output layers detectables en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar output layers: {e}")

    # Enumerar normalizations del modelo
    print("\n" + "="*60)
    print("AI NORMALIZATION LAYERS")
    print("-"*60)
    try:
        norm_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('norm', 'layernorm', 'rmsnorm', 'ln')) and not a.startswith("_")]
        if norm_attrs:
            for name in sorted(set(norm_attrs)):
                try:
                    norm = getattr(model, name)
                    print(f"Normalization: {name} -> {type(norm).__name__}")
                except Exception:
                    print(f"Normalization: {name} -> <unreadable>")
        else:
            print("No se encontraron normalizaciones detectables en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar normalization layers: {e}")

    # Enumerar attention heads del modelo
    print("\n" + "="*60)
    print("AI ATTENTION HEADS")
    print("-"*60)
    try:
        attn_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'heads')) and not a.startswith("_")]
        if attn_attrs:
            for name in sorted(set(attn_attrs)):
                try:
                    attn = getattr(model, name)
                    print(f"Attention: {name} -> {type(attn).__name__}")
                except Exception:
                    print(f"Attention: {name} -> <unreadable>")
        else:
            print("No se encontraron attention heads detectables en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar attention heads: {e}")

    # Enumerar configuración del modelo
    print("\n" + "="*60)
    print("AI CONFIGURATION")
    print("-"*60)
    try:
        config_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('config', 'vocab', 'context', 'max_seq', 'n_heads', 'n_layers', 'quant')) and not a.startswith("_")]
        if hasattr(model, 'config'):
            try:
                cfg = model.config
                print(f"Config: {type(cfg).__name__}")
                print_object_config("  ", cfg)
            except Exception:
                print("Config attribute exists but could not be read")
        elif config_attrs:
            for name in sorted(set(config_attrs)):
                try:
                    value = getattr(model, name)
                    value = call_no_args(value)
                    if isinstance(value, (str, int, float, bool, type(None))):
                        print(f"Config: {name} -> {value}")
                    elif isinstance(value, (dict, list, tuple)):
                        print(f"Config: {name} -> {type(value).__name__} -> {format_value(value)}")
                    else:
                        print(f"Config: {name} -> {type(value).__name__}")
                        print_object_config("    ", value)
                except Exception:
                    print(f"Config: {name} -> <unreadable>")
        else:
            print("No detectable configurations found in the model")
    except Exception as e:
        logger.exception(f"Error enumerating configuration: {e}")

    # Enumerar device map del modelo
    print("\n" + "="*60)
    print("AI DEVICE MAP")
    print("-"*60)
    try:
        device_attrs = [a for a in dir(model) if any(keyword in a.lower() for keyword in ('device', 'gpu', 'cuda', 'n_gpu', 'map')) and not a.startswith("_")]
        if device_attrs:
            for name in sorted(set(device_attrs)):
                try:
                    value = getattr(model, name)
                    print(f"Device: {name} -> {value}")
                except Exception:
                    print(f"Device: {name} -> <unreadable>")
        else:
            print("No se encontraron device map detectables en el modelo")
    except Exception as e:
        logger.exception(f"Error al enumerar device map: {e}")

    print("\n" + "="*60)

    # Enumerar Total Layers
    print("\n" + "="*60)
    print("AI TOTAL LAYERS")
    print("-"*60)
    inferred_layers = infer_model_layer_count(model)
    print(f"Inferred total layers: {inferred_layers}")
    print("\n" + "="*60)

except Exception as e:
    logger.exception("failed loading model")
    raise

# Pydantic models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
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
    parts.append("assistant: ")
    return "\n".join(parts)


def normalize_stop(stop: Optional[Any]) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(item) for item in stop]
    return [str(stop)]


def get_chat_stop(stop: Optional[Any]) -> List[str]:
    normalized = normalize_stop(stop)
    return normalized if normalized is not None else DEFAULT_STOP


def truncate_text_by_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            return text[:idx]
    return text


def build_text_completion_response(text: str, raw: Optional[Any] = None) -> Dict[str, Any]:
    response = {
        "id": None,
        "object": "text.completion",
        "model": MODEL_NAME,
        "choices": [{"text": text, "index": 0}]
    }
    if raw is not None:
        response["raw"] = raw
    return response


def build_chat_completion_response(text: str) -> Dict[str, Any]:
    return {
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


def extract_v1_prompt(data: Dict[str, Any]) -> Optional[str]:
    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    input_value = data.get("input")
    if isinstance(input_value, str) and input_value:
        return input_value
    return None


def run_model_sync(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None):
    out = model_create_compat_sync(prompt, max_tokens, temperature, top_p, stop)
    text = out["choices"][0].get("text", "") or ""
    return truncate_text_by_stop(text, stop), out


async def run_model_async(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None):
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(run_model_sync, prompt, max_tokens, temperature, top_p, stop)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(run_model_sync, prompt, max_tokens, temperature, top_p, stop))


def text_stream_event_stream(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]):
    async def event_stream():
        async for chunk in stream_generator_wrapper_sync_to_async(llama_create_stream_sync, prompt, max_tokens, temperature, top_p, stop):
            yield json.dumps({"id": None, "object": "text.chunk", "text": chunk}, ensure_ascii=False) + "\n"
        yield json.dumps({"id": None, "object": "text.complete"}, ensure_ascii=False) + "\n"
    return event_stream


def chat_stream_event_stream(prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]):
    async def event_stream():
        first_sent = True
        async for chunk in stream_generator_wrapper_sync_to_async(llama_create_stream_sync, prompt, max_tokens, temperature, top_p, stop):
            payload = {"id": None, "object": "chat.completion.chunk", "delta": {"content": chunk}}
            if first_sent:
                payload["delta"]["role"] = "assistant"
            yield json.dumps(payload, ensure_ascii=False) + "\n"
            first_sent = False
        yield json.dumps({"id": None, "object": "chat.completion.complete"}, ensure_ascii=False) + "\n"
    return event_stream


def remove_excessive_repetition(text: str, max_repetitions: int = 1) -> str:
    """Remove excessively repeated sentences to avoid token-padding artifacts.
    Detects and removes sentences that appear more than max_repetitions times.
    """
    if not text:
        return text
    
    # Split by sentence-like patterns (., !, ?)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    seen_sentences = {}
    filtered_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Normalize for comparison (lowercase, remove extra spaces)
        normalized = re.sub(r'\s+', ' ', sentence.lower())
        
        if normalized in seen_sentences:
            seen_sentences[normalized] += 1
            # Only keep first occurrence(s)
            if seen_sentences[normalized] <= max_repetitions:
                filtered_sentences.append(sentence)
        else:
            seen_sentences[normalized] = 1
            filtered_sentences.append(sentence)
    
    result = ' '.join(filtered_sentences).strip()
    return result


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

def call_model_with_signatures(prompt: str, max_tokens: int, temperature: float, top_p: float, stream: bool = False, stop: Optional[List[str]] = None):
    """
    Try several possible model completion signatures and return whatever the model returns.
    Synchronous function suitable to be run inside run_in_executor.
    """
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
        resp = call_model_with_signatures(prompt, max_tokens, temperature, top_p, False, stop=stop)
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
    out = model_create_compat_sync(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
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
        return make_json_response({"error": "no prompt provided"}, status_code=400)

    if req.stream:
        return StreamingResponse(
            text_stream_event_stream(req.prompt, req.max_tokens or 256, req.temperature or 0.0, req.top_p or 1.0, req.stop),
            media_type="application/x-ndjson"
        )

    try:
        text, out = await run_model_async(req.prompt, req.max_tokens or 256, req.temperature or 0.0, req.top_p or 1.0, req.stop)
    except Exception as e:
        logger.exception("generation failed")
        return make_json_response({"error": str(e)}, status_code=500)

    return make_json_response(build_text_completion_response(text, raw=out))

# /api/chat (proxy)
@app.post("/api/chat")
async def api_chat(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return make_json_response({"error": "invalid JSON body"}, status_code=400)

    messages = data.get("messages", [])
    if not messages:
        return make_json_response({"error": "no messages provided"}, status_code=400)

    prompt = build_prompt_from_messages(messages)
    stop = get_chat_stop(data.get("stop"))
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stream = data.get("stream", False)

    if stream:
        return StreamingResponse(
            chat_stream_event_stream(prompt, max_tokens, temperature, top_p, stop),
            media_type="application/x-ndjson"
        )

    try:
        text, _ = await run_model_async(prompt, max_tokens, temperature, top_p, stop)
    except Exception as e:
        logger.exception("chat generation failed")
        return make_json_response({"error": str(e)}, status_code=500)

    text = remove_excessive_repetition(text)
    return make_json_response(build_chat_completion_response(text))

# /api/chat/completions (ollama-style)
@app.post("/api/chat/completions")
async def chat_completions(req: ChatRequest):
    if not req.messages:
        return make_json_response({"error": "no messages provided"}, status_code=400)

    prompt = build_prompt_from_messages([m.dict() for m in req.messages])
    stop = get_chat_stop(req.stop)

    if req.stream:
        return StreamingResponse(
            chat_stream_event_stream(prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0, stop),
            media_type="application/x-ndjson"
        )

    try:
        text, _ = await run_model_async(prompt, req.max_tokens or 512, req.temperature or 0.0, req.top_p or 1.0, stop)
    except Exception as e:
        logger.exception("chat completions failed")
        return make_json_response({"error": str(e)}, status_code=500)

    text = remove_excessive_repetition(text)
    return make_json_response(build_chat_completion_response(text))

# v1 endpoints for OpenAI compatibility
@app.get("/v1/health")
def v1_health():
    return {"status": "ok"}

@app.get("/v1/models")
def v1_models():
    return models()

@app.get("/v1/version")
def v1_version():
    return version()


@app.post("/v1/completions")
async def v1_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return make_json_response({"error": "invalid JSON body"}, status_code=400)

    prompt = extract_v1_prompt(data)
    if not prompt:
        return make_json_response({"error": "no prompt provided"}, status_code=400)

    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stop = normalize_stop(data.get("stop"))
    stream = data.get("stream", False)

    if stream:
        return StreamingResponse(
            text_stream_event_stream(prompt, max_tokens, temperature, top_p, stop),
            media_type="application/x-ndjson"
        )

    try:
        text, out = await run_model_async(prompt, max_tokens, temperature, top_p, stop)
    except Exception as e:
        logger.exception("v1/completions generation failed")
        return make_json_response({"error": str(e)}, status_code=500)

    return make_json_response(build_text_completion_response(text, raw=out))


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return make_json_response({"error": "invalid JSON body"}, status_code=400)

    messages = data.get("messages", [])
    if not messages:
        return make_json_response({"error": "no messages provided"}, status_code=400)

    prompt = build_prompt_from_messages(messages)
    stop = get_chat_stop(data.get("stop"))
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    stream = data.get("stream", False)

    if stream:
        return StreamingResponse(
            chat_stream_event_stream(prompt, max_tokens, temperature, top_p, stop),
            media_type="application/x-ndjson"
        )

    try:
        text, _ = await run_model_async(prompt, max_tokens, temperature, top_p, stop)
    except Exception as e:
        logger.exception("v1/chat/completions failed")
        return make_json_response({"error": str(e)}, status_code=400)

    text = remove_excessive_repetition(text)
    return make_json_response(build_chat_completion_response(text))
