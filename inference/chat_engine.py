"""
Inference module for MyIAModelChat.

Provides the ChatEngine class and ChatConfig dataclass for chat/inference.
Refactored from main_chat.py to remove CLI and FastAPI dependencies.
"""

import os
import warnings
import logging
import importlib
import re
import json
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import psutil
import torch

from commons.model.chatmodel import ChatModel
from commons.dialogue.dialogmanager import DialogueManager
try:
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
except ImportError:
    SentencePieceTokenizerWrapper = None
from transformers import AutoTokenizer, pipeline

# Configuration constants
CKPT_PATH = os.path.join('checkpoints', 'chat_model.pth')
MODELS_DIR = 'models'
CHECKPOINTS_DIR = 'checkpoints'
MAX_RAM_GB = None

# Setup logging (configured by main.py)
logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Configuration for chat/inference (no CLI args)."""
    model_name: Optional[str] = None
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] or None=auto
    use_vulkan: bool = False
    show_thinking: bool = False
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64


def parse_thinking_response(text: str) -> Dict[str, Optional[str]]:
    """Parse <thinking> tags from response text."""
    if '<thinking>' not in text or '</thinking>' not in text:
        return {'thinking': None, 'response': text}
    try:
        thinking = text.split('<thinking>')[1].split('</thinking>')[0]
        response = text.split('</thinking>')[1].strip()
        return {'thinking': thinking, 'response': response}
    except (IndexError, ValueError):
        return {'thinking': None, 'response': text}


def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Build prompt from message list."""
    parts: List[str] = []
    for message in messages:
        role = str(message.get('role', 'user')).lower()
        content = str(message.get('content', ''))
        if role == 'system':
            parts.append(f"System: {content}")
        elif role == 'assistant':
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append('Assistant:')
    return '\n'.join(parts)


def truncate_stop_sequences(text: str, stop: Optional[List[str]]) -> str:
    """Truncate text at stop sequences."""
    if not stop:
        return text
    for s in stop:
        if s and s in text:
            text = text.split(s, 1)[0]
    return text


class ChatEngine:
    """Chat inference engine."""
    
    def __init__(self, config: ChatConfig):
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

        self.config = config
        self.device_mode = config.device_mode
        self.gpu_indices = config.gpu_indices or []
        self.use_vulkan = config.use_vulkan
        self.show_thinking = config.show_thinking

        # Dynamic model resolution
        self.model_name = config.model_name
        if self.model_name:
            self.ckpt_path = self._resolve_model_path(self.model_name)
        else:
            self.ckpt_path = CKPT_PATH
        logger.info(f"Loading model from: {self.ckpt_path}")

        device = self.get_device()

        # Optional memory limit
        if MAX_RAM_GB is not None:
            try:
                p = psutil.Process()
                mem_bytes = MAX_RAM_GB * 1024**3
                p.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
                logger.info("Applied RAM limit: %s GB", MAX_RAM_GB)
            except Exception as e:
                logger.warning("Could not enforce RAM limit via psutil: %s", e)

        # Tokenizer and model
        self.tokenizer = None
        vocab_file = os.path.join('checkpoints', 'tokenizer_vocab.json')

        if os.path.exists(vocab_file):
            try:
                import json as _json
                with open(vocab_file, 'r', encoding='utf-8') as _f:
                    _vocab_meta = _json.load(_f)
                sp_path = _vocab_meta.get('sentencepiece_model') if isinstance(_vocab_meta, dict) else None
                if sp_path and SentencePieceTokenizerWrapper is not None and os.path.exists(sp_path):
                    self.tokenizer = SentencePieceTokenizerWrapper(sp_path)
                    logger.info(f"Loaded SentencePiece BPE tokenizer from {sp_path}, vocab_size: {self.tokenizer.vocab_size}")
                else:
                    raise RuntimeError(
                        f"No SentencePiece model found in {vocab_file}. "
                        "Run: python main.py --prepare-data --aiml --hf"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Error loading tokenizer from {vocab_file}: {e}. "
                    "Run: python main.py --prepare-data --aiml --hf"
                )

        try:
            ckpt = self.try_load_checkpoint(self.ckpt_path, device, False)
            if isinstance(ckpt, dict):
                # Load tokenizer from path saved in checkpoint (avoids unpickling custom objects)
                if 'tokenizer_path' in ckpt and ckpt['tokenizer_path']:
                    sp_path = ckpt['tokenizer_path']
                    if not os.path.isabs(sp_path):
                        sp_path = os.path.join(os.getcwd(), sp_path)
                    if SentencePieceTokenizerWrapper is not None and os.path.exists(sp_path):
                        self.tokenizer = SentencePieceTokenizerWrapper(sp_path)
                        logger.info(f"Loaded tokenizer from checkpoint path: {sp_path}, vocab_size: {self.tokenizer.vocab_size}")
                elif 'tokenizer' in ckpt and ckpt['tokenizer'] is not None:
                    self.tokenizer = ckpt['tokenizer']
                    logger.info(f"Loaded tokenizer from checkpoint (legacy), vocab_size: {self.tokenizer.vocab_size}")
                state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', None))
            else:
                state_dict = ckpt

            if self.tokenizer is None:
                raise RuntimeError(
                    "No tokenizer found in checkpoint or vocab file. "
                    "Run: python main.py --prepare-data --aiml --hf"
                )

            arch = ckpt.get('architecture', {}) if isinstance(ckpt, dict) else {}
            self.model = ChatModel(self.tokenizer,
                embed_size=arch.get('embed_size', 256),
                num_layers=arch.get('num_layers', 4))

            if state_dict is None:
                raise ValueError('Checkpoint does not contain model state dict')

            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state[new_key] = v

            self.model.load_state_dict(new_state)
            self.model.to(device)
            self.model.eval()
            logger.info("Pre-trained model loaded successfully.")

        except Exception as e:
            logger.exception("Error loading pre-trained model: %s", e)
            if self.tokenizer is None:
                raise RuntimeError(
                    f"Error loading model checkpoint: {e}. "
                    "Ensure the model was trained with BPE tokenizer."
                )
            self.model = ChatModel(self.tokenizer, embed_size=256)
            logger.info("Initializing model with random weights...")
            init_fn = getattr(self.tokenizer, "init_weights", None)
            if callable(init_fn):
                self.model.apply(init_fn)
            else:
                def default_init(m):
                    if hasattr(m, "weight") and m.weight is not None:
                        try:
                            torch.nn.init.xavier_uniform_(m.weight)
                        except Exception:
                            pass
                    if hasattr(m, "bias") and m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                self.model.apply(default_init)
            self.model.to(device)
            self.model.eval()

        logger.info("Using device: %s", device)

        # Load auxiliary pipelines
        from commons.registry.model_downloader import ensure_model_local

        sentiment_model_path = ensure_model_local(
            'nlptown/bert-base-multilingual-uncased-sentiment',
            'models/sentiment'
        )
        intent_model_path = sentiment_model_path

        pipe_device = 0 if device.type == 'cuda' else -1
        shared_bert = pipeline(
            'text-classification',
            model=sentiment_model_path,
            device=pipe_device,
            )
        intent_classifier = shared_bert
        sentiment_analyzer = shared_bert

        self.dialogue_manager = DialogueManager(
            model=self.model,
            device=device,
            tokenizer=self.tokenizer,
            intent_classifier=intent_classifier,
            sentiment_analyzer=sentiment_analyzer,
            persona={
                "name": "Eduardo Piñera Aznárez",
                "age": 52,
                "occupation": "AI assistant",
                "interests": ["IT technology", "MS Office", "Libre Office", "Games", "Humanity simulation"]
            },
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            max_len=128,
            min_length=3,
            no_repeat_ngram_size=3,
            default_response="Lo siento, no puedo responder ahora.",
            thinking_enabled=config.thinking_enabled,
            thinking_max_tokens=config.thinking_max_tokens,
        )

    def _resolve_model_path(self, model_name):
        """Resolve model name to checkpoint file path."""
        if model_name.endswith('.pth'):
            return model_name
        for path in [
            os.path.join(CHECKPOINTS_DIR, f'{model_name}.pth'),
            os.path.join(MODELS_DIR, f'{model_name}.pth'),
            f'{model_name}.pth',
        ]:
            if os.path.exists(path):
                return path
        logger.warning(f"Model '{model_name}' not found, falling back to {CKPT_PATH}")
        return CKPT_PATH

    def get_device(self):
        """Get appropriate device based on configuration."""
        from commons.utils.device_utils import check_vulkan_available

        device_mode = self.device_mode
        use_vulkan = self.use_vulkan
        gpu_indices = self.gpu_indices

        if device_mode == 'cpu':
            logger.info("CPU-only mode enabled")
            return torch.device('cpu')

        if use_vulkan:
            if check_vulkan_available():
                logger.info("Using Vulkan backend")
                return torch.device('vulkan')
            else:
                raise RuntimeError("Vulkan requested but not available")

        if torch.cuda.is_available():
            if gpu_indices:
                device = torch.device(f'cuda:{gpu_indices[0]}')
                logger.info(f"Using CUDA device {gpu_indices[0]}: {torch.cuda.get_device_name(gpu_indices[0])}")
            else:
                device = torch.device('cuda')
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return device

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon)")
            return torch.device('mps')

        logger.info("No accelerator available, using CPU")
        return torch.device('cpu')

    def try_load_checkpoint(self, path, device, trusted):
        """Load model checkpoint with safety measures."""
        try:
            if trusted:
                logger.warning("Loading checkpoint with weights_only=False (trusted path): %s", path)
                return torch.load(path, map_location=device, weights_only=False)
            else:
                try:
                    return torch.load(path, map_location=device, weights_only=True)
                except Exception:
                    logger.warning("weights_only=True failed, falling back to weights_only=False: %s", path)
                    return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL\s+([\w\.]+)\s+was", msg)
            if m:
                full_name = m.group(1)
                try:
                    module_name, class_name = full_name.rsplit('.', 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    with torch.serialization.safe_globals([cls]):
                        logger.warning("Loading checkpoint with weights_only=False (allowlisted class): %s", path)
                        return torch.load(path, map_location=device, weights_only=False)
                except Exception:
                    logger.exception("Allowlisting failed for %s", full_name)
                    raise

            try:
                allowed = []
                if SentencePieceTokenizerWrapper is not None:
                    allowed.append(SentencePieceTokenizerWrapper)
                if allowed:
                    with torch.serialization.safe_globals(allowed):
                        logger.warning("Loading checkpoint with weights_only=False (allowlisted tokenizer): %s", path)
                        return torch.load(path, map_location=device, weights_only=False)
            except Exception:
                pass

            raise

    def start_chat_loop(self):
        """Start interactive chat loop."""
        logger.info("Starting chat interface. Type 'exit' or press ESC to exit.")
        model_display = self.model_name or 'chat_model'
        print(f"\n{'='*50}")
        print(f"  Active model: {model_display}")
        print(f"  Model file: {self.ckpt_path}")
        print(f"{'='*50}\n")
        try:
            import msvcrt
            use_msvcrt = True
        except ImportError:
            use_msvcrt = False

        try:
            while True:
                try:
                    if use_msvcrt:
                        print("You: ", end="", flush=True)
                        chars = []
                        while True:
                            if msvcrt.kbhit():
                                ch = msvcrt.getwch()
                                if ord(ch) == 27:
                                    print("\nExiting chat loop.")
                                    return
                                elif ch == '\r':
                                    print()
                                    break
                                elif ch == '\b':
                                    if chars:
                                        chars.pop()
                                        print('\b \b', end="", flush=True)
                                else:
                                    chars.append(ch)
                                    print(ch, end="", flush=True)
                        user_input = "".join(chars)
                    else:
                        user_input = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    logger.info("Exiting chat loop.")
                    break

                if user_input.lower() in ('quit', 'exit'):
                    break

                response = self.dialogue_manager.generate_response(user_input)

                if isinstance(response, dict):
                    thinking = response.get('thinking')
                    resp_text = response.get('response', '')
                    if self.show_thinking and thinking:
                        print(f"Bot [thinking]: {thinking}")
                    print(f"Bot: {resp_text}")
                else:
                    if self.show_thinking:
                        parsed = parse_thinking_response(response)
                        if parsed['thinking']:
                            print(f"Bot [thinking]: {parsed['thinking']}")
                        print(f"Bot: {parsed['response']}")
                    else:
                        print("Bot:", response)
        except KeyboardInterrupt:
            logger.info("Chat terminated by user.")

    def generate_response(self, prompt: str, full: bool = False) -> Union[str, Dict[str, Optional[str]]]:
        """Generate response for a prompt."""
        result = self.dialogue_manager.generate_response(prompt)
        if isinstance(result, dict):
            if full:
                return result
            return result.get('response', '')
        return result

    def close(self):
        """Release GPU memory and cleanup resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'dialogue_manager') and self.dialogue_manager is not None:
            dm = self.dialogue_manager
            if hasattr(dm, 'intent_classifier') and dm.intent_classifier is not None:
                del dm.intent_classifier
            if hasattr(dm, 'sentiment_analyzer') and dm.sentiment_analyzer is not None:
                del dm.sentiment_analyzer
            del self.dialogue_manager
            self.dialogue_manager = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ChatEngine resources released")


# Streaming helper functions for SSE format
def stream_chat_with_thinking(thinking: str, response: str):
    """Generate SSE chunks for streaming with thinking/reasoning.
    
    Yields JSON strings for Server-Sent Events format.
    """
    import hashlib
    request_id = hashlib.md5(f"{thinking}{response}".encode()).hexdigest()[:12]
    
    # First chunk with reasoning
    yield json.dumps({
        'id': request_id,
        'object': 'chat.completion.chunk',
        'delta': {'reasoning': thinking},
        'finish_reason': None
    }) + '\n'
    
    # Content chunks (split response into words for demo)
    words = response.split()
    for i, word in enumerate(words):
        content = word + (' ' if i < len(words) - 1 else '')
        yield json.dumps({
            'id': request_id,
            'object': 'chat.completion.chunk',
            'delta': {'content': content},
            'finish_reason': None
        }) + '\n'
    
    # Finish chunk
    yield json.dumps({
        'id': request_id,
        'object': 'chat.completion.chunk',
        'delta': {},
        'finish_reason': 'stop'
    }) + '\n'
    
    # Done marker
    yield 'data: [DONE]\n\n'


def stream_chat_text(text: str):
    """Generate SSE chunks for streaming text content.
    
    Yields JSON strings for Server-Sent Events format.
    """
    import hashlib
    request_id = hashlib.md5(text.encode()).hexdigest()[:12]
    
    # Content chunks
    words = text.split()
    for i, word in enumerate(words):
        content = word + (' ' if i < len(words) - 1 else '')
        yield json.dumps({
            'id': request_id,
            'object': 'chat.completion.chunk',
            'delta': {'content': content},
            'finish_reason': None
        }) + '\n'
    
    # Finish chunk
    yield json.dumps({
        'id': request_id,
        'object': 'chat.completion.chunk',
        'delta': {},
        'finish_reason': 'stop'
    }) + '\n'
    
    # Done marker
    yield 'data: [DONE]\n\n'


# Singleton for FastAPI integration
_init_lock = threading.Lock()
_chat_engine_instance = None

def get_chat_engine_instance(device_mode: str = 'auto', gpu_indices: Optional[List[int]] = None, model: Optional[str] = None):
    """Get or create singleton ChatEngine instance."""
    global _chat_engine_instance
    if _chat_engine_instance is None:
        with _init_lock:
            if _chat_engine_instance is None:
                config = ChatConfig(device_mode=device_mode, gpu_indices=gpu_indices, model_name=model)
                _chat_engine_instance = ChatEngine(config)
    return _chat_engine_instance


def reset_chat_engine():
    """Release singleton ChatEngine resources."""
    global _chat_engine_instance
    with _init_lock:
        if _chat_engine_instance is not None:
            _chat_engine_instance.close()
            _chat_engine_instance = None
