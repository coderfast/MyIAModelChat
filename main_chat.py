import os
import warnings
import argparse
import logging
import importlib
import json
import re
import time
import hashlib
import psutil
import torch
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dialogmanager import DialogueManager
from chatmodel import ChatModel
try:
    from bpe_tokenizer import SentencePieceTokenizerWrapper
except ImportError:
    SentencePieceTokenizerWrapper = None
from transformers import pipeline

# Configuration constants
SYSTEM_CONFIG = {
    'default_cores_fraction': 0.5,
    'min_cores': 1,
    'min_threads': 1,
    'max_ram_fraction': 0.75,
}

# Configuración principal
use_trusted = False  # Cambia a True solo si confías totalmente en el checkpoint
CKPT_PATH = 'chat_model.pth'
MODELS_DIR = 'models'
MAX_RAM_GB = None  # Ej: 4 para limitar a 4GB; None para no aplicar límite
MODEL_NAME = 'myiamodelchat-local'
OLLAMA_VERSION = '0.6.4'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="MyIAModelChat Ollama-compatible Server")
main_chat_instance = None


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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    include_thinking: Optional[bool] = False

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    include_thinking: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    user: Optional[str] = None

class MainChat:
    def __init__(self, args):
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

        self.chat = args.chat
        self.use_cpuonly = getattr(args, 'use_cpuonly', False)
        self.cuda_device = getattr(args, 'cuda_device', None)
        self.show_thinking = getattr(args, 'show_thinking', False)

        # Dynamic model resolution
        self.model_name = getattr(args, 'model', None)
        if self.model_name:
            self.ckpt_path = self._resolve_model_path(self.model_name)
        else:
            self.ckpt_path = CKPT_PATH
        logger.info(f"Loading model from: {self.ckpt_path}")

        if self.cuda_device is not None and not self.use_cpuonly:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
            logger.info("Using CUDA devices limited to: %s", self.cuda_device)

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
            ckpt = self.try_load_checkpoint(self.ckpt_path, device, use_trusted)
            if isinstance(ckpt, dict):
                # Load tokenizer from checkpoint if included
                if 'tokenizer' in ckpt and ckpt['tokenizer'] is not None:
                    self.tokenizer = ckpt['tokenizer']
                    logger.info(f"Loaded tokenizer from checkpoint, vocab_size: {self.tokenizer.vocab_size}")

                # Extract state dict for the model
                state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', None))
            else:
                state_dict = ckpt

            if self.tokenizer is None:
                raise RuntimeError(
                    "No tokenizer found in checkpoint or vocab file. "
                    "Run: python main.py --prepare-data --aiml --hf"
                )

            # Instantiate model with tokenizer-derived vocab
            self.model = ChatModel(self.tokenizer, embed_size=256, hidden_size=512, num_layers=4)

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
            self.model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256)
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

        # Pipelines auxiliares — load from local cache
        from model_downloader import ensure_model_local

        sentiment_model_path = ensure_model_local(
            'nlptown/bert-base-multilingual-uncased-sentiment',
            'models/sentiment'
        )
        intent_model_path = ensure_model_local(
            'nlptown/bert-base-multilingual-uncased-sentiment',
            'models/intent'
        )

        intent_classifier = pipeline('text-classification', model=intent_model_path)
        sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model_path)

        # DialogueManager configurable
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
            default_response="Lo siento, no puedo responder ahora."
        )

    def _resolve_model_path(self, model_name):
        """Resolve model name to checkpoint file path."""
        if model_name.endswith('.pth'):
            return model_name
        # Check models/ directory first, then root
        for path in [os.path.join(MODELS_DIR, f'{model_name}.pth'), f'{model_name}.pth']:
            if os.path.exists(path):
                return path
        # Fallback to chat_model.pth
        logger.warning(f"Model '{model_name}' not found, falling back to {CKPT_PATH}")
        return CKPT_PATH

    def get_device(self):
        if self.use_cpuonly:
            logger.info("CPU-only mode enabled")
            return torch.device('cpu')
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                logger.info("Using CUDA: %s", name)
            except Exception:
                logger.info("Using CUDA")
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            logger.info("Using MPS")
            return torch.device('mps')
        logger.info("Using CPU")
        return torch.device('cpu')

    def try_load_checkpoint(self, path, device, trusted):
        try:
            if trusted:
                # Load complete checkpoint (includes tokenizer object)
                return torch.load(path, map_location=device, weights_only=False)
            else:
                # Prefer weights-only for safety, fallback if it fails for local trusted checkpoint
                try:
                    return torch.load(path, map_location=device, weights_only=True)
                except Exception:
                    return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            # Attempt to parse unsupported globals and allowlist them
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL\s+([\w\.]+)\s+was", msg)
            if m:
                full_name = m.group(1)
                try:
                    module_name, class_name = full_name.rsplit('.', 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    with torch.serialization.safe_globals([cls]):
                        return torch.load(path, map_location=device, weights_only=False)
                except Exception:
                    logger.exception("Allowlisting failed for %s", full_name)
                    raise

            # Allow SentencePiece tokenizer for deserialization
            try:
                allowed = []
                if SentencePieceTokenizerWrapper is not None:
                    allowed.append(SentencePieceTokenizerWrapper)
                if allowed:
                    with torch.serialization.safe_globals(allowed):
                        return torch.load(path, map_location=device, weights_only=False)
            except Exception:
                pass

            raise

    def performMainChat(self):
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
                                if ord(ch) == 27:  # ESC
                                    print("\nExiting chat loop.")
                                    return
                                elif ch == '\r':  # Enter
                                    print()
                                    break
                                elif ch == '\b':  # Backspace
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

                # Parse and display thinking if enabled
                if self.show_thinking:
                    parsed = parse_thinking_response(response)
                    if parsed['thinking']:
                        print(f"Bot [thinking]: {parsed['thinking']}")
                    print(f"Bot: {parsed['response']}")
                else:
                    print("Bot:", response)
        except KeyboardInterrupt:
            logger.info("Chat terminated by user.")

    def generate_response(self, prompt: str) -> str:
        return self.dialogue_manager.generate_response(prompt)


def get_main_chat_instance(use_cpuonly: bool = False, cuda_device: Optional[int] = None, model: Optional[str] = None):
    global main_chat_instance
    if main_chat_instance is None:
        args = argparse.Namespace(chat=False, use_cpuonly=use_cpuonly, cuda_device=cuda_device, model=model)
        main_chat_instance = MainChat(args)
    return main_chat_instance


def build_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
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
    if not stop:
        return text
    for s in stop:
        if s and s in text:
            text = text.split(s, 1)[0]
    return text


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
        response = text.split('效益')[1].strip()
        return {'thinking': thinking, 'response': response}
    except (IndexError, ValueError):
        return {'thinking': None, 'response': text}


def make_usage(prompt_text: str, completion_text: str) -> Dict[str, int]:
    prompt_tokens = len(prompt_text.split())
    completion_tokens = len(completion_text.split())
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens
    }


def stream_chat_text(text: str, request_id: Optional[str] = None):
    def gen():
        chunk_size = 64
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            payload = {
                'id': request_id,
                'object': 'chat.completion.chunk',
                'delta': {'role': 'assistant', 'content': chunk}
            }
            yield json.dumps(payload) + '\n'
            time.sleep(0.01)
        yield json.dumps({'id': request_id, 'object': 'chat.completion.complete'}) + '\n'
    return gen()


def encode_text_embedding(text: str, chat: MainChat) -> List[float]:
    default_size = 512
    try:
        token_ids = chat.tokenizer.encode(text)
        if not token_ids:
            return [0.0] * default_size
        emb_weights = chat.model.model.transformer.wte.weight
        tokens_tensor = torch.LongTensor(token_ids).to(emb_weights.device)
        token_embeds = emb_weights[tokens_tensor]
        avg_embed = token_embeds.mean(dim=0)
        vector = avg_embed.detach().cpu().tolist()
        if len(vector) < default_size:
            vector.extend([0.0] * (default_size - len(vector)))
        return [float(x) for x in vector[:default_size]]
    except Exception:
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        vector = [b / 255.0 for b in digest]
        if len(vector) < default_size:
            vector.extend([0.0] * (default_size - len(vector)))
        return vector[:default_size]


@app.get('/v1/health')
def v1_health():
    return {'status': 'ok'}


@app.get('/v1/models')
def v1_models():
    return [
        {
            'name': MODEL_NAME,
            'id': MODEL_NAME,
            'description': 'MyIAModelChat local model',
            'size': 'custom',
            'family': 'myiamodelchat'
        }
    ]


@app.get('/v1/version')
def v1_version():
    return {'name': MODEL_NAME, 'version': OLLAMA_VERSION}


@app.post('/v1/chat/completions')
@app.post('/api/chat/completions')
async def chat_completions(req: ChatRequest):
    prompt = build_prompt_from_messages([m.dict() for m in req.messages])
    chat = get_main_chat_instance(use_cpuonly=os.getenv('USE_CPUONLY', 'false').lower() in ('1', 'true', 'yes'))
    text = chat.generate_response(prompt)
    text = truncate_stop_sequences(text, req.stop)

    # Parse thinking if requested
    parsed = parse_thinking_response(text)
    content = parsed['response'] if not req.include_thinking else text
    thinking = parsed['thinking']

    if req.stream:
        return StreamingResponse(stream_chat_text(content), media_type='application/json')

    message = {'role': 'assistant', 'content': content}
    if req.include_thinking and thinking:
        message['reasoning'] = thinking

    response = {
        'id': None,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': req.model or MODEL_NAME,
        'choices': [
            {
                'index': 0,
                'message': message,
                'finish_reason': 'stop'
            }
        ],
        'usage': make_usage(prompt, text)
    }
    return make_json_response(response)


@app.post('/api/chat')
async def api_chat(request: Request):
    data = await request.json()
    messages = data.get('messages', [])
    prompt = build_prompt_from_messages(messages)
    chat = get_main_chat_instance(use_cpuonly=os.getenv('USE_CPUONLY', 'false').lower() in ('1', 'true', 'yes'))
    text = chat.generate_response(prompt)
    text = truncate_stop_sequences(text, data.get('stop'))

    # Parse thinking if requested
    include_thinking = data.get('include_thinking', False)
    parsed = parse_thinking_response(text)
    content = parsed['response'] if not include_thinking else text
    thinking = parsed['thinking']

    if data.get('stream', False):
        return StreamingResponse(stream_chat_text(content), media_type='application/json')

    message = {'role': 'assistant', 'content': content}
    if include_thinking and thinking:
        message['reasoning'] = thinking

    response = {
        'id': None,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': data.get('model', MODEL_NAME),
        'choices': [
            {
                'index': 0,
                'message': message,
                'finish_reason': 'stop'
            }
        ],
        'usage': make_usage(prompt, text)
    }
    return make_json_response(response)


@app.post('/v1/embeddings')
async def v1_embeddings(req: EmbeddingRequest):
    chat = get_main_chat_instance(use_cpuonly=os.getenv('USE_CPUONLY', 'false').lower() in ('1', 'true', 'yes'))
    inputs = [req.input] if isinstance(req.input, str) else req.input
    embeddings = []
    for index, item in enumerate(inputs):
        text = str(item)
        vector = encode_text_embedding(text, chat)
        embeddings.append({'object': 'embedding', 'embedding': vector, 'index': index})
    return {
        'object': 'list',
        'data': embeddings,
        'model': req.model or MODEL_NAME
    }


@app.post('/api/generate')
async def api_generate(req: GenerateRequest):
    chat = get_main_chat_instance(use_cpuonly=os.getenv('USE_CPUONLY', 'false').lower() in ('1', 'true', 'yes'))
    text = chat.generate_response(req.prompt)
    text = truncate_stop_sequences(text, req.stop)

    # Parse thinking if requested
    parsed = parse_thinking_response(text)

    if req.stream:
        return StreamingResponse(stream_chat_text(parsed['response']), media_type='application/json')

    result = {
        'id': None,
        'model': MODEL_NAME,
        'object': 'text_completion',
        'text': parsed['response'] if not req.include_thinking else text,
        'usage': make_usage(req.prompt, text)
    }
    if req.include_thinking and parsed['thinking']:
        result['reasoning'] = parsed['thinking']

    return result


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--chat', action='store_true', help='Run Ollama-compatible HTTP server')
    parser.add_argument('--console', action='store_true', help='Run terminal chat loop')
    
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11434)
    
    parser.add_argument('--use-cpuonly', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=None)
    parser.add_argument('--show-thinking', action='store_true', help='Show <think> reasoning in console chat')
    
    # CPU configuration
    parser.add_argument("--num_cores", type=int, default=default_num_cores, help=f"CPU cores (default: {default_num_cores})")
    parser.add_argument("--num_threads", type=int, default=default_num_threads, help=f"Threads (default: {default_num_threads})")
    return parser.parse_args()


if __name__ == '__main__':

    def get_memory_limit_bytes(max_ram_fraction):
        """Return the maximum memory limit in bytes based on available system RAM."""
        if psutil is None:
            logger.warning("psutil is not installed. Cannot enforce RAM usage limits precisely.")
            return None

        total_bytes = psutil.virtual_memory().total
        limited_bytes = int(total_bytes * max_ram_fraction)
        logger.info(f"System memory: {total_bytes / (1024 ** 3):.2f} GB")
        logger.info(f"Applying max RAM usage fraction: {max_ram_fraction * 100:.0f}% => {limited_bytes / (1024 ** 3):.2f} GB")
        return limited_bytes

    def setup_cpu_configuration(args):
        """Configure CPU threading based on arguments."""
        if args.num_cores == 0:
            args.num_cores = mp.cpu_count()
            logger.info(f"Auto-detected CPU cores: {args.num_cores}")
        
        if args.num_threads == 0:
            args.num_threads = mp.cpu_count()
            logger.info(f"Auto-detected threads: {args.num_threads}")
        
        logger.info(f"\n{'='*80}")
        logger.info("CPU CONFIGURATION")
        logger.info(f"{'='*80}")
        
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_cores)
        torch.set_num_interop_threads(args.num_cores)
        
        # Enforce memory limit (75% of system RAM by default)
        args.max_ram_bytes = get_memory_limit_bytes(getattr(args, 'max_ram_fraction', SYSTEM_CONFIG['max_ram_fraction']))
        if args.max_ram_bytes is not None:
            current_used = psutil.virtual_memory().used
            if current_used > args.max_ram_bytes:
                logger.warning(f"Current memory usage ({current_used/(1024**3):.2f} GB) exceeds set limit ({args.max_ram_bytes/(1024**3):.2f} GB).\n"
                            "Consider closing other programs before training.")

        logger.info(f"OMP_NUM_THREADS (PyTorch): {args.num_threads}")
        logger.info(f"MKL_NUM_THREADS (NumPy): {args.num_cores}")
        logger.info(f"PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"Available CPUs: {mp.cpu_count()}")
        logger.info(f"{'='*80}\n")

    # Calculate default CPU configuration
    system_cpu_count = mp.cpu_count()
    default_num_cores = max(
        SYSTEM_CONFIG['min_cores'],
        int(system_cpu_count * SYSTEM_CONFIG['default_cores_fraction'])
    )
    default_num_threads = max(
        SYSTEM_CONFIG['min_threads'],
        int(system_cpu_count * SYSTEM_CONFIG['default_cores_fraction'])
    )

    # get command line arguments
    args = parse_args()

    # Setup CPU configuration
    setup_cpu_configuration(args)

    if args.chat:
        logger.info('Starting Ollama-compatible server on %s:%s', args.host, args.port)
        get_main_chat_instance(use_cpuonly=args.use_cpuonly, cuda_device=args.cuda_device)
        import uvicorn
        uvicorn.run('main_chat:app', host=args.host, port=args.port, log_level='info')
    elif args.console:
        main = MainChat(args)
        main.performMainChat()
    else:
        logger.info('No mode selected. Use --chat for server or --console for terminal chat.')
