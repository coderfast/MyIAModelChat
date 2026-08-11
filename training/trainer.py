import os
import pickle
import sys
import time
import math
import threading
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler
from datasets import Dataset
from commons.model.chatmodel import ChatModel
import logging
import shutil

from dataclasses import dataclass, field
from typing import Optional, List

from datetime import datetime

# Optional SentencePiece support
try:
    import sentencepiece as spm
    SP_AVAILABLE = True
except Exception:
    spm = None
    SP_AVAILABLE = False

try:
    from commons.tokenizer.bpe_tokenizer import SentencePieceTokenizerWrapper
except ImportError:
    SentencePieceTokenizerWrapper = None

class TrainingStopRequested(Exception):
    """Raised when a stop request is issued from the main thread."""


# Setup logging (configured by main.py)
logger = logging.getLogger(__name__)

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_TOKENIZED_DATASET_DIR = os.path.join(CACHE_DIR, 'prepared_dataset_tokenized')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.pkl')

# Training configuration constants
TRAINING_CONFIG = {
    'batch_size': 4,
    'accumulation_steps': 8,
    'learning_rate': 1e-3,
    'embed_size': 256,
    'hidden_size': 512,
    'grad_clip_norm': 1.0,
    'memory_cleanup_interval': 10,
    # Warm-up settings
    'warm_up': True,
    'warm_up_ratio': 0.1,  # use 10% of dataset for warm-up
    'warm_up_steps': 100,  # maximum batches for warm-up phase
    # Thinking settings
    'thinking_loss_weight': 1.0,  # loss weight for thinking tokens (0.0-1.0)
}

# Model checkpoint configuration
MODEL_CHECKPOINT_DIR = 'checkpoints'
TOKENIZER_VOCAB_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'tokenizer_vocab.json')
LATEST_MODEL_FILE = 'chat_model.pth'


class TokenPairIterableDataset(IterableDataset):
    """Iterable dataset that yields input-output token pairs without materializing all in memory."""
    def __init__(self, sequence_generator, length=None, rank=0, world_size=1):
        self.sequence_generator = sequence_generator
        self._length = length
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        if self._world_size > 1:
            return self._sharded_iter()
        return iter(self.sequence_generator())

    def _sharded_iter(self):
        for i, item in enumerate(self.sequence_generator()):
            if i % self._world_size == self._rank:
                yield item

    def __len__(self):
        if self._length is not None:
            if self._world_size > 1:
                return self._length // self._world_size
            return self._length
        raise TypeError("TokenPairIterableDataset length not set")



@dataclass
class TrainingConfig:
    """Configuration for model training (no CLI args)."""
    epochs: int = 1
    checkpoint_name: str = 'chat_model'
    dataset_source: str = 'dataset_cache'
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] or None=auto
    use_vulkan: bool = False
    num_cores: int = 0
    num_threads: int = 0
    max_ram_fraction: float = 0.75
    max_ram_bytes: Optional[int] = None
    pretrained: bool = False
    thinking_loss_weight: float = 1.0
    thinking_enabled: bool = True
    thinking_max_tokens: int = 64
    statistics: bool = False
    agent_enabled: bool = False
    agent_loss_weight: float = 1.0
    agent_ratio: float = 0.3
    # MoE (Mixture of Experts) fields
    moe_enabled: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_load_balance_weight: float = 0.01
    moe_freeze_attention: bool = False
    # DDP fields (single-machine multi-GPU or multi-node)
    rank: int = 0                # global rank
    local_rank: int = 0          # rank within this machine
    world_size: int = 1          # total number of processes
    master_addr: str = 'localhost'
    master_port: int = 29500

class Trainer:

    def __init__(self, config: TrainingConfig):

        logger.info("MainTrain initializing...")

        # Store config
        self.config = config

        # Check if the tokenizer and cached dataset exist
        self.tokenizer = None
        self.tokenized_data = None
        self.loaded_dataset = None
        self.stop_event = threading.Event()

        # Device and training configuration tracking
        self.use_gpu = False
        self.use_mixed_precision = False
        self.use_gradient_checkpointing = False
        self.best_loss = float('inf')

        # Thinking detection
        self.has_thinking_data = False
        self.thinking_sample_count = 0
        self.thinking_loss_weight = getattr(self.config, 'thinking_loss_weight', None) or TRAINING_CONFIG.get('thinking_loss_weight', 0.5)

        # Memory cap for entire application
        self.max_ram_fraction = getattr(self.config, 'max_ram_fraction', 0.75)
        self.max_ram_bytes = getattr(self.config, 'max_ram_bytes', None)

        # Training parameters
        self.epochs = self.config.epochs
        self.device_mode = getattr(self.config, 'device_mode', 'auto')
        self.gpu_indices = getattr(self.config, 'gpu_indices', None) or []
        self.use_vulkan = getattr(self.config, 'use_vulkan', False)

        # DDP parameters
        self.rank = getattr(self.config, 'rank', 0)
        self.local_rank = getattr(self.config, 'local_rank', 0)
        self.world_size = getattr(self.config, 'world_size', 1)

        # Dynamic checkpoint naming
        self.checkpoint_name = getattr(self.config, 'checkpoint_name', 'chat_model')
        self.dataset_source = getattr(self.config, 'dataset_source', 'dataset_cache')
        self.model_output_path = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}.pth')
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

        # Load cached dataset and metadata (if present)
        self.cache_metadata = {}
        print(f"Loading dataset from cache...")
        self._load_cached_dataset()

        # Create Tokenizer
        print(f"Create Tokenizer")
        # If cache metadata points to a SentencePiece model and SP is available, use it
        bpe_path = None
        try:
            bpe_path = self.cache_metadata.get('bpe_model_path') if isinstance(self.cache_metadata, dict) else None
        except Exception:
            bpe_path = None

        if bpe_path and SP_AVAILABLE and SentencePieceTokenizerWrapper is not None and os.path.exists(bpe_path):
            try:
                self.tokenizer = SentencePieceTokenizerWrapper(bpe_path)
                logger.info(f"Using SentencePiece tokenizer from {bpe_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize SentencePiece tokenizer: {e}. "
                    "Install sentencepiece: pip install sentencepiece"
                )
        else:
            if bpe_path and not SP_AVAILABLE:
                raise RuntimeError(
                    "cache_metadata indicates a BPE model but 'sentencepiece' is not installed. "
                    "Install it: pip install sentencepiece"
                )
            raise RuntimeError(
                "No BPE model found. Run prepare-data first:\n"
                "  python main.py --prepare-data --aiml\n"
                "  python main.py --prepare-data --aiml --hf --pdf --epub\n"
                "  python main.py --prepare-data --aiml --bpe-vocab-size 8000\n"
                "Run 'python main.py --prepare-data --help' for all options."
            )

        # Detect thinking data now that tokenizer is available
        # (must be after tokenizer creation so get_thinking_index works for pre-tokenized data)
        self._detect_thinking_data()

        print(f"MainTrain initialized...")

    def request_stop(self):
        """Request that training stop gracefully."""
        self.stop_event.set()

    def _load_cached_dataset(self):
        """Load dataset from pre-prepared cache."""
        try:
            if not os.path.exists(CACHE_DATASET_FILE):
                raise FileNotFoundError(
                    f"Cached dataset not found at {CACHE_DATASET_FILE}\n"
                    "No cached dataset available for training.\n\n"
                    "Prepare your dataset first:\n"
                    "  python main.py --prepare-data --aiml\n"
                    "  python main.py --prepare-data --aiml --hf --pdf --epub\n"
                    "  python main.py --prepare-data --aiml --bpe-vocab-size 8000\n\n"
                    "Run 'python main.py --prepare-data --help' for all options."
                )
            
            logger.info(f"Loading cached dataset from: {CACHE_DATASET_FILE}")
            self.loaded_dataset = Dataset.load_from_disk(CACHE_DATASET_FILE)
            logger.info(f"Loaded cached dataset: {len(self.loaded_dataset)} samples")
            
            # Load cache metadata if available
            if os.path.exists(CACHE_METADATA_FILE):
                try:
                    with open(CACHE_METADATA_FILE, 'rb') as f:
                        self.cache_metadata = pickle.load(f)
                    logger.info("Cache metadata loaded")
                except Exception as me:
                    logger.warning(f"Could not read cache metadata: {me}")

            # Detect whether cached dataset already contains tokenized ids
            if 'token_ids' in getattr(self.loaded_dataset, 'column_names', []):
                logger.info("Cached dataset contains 'token_ids' - will use pre-tokenized data for training")
            # fallback: if metadata includes a bpe model path, prefer tokenized flow
            elif isinstance(self.cache_metadata, dict) and self.cache_metadata.get('bpe_model_path'):
                logger.info("Cache metadata indicates BPE model present; training will prefer tokenized cache if available")

            # Load statistics if available
            if os.path.exists(CACHE_STATS_FILE):
                with open(CACHE_STATS_FILE, 'rb') as f:
                    stats = pickle.load(f)
                logger.info(f"Dataset statistics loaded")
                logger.info(f"  Total samples: {stats.get('total_samples', 0):,}")

            # NOTE: _detect_thinking_data() is called AFTER tokenizer creation
            # in __init__ because it needs tokenizer.get_thinking_index() to
            # detect thinking tokens in pre-tokenized data.

        except Exception as e:
            logger.error(f"Error loading cached dataset: {e}")
            sys.exit(1)

    def _detect_thinking_data(self):
        """Detect if the dataset contains <thinking> or mode tokens (<|context|>/<|thinking|>)."""
        if self.loaded_dataset is None:
            return

        # Check cache metadata for thinking flag
        if isinstance(self.cache_metadata, dict) and self.cache_metadata.get('has_thinking_tokens'):
            self.has_thinking_data = True
            logger.info("Thinking data detected (from cache metadata)")
            return

        # Heuristic: sample first 100 items and check for thinking/mode tokens
        sample_size = min(100, len(self.loaded_dataset))
        thinking_count = 0
        thinking_id = getattr(self.tokenizer, 'get_thinking_index', lambda: -1)()
        thinking_end_id = getattr(self.tokenizer, 'get_thinking_end_index', lambda: -1)()
        context_id = getattr(self.tokenizer, 'get_context_index', lambda: -1)()
        thinking_mode_id = getattr(self.tokenizer, 'get_thinking_mode_index', lambda: -1)()
        answer_id = getattr(self.tokenizer, 'get_answer_index', lambda: -1)()

        for i in range(sample_size):
            item = self.loaded_dataset[i]
            value = item.get('input_ids', item.get('token_ids', ''))
            if isinstance(value, str) and ('<thinking>' in value or '<|thinking|>' in value or '<|context|>' in value):
                thinking_count += 1
            elif isinstance(value, list):
                if thinking_id >= 0 and thinking_end_id >= 0:
                    if thinking_id in value and thinking_end_id in value:
                        thinking_count += 1
                elif context_id >= 0 or thinking_mode_id >= 0:
                    if context_id in value or thinking_mode_id in value:
                        thinking_count += 1

        if thinking_count > 0:
            self.has_thinking_data = True
            self.thinking_sample_count = thinking_count
            logger.info(f"Thinking data detected ({thinking_count}/{sample_size} samples contain <thinking>)")
        else:
            logger.info("No thinking data detected in dataset")

    def _warn_memory_usage(self, stage="training"):
        """Check and warn about memory usage compared to configured max RAM."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if self.max_ram_bytes is not None:
                usage = mem.used
                if usage > self.max_ram_bytes:
                    logger.warning(
                        f"Memory usage ({usage/(1024**3):.2f} GB) above configured max ({self.max_ram_bytes/(1024**3):.2f} GB) during {stage}."
                    )
                else:
                    logger.info(
                        f"Memory usage ({usage/(1024**3):.2f} GB) within limit ({self.max_ram_bytes/(1024**3):.2f} GB) during {stage}."
                    )
        except ImportError:
            logger.warning("psutil unavailable; cannot monitor RAM usage")

    def _limit_num_workers_by_memory(self, default_workers: int):
        """Heuristic: reduce num_workers when memory limit is low."""
        if self.max_ram_bytes is None:
            return default_workers

        try:
            import psutil
            mem = psutil.virtual_memory()
            free = mem.available
            if free < (self.max_ram_bytes * 0.25):
                return max(1, int(default_workers // 2))
        except ImportError:
            pass
        return default_workers

    def _get_num_proc(self):
        """Choose a safe number of processes for dataset map operations."""
        if sys.version_info >= (3, 14):
            return 0  # dill incompatible
        else:
            try:
                cpus = mp.cpu_count()
                if cpus <= 2:
                    return 1
                return min(4, max(1, cpus // 2))
            except Exception:
                return 0

    def _sample_generator(self):
        """Yield text or tokenized sequences from the loaded dataset in streaming mode."""
        for item in self.loaded_dataset:
            value = item.get('token_ids', item.get('input_ids', None))
            if value is None:
                continue

            if isinstance(value, str):
                raw = value.strip()
                if raw:
                    yield raw, None
            elif isinstance(value, dict):
                input_text = value.get('input', '').strip()
                output_text = value.get('output', '').strip()
                merged = f"{input_text} {output_text}".strip()
                if merged:
                    yield merged, None
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue

                if all(isinstance(v, int) for v in value):
                    yield None, list(value)
                else:
                    merged = " ".join(str(v).strip() for v in value if isinstance(v, str) and str(v).strip())
                    if merged:
                        yield merged, None
            elif hasattr(value, 'tolist'):
                seq = list(value.tolist())
                if seq and all(isinstance(v, int) for v in seq):
                    yield None, seq
                else:
                    text_tokens = " ".join(str(v).strip() for v in seq if str(v).strip())
                    if text_tokens:
                        yield text_tokens, None

    def _setup_device_and_config(self):
        """Setup device configuration with support for CPU, GPU, Vulkan, DDP, and CPU+GPU."""
        from commons.utils.device_utils import check_vulkan_available, resolve_device

        device_mode = self.device_mode
        gpu_indices = self.gpu_indices or []
        use_vulkan = self.use_vulkan

        # --- CPU puro ---
        if device_mode == 'cpu':
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("CPU-ONLY MODE ENABLED")
                logger.info("=" * 80)
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('cpu')

        # --- Vulkan ---
        if use_vulkan:
            if not check_vulkan_available():
                raise RuntimeError("Vulkan requested but not available on this system")
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("VULKAN MODE ENABLED")
                logger.info("=" * 80)
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('vulkan')

        # --- GPU (una o multiples) ---
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            return torch.device('cpu')

        # Set device for this DDP process
        torch.cuda.set_device(self.local_rank)

        # Log detected GPUs (rank-0 only)
        num_gpus = torch.cuda.device_count()
        if self.rank == 0:
            logger.info("=" * 80)
            logger.info(f"GPU DETECTED: {num_gpus} device(s)")
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                mem = props.total_memory / (1024 ** 3)
                cc = f"{props.major}.{props.minor}"
                logger.info(f"  GPU {i}: {name} | VRAM: {mem:.2f} GB | Compute: {cc}")
            logger.info("=" * 80)

        # DDP info
        if self.world_size > 1 and self.rank == 0:
            logger.info(f"DDP mode: {self.world_size} processes, local_rank={self.local_rank}")

        # Configure for specific GPU types
        device_idx = self.local_rank
        device_name = torch.cuda.get_device_name(device_idx)

        if "K80" in device_name or "Tesla" in device_name:
            if self.rank == 0:
                logger.info("Tesla K80 GPU detected - using optimized configuration")
            self.use_gradient_checkpointing = True
            torch.cuda.set_per_process_memory_fraction(0.9)
        else:
            self.use_gradient_checkpointing = False

        self.use_gpu = True
        self.use_mixed_precision = True

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if self.rank == 0:
            logger.info("CuDNN optimization enabled")
            logger.info(f"Mixed Precision Training: {'ENABLED' if self.use_mixed_precision else 'DISABLED'}")
            logger.info(f"Gradient Checkpointing: {'ENABLED' if self.use_gradient_checkpointing else 'DISABLED'}")

        device = torch.device(f'cuda:{device_idx}')
        if self.rank == 0:
            logger.info(f"Training Device: {device}")
        return device

    def _setup_model_with_device_strategy(self, model, device):
        """Setup model with CPU, GPU, DDP, or CPU+GPU device strategy."""
        from commons.utils.device_utils import calculate_layers_for_vram

        device_mode = self.device_mode

        # --- CPU puro ---
        if device_mode == 'cpu':
            model = model.to(device)
            if self.rank == 0:
                logger.info("Model deployed on CPU")
            return model

        # --- CPU+GPU: distribucion por capas ---
        if device_mode == 'cpu+gpu':
            return self._setup_model_cpu_gpu_split(model, device)

        # --- GPU estandar (1 o DDP) ---
        try:
            model = model.to(device)

            # DDP si multiples procesos
            if self.world_size > 1:
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel as DDP

                if not dist.is_initialized():
                    os.environ['MASTER_ADDR'] = getattr(self.config, 'master_addr', 'localhost')
                    os.environ['MASTER_PORT'] = str(getattr(self.config, 'master_port', 29500))
                    dist.init_process_group(
                        backend='nccl',
                        rank=self.rank,
                        world_size=self.world_size
                    )

            # Enable gradient checkpointing BEFORE DDP wrapping
            if self.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if self.rank == 0:
                    logger.info("Gradient checkpointing enabled for memory efficiency")

            if self.world_size > 1:
                model = DDP(model, device_ids=[self.local_rank])
                if self.rank == 0:
                    logger.info(f"Model wrapped with DDP on {self.world_size} processes, local_rank={self.local_rank}")

            if self.rank == 0:
                logger.info(f"Model deployed on GPU: {device}")
            return model

        except RuntimeError as e:
            logger.warning(f"Could not move model to GPU: {e}, falling back to CPU")
            model = model.to(torch.device("cpu"))
            self.use_gpu = False
            self.use_mixed_precision = False
            return model

    def _setup_model_cpu_gpu_split(self, model, gpu_device):
        """Distribute model layers across CPU and GPU based on VRAM budget."""
        import gc
        from commons.utils.device_utils import calculate_layers_for_vram

        model = model.to('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get transformer layers
        transformer_layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer_layers = model.model.transformer.h
        if transformer_layers is None or len(transformer_layers) == 0:
            logger.warning("Cannot split model: no transformer layers found. Using GPU only.")
            return model.to(gpu_device)

        num_layers = len(transformer_layers)
        gpu_props = torch.cuda.get_device_properties(gpu_device)
        gpu_memory_gb = gpu_props.total_memory / (1024 ** 3)
        vram_budget = gpu_memory_gb * 0.80  # 80% usable

        # Estimate memory per layer (params * 4 bytes FP32)
        total_layer_params = sum(p.numel() for p in transformer_layers.parameters())
        params_per_layer = total_layer_params / num_layers
        mem_per_layer_gb = (params_per_layer * 4) / (1024 ** 3)

        # Embeddings + lm_head memory
        overhead_params = 0
        if hasattr(model.model.transformer, 'wte'):
            overhead_params += sum(p.numel() for p in model.model.transformer.wte.parameters())
        if hasattr(model.model.transformer, 'wpe'):
            overhead_params += sum(p.numel() for p in model.model.transformer.wpe.parameters())
        if hasattr(model, 'lm_head'):
            overhead_params += sum(p.numel() for p in model.lm_head.parameters())
        overhead_gb = (overhead_params * 4) / (1024 ** 3)

        # Calculate layers that fit
        available_for_layers = vram_budget - overhead_gb
        layers_on_gpu = min(num_layers, max(1, int(available_for_layers / mem_per_layer_gb))) if available_for_layers > 0 else 0

        logger.info("=" * 80)
        logger.info("CPU+GPU MODEL SPLIT")
        logger.info(f"  GPU: {gpu_props.name} ({gpu_memory_gb:.2f} GB)")
        logger.info(f"  Total layers: {num_layers}")
        logger.info(f"  VRAM budget: {vram_budget:.2f} GB")
        logger.info(f"  Overhead (embeddings + head): {overhead_gb:.4f} GB")
        logger.info(f"  Layers on GPU: {layers_on_gpu}")
        logger.info(f"  Layers on CPU: {num_layers - layers_on_gpu}")
        logger.info("=" * 80)

        # Move embeddings + head to GPU
        if hasattr(model.model.transformer, 'wte'):
            model.model.transformer.wte = model.model.transformer.wte.to(gpu_device)
        if hasattr(model.model.transformer, 'wpe'):
            model.model.transformer.wpe = model.model.transformer.wpe.to(gpu_device)
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(gpu_device)

        # Move first N layers to GPU
        for i in range(layers_on_gpu):
            transformer_layers[i] = transformer_layers[i].to(gpu_device)

        self._gpu_layers_count = layers_on_gpu
        self._gpu_device = gpu_device

        return model

    def _compute_loss(self, model, inputs, targets, criterion):
        """Unified loss computation with thinking-aware, mode-aware, and agentic loss weighting.

        Loss rules by sample type:
        - CONTEXT samples (<|context|> prefix): tokens before <|answer|> get weight 0.0,
          <|answer|> and answer tokens get weight 1.0.
        - THINKING samples (<|thinking|> prefix): tokens before <thinking> get weight 0.0,
          <thinking>...</thinking> tokens get thinking_loss_weight, <|answer|> and answer
          tokens get weight 1.0.
        - AGENT samples (with <tool_call>): thinking gets thinking_loss_weight, tool_call and
          observation tokens get full weight (1.0), answer tokens get full weight (1.0).
        - MoE models: adds load balancing loss to encourage uniform expert usage.
        """
        # Handle MoE model output (returns logits + gate_scores)
        gate_scores = None
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs, gate_scores = outputs
        raw_outputs = outputs

        weights = torch.ones_like(targets, dtype=torch.float)

        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()
        context_id = getattr(self.tokenizer, 'get_context_index', lambda: -1)()
        answer_id = getattr(self.tokenizer, 'get_answer_index', lambda: -1)()
        thinking_mode_id = getattr(self.tokenizer, 'get_thinking_mode_index', lambda: -1)()
        tool_call_id = getattr(self.tokenizer, 'get_tool_call_index', lambda: -1)()
        tool_call_end_id = getattr(self.tokenizer, 'get_tool_call_end_index', lambda: -1)()
        observation_id = getattr(self.tokenizer, 'get_observation_index', lambda: -1)()
        observation_end_id = getattr(self.tokenizer, 'get_observation_end_index', lambda: -1)()

        agent_enabled = getattr(self.config, 'agent_enabled', False)
        agent_loss_weight = getattr(self.config, 'agent_loss_weight', 1.0)

        batch_size, seq_len = targets.shape

        for b in range(batch_size):
            has_context_prefix = False
            has_thinking_prefix = False

            # Detect mode from prefix token
            first_non_pad = -1
            for s in range(seq_len):
                tok = targets[b, s].item()
                if tok != self.tokenizer.get_pad_index() and tok != self.tokenizer.get_unk_index():
                    first_non_pad = s
                    break

            if first_non_pad >= 0:
                first_token = targets[b, first_non_pad].item()
                if first_token == context_id:
                    has_context_prefix = True
                elif first_token == thinking_mode_id:
                    has_thinking_prefix = True

            if has_context_prefix:
                # CONTEXT mode: zero loss for everything before <|answer|>
                in_preamble = True
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == answer_id:
                        in_preamble = False
                        weights[b, s] = 1.0  # <|answer|> delimiter gets full weight
                    elif in_preamble:
                        weights[b, s] = 0.0

            elif has_thinking_prefix:
                # THINKING mode: zero loss for preamble (before <thinking>),
                # reduced weight for thinking content, full weight after <|answer|>
                in_preamble = True
                in_thinking = False
                past_answer = False

                for s in range(seq_len):
                    token = targets[b, s].item()

                    if token == thinking_id:
                        in_preamble = False
                        in_thinking = True
                        weights[b, s] = 1.0  # <thinking> delimiter gets full weight
                    elif token == thinking_end_id:
                        in_thinking = False
                        weights[b, s] = 1.0  # </thinking> delimiter gets full weight
                    elif token == answer_id:
                        past_answer = True
                        in_thinking = False
                        weights[b, s] = 1.0  # <|answer|> delimiter gets full weight
                    elif in_preamble:
                        weights[b, s] = 0.0
                    elif in_thinking:
                        weights[b, s] = self.thinking_loss_weight
                    else:
                        weights[b, s] = 1.0

            elif self.has_thinking_data:
                # Legacy mode: detect <thinking>...</thinking> without mode prefix
                in_thinking = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == thinking_end_id:
                        in_thinking = False
                    if token == thinking_id:
                        in_thinking = True
                    if in_thinking and token != thinking_id and token != thinking_end_id:
                        weights[b, s] = self.thinking_loss_weight

            # Agentic masking: override weights for tool_call and observation tokens
            if agent_enabled and tool_call_id >= 0:
                in_tool_call = False
                in_observation = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == tool_call_id:
                        in_tool_call = True
                        weights[b, s] = agent_loss_weight  # tool_call gets full weight
                    elif token == tool_call_end_id:
                        in_tool_call = False
                        weights[b, s] = agent_loss_weight
                    elif token == observation_id:
                        in_observation = True
                        weights[b, s] = agent_loss_weight
                    elif token == observation_end_id:
                        in_observation = False
                        weights[b, s] = agent_loss_weight
                    elif in_tool_call:
                        weights[b, s] = agent_loss_weight  # JSON inside tool_call
                    elif in_observation:
                        weights[b, s] = agent_loss_weight  # observation content

        # Flatten outputs and targets
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.contiguous().view(-1)
        weights = weights.contiguous().view(-1)

        # Ignore padded elements
        non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
        outputs = outputs[non_pad_mask]
        targets = targets[non_pad_mask]
        weights = weights[non_pad_mask]

        # Weighted cross-entropy loss
        token_losses = criterion(outputs, targets)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True), raw_outputs
        loss = (token_losses * weights).sum() / weight_sum
        
        # Add load balancing loss for MoE models
        moe_loss = torch.tensor(0.0, device=loss.device)
        if gate_scores is not None and hasattr(model, 'get_load_balancing_loss'):
            moe_loss = model.get_load_balancing_loss(torch.stack(gate_scores))
            loss = loss + self.config.moe_load_balance_weight * moe_loss
        
        return loss, raw_outputs

    def _compute_thinking_metrics(self, model, inputs, targets, device, logits=None):
        """Compute metrics for thinking token generation if thinking data is present."""
        if not self.has_thinking_data:
            return {}

        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()

        if thinking_id < 0 or thinking_end_id < 0:
            return {}

        with torch.no_grad():
            outputs = logits if logits is not None else model(inputs)
            # Handle MoE model output (returns logits + gate_scores)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=-1)

            thinking_open_correct = 0
            thinking_close_correct = 0
            total_thinking_positions = 0
            thinking_token_count = 0
            response_token_count = 0
            thinking_correct = 0
            response_correct = 0

            for i in range(targets.size(0)):
                in_thinking = False
                for j in range(targets.size(1)):
                    target_token = targets[i, j].item()
                    pred_token = predictions[i, j].item()

                    if target_token == thinking_id:
                        in_thinking = True
                        total_thinking_positions += 1
                        thinking_token_count += 1
                        if pred_token == thinking_id:
                            thinking_open_correct += 1
                            thinking_correct += 1
                    elif target_token == thinking_end_id:
                        in_thinking = False
                        total_thinking_positions += 1
                        thinking_token_count += 1
                        if pred_token == thinking_end_id:
                            thinking_close_correct += 1
                            thinking_correct += 1
                    elif in_thinking:
                        thinking_token_count += 1
                        if pred_token == target_token:
                            thinking_correct += 1
                    else:
                        response_token_count += 1
                        if pred_token == target_token:
                            response_correct += 1

            metrics = {}
            if total_thinking_positions > 0:
                metrics['thinking_token_accuracy'] = (thinking_open_correct + thinking_close_correct) / total_thinking_positions
                metrics['thinking_open_accuracy'] = thinking_open_correct / max(1, sum(1 for t in targets.flatten() if t.item() == thinking_id))
                metrics['thinking_close_accuracy'] = thinking_close_correct / max(1, sum(1 for t in targets.flatten() if t.item() == thinking_end_id))
                metrics['thinking_positions'] = total_thinking_positions

            if thinking_token_count > 0:
                metrics['thinking_length_avg'] = thinking_token_count / targets.size(0)
                metrics['thinking_coverage'] = thinking_token_count / max(1, thinking_token_count + response_token_count)
                metrics['thinking_token_accuracy_full'] = thinking_correct / thinking_token_count

            if response_token_count > 0:
                metrics['response_token_accuracy'] = response_correct / response_token_count

            return metrics

    def _compute_agent_metrics(self, model, inputs, targets, device, logits=None):
        """Compute metrics for agentic token generation if agent data is present."""
        agent_enabled = getattr(self.config, 'agent_enabled', False)
        if not agent_enabled:
            return {}

        tool_call_id = getattr(self.tokenizer, 'get_tool_call_index', lambda: -1)()
        tool_call_end_id = getattr(self.tokenizer, 'get_tool_call_end_index', lambda: -1)()
        observation_id = getattr(self.tokenizer, 'get_observation_index', lambda: -1)()

        if tool_call_id < 0:
            return {}

        with torch.no_grad():
            outputs = logits if logits is not None else model(inputs)
            # Handle MoE model output (returns logits + gate_scores)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=-1)

            tool_call_count = 0
            tool_call_correct = 0
            tool_name_count = 0
            tool_name_correct = 0
            observation_count = 0
            observation_correct = 0
            total_agent_tokens = 0

            for i in range(targets.size(0)):
                in_tool_call = False
                in_observation = False
                is_tool_name = False

                for j in range(targets.size(1)):
                    target_token = targets[i, j].item()
                    pred_token = predictions[i, j].item()

                    if target_token == tool_call_id:
                        in_tool_call = True
                        is_tool_name = True
                        tool_call_count += 1
                        total_agent_tokens += 1
                        if pred_token == tool_call_id:
                            tool_call_correct += 1
                    elif target_token == tool_call_end_id:
                        in_tool_call = False
                        is_tool_name = False
                        tool_call_count += 1
                        total_agent_tokens += 1
                        if pred_token == tool_call_end_id:
                            tool_call_correct += 1
                    elif target_token == observation_id:
                        in_observation = True
                        in_tool_call = False
                        observation_count += 1
                        total_agent_tokens += 1
                        if pred_token == observation_id:
                            observation_correct += 1
                    elif in_tool_call:
                        total_agent_tokens += 1
                        if pred_token == target_token:
                            tool_call_correct += 1
                    elif in_observation:
                        total_agent_tokens += 1
                        if pred_token == target_token:
                            observation_correct += 1

            metrics = {}
            if tool_call_count > 0:
                metrics['agent_tool_call_accuracy'] = tool_call_correct / tool_call_count
                metrics['agent_tool_call_count'] = tool_call_count
            if observation_count > 0:
                metrics['agent_observation_accuracy'] = observation_correct / observation_count
            if total_agent_tokens > 0:
                metrics['agent_total_tokens'] = total_agent_tokens
                metrics['agent_ratio'] = total_agent_tokens / max(1, targets.size(0) * targets.size(1))

            return metrics
    
    def _backward_pass(self, loss, optimizer, scaler, accumulation_step=1):
        """Unified backward pass handling for mixed and standard precision."""
        # Scale loss for gradient accumulation
        loss = loss / accumulation_step
        
        if self.use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss * accumulation_step  # Return original loss for tracking

    # Training function with mixed precision support
    def train(self, model, dataloader, criterion, optimizer, device, scaler=None, accumulation_steps=1, num_batches_override=None, is_warmup=False):
        """Train function with support for mixed precision training and gradient accumulation."""

        total_loss = 0
        total_batches = 0
        num_batches = 0
        phase_label = "Light training (warm-up)" if is_warmup else "Training"
        # Try to get total batch count
        known_total = num_batches_override
        if known_total is None:
            try:
                known_total = len(dataloader)
            except (TypeError, AttributeError):
                known_total = None
        model.train()
        optimizer.zero_grad()
        epoch_start_time = time.time()
        if self.rank == 0:
            logger.info("Training loop started; the model is actively processing batches")

        # Timing accumulators for micro-steps and optimizer updates (running stats)
        micro_step_count = 0
        micro_step_sum = 0.0
        micro_step_min = float('inf')
        micro_step_max = 0.0
        micro_step_last = 0.0
        optimizer_step_count = 0
        optimizer_step_sum = 0.0
        optimizer_step_min = float('inf')
        optimizer_step_max = 0.0
        optimizer_step_last = 0.0

        # Thinking metrics accumulators
        thinking_metrics_accum = {}
        agent_metrics_accum = {}
        last_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if self.stop_event.is_set():
                if self.rank == 0:
                    logger.info("Stop requested; exiting current training epoch early")
                break

            if (batch_idx == 0 or (batch_idx + 1) % 10 == 0) and self.rank == 0:
                elapsed = time.time() - epoch_start_time
                current_batch = batch_idx + 1
                # Show total if we know it; otherwise just show current
                if known_total is not None and known_total > 1:
                    total_str = f"/{known_total}"
                    remaining = max(0, known_total - current_batch)
                else:
                    total_str = ""
                    remaining = 0
                if batch_idx > 0:
                    rate = current_batch / elapsed
                    if remaining > 0:
                        eta_seconds = remaining / rate
                        eta_m, eta_s = divmod(int(eta_seconds), 60)
                        eta_str = f"{eta_m}m {eta_s}s" if eta_m > 0 else f"{eta_s}s"
                    else:
                        # Unknown remaining: just show elapsed
                        e_m, e_s = divmod(int(elapsed), 60)
                        eta_str = f"elapsed {e_m}m {e_s}s" if e_m > 0 else f"elapsed {e_s}s"
                    loss_str = f" | loss: {last_loss:.4f}" if last_loss > 0 else ""
                    logger.info(f"{phase_label} batch {current_batch}{total_str} | ETA: {eta_str}{loss_str}")
                else:
                    logger.info(f"{phase_label} batch {current_batch}{total_str} in progress...")

            total_batches += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            # --- Micro-step timing (forward + backward) ---
            micro_step_start = time.time()

            # Determine if this is the last micro-step (should sync gradients)
            is_last_micro_step = ((batch_idx + 1) % accumulation_steps == 0)

            # Forward pass
            if self.use_mixed_precision and scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss, logits = self._compute_loss(model, inputs, targets, criterion)
            else:
                loss, logits = self._compute_loss(model, inputs, targets, criterion)

            last_loss = loss.item()

            # Compute thinking metrics periodically (reuse logits from _compute_loss)
            if self.has_thinking_data and (batch_idx + 1) % 50 == 0:
                thinking_metrics = self._compute_thinking_metrics(model, inputs, targets, device, logits=logits)
                for key, value in thinking_metrics.items():
                    if key not in thinking_metrics_accum:
                        thinking_metrics_accum[key] = []
                    thinking_metrics_accum[key].append(value)

            # Compute agent metrics periodically
            agent_enabled = getattr(self.config, 'agent_enabled', False)
            if agent_enabled and (batch_idx + 1) % 50 == 0:
                agent_metrics = self._compute_agent_metrics(model, inputs, targets, device, logits=logits)
                for key, value in agent_metrics.items():
                    if key not in agent_metrics_accum:
                        agent_metrics_accum[key] = []
                    agent_metrics_accum[key].append(value)

            # Backward pass — use no_sync() for DDP when not at accumulation boundary
            if self.world_size > 1 and hasattr(model, 'no_sync') and not is_last_micro_step:
                with model.no_sync():
                    original_loss = self._backward_pass(loss, optimizer, scaler, accumulation_steps)
            else:
                original_loss = self._backward_pass(loss, optimizer, scaler, accumulation_steps)
                total_loss += original_loss.item()

            micro_step_end = time.time()
            micro_step_elapsed = micro_step_end - micro_step_start
            micro_step_count += 1
            micro_step_sum += micro_step_elapsed
            micro_step_min = min(micro_step_min, micro_step_elapsed)
            micro_step_max = max(micro_step_max, micro_step_elapsed)
            micro_step_last = micro_step_elapsed

            # Log micro-step only if --statistics is enabled (rank-0 only)
            if self.config.statistics and self.rank == 0:
                logger.info(
                    f"  [micro-step {batch_idx + 1}] "
                    f"loss={original_loss.item():.4f} | "
                    f"time={micro_step_elapsed*1000:.1f}ms"
                )

            # --- Optimizer step timing (every accumulation_steps) ---
            if (batch_idx + 1) % accumulation_steps == 0:
                opt_step_start = time.time()

                if self.use_mixed_precision and scaler is not None:
                    scaler.unscale_(optimizer)

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG['grad_clip_norm'])

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                opt_step_end = time.time()
                opt_step_elapsed = opt_step_end - opt_step_start
                optimizer_step_count += 1
                optimizer_step_sum += opt_step_elapsed
                optimizer_step_min = min(optimizer_step_min, opt_step_elapsed)
                optimizer_step_max = max(optimizer_step_max, opt_step_elapsed)
                optimizer_step_last = opt_step_elapsed

                # Log optimizer step only if --statistics is enabled (rank-0 only)
                if self.config.statistics and self.rank == 0:
                    accum_group = (batch_idx + 1) // accumulation_steps
                    logger.info(
                        f"  [optimizer step #{accum_group}] "
                        f"opt_time={opt_step_elapsed*1000:.1f}ms | "
                        f"micro_avg={micro_step_last*1000:.1f}ms | "
                        f"ratio(opt/total)={opt_step_elapsed/(opt_step_elapsed + micro_step_last)*100:.1f}%"
                    )

            num_batches = max(1, total_batches)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Memory cleanup at epoch boundary
        if self.use_gpu:
            torch.cuda.empty_cache()

        # Log thinking metrics summary (rank-0 only)
        if thinking_metrics_accum and self.rank == 0:
            logger.info("Thinking Metrics Summary:")
            for key, values in thinking_metrics_accum.items():
                avg_val = sum(values) / len(values) if values else 0
                logger.info(f"    {key}: {avg_val:.4f}")

        # Log agent metrics summary (rank-0 only)
        if agent_metrics_accum and self.rank == 0:
            logger.info("Agent Metrics Summary:")
            for key, values in agent_metrics_accum.items():
                avg_val = sum(values) / len(values) if values else 0
                logger.info(f"    {key}: {avg_val:.4f}")

        # --- Timing summary (rank-0 only) ---
        total_elapsed = time.time() - epoch_start_time
        if self.rank == 0:
            if micro_step_count > 0:
                micro_avg = micro_step_sum / micro_step_count
                logger.info("=" * 80)
                logger.info(f"TIMING SUMMARY ({phase_label})")
                logger.info(f"  Micro-steps (forward+backward):")
                logger.info(f"    Count:    {micro_step_count}")
                logger.info(f"    Avg:      {micro_avg*1000:.1f}ms")
                logger.info(f"    Min:      {micro_step_min*1000:.1f}ms")
                logger.info(f"    Max:      {micro_step_max*1000:.1f}ms")
                logger.info(f"    Total:    {micro_step_sum*1000:.1f}ms")

            if optimizer_step_count > 0:
                opt_avg = optimizer_step_sum / optimizer_step_count
                logger.info(f"  Optimizer updates:")
                logger.info(f"    Count:    {optimizer_step_count}")
                logger.info(f"    Avg:      {opt_avg*1000:.1f}ms")
                logger.info(f"    Min:      {optimizer_step_min*1000:.1f}ms")
                logger.info(f"    Max:      {optimizer_step_max*1000:.1f}ms")
                logger.info(f"    Total:    {optimizer_step_sum*1000:.1f}ms")

                # Overhead analysis
                compute_total = micro_step_sum
                overhead = total_elapsed - compute_total - optimizer_step_sum
                logger.info(f"  Overhead (data loading, logging, etc): {overhead*1000:.1f}ms ({overhead/total_elapsed*100:.1f}%)")
                logger.info(f"  Compute fraction:  {compute_total/total_elapsed*100:.1f}%")
                logger.info(f"  Optimizer fraction: {optimizer_step_sum/total_elapsed*100:.1f}%")

        if self.rank == 0:
            logger.info(f"  Total epoch time: {total_elapsed:.2f}s")
            logger.info("=" * 80)

            logger.info(f"Training loop completed after {num_batches} batches")
        return avg_loss


    def collate_fn(self, batch):
        """Collate function for DataLoader with proper padding and truncation."""
        input_sequences, output_sequences = zip(*batch)

        max_positions = 512  # n_positions from GPT2Config

        # Truncate sequences that exceed max_positions
        input_sequences = [seq[:max_positions] for seq in input_sequences]
        output_sequences = [seq[:max_positions] for seq in output_sequences]

        # Find max length across all sequences
        max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))

        # Pad sequences to max length
        pad_idx = self.tokenizer.get_pad_index()
        padded_inputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in input_sequences]
        padded_outputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in output_sequences]

        # Convert to tensors
        input_tensor = torch.LongTensor(padded_inputs)
        output_tensor = torch.LongTensor(padded_outputs)

        return input_tensor, output_tensor

    def _get_csv_dataloader(self, batch_size):
        """Build a dataloader from CSV data for fine-tuning."""
        if getattr(self, 'raw_dataset', None) is None:
            return None

        def csv_pair_generator():
            for item in self.raw_dataset:
                value = item.get('input_ids', None)
                if not isinstance(value, str):
                    continue
                if not value.startswith('Pregunta:'):
                    continue
                token_seq = self.tokenizer.encode(value)
                if not token_seq or len(token_seq) <= 1:
                    continue
                yield token_seq[:-1], token_seq[1:]

        csv_length = len(self.raw_dataset) if hasattr(self.raw_dataset, '__len__') else None
        return DataLoader(
            TokenPairIterableDataset(csv_pair_generator, length=csv_length, rank=self.rank, world_size=self.world_size),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.use_gpu,
            num_workers=0
        )

    def performMainTrain(self):
        """Main training loop with proper error handling and model checkpointing."""
        
        try:
            # Use the cached dataset
            _pre_tokenized_dataset = self.loaded_dataset

            logger.info(f"Dataset size: {len(_pre_tokenized_dataset)} samples")
            logger.info(f"Dataset columns: {_pre_tokenized_dataset.column_names}")

            # Log thinking data status
            if self.has_thinking_data:
                logger.info(f" Thinking data: ENABLED (model will learn <thinking>...</thinking> structure)")
            else:
                logger.info(f"Thinking data: NOT detected (standard training mode)")

            # Extract and normalize records from cached dataset in streaming mode
            logger.info("Extracting and validating dataset samples (streaming mode)...")

            # Fit tokenizer incrementally to avoid memory spikes
            tokenizer_batch = []
            tokenizer_batch_size = 1000
            if self.max_ram_bytes:
                tokenizer_batch_size = max(128, int((self.max_ram_bytes / (1024**2)) // 10))

            sample_count = 0
            for text, token_seq in self._sample_generator():
                sample_count += 1
                if text is not None:
                    tokenizer_batch.append(text)
                    if len(tokenizer_batch) >= tokenizer_batch_size:
                        self.tokenizer.fit(tokenizer_batch)
                        tokenizer_batch.clear()

            if tokenizer_batch:
                self.tokenizer.fit(tokenizer_batch)
                tokenizer_batch.clear()

            if sample_count == 0:
                logger.error("No valid text/token sequences found in cached dataset")
                sys.exit(1)

            if self.rank == 0:
                logger.info(f" Processed {sample_count} samples for tokenizer fitting")
                logger.info(f" Tokenizer vocabulary size: {self.tokenizer.vocab_size}")

            # Save tokenizer vocabulary for chat loading inside checkpoints
            if self.rank == 0:
                self.tokenizer.save_vocabulary(TOKENIZER_VOCAB_FILE)
                logger.info(f" Tokenizer vocabulary saved to {TOKENIZER_VOCAB_FILE}")

            # Preserve raw dataset text for special facts fine-tuning before tokenization
            self.raw_dataset = _pre_tokenized_dataset

            # Build or load pretokenized cache for faster training iterations
            self.loaded_dataset = self._get_tokenized_dataset()
            self._warn_memory_usage(stage="data preparation")

            def token_pair_generator():
                for text, token_seq in self._sample_generator():
                    if token_seq is None:
                        token_seq = self.tokenizer.encode(text)
                    if not token_seq or len(token_seq) <= 1:
                        continue
                    input_ids = token_seq[:-1]
                    output_ids = token_seq[1:]
                    yield input_ids, output_ids

            dataset_length = len(self.loaded_dataset) if hasattr(self.loaded_dataset, '__len__') else None
            iterable_dataset = TokenPairIterableDataset(
                token_pair_generator,
                length=dataset_length,
                rank=self.rank,
                world_size=self.world_size
            )

            # Create DataLoader with custom collate function
            if self.rank == 0:
                logger.info(f"Creating DataLoader (batch_size={TRAINING_CONFIG['batch_size']})...")
            pin_memory = self.use_gpu  # Only use pin_memory if GPU is available
            # num_workers = self._limit_num_workers_by_memory(max(1, min(2, mp.cpu_count() // 2)))

            dataloader = DataLoader(
                iterable_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=False,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                num_workers=0
            )

            if self.rank == 0:
                logger.info(" DataLoader created")
        
        except Exception as e:
            logger.error(f"Error in data preparation: {e}", exc_info=True)
            sys.exit(1)

        try:
            # CPU configuration already set in main.py entry point
            var_num_cores = os.environ.get("MKL_NUM_THREADS", mp.cpu_count())
            var_num_threads = os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())
            if self.rank == 0:
                logger.info(f"CPU configuration - Cores: {var_num_cores}, Threads: {var_num_threads}")

            # Setup device with improved configuration for CPU-GPU combined training
            device = self._setup_device_and_config()

            # Initialize model — resume from checkpoint if it exists (skip if --pretrained)
            resume_checkpoint = None
            use_pretrained = getattr(self.config, 'pretrained', False)
            if use_pretrained and self.rank == 0:
                logger.info("--pretrained flag detected, ignoring existing checkpoint (fresh start with GPT-2 weights)")

            if not use_pretrained and os.path.exists(self.model_output_path):
                if self.rank == 0:
                    logger.info(f"Found existing checkpoint: {self.model_output_path}")
                    logger.info("Resuming training from checkpoint...")
                resume_checkpoint = torch.load(self.model_output_path, map_location='cpu', weights_only=False)
                arch = resume_checkpoint.get('architecture', {})
                embed_size = arch.get('embed_size', TRAINING_CONFIG['embed_size'])
                num_layers = arch.get('num_layers', 4)
                checkpoint_vocab_size = arch.get('vocab_size', self.tokenizer.vocab_size)
                current_vocab_size = self.tokenizer.vocab_size
                
                model = ChatModel(self.tokenizer, embed_size=embed_size, num_layers=num_layers)
                
                # Handle vocab_size mismatch
                if checkpoint_vocab_size != current_vocab_size:
                    if self.rank == 0:
                        logger.warning(f" vocab_size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}")
                        logger.info(" Loading only transformer layers (skipping embedding/head layers)...")
                    
                    # Filter out embedding and head layers
                    state_dict = resume_checkpoint['model_state_dict']
                    filtered_state_dict = {}
                    for k, v in state_dict.items():
                        # Skip embedding and head layers if vocab_size mismatch
                        if 'wte.weight' in k or 'lm_head.weight' in k:
                            if v.shape[0] != current_vocab_size:
                                if self.rank == 0:
                                    logger.info(f"   Skipping {k}: shape {v.shape} -> will be retrained")
                                continue
                        filtered_state_dict[k] = v
                    
                    # Load with strict=False to allow missing keys
                    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                    if self.rank == 0 and missing:
                        logger.info(f" Missing keys (will be retrained): {len(missing)}")
                else:
                    model.load_state_dict(resume_checkpoint['model_state_dict'])
                
                if self.rank == 0:
                    logger.info(f" Model loaded from checkpoint (epoch {resume_checkpoint.get('epoch', '?')}, loss {resume_checkpoint.get('loss', '?'):.4f})")
            else:
                if self.rank == 0:
                    msg = f"Initializing ChatModel (embed_size={TRAINING_CONFIG['embed_size']}, num_layers=4"
                    if use_pretrained:
                        msg += ", pretrained=True"
                    logger.info(f"{msg})...")
                model = ChatModel(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], num_layers=4, pretrained=use_pretrained)

            model = self._setup_model_with_device_strategy(model, device)
            if self.rank == 0:
                logger.info(f" Model initialized and deployed")

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_index(), reduction='none')
            optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
            
            # Add learning rate scheduler
            scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, self.epochs // 3), gamma=0.1)

            # Restore optimizer and scheduler state if resuming
            if resume_checkpoint is not None:
                # Skip optimizer state if vocab_size mismatch (shapes won't match)
                if checkpoint_vocab_size != current_vocab_size:
                    if self.rank == 0:
                        logger.info(" Skipping optimizer/scheduler restore due to vocab_size mismatch")
                else:
                    if 'optimizer_state_dict' in resume_checkpoint:
                        try:
                            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                            if self.rank == 0:
                                logger.info(" Optimizer state restored")
                        except Exception as e:
                            if self.rank == 0:
                                logger.warning(f" Could not restore optimizer state: {e}")
                    if 'scheduler_state_dict' in resume_checkpoint:
                        try:
                            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                            if self.rank == 0:
                                logger.info(" Scheduler state restored")
                        except Exception as e:
                            if self.rank == 0:
                                logger.warning(f" Could not restore scheduler state: {e}")
                
                if 'loss' in resume_checkpoint:
                    self.best_loss = resume_checkpoint['loss']
                    if self.rank == 0:
                        logger.info(f" Best loss restored: {self.best_loss:.4f}")

            if self.rank == 0:
                logger.info(f" Learning rate scheduler: StepLR (step_size={scheduler.step_size}, gamma={scheduler.gamma})")

            # Initialize gradient scaler for mixed precision training
            scaler = GradScaler() if self.use_mixed_precision else None
            if scaler and self.rank == 0:
                logger.info(" Gradient scaler initialized for mixed precision training")

            # Checkpoint directory already created in __init__
            if self.rank == 0:
                logger.info(f" Checkpoint directory: {MODEL_CHECKPOINT_DIR}")

            # Warm-up phase (optional light training)
            if TRAINING_CONFIG.get('warm_up', False):
                warm_up_ratio = float(TRAINING_CONFIG.get('warm_up_ratio', 0.1))
                warm_up_steps = int(TRAINING_CONFIG.get('warm_up_steps', 100))
                warmup_limit = max(1, int(len(_pre_tokenized_dataset) * warm_up_ratio))

                if self.rank == 0:
                    logger.info(f"Starting warm-up phase (light training) with {warmup_limit} samples and up to {warm_up_steps} batches...")

                def warmup_pair_generator():
                    idx = 0
                    for input_ids, output_ids in token_pair_generator():
                        if idx >= warmup_limit:
                            break
                        yield input_ids, output_ids
                        idx += 1

                warmup_dataloader = DataLoader(
                    TokenPairIterableDataset(warmup_pair_generator, rank=self.rank, world_size=self.world_size),
                    batch_size=TRAINING_CONFIG['batch_size'],
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    pin_memory=pin_memory,
                    num_workers=0
                )

                warmup_num_batches = max(1, math.ceil(warmup_limit / TRAINING_CONFIG['batch_size']))
                _ = self.train(model, warmup_dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'], num_batches_override=warmup_num_batches, is_warmup=True)
                if self.rank == 0:
                    logger.info(" Warm-up phase completed")

            # Main training loop
            if self.rank == 0:
                logger.info(f"Starting main training phase ({self.epochs} epochs)...")
                logger.info("=" * 80)
            
            num_epochs = self.epochs
            for epoch in range(num_epochs):
                if self.stop_event.is_set():
                    logger.warning("Training stop requested; ending before next epoch")
                    break

                # Training
                loss = self.train(model, dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                
                # Update learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Memory and performance information (rank-0 only)
                if self.rank == 0:
                    if self.use_gpu:
                        gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                        logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Training loss (avg per batch): {loss:.4f} | Learning rate (LR): {current_lr:.2e} | GPU mem used: {gpu_memory:.2f}GB / reserved: {gpu_memory_reserved:.2f}GB")
                    else:
                        logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Training loss (avg per batch): {loss:.4f} | Learning rate (LR): {current_lr:.2e}")

                # Save best model checkpoint after each epoch (rank-0 only)
                if loss < self.best_loss:
                    self.best_loss = loss
                    if self.rank == 0:
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        epoch_filename = f"{self.checkpoint_name}_epoch_{epoch+1}_{timestamp}.pth"
                        epoch_path = os.path.join(MODEL_CHECKPOINT_DIR, epoch_filename)
                        # Use model.module.state_dict() for DDP to remove 'module.' prefix
                        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss,
                            'tokenizer_path': os.path.join(CACHE_DIR, 'sentencepiece.model'),
                            'model_name': self.checkpoint_name,
                            'architecture': {
                                'embed_size': TRAINING_CONFIG['embed_size'],
                                'hidden_size': TRAINING_CONFIG['hidden_size'],
                                'num_layers': TRAINING_CONFIG.get('num_layers', 4),
                                'n_head': TRAINING_CONFIG.get('n_head', 4),
                                'n_positions': TRAINING_CONFIG.get('n_positions', 512),
                                'vocab_size': self.tokenizer.vocab_size,
                            },
                            'dataset_source': self.dataset_source,
                            'used_pretrained': getattr(self.config, 'pretrained', False),
                        }, epoch_path)
                        logger.info(f"Epoch {epoch+1} checkpoint saved to {epoch_path}")
                    # Synchronize all processes after checkpoint save
                    if self.world_size > 1:
                        import torch.distributed as dist
                        dist.barrier()

            # Stop requested? Do not write a partial final checkpoint.
            if self.stop_event.is_set():
                if self.rank == 0:
                    logger.info("Stop requested; skipping final model save")
                return

            # Fine-tune on CSV data if available
            csv_dataloader = self._get_csv_dataloader(batch_size=TRAINING_CONFIG['batch_size'])
            if csv_dataloader is not None:
                if self.rank == 0:
                    logger.info("Starting CSV data fine-tuning...")
                try:
                    _ = self.train(model, csv_dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                    if self.rank == 0:
                        logger.info(" CSV data fine-tuning completed")
                except Exception as e:
                    logger.warning(f"CSV data fine-tuning failed: {e}")
                finally:
                    self.raw_dataset = None  # Free memory after CSV fine-tuning

            # Save final model + tokenizer state for consistent inference (rank-0 only)
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("[OK] Training completed! Saving final model and tokenizer...")

                # Use model.module.state_dict() for DDP to remove 'module.' prefix
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': self.best_loss,
                    'tokenizer_path': os.path.join(CACHE_DIR, 'sentencepiece.model'),
                    'model_name': self.checkpoint_name,
                    'architecture': {
                        'embed_size': TRAINING_CONFIG['embed_size'],
                        'hidden_size': TRAINING_CONFIG['hidden_size'],
                        'num_layers': TRAINING_CONFIG.get('num_layers', 4),
                        'n_head': TRAINING_CONFIG.get('n_head', 4),
                        'n_positions': TRAINING_CONFIG.get('n_positions', 512),
                        'vocab_size': self.tokenizer.vocab_size,
                    },
                    'dataset_source': self.dataset_source,
                    'used_pretrained': getattr(self.config, 'pretrained', False),
                }, self.model_output_path)
                logger.info(f" Model + tokenizer saved to {self.model_output_path}")

            # Synchronize all processes after final checkpoint save
            if self.world_size > 1:
                import torch.distributed as dist
                dist.barrier()

        except TrainingStopRequested:
            logger.warning("\nTraining interrupted by request")
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise
        finally:
            # Cleanup DDP process group
            if self.world_size > 1:
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        if self.rank == 0:
                            logger.info("DDP process group destroyed")
                except Exception as e:
                    logger.warning(f"Error destroying DDP process group: {e}")

    def _tokenize_dataset_item(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            if all(isinstance(v, int) for v in value):
                return value
            return self.tokenizer.encode(" ".join(str(v) for v in value))
        if isinstance(value, dict):
            input_text = value.get('input', '').strip()
            output_text = value.get('output', '').strip()
            merged = f"{input_text} {output_text}".strip()
            return self.tokenizer.encode(merged) if merged else []
        if isinstance(value, str):
            return self.tokenizer.encode(value)
        if hasattr(value, 'tolist'):
            seq = list(value.tolist())
            if seq and all(isinstance(v, int) for v in seq):
                return seq
            return self.tokenizer.encode(" ".join(str(v) for v in seq))
        return []

    def _tokenize_batch(self, batch):
        if 'token_ids' in batch:
            return {'token_ids': batch['token_ids']}

        if 'input_ids' in batch:
            if all(isinstance(value, str) for value in batch['input_ids']):
                tokenized = self.tokenizer.batch_encode(
                    batch['input_ids'],
                    add_language_token=False,
                    remove_accents_flag=False,
                    pad=False,
                    return_tensors=False
                )
            else:
                tokenized = [self._tokenize_dataset_item(value) for value in batch['input_ids']]
            return {'token_ids': tokenized}

        if 'input' in batch and 'output' in batch:
            texts = [
                f"{inp.strip()} {out.strip()}".strip()
                for inp, out in zip(batch['input'], batch['output'])
            ]
        elif 'text' in batch:
            texts = batch['text']
        elif 'sentence' in batch:
            texts = batch['sentence']
        else:
            columns = [k for k in batch.keys() if k not in ('__index_level_0__', 'token_ids')]
            length = len(batch[next(iter(batch))]) if batch else 0
            texts = []
            for i in range(length):
                pieces = []
                for key in columns:
                    value = batch[key][i]
                    if value is None:
                        continue
                    pieces.append(str(value))
                texts.append(' '.join(pieces).strip())

        tokenized = self.tokenizer.batch_encode(
            texts,
            add_language_token=False,
            remove_accents_flag=False,
            pad=False,
            return_tensors=False
        )
        return {'token_ids': tokenized}

    def _get_tokenized_dataset(self):
        if os.path.exists(CACHE_TOKENIZED_DATASET_DIR):
            logger.info(f"Loading tokenized dataset from cache: {CACHE_TOKENIZED_DATASET_DIR}")
            tokenized_ds = Dataset.load_from_disk(CACHE_TOKENIZED_DATASET_DIR)
            logger.info(f" Loaded tokenized dataset with {len(tokenized_ds)} samples")
            return tokenized_ds

        if 'token_ids' in self.loaded_dataset.column_names:
            logger.info("Dataset already contains token_ids; skipping tokenization")
            return self.loaded_dataset

        logger.info("Tokenizing cached dataset for faster training...")

        PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'question', 'answer', 'type', 'thinking', 'source', 'has_tool_call')
        columns_to_remove = [c for c in self.loaded_dataset.column_names if c not in PRESERVED_COLUMNS]
        num_proc = self._get_num_proc()
        try:
            tokenized_ds = self.loaded_dataset.map(
                self._tokenize_batch,
                batched=True,
                batch_size=512,
                num_proc=num_proc,
                remove_columns=columns_to_remove
            )
        except Exception as e:
            logger.warning(f"Could not tokenize dataset with num_proc={num_proc}: {e}. Falling back to num_proc=1")
            tokenized_ds = self.loaded_dataset.map(
                self._tokenize_batch,
                batched=True,
                batch_size=512,
                num_proc=1,
                remove_columns=columns_to_remove
            )

        os.makedirs(CACHE_TOKENIZED_DATASET_DIR, exist_ok=True)
        tokenized_ds.save_to_disk(CACHE_TOKENIZED_DATASET_DIR)
        logger.info(f" Saved tokenized dataset to {CACHE_TOKENIZED_DATASET_DIR}")
        return tokenized_ds

