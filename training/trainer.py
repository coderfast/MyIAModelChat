import os
import pickle
import sys
import time
import math
import csv
import json
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
from commons.model.chatmodel_moe import ChatModelMoE
from commons.model.chatmodel_mtp import ChatModelMTP
from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP
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

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

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
    'weight_decay': 0.01,
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


class _RangedIterableDataset(IterableDataset):
    """Wraps a generator factory to yield only items in [offset, offset+limit)."""
    def __init__(self, base_factory, offset, limit, length=None):
        self._base_factory = base_factory
        self._offset = offset
        self._limit = limit
        self._length = length

    def __iter__(self):
        count = 0
        for i, item in enumerate(self._base_factory()):
            if i < self._offset:
                continue
            yield item
            count += 1
            if self._limit > 0 and count >= self._limit:
                break

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("_RangedIterableDataset length not set")



@dataclass
class TrainingConfig:
    """Configuration for model training (no CLI args)."""
    epochs: int = 30
    checkpoint_name: str = 'chat_model'
    dataset_source: str = 'dataset_cache'
    device_mode: str = 'auto'           # 'cpu' | 'gpu' | 'cpu+gpu' | 'auto'
    gpu_indices: Optional[List[int]] = None  # [0, 1, 2] or None=auto
    use_vulkan: bool = False
    num_cores: int = 0
    num_threads: int = 0
    max_ram_fraction: float = 0.75
    max_ram_bytes: Optional[int] = None
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
    # MTP (Multi-Token Prediction) fields
    mtp_enabled: bool = False
    mtp_num_heads: int = 4
    mtp_loss_weight: float = 0.3
    # Draft model (speculative decoding) fields
    draft_enabled: bool = False
    draft_num_layers: int = 2
    draft_embed_size: int = 128
    draft_hidden_size: int = 256
    draft_n_head: int = 2
    draft_kd_enabled: bool = False
    draft_kd_temperature: float = 2.0
    draft_kd_loss_weight: float = 0.5
    draft_kd_epochs: int = 10
    # Validation and metrics
    val_split: float = 0.1              # 10% del dataset para validacion (0 = sin validacion)
    val_batches: int = 0                # 0 = usar todo el split; >0 = limitar batches de val
    early_stopping_patience: int = 5    # 0 = deshabilitado; N = parar si val_loss no mejora en N epochs
    log_metrics_csv: bool = True        # Guardar metricas en metrics.csv
    # LR Scheduler settings
    scheduler_type: str = 'cosine'      # 'cosine' | 'step' | 'exponential' | 'plateau' | 'onecycle'
    scheduler_eta_min: float = 1e-6     # LR minimo para cosine/onecycle
    scheduler_step_size: int = 0        # Para step: cada cuantos epochs (0 = auto = epochs//3)
    scheduler_gamma: float = 0.5        # Factor de reduccion para step/exponential
    scheduler_patience: int = 5         # Para plateau: epochs sin mejora antes de reducir
    scheduler_factor: float = 0.5       # Para plateau: factor de reduccion
    # TensorBoard fields
    tensorboard_enabled: bool = False
    tensorboard_log_dir: str = 'runs'           # root log directory
    tensorboard_comment: str = ''               # optional comment suffix (e.g. 'run1')
    tensorboard_freq: int = 1                   # log every N epochs (1 = every epoch)
    # DDP fields (single-machine multi-GPU or multi-node)
    rank: int = 0                # global rank
    local_rank: int = 0          # rank within this machine
    world_size: int = 1          # total number of processes
    master_addr: str = 'localhost'
    master_port: int = 29500

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    # Mapping: nested JSON path → flat TrainingConfig field
    _JSON_TO_FIELD = {
        # training
        'training.epochs': 'epochs',
        'training.checkpoint_name': 'checkpoint_name',
        'training.dataset_source': 'dataset_source',
        # device
        'device.mode': 'device_mode',
        'device.gpu_indices': 'gpu_indices',
        'device.use_vulkan': 'use_vulkan',
        'device.num_cores': 'num_cores',
        'device.num_threads': 'num_threads',
        'device.max_ram_fraction': 'max_ram_fraction',
        'device.max_ram_bytes': 'max_ram_bytes',
        # thinking
        'thinking.enabled': 'thinking_enabled',
        'thinking.loss_weight': 'thinking_loss_weight',
        'thinking.max_tokens': 'thinking_max_tokens',
        # agent
        'agent.enabled': 'agent_enabled',
        'agent.loss_weight': 'agent_loss_weight',
        'agent.ratio': 'agent_ratio',
        # moe
        'moe.enabled': 'moe_enabled',
        'moe.num_experts': 'moe_num_experts',
        'moe.top_k': 'moe_top_k',
        'moe.load_balance_weight': 'moe_load_balance_weight',
        'moe.freeze_attention': 'moe_freeze_attention',
        # mtp
        'mtp.enabled': 'mtp_enabled',
        'mtp.num_heads': 'mtp_num_heads',
        'mtp.loss_weight': 'mtp_loss_weight',
        # draft
        'draft.enabled': 'draft_enabled',
        'draft.num_layers': 'draft_num_layers',
        'draft.embed_size': 'draft_embed_size',
        'draft.hidden_size': 'draft_hidden_size',
        'draft.n_head': 'draft_n_head',
        'draft.kd_enabled': 'draft_kd_enabled',
        'draft.kd_temperature': 'draft_kd_temperature',
        'draft.kd_loss_weight': 'draft_kd_loss_weight',
        'draft.kd_epochs': 'draft_kd_epochs',
        # validation
        'validation.split': 'val_split',
        'validation.batches': 'val_batches',
        'validation.early_stopping_patience': 'early_stopping_patience',
        # scheduler
        'scheduler.type': 'scheduler_type',
        'scheduler.eta_min': 'scheduler_eta_min',
        'scheduler.step_size': 'scheduler_step_size',
        'scheduler.gamma': 'scheduler_gamma',
        'scheduler.patience': 'scheduler_patience',
        'scheduler.factor': 'scheduler_factor',
        # tensorboard
        'tensorboard.enabled': 'tensorboard_enabled',
        'tensorboard.log_dir': 'tensorboard_log_dir',
        'tensorboard.comment': 'tensorboard_comment',
        'tensorboard.freq': 'tensorboard_freq',
        # logging
        'logging.metrics_csv': 'log_metrics_csv',
        'logging.statistics': 'statistics',
    }

    # Reverse mapping: flat field → nested JSON path
    _FIELD_TO_JSON = {v: k for k, v in _JSON_TO_FIELD.items()}

    @classmethod
    def _flatten_json(cls, data):
        """Flatten nested JSON dict to flat field dict."""
        flat = {}
        for section, values in data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    path = f"{section}.{key}"
                    if path in cls._JSON_TO_FIELD:
                        flat[cls._JSON_TO_FIELD[path]] = value
            elif section in cls._JSON_TO_FIELD:
                flat[cls._JSON_TO_FIELD[section]] = values
        return flat

    def _nest_dict(self):
        """Convert flat config dict to nested JSON structure."""
        nested = {}
        for field_name, value in self.to_dict().items():
            if field_name in self._FIELD_TO_JSON:
                path = self._FIELD_TO_JSON[field_name]
                section, key = path.split('.', 1)
                if section not in nested:
                    nested[section] = {}
                nested[section][key] = value
            else:
                # Fields not in mapping go to root (DDP fields)
                nested[field_name] = value
        return nested

    @classmethod
    def from_json(cls, json_path):
        """Load config from nested JSON file, overriding defaults."""
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        flat = cls._flatten_json(data)
        config = cls(**flat)
        return config

    def save_json(self, json_path):
        """Save config to nested JSON file."""
        import json
        nested = self._nest_dict()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(nested, f, indent=2, ensure_ascii=False)

    @classmethod
    def generate_default(cls, json_path):
        """Generate default config JSON file."""
        config = cls()
        config.save_json(json_path)
        return config

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

        # Detect next epoch number from existing checkpoints
        self._next_epoch = self._get_next_epoch_number()

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

    def _get_next_epoch_number(self):
        """Read last epoch number from training_state.json."""
        state_path = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}_state.json')
        if os.path.exists(state_path):
            try:
                import json
                with open(state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('last_epoch', 0)
            except Exception:
                pass
        return 0

    def _save_training_state(self, epoch):
        """Save last epoch number to training_state.json."""
        import json
        state_path = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}_state.json')
        try:
            data = {}
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            data['last_epoch'] = epoch
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

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
        """Detect if the dataset contains thinking or mode tokens (<|thinking|>/<|final|>)."""
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
                    if thinking_id in value:
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
                # Reduce parallelism on low-RAM systems (e.g. HF Spaces 2GB tier)
                try:
                    import psutil
                    mem_gb = psutil.virtual_memory().total / (1024**3)
                    if mem_gb < 4:
                        return 1
                except Exception:
                    pass
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

        # Install custom forward wrapper to move activations between CPU/GPU
        self._install_cpu_gpu_forward_wrapper(model)

        return model

    def _install_cpu_gpu_forward_wrapper(self, model):
        """Wrap transformer forward to move activations at CPU/GPU boundary."""
        split_index = self._gpu_layers_count
        gpu_device = self._gpu_device

        transformer = model.model.transformer
        original_transformer_forward = transformer.forward

        def cpu_gpu_forward(*args, **kwargs):
            # For GPT-2, call the original forward but intercept hidden states
            # at the split boundary by hooking into the layer list
            from transformers.modeling_outputs import BaseModelOutputWithPast

            # Replicate the GPT-2 forward logic with device transfers
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            attention_mask = kwargs.get('attention_mask', None)
            position_ids = kwargs.get('position_ids', None)

            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

                hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)

                for i, layer in enumerate(transformer.h):
                    # Transfer to CPU when crossing from GPU to CPU boundary
                    if i == split_index and hidden_states.device.type != 'cpu':
                        hidden_states = hidden_states.to('cpu')
                    layer_output = layer(hidden_states)
                    hidden_states = layer_output[0]

                hidden_states = transformer.ln_f(hidden_states)

                # Transfer back to GPU for lm_head
                if hidden_states.device.type != gpu_device.type:
                    hidden_states = hidden_states.to(gpu_device)

                return BaseModelOutputWithPast(last_hidden_state=hidden_states)

            # Fallback to original forward
            return original_transformer_forward(*args, **kwargs)

        transformer.forward = cpu_gpu_forward
        logger.info(f"CPU/GPU forward wrapper installed (split at layer {split_index})")

    def _compute_loss(self, model, inputs, targets, criterion):
        """Unified loss computation with thinking-aware, mode-aware, and agentic loss weighting.

        Loss rules by sample type:
        - CONTEXT samples (<|problem|> prefix, no <|thinking|>): tokens before <|final|> get
          weight 0.0, <|final|> and answer tokens get weight 1.0.
        - THINKING samples (contain <|thinking|>): tokens before <|thinking|> get weight 0.0,
          <|thinking|>...<|final|> tokens get thinking_loss_weight, <|final|> and answer
          tokens get weight 1.0.
        - AGENT samples (with <tool_call>): thinking gets thinking_loss_weight, tool_call and
          observation tokens get full weight (1.0), answer tokens get full weight (1.0).
        - MoE models: adds load balancing loss to encourage uniform expert usage.
        """
        # Handle MoE model output (returns logits + gate_scores)
        gate_scores = None
        mtp_logits_list = None
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            if len(outputs) == 2 and isinstance(outputs[1], list):
                # Check if this is MoE+MTP (list of MTP logits) or pure MoE (list of gate_scores)
                # MoE+MTP: model has both get_expert_utilization and mtp_heads
                if hasattr(model, 'get_expert_utilization') and hasattr(model, 'mtp_heads'):
                    # MoE+MTP: extract gate_scores from model._all_gate_scores
                    outputs, mtp_logits_list = outputs
                    gate_scores = model._all_gate_scores if model._all_gate_scores else None
                else:
                    # Pure MTP model: (primary_logits, [mtp_head_logits...])
                    outputs, mtp_logits_list = outputs
            elif len(outputs) == 2:
                # MoE model: (logits, gate_scores)
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
        end_id = getattr(self.tokenizer, 'get_end_index', lambda: -1)()
        assistant_id = getattr(self.tokenizer, 'get_assistant_index', lambda: -1)()

        agent_enabled = getattr(self.config, 'agent_enabled', False)
        agent_loss_weight = getattr(self.config, 'agent_loss_weight', 1.0)

        batch_size, seq_len = targets.shape

        for b in range(batch_size):
            has_context_prefix = False
            has_thinking_prefix = False

            # Detect thinking by presence of <|thinking|> token anywhere in the row
            # (Formato 3 thinking samples start with <|problem|>, not <|thinking|>)
            if thinking_id >= 0 and thinking_id in targets[b]:
                has_thinking_prefix = True
            elif context_id >= 0:
                # Detect mode from prefix token (only when no thinking present)
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

            # Agentic masking: override weights for tool_call and observation tokens.
            # Observation is a prefix-only marker (<|tool_result|>) that runs until
            # the next turn marker (<|end|> or <|assistant|>).
            if agent_enabled and tool_call_id >= 0:
                in_tool_call = False
                in_observation = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == tool_call_id:
                        in_tool_call = True
                        in_observation = False
                        weights[b, s] = agent_loss_weight  # tool_call gets full weight
                    elif token == tool_call_end_id:
                        in_tool_call = False
                        in_observation = False
                        weights[b, s] = agent_loss_weight
                    elif token == observation_id:
                        in_tool_call = False
                        in_observation = True
                        weights[b, s] = agent_loss_weight
                    elif end_id >= 0 and token == end_id:
                        in_tool_call = False
                        in_observation = False
                        weights[b, s] = agent_loss_weight  # <|end|> turn delimiter
                    elif assistant_id >= 0 and token == assistant_id:
                        in_tool_call = False
                        in_observation = False
                        weights[b, s] = agent_loss_weight  # <|assistant|> new turn
                    elif in_tool_call:
                        weights[b, s] = agent_loss_weight  # JSON inside tool_call
                    elif in_observation:
                        weights[b, s] = agent_loss_weight  # observation content

        # Save original 2D targets for MTP loss computation (before flattening)
        targets_2d = targets.clone() if mtp_logits_list and getattr(self.config, 'mtp_enabled', False) else None

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
            return torch.tensor(0.0, device=outputs.device, requires_grad=True), raw_outputs, torch.tensor(0.0, device=outputs.device)
        loss = (token_losses * weights).sum() / weight_sum
        
        # Add load balancing loss for MoE models
        moe_loss = torch.tensor(0.0, device=loss.device)
        if gate_scores is not None and hasattr(model, 'get_load_balancing_loss'):
            moe_loss = model.get_load_balancing_loss(torch.stack(gate_scores))
            loss = loss + self.config.moe_load_balance_weight * moe_loss

        # Add MTP auxiliary loss
        mtp_loss = torch.tensor(0.0, device=loss.device)
        if mtp_logits_list and getattr(self.config, 'mtp_enabled', False) and targets_2d is not None:
            mtp_total = torch.tensor(0.0, device=loss.device)
            mtp_head_count = 0
            for k, head_logits in enumerate(mtp_logits_list):
                shift = k + 2  # head 0 predicts t+2, head 1 predicts t+3, ...
                if targets_2d.size(1) > shift:
                    mtp_preds = head_logits[:, :-shift].contiguous().view(-1, head_logits.size(-1))
                    mtp_targets = targets_2d[:, shift:].contiguous().view(-1)
                    non_pad = mtp_targets.ne(self.tokenizer.get_pad_index())
                    if non_pad.any():
                        mtp_token_losses = criterion(mtp_preds[non_pad], mtp_targets[non_pad])
                        mtp_loss_k = mtp_token_losses.mean()
                        mtp_total = mtp_total + mtp_loss_k
                        mtp_head_count += 1
            if mtp_head_count > 0:
                mtp_loss = mtp_total / mtp_head_count
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        return loss, raw_outputs, mtp_loss.item()

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
        mtp_metrics_accum = {}
        last_loss = 0.0
        total_tokens = 0
        last_grad_norm = 0.0

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
                    loss, logits, mtp_loss_val = self._compute_loss(model, inputs, targets, criterion)
            else:
                loss, logits, mtp_loss_val = self._compute_loss(model, inputs, targets, criterion)

            last_loss = loss.item()
            total_tokens += inputs.size(0) * inputs.size(1)

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

            # Accumulate MTP loss
            if mtp_loss_val > 0:
                if 'mtp_loss' not in mtp_metrics_accum:
                    mtp_metrics_accum['mtp_loss'] = []
                mtp_metrics_accum['mtp_loss'].append(mtp_loss_val)

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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG['grad_clip_norm'])
                last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

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

        # Compute average thinking metrics for this epoch
        thinking_epoch = {}
        for key, values in thinking_metrics_accum.items():
            thinking_epoch[key] = sum(values) / len(values) if values else 0.0

        # Compute average agent metrics for this epoch
        agent_epoch = {}
        for key, values in agent_metrics_accum.items():
            agent_epoch[key] = sum(values) / len(values) if values else 0.0

        # Compute average MTP metrics for this epoch
        mtp_epoch = {}
        for key, values in mtp_metrics_accum.items():
            mtp_epoch[key] = sum(values) / len(values) if values else 0.0

        # Collect MoE metrics BEFORE validation (which clears _all_gate_scores)
        moe_epoch = {}
        if hasattr(self.config, 'moe_enabled') and self.config.moe_enabled:
            try:
                if hasattr(model, 'get_expert_utilization'):
                    utilization = model.get_expert_utilization()
                    for expert_id, util in utilization.items():
                        moe_epoch[f'moe_expert_{expert_id}_util'] = util
                if hasattr(model, '_all_gate_scores') and model._all_gate_scores:
                    import math as _math
                    all_scores = torch.stack([s.detach() if s.requires_grad else s for s in model._all_gate_scores])
                    probs = all_scores.mean(dim=[0, 1, 2])
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    moe_epoch['moe_gate_entropy'] = entropy
                    max_entropy = _math.log(getattr(self.config, 'moe_num_experts', 4))
                    moe_epoch['moe_gate_entropy_norm'] = entropy / max_entropy if max_entropy > 0 else 0
            except Exception:
                pass

        return avg_loss, total_tokens, last_grad_norm, thinking_epoch, agent_epoch, mtp_epoch, moe_epoch

    def validate(self, model, dataloader, criterion, device, max_batches=0):
        """Run validation loop. Returns (avg_loss, perplexity, num_batches)."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                loss, _, _ = self._compute_loss(model, inputs, targets, criterion)
                total_loss += loss.item()
                total_batches += 1

        model.train()
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
        return avg_loss, perplexity, total_batches


    def _log_metrics_csv(self, metrics_row, csv_path):
        """Append a row of metrics to CSV file. Creates header if file doesn't exist.
        Correlates epoch numbers when retraining (continues from last session)."""
        file_exists = os.path.exists(csv_path)
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'train_perplexity', 'val_perplexity',
            'gap', 'lr', 'grad_norm', 'tokens_per_sec', 'best_loss', 'early_stop_patience',
            # Thinking metrics
            'thinking_accuracy', 'thinking_open_acc', 'thinking_close_acc',
            'thinking_coverage', 'response_accuracy',
            # Agent metrics
            'agent_tool_call_acc', 'agent_observation_acc', 'agent_ratio',
            # MoE metrics
            'moe_gate_entropy_norm',
            # MTP metrics
            'mtp_loss',
        ]
        # Add per-expert utilization columns dynamically
        for key in sorted(metrics_row.keys()):
            if key.startswith('moe_expert_') and key not in fieldnames:
                fieldnames.append(key)

        # If file exists, migrate missing columns and read last epoch number
        epoch_offset = 0
        if file_exists:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_fields = reader.fieldnames or []
                    rows = list(reader)
                    last_epoch = 0
                    for row in rows:
                        try:
                            last_epoch = int(row.get('epoch', 0))
                        except (ValueError, TypeError):
                            pass
                    epoch_offset = last_epoch

                # Migrate: add missing columns to existing CSV
                missing = [col for col in fieldnames if col not in existing_fields]
                if missing:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in rows:
                            # DictWriter fills missing keys with None (empty string via restval)
                            writer.writerow(row)
            except Exception:
                epoch_offset = 0

        # Epoch number is already correlated (current_epoch_num = _next_epoch + epoch + 1)
        # No additional offset needed - just write the row as-is

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)

    def _generate_training_report(self, csv_path):
        """Generate an HTML report with training graphs from CSV metrics."""
        if not os.path.exists(csv_path):
            return

        # Read CSV data
        epochs = []
        train_losses = []
        val_losses = []
        train_perplexities = []
        val_perplexities = []
        gaps = []
        lrs = []
        grad_norms = []
        tokens_per_secs = []
        # Thinking metrics
        thinking_accuracies = []
        thinking_open_accs = []
        thinking_close_accs = []
        thinking_coverages = []
        response_accuracies = []
        # Agent metrics
        agent_tool_call_accs = []
        agent_observation_accs = []
        agent_ratios = []
        # MoE metrics
        moe_gate_entropy_norms = []
        moe_expert_utils = {}  # expert_id -> list of utilization values
        # MTP metrics
        mtp_losses = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        epochs.append(int(row.get('epoch', 0)))
                        train_losses.append(float(row.get('train_loss', 0)))
                        val_loss = row.get('val_loss', '').strip()
                        val_losses.append(float(val_loss) if val_loss else None)
                        train_perplexities.append(float(row.get('train_perplexity', 0)))
                        val_perp = row.get('val_perplexity', '').strip()
                        val_perplexities.append(float(val_perp) if val_perp else None)
                        gap = row.get('gap', '').strip()
                        gaps.append(float(gap) if gap else None)
                        lrs.append(float(row.get('lr', 0)))
                        grad_norms.append(float(row.get('grad_norm', 0)))
                        tokens_per_secs.append(float(row.get('tokens_per_sec', 0)))
                        # Thinking metrics
                        ta = row.get('thinking_accuracy', '').strip()
                        thinking_accuracies.append(float(ta) if ta else None)
                        to = row.get('thinking_open_acc', '').strip()
                        thinking_open_accs.append(float(to) if to else None)
                        tc = row.get('thinking_close_acc', '').strip()
                        thinking_close_accs.append(float(tc) if tc else None)
                        tv = row.get('thinking_coverage', '').strip()
                        thinking_coverages.append(float(tv) if tv else None)
                        ra = row.get('response_accuracy', '').strip()
                        response_accuracies.append(float(ra) if ra else None)
                        # Agent metrics
                        at = row.get('agent_tool_call_acc', '').strip()
                        agent_tool_call_accs.append(float(at) if at else None)
                        ao = row.get('agent_observation_acc', '').strip()
                        agent_observation_accs.append(float(ao) if ao else None)
                        ar = row.get('agent_ratio', '').strip()
                        agent_ratios.append(float(ar) if ar else None)
                        # MoE metrics
                        me = row.get('moe_gate_entropy_norm', '').strip()
                        moe_gate_entropy_norms.append(float(me) if me else None)
                        # Per-expert utilization
                        for key, value in row.items():
                            if key.startswith('moe_expert_') and key.endswith('_util'):
                                expert_id = key.replace('moe_expert_', '').replace('_util', '')
                                if expert_id not in moe_expert_utils:
                                    moe_expert_utils[expert_id] = []
                                ev = value.strip()
                                moe_expert_utils[expert_id].append(float(ev) if ev else None)
                        # MTP metrics
                        mt = row.get('mtp_loss', '').strip()
                        mtp_losses.append(float(mt) if mt else None)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            return

        if not epochs:
            return

        # Determine training status
        has_val = any(v is not None for v in val_losses)
        if has_val and len(val_losses) >= 2:
            first_val = next(v for v in val_losses if v is not None)
            last_val = next(v for v in reversed(val_losses) if v is not None)
            val_change_pct = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
            if last_val < first_val * 0.9:
                status = "healthy"
                status_text = f"TRAINING IS GOING WELL - Val loss decreased {abs(val_change_pct):.1f}%"
                status_detail = "The model is learning and generalizing to new data."
            elif last_val > first_val * 1.1:
                status = "overfitting"
                status_text = f"WARNING: OVERFITTING DETECTED - Val loss increased {val_change_pct:.1f}%"
                status_detail = "The model is memorizing training data but failing on new data. Reduce --val-split or add more data."
            else:
                status = "stable"
                status_text = f"TRAINING IS STABLE - Val loss changed {val_change_pct:+.1f}%"
                status_detail = "Loss is plateauing. Try more epochs or adjust learning rate."
        else:
            first_loss = train_losses[0] if train_losses else 0
            last_loss = train_losses[-1] if train_losses else 0
            loss_change_pct = ((last_loss - first_loss) / first_loss) * 100 if first_loss > 0 else 0
            if last_loss < first_loss * 0.9:
                status = "healthy"
                status_text = f"TRAINING IS GOING WELL - Loss decreased {abs(loss_change_pct):.1f}%"
                status_detail = "The model is learning. Add --val-split 0.1 to monitor generalization."
            elif last_loss > first_loss * 1.1:
                status = "overfitting"
                status_text = f"WARNING: LOSS INCREASING - Loss increased {loss_change_pct:.1f}%"
                status_detail = "Training is diverging. Reduce learning rate or check data quality."
            else:
                status = "stable"
                status_text = f"TRAINING IS STABLE - Loss changed {loss_change_pct:+.1f}%"
                status_detail = "Loss plateau. Try more epochs or adjust learning rate."

        # Thinking status
        has_thinking = any(v is not None for v in thinking_accuracies)
        if has_thinking and len(thinking_accuracies) >= 2:
            first_ta = next(v for v in thinking_accuracies if v is not None)
            last_ta = next(v for v in reversed(thinking_accuracies) if v is not None)
            last_tc = next((v for v in reversed(thinking_close_accs) if v is not None), 0)
            if last_ta > 0.8 and last_tc > 0.7:
                thinking_status = "healthy"
                thinking_status_text = f"THINKING IS LEARNING - Accuracy: {last_ta*100:.0f}% (close: {last_tc*100:.0f}%)"
                thinking_status_detail = "Model is learning to generate thinking blocks correctly."
            elif last_ta < 0.5 or last_tc < 0.3:
                thinking_status = "overfitting"
                thinking_status_text = f"THINKING NEEDS ATTENTION - Accuracy: {last_ta*100:.0f}% (close: {last_tc*100:.0f}%)"
                thinking_status_detail = "Model struggles with thinking delimiters. Check thinking data quality."
            else:
                thinking_status = "stable"
                thinking_status_text = f"THINKING IS STABLE - Accuracy: {last_ta*100:.0f}%"
                thinking_status_detail = "Thinking accuracy is moderate. More epochs may help."
        else:
            thinking_status = None
            thinking_status_text = None
            thinking_status_detail = None

        # Agent status
        has_agent = any(v is not None for v in agent_tool_call_accs)
        if has_agent and len(agent_tool_call_accs) >= 2:
            first_at = next(v for v in agent_tool_call_accs if v is not None)
            last_at = next(v for v in reversed(agent_tool_call_accs) if v is not None)
            last_ao = next((v for v in reversed(agent_observation_accs) if v is not None), 0)
            if last_at > 0.7 and last_ao > 0.6:
                agent_status = "healthy"
                agent_status_text = f"AGENT IS LEARNING - Tool Call: {last_at*100:.0f}% | Observation: {last_ao*100:.0f}%"
                agent_status_detail = "Model is learning tool calls and observations correctly."
            elif last_at < 0.4 or last_ao < 0.3:
                agent_status = "overfitting"
                agent_status_text = f"AGENT NEEDS ATTENTION - Tool Call: {last_at*100:.0f}% | Observation: {last_ao*100:.0f}%"
                agent_status_detail = "Model struggles with agentic tokens. Check agent data quality."
            else:
                agent_status = "stable"
                agent_status_text = f"AGENT IS STABLE - Tool Call: {last_at*100:.0f}%"
                agent_status_detail = "Agent accuracy is moderate. More epochs may help."
        else:
            agent_status = None
            agent_status_text = None
            agent_status_detail = None

        # MoE status
        has_moe = any(v is not None for v in moe_gate_entropy_norms)
        if has_moe and len(moe_gate_entropy_norms) >= 2:
            last_entropy = next(v for v in reversed(moe_gate_entropy_norms) if v is not None)
            # Check if experts are balanced (entropy > 0.7 means good balance)
            if last_entropy > 0.7:
                moe_status = "healthy"
                moe_status_text = f"MoE IS BALANCED - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "Experts are being used evenly. Load balancing is working."
            elif last_entropy < 0.4:
                moe_status = "overfitting"
                moe_status_text = f"MoE EXPERT COLLAPSE - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "One or more experts dominate. Increase load_balance_weight."
            else:
                moe_status = "stable"
                moe_status_text = f"MoE IS MODERATE - Gate entropy: {last_entropy*100:.0f}% of max"
                moe_status_detail = "Expert distribution is moderate. Monitor for collapse."
        else:
            moe_status = None
            moe_status_text = None
            moe_status_detail = None

        # MTP status
        has_mtp = any(v is not None for v in mtp_losses)
        if has_mtp and len(mtp_losses) >= 2:
            first_mtp = next(v for v in mtp_losses if v is not None)
            last_mtp = next(v for v in reversed(mtp_losses) if v is not None)
            mtp_change_pct = ((last_mtp - first_mtp) / first_mtp) * 100 if first_mtp > 0 else 0
            if last_mtp < first_mtp * 0.9:
                mtp_status = "healthy"
                mtp_status_text = f"MTP IS LEARNING - Loss decreased {abs(mtp_change_pct):.1f}%"
                mtp_status_detail = "Multi-Token Prediction heads are learning future token patterns."
            elif last_mtp > first_mtp * 1.1:
                mtp_status = "overfitting"
                mtp_status_text = f"MTP LOSS INCREASING - Loss increased {mtp_change_pct:.1f}%"
                mtp_status_detail = "MTP heads may be overfitting. Reduce mtp_loss_weight or add more data."
            else:
                mtp_status = "stable"
                mtp_status_text = f"MTP IS STABLE - Loss changed {mtp_change_pct:+.1f}%"
                mtp_status_detail = "MTP loss is plateauing. More epochs may help."
        else:
            mtp_status = None
            mtp_status_text = None
            mtp_status_detail = None

        # Precompute display values for HTML template (avoid f-string ternary issues)
        final_train_loss_str = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
        final_val_loss_str = f"{next(v for v in reversed(val_losses) if v is not None):.4f}" if has_val else "N/A"
        final_perplexity_str = f"{train_perplexities[-1]:.2f}" if train_perplexities else "N/A"
        final_tokens_str = f"{tokens_per_secs[-1]:.0f}" if tokens_per_secs else "N/A"
        total_epochs_str = str(epochs[-1]) if epochs else "0"

        # Serialize data for JavaScript (Python None → null, proper JS syntax)
        epochs_json = json.dumps(epochs)
        train_losses_json = json.dumps(train_losses)
        val_losses_json = json.dumps(val_losses)
        train_perplexities_json = json.dumps(train_perplexities)
        val_perplexities_json = json.dumps(val_perplexities)
        gaps_json = json.dumps(gaps)
        lrs_json = json.dumps(lrs)
        tokens_per_secs_json = json.dumps(tokens_per_secs)
        grad_norms_json = json.dumps(grad_norms)
        thinking_accuracies_json = json.dumps(thinking_accuracies)
        thinking_open_accs_json = json.dumps(thinking_open_accs)
        thinking_close_accs_json = json.dumps(thinking_close_accs)
        thinking_coverages_json = json.dumps(thinking_coverages)
        response_accuracies_json = json.dumps(response_accuracies)
        agent_tool_call_accs_json = json.dumps(agent_tool_call_accs)
        agent_observation_accs_json = json.dumps(agent_observation_accs)
        agent_ratios_json = json.dumps(agent_ratios)
        moe_gate_entropy_norms_json = json.dumps(moe_gate_entropy_norms)
        moe_expert_utils_json = json.dumps(moe_expert_utils)
        mtp_losses_json = json.dumps(mtp_losses)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {self.checkpoint_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .status {{ padding: 15px 20px; border-radius: 8px; margin: 15px 0; font-weight: bold; border-left: 5px solid; }}
        .status.healthy {{ background: #d4edda; color: #155724; border-color: #28a745; }}
        .status.overfitting {{ background: #f8d7da; color: #721c24; border-color: #dc3545; }}
        .status.stable {{ background: #fff3cd; color: #856404; border-color: #ffc107; }}
        .status-main {{ font-size: 18px; margin-bottom: 5px; }}
        .status-detail {{ font-size: 14px; font-weight: normal; opacity: 0.9; }}
        .status-banner {{ display: flex; align-items: center; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
        .status-banner.healthy {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; }}
        .status-banner.overfitting {{ background: linear-gradient(135deg, #dc3545, #e83e8c); color: white; }}
        .status-banner.stable {{ background: linear-gradient(135deg, #ffc107, #fd7e14); color: white; }}
        .banner-icon {{ font-size: 36px; margin-right: 20px; opacity: 0.9; }}
        .banner-content {{ flex: 1; }}
        .banner-title {{ font-size: 20px; font-weight: bold; margin-bottom: 5px; }}
        .banner-detail {{ font-size: 14px; opacity: 0.9; }}
        .section-divider {{ font-size: 20px; font-weight: bold; color: #333; margin: 30px 0 15px 0; padding: 10px 0; border-bottom: 3px solid #3498db; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
        .zoom-controls {{ position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; z-index: 10; }}
        .zoom-btn {{ padding: 4px 10px; font-size: 14px; font-weight: bold; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; opacity: 0.8; min-width: 28px; text-align: center; }}
        .zoom-btn:hover {{ opacity: 1; background: #2980b9; }}
        .zoom-btn.reset {{ background: #95a5a6; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        canvas {{ max-height: 300px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }}
        .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div id="google_translate_element" style="position:fixed;top:10px;right:10px;z-index:9999;background:white;padding:5px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;"></div>
    <script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <script>
    function googleTranslateElementInit() {{
        new google.translate.TranslateElement({{pageLanguage: 'en', includedLanguages: 'es,fr,de,it,pt,ru,ja,ko,zh-CN,ar,hi,th,vi,nl,pl,sv,da,no,fi,tr,uk,cs,ro,hu,el,bg,hr,sk,sl,lt,lv,et,mt,ga,cy,eu,ca,gl,af,sq,bs,is,lb,mk,sr,be,kk,ky,tg,uz,tk,ka,hy,az', autoDisplay: false}}, 'google_translate_element');
    }}
    </script>
    <style>
        .skiptranslate {{ display: inline !important; }}
        .goog-te-gadget {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; }}
        .goog-te-gadget-simple {{ border: 1px solid #ddd !important; border-radius: 6px !important; padding: 2px 8px !important; background: #f8f8f8 !important; }}
        .goog-te-gadget-simple:hover {{ background: #e8e8e8 !important; }}
        .goog-te-combo {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; border: none !important; background: transparent !important; cursor: pointer !important; }}
        body {{ top: 0 !important; }}
    </style>
    <div class="container">
        <div class="header">
            <h1>Training Report: {self.checkpoint_name}</h1>
            <div class="status {status}">
                <div class="status-main">{status_text}</div>
                <div class="status-detail">{status_detail}</div>
            </div>
            <p>Total epochs: {total_epochs_str} | Final train loss: {final_train_loss_str} | Final val loss: {final_val_loss_str}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{final_train_loss_str}</div>
                <div class="metric-label">Final Train Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_val_loss_str}</div>
                <div class="metric-label">Final Val Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_perplexity_str}</div>
                <div class="metric-label">Final Perplexity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_tokens_str}</div>
                <div class="metric-label">Tokens/sec</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Loss Over Time</h2>
            <canvas id="lossChart"></canvas>
            <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lossChart')" title="Reset">&#8634;</button></div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h2>Perplexity</h2>
                <canvas id="perplexityChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('perplexityChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('perplexityChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('perplexityChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Train/Val Gap</h2>
                <canvas id="gapChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('gapChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('gapChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('gapChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h2>Learning Rate</h2>
                <canvas id="lrChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lrChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lrChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lrChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Training Speed (tokens/s)</h2>
                <canvas id="speedChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('speedChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('speedChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('speedChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        {f'''
        <div class="section-divider">THINKING TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Thinking Accuracy</h2>
                <canvas id="thinkingAccuracyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('thinkingAccuracyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('thinkingAccuracyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('thinkingAccuracyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Thinking Coverage</h2>
                <canvas id="thinkingCoverageChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('thinkingCoverageChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('thinkingCoverageChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('thinkingCoverageChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {thinking_status}">
            <div class="banner-icon">{'OK' if thinking_status == 'healthy' else 'WARNING' if thinking_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{thinking_status_text}</div>
                <div class="banner-detail">{thinking_status_detail}</div>
            </div>
        </div>
        ''' if has_thinking else ''}

        {f'''
        <div class="section-divider">AGENT TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Agent Accuracy</h2>
                <canvas id="agentAccuracyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('agentAccuracyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('agentAccuracyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('agentAccuracyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Agent Token Ratio</h2>
                <canvas id="agentRatioChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('agentRatioChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('agentRatioChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('agentRatioChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {agent_status}">
            <div class="banner-icon">{'OK' if agent_status == 'healthy' else 'WARNING' if agent_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{agent_status_text}</div>
                <div class="banner-detail">{agent_status_detail}</div>
            </div>
        </div>
        ''' if has_agent else ''}

        {f'''
        <div class="section-divider">MoE TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Gate Entropy (normalized)</h2>
                <canvas id="moeEntropyChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('moeEntropyChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('moeEntropyChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('moeEntropyChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <h2>Expert Utilization</h2>
                <canvas id="moeUtilChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('moeUtilChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('moeUtilChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('moeUtilChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {moe_status}">
            <div class="banner-icon">{'OK' if moe_status == 'healthy' else 'WARNING' if moe_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{moe_status_text}</div>
                <div class="banner-detail">{moe_status_detail}</div>
            </div>
        </div>
        ''' if has_moe else ''}

        {f'''
        <div class="section-divider">MTP TRAINING PROGRESS</div>
        <div class="charts-grid">
            <div class="chart-container">
                <h2>MTP Loss</h2>
                <canvas id="mtpLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('mtpLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('mtpLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('mtpLossChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
        <div class="status-banner {mtp_status}">
            <div class="banner-icon">{'OK' if mtp_status == 'healthy' else 'WARNING' if mtp_status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{mtp_status_text}</div>
                <div class="banner-detail">{mtp_status_detail}</div>
            </div>
        </div>
        ''' if has_mtp else ''}

        <div class="status-banner {status}">
            <div class="banner-icon">{'OK' if status == 'healthy' else 'WARNING' if status == 'overfitting' else 'INFORMATION'}</div>
            <div class="banner-content">
                <div class="banner-title">{status_text}</div>
                <div class="banner-detail">{status_detail}</div>
            </div>
        </div>
    </div>

    <script>
        const epochs = {epochs_json};
        const trainLoss = {train_losses_json};
        const valLoss = {val_losses_json};
        const trainPerplexity = {train_perplexities_json};
        const valPerplexity = {val_perplexities_json};
        const gaps = {gaps_json};
        const lrs = {lrs_json};
        const tokensPerSec = {tokens_per_secs_json};
        const gradNorms = {grad_norms_json};
        const thinkingAccuracy = {thinking_accuracies_json};
        const thinkingOpenAcc = {thinking_open_accs_json};
        const thinkingCloseAcc = {thinking_close_accs_json};
        const thinkingCoverage = {thinking_coverages_json};
        const responseAccuracy = {response_accuracies_json};
        const agentToolCallAcc = {agent_tool_call_accs_json};
        const agentObsAcc = {agent_observation_accs_json};
        const agentRatio = {agent_ratios_json};
        const moeEntropyNorm = {moe_gate_entropy_norms_json};
        const moeExpertUtils = {moe_expert_utils_json};
        const mtpLosses = {mtp_losses_json};

        // Reusable zoom/pan configuration
        const zoomOptions = {{
            zoom: {{
                wheel: {{ enabled: true }},
                pinch: {{ enabled: true }},
                drag: {{ enabled: false }},
                mode: 'xy',
                scaleMode: 'xy'
            }},
            pan: {{
                enabled: true,
                mode: 'xy'
            }}
        }};

        function resetZoom(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.resetZoom();
        }}
        function zoomIn(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.zoom(1.2);
        }}
        function zoomOut(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.zoom(0.8);
        }}

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Train Loss',
                        data: trainLoss,
                        borderColor: '#3498db',
                        tension: 0.3,
                        fill: false
                    }},
                    {{
                        label: 'Val Loss',
                        data: valLoss,
                        borderColor: '#e74c3c',
                        tension: 0.3,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true, plugins: {{ zoom: zoomOptions }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Loss' }} }}
                }}
            }}
        }});

        // Perplexity Chart
        new Chart(document.getElementById('perplexityChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Train Perplexity',
                        data: trainPerplexity,
                        borderColor: '#2ecc71',
                        tension: 0.3,
                        fill: false
                    }},
                    {{
                        label: 'Val Perplexity',
                        data: valPerplexity,
                        borderColor: '#e67e22',
                        tension: 0.3,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true, plugins: {{ zoom: zoomOptions }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Perplexity' }} }}
                }}
            }}
        }});

        // Gap Chart
        new Chart(document.getElementById('gapChart'), {{
            type: 'bar',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Val - Train Loss',
                    data: gaps,
                    backgroundColor: gaps.map(v => v === null ? 'transparent' : v > 0 ? 'rgba(231, 76, 60, 0.6)' : 'rgba(46, 204, 113, 0.6)'),
                    borderColor: gaps.map(v => v === null ? 'transparent' : v > 0 ? '#e74c3c' : '#2ecc71'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true, plugins: {{ zoom: zoomOptions }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Gap' }} }}
                }},
                plugins: {{
                    annotation: {{
                        annotations: {{
                            zeroLine: {{
                                type: 'line',
                                yMin: 0,
                                yMax: 0,
                                borderColor: '#999',
                                borderDash: [5, 5]
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Learning Rate Chart
        new Chart(document.getElementById('lrChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Learning Rate',
                    data: lrs,
                    borderColor: '#9b59b6',
                    tension: 0.3,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true, plugins: {{ zoom: zoomOptions }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Learning Rate' }} }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return 'LR: ' + ctx.parsed.y.toFixed(10);
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Speed Chart
        new Chart(document.getElementById('speedChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Tokens/sec',
                    data: tokensPerSec,
                    borderColor: '#1abc9c',
                    tension: 0.3,
                    fill: true,
                    backgroundColor: 'rgba(26, 188, 156, 0.2)'
                }}]
            }},
            options: {{
                responsive: true, plugins: {{ zoom: zoomOptions }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Tokens/sec' }} }}
                }}
            }}
        }});

        // Thinking Accuracy Chart
        if (thinkingAccuracy.some(v => v !== null)) {{
            new Chart(document.getElementById('thinkingAccuracyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [
                        {{ label: 'Overall', data: thinkingAccuracy, borderColor: '#3498db', tension: 0.3, fill: false }},
                        {{ label: 'Open <thinking>', data: thinkingOpenAcc, borderColor: '#2ecc71', tension: 0.3, fill: false }},
                        {{ label: 'Close </thinking>', data: thinkingCloseAcc, borderColor: '#e74c3c', tension: 0.3, fill: false }},
                        {{ label: 'Response', data: responseAccuracy, borderColor: '#9b59b6', tension: 0.3, fill: false }}
                    ]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
        }}

        // Thinking Coverage Chart
        if (thinkingCoverage.some(v => v !== null)) {{
            new Chart(document.getElementById('thinkingCoverageChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Thinking Coverage',
                        data: thinkingCoverage,
                        borderColor: '#f39c12',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(243, 156, 18, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Coverage' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
        }}

        // Agent Accuracy Chart
        if (agentToolCallAcc.some(v => v !== null)) {{
            new Chart(document.getElementById('agentAccuracyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [
                        {{ label: 'Tool Call', data: agentToolCallAcc, borderColor: '#e67e22', tension: 0.3, fill: false }},
                        {{ label: 'Observation', data: agentObsAcc, borderColor: '#1abc9c', tension: 0.3, fill: false }}
                    ]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
        }}

        // Agent Ratio Chart
        if (agentRatio.some(v => v !== null)) {{
            new Chart(document.getElementById('agentRatioChart'), {{
                type: 'bar',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Agent Token Ratio',
                        data: agentRatio,
                        backgroundColor: 'rgba(230, 126, 34, 0.6)',
                        borderColor: '#e67e22',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Ratio' }}, min: 0 }}
                    }}
                }}
            }});
        }}

        // MoE Entropy Chart
        if (moeEntropyNorm.some(v => v !== null)) {{
            new Chart(document.getElementById('moeEntropyChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Normalized Entropy',
                        data: moeEntropyNorm,
                        borderColor: '#8e44ad',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(142, 68, 173, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'Entropy (0=collapsed, 1=balanced)' }}, min: 0, max: 1 }}
                    }}
                }}
            }});
        }}

        // MoE Expert Utilization Chart
        if (Object.keys(moeExpertUtils).length > 0) {{
            const expertColors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
            const datasets = Object.keys(moeExpertUtils).map((key, i) => ({{
                label: 'Expert ' + key,
                data: moeExpertUtils[key],
                backgroundColor: expertColors[i % expertColors.length] + '99',
                borderColor: expertColors[i % expertColors.length],
                borderWidth: 1
            }}));
            new Chart(document.getElementById('moeUtilChart'), {{
                type: 'bar',
                data: {{ labels: epochs, datasets: datasets }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ stacked: true, title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ stacked: true, title: {{ display: true, text: 'Utilization' }}, min: 0 }}
                    }}
                }}
            }});
        }}

        // MTP Loss Chart
        if (mtpLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('mtpLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'MTP Loss',
                        data: mtpLosses,
                        borderColor: '#e74c3c',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(231, 76, 60, 0.2)'
                    }}]
                }},
                options: {{
                    responsive: true, plugins: {{ zoom: zoomOptions }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Epoch' }} }},
                        y: {{ title: {{ display: true, text: 'MTP Loss' }} }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""

        # Write HTML file
        html_path = csv_path.replace('.csv', '_report.html')
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if self.rank == 0:
                logger.info(f"  Training report saved to {html_path}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"  Could not generate training report: {e}")


    def _log_draft_csv(self, metrics_row, csv_path):
        """Append a row of draft model metrics to CSV file."""
        file_exists = os.path.exists(csv_path)
        fieldnames = [
            'epoch', 'loss', 'lr', 'kd_loss', 'hard_loss', 'tokens_per_sec',
        ]
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_row)

    def _generate_draft_report(self, csv_path):
        """Generate an HTML report for draft model training."""
        if not os.path.exists(csv_path):
            return

        epochs = []
        losses = []
        lrs = []
        kd_losses = []
        hard_losses = []
        tokens_per_secs = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        epochs.append(int(row.get('epoch', 0)))
                        losses.append(float(row.get('loss', 0)))
                        lrs.append(float(row.get('lr', 0)))
                        kl = row.get('kd_loss', '').strip()
                        kd_losses.append(float(kl) if kl else None)
                        hl = row.get('hard_loss', '').strip()
                        hard_losses.append(float(hl) if hl else None)
                        ts = row.get('tokens_per_sec', '').strip()
                        tokens_per_secs.append(float(ts) if ts else 0)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            return

        if not epochs:
            return

        # Determine status
        if len(losses) >= 2:
            change_pct = ((losses[-1] - losses[0]) / losses[0]) * 100 if losses[0] > 0 else 0
            if losses[-1] < losses[0] * 0.9:
                status = "healthy"
                status_text = f"DRAFT TRAINING GOING WELL - Loss decreased {abs(change_pct):.1f}%"
                status_detail = "The draft model is learning from the target model."
            elif losses[-1] > losses[0] * 1.1:
                status = "overfitting"
                status_text = f"WARNING - Loss increased {change_pct:.1f}%"
                status_detail = "Draft model may be overfitting. Try reducing kd_epochs."
            else:
                status = "stable"
                status_text = f"STABLE - Loss changed {change_pct:.1f}%"
                status_detail = "Draft model training is stable."
        else:
            status = "stable"
            status_text = "DRAFT MODEL TRAINING"
            status_detail = "Insufficient data for status determination."

        # Summary values
        final_loss_str = f"{losses[-1]:.4f}" if losses else "N/A"
        final_lr_str = f"{lrs[-1]:.2e}" if lrs else "N/A"
        total_epochs_str = str(epochs[-1]) if epochs else "0"
        min_loss_str = f"{min(losses):.4f}" if losses else "N/A"

        # Serialize data for JavaScript
        epochs_json = json.dumps(epochs)
        losses_json = json.dumps(losses)
        lrs_json = json.dumps(lrs)
        kd_losses_json = json.dumps(kd_losses)
        hard_losses_json = json.dumps(hard_losses)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draft Model Training Report - {self.checkpoint_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .status {{ padding: 15px 20px; border-radius: 8px; margin: 15px 0; font-weight: bold; border-left: 5px solid; }}
        .status.healthy {{ background: #d4edda; color: #155724; border-color: #28a745; }}
        .status.overfitting {{ background: #f8d7da; color: #721c24; border-color: #dc3545; }}
        .status.stable {{ background: #fff3cd; color: #856404; border-color: #ffc107; }}
        .status-main {{ font-size: 18px; margin-bottom: 5px; }}
        .status-detail {{ font-size: 14px; font-weight: normal; opacity: 0.9; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
        .zoom-controls {{ position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; z-index: 10; }}
        .zoom-btn {{ padding: 4px 10px; font-size: 14px; font-weight: bold; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; opacity: 0.8; min-width: 28px; text-align: center; }}
        .zoom-btn:hover {{ opacity: 1; background: #2980b9; }}
        .zoom-btn.reset {{ background: #95a5a6; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        canvas {{ max-height: 300px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }}
        .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px; }}
        .badge-draft {{ background: #17a2b8; color: white; }}
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div id="google_translate_element" style="position:fixed;top:10px;right:10px;z-index:9999;background:white;padding:5px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;"></div>
    <script src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <script>
    function googleTranslateElementInit() {{
        new google.translate.TranslateElement({{pageLanguage: 'en', includedLanguages: 'es,fr,de,it,pt,ru,ja,ko,zh-CN,ar,hi,th,vi,nl,pl,sv,da,no,fi,tr,uk,cs,ro,hu,el,bg,hr,sk,sl,lt,lv,et,mt,ga,cy,eu,ca,gl,af,sq,bs,is,lb,mk,sr,be,kk,ky,tg,uz,tk,ka,hy,az', autoDisplay: false}}, 'google_translate_element');
    }}
    </script>
    <style>
        .skiptranslate {{ display: inline !important; }}
        .goog-te-gadget {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; }}
        .goog-te-gadget-simple {{ border: 1px solid #ddd !important; border-radius: 6px !important; padding: 2px 8px !important; background: #f8f8f8 !important; }}
        .goog-te-gadget-simple:hover {{ background: #e8e8e8 !important; }}
        .goog-te-combo {{ font-family: Roboto, sans-serif !important; font-size: 13px !important; border: none !important; background: transparent !important; cursor: pointer !important; }}
        body {{ top: 0 !important; }}
    </style>
    <div class="container">
        <div class="header">
            <h1>Draft Model Training Report: {self.checkpoint_name} <span class="badge badge-draft">DRAFT</span></h1>
            <div class="status {status}">
                <div class="status-main">{status_text}</div>
                <div class="status-detail">{status_detail}</div>
            </div>
            <p>Total epochs: {total_epochs_str} | Final loss: {final_loss_str} | Min loss: {min_loss_str}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{final_loss_str}</div>
                <div class="metric-label">Final Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{min_loss_str}</div>
                <div class="metric-label">Min Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{final_lr_str}</div>
                <div class="metric-label">Final LR</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_epochs_str}</div>
                <div class="metric-label">Total Epochs</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lossChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <canvas id="lrChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('lrChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('lrChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('lrChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="kdLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('kdLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('kdLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('kdLossChart')" title="Reset">&#8634;</button></div>
            </div>
            <div class="chart-container">
                <canvas id="hardLossChart"></canvas>
                <div class="zoom-controls"><button class="zoom-btn" onclick="zoomIn('hardLossChart')" title="Zoom In">+</button><button class="zoom-btn" onclick="zoomOut('hardLossChart')" title="Zoom Out">-</button><button class="zoom-btn reset" onclick="resetZoom('hardLossChart')" title="Reset">&#8634;</button></div>
            </div>
        </div>
    </div>

    <script>
        const epochs = {epochs_json};
        const losses = {losses_json};
        const lrs = {lrs_json};
        const kdLosses = {kd_losses_json};
        const hardLosses = {hard_losses_json};

        // Reusable zoom/pan configuration
        const zoomOptions = {{
            zoom: {{
                wheel: {{ enabled: true }},
                pinch: {{ enabled: true }},
                drag: {{ enabled: false }},
                mode: 'xy',
                scaleMode: 'xy'
            }},
            pan: {{
                enabled: true,
                mode: 'xy'
            }}
        }};

        function resetZoom(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.resetZoom();
        }}
        function zoomIn(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.zoom(1.2);
        }}
        function zoomOut(chartId) {{
            const chart = Chart.getChart(chartId);
            if (chart) chart.zoom(0.8);
        }}

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Total Loss',
                    data: losses,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231,76,60,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Draft Model - Total Loss' }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Loss' }} }},
                    x: {{ title: {{ display: true, text: 'Epoch' }} }}
                }}
            }}
        }});

        // Learning Rate Chart
        new Chart(document.getElementById('lrChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [{{
                    label: 'Learning Rate',
                    data: lrs,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52,152,219,0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Draft Model - Learning Rate' }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return 'LR: ' + ctx.parsed.y.toFixed(10);
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'LR' }} }},
                    x: {{ title: {{ display: true, text: 'Epoch' }} }}
                }}
            }}
        }});

        // KD Loss Chart (only if KD enabled)
        if (kdLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('kdLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'KD Loss',
                        data: kdLosses,
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155,89,182,0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Draft Model - Knowledge Distillation Loss' }} }},
                    scales: {{
                        y: {{ title: {{ display: true, text: 'KD Loss' }} }},
                        x: {{ title: {{ display: true, text: 'Epoch' }} }}
                    }}
                }}
            }});
        }} else {{
            document.getElementById('kdLossChart').parentElement.style.display = 'none';
        }}

        // Hard Loss Chart (only if KD enabled)
        if (hardLosses.some(v => v !== null)) {{
            new Chart(document.getElementById('hardLossChart'), {{
                type: 'line',
                data: {{
                    labels: epochs,
                    datasets: [{{
                        label: 'Hard Loss',
                        data: hardLosses,
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230,126,34,0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ title: {{ display: true, text: 'Draft Model - Next-Token Loss' }} }},
                    scales: {{
                        y: {{ title: {{ display: true, text: 'Hard Loss' }} }},
                        x: {{ title: {{ display: true, text: 'Epoch' }} }}
                    }}
                }}
            }});
        }} else {{
            document.getElementById('hardLossChart').parentElement.style.display = 'none';
        }}
    </script>
</body>
</html>"""

        # Write HTML file
        html_path = csv_path.replace('.csv', '_report.html')
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            if self.rank == 0:
                logger.info(f"  Draft training report saved to {html_path}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"  Could not generate draft training report: {e}")


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

            # Split into train/validation
            val_split = getattr(self.config, 'val_split', 0.1)
            val_batches_config = getattr(self.config, 'val_batches', 0)
            val_dataloader = None

            if val_split > 0 and dataset_length is not None and dataset_length > 10:
                val_size = max(1, int(dataset_length * val_split))
                train_size = dataset_length - val_size
                if self.rank == 0:
                    logger.info(f"Dataset split: {train_size} train / {val_size} validation ({val_split*100:.0f}%)")
                    if val_batches_config == 0:
                        logger.info(f"  val_batches=0: evaluating ALL {val_size // TRAINING_CONFIG['batch_size']} validation batches per epoch")
                    else:
                        logger.info(f"  val_batches={val_batches_config}: limiting validation to {val_batches_config} batches per epoch")

                train_dataset = _RangedIterableDataset(token_pair_generator, offset=0, limit=train_size, length=train_size)
                val_dataset = _RangedIterableDataset(token_pair_generator, offset=train_size, limit=val_size, length=val_size)

                iterable_dataset = train_dataset
            else:
                if val_split > 0 and self.rank == 0:
                    logger.info(f"Validation split disabled (dataset too small: {dataset_length} samples)")
                iterable_dataset = TokenPairIterableDataset(
                    token_pair_generator,
                    length=dataset_length,
                    rank=self.rank,
                    world_size=self.world_size
                )

            # Create train DataLoader
            if self.rank == 0:
                logger.info(f"Creating DataLoader (batch_size={TRAINING_CONFIG['batch_size']})...")
            pin_memory = self.use_gpu

            dataloader = DataLoader(
                iterable_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=False,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                num_workers=0
            )

            # Create validation DataLoader if split was applied
            if val_split > 0 and dataset_length is not None and dataset_length > 10:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=TRAINING_CONFIG['batch_size'],
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    pin_memory=pin_memory,
                    num_workers=0
                )
                if self.rank == 0:
                    logger.info("Validation DataLoader created")

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

            def _filter_state_dict(sd, model):
                """Filter state_dict to only include keys with matching shapes."""
                model_sd = model.state_dict()
                filtered = {}
                skipped = 0
                for k, v in sd.items():
                    if k in model_sd and v.shape == model_sd[k].shape:
                        filtered[k] = v
                    else:
                        skipped += 1
                if self.rank == 0 and skipped:
                    logger.info(f" Filtered {skipped} keys with shape mismatch (will be retrained)")
                return filtered

            # Initialize model — resume from checkpoint if it exists
            resume_checkpoint = None
            if os.path.exists(self.model_output_path):
                if self.rank == 0:
                    logger.info(f"Found existing checkpoint: {self.model_output_path}")
                    logger.info("Resuming training from checkpoint...")
                resume_checkpoint = torch.load(self.model_output_path, map_location='cpu', weights_only=False)
                arch = resume_checkpoint.get('architecture', {})
                embed_size = arch.get('embed_size', TRAINING_CONFIG['embed_size'])
                num_layers = arch.get('num_layers', 4)
                checkpoint_vocab_size = arch.get('vocab_size', self.tokenizer.vocab_size)
                current_vocab_size = self.tokenizer.vocab_size

                # Detect MoE from actual state_dict keys, not metadata
                sd_keys = set(resume_checkpoint['model_state_dict'].keys())
                checkpoint_has_moe = any('mlp.experts' in k or 'mlp.gate' in k for k in sd_keys)
                checkpoint_has_mtp = any('mtp_heads' in k for k in sd_keys)
                use_moe = checkpoint_has_moe or self.config.moe_enabled
                use_mtp = checkpoint_has_mtp or self.config.mtp_enabled

                if use_moe and use_mtp:
                    model = ChatModelMoEMTP(self.tokenizer, embed_size=embed_size, num_layers=num_layers,
                                            num_experts=arch.get('moe_num_experts', self.config.moe_num_experts),
                                            top_k=arch.get('moe_top_k', self.config.moe_top_k),
                                            load_balance_weight=arch.get('moe_load_balance_weight', self.config.moe_load_balance_weight),
                                            mtp_num_heads=arch.get('mtp_num_heads', self.config.mtp_num_heads),
                                            mtp_loss_weight=arch.get('mtp_loss_weight', self.config.mtp_loss_weight))
                elif use_mtp:
                    model = ChatModelMTP(self.tokenizer, embed_size=embed_size, num_layers=num_layers,
                                         mtp_num_heads=arch.get('mtp_num_heads', self.config.mtp_num_heads),
                                         mtp_loss_weight=arch.get('mtp_loss_weight', self.config.mtp_loss_weight))
                elif use_moe:
                    model = ChatModelMoE(self.tokenizer, embed_size=embed_size, num_layers=num_layers,
                                         num_experts=arch.get('moe_num_experts', self.config.moe_num_experts),
                                         top_k=arch.get('moe_top_k', self.config.moe_top_k),
                                         load_balance_weight=arch.get('moe_load_balance_weight', self.config.moe_load_balance_weight))
                else:
                    model = ChatModel(self.tokenizer, embed_size=embed_size, num_layers=num_layers)
                
                # Handle vocab_size mismatch
                if checkpoint_vocab_size != current_vocab_size:
                    if self.rank == 0:
                        logger.warning(f" vocab_size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}")
                        logger.info(" Loading only transformer layers (skipping embedding/head layers)...")

                    filtered = _filter_state_dict(resume_checkpoint['model_state_dict'], model)
                    missing, unexpected = model.load_state_dict(filtered, strict=False)
                    if self.rank == 0 and missing:
                        logger.info(f" Missing keys (will be retrained): {len(missing)}")
                else:
                    # Detect MoE from actual state_dict keys, not metadata
                    sd_keys = set(resume_checkpoint['model_state_dict'].keys())
                    checkpoint_has_moe = any('mlp.experts' in k or 'mlp.gate' in k for k in sd_keys)
                    current_moe = self.config.moe_enabled
                    if checkpoint_has_moe != current_moe:
                        if self.rank == 0:
                            logger.warning(f" Architecture mismatch: checkpoint MoE={checkpoint_has_moe}, current MoE={current_moe}")
                    filtered = _filter_state_dict(resume_checkpoint['model_state_dict'], model)
                    missing, unexpected = model.load_state_dict(filtered, strict=False)
                    if self.rank == 0 and missing:
                        logger.info(f" Missing keys (will be retrained): {len(missing)}")
                
                if self.rank == 0:
                    logger.info(f" Model loaded from checkpoint (epoch {resume_checkpoint.get('epoch', '?')}, loss {resume_checkpoint.get('loss', '?'):.4f})")
            else:
                model_type = 'ChatModel'
                if self.config.moe_enabled and self.config.mtp_enabled:
                    model_type = 'ChatModel(MoE+MTP)'
                elif self.config.moe_enabled:
                    model_type = 'ChatModelMoE'
                elif self.config.mtp_enabled:
                    model_type = 'ChatModelMTP'
                if self.rank == 0:
                    logger.info(f"Initializing {model_type} (embed_size={TRAINING_CONFIG['embed_size']}, num_layers=4)...")
                if self.config.moe_enabled and self.config.mtp_enabled:
                    model = ChatModelMoEMTP(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], num_layers=4,
                                            num_experts=self.config.moe_num_experts, top_k=self.config.moe_top_k,
                                            load_balance_weight=self.config.moe_load_balance_weight,
                                            mtp_num_heads=self.config.mtp_num_heads,
                                            mtp_loss_weight=self.config.mtp_loss_weight)
                elif self.config.mtp_enabled:
                    model = ChatModelMTP(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], num_layers=4,
                                         mtp_num_heads=self.config.mtp_num_heads,
                                         mtp_loss_weight=self.config.mtp_loss_weight)
                elif self.config.moe_enabled:
                    model = ChatModelMoE(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], num_layers=4,
                                         num_experts=self.config.moe_num_experts, top_k=self.config.moe_top_k,
                                         load_balance_weight=self.config.moe_load_balance_weight)
                    if self.config.moe_freeze_attention:
                        model.freeze_attention()
                else:
                    model = ChatModel(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], num_layers=4)

            model = self._setup_model_with_device_strategy(model, device)
            if self.rank == 0:
                logger.info(f" Model initialized and deployed")

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_index(), reduction='none')
            optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], weight_decay=TRAINING_CONFIG.get('weight_decay', 0.01))
            
            # Add learning rate scheduler
            sched_type = self.config.scheduler_type.lower()
            if sched_type == 'step':
                step_size = self.config.scheduler_step_size if self.config.scheduler_step_size > 0 else max(1, self.epochs // 3)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.config.scheduler_gamma)
            elif sched_type == 'exponential':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config.scheduler_gamma)
            elif sched_type == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.config.scheduler_patience, factor=self.config.scheduler_factor)
            elif sched_type == 'onecycle':
                scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=TRAINING_CONFIG['learning_rate'], total_steps=self.epochs)
            else:  # cosine (default)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.config.scheduler_eta_min)

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
                            # PyTorch 2.x CosineAnnealingLR uses group["lr"] (not base_lrs),
                            # so restored lr=eta_min would make cosine stuck at eta_min forever.
                            # We reset lr here and recompute the correct value below.
                            for group in optimizer.param_groups:
                                group['lr'] = TRAINING_CONFIG['learning_rate']
                            if self.rank == 0:
                                logger.info(f" Optimizer state restored")
                        except Exception as e:
                            if self.rank == 0:
                                logger.warning(f" Could not restore optimizer state: {e}")
                    if 'scheduler_state_dict' in resume_checkpoint and sched_type == 'cosine':
                        try:
                            restored_last_epoch = resume_checkpoint['scheduler_state_dict'].get('last_epoch', 0)
                            if restored_last_epoch > 0:
                                has_T_max = 'scheduler_T_max' in resume_checkpoint
                                old_T_max = resume_checkpoint.get('scheduler_T_max', 0)
                                lr_max = TRAINING_CONFIG['learning_rate']
                                eta_min = self.config.scheduler_eta_min

                                if has_T_max and restored_last_epoch >= old_T_max:
                                    # Cycle completed: start fresh new cycle
                                    new_T_max = self.epochs
                                    scheduler = lr_scheduler.CosineAnnealingLR(
                                        optimizer, T_max=new_T_max, eta_min=eta_min
                                    )
                                    for group in optimizer.param_groups:
                                        group['lr'] = lr_max
                                    if self.rank == 0:
                                        logger.info(
                                            f" Scheduler restarted: T_max={new_T_max}, "
                                            f"lr={lr_max:.2e} (old cycle completed at epoch {restored_last_epoch})"
                                        )
                                else:
                                    # Mid-cycle: decay from old lr to eta_min over new epochs
                                    # Get the actual lr from the restored optimizer state
                                    old_lr = resume_checkpoint['optimizer_state_dict']['param_groups'][0].get('lr', lr_max)
                                    new_T_max = self.epochs
                                    scheduler = lr_scheduler.CosineAnnealingLR(
                                        optimizer, T_max=new_T_max, eta_min=eta_min
                                    )
                                    scheduler.base_lrs = [old_lr]
                                    for group in optimizer.param_groups:
                                        group['lr'] = old_lr
                                    if self.rank == 0:
                                        reason = "legacy checkpoint" if not has_T_max else "mid-cycle"
                                        logger.info(
                                            f" Scheduler continued: T_max={new_T_max}, "
                                            f"lr={old_lr:.2e} -> {eta_min:.2e} over {new_T_max} epochs ({reason})"
                                        )
                        except Exception as e:
                            if self.rank == 0:
                                logger.warning(f" Could not restore scheduler state: {e}")
                    elif 'scheduler_state_dict' in resume_checkpoint:
                        try:
                            if self.rank == 0:
                                restored_epoch = resume_checkpoint['scheduler_state_dict'].get('last_epoch', 0)
                                logger.info(f" Scheduler not restored (non-cosine type); was at epoch {restored_epoch}")
                        except Exception as e:
                            if self.rank == 0:
                                logger.warning(f" Could not restore scheduler state: {e}")
                
                if 'loss' in resume_checkpoint:
                    self.best_loss = resume_checkpoint['loss']
                    if self.rank == 0:
                        logger.info(f" Best loss restored: {self.best_loss:.4f}")

            if self.rank == 0:
                if sched_type == 'step':
                    logger.info(f" Learning rate scheduler: StepLR (step_size={scheduler.step_size}, gamma={scheduler.gamma})")
                elif sched_type == 'exponential':
                    logger.info(f" Learning rate scheduler: ExponentialLR (gamma={scheduler.gamma})")
                elif sched_type == 'plateau':
                    logger.info(f" Learning rate scheduler: ReduceLROnPlateau (patience={scheduler.patience}, factor={scheduler.factor})")
                elif sched_type == 'onecycle':
                    logger.info(f" Learning rate scheduler: OneCycleLR (max_lr={TRAINING_CONFIG['learning_rate']})")
                else:
                    logger.info(f" Learning rate scheduler: CosineAnnealingLR (T_max={self.epochs}, eta_min={self.config.scheduler_eta_min})")

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

            # Early stopping state
            early_stopping_patience = getattr(self.config, 'early_stopping_patience', 0)
            best_val_loss = float('inf')
            patience_counter = 0

            # CSV metrics path
            metrics_csv_path = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}_metrics.csv')
            log_csv = getattr(self.config, 'log_metrics_csv', True) and self.rank == 0

            # TensorBoard initialization
            tb_writer = None
            if getattr(self.config, 'tensorboard_enabled', False) and self.rank == 0:
                if TENSORBOARD_AVAILABLE:
                    tb_comment = getattr(self.config, 'tensorboard_comment', '')
                    tb_log_dir = getattr(self.config, 'tensorboard_log_dir', 'runs')
                    if not os.path.isabs(tb_log_dir):
                        tb_log_dir = os.path.join(MODEL_CHECKPOINT_DIR, tb_log_dir)
                    tb_run_name = f"{self.checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if tb_comment:
                        tb_run_name += f"_{tb_comment}"
                    tb_writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, tb_run_name))
                    tb_freq = getattr(self.config, 'tensorboard_freq', 1)
                    logger.info(f"  TensorBoard enabled: log_dir={os.path.join(tb_log_dir, tb_run_name)} (every {tb_freq} epochs)")
                    tb_writer.add_text("config", str(self.config.to_dict()), 0)
                else:
                    logger.warning("  TensorBoard requested but tensorboard package not installed. Install: pip install tensorboard")

            num_epochs = self.epochs
            for epoch in range(num_epochs):
                if self.stop_event.is_set():
                    logger.warning("Training stop requested; ending before next epoch")
                    break

                epoch_start_time = time.time()

                # Training
                train_loss, total_tokens, grad_norm, thinking_metrics, agent_metrics, mtp_metrics, moe_metrics = self.train(model, dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])

                # Update learning rate
                if sched_type == 'plateau':
                    # Plateau scheduler needs val_loss
                    val_metric = val_loss if val_dataloader is not None else train_loss
                    scheduler.step(val_metric)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                # Validation
                val_loss = 0.0
                val_perplexity = 0.0
                if val_dataloader is not None:
                    val_loss, val_perplexity, val_batches_run = self.validate(
                        model, val_dataloader, criterion, device,
                        max_batches=val_batches_config if val_batches_config > 0 else 0
                    )

                # Compute metrics
                train_perplexity = math.exp(min(train_loss, 20))
                gap = val_loss - train_loss if val_dataloader is not None else 0.0
                epoch_time = time.time() - epoch_start_time
                tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0

                # MoE metrics (already collected inside train() before validation)
                # Log MoE summary (rank-0 only)
                if moe_metrics and self.rank == 0:
                    moe_entropy_norm = moe_metrics.get('moe_gate_entropy_norm', 0)
                    logger.info(f"  MoE Gate Entropy (norm): {moe_entropy_norm:.4f}")

                # MTP metrics (if enabled)
                if self.config.mtp_enabled and mtp_metrics and self.rank == 0:
                    mtp_loss_val = mtp_metrics.get('mtp_loss', 0)
                    logger.info(f"  MTP Loss: {mtp_loss_val:.6f}")

                # Log enhanced metrics (rank-0 only)
                if self.rank == 0:
                    current_epoch_num = self._next_epoch + epoch + 1
                    total_epochs_display = self._next_epoch + num_epochs
                    if val_dataloader is not None:
                        gap_sign = "+" if gap >= 0 else ""
                        logger.info(
                            f"Epoch {current_epoch_num:2d}/{total_epochs_display} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Perplexity: {train_perplexity:.2f}/{val_perplexity:.2f} | "
                            f"Gap: {gap_sign}{gap:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens/s: {tokens_per_sec:.0f} | "
                            f"Grad: {grad_norm:.2f}"
                        )
                    else:
                        if self.use_gpu:
                            gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                            logger.info(
                                f"Epoch {current_epoch_num:2d}/{total_epochs_display} | "
                                f"Train Loss: {train_loss:.4f} | "
                                f"Perplexity: {train_perplexity:.2f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Tokens/s: {tokens_per_sec:.0f} | "
                                f"Grad: {grad_norm:.2f} | "
                                f"GPU: {gpu_memory:.2f}GB"
                            )
                        else:
                            logger.info(
                                f"Epoch {current_epoch_num:2d}/{total_epochs_display} | "
                                f"Train Loss: {train_loss:.4f} | "
                                f"Perplexity: {train_perplexity:.2f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Tokens/s: {tokens_per_sec:.0f} | "
                                f"Grad: {grad_norm:.2f}"
                            )

                    # Save best model checkpoint
                    best_metric = val_loss if val_dataloader is not None else train_loss
                    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    checkpoint_data = {
                        'epoch': current_epoch_num,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scheduler_T_max': scheduler.T_max if sched_type == 'cosine' else None,
                        'loss': best_metric,
                        'train_loss': train_loss,
                        'val_loss': val_loss if val_dataloader is not None else None,
                        'tokenizer_path': os.path.join(CACHE_DIR, 'sentencepiece.model'),
                        'model_name': self.checkpoint_name,
                        'architecture': {
                            'embed_size': TRAINING_CONFIG['embed_size'],
                            'hidden_size': TRAINING_CONFIG['hidden_size'],
                            'num_layers': TRAINING_CONFIG.get('num_layers', 4),
                            'n_head': TRAINING_CONFIG.get('n_head', 4),
                            'n_positions': TRAINING_CONFIG.get('n_positions', 512),
                            'vocab_size': self.tokenizer.vocab_size,
                            'moe_enabled': self.config.moe_enabled,
                            'moe_num_experts': self.config.moe_num_experts,
                            'moe_top_k': self.config.moe_top_k,
                            'moe_load_balance_weight': self.config.moe_load_balance_weight,
                            'mtp_enabled': self.config.mtp_enabled,
                            'mtp_num_heads': self.config.mtp_num_heads,
                            'mtp_loss_weight': self.config.mtp_loss_weight,
                        },
                        'dataset_source': self.dataset_source,
                    }

                    # Save epoch-specific checkpoint (only if best)
                    if best_metric < self.best_loss:
                        self.best_loss = best_metric
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        epoch_filename = f"{self.checkpoint_name}_epoch_{current_epoch_num}_{timestamp}.pth"
                        epoch_path = os.path.join(MODEL_CHECKPOINT_DIR, epoch_filename)
                        torch.save(checkpoint_data, epoch_path)
                        logger.info(f"  Best model saved to {epoch_path}")

                    # Always overwrite chat_model.pth with latest epoch
                    torch.save(checkpoint_data, self.model_output_path)

                    # Save training state for epoch correlation
                    self._save_training_state(current_epoch_num)

                    # CSV metrics logging
                    if log_csv:
                        metrics_row = {
                            'epoch': current_epoch_num,
                            'train_loss': f"{train_loss:.6f}",
                            'val_loss': f"{val_loss:.6f}" if val_dataloader is not None else "",
                            'train_perplexity': f"{train_perplexity:.4f}",
                            'val_perplexity': f"{val_perplexity:.4f}" if val_dataloader is not None else "",
                            'gap': f"{gap:.6f}" if val_dataloader is not None else "",
                            'lr': f"{current_lr:.8f}",
                            'grad_norm': f"{grad_norm:.4f}",
                            'tokens_per_sec': f"{tokens_per_sec:.0f}",
                            'best_loss': f"{self.best_loss:.6f}",
                            'early_stop_patience': f"{patience_counter}",
                            # Thinking metrics
                            'thinking_accuracy': f"{thinking_metrics.get('thinking_token_accuracy', 0):.4f}" if thinking_metrics else "",
                            'thinking_open_acc': f"{thinking_metrics.get('thinking_open_accuracy', 0):.4f}" if thinking_metrics else "",
                            'thinking_close_acc': f"{thinking_metrics.get('thinking_close_accuracy', 0):.4f}" if thinking_metrics else "",
                            'thinking_coverage': f"{thinking_metrics.get('thinking_coverage', 0):.4f}" if thinking_metrics else "",
                            'response_accuracy': f"{thinking_metrics.get('response_token_accuracy', 0):.4f}" if thinking_metrics else "",
                            # Agent metrics
                            'agent_tool_call_acc': f"{agent_metrics.get('agent_tool_call_accuracy', 0):.4f}" if agent_metrics else "",
                            'agent_observation_acc': f"{agent_metrics.get('agent_observation_accuracy', 0):.4f}" if agent_metrics else "",
                            'agent_ratio': f"{agent_metrics.get('agent_ratio', 0):.6f}" if agent_metrics else "",
                            # MoE metrics
                            'moe_gate_entropy_norm': f"{moe_metrics.get('moe_gate_entropy_norm', 0):.4f}" if moe_metrics else "",
                            # MTP metrics
                            'mtp_loss': f"{mtp_metrics.get('mtp_loss', 0):.6f}" if mtp_metrics else "",
                        }
                        # Add per-expert utilization
                        for expert_id in range(self.config.moe_num_experts):
                            key = f'moe_expert_{expert_id}_util'
                            metrics_row[key] = f"{moe_metrics.get(key, 0):.4f}" if moe_metrics else ""
                        self._log_metrics_csv(metrics_row, metrics_csv_path)

                    # TensorBoard logging
                    if tb_writer is not None:
                        tb_freq = getattr(self.config, 'tensorboard_freq', 1)
                        if (epoch + 1) % tb_freq == 0:
                            step = current_epoch_num
                            tb_writer.add_scalars('loss', {'train': train_loss, 'val': val_loss if val_dataloader is not None else train_loss}, step)
                            tb_writer.add_scalar('perplexity/train', train_perplexity, step)
                            if val_dataloader is not None:
                                tb_writer.add_scalar('perplexity/val', val_perplexity, step)
                            tb_writer.add_scalar('gap', gap, step)
                            tb_writer.add_scalar('learning_rate', current_lr, step)
                            tb_writer.add_scalar('grad_norm', grad_norm, step)
                            tb_writer.add_scalar('tokens_per_sec', tokens_per_sec, step)
                            tb_writer.add_scalar('best_loss', self.best_loss, step)
                            # Thinking metrics
                            if thinking_metrics:
                                for k, v in thinking_metrics.items():
                                    if v != 0:
                                        tb_writer.add_scalar(f'thinking/{k}', v, step)
                            # Agent metrics
                            if agent_metrics:
                                for k, v in agent_metrics.items():
                                    if v != 0:
                                        tb_writer.add_scalar(f'agent/{k}', v, step)
                            # MoE metrics
                            if moe_metrics:
                                for k, v in moe_metrics.items():
                                    if v != 0:
                                        tb_writer.add_scalar(f'moe/{k}', v, step)
                            # MTP metrics
                            if mtp_metrics:
                                for k, v in mtp_metrics.items():
                                    if v != 0:
                                        tb_writer.add_scalar(f'mtp/{k}', v, step)

                # Early stopping check
                if early_stopping_patience > 0 and val_dataloader is not None:
                    current_best = val_loss
                    if current_best < best_val_loss:
                        best_val_loss = current_best
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.rank == 0:
                            logger.info(f"  Early stopping patience: {patience_counter}/{early_stopping_patience}")
                        if patience_counter >= early_stopping_patience:
                            if self.rank == 0:
                                logger.info(f"Early stopping triggered at epoch {current_epoch_num} (no improvement in {early_stopping_patience} epochs)")
                            break

                # DDP barrier
                if self.world_size > 1:
                    import torch.distributed as dist
                    dist.barrier()

            # Generate training report after epoch loop
            if log_csv and self.rank == 0:
                self._generate_training_report(metrics_csv_path)

            # Close TensorBoard writer
            if tb_writer is not None:
                tb_writer.close()
                if self.rank == 0:
                    logger.info("  TensorBoard log closed")

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

            # Train draft model for speculative decoding (after main training)
            if self.config.draft_enabled and not self.stop_event.is_set():
                if dataloader is not None:
                    self._train_draft_model(model, dataloader, device)
                elif self.rank == 0:
                    logger.warning("Cannot train draft model: no training data available")

            # Save final model + tokenizer state for consistent inference (rank-0 only)
            if self.rank == 0:
                logger.info("=" * 80)
                logger.info("[OK] Training completed! Saving final model and tokenizer...")

                # Use model.module.state_dict() for DDP to remove 'module.' prefix
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'epoch': self._next_epoch + num_epochs,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scheduler_T_max': scheduler.T_max if sched_type == 'cosine' else None,
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
                        'moe_enabled': self.config.moe_enabled,
                        'moe_num_experts': self.config.moe_num_experts,
                        'moe_top_k': self.config.moe_top_k,
                        'moe_load_balance_weight': self.config.moe_load_balance_weight,
                        'mtp_enabled': self.config.mtp_enabled,
                        'mtp_num_heads': self.config.mtp_num_heads,
                        'mtp_loss_weight': self.config.mtp_loss_weight,
                    },
                    'dataset_source': self.dataset_source,
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

    def _train_draft_model(self, target_model, train_dataloader, device, num_epochs=None):
        """Train a draft model for speculative decoding.

        The draft model is a small ChatModel (same architecture, fewer layers/smaller embed).
        When KD is enabled, it is trained with soft labels from the target model.
        """
        if not self.config.draft_enabled:
            return

        draft_num_layers = self.config.draft_num_layers
        draft_embed_size = self.config.draft_embed_size
        draft_hidden_size = self.config.draft_hidden_size
        draft_n_head = self.config.draft_n_head
        kd_enabled = self.config.draft_kd_enabled
        kd_temperature = self.config.draft_kd_temperature
        kd_loss_weight = self.config.draft_kd_loss_weight
        kd_epochs = num_epochs if num_epochs is not None else self.config.draft_kd_epochs

        if self.rank == 0:
            logger.info("=" * 80)
            logger.info(f"Training DRAFT MODEL for speculative decoding")
            logger.info(f"  Draft config: layers={draft_num_layers}, embed={draft_embed_size}, "
                        f"hidden={draft_hidden_size}, heads={draft_n_head}")
            if kd_enabled:
                logger.info(f"  KD enabled: temperature={kd_temperature}, loss_weight={kd_loss_weight}, epochs={kd_epochs}")
            else:
                logger.info(f"  Simple training (no KD): epochs={kd_epochs}")

        # Create draft model (small ChatModel)
        draft_model = ChatModel(self.tokenizer, embed_size=draft_embed_size, num_layers=draft_num_layers)
        draft_model = draft_model.to(device)

        draft_param_count = sum(p.numel() for p in draft_model.parameters())
        if self.rank == 0:
            logger.info(f"  Draft model parameters: {draft_param_count:,}")

        # Setup optimizer for draft
        draft_optimizer = torch.optim.AdamW(
            draft_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        draft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            draft_optimizer, T_max=kd_epochs, eta_min=1e-6
        )
        draft_criterion = torch.nn.CrossEntropyLoss()

        # Prepare target model for KD (frozen, eval mode)
        if kd_enabled and target_model is not None:
            target_model.eval()
            for param in target_model.parameters():
                param.requires_grad = False

        # Clear old draft CSV to avoid duplicate epochs across training runs
        draft_csv_path = os.path.join(
            MODEL_CHECKPOINT_DIR,
            f"{self.checkpoint_name}_draft_metrics.csv"
        )
        if os.path.exists(draft_csv_path):
            os.remove(draft_csv_path)

        # TensorBoard for draft model
        draft_tb_writer = None
        if getattr(self.config, 'tensorboard_enabled', False) and self.rank == 0 and TENSORBOARD_AVAILABLE:
            draft_tb_log_dir = getattr(self.config, 'tensorboard_log_dir', 'runs')
            if not os.path.isabs(draft_tb_log_dir):
                draft_tb_log_dir = os.path.join(MODEL_CHECKPOINT_DIR, draft_tb_log_dir)
            draft_tb_dir = os.path.join(
                draft_tb_log_dir,
                f"{self.checkpoint_name}_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            draft_tb_writer = SummaryWriter(log_dir=draft_tb_dir)
            if self.rank == 0:
                logger.info(f"  Draft TensorBoard: {draft_tb_dir}")

        draft_model.train()
        for epoch in range(kd_epochs):
            if self.stop_event.is_set():
                break

            epoch_loss = 0.0
            epoch_kd_loss = 0.0
            epoch_hard_loss = 0.0
            epoch_batches = 0
            epoch_start = time.time()
            try:
                known_total = len(train_dataloader)
            except (TypeError, AttributeError):
                known_total = None

            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                if self.stop_event.is_set():
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass on draft
                draft_logits = draft_model(inputs)

                if kd_enabled and target_model is not None:
                    # Knowledge Distillation: soft labels from target
                    with torch.no_grad():
                        target_logits = target_model(inputs)
                        if isinstance(target_logits, tuple):
                            target_logits = target_logits[0]  # Extract primary logits

                    # Soft target distribution (temperature scaled)
                    soft_targets = torch.nn.functional.softmax(target_logits / kd_temperature, dim=-1)
                    draft_log_probs = torch.nn.functional.log_softmax(draft_logits, dim=-1)
                    kd_loss = torch.nn.functional.kl_div(draft_log_probs, soft_targets, reduction='batchmean')

                    # Hard target loss (next-token prediction)
                    hard_loss = draft_criterion(draft_logits.view(-1, draft_logits.size(-1)), targets.view(-1))

                    # Combined loss
                    loss = kd_loss_weight * kd_loss + (1.0 - kd_loss_weight) * hard_loss
                    epoch_kd_loss += kd_loss.item()
                    epoch_hard_loss += hard_loss.item()
                else:
                    # Simple next-token prediction
                    loss = draft_criterion(draft_logits.view(-1, draft_logits.size(-1)), targets.view(-1))
                    epoch_kd_loss = 0.0
                    epoch_hard_loss = loss.item()

                # Backward pass
                draft_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(draft_model.parameters(), TRAINING_CONFIG['grad_clip_norm'])
                draft_optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

                if self.rank == 0 and (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / epoch_batches
                    elapsed = time.time() - epoch_start
                    rate = (batch_idx + 1) / elapsed
                    current_batch = batch_idx + 1
                    if known_total is not None and known_total > 1:
                        remaining = max(0, known_total - current_batch)
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_m, eta_s = divmod(int(eta_seconds), 60)
                        eta_str = f"{eta_m}m {eta_s}s" if eta_m > 0 else f"{eta_s}s"
                        total_str = f"/{known_total}"
                    else:
                        eta_str = "..."
                        total_str = ""
                    logger.info(f"  Draft epoch {epoch+1}/{kd_epochs} batch {current_batch}{total_str} | ETA: {eta_str} | loss: {avg_loss:.4f}")

            draft_scheduler.step()
            elapsed = time.time() - epoch_start
            avg_loss = epoch_loss / max(epoch_batches, 1)
            avg_kd_loss = epoch_kd_loss / max(epoch_batches, 1) if kd_enabled else 0.0
            avg_hard_loss = epoch_hard_loss / max(epoch_batches, 1)
            tokens_per_sec = (epoch_batches * inputs.size(0) * inputs.size(1)) / elapsed if elapsed > 0 else 0
            current_lr = draft_optimizer.param_groups[0]['lr']

            if self.rank == 0:
                logger.info(f"  Draft epoch {epoch+1}/{kd_epochs} completed | avg_loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

                # Log to CSV
                draft_metrics = {
                    'epoch': epoch + 1,
                    'loss': f"{avg_loss:.6f}",
                    'lr': f"{current_lr:.2e}",
                    'kd_loss': f"{avg_kd_loss:.6f}" if kd_enabled else '',
                    'hard_loss': f"{avg_hard_loss:.6f}",
                    'tokens_per_sec': f"{tokens_per_sec:.0f}",
                }
                self._log_draft_csv(draft_metrics, draft_csv_path)

                # TensorBoard logging for draft
                if draft_tb_writer is not None:
                    draft_step = epoch + 1
                    draft_tb_writer.add_scalar('draft/loss', avg_loss, draft_step)
                    draft_tb_writer.add_scalar('draft/lr', current_lr, draft_step)
                    draft_tb_writer.add_scalar('draft/hard_loss', avg_hard_loss, draft_step)
                    if kd_enabled:
                        draft_tb_writer.add_scalar('draft/kd_loss', avg_kd_loss, draft_step)
                    draft_tb_writer.add_scalar('draft/tokens_per_sec', tokens_per_sec, draft_step)

        # Generate HTML report
        if self.rank == 0:
            self._generate_draft_report(draft_csv_path)

        # Close TensorBoard writer
        if draft_tb_writer is not None:
            draft_tb_writer.close()

        # Save draft checkpoint
        if self.rank == 0:
            draft_output_path = os.path.join(
                MODEL_CHECKPOINT_DIR,
                f"{self.checkpoint_name}_draft.pth"
            )
            state_dict = draft_model.module.state_dict() if hasattr(draft_model, 'module') else draft_model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'architecture': {
                    'embed_size': draft_embed_size,
                    'hidden_size': draft_hidden_size,
                    'num_layers': draft_num_layers,
                    'n_head': draft_n_head,
                    'n_positions': TRAINING_CONFIG.get('n_positions', 512),
                    'vocab_size': self.tokenizer.vocab_size,
                },
                'is_draft': True,
                'target_model': self.checkpoint_name,
                'tokenizer_path': os.path.join(CACHE_DIR, 'sentencepiece.model'),
                'model_name': f"{self.checkpoint_name}_draft",
                'draft_config': {
                    'kd_enabled': kd_enabled,
                    'kd_temperature': kd_temperature,
                    'kd_loss_weight': kd_loss_weight,
                    'kd_epochs': kd_epochs,
                },
            }, draft_output_path)
            logger.info(f"  Draft model saved to: {draft_output_path}")

        # Cleanup
        del draft_model, draft_optimizer, draft_scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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

        PRESERVED_COLUMNS = ('input_ids', 'token_ids', 'question', 'answer', 'type', 'thinking', 'has_tool_call')
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

