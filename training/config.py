"""Training configuration constants and dataclass."""
import os
from dataclasses import dataclass
from typing import Optional, List


class TrainingStopRequested(Exception):
    """Raised when a stop request is issued from the main thread."""


# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_TOKENIZED_DATASET_DIR = os.path.join(CACHE_DIR, 'prepared_dataset_tokenized')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.pkl')

# Model checkpoint configuration
MODEL_CHECKPOINT_DIR = 'checkpoints'
TOKENIZER_VOCAB_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'tokenizer_vocab.json')
LATEST_MODEL_FILE = 'chat_model.pth'


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
    # Core training hyperparameters (previously in TRAINING_CONFIG dict)
    batch_size: int = 4
    accumulation_steps: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 4
    n_head: int = 4
    n_positions: int = 512
    grad_clip_norm: float = 1.0
    memory_cleanup_interval: int = 10
    warm_up: bool = True
    warm_up_ratio: float = 0.1
    warm_up_steps: int = 100
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
    val_split: float = 0.1
    val_batches: int = 0
    early_stopping_patience: int = 5
    log_metrics_csv: bool = True
    # LR Scheduler settings
    scheduler_type: str = 'cosine'
    scheduler_eta_min: float = 1e-6
    scheduler_step_size: int = 0
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    # TensorBoard fields
    tensorboard_enabled: bool = False
    tensorboard_log_dir: str = 'runs'
    tensorboard_comment: str = ''
    tensorboard_freq: int = 1
    # DDP fields (single-machine multi-GPU or multi-node)
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master_addr: str = 'localhost'
    master_port: int = 29500

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    # Mapping: nested JSON path -> flat TrainingConfig field
    _JSON_TO_FIELD = {
        # training
        'training.epochs': 'epochs',
        'training.checkpoint_name': 'checkpoint_name',
        'training.dataset_source': 'dataset_source',
        # training hyperparameters
        'training.batch_size': 'batch_size',
        'training.accumulation_steps': 'accumulation_steps',
        'training.learning_rate': 'learning_rate',
        'training.weight_decay': 'weight_decay',
        'training.embed_size': 'embed_size',
        'training.hidden_size': 'hidden_size',
        'training.num_layers': 'num_layers',
        'training.n_head': 'n_head',
        'training.n_positions': 'n_positions',
        'training.grad_clip_norm': 'grad_clip_norm',
        'training.memory_cleanup_interval': 'memory_cleanup_interval',
        'training.warm_up': 'warm_up',
        'training.warm_up_ratio': 'warm_up_ratio',
        'training.warm_up_steps': 'warm_up_steps',
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

    # Reverse mapping: flat field -> nested JSON path
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
