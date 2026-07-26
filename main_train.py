import os
import pickle
import sys
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler
from datasets import Dataset
from chatmodel import ChatModel
import logging
import shutil
from datetime import datetime

# Optional SentencePiece support
try:
    import sentencepiece as spm
    SP_AVAILABLE = True
except Exception:
    spm = None
    SP_AVAILABLE = False

try:
    from bpe_tokenizer import SentencePieceTokenizerWrapper
except ImportError:
    SentencePieceTokenizerWrapper = None

class TrainingStopRequested(Exception):
    """Raised when a stop request is issued from the main thread."""


# Setup logging
logging.basicConfig(level=logging.INFO)
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
    'thinking_loss_weight': 0.5,  # loss weight for thinking tokens (0.0-1.0)
}

# Model checkpoint configuration
MODEL_CHECKPOINT_DIR = 'checkpoints'
TOKENIZER_VOCAB_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'tokenizer_vocab.json')
LATEST_MODEL_FILE = 'chat_model.pth'


class TokenPairIterableDataset(IterableDataset):
    """Iterable dataset that yields input-output token pairs without materializing all in memory."""
    def __init__(self, sequence_generator, length=None):
        self.sequence_generator = sequence_generator
        self._length = length

    def __iter__(self):
        return iter(self.sequence_generator())

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("TokenPairIterableDataset length not set")


class MainTrain:

    def __init__(self, args):

        logger.info("MainTrain initializing...")

        # Check if the tokenizer and cached dataset exist
        self.tokenizer = None
        self.tokenized_data = None
        self.loaded_dataset = None
        self.stop_event = threading.Event()

        # Device and training configuration tracking
        self.use_gpu = False
        self.use_mixed_precision = False
        self.use_gradient_checkpointing = False
        self.use_cpuonly = False
        self.best_loss = float('inf')

        # Thinking detection
        self.has_thinking_data = False
        self.thinking_sample_count = 0
        self.thinking_loss_weight = getattr(args, 'thinking_loss_weight', None) or TRAINING_CONFIG.get('thinking_loss_weight', 0.5)

        # Memory cap for entire application
        self.max_ram_fraction = getattr(args, 'max_ram_fraction', 0.75)
        self.max_ram_bytes = getattr(args, 'max_ram_bytes', None)

        # Parse command-line arguments
        self.aiml = args.aiml
        self.hf = args.hf
        self.onlytokenize = args.onlytokenize
        self.epochs = args.epochs
        self.use_cpuonly = getattr(args, 'use_cpuonly', False)
        self.cuda_device = getattr(args, 'cuda_device', None)

        # Dynamic checkpoint naming
        self.checkpoint_name = getattr(args, 'checkpoint_name', 'chat_model')
        self.dataset_source = getattr(args, 'dataset', 'dataset_cache')
        self.model_output_path = os.path.join(MODEL_CHECKPOINT_DIR, f'{self.checkpoint_name}.pth')
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

        # Load cached dataset and metadata (if present)
        self.cache_metadata = {}
        self.cache_has_token_ids = False
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
                "No BPE model found. Run: python main.py --prepare-data --aiml --hf"
            )

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
                    f"Please run: python main.py --prepare-data --aiml --hf"
                )
            
            logger.info(f"Loading cached dataset from: {CACHE_DATASET_FILE}")
            self.loaded_dataset = Dataset.load_from_disk(CACHE_DATASET_FILE)
            logger.info(f"✓ Loaded cached dataset: {len(self.loaded_dataset)} samples")
            
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
                self.cache_has_token_ids = True
                logger.info("Cached dataset contains 'token_ids' - will use pre-tokenized data for training")
            # fallback: if metadata includes a bpe model path, prefer tokenized flow
            elif isinstance(self.cache_metadata, dict) and self.cache_metadata.get('bpe_model_path'):
                self.cache_has_token_ids = True
                logger.info("Cache metadata indicates BPE model present; training will prefer tokenized cache if available")

            # Load statistics if available
            if os.path.exists(CACHE_STATS_FILE):
                with open(CACHE_STATS_FILE, 'rb') as f:
                    stats = pickle.load(f)
                logger.info(f"Dataset statistics loaded")
                logger.info(f"  Total samples: {stats.get('total_samples', 0):,}")

            # Detect thinking data in dataset
            self._detect_thinking_data()
            
        except Exception as e:
            logger.error(f"❌ Error loading cached dataset: {e}")
            sys.exit(1)

    def _detect_thinking_data(self):
        """Detect if the dataset contains <think> thinking data."""
        if self.loaded_dataset is None:
            return

        # Check cache metadata for thinking flag
        if isinstance(self.cache_metadata, dict) and self.cache_metadata.get('has_thinking_tokens'):
            self.has_thinking_data = True
            logger.info("✓ Thinking data detected (from cache metadata)")
            return

        # Heuristic: sample first 100 items and check for <think> tags
        sample_size = min(100, len(self.loaded_dataset))
        thinking_count = 0
        # Get thinking token IDs for detecting pre-tokenized thinking data
        thinking_id = getattr(self.tokenizer, 'get_thinking_index', lambda: -1)()
        thinking_end_id = getattr(self.tokenizer, 'get_thinking_end_index', lambda: -1)()
        for i in range(sample_size):
            item = self.loaded_dataset[i]
            value = item.get('input_ids', item.get('token_ids', ''))
            if isinstance(value, str) and '<think>' in value:
                thinking_count += 1
            elif isinstance(value, list) and thinking_id >= 0 and thinking_end_id >= 0:
                if thinking_id in value and thinking_end_id in value:
                    thinking_count += 1

        if thinking_count > 0:
            self.has_thinking_data = True
            self.thinking_sample_count = thinking_count
            logger.info(f"✓ Thinking data detected ({thinking_count}/{sample_size} samples contain <think>)")
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
        try:
            cpus = mp.cpu_count()
            if cpus <= 2:
                return 1
            return min(4, max(1, cpus // 2))
        except Exception:
            return 1

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
        """Setup device configuration with CPU-GPU combined strategy and mixed precision support."""
        
        # Force CPU if requested
        if self.use_cpuonly:
            cuda_available = False
        else:
            # If user specified --cuda-device, torch should read CUDA_VISIBLE_DEVICES set by main.py
            cuda_available = torch.cuda.is_available()

        # If explicit device is set and available, force it on torch
        if self.cuda_device is not None and not self.use_cpuonly:
            try:
                selected = int(str(self.cuda_device).split(',')[0])
                torch.cuda.set_device(selected)
            except Exception:
                pass

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"\n{'='*80}")
            logger.info(f"GPU DETECTED: {device_name}")
            logger.info(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
            logger.info(f"  Total Memory: {total_memory:.2f} GB")
            logger.info(f"{'='*80}\n")
            
            # Check for Tesla K80
            if "K80" in device_name or "Tesla" in device_name:
                logger.info("✓ Tesla K80 GPU detected - using optimized configuration")
                self.use_gpu = True
                self.use_mixed_precision = True
                self.use_gradient_checkpointing = True
                # Tesla K80 has 24GB per GPU, enable memory efficient training
                torch.cuda.set_per_process_memory_fraction(0.9)
            else:
                logger.info("✓ CUDA GPU detected - using optimized configuration")
                self.use_gpu = True
                self.use_mixed_precision = True
                self.use_gradient_checkpointing = False
            
            device = torch.device("cuda:0")
        else:
            logger.info(f"\n{'='*80}")
            logger.info("NO GPU DETECTED - Using CPU training")
            logger.info(f"{'='*80}\n")
            self.use_gpu = False
            self.use_mixed_precision = False
            self.use_gradient_checkpointing = False
            device = torch.device("cpu")
        
        # Enable cudnn benchmarking for faster training
        if cuda_available:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("✓ CuDNN optimization enabled")
        
        logger.info(f"Mixed Precision Training: {'ENABLED' if self.use_mixed_precision else 'DISABLED'}")
        logger.info(f"Gradient Checkpointing: {'ENABLED' if self.use_gradient_checkpointing else 'DISABLED'}")
        logger.info(f"Training Device: {device}")
        
        return device

    def _setup_model_with_device_strategy(self, model, device):
        """Setup model with CPU-GPU combined device strategy for efficient memory usage."""
        
        if not self.use_gpu:
            # CPU only - move entire model to CPU
            model = model.to(device)
            logger.info("Model deployed on CPU")
            return model
        
        # GPU available - use intelligent device placement
        try:
            # Move model to GPU
            model = model.to(device)
            logger.info(f"Model deployed on GPU: {device}")
            
            # Enable gradient checkpointing if available
            if self.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled for memory efficiency")
            
            # For very large models, we can implement layer-wise CPU-GPU distribution
            # This is useful for models that don't fit in GPU memory
            if hasattr(model, 'get_total_params'):
                total_params = model.get_total_params()
                gpu_memory_available = torch.cuda.get_device_properties(device).total_memory / 1e9
                estimated_model_memory = (total_params * 4) / 1e9  # Rough estimate in GB
                
                if estimated_model_memory > gpu_memory_available * 0.8:
                    logger.warning(f"Model size ({estimated_model_memory:.2f}GB) approaches GPU memory ({gpu_memory_available:.2f}GB)")
                    logger.info("Implementing model parallellism across CPU and GPU...")
                    # Keep model on GPU but use CPU offloading where possible
                    if hasattr(model, 'cpu_offload'):
                        model.cpu_offload()
            
            return model
            
        except RuntimeError as e:
            logger.warning(f"Could not move model to GPU: {e}")
            logger.info("Falling back to CPU training...")
            model = model.to(torch.device("cpu"))
            self.use_gpu = False
            self.use_mixed_precision = False
            return model

    def _compute_loss(self, model, inputs, targets, criterion):
        """Unified loss computation with thinking-aware loss weighting.

        Thinking tokens (between <think> and </think>) receive a lower loss weight
        (thinking_loss_weight) so the model focuses more on learning the response.
        """
        outputs = model(inputs)

        # Build weight mask on 2D tensors before flattening
        weights = torch.ones_like(targets, dtype=torch.float)
        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()

        if thinking_id >= 0 and thinking_end_id >= 0 and self.has_thinking_data:
            batch_size, seq_len = targets.shape
            for b in range(batch_size):
                in_thinking = False
                for s in range(seq_len):
                    token = targets[b, s].item()
                    if token == thinking_end_id:
                        in_thinking = False
                    if token == thinking_id:
                        in_thinking = True
                    # Apply lower weight only to tokens BETWEEN delimiters (not the delimiters themselves)
                    if in_thinking and token != thinking_id and token != thinking_end_id:
                        weights[b, s] = self.thinking_loss_weight

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
        loss = (token_losses * weights).sum() / weights.sum()
        return loss

    def _compute_thinking_metrics(self, model, inputs, targets, device):
        """Compute metrics for thinking token generation if thinking data is present."""
        if not self.has_thinking_data:
            return {}

        thinking_id = self.tokenizer.get_thinking_index()
        thinking_end_id = self.tokenizer.get_thinking_end_index()

        if thinking_id < 0 or thinking_end_id < 0:
            return {}

        with torch.no_grad():
            outputs = model(inputs)
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
    def train(self, model, dataloader, criterion, optimizer, device, scaler=None, accumulation_steps=1):
        """Train function with support for mixed precision training and gradient accumulation."""

        total_loss = 0
        total_batches = 0
        try:
            num_batches = len(dataloader)
        except (TypeError, AttributeError):
            num_batches = float('inf')
        model.train()
        optimizer.zero_grad()
        epoch_start_time = time.time()
        logger.info("Training loop started; the model is actively processing batches")

        # Thinking metrics accumulators
        thinking_metrics_accum = {}

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if self.stop_event.is_set():
                logger.info("Stop requested; exiting current training epoch early")
                break

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                batch_str = f"{batch_idx + 1}/{num_batches}" if num_batches != float('inf') else f"{batch_idx + 1}"
                elapsed = time.time() - epoch_start_time
                if batch_idx > 0 and num_batches != float('inf'):
                    eta_seconds = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)
                    eta_m, eta_s = divmod(int(eta_seconds), 60)
                    eta_str = f"{eta_m}m {eta_s}s" if eta_m > 0 else f"{eta_s}s"
                    logger.info(f"Training batch {batch_str} | ETA: {eta_str}")
                else:
                    logger.info(f"Training batch {batch_str} in progress...")

            total_batches += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            if self.use_mixed_precision and scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = self._compute_loss(model, inputs, targets, criterion)
            else:
                loss = self._compute_loss(model, inputs, targets, criterion)

            # Compute thinking metrics periodically
            if self.has_thinking_data and (batch_idx + 1) % 50 == 0:
                thinking_metrics = self._compute_thinking_metrics(model, inputs, targets, device)
                for key, value in thinking_metrics.items():
                    if key not in thinking_metrics_accum:
                        thinking_metrics_accum[key] = []
                    thinking_metrics_accum[key].append(value)

            # Backward pass
            original_loss = self._backward_pass(loss, optimizer, scaler, accumulation_steps)
            total_loss += original_loss.item()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
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

            # Memory cleanup on GPU
            if self.use_gpu and (batch_idx + 1) % TRAINING_CONFIG['memory_cleanup_interval'] == 0:
                torch.cuda.empty_cache()

        num_batches = max(1, total_batches)
        avg_loss = total_loss / num_batches

        # Log thinking metrics summary
        if thinking_metrics_accum:
            logger.info("\n  Thinking Metrics Summary:")
            for key, values in thinking_metrics_accum.items():
                avg_val = sum(values) / len(values) if values else 0
                logger.info(f"    {key}: {avg_val:.4f}")

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
            TokenPairIterableDataset(csv_pair_generator, length=csv_length),
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
            loaded_dataset = self.loaded_dataset

            logger.info(f"Dataset size: {len(loaded_dataset)} samples")
            logger.info(f"Dataset columns: {loaded_dataset.column_names}")

            # Log thinking data status
            if self.has_thinking_data:
                logger.info(f"✓ Thinking data: ENABLED (model will learn <think>...</think> structure)")
            else:
                logger.info(f"ℹ Thinking data: NOT detected (standard training mode)")

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

            logger.info(f"✓ Processed {sample_count} samples for tokenizer fitting")
            logger.info(f"✓ Tokenizer vocabulary size: {self.tokenizer.vocab_size}")

            # Save tokenizer vocabulary for chat loading inside checkpoints
            os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
            self.tokenizer.save_vocabulary(TOKENIZER_VOCAB_FILE)
            logger.info(f"✓ Tokenizer vocabulary saved to {TOKENIZER_VOCAB_FILE}")

            # Preserve raw dataset text for special facts fine-tuning before tokenization
            self.raw_dataset = loaded_dataset

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
            iterable_dataset = TokenPairIterableDataset(token_pair_generator, length=dataset_length)

            # Create DataLoader with custom collate function
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

            logger.info("✓ DataLoader created")
        
        except Exception as e:
            logger.error(f"Error in data preparation: {e}", exc_info=True)
            sys.exit(1)

        try:
            # CPU configuration already set in main.py entry point
            var_num_cores = os.environ.get("MKL_NUM_THREADS", mp.cpu_count())
            var_num_threads = os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())
            logger.info(f"CPU configuration - Cores: {var_num_cores}, Threads: {var_num_threads}")

            # Setup device with improved configuration for CPU-GPU combined training
            device = self._setup_device_and_config()

            # Initialize model with device strategy
            logger.info(f"Initializing ChatModel (embed_size={TRAINING_CONFIG['embed_size']}, hidden_size={TRAINING_CONFIG['hidden_size']}, num_layers=4)...")
            model = ChatModel(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], hidden_size=TRAINING_CONFIG['hidden_size'], num_layers=4)
            model = self._setup_model_with_device_strategy(model, device)
            logger.info(f"✓ Model initialized and deployed")

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_index())
            optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
            
            # Add learning rate scheduler
            scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, self.epochs // 3), gamma=0.1)
            logger.info(f"✓ Learning rate scheduler: StepLR (step_size={scheduler.step_size}, gamma={scheduler.gamma})")

            # Initialize gradient scaler for mixed precision training
            scaler = GradScaler() if self.use_mixed_precision else None
            if scaler:
                logger.info("✓ Gradient scaler initialized for mixed precision training")

            # Create checkpoint directory
            os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
            logger.info(f"✓ Checkpoint directory: {MODEL_CHECKPOINT_DIR}")

            # Warm-up phase (optional light training)
            if TRAINING_CONFIG.get('warm_up', False):
                warm_up_ratio = float(TRAINING_CONFIG.get('warm_up_ratio', 0.1))
                warm_up_steps = int(TRAINING_CONFIG.get('warm_up_steps', 100))
                warmup_limit = max(1, int(len(loaded_dataset) * warm_up_ratio))

                logger.info(f"\nStarting warm-up phase (light training) with {warmup_limit} samples and up to {warm_up_steps} batches...")

                def warmup_pair_generator():
                    idx = 0
                    for input_ids, output_ids in token_pair_generator():
                        if idx >= warmup_limit:
                            break
                        yield input_ids, output_ids
                        idx += 1

                warmup_dataloader = DataLoader(
                    TokenPairIterableDataset(warmup_pair_generator),
                    batch_size=TRAINING_CONFIG['batch_size'],
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    pin_memory=pin_memory,
                    num_workers=0
                )

                _ = self.train(model, warmup_dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                logger.info("✓ Warm-up phase completed")

            # Main training loop
            logger.info(f"\nStarting main training phase ({self.epochs} epochs)...")
            logger.info(f"{'='*80}")
            
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
                
                # Memory and performance information
                if self.use_gpu:
                    gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Training loss (avg per batch): {loss:.4f} | Learning rate (LR): {current_lr:.2e} | GPU mem used: {gpu_memory:.2f}GB / reserved: {gpu_memory_reserved:.2f}GB")
                else:
                    logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Training loss (avg per batch): {loss:.4f} | Learning rate (LR): {current_lr:.2e}")

                # Save best model checkpoint after each epoch
                if loss < self.best_loss:
                    self.best_loss = loss
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    epoch_filename = f"{self.checkpoint_name}_epoch_{epoch+1}_{timestamp}.pth"
                    epoch_path = os.path.join(MODEL_CHECKPOINT_DIR, epoch_filename)
                    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'tokenizer': self.tokenizer,
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
                    }, epoch_path)
                    logger.info(f"  ✓ Epoch {epoch+1} checkpoint saved to {epoch_path}")

            # Stop requested? Do not write a partial final checkpoint.
            if self.stop_event.is_set():
                logger.info("\nStop requested; skipping final model save")
                return

            # Fine-tune on CSV data if available
            csv_dataloader = self._get_csv_dataloader(batch_size=TRAINING_CONFIG['batch_size'])
            if csv_dataloader is not None:
                logger.info("\nStarting CSV data fine-tuning...")
                try:
                    _ = self.train(model, csv_dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                    logger.info("✓ CSV data fine-tuning completed")
                except Exception as e:
                    logger.warning(f"CSV data fine-tuning failed: {e}")

            # Save final model + tokenizer state for consistent inference
            logger.info(f"\n{'='*80}")
            logger.info("Training completed! Saving final model and tokenizer...")

            # Save final checkpoint with metadata
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            epoch_filename = f"{self.checkpoint_name}_final_{timestamp}.pth"
            epoch_path = os.path.join(MODEL_CHECKPOINT_DIR, epoch_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': self.best_loss,
                'tokenizer': self.tokenizer,
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
            }, epoch_path)
            logger.info(f"  Final epoch checkpoint saved to {epoch_path}")

            # Save final model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': self.best_loss,
                'tokenizer': self.tokenizer,
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
            }, self.model_output_path)
            logger.info(f"✓ Model + tokenizer saved to {self.model_output_path}")

        except TrainingStopRequested:
            logger.warning("\nTraining interrupted by request")
            return
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            self.stop_event.set()
            return
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

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
            logger.info(f"✓ Loaded tokenized dataset with {len(tokenized_ds)} samples")
            return tokenized_ds

        if 'token_ids' in self.loaded_dataset.column_names:
            logger.info("Dataset already contains token_ids; skipping tokenization")
            return self.loaded_dataset

        logger.info("Tokenizing cached dataset for faster training...")

        columns_to_remove = [c for c in self.loaded_dataset.column_names if c not in ('input_ids', 'token_ids')]
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
        logger.info(f"✓ Saved tokenized dataset to {CACHE_TOKENIZED_DATASET_DIR}")
        return tokenized_ds
