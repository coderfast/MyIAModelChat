import os
import pickle
import sys
import time
import math
import json
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from datasets import Dataset
from commons.model.chatmodel import ChatModel
from commons.model.chatmodel_moe import ChatModelMoE
from commons.model.chatmodel_mtp import ChatModelMTP
from commons.model.chatmodel_moe_mtp import ChatModelMoEMTP
import logging

from datetime import datetime

from training.config import TrainingConfig, TrainingStopRequested, CACHE_DIR, CACHE_DATASET_FILE, CACHE_TOKENIZED_DATASET_DIR, CACHE_STATS_FILE, CACHE_METADATA_FILE, MODEL_CHECKPOINT_DIR, TOKENIZER_VOCAB_FILE
from training.datasets import TokenPairIterableDataset, _RangedIterableDataset, DatasetMixin
from training.loss import LossMixin
from training.device import DeviceMixin
from training.reporting import ReportingMixin

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

logger = logging.getLogger(__name__)

class Trainer(LossMixin, DeviceMixin, DatasetMixin, ReportingMixin):

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
        self.thinking_loss_weight = self.config.thinking_loss_weight

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
                with open(state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('last_epoch', 0)
            except Exception:
                pass
        return 0

    def _save_training_state(self, epoch):
        """Save last epoch number to training_state.json."""
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
        problem_id = getattr(self.tokenizer, 'get_problem_index', lambda: -1)()
        thinking_mode_id = getattr(self.tokenizer, 'get_thinking_mode_index', lambda: -1)()
        final_id = getattr(self.tokenizer, 'get_final_index', lambda: -1)()

        for i in range(sample_size):
            item = self.loaded_dataset[i]
            value = item.get('input_ids', item.get('token_ids', ''))
            if isinstance(value, str) and ('<|thinking|>' in value or '<|problem|>' in value):
                thinking_count += 1
            elif isinstance(value, list):
                if thinking_id >= 0 and thinking_end_id >= 0:
                    if thinking_id in value:
                        thinking_count += 1
                elif problem_id >= 0 or thinking_mode_id >= 0:
                    if problem_id in value or thinking_mode_id in value:
                        thinking_count += 1

        if thinking_count > 0:
            self.has_thinking_data = True
            self.thinking_sample_count = thinking_count
            logger.info(f"Thinking data detected ({thinking_count}/{sample_size} samples contain <|thinking|>)")
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_clip_norm)
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


    def performMainTrain(self):
        """Main training loop with proper error handling and model checkpointing."""
        
        try:
            # Use the cached dataset
            _pre_tokenized_dataset = self.loaded_dataset

            logger.info(f"Dataset size: {len(_pre_tokenized_dataset)} samples")
            logger.info(f"Dataset columns: {_pre_tokenized_dataset.column_names}")

            # Log thinking data status
            if self.has_thinking_data:
                logger.info(f" Thinking data: ENABLED (model will learn <|thinking|>/<|final|> structure)")
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
                        logger.info(f"  val_batches=0: evaluating ALL {val_size // self.config.batch_size} validation batches per epoch")
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
                logger.info(f"Creating DataLoader (batch_size={self.config.batch_size})...")
            pin_memory = self.use_gpu

            dataloader = DataLoader(
                iterable_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                num_workers=0
            )

            # Create validation DataLoader if split was applied
            if val_split > 0 and dataset_length is not None and dataset_length > 10:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
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
                embed_size = arch.get('embed_size', self.config.embed_size)
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
                    logger.info(f"Initializing {model_type} (embed_size={self.config.embed_size}, num_layers={self.config.num_layers})...")
                if self.config.moe_enabled and self.config.mtp_enabled:
                    model = ChatModelMoEMTP(self.tokenizer, embed_size=self.config.embed_size, num_layers=self.config.num_layers,
                                            num_experts=self.config.moe_num_experts, top_k=self.config.moe_top_k,
                                            load_balance_weight=self.config.moe_load_balance_weight,
                                            mtp_num_heads=self.config.mtp_num_heads,
                                            mtp_loss_weight=self.config.mtp_loss_weight)
                elif self.config.mtp_enabled:
                    model = ChatModelMTP(self.tokenizer, embed_size=self.config.embed_size, num_layers=self.config.num_layers,
                                         mtp_num_heads=self.config.mtp_num_heads,
                                         mtp_loss_weight=self.config.mtp_loss_weight)
                elif self.config.moe_enabled:
                    model = ChatModelMoE(self.tokenizer, embed_size=self.config.embed_size, num_layers=self.config.num_layers,
                                         num_experts=self.config.moe_num_experts, top_k=self.config.moe_top_k,
                                         load_balance_weight=self.config.moe_load_balance_weight)
                    if self.config.moe_freeze_attention:
                        model.freeze_attention()
                else:
                    model = ChatModel(self.tokenizer, embed_size=self.config.embed_size, num_layers=self.config.num_layers)

            model = self._setup_model_with_device_strategy(model, device)
            if self.rank == 0:
                logger.info(f" Model initialized and deployed")

            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_index(), reduction='none')
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            
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
                scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.learning_rate, total_steps=self.epochs)
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
                                group['lr'] = self.config.learning_rate
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
                                lr_max = self.config.learning_rate
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
                    logger.info(f" Learning rate scheduler: OneCycleLR (max_lr={self.config.learning_rate})")
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
            if self.config.warm_up:
                warm_up_ratio = float(self.config.warm_up_ratio)
                warm_up_steps = int(self.config.warm_up_steps)
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
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    pin_memory=pin_memory,
                    num_workers=0
                )

                warmup_num_batches = max(1, math.ceil(warmup_limit / self.config.batch_size))
                _ = self.train(model, warmup_dataloader, criterion, optimizer, device, scaler, self.config.accumulation_steps, num_batches_override=warmup_num_batches, is_warmup=True)
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
                train_loss, total_tokens, grad_norm, thinking_metrics, agent_metrics, mtp_metrics, moe_metrics = self.train(model, dataloader, criterion, optimizer, device, scaler, self.config.accumulation_steps)

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
                            'embed_size': self.config.embed_size,
                            'hidden_size': self.config.hidden_size,
                            'num_layers': self.config.num_layers,
                            'n_head': self.config.n_head,
                            'n_positions': self.config.n_positions,
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
            csv_dataloader = self._get_csv_dataloader(batch_size=self.config.batch_size)
            if csv_dataloader is not None:
                if self.rank == 0:
                    logger.info("Starting CSV data fine-tuning...")
                try:
                    _ = self.train(model, csv_dataloader, criterion, optimizer, device, scaler, self.config.accumulation_steps)
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
                        'embed_size': self.config.embed_size,
                        'hidden_size': self.config.hidden_size,
                        'num_layers': self.config.num_layers,
                        'n_head': self.config.n_head,
                        'n_positions': self.config.n_positions,
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
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
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
                torch.nn.utils.clip_grad_norm_(draft_model.parameters(), self.config.grad_clip_norm)
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
                    'n_positions': self.config.n_positions,
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

