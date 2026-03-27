import os
import argparse
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from aimlloder import *
from dialogmanager import DialogueManager
from simpletokenizer import *
from chatmodel import *
from chatdataset import *
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import pipeline
import logging

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')

# Training configuration constants
TRAINING_CONFIG = {
    'batch_size': 4,
    'accumulation_steps': 8,
    'learning_rate': 1e-3,
    'embed_size': 128,
    'hidden_size': 256,
    'grad_clip_norm': 1.0,
    'memory_cleanup_interval': 10,
    'warm_up_warmup': True,
}

# Model checkpoint configuration
MODEL_CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_FILE = os.path.join(MODEL_CHECKPOINT_DIR, 'best_model.pth')
LATEST_MODEL_FILE = 'chat_model.pth'


class TokenPairIterableDataset(IterableDataset):
    """Iterable dataset that yields input-output token pairs without materializing all in memory."""
    def __init__(self, sequence_generator):
        self.sequence_generator = sequence_generator

    def __iter__(self):
        return iter(self.sequence_generator())


class MainTrain:
    
    def __init__(self, args):
        
        logger.info("MainTrain initializing...")

        # Check if the tokenizer and cached dataset exist
        self.tokenizer = None
        self.tokenized_data = None
        self.loaded_dataset = None
        
        # Device and training configuration tracking
        self.use_gpu = False
        self.use_mixed_precision = False
        self.use_gradient_checkpointing = False
        self.use_cpuonly = False
        self.best_loss = float('inf')

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

        # Load cached dataset
        print(f"Loading dataset from cache...")
        self._load_cached_dataset()

        # Create Tokenizer
        print(f"Create Tokenizer")
        if self.tokenizer == None:
            self.tokenizer = SimpleTokenizer()

        print(f"MainTrain initialized...")

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
            
            # Load statistics if available
            if os.path.exists(CACHE_STATS_FILE):
                with open(CACHE_STATS_FILE, 'rb') as f:
                    stats = pickle.load(f)
                logger.info(f"Dataset statistics loaded")
                logger.info(f"  Total samples: {stats.get('total_samples', 0):,}")
            
        except Exception as e:
            logger.error(f"❌ Error loading cached dataset: {e}")
            sys.exit(1)

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

    def _sample_generator(self):
        """Yield text or tokenized sequences from the loaded dataset in streaming mode."""
        for item in self.loaded_dataset:
            value = item.get('input_ids', None)
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
        """Unified loss computation for both mixed and standard precision."""
        outputs = model(inputs)
        
        # Reshape outputs and targets
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.contiguous().view(-1)
        
        # Ignore padded elements
        non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
        outputs = outputs[non_pad_mask]
        targets = targets[non_pad_mask]
        
        return criterion(outputs, targets)
    
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
        model.train()
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            if self.use_mixed_precision and scaler is not None:
                with autocast(dtype=torch.float16):
                    loss = self._compute_loss(model, inputs, targets, criterion)
            else:
                loss = self._compute_loss(model, inputs, targets, criterion)
            
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

        return total_loss / max(1, len(dataloader))


    def collate_fn(self, batch):
        """Collate function for DataLoader with proper padding."""
        input_sequences, output_sequences = zip(*batch)
        
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


    def performMainTrain(self):
        """Main training loop with proper error handling and model checkpointing."""
        
        try:
            # Use the cached dataset
            loaded_dataset = self.loaded_dataset

            logger.info(f"Dataset size: {len(loaded_dataset)} samples")
            logger.info(f"Dataset columns: {loaded_dataset.column_names}")

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

            # Create iterable dataset for training pairs without materializing all pairs
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

            iterable_dataset = TokenPairIterableDataset(token_pair_generator)

            # Create DataLoader with custom collate function
            logger.info(f"Creating DataLoader (batch_size={TRAINING_CONFIG['batch_size']})...")
            pin_memory = self.use_gpu  # Only use pin_memory if GPU is available
            num_workers = self._limit_num_workers_by_memory(max(1, min(2, mp.cpu_count() // 2)))

            dataloader = DataLoader(
                iterable_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=True,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                num_workers=num_workers
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
            logger.info(f"Initializing ChatModel (embed_size={TRAINING_CONFIG['embed_size']}, hidden_size={TRAINING_CONFIG['hidden_size']})...")
            model = ChatModel(self.tokenizer, embed_size=TRAINING_CONFIG['embed_size'], hidden_size=TRAINING_CONFIG['hidden_size'])
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
            if TRAINING_CONFIG['warm_up_warmup']:
                logger.info(f"\nStarting warm-up phase (1 epoch)...")
                _ = self.train(model, dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                logger.info("✓ Warm-up phase completed")

            # Main training loop
            logger.info(f"\nStarting main training phase ({self.epochs} epochs)...")
            logger.info(f"{'='*80}")
            
            num_epochs = self.epochs
            for epoch in range(num_epochs):
                
                # Check for user interrupt
                if KEYBOARD_AVAILABLE:
                    try:
                        if keyboard.is_pressed('esc'):
                            logger.info("\n⚠ Training interrupted by user (ESC pressed)")
                            break
                    except:
                        pass  # Keyboard library may not work in all environments

                # Training
                loss = self.train(model, dataloader, criterion, optimizer, device, scaler, TRAINING_CONFIG['accumulation_steps'])
                
                # Update learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Memory and performance information
                if self.use_gpu:
                    gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {loss:.4f} | LR: {current_lr:.2e} | GPU: {gpu_memory:.2f}GB / {gpu_memory_reserved:.2f}GB")
                else:
                    logger.info(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {loss:.4f} | LR: {current_lr:.2e}")

                # Save best model checkpoint
                if loss < self.best_loss:
                    self.best_loss = loss
                    logger.info(f"  ✓ New best loss! Saving checkpoint to {BEST_MODEL_FILE}")
                    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                    }, BEST_MODEL_FILE)

            # Save final model and tokenizer
            logger.info(f"\n{'='*80}")
            logger.info("Training completed! Saving final models...")
            torch.save(model.state_dict(), LATEST_MODEL_FILE)
            torch.save(self.tokenizer, 'tokenizer.pth')
            torch.save(self.tokenizer.embedding.weight.data, 'pretrained_embeddings.pth')
            logger.info("✓ Models saved successfully")
        
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            sys.exit(1)