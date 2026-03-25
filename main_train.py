import os
import argparse
import pickle
import sys
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from aimlloder import *
from dialogmanager import DialogueManager
from simpletokenizer import *
from chatmodel import *
from chatdataset import *
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = 'dataset_cache'
CACHE_DATASET_FILE = os.path.join(CACHE_DIR, 'prepared_dataset')
CACHE_STATS_FILE = os.path.join(CACHE_DIR, 'dataset_stats.pkl')


class MainTrain:
    
    def __init__(self, args):
        
        print(f"MainTrain initializing...")

        # Check if the tokenizer and cached dataset exist
        self.tokenizer = None
        self.tokenized_data = None
        self.merged_dataset = None
        
        # Device and training configuration tracking
        self.use_gpu = False
        self.use_mixed_precision = False
        self.use_gradient_checkpointing = False

        # Parse command-line arguments
        self.aiml = args.aiml
        self.hf = args.hf
        self.onlytokenize = args.onlytokenize
        self.epochs = args.epochs

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
            self.merged_dataset = Dataset.load_from_disk(CACHE_DATASET_FILE)
            logger.info(f"✓ Loaded cached dataset: {len(self.merged_dataset)} samples")
            
            # Load statistics if available
            if os.path.exists(CACHE_STATS_FILE):
                with open(CACHE_STATS_FILE, 'rb') as f:
                    stats = pickle.load(f)
                logger.info(f"Dataset statistics loaded")
                logger.info(f"  Total samples: {stats.get('total_samples', 0):,}")
            
        except Exception as e:
            logger.error(f"❌ Error loading cached dataset: {e}")
            sys.exit(1)

    def _setup_device_and_config(self):
        """Setup device configuration with CPU-GPU combined strategy and mixed precision support."""
        
        # Detect GPU availability
        cuda_available = torch.cuda.is_available()
        
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

    # Training function with mixed precision support
    def train(self, model, dataloader, criterion, optimizer, device, scaler=None):
        """Train function with support for mixed precision training."""
        
        total_loss = 0
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            # Use mixed precision if available
            if self.use_mixed_precision and scaler is not None:
                with autocast(dtype=torch.float16):
                    outputs = model(inputs)
                    
                    # Reshape outputs and targets
                    outputs = outputs.contiguous().view(-1, outputs.size(-1))
                    targets = targets.contiguous().view(-1)
                    
                    # Ignore padded elements
                    non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
                    outputs = outputs[non_pad_mask]
                    targets = targets[non_pad_mask]
                    
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(inputs)
                
                # Reshape outputs and targets
                outputs = outputs.contiguous().view(-1, outputs.size(-1))
                targets = targets.contiguous().view(-1)
                
                # Ignore padded elements
                non_pad_mask = targets.ne(self.tokenizer.get_pad_index())
                outputs = outputs[non_pad_mask]
                targets = targets[non_pad_mask]
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            total_loss += loss.item()
            
            # Memory cleanup every 10 batches on GPU
            if self.use_gpu and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        return total_loss / len(dataloader)


    # Collate function for DataLoader
    def collate_fn(self, batch):

        input_sequences, output_sequences = zip(*batch)

        # Find max lengths
        # max_input_len = max(len(seq) for seq in input_sequences)
        # max_output_len = max(len(seq) for seq in output_sequences)
        max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))

        # Pad sequences
        ## padded_inputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_input_len - len(seq)) for seq in input_sequences]
        ## padded_outputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_output_len - len(seq)) for seq in output_sequences]
        # padded_inputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in input_sequences]
        # padded_outputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in output_sequences]
        padded_inputs = [seq + [self.tokenizer.get_pad_index()] * (max_len - len(seq)) for seq in input_sequences]
        padded_outputs = [seq + [self.tokenizer.get_pad_index()] * (max_len - len(seq)) for seq in output_sequences]

        # Convert to tensors
        input_tensor = torch.LongTensor(padded_inputs)
        output_tensor = torch.LongTensor(padded_outputs)

        return input_tensor, output_tensor


    # def encode(self, text):
    #     if isinstance(text, str):
    #         return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]
    #     else:
    #         return [self.encode(str(item)) for item in text]


    def performMainTrain(self):

        # Use the cached dataset
        merged_dataset = self.merged_dataset

        print(f"Length of dataset: {len(merged_dataset)}")
        print(f"Dataset columns: {merged_dataset.column_names}")

        # Prepare data for PyTorch
        print(f"Preparando datos para PyTorch, codificando datos")
        
        # Extract text data from cached dataset
        all_texts = []
        for item in merged_dataset:
            text = item['input_ids']
            if isinstance(text, str) and len(text.strip()) > 0:
                all_texts.append(text)
        
        if not all_texts:
            logger.error("No valid text data found in cached dataset")
            sys.exit(1)
        
        logger.info(f"Extracted {len(all_texts)} text samples from cache")

        # Fit tokenizer on all texts
        self.tokenizer.fit(all_texts)
        logger.info("Tokenizer fitted on cached dataset")

        # Create input-output pairs for training
        # Use each text as both input and output for language model training
        encoded_data = []
        for i, text in enumerate(all_texts):
            encoded_text = self.tokenizer.encode(text)
            if encoded_text and len(encoded_text) > 1:
                # Input is all but last token, output is all but first token
                input_ids = encoded_text[:-1]
                output_ids = encoded_text[1:]
                encoded_data.append((input_ids, output_ids))
            
            if (i + 1) % max(1, len(all_texts) // 10) == 0:
                logger.info(f"  Encoded {i + 1}/{len(all_texts)} samples")
        
        logger.info(f"Created {len(encoded_data)} training pairs")
        
        if not encoded_data:
            logger.error("No valid encoded data generated")
            sys.exit(1)

        print(f"create dataloader")
        # Create DataLoader with custom collate function
        batch_size = 4  # Reduce from 32 to 4 (or 8 if it fits)
        dataloader = DataLoader(encoded_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn, pin_memory=True, num_workers=2)
        print(f"created dataloader")

        # CPU configuration already set in main.py entry point
        var_num_cores = os.environ.get("MKL_NUM_THREADS", mp.cpu_count())
        var_num_threads = os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())
        logger.info(f"Using CPU configuration - Cores: {var_num_cores}, Threads: {var_num_threads}")

        # Setup device with improved configuration for CPU-GPU combined training
        device = self._setup_device_and_config()

        # Initialize model with device strategy
        model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256)
        model = self._setup_model_with_device_strategy(model, device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_index())
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler() if self.use_mixed_precision else None
        
        if scaler:
            logger.info("✓ Gradient scaler initialized for mixed precision training")

        # Add gradient accumulation
        accumulation_steps = 8  # Simulate batch_size=32 (4 * 8)
        optimizer.zero_grad()

        logger.info(f"\nStarting pre-training warm-up phase...")
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if self.use_mixed_precision and scaler:
                with autocast(dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Memory cleanup
            if self.use_gpu and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # Training loop
        logger.info(f"\nStarting main training phase ({self.epochs} epochs)...")
        num_epochs = self.epochs
        for epoch in range(num_epochs):

            if keyboard.is_pressed('esc'):
                logger.info("Training interrupted by user")
                break

            loss = self.train(model, dataloader, criterion, optimizer, device, scaler)
            
            # Memory information
            if self.use_gpu:
                gpu_memory = torch.cuda.memory_allocated(device) / 1e9
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f} | GPU Memory: {gpu_memory:.2f}GB / {gpu_memory_reserved:.2f}GB (reserved)")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

            # Save the model, tokenizer, and pre-trained embeddings
            torch.save(model.state_dict(), 'chat_model.pth')
            torch.save(self.tokenizer, 'tokenizer.pth')
            torch.save(self.tokenizer.embedding.weight.data, 'pretrained_embeddings.pth')