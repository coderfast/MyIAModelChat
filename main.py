"""
MyIAModelChat - Main Entry Point

This is the only CLI entry point for the application.
All operations (training, chat, data preparation, etc.) are controlled from here.
"""

import os
import gc
import argparse
import sys
import threading
import logging
import multiprocessing as mp
from config import OLLAMA_MODEL

try:
    import psutil
except ImportError:
    psutil = None

import torch

# Import from new module structure
from commons.registry.model_registry import list_models_cli
from commons.registry.model_merge import merge_from_names, parse_merge_spec
from commons.registry.model_export import export_cli
from dataset_preparer.data_preparer import prepare_datasets_for_training, DataPreparer
from training.trainer import Trainer, TrainingConfig
from inference.chat_engine import ChatEngine, ChatConfig

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
SYSTEM_CONFIG = {
    'default_cores_fraction': 0.5,
    'min_cores': 1,
    'min_threads': 1,
    'max_ram_fraction': 0.75,
}


def validate_arguments(args):
    """Validate command-line arguments for consistency."""
    if not (args.train or args.chat or args.prepare_data or args.clear_cache or args.list_models or args.export):
        return False, "Please specify: --train, --chat, --prepare-data, --clear-cache, --list-models, or --export"

    if args.prepare_data and not (args.aiml or args.hf or args.pdf or args.epub or args.web):
        return False, "Specify data source for --prepare-data: --aiml, --hf, --pdf, --epub, or --web"

    if args.train:
        cache_path = os.path.join('dataset_cache', 'prepared_dataset')
        if not os.path.exists(cache_path):
            return False, (
                "No cached dataset found for training.\n\n"
                "Prepare your dataset first:\n"
                "  python main.py --prepare-data --aiml\n"
                "  python main.py --prepare-data --aiml --hf --pdf --epub\n"
                "  python main.py --prepare-data --aiml --bpe-vocab-size 8000\n\n"
                "Run 'python main.py --prepare-data --help' for all options."
            )

    if args.num_cores < 0 or args.num_threads < 0:
        return False, "--num_cores and --num_threads must be >= 0"

    if args.max_ram_fraction < 0 or args.max_ram_fraction > 1:
        return False, "--max-ram-fraction must be between 0 and 1"

    if args.cuda_device is not None:
        try:
            device_ids = [int(x) for x in str(args.cuda_device).split(',')]
            if any(d < 0 for d in device_ids):
                raise ValueError
        except Exception:
            return False, "--cuda-device must be kernel indices like '0' or '0,1'"

    if args.epochs < 1 and args.train:
        return False, "--epochs must be >= 1"
    
    return True, None


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
    try:
        torch.set_num_interop_threads(args.num_cores)
    except RuntimeError:
        pass
    
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


if __name__ == '__main__':
    try:
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

        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description='MyIAModelChat - AI Chat Model Training and Inference',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
OPERATIONS:
  --train              Train the neural network model
  --chat               Run interactive chat interface
  --prepare-data       Prepare and validate datasets only
  --clear-cache        Clear cached datasets and exit
  --list-models        List available trained models
  --export NAME        Export model to GGUF/ONNX (use NAME+NAME for merge)

MODEL LIBRARY:
  --model NAME         Load specific model for chat (e.g. ciencias_naturales)
  --checkpoint-name N  Name for saved checkpoint (default: chat_model)
  --dataset PATH       Path to dataset directory (default: dataset_cache)
  --formats F1,F2      Export formats: gguf, onnx, onnx_int8 (default: gguf,onnx)

DATA SOURCES (required with --train or --prepare-data):
  --aiml               Include AIML data from datasets_source/aiml directory
  --hf                 Include Hugging Face datasets
  --pdf                Include PDF data from datasets_source/pdf directory
  --epub               Include EPUB data from datasets_source/epub directory
  --web                Include web documentation data (scrapes from URL)
  --web-url URL        Seed URL to scrape (reads datasets_source/web/urls.txt if not set)
  --web-max-pages N    Max pages per URL (default: 50)
  --web-max-depth N    Max link-following depth (default: 3)

TRAINING OPTIONS:
  --epochs NUM         Number of training epochs (default: 1)
  --use-cpuonly        Force CPU-only execution (disable GPU)

CPU CONFIGURATION:
  --num_cores NUM      CPU cores for processing (default: {default_num_cores}, 0=all)
  --num_threads NUM    Threads per worker (default: {default_num_threads}, 0=all)

EXAMPLES:
  # Prepare data (create/update cache)
  python main.py --prepare-data --aiml --hf --pdf --epub
  python main.py --prepare-data --aiml --bpe-vocab-size 8000
  python main.py --prepare-data --web  (scrapes URLs from datasets_source/web/urls.txt)

  # Train model (uses cached dataset)
  python main.py --train --epochs 10
  python main.py --train --dataset datasets_source/ciencias/ --checkpoint-name ciencias_naturales --epochs 10
  python main.py --train --use-cpuonly --num_cores 4 --num_threads 4

  python main.py --chat --model ciencias_naturales
  python main.py --chat --model ciencias_naturales+programacion
  python main.py --chat --use-cpuonly --num_cores 4 --num_threads 4

  python main.py --list-models
  python main.py --export ciencias_naturales --formats gguf,onnx,onnx_int8
            """
        )
        
        # Operation arguments
        parser.add_argument("--train", action='store_true', help="Train the model")
        parser.add_argument("--chat", action='store_true', help="Run chat interface")
        parser.add_argument("--prepare-data", action='store_true', help="Prepare datasets only")
        parser.add_argument("--clear-cache", action='store_true', help="Clear cached datasets")
        parser.add_argument("--list-models", action='store_true', help="List available trained models")
        parser.add_argument("--export", type=str, default=None, help="Export model to GGUF/ONNX (name or name+name for merge)")
        parser.add_argument("--formats", type=str, default="gguf,onnx", help="Export formats (default: gguf,onnx)")

        # Model library arguments
        parser.add_argument("--model", type=str, default=None, help="Model name to load for chat (e.g. ciencias_naturales)")
        parser.add_argument("--checkpoint-name", type=str, default="chat_model", help="Name for saved checkpoint (default: chat_model)")
        parser.add_argument("--dataset", type=str, default="dataset_cache", help="Path to dataset directory (default: dataset_cache)")
        
        # CPU configuration
        parser.add_argument("--num_cores", type=int, default=default_num_cores, help=f"CPU cores (default: {default_num_cores})")
        parser.add_argument("--num_threads", type=int, default=default_num_threads, help=f"Threads (default: {default_num_threads})")
        
        # Data sources
        parser.add_argument("--aiml", action='store_true', help="Include AIML data")
        parser.add_argument("--hf", action='store_true', help="Include HuggingFace data")
        parser.add_argument("--pdf", action='store_true', help="Include PDF data")
        parser.add_argument("--epub", action='store_true', help="Include EPUB data")
        parser.add_argument("--web", action='store_true', help="Include web documentation data")
        parser.add_argument("--csv", action='store_true', help="Include CSV data")
        parser.add_argument("--web-url", type=str, default=None,
                            help="Seed URL to scrape (reads from datasets_source/web/urls.txt if not set)")
        parser.add_argument("--web-max-pages", type=int, default=50,
                            help="Maximum pages to scrape per URL (default: 50)")
        parser.add_argument("--web-max-depth", type=int, default=3,
                            help="Maximum link-following depth (default: 3)")
        
        # Training options
        parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
        parser.add_argument("--use-cache", action='store_true', help="Load cached dataset (for --prepare-data)")
        parser.add_argument("--refresh-cache", action='store_true', help="Rebuild cache (for --prepare-data)")
        parser.add_argument("--use-cpuonly", action='store_true', help="CPU-only execution")
        parser.add_argument("--bpe-vocab-size", type=int, default=8000, help="Vocabulary size for BPE tokenizer (default: 8000)")
        parser.add_argument("--cuda-device", type=str, default=None,
                            help="CUDA device index(es), e.g. '0' or '0,1'; ignored with --use-cpuonly")
        parser.add_argument("--max-ram-fraction", type=float, default=SYSTEM_CONFIG['max_ram_fraction'],
                            help="Maximum fraction of total RAM to use (0-1, default 0.75)")
        parser.add_argument("--show-thinking", action='store_true', help="Show <thinking> reasoning in chat")

        # Thinking configuration
        parser.add_argument("--thinking-loss-weight", type=float, default=0.5,
                            help="Loss weight for thinking tokens (0.0-1.0, default: 0.5)")
        parser.add_argument("--thinking-enabled", action='store_true', default=True,
                            help="Enable thinking generation (default: True)")
        parser.add_argument("--thinking-no-enabled", dest='thinking_enabled', action='store_false',
                            help="Disable thinking generation")
        parser.add_argument("--thinking-max-tokens", type=int, default=64,
                            help="Max tokens for thinking phase (default: 64)")
        parser.add_argument("--generate-thinking", action='store_true',
                            help="Generate real thinking data during preparation")
        parser.add_argument("--thinking-mode", type=str, default='nlp',
                            choices=['nlp', 'template', 'hf', 'ollama'],
                            help="Mode for thinking generation (default: nlp - ThinkingEngine)")
        parser.add_argument("--thinking-model", type=str, default=OLLAMA_MODEL,
                            help=f"Teacher model for thinking generation (default: {OLLAMA_MODEL})")
        parser.add_argument("--thinking-depth", type=str, default='adaptive',
                            choices=['basic', 'adaptive', 'detailed'],
                            help="Thinking depth level (default: adaptive)")
        parser.add_argument("--thinking-ollama", action='store_true',
                            help="Use Ollama teacher for enhanced thinking (optional)")
        parser.add_argument("--validate-sources", action='store_true',
                            help="Validate and clean each data source before training")

        # Text processing options
        parser.add_argument("--enable-chunking", action='store_true',
                            help="Enable text chunking by tokens (for PDF/EPUB)")
        parser.add_argument("--chunk-max-tokens", type=int, default=512,
                            help="Maximum tokens per chunk when chunking is enabled (default: 512)")
        parser.add_argument("--chunk-overlap", type=int, default=50,
                            help="Number of overlapping tokens between chunks (default: 50)")
        parser.add_argument("--enable-dedup", action='store_true',
                            help="Enable deduplication of similar texts")
        parser.add_argument("--dedup-threshold", type=float, default=0.8,
                            help="Similarity threshold for deduplication (0-1, default: 0.8)")
        parser.add_argument("--enable-quality-filter", action='store_true',
                            help="Enable quality filtering of texts")
        parser.add_argument("--min-words", type=int, default=5,
                            help="Minimum words per text for quality filter (default: 5)")
        parser.add_argument("--max-words", type=int, default=1000,
                            help="Maximum words per text for quality filter (default: 1000)")
        parser.add_argument("--preserve-metadata", action='store_true',
                            help="Preserve document metadata (title, author, etc.) in dataset")
        parser.add_argument("--enable-lang-filter", action='store_true',
                            help="Enable language filtering")
        parser.add_argument("--allowed-languages", type=str, nargs='+', default=['es', 'en'],
                            help="Allowed language codes for filtering (default: es en)")
        
        args = parser.parse_args()

        # Validate arguments
        is_valid, error_msg = validate_arguments(args)
        if not is_valid:
            logger.error(f"Argument error: {error_msg}")
            parser.print_help()
            sys.exit(1)

        # Handle --list-models
        if args.list_models:
            list_models_cli()
            sys.exit(0)

        # Handle --export
        if args.export:
            formats = [f.strip() for f in args.formats.split(',')]
            export_cli(args.export, formats)
            sys.exit(0)

        # Garbage collection
        gc.collect()
        
        # Display system information
        logger.info(f"\n{'='*80}")
        logger.info("SYSTEM INFORMATION")
        logger.info(f"{'='*80}")
        logger.info(f"Total available CPU cores: {system_cpu_count}")
        logger.info(f"Specified --num_cores: {args.num_cores}")
        logger.info(f"Specified --num_threads: {args.num_threads}")
        logger.info(f"{'='*80}\n")
        
        # Handle CPU-only mode or explicit CUDA device selection
        if args.use_cpuonly:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.info("CPU-only mode enabled (GPU disabled)")
        elif args.cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
            logger.info(f"CUDA device(s) forced: {args.cuda_device}")
        
        # Setup CPU configuration
        setup_cpu_configuration(args)
        
        # Clear cache if requested
        if args.clear_cache:
            preparer = DataPreparer(args)
            preparer._clear_cache()
            logger.info("Cache cleared successfully")
            sys.exit(0)
        
        # Prepare data only when explicitly requested
        if args.prepare_data:
            logger.info("Preparing datasets...")
            dataset, stats = prepare_datasets_for_training(args)
            if dataset is not None:
                logger.info("Dataset preparation completed!")
                logger.info(f"Total samples: {stats.get('total_samples', 0):,}")
                if stats.get('from_cache'):
                    logger.info("(Loaded from cache)")
            else:
                logger.error("Dataset preparation failed!")
                sys.exit(1)
        
        # Train
        if args.train:
            logger.info("Initializing training...")
            config = TrainingConfig(
                epochs=args.epochs,
                checkpoint_name=args.checkpoint_name,
                dataset_source=args.dataset,
                use_cpuonly=args.use_cpuonly,
                cuda_device=args.cuda_device,
                num_cores=args.num_cores,
                num_threads=args.num_threads,
                max_ram_fraction=args.max_ram_fraction,
                max_ram_bytes=getattr(args, 'max_ram_bytes', None),
                thinking_loss_weight=args.thinking_loss_weight,
                thinking_enabled=args.thinking_enabled,
                thinking_max_tokens=args.thinking_max_tokens,
            )
            trainer = Trainer(config)
            training_thread = threading.Thread(target=trainer.performMainTrain, name="TrainingThread", daemon=True)
            training_thread.start()
            try:
                while training_thread.is_alive():
                    training_thread.join(timeout=1)
                    if KEYBOARD_AVAILABLE:
                        try:
                            if keyboard.is_pressed('esc'):
                                logger.warning("\nTraining interrupted by user (ESC pressed)")
                                trainer.request_stop()
                                training_thread.join(timeout=10)
                                if training_thread.is_alive():
                                    logger.warning("Training thread did not stop within timeout")
                                sys.exit(0)
                        except Exception:
                            pass
            except KeyboardInterrupt:
                logger.warning("\nTraining interrupted by user (Ctrl+C)")
                trainer.request_stop()
                training_thread.join(timeout=10)
                if training_thread.is_alive():
                    logger.warning("Training thread did not stop within timeout")
                sys.exit(0)

        # Chat
        if args.chat:
            # Handle model merging if + is in model name
            if args.model and '+' in args.model:
                names, weights = parse_merge_spec(args.model)
                logger.info(f"Merging models: {names}")
                merged_path = merge_from_names(names, weights)
                args.model = merged_path.replace('.pth', '')

            logger.info("Starting chat interface...")
            config = ChatConfig(
                model_name=args.model,
                use_cpuonly=args.use_cpuonly,
                cuda_device=args.cuda_device,
                show_thinking=args.show_thinking,
                thinking_enabled=args.thinking_enabled,
                thinking_max_tokens=args.thinking_max_tokens,
            )
            engine = ChatEngine(config)
            engine.start_chat_loop()
        
        logger.info("\n" + "="*80)
        logger.info("Program completed successfully")
        logger.info("="*80)
    
    except KeyboardInterrupt:
        logger.warning("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
