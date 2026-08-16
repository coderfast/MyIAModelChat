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

# Force CPU-only mode BEFORE torch import to prevent CUDA context initialization
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

try:
    import psutil
except ImportError:
    psutil = None

import torch

# Import from new module structure
from commons.registry.model_registry import list_models_cli, display_model_info
from commons.registry.model_merge import merge_from_names, parse_merge_spec
from commons.registry.model_export import export_cli
from dataset_preparer.data_preparer import prepare_datasets_for_training, DataPreparer
from commons.utils.device_utils import enumerate_gpus
from training.trainer import Trainer, TrainingConfig
from inference.chat_engine import ChatEngine, ChatConfig

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Setup logging with colors
class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output."""
    COLORS = {
        'INFO': '',              # White/default
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m',  # Red
        'DEBUG': '\033[90m',     # Gray
    }
    SUCCESS_COLOR = '\033[92m'   # Green for [OK] messages
    SYMBOLS = {
        'ERROR': '[ERROR] ',
        'CRITICAL': '[CRITICAL] ',
        'WARNING': '[WARN] ',
        'INFO': '',
        'DEBUG': '',
    }
    RESET = '\033[0m'

    def format(self, record):
        reset = self.RESET
        symbol = self.SYMBOLS.get(record.levelname, '')

        # Green for messages starting with [OK]
        if record.levelname == 'INFO' and str(record.msg).startswith('[OK]'):
            color = self.SUCCESS_COLOR
        else:
            color = self.COLORS.get(record.levelname, '')

        record.msg = f"{color}{symbol}{record.msg}{reset}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    root_logger.addHandler(handler)
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
    if not (args.train or args.chat or args.prepare_data or args.clear_cache or args.list_models or args.model_info or args.export):
        return False, "Please specify: --train, --chat, --prepare-data, --clear-cache, --list-models, --model-info, or --export"

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
        return False, "--num-cores and --num-threads must be >= 0"

    if args.max_ram_fraction < 0 or args.max_ram_fraction > 1:
        return False, "--max-ram-fraction must be between 0 and 1"

    # Device mutual exclusion — only one of --cpu, --gpu, --cpu+gpu, --vulkan
    device_flags = []
    if args.cpu:
        device_flags.append('--cpu')
    if args.gpu is not None:
        device_flags.append('--gpu')
    if args.cpu_gpu is not None:
        device_flags.append('--cpu+gpu')
    if args.vulkan:
        device_flags.append('--vulkan')
    if len(device_flags) > 1:
        return False, f"Only one device flag allowed at a time: {', '.join(device_flags)}"

    # Validate GPU indices
    if args.gpu is not None and args.gpu != '':
        try:
            device_ids = [int(x) for x in args.gpu.split(',')]
            if any(d < 0 for d in device_ids):
                raise ValueError
        except Exception:
            return False, "--gpu must be indices like '0' or '0,1'"

    if args.cpu_gpu is not None:
        try:
            device_ids = [int(x) for x in args.cpu_gpu.split(',')]
            if any(d < 0 for d in device_ids):
                raise ValueError
        except Exception:
            return False, "--cpu+gpu must be indices like '0' or '0,1'"

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
    
    logger.info("=" * 80)
    logger.info("CPU CONFIGURATION")
    logger.info("=" * 80)
    
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
    logger.info("=" * 80)


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
  --quantization TYPE  GGUF quantization: f32, f16, q4_0, q4_k_m, q5_k_m, q6_k, q8_0, etc. (default: q8_0)

DATA SOURCES (required with --prepare-data only):
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

DEVICE SELECTION:
  --cpu                Force CPU-only execution (disable GPU)
  --gpu [N,N,...]      Use GPU (no args=default GPU, or specific indices like 0,1)
  --vulkan             Force Vulkan backend (if available)
  --gpu-enum           Enumerate available GPUs and exit
  --cpu+gpu N,N,...    CPU+GPU hybrid: split layers by VRAM (e.g. --cpu+gpu 0,1)

CPU CONFIGURATION:
  --num-cores NUM       CPU cores for processing (default: {default_num_cores}, 0=all)
  --num-threads NUM     Threads per worker (default: {default_num_threads}, 0=all)

STATISTICS:
  --statistics          Show detailed per-step timing during training

EXAMPLES:
  # Prepare data (create/update cache)
  python main.py --prepare-data --aiml --hf --pdf --epub
  python main.py --prepare-data --aiml --bpe-vocab-size 8000
  python main.py --prepare-data --web  (scrapes URLs from datasets_source/web/urls.txt)

  # Train model (always uses cached dataset from prepare-data)
  python main.py --train --epochs 10
  python main.py --train --checkpoint-name ciencias_naturales --epochs 10
  python main.py --train --cpu --num-cores 4 --num-threads 4
  python main.py --train --gpu 0 --epochs 10
  python main.py --train --gpu 0,1 --epochs 10       (DDP multi-GPU)
  python main.py --train --cpu+gpu 0 --epochs 10     (CPU+GPU hybrid)

  python main.py --chat --model ciencias_naturales
  python main.py --chat --model ciencias_naturales+programacion
  python main.py --chat --cpu --num-cores 4 --num-threads 4

  python main.py --gpu-enum
  python main.py --list-models
  python main.py --model-info chat_model
  python main.py --export ciencias_naturales --formats gguf,onnx,onnx_int8
  python main.py --export chat_model --formats gguf --quantization q4_k_m
  python main.py --export chat_model --formats onnx --onnx-quant-type int8 --onnx-static
  python main.py --export chat_model --formats onnx --onnx-quant-type fp8_e4m3fn --onnx-static
            """
        )
        
        # Operation arguments
        parser.add_argument("--train", action='store_true', help="Train the model")
        parser.add_argument("--chat", action='store_true', help="Run chat interface")
        parser.add_argument("--prepare-data", action='store_true', help="Prepare datasets only")
        parser.add_argument("--clear-cache", action='store_true', help="Clear cached datasets")
        parser.add_argument("--list-models", action='store_true', help="List available trained models")
        parser.add_argument("--model-info", type=str, default=None, metavar='NAME',
                            help="Show detailed layer info for a specific model")
        parser.add_argument("--export", type=str, default=None, help="Export model to GGUF/ONNX (name or name+name for merge)")
        parser.add_argument("--formats", type=str, default="gguf,onnx",
                            help="Export formats: gguf, onnx, onnx_int8, onnx_uint8, onnx_int4, onnx_uint4, "
                                 "onnx_static_int8, onnx_static_int4, onnx_fp8, onnx_fp8_mixed, "
                                 "onnx_per_channel, onnx_mixed (default: gguf,onnx)")
        parser.add_argument("--quantization", type=str, default="q8_0",
                            choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
                                     "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
                                     "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq2_s",
                                     "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"],
                            help="GGUF quantization type (default: q8_0)")
        parser.add_argument("--onnx-quant-type", type=str, default="int8",
                            choices=["int8", "uint8", "int4", "uint4",
                                     "static_int8_qdq", "static_int8_qoperator",
                                     "static_int4_qoperator", "static_int4_qdq",
                                     "fp8_e4m3fn", "fp8_e5m2", "fp8_mixed",
                                     "fp8_e4m3fnuz", "fp8_e5m2fnuz",
                                     "per_channel_int8", "per_channel_int4",
                                     "mixed_int8_int4", "mixed_fp8_int8", "tensor_overrides"],
                            help="ONNX quantization type (default: int8)")
        parser.add_argument("--onnx-static", action='store_true',
                            help="Use static quantization for ONNX (default: dynamic)")
        parser.add_argument("--onnx-per-channel", action='store_true',
                            help="Use per-channel quantization for ONNX")
        parser.add_argument("--onnx-block-size", type=int, default=128,
                            help="Block size for INT4/UINT4 ONNX quantization (default: 128)")

        # Model library arguments
        parser.add_argument("--model", type=str, default=None, help="Model name to load for chat (e.g. ciencias_naturales)")
        parser.add_argument("--checkpoint-name", type=str, default="chat_model", help="Name for saved checkpoint (default: chat_model)")
        parser.add_argument("--dataset", type=str, default="dataset_cache", help="Path to dataset directory (default: dataset_cache)")
        
        # CPU configuration
        parser.add_argument("--num-cores", type=int, default=default_num_cores, help=f"CPU cores (default: {default_num_cores})")
        parser.add_argument("--num-threads", type=int, default=default_num_threads, help=f"Threads (default: {default_num_threads})")
        
        # Data sources
        parser.add_argument("--aiml", action='store_true', help="Include AIML data")
        parser.add_argument("--hf", action='store_true', help="Include HuggingFace data")
        parser.add_argument("--pdf", action='store_true', help="Include PDF data")
        parser.add_argument("--epub", action='store_true', help="Include EPUB data")
        parser.add_argument("--web", action='store_true', help="Include web documentation data")
        parser.add_argument("--csv", action='store_true', help="Include CSV data")
        parser.add_argument("--tagengo", action='store_true', help="Include Tagengo-GPT4 multilingual conversations (EN/ES)")
        parser.add_argument("--web-url", type=str, default=None,
                            help="Seed URL to scrape (reads from datasets_source/web/urls.txt if not set)")
        parser.add_argument("--web-max-pages", type=int, default=50,
                            help="Maximum pages to scrape per URL (default: 50)")
        parser.add_argument("--web-max-depth", type=int, default=3,
                            help="Maximum link-following depth (default: 3)")
        
        # Training options
        parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
        parser.add_argument("--refresh-cache", action='store_true', help="Rebuild cache (for --prepare-data)")
        parser.add_argument("--statistics", action='store_true', help="Show detailed per-step timing statistics during training")
        parser.add_argument("--bpe-vocab-size", type=int, default=8000, help="Vocabulary size for BPE tokenizer (default: 8000)")
        parser.add_argument("--pretrained", action='store_true',
                            help="Initialize model with pre-trained GPT-2 weights from HuggingFace")

        # Device selection
        parser.add_argument("--cpu", action='store_true', help="Force CPU-only execution (disable GPU)")
        parser.add_argument("--gpu", type=str, nargs='?', const='', default=None,
                            help="Use GPU: no arg=auto default GPU, or indices like '0' or '0,1'")
        parser.add_argument("--vulkan", action='store_true', help="Force Vulkan backend (if available)")
        parser.add_argument("--gpu-enum", action='store_true', help="Enumerate available GPUs and exit")
        parser.add_argument("--cpu+gpu", type=str, default=None, dest='cpu_gpu',
                            help="CPU+GPU hybrid mode: split layers by VRAM (e.g. '0' or '0,1')")

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
        parser.add_argument("--generate-agent-data", action='store_true',
                            help="Generate agentic data with tool calls during preparation")
        parser.add_argument("--agent-ratio", type=float, default=0.3,
                            help="Ratio of agentic samples in dataset (default: 0.3)")
        parser.add_argument("--agent-enabled", action='store_true',
                            help="Enable agentic mode during chat/training")
        parser.add_argument("--agent-show-tool-calls", action='store_true', default=True,
                            help="Show tool calls during chat (default: True)")

        # MoE (Mixture of Experts) configuration
        parser.add_argument("--moe-enabled", action='store_true',
                            help="Enable Mixture of Experts architecture")
        parser.add_argument("--moe-num-experts", type=int, default=4,
                            help="Number of experts per MoE layer (default: 4)")
        parser.add_argument("--moe-top-k", type=int, default=2,
                            help="Number of experts to route each token to (default: 2)")
        parser.add_argument("--moe-load-balance-weight", type=float, default=0.01,
                            help="Weight for load balancing loss (default: 0.01)")
        parser.add_argument("--moe-freeze-attention", action='store_true',
                            help="Freeze attention layers during MoE training")
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


        # Contamination filtering options (NEW)
        parser.add_argument("--filter-noise", action='store_true',
                            help="Enable noise filter (URLs, emails, code, boilerplate)")
        parser.add_argument("--noise-categories", type=str, default=None,
                            help="Noise categories to filter (comma-separated: urls,emails,phones,paths,code,boilerplate,corruption)")
        parser.add_argument("--filter-contamination", action='store_true',
                            help="Enable extended quality filter (low alpha, repetition, diversity)")
        parser.add_argument("--filter-dedup", action='store_true',
                            help="Enable cross-source deduplication")
        parser.add_argument("--dedup-mode", type=str, default='all',
                            choices=['exact', 'near', 'cross', 'all'],
                            help="Deduplication mode (default: all)")
        parser.add_argument("--filter-balance", action='store_true',
                            help="Enable source balance control")
        parser.add_argument("--max-source-ratio", type=float, default=0.3,
                            help="Max fraction per source (default: 0.3)")
        parser.add_argument("--filter-leakage", action='store_true',
                            help="Enable data leakage detection")
        parser.add_argument("--leakage-threshold", type=float, default=0.5,
                            help="Leakage overlap threshold (default: 0.5)")
        parser.add_argument("--audit-report", action='store_true',
                            help="Generate audit report for filtering pipeline")
        parser.add_argument("--audit-dir", type=str, default=None,
                            help="Directory for audit reports (default: dataset_preparer/contamination/reports/)")
        
        args = parser.parse_args()

        # Force agent_ratio=0 for small models — can't learn agentic patterns
        # Default embed_size is 256 (< 300), so agentic is disabled unless user overrides
        if hasattr(args, 'agent_ratio') and args.agent_ratio > 0:
            args.agent_ratio = 0.0
            args.agent_enabled = False
            logger.info("Agent ratio forced to 0.0 (model too small for agentic patterns)")

        # Handle --gpu-enum before validation
        if args.gpu_enum:
            gpus = enumerate_gpus()
            if not gpus:
                print("No GPUs detected.")
            else:
                print("=" * 60)
                print("GPU ENUMERATION")
                print("=" * 60)
                for g in gpus:
                    rec = "RECOMMENDED" if g['ai_recommended'] else "NOT RECOMMENDED"
                    print(f"[{g['index']}] {g['name']}")
                    print(f"    VRAM: {g['vram_total_gb']:.2f} GB ({g['vram_free_gb']:.2f} GB free)")
                    print(f"    Compute Capability: {g['compute_capability'][0]}.{g['compute_capability'][1]}")
                    print(f"    Status: {g['status']}")
                    print(f"    AI Level: {g['ai_level']} — {g['max_model_config']}")
                    if g['notes']:
                        print(f"    Notes: {g['notes']}")
                    print()
                print("=" * 60)
            sys.exit(0)

        # Resolve device mode from CLI args
        if args.cpu:
            args.device_mode = 'cpu'
            args.gpu_indices = None
            args.use_vulkan = False
        elif args.cpu_gpu is not None:
            args.device_mode = 'cpu+gpu'
            args.gpu_indices = [int(x) for x in args.cpu_gpu.split(',')]
            args.use_vulkan = False
        elif args.gpu is not None:
            if args.gpu == '':
                args.device_mode = 'gpu'
                args.gpu_indices = None  # auto-detect
            else:
                args.device_mode = 'gpu'
                args.gpu_indices = [int(x) for x in args.gpu.split(',')]
            args.use_vulkan = False
        elif args.vulkan:
            args.device_mode = 'gpu'
            args.gpu_indices = None
            args.use_vulkan = True
        else:
            args.device_mode = 'auto'
            args.gpu_indices = None
            args.use_vulkan = False

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

        # Handle --model-info
        if args.model_info:
            display_model_info(args.model_info)
            sys.exit(0)

        # Handle --export
        if args.export:
            formats = [f.strip() for f in args.formats.split(',')]
            export_cli(
                args.export,
                formats,
                quantization=args.quantization,
                onnx_quant_type=args.onnx_quant_type,
                onnx_static=args.onnx_static,
                onnx_per_channel=args.onnx_per_channel,
                onnx_block_size=args.onnx_block_size,
            )
            sys.exit(0)

        # Garbage collection
        gc.collect()
        
        # Display system information
        logger.info("=" * 80)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 80)
        logger.info(f"Total available CPU cores: {system_cpu_count}")
        logger.info(f"Specified --num-cores: {args.num_cores}")
        logger.info(f"Specified --num-threads: {args.num_threads}")
        logger.info("=" * 80)
        
        # Handle device mode environment setup
        if args.device_mode == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.info("CPU-only mode enabled (GPU disabled)")
        elif args.gpu_indices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu_indices)
            logger.info(f"CUDA device(s) forced: {args.gpu_indices}")
        
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
                logger.info("[OK] Dataset preparation completed!")
                logger.info(f"Total samples: {stats.get('total_samples', 0):,}")
                if stats.get('from_cache'):
                    logger.info("(Loaded from cache)")
            else:
                logger.error("Dataset preparation failed!")
                sys.exit(1)
        
        # Train
        if args.train:
            # Detect DDP from torchrun environment
            if 'RANK' in os.environ:
                # Executed with torchrun - read config from environment
                args.rank = int(os.environ['RANK'])
                args.local_rank = int(os.environ['LOCAL_RANK'])
                args.world_size = int(os.environ['WORLD_SIZE'])
                args.master_addr = os.environ.get('MASTER_ADDR', 'localhost')
                args.master_port = int(os.environ.get('MASTER_PORT', 29500))
                logger.info(f"DDP detected: rank={args.rank}, local_rank={args.local_rank}, world_size={args.world_size}")
            else:
                # Single-process mode
                args.rank = 0
                args.local_rank = 0
                args.world_size = len(args.gpu_indices) if args.gpu_indices and len(args.gpu_indices) > 1 else 1
                args.master_addr = 'localhost'
                args.master_port = 29500

            logger.info("Initializing training...")
            config = TrainingConfig(
                epochs=args.epochs,
                checkpoint_name=args.checkpoint_name,
                dataset_source=args.dataset,
                device_mode=args.device_mode,
                gpu_indices=args.gpu_indices,
                use_vulkan=args.use_vulkan,
                num_cores=args.num_cores,
                num_threads=args.num_threads,
                max_ram_fraction=args.max_ram_fraction,
                max_ram_bytes=getattr(args, 'max_ram_bytes', None),
                pretrained=getattr(args, 'pretrained', False),
                thinking_loss_weight=args.thinking_loss_weight,
                thinking_enabled=args.thinking_enabled,
                thinking_max_tokens=args.thinking_max_tokens,
                statistics=args.statistics,
                agent_enabled=getattr(args, 'agent_enabled', False),
                agent_loss_weight=getattr(args, 'agent_loss_weight', 1.0),
                agent_ratio=getattr(args, 'agent_ratio', 0.3),
                moe_enabled=getattr(args, 'moe_enabled', False),
                moe_num_experts=getattr(args, 'moe_num_experts', 4),
                moe_top_k=getattr(args, 'moe_top_k', 2),
                moe_load_balance_weight=getattr(args, 'moe_load_balance_weight', 0.01),
                moe_freeze_attention=getattr(args, 'moe_freeze_attention', False),
                rank=args.rank,
                local_rank=args.local_rank,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
            trainer = Trainer(config)
            training_thread = threading.Thread(target=trainer.performMainTrain, name="TrainingThread", daemon=False)
            training_thread.start()
            try:
                while training_thread.is_alive():
                    training_thread.join(timeout=30)
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
                device_mode=args.device_mode,
                gpu_indices=args.gpu_indices,
                use_vulkan=args.use_vulkan,
                show_thinking=args.show_thinking,
                thinking_enabled=args.thinking_enabled,
                thinking_max_tokens=args.thinking_max_tokens,
                agent_enabled=getattr(args, 'agent_enabled', False),
                agent_max_iterations=getattr(args, 'agent_max_iterations', 5),
                agent_show_tool_calls=getattr(args, 'agent_show_tool_calls', True),
            )
            engine = ChatEngine(config)
            engine.start_chat_loop()
        
        logger.info("=" * 80)
        logger.info("[OK] Program completed successfully")
        logger.info("=" * 80)
    
    except KeyboardInterrupt:
        logger.warning("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
