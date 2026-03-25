import os
import gc
import argparse
import pickle
import sys
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from aimlloder import *
from dialogmanager import DialogueManager
from simpletokenizer import *
from chatmodel import *
from chatdataset import *
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import pipeline
from main_train import *
from main_chat import *
from data_preparer import prepare_datasets_for_training



# Main
if __name__ == '__main__':

    # Calculate default CPU configuration (half of available cores)
    system_cpu_count = mp.cpu_count()
    default_num_cores = max(1, system_cpu_count // 2)
    default_num_threads = max(1, system_cpu_count // 2)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='MyIAModelChat - AI Chat Model Training and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train --aiml --hf --epochs 10
  python main.py --train --aiml --hf --epochs 5 --num_cores 8 --num_threads 8
  python main.py --train --aiml --hf --epochs 10 --use-cache
  python main.py --prepare-data --aiml --hf
  python main.py --prepare-data --aiml --pdf
  python main.py --prepare-data --aiml --epub
  python main.py --chat
  
Performance Tips:
  - Use --use-cache after first data preparation for faster training starts
  - Adjust --num_cores and --num_threads based on your CPU capabilities
  - Default: Half of available system cores (currently {half_cores} cores)
  - Use explicit values to override: --num_cores 4 --num_threads 4
  - Use 0 for auto-detection (all available): --num_cores 0 --num_threads 0
        """.format(half_cores=default_num_cores)
    )
    parser.add_argument("--train", action='store_true', help="Train the model")
    parser.add_argument("--chat", action='store_true', help="Run chat interface")
    parser.add_argument("--prepare-data", action='store_true', help="Prepare and validate datasets only")
    parser.add_argument("--num_cores", type=int, default=default_num_cores, help=f"Number of CPU cores for parallel processing (default: {default_num_cores} - half of system, 0 for all)")
    parser.add_argument("--num_threads", type=int, default=default_num_threads, help=f"Number of threads per worker (default: {default_num_threads} - half of system, 0 for all)")
    parser.add_argument("--aiml", action='store_true', help="Include AIML data")
    parser.add_argument("--hf", action='store_true', help="Include Hugging Face datasets")
    parser.add_argument("--pdf", action='store_true', help="Include PDF data from 'pdfs' directory")
    parser.add_argument("--epub", action='store_true', help="Include EPUB data from 'epub' directory")
    parser.add_argument("--onlytokenize", action='store_true', help="Build vocabulary only")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--use-cache", action='store_true', help="Load cached dataset if available (skip data loading)")
    parser.add_argument("--refresh-cache", action='store_true', help="Rebuild cache from scratch")
    parser.add_argument("--clear-cache", action='store_true', help="Clear cached datasets and exit")
    args = parser.parse_args()

    # Garbage collector
    gc.collect()
    
    # Display system information
    print(f"\n{'='*80}")
    print(f"SYSTEM INFORMATION")
    print(f"{'='*80}")
    print(f"Total available CPU cores: {system_cpu_count}")
    print(f"Specified --num_cores: {args.num_cores}")
    print(f"Specified --num_threads: {args.num_threads}")
    print(f"{'='*80}\n")
    
    # Auto-detect CPU cores and threads if set to 0
    if args.num_cores == 0:
        args.num_cores = system_cpu_count
        print(f"✓ Auto-detected CPU cores (0 specified): {args.num_cores}")
    
    if args.num_threads == 0:
        args.num_threads = system_cpu_count
        print(f"✓ Auto-detected threads (0 specified): {args.num_threads}")

    # Setup CPU configuration globally for all components
    print(f"\n{'='*80}")
    print(f"CPU CONFIGURATION SETUP")
    print(f"{'='*80}")
    
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cores)
    torch.set_num_interop_threads(args.num_cores)
    
    print(f"OMP_NUM_THREADS (PyTorch): {args.num_threads}")
    print(f"MKL_NUM_THREADS (NumPy): {args.num_cores}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Available CPU count: {system_cpu_count}")
    print(f"{'='*80}\n")

    # Clear cache if requested
    if args.clear_cache:
        from data_preparer import CACHE_DIR, DataPreparer
        preparer = DataPreparer(args)
        preparer._clear_cache()
        print("✅ Cache cleared successfully")
        sys.exit(0)

    # Prepare Data
    if args.prepare_data:
        dataset, stats = prepare_datasets_for_training(args)
        if dataset is not None:
            print("\n✅ Dataset preparation completed!")
            print(f"Total samples: {stats.get('total_samples', 0):,}")
            if stats.get('from_cache'):
                print("(Loaded from cache)")
        else:
            print("\n❌ Dataset preparation failed!")
            sys.exit(1)
    
    # Train
    elif args.train:
        iMainTrain = MainTrain(args)
        iMainTrain.performMainTrain()

    # Chat
    if args.chat:
        iMainChat = MainChat(args)
        iMainChat.performMainChat()