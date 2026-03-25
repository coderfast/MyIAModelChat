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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='MyIAModelChat - AI Chat Model Training and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train --aiml --epochs 10
  python main.py --prepare-data --aiml --hf
  python main.py --prepare-data --aiml --hf --use-cache
  python main.py --chat
        """
    )
    parser.add_argument("--train", action='store_true', help="Train the model")
    parser.add_argument("--chat", action='store_true', help="Run chat interface")
    parser.add_argument("--prepare-data", action='store_true', help="Prepare and validate datasets only")
    parser.add_argument("--num_cores", type=int, default=4, help="Number of CPU cores (default: 4)")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads per worker (default: 4)")
    parser.add_argument("--aiml", action='store_true', help="Include AIML data")
    parser.add_argument("--hf", action='store_true', help="Include Hugging Face datasets")
    parser.add_argument("--onlytokenize", action='store_true', help="Build vocabulary only")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--use-cache", action='store_true', help="Load cached dataset if available (skipdata loading)")
    parser.add_argument("--refresh-cache", action='store_true', help="Rebuild cache from scratch")
    parser.add_argument("--clear-cache", action='store_true', help="Clear cached datasets and exit")
    args = parser.parse_args()

    # Garbage collector
    gc.collect()

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