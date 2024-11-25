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



# Main
if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--chat", action='store_true')
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--aiml", action='store_true')
    parser.add_argument("--hf", action='store_true')
    parser.add_argument("--onlytokenize", action='store_true')
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Garbage collector
    gc.collect()

    # Train
    if args.train:
        iMainTrain = MainTrain(args)
        iMainTrain.performMainTrain()

    # Chat
    if args.chat:
        iMainChat = MainChat(args)
        iMainChat.performMainChat()