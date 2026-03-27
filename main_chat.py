import os
import argparse
import pickle
import sys
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from aimlloder import *
from dialogmanager import DialogueManager
from simpletokenizer import *
from chatmodel import *
from chatdataset import *
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import pipeline, BertTokenizer


class MainChat:

    tokenizer = None
    tokenized_data = None
    dialogue_manager = None

    def __init__(self, args):

        # Don't show the next wanings
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

        # Check if the tokenizer and tokenized data files exist
        self.tokenizer = None
        self.tokenized_data = None

        # Parse command-line arguments
        self.chat = args.chat
        self.use_cpuonly = getattr(args, 'use_cpuonly', False)
        self.cuda_device = getattr(args, 'cuda_device', None)

        # If explicit CUDA device is requested and CPU-only is not set, enforce it in environment
        if self.cuda_device is not None and not self.use_cpuonly:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
            print(f"Using CUDA devices limited to: {self.cuda_device}")

        # Initialize and Load pre-trained model
        def get_device():
            """Select the best available device for computation."""
            if self.use_cpuonly:
                device = torch.device('cpu')
                print("Using CPU (forced by --use-cpuonly)")
                return device

            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using MPS (Metal Performance Shaders)")
            else:
                device = torch.device('cpu')
                print("Using CPU")
            return device
        
        device = get_device()

        # Check if tokenizer file exists
        tokenizer_path = 'tokenizer.pth'
        if os.path.exists(tokenizer_path):
            try:
                # Attempt to load existing tokenizer
                self.tokenizer = torch.load(tokenizer_path)
                print(f"Tokenizer loaded from {tokenizer_path}")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
        else:
            # Create new tokenizer instance
            self.tokenizer = SimpleTokenizer(max_vocab_size=128, embedding_dim=128)

        # Load model and weights
        try:
            # Create the model
            self.model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256).to(device)

            # Load the pre-trained embeddings
            # pretrained_embeddings = torch.load('pretrained_embeddings.pth', map_location=device, pickle_module=pickle)
            pretrained_embeddings = torch.load('pretrained_embeddings.pth', map_location=device)
            
            # Load the model
            # self.model.load_state_dict(torch.load('chat_model.pth', map_location=device))

            # Set the pre-trained embeddings in the tokenizer
            self.tokenizer.embedding.weight.data.copy_(pretrained_embeddings)

            print("Pre-trained model loaded successfully.")
            self.model.eval()
        except RuntimeError as e:
            print(f"Error loading pre-trained model: {e}")
            print("Initializing model with random weights...")
            self.model.apply(self.tokenizer.init_weights)

        print(f"Using {device} device")

        # Initialize Dialogue Manager
        intent_classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
        sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        persona = {
            "name": "Eduardo Piñera Aznárez",
            "age": 51,
            "occupation": "AI assistant",
            "interests": ["IT technology", "MS Office", "Libre Office", "Games", "Humanity simulation"]
        }
        self.dialogue_manager = DialogueManager(self.model, device, self.tokenizer, intent_classifier, sentiment_analyzer, persona)


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


    def performMainChat(self):

        # Test the model
        while not keyboard.is_pressed('esc'):
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            # response = generate_response(model, tokenizer, user_input)
            response = self.dialogue_manager.generate_response(user_input)
            print("Bot:", response)