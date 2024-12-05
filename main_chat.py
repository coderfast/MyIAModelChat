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
        self.num_cores = args.num_cores
        self.num_threads = args.num_threads

        # print(f"create dataloader")
        # Create DataLoader with custom collate function
        # dataloader = DataLoader(encoded_data, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        # print(f"created dataloader")

        # Set the number of CPU threads and cores to use
        var_num_threads = args.num_threads
        var_num_cores = args.num_cores

        os.environ["OMP_NUM_THREADS"] = str(var_num_threads)
        torch.set_num_threads(var_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(var_num_cores)
        torch.set_num_interop_threads(var_num_cores)

        # Get the number of CPU cores and threads
        var_num_cores = os.environ.get("MKL_NUM_THREADS", mp.cpu_count())
        var_num_threads = os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())

        # Print the information
        print(f"Number of CPU cores: {var_num_cores}")
        print(f"Number of CPU threads: {var_num_threads}")

        # Initialize and Load pre-trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xla" if "XLA_AVAILABLE" in os.environ else "rocm" if torch.version.hip is not None else "cpu" if torch.backends.mkldnn.is_available() else "opengl" if torch.backends.opengl.is_available() else "opencl" if torch.backends.opencl.is_available() else "ideep" if torch.backends.ideep.is_available() else "hip" if torch.version.hip is not None else "ve" if torch.version.ve is not None else "fpga" if torch.version.fpga is not None else "ort" if torch.version.ort is not None else "lazy" if torch.version.lazy is not None else "vulkan" if torch.version.vulkan is not None else "meta" if torch.version.meta is not None else "hpu" if torch.version.hpu is not None else "mtia" if torch.version.mtia is not None else "privateuse" if torch.version.privateuse is not None else "openmp" if torch.backends.openmp.is_available() else "cpu")

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