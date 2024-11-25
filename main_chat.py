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

        self.tokenizer = SimpleTokenizer()

        # Parse command-line arguments
        self.chat = args.chat
        self.num_cores = args.num_cores
        self.num_threads = args.num_threads
        
        # print(f"create dataloader")
        # Create DataLoader with custom collate function
        # dataloader = DataLoader(encoded_data, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        # print(f"created dataloader")

        # Set the number of CPU threads and cores to use
        # num_threads = 8
        # num_cores = 8
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
        # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xla" if "XLA_AVAILABLE" in os.environ else "rocm" if torch.version.hip is not None else "cpu" if torch.backends.mkldnn.is_available() else "opengl" if torch.backends.opengl.is_available() else "opencl" if torch.backends.opencl.is_available() else "ideep" if torch.backends.ideep.is_available() else "hip" if torch.version.hip is not None else "ve" if torch.version.ve is not None else "fpga" if torch.version.fpga is not None else "ort" if torch.version.ort is not None else "lazy" if torch.version.lazy is not None else "vulkan" if torch.version.vulkan is not None else "meta" if torch.version.meta is not None else "hpu" if torch.version.hpu is not None else "mtia" if torch.version.mtia is not None else "privateuse" if torch.version.privateuse is not None else "openmp" if torch.backends.openmp.is_available() else "cpu")
        self.model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256).to(device)        
        model_path = 'chat_model.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        print(f"Using {device} device")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word2idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters())

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
        # padded_inputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_input_len - len(seq)) for seq in input_sequences]
        # padded_outputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_output_len - len(seq)) for seq in output_sequences]
        padded_inputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in input_sequences]
        padded_outputs = [seq + [self.tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in output_sequences]

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