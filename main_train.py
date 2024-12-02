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
from aimlloder import *
from dialogmanager import DialogueManager
from simpletokenizer import *
from chatmodel import *
from chatdataset import *
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import pipeline


class MainTrain:
    
    def __init__(self, args):
        
        print(f"MainTrain initializing...")

        # Check if the tokenizer and tokenized data files exist
        self.tokenizer = None
        self.tokenized_data = None
        self.aiml_list_datasets_objects = Dataset.from_list([])

        # final Huggin Face datasets merged
        # final_hf_datasets = []

        # Parse command-line arguments
        self.num_cores = args.num_cores
        self.num_threads = args.num_threads
        self.aiml = args.aiml
        self.hf = args.hf
        self.onlytokenize = args.onlytokenize
        self.epochs = args.epochs

        # Tokenize data
        print(f"Create Tokenizer")
        if self.tokenizer == None:
            self.tokenizer = SimpleTokenizer()

        # process AIML files
        self.processAiml()

        print(f"MainTrain initialized...")


    def processAiml(self):

        print(f"processAiml...")
        
        # Create a data list
        data_dir = 'aiml_dev'
        self.aiml_list_datasets_objects = Dataset.from_list([])
        self.aiml_list_datasets_data = []

        # Load AIML files
        if self.aiml:
            aiml_loader = AIMLLoader('aiml_dev')
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.datasets'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'rb') as f:
                    tokenized_data = pickle.load(f)

                    # Create a list of dictionaries from the tokenized data
                    data = [{'input_ids': token_ids} for token_ids in tokenized_data]
                    
                    # Imprimir el contenido de data
                    print(f"Contenido de data:")
                    for d in data:
                        print(d)
                    print()

                    self.aiml_list_datasets_objects = concatenate_datasets([self.aiml_list_datasets_objects, Dataset.from_list(data)])
                    # self.aiml_list_datasets_objects = Dataset.from_list(data)

        # Imprimir los primeros 5 ejemplos
        # print("Primeros 5 ejemplos de merged_dataset:")
        # print(f"IDs: {self.aiml_list_datasets_objects}")
        # print("Primeros 5 ejemplos de merged_dataset:")
        # first_5 = self.aiml_list_datasets_objects.take(5)
        # for example in first_5:
        #     print(f"Input IDs: {example['input_ids']}")
        #     if 'output' in example:
        #         print(f"Output: {example['output']}")
        #     print()
        # sys.exit("End 1")

        print(f"processAiml done...")


    # Training function
    def train(self, model, dataloader, criterion, optimizer, device):

        total_loss = 0

        model.train()
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs and targets
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            targets = targets.contiguous().view(-1)

            # Ignore padded elements
            non_pad_mask = targets.ne(self.tokenizer.word2idx['<PAD>'])
            outputs = outputs[non_pad_mask]
            targets = targets[non_pad_mask]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print(f"total_loss: {total_loss:.4f}")

        return total_loss / len(dataloader)


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


    def encode(self, text):
        if isinstance(text, str):
            return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]
        else:
            return [self.encode(str(item)) for item in text]


    def performMainTrain(self):

        # Merge the datasets
        merged_dataset = concatenate_datasets([self.aiml_list_datasets_objects])

        print(f"Length of merged dataset: {len(merged_dataset)}")

        # Imprimir los primeros 5 ejemplos
        # print("Primeros 5 ejemplos de merged_dataset:")
        # print(f"IDs: {self.aiml_list_datasets_objects}")
        # first_5 = merged_dataset.take(5)
        # for example in first_5:
        #     print(f"Input IDs: {example['input_ids']}")
        #     if 'output' in example:
        #         print(f"Output: {example['output']}")
        #     print()

        # Prepare data for PyTorch
        print(f"Preparando datos para PyTorch, codificando datos")
        for i in range(len(merged_dataset)):
            self.tokenizer.fit(merged_dataset[i]['input_ids']['input'] + merged_dataset[i]['input_ids']['output'])

        encoded_data = []
        for i in range(len(merged_dataset)):
            if 'input' in merged_dataset[i] and 'output' in merged_dataset[i]:
                input_ids = self.tokenizer.encode(str(merged_dataset[i]['input']))
                output_ids = self.tokenizer.encode(str(merged_dataset[i]['output']))
                # print(f"input1: {input_ids}")
                # print(f"output1: {output_ids}")
                encoded_data.append((input_ids, output_ids))
            else:
                keys = list(merged_dataset[i].keys())
                if len(keys) == 2 and 'input_ids' in keys and 'output' in keys:
                    input_ids = self.tokenizer.encode(str(merged_dataset[i]['input_ids']['input']))
                    output_ids = self.tokenizer.encode(str(merged_dataset[i]['input_ids']['output']))
                    # print(f"input1: {input_ids}")
                    # print(f"output1: {output_ids}")
                    encoded_data.append((input_ids, output_ids))
                else:
                    input_ids = self.tokenizer.encode(str(merged_dataset[i]['input_ids']['input']))
                    output_ids = self.tokenizer.encode(str(merged_dataset[i]['input_ids']['output']))
                    # print(f"input1: {input_ids}")
                    # print(f"output1: {output_ids}")
                    encoded_data.append((input_ids, output_ids))
        print(f"Prepared data for PyTorch, encoded data")

        # Inspeccionar el contenido de encoded_data
        # print("Contenido de los primeros 5 ejemplos de encoded_data:")
        # print(f"encoded_data: {encoded_data}")
        # for i in range(5):
        #     input_ids, output_ids = encoded_data[i]
        #     print(f"Input IDs: {input_ids}")
        #     print(f"Output IDs: {output_ids}")
        #     print()
        # sys.exit("End 2")

        print(f"create dataloader")
        # Create DataLoader with custom collate function
        dataloader = DataLoader(encoded_data, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        print(f"created dataloader")

        # Set the number of CPU threads and cores to use
        var_num_threads = self.num_threads
        var_num_cores = self.num_cores
        
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

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xla" if "XLA_AVAILABLE" in os.environ else "rocm" if torch.version.hip is not None else "cpu" if torch.backends.mkldnn.is_available() else "opengl" if torch.backends.opengl.is_available() else "opencl" if torch.backends.opencl.is_available() else "ideep" if torch.backends.ideep.is_available() else "hip" if torch.version.hip is not None else "ve" if torch.version.ve is not None else "fpga" if torch.version.fpga is not None else "ort" if torch.version.ort is not None else "lazy" if torch.version.lazy is not None else "vulkan" if torch.version.vulkan is not None else "meta" if torch.version.meta is not None else "hpu" if torch.version.hpu is not None else "mtia" if torch.version.mtia is not None else "privateuse" if torch.version.privateuse is not None else "openmp" if torch.backends.openmp.is_available() else "cpu")
        model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256).to(device)
        print(f"Using {device} device")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word2idx['<PAD>'])
        optimizer = optim.Adam(model.parameters())

        # Training loop
        num_epochs = self.epochs
        for epoch in range(num_epochs):

            if keyboard.is_pressed('esc'):
                break;

            loss = self.train(model, dataloader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
            
            # Save the model, tokenizer, and pre-trained embeddings
            torch.save(model.state_dict(), 'chat_model.pth')
            torch.save(self.tokenizer, 'tokenizer.pth')
            torch.save(self.tokenizer.embedding.weight.data, 'pretrained_embeddings.pth')