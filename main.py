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
from datasets import concatenate_datasets, load_dataset
from transformers import pipeline


# Training function
def train(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape outputs and targets
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        targets = targets.contiguous().view(-1)

        # Ignore padded elements
        non_pad_mask = targets.ne(tokenizer.word2idx['<PAD>'])
        outputs = outputs[non_pad_mask]
        targets = targets[non_pad_mask]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Collate function for DataLoader
def collate_fn(batch):

    input_sequences, output_sequences = zip(*batch)

    # Find max lengths
    # max_input_len = max(len(seq) for seq in input_sequences)
    # max_output_len = max(len(seq) for seq in output_sequences)
    max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))

    # Pad sequences
    # padded_inputs = [seq + [tokenizer.word2idx['<PAD>']] * (max_input_len - len(seq)) for seq in input_sequences]
    # padded_outputs = [seq + [tokenizer.word2idx['<PAD>']] * (max_output_len - len(seq)) for seq in output_sequences]
    padded_inputs = [seq + [tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in input_sequences]
    padded_outputs = [seq + [tokenizer.word2idx['<PAD>']] * (max_len - len(seq)) for seq in output_sequences]

    # Convert to tensors
    input_tensor = torch.LongTensor(padded_inputs)
    output_tensor = torch.LongTensor(padded_outputs)

    return input_tensor, output_tensor


# Function to generate responses
# def generate_response(model, tokenizer, input_text, max_length=50):

#     model.eval()
#     input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)

#     with torch.no_grad():
#         for _ in range(max_length):
#             outputs = model(input_ids)
#             next_token_id = outputs[0, -1, :].argmax().item()
#             if next_token_id == tokenizer.word2idx['<PAD>']:
#                 break
#             input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)

#     return tokenizer.decode([token for token in input_ids[0].tolist() if token != tokenizer.word2idx['<PAD>']])


# Main
if __name__ == '__main__':
    
    final_hf_datasets = []

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--aiml", action='store_true')
    parser.add_argument("--hf", action='store_true')
    parser.add_argument("--onlytokenize", action='store_true')
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # print(f"var_process_aiml: {var_process_aiml}")
    # if (var_process_aiml == True):
    #     sys.exit("tokenized")
    # sys.exit("fin")

    # Load AIML files
    if args.aiml:
        aiml_loader = AIMLLoader('aiml')

        # Create Hugging Face Dataset from AIMLs
        aiml_hf_dataset = aiml_loader.create_hf_dataset()
        final_hf_datasets.append( aiml_hf_dataset )

    # Load a Hugging Face Dataset
    if args.hf:
        huggingface_hf_dataset = load_dataset("wikimedia/wikipedia", "20231101.es")
        final_hf_datasets.append( huggingface_hf_dataset )

    # Merge the datasets
    # merged_dataset = concatenate_datasets([aiml_hf_dataset, huggingface_hf_dataset])
    # merged_dataset = concatenate_datasets([aiml_hf_dataset])
    merged_dataset = concatenate_datasets(final_hf_datasets)

    print(f"Length of merged dataset: {len(merged_dataset)}")

    # Check if the tokenizer and tokenized data files exist
    tokenizer = None
    tokenized_data = None
    if os.path.exists('tokenizer.pkl') and os.path.exists('tokenized_data.pkl'):

        # Load the tokenizer from the file
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        # Load the tokenized data from the file
        with open('tokenized_data.pkl', 'rb') as f:
            tokenized_data = pickle.load(f)

    print(f"Tokenizing data")
    # Tokenize data
    if tokenizer == None:
        tokenizer = SimpleTokenizer()

    # Tokenizing data
    all_texts = []

    # all_texts = [merged_dataset['input'][i] + ' ' + merged_dataset['output'][i] for i in range(len(merged_dataset))]
    for i in range(len(merged_dataset)):
        print(f"Input: {merged_dataset['input'][i]}")
        print(f"Output: {merged_dataset['output'][i]}")
        print()
        input_text = merged_dataset['input'][i]
        output_text = merged_dataset['output'][i]
        combined_text = input_text + ' ' + output_text
        all_texts.append(combined_text)

        # save in each loop 
        tokenizer.fit(all_texts)
        print(f"Tokenized data")

        # Save the tokenizer to a file
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        # Save the tokenized data to a file
        # tokenized_data = tokenizer.encode_batch(all_texts)
        tokenized_data = [tokenizer.encode(text) for text in all_texts]
        with open('tokenized_data.pkl', 'wb') as f:
            pickle.dump(tokenized_data, f)

    tokenizer.fit(all_texts)
    print(f"Tokenized data")

    # Save the tokenizer to a file
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save the tokenized data to a file
    # tokenized_data = tokenizer.encode_batch(all_texts)
    tokenized_data = [tokenizer.encode(text) for text in all_texts]
    with open('tokenized_data.pkl', 'wb') as f:
        pickle.dump(tokenized_data, f)

    if args.onlytokenize:
        sys.exit("only tokenized, all done, exit")


    print(f"preparing data for pytorch, encoding data")
    # Prepare data for PyTorch
    encoded_data = [(tokenizer.encode(merged_dataset['input'][i]), tokenizer.encode(merged_dataset['output'][i])) for i in range(len(merged_dataset))]
    print(f"prepared data for pytorch, encoded data")

    print(f"create dataloader")
    # Create DataLoader with custom collate function
    dataloader = DataLoader(encoded_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print(f"created dataloader")

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

    # Initialize model
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xla" if "XLA_AVAILABLE" in os.environ else "rocm" if torch.version.hip is not None else "cpu" if torch.backends.mkldnn.is_available() else "opengl" if torch.backends.opengl.is_available() else "opencl" if torch.backends.opencl.is_available() else "ideep" if torch.backends.ideep.is_available() else "hip" if torch.version.hip is not None else "ve" if torch.version.ve is not None else "fpga" if torch.version.fpga is not None else "ort" if torch.version.ort is not None else "lazy" if torch.version.lazy is not None else "vulkan" if torch.version.vulkan is not None else "meta" if torch.version.meta is not None else "hpu" if torch.version.hpu is not None else "mtia" if torch.version.mtia is not None else "privateuse" if torch.version.privateuse is not None else "openmp" if torch.backends.openmp.is_available() else "cpu")
    model = ChatModel(tokenizer, embed_size=128, hidden_size=256).to(device)
    print(f"Using {device} device")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        
        if keyboard.is_pressed('esc'):
            break;

        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'chat_model.pth')
    
    
    # Initialize Dialogue Manager
    intent_classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    knowledge_base = {"France": "The capital of France is Paris."}
    persona = {
        "name": "Claude",
        "age": 30,
        "occupation": "AI assistant",
        "interests": ["technology", "science", "philosophy"]
    }
    dialogue_manager = DialogueManager(model, tokenizer, intent_classifier, sentiment_analyzer, knowledge_base, persona)

    # Test the model
    while not keyboard.is_pressed('esc'):
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # response = generate_response(model, tokenizer, user_input)
        response = dialogue_manager.generate_response(user_input)
        print("Bot:", response)
