import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import aiml
import os

# AIML Loader
class AIMLLoader:
    def __init__(self, aiml_dir):
        self.kernel = aiml.Kernel()
        for file in os.listdir(aiml_dir):
            if file.endswith('.aiml'):
                self.kernel.learn(os.path.join(aiml_dir, file))
    
    def get_response(self, input_text):
        return self.kernel.respond(input_text)

# Dataset
class ChatDataset(Dataset):
    def __init__(self, aiml_loader, num_samples=1000):
        self.aiml_loader = aiml_loader
        self.num_samples = num_samples
        self.data = self.generate_data()
    
    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            input_text = f"Random input {_}"
            output_text = self.aiml_loader.get_response(input_text)
            data.append((input_text, output_text))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
    
    def fit(self, texts):
        words = set()
        for text in texts:
            words.update(text.split())
        
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]
    
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))
        return sequence[:max_length]

# Model
class ChatModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ChatModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

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
def generate_response(model, tokenizer, input_text, max_length=50):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_id = outputs[0, -1, :].argmax().item()
            if next_token_id == tokenizer.word2idx['<PAD>']:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
    
    return tokenizer.decode([token for token in input_ids[0].tolist() if token != tokenizer.word2idx['<PAD>']])

# Main
if __name__ == '__main__':
    # Load AIML files
    aiml_loader = AIMLLoader('aiml')
    
    # Create dataset
    dataset = ChatDataset(aiml_loader)
    
    # Tokenize data
    tokenizer = SimpleTokenizer()
    all_texts = [item[0] + ' ' + item[1] for item in dataset]
    tokenizer.fit(all_texts)
    
    # Prepare data for PyTorch
    encoded_data = [(tokenizer.encode(item[0]), tokenizer.encode(item[1])) for item in dataset]
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(encoded_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xla" if "XLA_AVAILABLE" in os.environ else "rocm" if torch.version.hip is not None else "cpu" if torch.backends.mkldnn.is_available() else "opengl" if torch.backends.opengl.is_available() else "opencl" if torch.backends.opencl.is_available() else "ideep" if torch.backends.ideep.is_available() else "hip" if torch.version.hip is not None else "ve" if torch.version.ve is not None else "fpga" if torch.version.fpga is not None else "ort" if torch.version.ort is not None else "lazy" if torch.version.lazy is not None else "vulkan" if torch.version.vulkan is not None else "meta" if torch.version.meta is not None else "hpu" if torch.version.hpu is not None else "mtia" if torch.version.mtia is not None else "privateuse" if torch.version.privateuse is not None else "openmp" if torch.backends.openmp.is_available() else "cpu")
    model = ChatModel(tokenizer.vocab_size, embed_size=128, hidden_size=256).to(device)
    print(f"Using {device} device")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 150
    for epoch in range(num_epochs):
        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'chat_model.pth')

    # Test the model
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        print("Bot:", response)