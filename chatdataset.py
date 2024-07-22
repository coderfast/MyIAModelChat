import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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