import torch
import torch.nn as nn
import torch.optim as optim
from simpletokenizer import *

# Model
class ChatModel(nn.Module):

    def __init__(self, tokenizer, embed_size, hidden_size):
        super(ChatModel, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embed_size, padding_idx=tokenizer.word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tokenizer.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)