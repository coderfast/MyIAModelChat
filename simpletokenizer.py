import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTokenizer:

    def __init__(self, max_vocab_size=128, embedding_dim=128):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 128
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.max_vocab_size, self.embedding_dim)

    def fit(self, texts, load_pretrained=False, pretrained_embeddings=None):
        words = set()
        for text in texts:
            words.update(text.split())

        for word in words:
            if word not in self.word2idx and len(self.word2idx) < self.max_vocab_size:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        # Inicializar los pesos de los embeddings
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        if load_pretrained and pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        else:
            self.embedding.weight.data.normal_(0, 1)  # Inicializar aleatoriamente

    def encode(self, text):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]

    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))
        return sequence[:max_length]

    def add_word(self, word):
        if word not in self.word2idx and len(self.word2idx) < self.max_vocab_size:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
            # Actualizar los pesos de los embeddings
            self.embedding.weight.data[self.vocab_size - 1] = torch.randn(self.embedding_dim)

    def load_pretrained_embeddings(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(pretrained_embeddings)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
        elif isinstance(module, nn.LSTM):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.normal_(module.bias, 0, 0.01)
