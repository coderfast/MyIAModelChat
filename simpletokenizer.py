import torch
import torch.nn as nn
from multiprocessing import Pool

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.index = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, index):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.index = index

    def get_index(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node.index
        return None

class SimpleTokenizer:

    def __init__(self, max_vocab_size=128, embedding_dim=128, num_workers=4):
        # self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        # self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.trie = Trie()
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.max_vocab_size, self.embedding_dim)
        self.num_workers = num_workers
        
        # Agregar el token de relleno al Trie
        self.trie.insert('<PAD>', 0)
        self.trie.insert('<UNK>', 1)
        self.vocab_size += 2

    def fit(self, texts):
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self.process_text, texts)

        for result in results:
            for word, index in result:
                if self.vocab_size < self.max_vocab_size:
                    self.trie.insert(word, index)
                    self.vocab_size += 1

    def process_text(self, text):
        word_indices = []
        for word in text.split():
            index = self.trie.get_index(word)
            if index is None:
                index = self.vocab_size
                word_indices.append((word, index))
        return word_indices

    def encode(self, text):
        return [self.trie.get_index(word) or self.trie.get_index('<UNK>') for word in text.split()]

    def decode(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        words = []
        for idx in indices:
            if idx is None:
                words.append('<UNK>')
            else:
                for word, node in self.trie.root.children.items():
                    if node.index == idx:
                        words.append(word)
                        break
        return ' '.join(words)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [None] * (max_length - len(sequence))
        return sequence[:max_length]

    def add_word(self, word):
        if self.vocab_size < self.max_vocab_size:
            self.trie.insert(word, self.vocab_size)
            self.vocab_size += 1
            self.embedding.weight.data[self.vocab_size - 1] = torch.randn(self.embedding_dim)

    def load_pretrained_embeddings(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(pretrained_embeddings)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # Uniform initialization for Embedding layer
            # nn.init.uniform_(module.weight, -0.1, 0.1)
            # Xavier initialization for Embedding layer
            # nn.init.xavier_uniform_(module.weight)
            # Kaiming initialization for Embedding layer
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='linear')
        elif isinstance(module, nn.LSTM):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param)
        elif isinstance(module, nn.Linear):
            # Uniform initialization for Linear layer
            # nn.init.uniform_(module.weight, -0.1, 0.1)
            # nn.init.xavier_uniform_(module.weight)
            # Kaiming initialization for Linear layer
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(module.bias, 0, 0.01)

    def get_pad_index(self):
        return self.trie.get_index('<PAD>')