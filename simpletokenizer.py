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

    def __init__(self, max_vocab_size=65536, embedding_dim=65536, num_workers=4):
        """Initialize SimpleTokenizer with vocabulary and embedding."""
        self.trie = Trie()
        self.idx2word = {}  # Inverse mapping for efficient O(1) decoding
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.max_vocab_size, self.embedding_dim)
        self.num_workers = num_workers
        
        # Add special tokens
        self._insert_word('<PAD>', 0)
        self._insert_word('<UNK>', 1)
    
    def _insert_word(self, word, index):
        """Insert a word into both trie and idx2word mapping."""
        self.trie.insert(word, index)
        self.idx2word[index] = word
        if index >= self.vocab_size:
            self.vocab_size = index + 1

    def fit(self, texts):
        """Build vocabulary from texts using multiprocessing (no race conditions)."""
        # Collect unique words from all texts
        all_words = set()
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self._extract_words, texts)
        
        # Merge results from all processes (no race conditions during merge)
        for words_set in results:
            all_words.update(words_set)
        
        # Build vocabulary sequentially
        for idx, word in enumerate(all_words, start=2):
            if self.vocab_size < self.max_vocab_size:
                self._insert_word(word, idx)
    
    def _extract_words(self, text):
        """Extract unique words from text (for multiprocessing)."""
        return set(text.split())

    def encode(self, text):
        """Encode text into token IDs."""
        return [self.trie.get_index(word) or 1 for word in text.split()]

    def decode(self, indices):
        """Decode token IDs into text (O(n) complexity using idx2word mapping)."""
        if isinstance(indices, int):
            indices = [indices]
        
        # Use idx2word mapping for O(1) lookup instead of O(vocab_size) trie search
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [None] * (max_length - len(sequence))
        return sequence[:max_length]

    def add_word(self, word):
        """Add a new word to the vocabulary."""
        if self.vocab_size < self.max_vocab_size:
            self._insert_word(word, self.vocab_size)
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