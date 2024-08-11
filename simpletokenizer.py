import torch
import torch.nn as nn
import torch.optim as optim

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
