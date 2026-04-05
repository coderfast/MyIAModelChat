import torch
import torch.nn as nn
from multiprocessing import Pool
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class BilingualTokenizer:
    """Improved tokenizer supporting English and Spanish with language detection and preprocessing."""

    def __init__(self, max_vocab_size=65536, embedding_dim=300, num_workers=4, 
                 use_language_tokens=True, multilingual_vocab=True):
        """
        Initialize BilingualTokenizer with vocabulary and embedding.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            embedding_dim: Embedding dimension
            num_workers: Number of workers for multiprocessing
            use_language_tokens: Add language-specific tokens (<EN>, <ES>)
            multilingual_vocab: Share vocabulary across languages vs separate vocabularies
        """
        self.trie = Trie()
        self.word2idx = {}
        self.idx2word = {}  # Inverse mapping for efficient O(1) decoding
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.max_vocab_size, self.embedding_dim)
        self.num_workers = num_workers
        self.use_language_tokens = use_language_tokens
        self.multilingual_vocab = multilingual_vocab
        
        # Language-specific counters for statistics
        self.language_stats = {'en': 0, 'es': 0}
        
        # Add special tokens
        self._insert_word('<PAD>', 0)
        self._insert_word('<UNK>', 1)
        self._insert_word('<START>', 2)
        self._insert_word('<END>', 3)
        
        # Add language-specific tokens if enabled
        if self.use_language_tokens:
            self._insert_word('<EN>', 4)
            self._insert_word('<ES>', 5)
    
    def _insert_word(self, word, index):
        """Insert a word into word2idx and idx2word mapping."""
        self.trie.insert(word, index)
        self.word2idx[word] = index
        self.idx2word[index] = word
        if index >= self.vocab_size:
            self.vocab_size = index + 1

    def get_index(self, word: str) -> Optional[int]:
        """Get the token index for a word using a direct dictionary lookup."""
        return self.word2idx.get(word)

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using simple heuristics.
        Returns 'en' for English or 'es' for Spanish.
        """
        # Spanish-specific characters and words
        spanish_chars = set('áéíóúñüçñ')
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'por', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'o', 'fue', 'este', 'sí', 'porque', 'esta', 'son', 'entre', 'está', 'cuando'}
        
        # English-specific words
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'}
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Count language-specific indicators
        spanish_indicator = sum(1 for char in text_lower if char in spanish_chars)
        spanish_words_count = len(words & spanish_words)
        english_words_count = len(words & english_words)
        
        # Decision logic
        if spanish_indicator > 0:
            return 'es'
        if spanish_words_count > english_words_count:
            return 'es'
        if english_words_count > spanish_words_count:
            return 'en'
        
        # Default to English if unclear
        return 'en'

    def remove_accents(self, text: str) -> str:
        """Remove accents from Spanish characters."""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

    def preprocess_text(self, text: str, language: Optional[str] = None, remove_accents_flag: bool = False) -> str:
        """
        Preprocess text for tokenization.
        
        Args:
            text: Input text
            language: Language code ('en', 'es', or None for auto-detection)
            remove_accents_flag: Whether to remove accents from Spanish text
        """
        if language is None:
            language = self.detect_language(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle accents for Spanish
        if language == 'es' and remove_accents_flag:
            text = self.remove_accents(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle contractions and special cases
        if language == 'en':
            # English contractions
            contractions_dict = {
                "ain't": "am not", "aren't": "are not", "can't": "cannot",
                "didn't": "did not", "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is",
                "i'd": "i would", "i'll": "i will", "i'm": "i am",
                "i've": "i have", "isn't": "is not", "it's": "it is",
                "let's": "let us", "she'd": "she would", "she'll": "she will",
                "she's": "she is", "that's": "that is", "they'd": "they would",
                "they'll": "they will", "they're": "they are", "they've": "they have",
                "wasn't": "was not", "we'd": "we would", "we'll": "we will",
                "we're": "we are", "we've": "we have", "weren't": "were not",
                "what's": "what is", "won't": "will not", "wouldn't": "would not",
                "you'd": "you would", "you'll": "you will", "you're": "you are",
                "you've": "you have"
            }
            for contraction, expansion in contractions_dict.items():
                text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        
        elif language == 'es':
            # Spanish contractions
            contractions_dict = {
                "al": "a el", "del": "de el"
            }
            for contraction, expansion in contractions_dict.items():
                text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        
        # Remove or replace punctuation (keep some for context)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fit(self, texts: List[str], languages: Optional[List[str]] = None, remove_accents_flag: bool = False):
        """
        Build vocabulary from texts using multiprocessing with language awareness.
        
        Args:
            texts: List of text samples
            languages: Optional list of language codes corresponding to texts
            remove_accents_flag: Whether to remove accents during preprocessing
        """
        if languages is None:
            languages = [self.detect_language(text) for text in texts]
        
        # Collect unique word frequencies from all texts
        word_counter = Counter()
        logger.info(f"Preprocessing {len(texts)} texts sequentially...")
        for text, lang in zip(texts, languages):
            preprocessed = self._preprocess_worker(text, lang, remove_accents_flag)
            word_counter.update(word for word in preprocessed.split() if word)
        
        # Sort by frequency and build vocabulary
        sorted_words = word_counter.most_common()
        reserved_tokens = 6 if self.use_language_tokens else 4
        for idx, (word, _) in enumerate(sorted_words, start=reserved_tokens):
            if self.vocab_size < self.max_vocab_size and word not in self.word2idx:
                self._insert_word(word, idx)
    
    def _preprocess_worker(self, text: str, language: str, remove_accents_flag: bool) -> str:
        """Worker function for multiprocessing preprocessing."""
        return self.preprocess_text(text, language, remove_accents_flag)

    def encode(self, text: str, language: Optional[str] = None, 
               add_language_token: bool = True, remove_accents_flag: bool = False) -> List[int]:
        """
        Encode text into token IDs with language awareness.
        
        Args:
            text: Input text
            language: Language code ('en', 'es', or None for auto-detection)
            add_language_token: Whether to prepend language token
            remove_accents_flag: Whether to remove accents
            
        Returns:
            List of token IDs
        """
        if language is None:
            language = self.detect_language(text)
        
        # Track language statistics
        self.language_stats[language] += 1
        
        # Preprocess text
        preprocessed = self.preprocess_text(text, language, remove_accents_flag)
        words = preprocessed.split()
        
        # Build token sequence
        tokens = []
        
        # Add language token if enabled
        if self.use_language_tokens and add_language_token:
            lang_token = '<EN>' if language == 'en' else '<ES>'
            lang_idx = self.get_index(lang_token)
            if lang_idx is not None:
                tokens.append(lang_idx)
        
        # Add START token
        start_idx = self.get_index('<START>')
        if start_idx is not None:
            tokens.append(start_idx)
        
        # Encode words
        for word in words:
            idx = self.get_index(word)
            if idx is None:
                # Use UNK token for unknown words
                tokens.append(1)
            else:
                tokens.append(idx)
        
        # Add END token
        end_idx = self.get_index('<END>')
        if end_idx is not None:
            tokens.append(end_idx)
        
        return tokens

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs into text.
        
        Args:
            indices: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if isinstance(indices, int):
            indices = [indices]
        
        special_tokens = {'<PAD>', '<UNK>', '<START>', '<END>', '<EN>', '<ES>'}
        
        # Use idx2word mapping for O(1) lookup
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)
        
        return ' '.join(words)
    
    def batch_encode(self, texts: List[str], languages: Optional[List[str]] = None,
                     add_language_token: bool = True, remove_accents_flag: bool = False,
                     max_length: Optional[int] = None, pad: bool = True,
                     return_tensors: bool = True):
        """
        Encode multiple texts into padded tensor or Python lists for batch processing.
        
        Args:
            texts: List of text samples
            languages: Optional list of language codes
            add_language_token: Whether to add language tokens
            remove_accents_flag: Whether to remove accents
            max_length: Maximum sequence length (auto-determined if None)
            pad: Whether to pad sequences
            return_tensors: If True, return a torch.Tensor; otherwise return List[List[int]]
            
        Returns:
            Tensor or list of token id lists
        """
        if languages is None:
            languages = [self.detect_language(text) for text in texts]
        
        # Encode all texts
        encoded_texts = [
            self.encode(text, lang, add_language_token, remove_accents_flag)
            for text, lang in zip(texts, languages)
        ]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_texts)
        
        # Pad sequences
        if pad:
            pad_idx = self.get_index('<PAD>')
            padded_texts = [
                seq + [pad_idx] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]
                for seq in encoded_texts
            ]
        else:
            padded_texts = encoded_texts
        
        if return_tensors:
            return torch.tensor(padded_texts, dtype=torch.long)
        return padded_texts
    
    def batch_decode(self, tensor: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of token IDs into texts.
        
        Args:
            tensor: Tensor of shape (batch_size, seq_length)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        return [self.decode(seq.tolist(), skip_special_tokens) for seq in tensor]

    def pad_sequence(self, sequence: List[int], max_length: int, pad_idx: Optional[int] = None) -> List[int]:
        """
        Pad or truncate sequence to max_length.
        
        Args:
            sequence: Input token sequence
            max_length: Target length
            pad_idx: Padding token index (uses <PAD> if None)
            
        Returns:
            Padded/truncated sequence
        """
        if pad_idx is None:
            pad_idx = self.get_index('<PAD>')
        
        if len(sequence) < max_length:
            return sequence + [pad_idx] * (max_length - len(sequence))
        return sequence[:max_length]

    def add_word(self, word: str, language: Optional[str] = None) -> int:
        """
        Add a new word to the vocabulary.
        
        Args:
            word: Word to add
            language: Language code for reference
            
        Returns:
            Token index of the new word
        """
        if self.vocab_size < self.max_vocab_size:
            idx = self.vocab_size
            self._insert_word(word, idx)
            # Initialize embedding for new word
            self.embedding.weight.data[idx] = torch.randn(self.embedding_dim) * 0.02
            return idx
        return None

    def get_vocabulary_size(self) -> int:
        """Get current vocabulary size."""
        return self.vocab_size

    def get_vocab_dict(self) -> Dict[str, int]:
        """Get word to index mapping."""
        return dict(self.word2idx)

    def get_language_stats(self) -> Dict[str, int]:
        """Get language statistics."""
        return self.language_stats.copy()

    def get_language_token_index(self, language: str) -> int:
        """Get token index for language marker."""
        lang_token = '<EN>' if language == 'en' else '<ES>'
        return self.get_index(lang_token)

    def load_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor):
        """Load pretrained embeddings."""
        self.embedding.weight.data.copy_(pretrained_embeddings)

    def get_embeddings(self) -> nn.Embedding:
        """Get embedding layer."""
        return self.embedding

    def get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding vector for a word."""
        idx = self.get_index(word)
        if idx is not None:
            return self.embedding.weight.data[idx]
        return None

    def init_weights(self, module):
        """Initialize weights for neural network modules."""
        if isinstance(module, nn.Embedding):
            # Kaiming initialization for Embedding layer
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='linear')
        elif isinstance(module, nn.LSTM):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param)
        elif isinstance(module, nn.Linear):
            # Kaiming initialization for Linear layer
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(module.bias, 0, 0.01)

    def get_pad_index(self) -> int:
        """Get padding token index."""
        return self.get_index('<PAD>')

    def get_unknown_index(self) -> int:
        """Get unknown token index."""
        return self.get_index('<UNK>')

    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file."""
        import json
        vocab_dict = self.get_vocab_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        # Rebuild trie and mappings
        self.trie = Trie()
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
        for word, idx in sorted(vocab_dict.items(), key=lambda item: item[1]):
            self._insert_word(word, idx)


# Backward compatibility alias
class SimpleTokenizer(BilingualTokenizer):
    """Backward compatibility wrapper for SimpleTokenizer."""
    def __init__(self, max_vocab_size=65536, embedding_dim=300, num_workers=4):
        super().__init__(max_vocab_size, embedding_dim, num_workers, 
                         use_language_tokens=False, multilingual_vocab=True)