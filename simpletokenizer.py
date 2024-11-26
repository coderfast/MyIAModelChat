"""
This code defines a simple tokenizer class called SimpleTokenizer that can be used for text preprocessing 
in a natural language processing (NLP) task. Here's a breakdown of the code:

Initialization: The __init__ method initializes the tokenizer with a predefined vocabulary of 
two tokens: <PAD> and <UNK>. The word2idx and idx2word dictionaries are used to map words to their 
corresponding indices and vice versa. The vocab_size variable keeps track of the current size of the 
vocabulary, and max_vocab_size sets the maximum allowed vocabulary size.

Fitting the Tokenizer: The fit method takes a list of texts (e.g., sentences or documents) and builds 
the vocabulary based on the unique words in the texts. The method ensures that the vocabulary size does 
not exceed the max_vocab_size limit.

Encoding and Decoding: The encode method takes a text input and returns a list of indices representing the 
words in the text. The decode method takes a list of indices and returns the corresponding text.

Padding Sequences: The pad_sequence method takes a sequence (e.g., a list of word indices) and pads it 
with the <PAD> token to a specified maximum length. This is useful for creating fixed-size input tensors 
for neural network models.

Adding New Words: The add_word method allows you to add a new word to the tokenizer's vocabulary, as long 
as the vocabulary size does not exceed the max_vocab_size limit.

This tokenizer can be used as a building block for more complex NLP models, such as language models or text 
classification models. The SimpleTokenizer class provides a basic implementation of text preprocessing, which 
can be extended or modified as needed for specific use cases.
"""

"""
Este código define una clase de tokenizador simple llamada SimpleTokenizer que se puede usar para el 
preprocesamiento de texto en una tarea de procesamiento de lenguaje natural (NLP). Veamos los detalles:

Inicialización: 
El método __init__ inicializa el tokenizador con un vocabulario predefinido de 
dos tokens: <PAD> y <UNK>. Los diccionarios word2idx y idx2word se utilizan para asignar 
palabras a sus índices correspondientes y viceversa. La variable vocab_size realiza un 
seguimiento del tamaño actual del vocabulario, y max_vocab_size establece el tamaño máximo 
permitido del vocabulario.

Ajuste del Tokenizador: 
El método fit toma una lista de textos (por ejemplo, oraciones o documentos) y construye 
el vocabulario en función de las palabras únicas en los textos. El método se asegura de que el 
tamaño del vocabulario no exceda el límite de max_vocab_size.

Codificación y Decodificación: 
El método encode toma una entrada de texto y devuelve una lista de índices que representan las 
palabras en el texto. El método decode toma una lista de índices y devuelve el texto correspondiente.

Rellenado de Secuencias: 
El método pad_sequence toma una secuencia (por ejemplo, una lista de índices de palabras) y la 
rellena con el token <PAD> hasta una longitud máxima especificada. Esto es útil para crear tensores 
de entrada de tamaño fijo para modelos de redes neuronales.

Agregar Nuevas Palabras: 
El método add_word le permite agregar una nueva palabra al vocabulario del tokenizador, siempre que 
el tamaño del vocabulario no exceda el límite de max_vocab_size.

Este tokenizador se puede usar como un bloque de construcción para modelos de NLP más complejos, como modelos 
de lenguaje o modelos de clasificación de texto. La clase SimpleTokenizer proporciona una implementación 
básica del preprocesamiento de texto, que se puede extender o modificar según sea necesario para 
casos de uso específicos. 
"""
import torch
import torch.nn as nn
import torch.optim as optim

# Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        self.max_vocab_size = 10000  # Límite máximo del vocabulario
    
    def fit(self, texts, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        words = set()
        for text in texts:
            words.update(text.split())

        for word in words:
            if word not in self.word2idx and len(self.word2idx) < self.max_vocab_size:
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

    def add_word(self, word):
        if word not in self.word2idx and len(self.word2idx) < self.max_vocab_size:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1