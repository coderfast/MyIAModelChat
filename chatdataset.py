"""
This code defines a PyTorch dataset class called ChatDataset that generates a dataset of 
input-output pairs for a chatbot. Here's a breakdown of the code:

Imports: The code imports the necessary PyTorch modules, including nn (neural networks), 
optim (optimization), and Dataset and DataLoader from torch.utils.data.

ChatDataset Class: The ChatDataset class inherits from the Dataset class, which 
is a PyTorch class for creating custom datasets.

init Method: The __init__ method initializes the dataset with an 
aiml_loader object (which is not shown in the provided code) and a num_samples parameter that determines the number of samples to generate.

generate_data Method: The generate_data method generates the dataset 
by creating num_samples number of input-output pairs. The input text is generated as a string, and the output text is obtained by calling the get_response method of the aiml_loader object with the input text.

len Method: The __len__ method returns the length of the dataset, which 
is the number of samples generated.

getitem Method: The __getitem__ method returns the 
input-output pair at the specified index idx.

This dataset class can be used to create a PyTorch DataLoader object, which can 
then be used to feed the data to a neural network model for training or inference. 

The DataLoader class takes care of batching the data, shuffling the samples, and 
other useful data-related operations.
"""

"""
Este código define una clase de conjunto de datos de PyTorch llamada ChatDataset que genera un conjunto 
de datos de pares de entrada-salida para un chatbot. Aquí hay un desglose del código:

Importaciones: El código importa los módulos de PyTorch necesarios, incluidos nn (redes neuronales), 
optim (optimización) y Dataset y DataLoader de torch.utils.data.

Clase ChatDataset: La clase ChatDataset hereda de la clase Dataset, que es una clase de PyTorch para crear 
conjuntos de datos personalizados.

Método init: El método init inicializa el conjunto de datos con un objeto aiml_loader (que no se muestra en el 
código proporcionado) y un parámetro num_samples que determina el número de muestras a generar.

Método generate_data: El método generate_data genera el conjunto de datos creando num_samples número de pares 
de entrada-salida. El texto de entrada se genera como una cadena y el texto de salida se obtiene llamando al método get_response del objeto aiml_loader con el texto de entrada.

Método len: El método len devuelve la longitud del conjunto de datos, que es el número de muestras generadas.

Método getitem: El método getitem devuelve el par de entrada-salida en el índice especificado idx.

Esta clase de conjunto de datos se puede usar para crear un objeto DataLoader de PyTorch, que luego 
se puede usar para alimentar los datos a un modelo de red neuronal para entrenamiento o inferencia. La clase DataLoader se encarga del procesamiento por lotes de los datos, la mezcla de las muestras y otras operaciones de datos útiles.
"""
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