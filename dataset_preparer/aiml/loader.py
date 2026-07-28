"""
The provided code defines an `AIMLLoader` class that is responsible for loading and processing AIML (Artificial Intelligence Markup Language) files. Here's a breakdown of the class and its methods:

1. `__init__(self, aiml_dir, tokenizer)`:
   - The constructor initializes the class with the directory containing the AIML files (`aiml_dir`) and an optional `tokenizer` object.
   - It creates an `aiml.Kernel()` object, which is the main interface for the AIML interpreter.
   - The `load_aiml_files()` method is called to load the AIML files.

2. `load_aiml_files(self)`:
   - This method iterates through the files in the `aiml_dir` directory and loads each AIML file using the `kernel.learn()` method.
   - After loading each file, it calls the `create_hf_dataset()` method.

3. `get_response(self, input_text)`:
   - This method takes an input text and uses the `kernel.respond()` method to generate a response.

4. `create_hf_dataset(self)`:
   - This method creates a Hugging Face dataset from the AIML files.
   - It iterates through the AIML files and extracts the pattern and template text from each `<category>` element.
   - The extracted data is stored in a list of dictionaries, where each dictionary has 'input' and 'output' keys.
   - If the corresponding dataset file (with the `.datasets` extension) does not exist, the method calls the `tokenize_data()` method to process the data.

5. `tokenize_data(self, data_hf_dataset, path)`:
   - This method takes a Hugging Face dataset and the file path (without the extension) as input.
   - It concatenates all the Hugging Face datasets into a single `self.finaldata_hf_datasets` variable.
   - The method is currently commented out and does not perform any tokenization. Instead, it saves the list of Hugging Face datasets to a file with the `.datasets` extension.

The purpose of this class is to provide a way to load and process AIML files, creating Hugging Face datasets that can be used for various natural language processing tasks. The `tokenize_data()` method is intended to tokenize the input and output text, but the current implementation only saves the Hugging Face datasets to a file.

To use this class, you would need to create an instance of the `AIMLLoader` class, passing the directory containing the AIML files as an argument. You can then call the `get_response()` method to generate responses based on the loaded AIML files.
"""
"""
El código proporcionado define una clase `AIMLLoader` que es responsable de cargar y procesar archivos AIML (Artificial Intelligence Markup Language). Aquí hay un desglose de la clase y sus métodos:

1. `__init__(self, aiml_dir, tokenizer)`:
   - El constructor inicializa la clase con el directorio que contiene los archivos AIML (`aiml_dir`) y un objeto `tokenizer` opcional.
   - Crea un objeto `aiml.Kernel()`, que es la interfaz principal para el intérprete AIML.
   - Se llama al método `load_aiml_files()` para cargar los archivos AIML.

2. `load_aiml_files(self)`:
   - Este método itera a través de los archivos en el directorio `aiml_dir` y carga cada archivo AIML usando el método `kernel.learn()`.
   - Después de cargar cada archivo, llama al método `create_hf_dataset()`.

3. `get_response(self, input_text)`:
   - Este método toma un texto de entrada y usa el método `kernel.respond()` para generar una respuesta.

4. `create_hf_dataset(self)`:
   - Este método crea un conjunto de datos de Hugging Face a partir de los archivos AIML.
   - Itera a través de los archivos AIML y extrae el texto del patrón y la plantilla de cada elemento `<category>`.
   - Los datos extraídos se almacenan en una lista de diccionarios, donde cada diccionario tiene claves 'input' y 'output'.
   - Si el archivo de conjunto de datos correspondiente (con la extensión `.datasets`) no existe, el método llama al método `tokenize_data()` para procesar los datos.

5. `tokenize_data(self, data_hf_dataset, path)`:
   - Este método toma un conjunto de datos de Hugging Face y la ruta del archivo (sin la extensión) como entrada.
   - Concatena todos los conjuntos de datos de Hugging Face en una sola variable `self.finaldata_hf_datasets`.
   - El método está actualmente comentado y no realiza ninguna tokenización. En su lugar, guarda la lista de conjuntos de datos de Hugging Face en un archivo con la extensión `.datasets`.

El propósito de esta clase es proporcionar una forma de cargar y procesar archivos AIML, creando conjuntos de datos de Hugging Face que se pueden utilizar para diversas tareas de procesamiento de lenguaje natural. El método `tokenize_data()` está diseñado para tokenizar el texto de entrada y salida, pero la implementación actual solo guarda los conjuntos de datos de Hugging Face en un archivo.

Para usar esta clase, necesitarías crear una instancia de la clase `AIMLLoader`, pasando el directorio que contiene los archivos AIML como argumento. Luego puedes llamar al método `get_response()` para generar respuestas basadas en los archivos AIML cargados.
"""
import os
import aiml
import pickle
from datasets import Dataset, concatenate_datasets, load_dataset
from xml.etree import ElementTree

class AIMLLoader:

    # def __init__(self, aiml_dir, tokenizer):
    def __init__(self, aiml_dir):
        self.finaldata_hf_datasets = []
        self.aiml_dir = aiml_dir
        # self.tokenizer = tokenizer
        self.kernel = aiml.Kernel()
        self.load_aiml_files()

    def load_aiml_files(self):
        for file in os.listdir(self.aiml_dir):
            if file.endswith('.aiml'):
                self.kernel.learn(os.path.join(self.aiml_dir, file))
                self.create_hf_dataset()

    def get_response(self, input_text):
        return self.kernel.respond(input_text)

    def create_hf_dataset(self):
        data = []
        for file in os.listdir(self.aiml_dir):
            if file.endswith('.aiml'):

                if not os.path.isfile( os.path.join(self.aiml_dir, file) + '.datasets' ):

                    tree = ElementTree.parse(os.path.join(self.aiml_dir, file))
                    root = tree.getroot()
                    for category in root.findall('category'):
                        pattern = category.find('pattern')
                        if pattern is not None and pattern.text is not None:
                            pattern_text = pattern.text.strip()
                            template = category.find('template')
                            if template is not None and template.text is not None:
                                template_text = template.text.strip()
                                data.append({'input': pattern_text, 'output': template_text})

                    # Aqui tokenizarlo y guardarlo en un archivo
                    self.tokenize_data( Dataset.from_list(data), os.path.join(self.aiml_dir, file) )

        # return Dataset.from_list(data)
        return

    
    def tokenize_data(self, data_hf_dataset, path):

        # Concatenate all the Hugging Face datasets
        self.finaldata_hf_datasets = data_hf_dataset

        # Tokenizing data
        # all_texts = []
        # for dataset in self.finaldata_hf_datasets:
        #     for i in range(len(dataset)):
        #         input_text = dataset['input'][i]
        #         output_text = dataset['output'][i]
        #         combined_text = input_text + ' ' + output_text
        #         all_texts.append(combined_text)

        # self.tokenizer.fit(all_texts)
        # print(f"Tokenized data")

        # Save the tokenizer to a file
        # with open(path + '.tkz', 'wb') as f:
        #     pickle.dump(self.tokenizer, f)

        # Save the tokenized data to a file
        # tokenized_data = [self.tokenizer.encode(text) for text in all_texts]
        # with open(path + '.pkl', 'wb') as f:
        #     pickle.dump(tokenized_data, f)

        # Save the list of Dataset objects
        with open(path + '.datasets', 'wb') as f:
            pickle.dump(self.finaldata_hf_datasets, f)

        return self.finaldata_hf_datasets