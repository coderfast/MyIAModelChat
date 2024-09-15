import os
import aiml
import pickle
from datasets import Dataset, concatenate_datasets, load_dataset

import os
from xml.etree import ElementTree

class AIMLLoader:

    def __init__(self, aiml_dir, tokenizer):
        self.finaldata_hf_datasets = []
        self.aiml_dir = aiml_dir
        self.tokenizer = tokenizer
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

    
    def tokenize_data(self, data_hf_datasets, path):

        # Concatenate all the Hugging Face datasets
        self.finaldata_hf_datasets = data_hf_datasets

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