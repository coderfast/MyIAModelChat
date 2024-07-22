import os
import aiml
from datasets import Dataset

import os
from xml.etree import ElementTree

class AIMLLoader:
    def __init__(self, aiml_dir):
        self.aiml_dir = aiml_dir
        self.kernel = aiml.Kernel()
        self.load_aiml_files()

    def load_aiml_files(self):
        for file in os.listdir(self.aiml_dir):
            if file.endswith('.aiml'):
                self.kernel.learn(os.path.join(self.aiml_dir, file))

    def get_response(self, input_text):
        return self.kernel.respond(input_text)

    def create_hf_dataset(self):
        data = []
        for file in os.listdir(self.aiml_dir):
            if file.endswith('.aiml'):
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

        return Dataset.from_list(data)