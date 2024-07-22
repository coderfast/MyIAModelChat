import os
import aiml

# AIML Loader
class AIMLLoader:

    def __init__(self, aiml_dir):
        self.kernel = aiml.Kernel()
        for file in os.listdir(aiml_dir):
            if file.endswith('.aiml'):
                self.kernel.learn(os.path.join(aiml_dir, file))
    
    def get_response(self, input_text):
        return self.kernel.respond(input_text)