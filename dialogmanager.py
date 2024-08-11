# Dialogue Manager
import torch
from collections import deque


class DialogueManager:

    def __init__(self, model, tokenizer, intent_classifier, sentiment_analyzer, max_history=5):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_classifier = intent_classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.history = deque(maxlen=max_history)

    def generate_response(self, user_input):
        self.history.append(user_input)

        # Intent recognition
        intent = self.intent_classifier(user_input)[0]['label']
        print(f"Detected intent: {intent}")

        # Sentiment analysis
        sentiment = self.sentiment_analyzer(user_input)[0]['label']
        print(f"Detected sentiment: {sentiment}")

        input_ids = [self.tokenizer.word2idx[token] for token in self.tokenizer.tokenize(" ".join(self.history))]
        input_tensor = torch.LongTensor([input_ids]).to(self.model.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output_ids = output[0].argmax(dim=-1).squeeze().tolist()

        response_tokens = [self.tokenizer.idx2word[idx] for idx in output_ids if idx != self.tokenizer.word2idx['<PAD>']]
        response = " ".join(response_tokens)
        self.history.append(response)
        return response
