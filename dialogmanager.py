# Dialogue Manager
import torch
from collections import deque


# Dialogue Manager
class DialogueManager:

    def __init__(self, model, device, tokenizer, intent_classifier, sentiment_analyzer, persona, max_history=5):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.intent_classifier = intent_classifier
        self.sentiment_analyzer = sentiment_analyzer
        # self.knowledge_base = knowledge_base
        self.persona = persona
        self.history = deque(maxlen=max_history)
        self.context = ""

    def generate_response(self, user_input):
        self.history.append(user_input)

        # Intent recognition
        intent = self.intent_classifier(user_input)[0]['label']
        print(f"Detected intent: {intent}")

        # Sentiment analysis
        sentiment = self.sentiment_analyzer(user_input)[0]['label']
        print(f"Detected sentiment: {sentiment}")

        # Update context
        self.context = " ".join(self.history)

        # Persona modeling
        persona_response = self.get_persona_response(intent, sentiment)
        print(f"Persona response: {persona_response}")

        # input_ids = [self.tokenizer.word2idx[token] for token in self.tokenizer.tokenize(self.context + " " + relevant_knowledge + " " + persona_response)]
        # input_ids = [self.tokenizer.word2idx[token] for token in self.tokenizer.encode(self.context + " " + persona_response)]
        input_ids = self.tokenizer.encode(self.context + " " + persona_response)
        input_tensor = torch.LongTensor([input_ids]).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output_id = output[0].argmax().item()

        response_token = self.tokenizer.idx2word.get(output_id, "<UNK>")
        response = response_token
        self.history.append(response)
        self.context = " ".join(self.history)
        return response

    def get_persona_response(self, intent, sentiment):
        if intent == "greeting":
            return self.persona.get_greeting()
        elif sentiment == "positive":
            return self.persona.get_positive_response()
        elif sentiment == "negative":
            return self.persona.get_negative_response()
        else:
            return ""