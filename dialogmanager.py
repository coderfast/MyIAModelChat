# Dialogue Manager
import torch
from collections import deque


# Dialogue Manager
class DialogueManager:

    def __init__(self, model, tokenizer, intent_classifier, sentiment_analyzer, knowledge_base, persona, max_history=5):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_classifier = intent_classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.knowledge_base = knowledge_base
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

        # Knowledge retrieval
        # relevant_knowledge = self.retrieve_relevant_knowledge(user_input)
        # print(f"Relevant knowledge: {relevant_knowledge}")

        # Persona modeling
        persona_response = self.get_persona_response(intent, sentiment)
        print(f"Persona response: {persona_response}")

        # input_ids = [self.tokenizer.word2idx[token] for token in self.tokenizer.tokenize(self.context + " " + relevant_knowledge + " " + persona_response)]
        input_ids = [self.tokenizer.word2idx[token] for token in self.tokenizer.encode(self.context + " " + persona_response)]
        input_tensor = torch.LongTensor([input_ids]).to(self.model.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output_ids = output[0].argmax(dim=-1).squeeze().tolist()

        response_tokens = [self.tokenizer.idx2word[idx] for idx in output_ids if idx != self.tokenizer.word2idx['<PAD>']]
        response = " ".join(response_tokens)
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