from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    def chat_with_model(self, question, history):
        message, history = self.model.chat(self.tokenizer, question, history=history)
        return message, history
