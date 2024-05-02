from core.model import AbstractModel

MODEL_PATH = "Fake-LLM"


class FakeTokenizer:
    pass


class FakeModel:
    def __init__(self):
        self.name = "FakeLLM"

    def chat(self, tokenizer, question, history):
        return f"{self.name}: {question}", history


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        self.model = model
        self.tokenizer = tokenizer
