from transformers import AutoTokenizer, AutoModel

from core.model import AbstractModel

MODEL_PATH = "ChatGLM3-6B"


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device='cuda:0')
        model = model.eval()
        self.model = model
        self.tokenizer = tokenizer
