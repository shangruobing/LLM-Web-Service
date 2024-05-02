from transformers import AutoTokenizer, AutoModel

from core.model import AbstractModel

MODEL_PATH = "weights/ChatGLM_deployment/ChatGLM3-6B"


class ChatModel(AbstractModel):

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device='cuda:0').eval()
