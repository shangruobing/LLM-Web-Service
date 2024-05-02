from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model import AbstractModel

MODEL_PATH = "Qwen-14B-Chat"


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
        model = model.to('cuda:0')
        self.model = model
        self.tokenizer = tokenizer
