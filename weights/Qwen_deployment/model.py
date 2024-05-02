from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model import AbstractModel

MODEL_PATH = "/weights/Qwen_deployment/Qwen-14B-ChatQwen-14B-Chat"


class ChatModel(AbstractModel):

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
        self.model = model.to('cuda:0')
