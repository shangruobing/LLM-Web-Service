import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.model import AbstractModel

MODEL_PATH = "InternLM-chat-20b"


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        model = model.eval()
        self.model = model
        self.tokenizer = tokenizer
