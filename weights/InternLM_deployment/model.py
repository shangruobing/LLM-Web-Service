import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.model import AbstractModel

MODEL_PATH = "weights/InternLM_deployment/InternLM-chat-20b"


class ChatModel(AbstractModel):

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.model = model.eval()
