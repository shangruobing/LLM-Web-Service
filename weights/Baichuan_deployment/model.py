import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from core.model import AbstractModel

MODEL_PATH = "Baichuan2-13B-Chat"


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
        model = model.to('cuda:0')
        self.model = model
        self.tokenizer = tokenizer

    def chat_with_model(self, question, history):
        messages = [{"role": "user", "content": question}]
        response = self.model.chat(self.tokenizer, messages)
        return response, None
