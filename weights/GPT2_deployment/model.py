from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

from core.model import AbstractModel

MODEL_PATH = "weights/GPT2_deployment/GPT2"


class ChatModel(AbstractModel):

    def _load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
        self.text_generator = TextGenerationPipeline(self.model, self.tokenizer)

    def chat_with_model(self, question, history):
        outputs = self.text_generator(text_inputs=question, max_length=256, do_sample=True)
        message = outputs[0].get("generated_text").replace(" ", "")
        return message, None
