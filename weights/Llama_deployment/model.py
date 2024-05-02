from typing import List, Optional, Union

from core.model import AbstractModel
from llama import Llama, Dialog
from llama.generation import Message

"""
Entry point of the program for generating text using a pretrained model.
ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
temperature (float, optional): The temperature value for controlling randomness in generation. Defaults to 0.6.
top_p (float, optional): The top-p sampling parameter for controlling diversity in generation. Defaults to 0.9.
max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
max_gen_len (int, optional): The maximum length of generated sequences. 
    If None, it will be set to the model's max sequence length. Defaults to None.
"""
ckpt_dir: str = "Llama-2-7b-chat"
tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
MODEL_PATH: str = "Llama-2-7b-chat"
temperature: float = 0
top_p: float = 0.9
max_seq_len: int = 550
max_batch_size: int = 8
max_gen_len: Optional[int] = None


def chat_with_llama(generator, question: Union[str, List[Message]]):
    if isinstance(question, list):
        dialogs: List[Dialog] = [
            question
        ]
    else:
        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": "You are a helpful assistant, please answer the question"},
                {"role": "user", "content": question},
            ],
        ]
    return generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )


class ChatModel(AbstractModel):

    def _load_model(self):
        self.model = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def chat_with_model(self, question, history):
        response = chat_with_llama(self.model, question=question)
        return response, None
