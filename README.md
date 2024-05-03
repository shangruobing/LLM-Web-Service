# LLM Web Service

LLM-Web-Service deploy various open-source Large Language Models (LLMs) with Flask.

# Introduction

We provide a simple browser interface and RESTful APIs for you to chat with LLMs.
> This repository doesn't provide the download of LLMs.
> You can download them from their official homepage or Huggingface.

## API

For each model, we provide the RESTful APIs for calling.

We use `Ping` API to test the connection of service, and use `Chat` API to chat with LLM.

> For each model, we provide the following APIs to use.

- `GET` http://127.0.0.1:5000/api/llm/ping
- `POST` http://127.0.0.1:5000/api/llm/chat

## Browser Interface

We provide a simple browser interface for you to chat with LLMs.
You can access it by visiting http://127.0.0.1:5000.

## LangServe

You can lunch the LangsServe in `serve.py` file.
You can access it by visiting http://127.0.0.1:8000/llm/playground.

# Quick Start

## Requirement

LLMs always has many parameters, so you must have at least one GPU to run them.

This table shows the GPU usage of each model on our experimental devices.

|                                     LLM                                      |       Device        | GPU Usage |
|:----------------------------------------------------------------------------:|:-------------------:|:---------:|
|     [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat)     | NVIDIA RTX 4090 24G |    16G    |
|           [ChatGLM3-7b](https://huggingface.co/THUDM/chatglm3-6b)            | NVIDIA RTX 4090 24G |    12G    |
| [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) |   NVIDIA A100 80G   |    55G    |
|          [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)          |   NVIDIA A100 80G   |    55G    |
| [InternLM-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b)  |   NVIDIA A100 80G   |    78G    |

## Install

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Launch

1. Assume your model is named `Chatbot`.
2. Create a folder named `Chatbot_deplotment` in `weights` folder.
3. Place the model weight files in the `Chatbot_deplotment` folder.
4. Write the model loading code in `model.py`.
5. Configure the model name in `config.py`.
6. Execute the following command to launch the service.

```shell
# Launch the service
python main.py
# or detach running
nohup python main.py > log.txt 2>&1 &
```

## Use with Shell

```shell
curl http://127.0.0.1:5000/api/llm/ping
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"question": "Hello!"}' \
     http://127.0.0.1:5000/api/llm/chat
```

## Use with Python

```python
import requests


def ping():
    print("ping")
    url = "http://127.0.0.1:5000/api/llm/ping"
    response = requests.get(url)
    print(response.text)


def chat():
    print("chat")
    url = "http://127.0.0.1:5000/api/llm/chat"
    headers = {"Content-Type": "application/json"}
    data = {"question": "Hello!"}
    response = requests.post(url, headers=headers, json=data)
    print(response.text)


if __name__ == '__main__':
    ping()
    chat()

```

## Coding Instruction

You need to implement the `AbstractModel` class in `core/model.py` and write the model loading code in `model.py`.

```python
from transformers import AutoTokenizer, AutoModel

from core.model import AbstractModel

MODEL_PATH = "Your-Weight-Path"


class ChatModel(AbstractModel):

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device='cuda:0')
        model = model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def chat_with_model(self, question, history):
        message, history = self.model.chat(self.tokenizer, question, history)
        return message, history

```