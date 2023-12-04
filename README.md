# LLM Web Service

LLM-Web-Service deploy various open-source Large Language Models (LLMs) with Flask.

# Introduction

This repository doesn't provide the download of LLMs, you can download them from their official homepage or Huggingface.

For each model, this repository provides the RESTful APIs for calling.

we use `Ping` API to test the connection of service, and use `Chat` API to chat with LLM.

> For each model, this repository provides the following APIs to use.

- GET http://127.0.0.1:5000/api/llm-name/ping
- POST http://127.0.0.1:5000/api/llm-name/chat

# Get Started

## Requirement

LLMs always has many parameters, so you must have at least one GPU to run them.

|     LLM      |       Device        | GPU Usage |
|:------------:|:-------------------:|:---------:|
|  llama2-7b   | Nvidia RTX 4090 24G |    16G    |
| chatglm2-7b  | Nvidia RTX 4090 24G |    12G    |
| baichuan-13b |   Nvidia A100 80G   |    55G    |

## Install

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Launch

```shell
python main.py
nohup python main.py > log.txt 2>&1 &disown
```

## Use with Shell

```shell
curl http://127.0.0.1:5000/api/llm-name/ping
curl -X POST -H "Content-Type: application/json" -d '{"question": "Hello!"}'http://127.0.0.1:5000/api/llm-name/chat
```

## Use with Python

```python
import json
import requests


def ping():
    print("ping")
    url = "http://127.0.0.1:5000/api/llm-name/ping"
    response = requests.get(url)
    print(response.text)


def chat():
    print("chat")
    url = "http://127.0.0.1:5000/api/llm-name/chat"
    headers = {"Content-Type": "application/json"}
    data = {"question": "Hello!"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


if __name__ == '__main__':
    ping()
    chat()

```
