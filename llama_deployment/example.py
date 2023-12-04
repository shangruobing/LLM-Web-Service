import json
import requests


def ping():
    print("ping")
    url = "http://127.0.0.1:5000/api/llama/ping"
    response = requests.get(url)
    print(response.text)


def chat():
    print("chat")
    url = "http://127.0.0.1:5000/api/llama/chat"
    headers = {"Content-Type": "application/json"}
    data = {"question": "Hello!"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


def dialog():
    print("dialog")
    url = "http://127.0.0.1:5000/api/llama/dialog"
    headers = {"Content-Type": "application/json"}
    data = {
        "dialog": [
            {"role": "system", "content": "You are a helpful assistant!"},
            {"role": "user", "content": "根据休息区位置和物体汉堡进行任务规划."},
            {"role": "assistant", "content": "step1: 移动到休息区。动作：移动，位置：休息区..."},
            {"role": "user", "content": "上方是一个例子，根据输入的指令中的物体和位置进行任务规划:去找到休息区"}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


if __name__ == '__main__':
    ping()
    chat()
    dialog()
