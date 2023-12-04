import json
import requests


def ping():
    print("ping")
    url = "http://127.0.0.1:7000/api/chatglm/ping"
    response = requests.get(url)
    print(response.text)


def chat():
    print("chat")
    url = "http://127.0.0.1:7000/api/chatglm/chat"
    headers = {"Content-Type": "application/json"}
    data = {"question": "Hello!"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.text)


if __name__ == '__main__':
    ping()
    chat()
