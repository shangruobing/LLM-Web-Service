import requests


def ping():
    print("ping")
    url = "http://127.0.0.1:5000/api/llm/ping"
    response = requests.get(url)
    print(response.text)


def chat(question, history: []):
    url = "http://127.0.0.1:5000/api/llm/chat"
    headers = {"Content-Type": "application/json"}
    data = {"question": question, "history": history}
    return requests.post(url, headers=headers, json=data)


response = chat("Hello!", [
    ["你好", "你好呀"],
])

response = response.json()
print(response)
print(response["message"])
print(response["history"])
