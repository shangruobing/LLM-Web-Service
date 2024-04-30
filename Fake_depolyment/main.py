from flask import jsonify

import sys

sys.path.append("..")

from webservice import WebService

MODEL_PATH = "Fake-LLM"


class FakeTokenizer:
    pass


class FakeModel:
    def __init__(self):
        self.name = "FakeLLM"

    def chat(self, tokenizer, question, history):
        return f"{self.name}: {question}", history


def __init_fake_llm():
    return FakeTokenizer(), FakeModel()


tokenizer, model = __init_fake_llm()

webService = WebService(__name__, MODEL_PATH, tokenizer, model)

app = webService.create_app()

app.view_functions.pop('ping')


@app.route('/api/llm/ping', endpoint="ping", methods=['GET'])
def custom_ping():
    return jsonify({"message": "custom ping"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
