from flask import request, jsonify
from transformers import AutoTokenizer, AutoModel

import sys

sys.path.append("..")

from core.webservice import WebService

MODEL_PATH = "ChatGLM3-6B"


def __init_chatglm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device='cuda:0')
    model = model.eval()
    return tokenizer, model


tokenizer, model = __init_chatglm()

webService = WebService(__name__, MODEL_PATH, tokenizer, model)

app = webService.create_app()

app.view_functions.pop('chat')


@app.route('/api/llm/chat', endpoint="chat", methods=['POST'])
def chat():
    try:
        question = request.get_json().get("question")
        history = request.get_json().get("history")
        response, history = model.chat(tokenizer, question, history=history)
        return jsonify({"message": response, "history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
