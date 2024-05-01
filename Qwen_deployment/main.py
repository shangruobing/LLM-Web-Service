from flask import request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.append("..")

from core.webservice import WebService

MODEL_PATH = "Qwen-14B-Chat"


def __init_Qwen():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
    model = model.to('cuda:0')
    return tokenizer, model


tokenizer, model = __init_Qwen()

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
