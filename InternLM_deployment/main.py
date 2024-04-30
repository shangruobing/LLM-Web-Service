import torch
from flask import request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys

sys.path.append("..")

from webservice import WebService

MODEL_PATH = "InternLM-chat-20b"


def __init_internlm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model = model.eval()
    return tokenizer, model


tokenizer, model = __init_internlm()

webService = WebService(__name__, MODEL_PATH, tokenizer, model)

app = webService.create_app()

app.view_functions.pop('chat')


@app.route('/api/llm/chat', endpoint="chat", methods=['POST'])
def chat():
    try:
        question = request.get_json().get("question")
        response, history = model.chat(tokenizer, question)
        return jsonify({"message": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
