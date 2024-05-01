import torch
from flask import request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys

sys.path.append("..")

from core.webservice import WebService

MODEL_PATH = "Baichuan2-13B-Chat"


def __init_baichuan():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model = model.to('cuda:0')
    return tokenizer, model


tokenizer, model = __init_baichuan()

webService = WebService(__name__, MODEL_PATH, tokenizer, model)

app = webService.create_app()

app.view_functions.pop('chat')


@app.route('/api/llm/chat', endpoint="chat", methods=['POST'])
def chat():
    try:
        question = request.json['question']
        messages = [{"role": "user", "content": question}]
        response = model.chat(tokenizer, messages)
        return jsonify({"message": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
