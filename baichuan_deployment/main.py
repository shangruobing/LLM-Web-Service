import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

app = Flask(__name__)
CORS(app)

MODEL_PATH = ""


def __init_baichuan():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model = model.to('cuda:0')
    return tokenizer, model


tokenizer, model = __init_baichuan()


@app.route('/api/baichuan/ping', methods=['GET'])
def ping():
    return jsonify({"message": "baichuan is running!"}), 200


@app.route('/api/baichuan/chat', methods=['POST'])
def chat():
    try:
        question = request.json['question']
        messages = [{"role": "user", "content": question}]
        response = model.chat(tokenizer, messages)
        return jsonify({"message": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
