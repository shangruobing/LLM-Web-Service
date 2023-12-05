import torch
from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

MODEL_PATH = "internlm-chat-20b"


def __init_internlm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    model = model.eval()
    return tokenizer, model


tokenizer, model = __init_internlm()


@app.route(f'/api/internlm/ping', methods=['GET'])
def ping():
    return jsonify({"message": f"{MODEL_PATH} is running!"}), 200


@app.route('/api/internlm/chat', methods=['POST'])
def chat():
    try:
        question = request.get_json().get("question")
        response, history = model.chat(tokenizer, question)
        return jsonify({"message": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
