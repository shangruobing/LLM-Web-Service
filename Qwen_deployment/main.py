from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

MODEL_PATH = "Qwen-14B-Chat"


def __init_Qwen():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
    model = model.to('cuda:0')
    return tokenizer, model


tokenizer, model = __init_Qwen()


@app.route('/api/qwen/ping', methods=['GET'])
def ping():
    return jsonify({"message": f"{MODEL_PATH} is running!"}), 200


@app.route('/api/qwen/chat', methods=['POST'])
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
