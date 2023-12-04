from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)


def __init_chatglm():
    tokenizer = AutoTokenizer.from_pretrained("chatglm3", trust_remote_code=True)
    model = AutoModel.from_pretrained("chatglm3", trust_remote_code=True, device='cuda:0')
    model = model.eval()
    return tokenizer, model


tokenizer, model = __init_chatglm()


@app.route('/api/chatglm/ping', methods=['GET'])
def ping():
    return jsonify({"message": "chatglm-3 is running!"}), 200


@app.route('/api/chatglm/chat', methods=['POST'])
def chat():
    try:
        question = request.get_json().get("question")
        history = request.get_json().get("history")
        response, history = model.chat(tokenizer, question, history=history)
        return jsonify({"message": response, "history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
