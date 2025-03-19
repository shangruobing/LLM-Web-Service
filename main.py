import importlib

from config import MODULE_PATH
from core.webservice import WebService

MODEL_PATH = MODULE_PATH.split("_")[0]

module = importlib.import_module("weights." + MODULE_PATH + ".model")

ChatModel = getattr(module, "ChatModel")

chatModel = ChatModel()

webService = WebService(__name__, MODEL_PATH, chatModel)

app = webService.create_app()

# DIY: modify the default ping endpoint
# app.view_functions.pop('ping')
#
#
# @app.route('/api/llm/ping', endpoint="ping", methods=['GET'])
# def custom_ping():
#     return jsonify({"message": "custom ping"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
