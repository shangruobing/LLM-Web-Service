from flask_cors import CORS
from flask import Flask, request, jsonify, render_template

from core.model import AbstractModel


class WebService:
    def __init__(self, name, model_path, model: AbstractModel):
        """
        Init
        Args:
            name: app module name
            model_path: model path
            model: model
        """
        self.name = name
        self.model_path = model_path
        self.model = model
        self.app = self._init_app()
        self.endpoints = [
            self._ping,
            self._chat,
            self._home,
        ]

    def create_app(self):
        """
        Create a Flask app
        Returns: A Flask instance

        """
        self._add_default_endpoints()
        return self.app

    def _add_default_endpoints(self):
        for endpoint in self.endpoints:
            endpoint()

    def _init_app(self):
        app = Flask(self.name)
        app.json.ensure_ascii = False
        # app.template_folder = "templates"
        # app.static_folder = "static"
        CORS(app)
        return app

    def _ping(self):
        @self.app.route(rule='/api/llm/ping', endpoint="ping", methods=['GET'])
        def ping():
            return jsonify({"message": f"{self.model_path} is running!"}), 200

    def _chat(self):
        @self.app.route(rule='/api/llm/chat', endpoint="chat", methods=['POST'])
        def chat():
            try:
                question = request.get_json().get("question")
                history = request.get_json().get("history")
                message, history = self.model.chat_with_model(question=question, history=history)
                return jsonify({"message": message, "history": history}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _home(self):
        @self.app.route(rule='/', endpoint="/", methods=['GET'])
        def home():
            return render_template(
                'LLM.html',
            )
