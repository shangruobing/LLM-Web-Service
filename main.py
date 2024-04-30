from abc import ABC, abstractmethod
from typing import Mapping, Any, Optional, List

import requests
from fastapi import FastAPI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langserve import add_routes


class BaseLLM(LLM, ABC):

    @property
    def _llm_type(self) -> str:
        """
        Declaration of the model type
        """
        return "BaseLLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Declaration of the model params
        """
        return {"name": self._llm_type}

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """
        Override Langchain's LLM model call interface
        """
        return self._chat(prompt)

    def request(self, url: str, prompt: str) -> dict:
        """
        Request HTTP interface
        Args:
            url: URL
            prompt: prompt to send
        """
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"question": prompt}
            return requests.post(url, headers=headers, json=payload).json()
        except Exception:
            raise ConnectionError(f"{self._llm_type} connect fail, URL: {url}")

    @abstractmethod
    def _chat(self, prompt: str) -> str:
        """
        Chat with Model
        Args:
            prompt: prompt
        """
        raise NotImplemented


class ChatGLM(BaseLLM):
    url = "http://127.0.0.1:5000/api/chatglm/chat"

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3-6b"

    def _chat(self, prompt: str) -> str:
        response = self.request(url=self.url, prompt=prompt)
        return response.get("message")


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatGLM(),
    path="/chatglm",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
