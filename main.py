import json
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
        模型名称声明
        """
        return "BaseLLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        模型参数
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
        Override Langchain的LLM模型调用接口
        """
        return self._chat(prompt)

    def request(self, url: str, prompt: str):
        """
        请求HTTP接口
        Args:
            url: URL
            prompt: 提示词
        """
        try:
            headers = {"Content-Type": "application/json"}
            data = {"question": prompt}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except Exception:
            raise ConnectionError(f"{self._llm_type} connect fail, URL: {url}")

    @abstractmethod
    def _chat(self, prompt: str):
        """
        调用模型进行对话
        Args:
            prompt: 提示词
        """
        raise NotImplemented


class ChatGLM(BaseLLM):
    url = "http://127.0.0.1:5000/api/chatglm/chat"

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3-6b"

    def _chat(self, prompt: str):
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

    uvicorn.run(app, host="localhost", port=7000)
