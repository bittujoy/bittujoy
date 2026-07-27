import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import requests


class LLMClientError(Exception):
    pass


@dataclass
class LLMResponse:
    raw_response: str
    latency_seconds: float
    provider: str


class LLMClient(ABC):
    def __init__(self, server_url: str, model_name: str, api_key: str, auth_header: str = "Authorization", auth_prefix: str = "Bearer") -> None:
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.auth_header = auth_header.strip() or "Authorization"
        self.auth_prefix = auth_prefix.strip()

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def connect(self) -> bool:
        url = f"{self.server_url}/v1/models/{self.model_name}"
        headers = {
            self.auth_header: self._build_auth_value(),
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200

    def analyze(self, prompt: str) -> LLMResponse:
        url = f"{self.server_url}/v1/chat/completions"
        headers = {
            self.auth_header: self._build_auth_value(),
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        start = time.monotonic()
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        latency = time.monotonic() - start
        if response.status_code != 200:
            raise LLMClientError(
                f"OpenAI request failed: {response.status_code} {response.text}"
            )
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(raw_response=message.strip(), latency_seconds=latency, provider="OpenAI")


    def _build_auth_value(self) -> str:
        if not self.api_key:
            return ""
        if self.auth_prefix:
            return f"{self.auth_prefix} {self.api_key}".strip()
        return self.api_key


class OllamaClient(LLMClient):
    def connect(self) -> bool:
        url = f"{self.server_url}/models/{self.model_name}"
        headers = {self.auth_header: self._build_auth_value()} if self.api_key else {}
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200

    def analyze(self, prompt: str) -> LLMResponse:
        url = f"{self.server_url}/api/models/{self.model_name}/generate"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.auth_header] = self._build_auth_value()

        payload = {
            "prompt": prompt,
            "max_tokens": 800,
            "temperature": 0.2,
            "stream": False,
        }
        start = time.monotonic()
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        latency = time.monotonic() - start
        if response.status_code != 200:
            raise LLMClientError(
                f"Ollama request failed: {response.status_code} {response.text}"
            )
        data = response.json()
        output = data.get("output")
        if isinstance(output, list):
            output = "\n".join(str(item) for item in output)
        return LLMResponse(raw_response=str(output).strip(), latency_seconds=latency, provider="Ollama")
