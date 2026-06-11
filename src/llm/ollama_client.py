from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class OllamaMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: List[OllamaMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
        raise ValueError(f"Unexpected Ollama response format: {data.keys()}")

    def is_available(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
