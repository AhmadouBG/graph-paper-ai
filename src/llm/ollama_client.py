from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class OllamaMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5vl:3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: List[Dict[str, str]] | List[OllamaMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
    ) -> str:
        ollama_messages: List[Dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, OllamaMessage):
                entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            else:
                entry = {"role": msg["role"], "content": msg["content"]}

            if images and i == len(messages) - 1 and entry["role"] == "user":
                encoded_images = []
                for img_path in images:
                    p = Path(img_path)
                    if p.exists():
                        encoded_images.append(base64.b64encode(p.read_bytes()).decode("utf-8"))
                if encoded_images:
                    entry["images"] = encoded_images

            ollama_messages.append(entry)

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": ollama_messages,
            "temperature": temperature,
            "stream": False,
        }
        if format:
            payload["format"] = format

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
