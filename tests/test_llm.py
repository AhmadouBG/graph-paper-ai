from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm.ollama_client import OllamaClient, OllamaMessage


class TestOllamaMessage:
    def test_creates_message(self):
        msg = OllamaMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestOllamaClient:
    def test_chat_sends_correct_payload(self):
        client = OllamaClient(base_url="http://test:11434", model="test-model")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "answer"}}

        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            result = client.chat([OllamaMessage(role="user", content="hi")])

        assert result == "answer"
        mock_post.assert_called_once_with(
            "http://test:11434/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.0,
                "stream": False,
            },
        )

    def test_chat_overrides_model(self):
        client = OllamaClient(base_url="http://test:11434", model="default-model")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "answer"}}

        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            client.chat([OllamaMessage(role="user", content="hi")], model="custom")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "custom"

    def test_chat_handles_legacy_response_format(self):
        client = OllamaClient(base_url="http://test:11434")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "legacy answer"}

        with patch.object(client._client, "post", return_value=mock_resp):
            result = client.chat([OllamaMessage(role="user", content="hi")])

        assert result == "legacy answer"

    def test_chat_raises_on_bad_status(self):
        client = OllamaClient(base_url="http://test:11434")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp,
        )

        with patch.object(client._client, "post", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                client.chat([OllamaMessage(role="user", content="hi")])

    def test_chat_raises_on_unexpected_format(self):
        client = OllamaClient(base_url="http://test:11434")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unknown": "format"}

        with patch.object(client._client, "post", return_value=mock_resp):
            with pytest.raises(ValueError, match="Unexpected Ollama response"):
                client.chat([OllamaMessage(role="user", content="hi")])

    def test_is_available_returns_true_on_200(self):
        client = OllamaClient(base_url="http://test:11434")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200

        with patch.object(client._client, "get", return_value=mock_resp):
            assert client.is_available()

    def test_is_available_returns_false_on_exception(self):
        client = OllamaClient(base_url="http://test:11434")

        with patch.object(client._client, "get", side_effect=Exception("connection refused")):
            assert not client.is_available()

    def test_close(self):
        client = OllamaClient(base_url="http://test:11434")
        with patch.object(client._client, "close") as mock_close:
            client.close()
        mock_close.assert_called_once()
