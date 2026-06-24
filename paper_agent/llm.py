from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests

from .config import LLMConfig

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str


class LLMClient:
    """Small OpenAI-compatible chat client.

    The project uses the Chat Completions shape because many providers expose
    the same `/chat/completions` endpoint. Swap `base_url`, `model`, and
    `api_key_env` in config to use another compatible provider.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.config.api_key_env)

    @property
    def is_configured(self) -> bool:
        return bool(self.config.enabled and self.api_key)

    def complete(self, messages: list[ChatMessage]) -> str:
        if not self.is_configured:
            raise RuntimeError(
                f"LLM is not configured. Set {self.config.api_key_env} or disable LLM calls."
            )
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()
