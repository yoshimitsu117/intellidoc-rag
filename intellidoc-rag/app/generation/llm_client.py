"""IntelliDoc RAG — LLM Client (OpenAI & Gemini)."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import openai

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenAI and Google Gemini."""

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.async_client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key
            )
            self.model = self.settings.openai_model
        elif self.provider == "gemini":
            self.client = openai.OpenAI(
                api_key=self.settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.async_client = openai.AsyncOpenAI(
                api_key=self.settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.model = self.settings.gemini_model
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(f"LLM client initialized: provider={self.provider}, model={self.model}")

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion synchronously."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def agenerate(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion asynchronously."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def astream(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream a completion token by token."""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
