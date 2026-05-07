"""统一 LLM 调用接口 — 封装 OpenAI-compatible 和 VLLM 后端"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("llm")
settings = get_settings()


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class LLMClient:
    """统一 LLM 客户端

    支持两种后端:
    - general: OpenAI-compatible API (意图识别 / 通用 Agent / 记忆整理)
    - vllm:    VLLM 部署的微调模型 (心理咨询 Agent)
    """

    def __init__(self) -> None:
        # 通用 LLM 客户端 (DeepSeek OpenAI-compatible)
        self._general = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        # VLLM 微调模型客户端
        self._vllm = AsyncOpenAI(
            api_key="EMPTY",  # VLLM 不需要 key
            base_url=settings.VLLM_BASE_URL,
        )

    # ---- 同步完整调用 ----

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        backend: str = "general",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """同步调用 LLM，返回完整响应"""
        client = self._vllm if backend == "vllm" else self._general
        model_name = model or (
            settings.GENERAL_MODEL if backend == "general" else settings.VLLM_MODEL
        )

        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            resp = await client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return LLMResponse(
                content=content,
                model=model_name,
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                },
                raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            )
        except Exception as e:
            await logger.aerror("llm_call_failed", backend=backend, model=model_name, error=str(e))
            raise


    # ---- 流式调用 ----

    async def stream(
        self,
        messages: list[dict],
        model: str | None = None,
        backend: str = "general",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """流式调用 LLM，逐 token yield"""
        client = self._vllm if backend == "vllm" else self._general
        model_name = model or (
            settings.GENERAL_MODEL if backend == "general" else settings.VLLM_MODEL
        )

        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            await logger.aerror("llm_stream_failed", backend=backend, model=model_name, error=str(e))
            raise

    # ---- JSON 结构化输出 ----

    async def complete_json(
        self,
        messages: list[dict],
        model: str | None = None,
        backend: str = "general",
        temperature: float = 0.3,
    ) -> dict:
        """调用 LLM 并解析 JSON 响应"""
        resp = await self.complete(
            messages=messages,
            model=model,
            backend=backend,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        content = resp.content.strip()

        # 剥离 markdown 代码块
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(l for l in lines if not l.startswith("```")).strip()

        # 标准解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 宽松解析：LLM 有时在字符串中输出字面换行符 / 控制字符
        try:
            return json.JSONDecoder(strict=False).decode(content)
        except json.JSONDecodeError:
            raise


# ---- 单例 ----

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
