"""Agent 基类 — 所有 Agent 的统一接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from app.core.state import AgentState
from app.llm.client import get_llm_client
from app.utils.logger import get_logger


class BaseAgent(ABC):
    """智能体基类

    每个 Agent 实现两个核心方法:
    - run(): 同步执行，返回完整响应
    - stream(): 流式执行，逐 token yield
    """

    def __init__(self, name: str):
        self.name = name
        self.llm = get_llm_client()
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """同步执行 Agent，更新 state 中的 response 字段"""
        ...

    @abstractmethod
    async def stream(self, state: AgentState) -> AsyncGenerator[str, None]:
        """流式执行 Agent，逐 token yield，同时更新 state"""
        ...

    def _get_user_message(self, state: AgentState) -> str:
        """从 state 中提取最新的用户消息"""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _get_short_term_memory(self, state: AgentState) -> list[dict]:
        return state.get("short_term_memory", [])

    def _get_long_term_memory(self, state: AgentState) -> list[dict]:
        return state.get("long_term_memory", [])

    def _get_user_profile(self, state: AgentState) -> dict:
        return state.get("user_profile", {})
