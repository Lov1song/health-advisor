"""通用健康 Agent — 覆盖运动、疾病科普、养生等综合问题"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from app.agents.base import BaseAgent
from app.core.state import AgentState
from app.prompts.system_prompts import GENERAL_AGENT_SYSTEM


class GeneralAgent(BaseAgent):
    """通用健康顾问 Agent

    特点:
    - 使用通用 LLM (GPT-4o-mini 等)
    - 无需 RAG / 工具调用，直接基于知识回答
    - 兜底 Agent: 当意图不明确时默认路由到此
    """

    def __init__(self):
        super().__init__("general")

    def _build_messages(self, state: AgentState) -> list[dict]:
        """构建 prompt"""
        messages: list[dict] = [{"role": "system", "content": GENERAL_AGENT_SYSTEM}]

        # 注入用户画像
        profile = self._get_user_profile(state)
        if profile:
            profile_parts = []
            for key in ("age", "gender", "height_cm", "weight_kg", "chronic_conditions"):
                if profile.get(key):
                    profile_parts.append(f"- {key}: {profile[key]}")
            if profile_parts:
                messages.append({
                    "role": "system",
                    "content": "## 用户信息\n" + "\n".join(profile_parts),
                })

        # 注入长期记忆摘要
        lt_memory = self._get_long_term_memory(state)
        if lt_memory:
            lt_lines = [f"- {m['summary']}" for m in lt_memory[-3:] if m.get("summary")]
            if lt_lines:
                messages.append({
                    "role": "system",
                    "content": "## 历史对话摘要\n" + "\n".join(lt_lines),
                })

        # 注入短期记忆
        memory = self._get_short_term_memory(state)
        if memory:
            mem_lines = []
            for m in memory[-5:]:
                role = "用户" if m.get("role") == "user" else "助手"
                mem_lines.append(f"{role}: {m.get('content', '')[:200]}")
            messages.append({
                "role": "system",
                "content": "## 近期对话\n" + "\n".join(mem_lines),
            })

        # 用户消息
        messages.append({"role": "user", "content": self._get_user_message(state)})
        return messages

    async def run(self, state: AgentState) -> AgentState:
        messages = self._build_messages(state)

        resp = await self.llm.complete(
            messages=messages,
            backend="general",
            temperature=0.7,
            max_tokens=1536,
        )

        state["response"] = resp.content
        state["tool_calls"] = []
        return state

    async def stream(self, state: AgentState) -> AsyncGenerator[str, None]:
        messages = self._build_messages(state)

        full_response = []
        async for token in self.llm.stream(
            messages=messages, backend="general", temperature=0.7, max_tokens=1536
        ):
            full_response.append(token)
            yield token

        state["response"] = "".join(full_response)
        state["tool_calls"] = []
