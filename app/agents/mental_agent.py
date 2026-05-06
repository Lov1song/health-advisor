"""心理咨询 Agent — 基于 VLLM 部署的 QLora 微调模型"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from app.agents.base import BaseAgent
from app.core.state import AgentState
from app.prompts.mental_prompts import build_mental_messages


# 危机关键词检测
_CRISIS_KEYWORDS = [
    "自杀", "自残", "不想活", "想死", "结束生命", "跳楼", "割腕",
    "活着没意思", "自我了断", "遗书", "告别",
]

_CRISIS_RESPONSE_SUFFIX = """

---
⚠️ **我注意到你可能正在经历非常困难的时刻。你的感受很重要，请一定要寻求专业帮助：**
- 🆘 全国 24 小时心理援助热线: **400-161-9995**
- 🆘 北京心理危机研究与干预中心: **010-82951332**
- 🆘 生命热线: **400-821-1215**

**你不是一个人，请立即拨打以上电话，会有专业人员帮助你。**
"""


class MentalAgent(BaseAgent):
    """心理咨询 Agent

    特点:
    - 使用 VLLM 部署的 Deepseek-8B-QLora 微调模型
    - 温度参数调低 (0.4) 保证回复稳定性
    - 内置危机关键词检测 + 安全兜底
    """

    def __init__(self):
        super().__init__("mental")

    def _detect_crisis(self, text: str) -> bool:
        """检测危机信号"""
        return any(kw in text for kw in _CRISIS_KEYWORDS)

    async def run(self, state: AgentState) -> AgentState:
        user_message = self._get_user_message(state)
        has_crisis = self._detect_crisis(user_message)

        messages = build_mental_messages(
            user_message=user_message,
            short_term_memory=self._get_short_term_memory(state),
            user_profile=self._get_user_profile(state),
        )

        try:
            resp = await self.llm.complete(
                messages=messages,
                backend="vllm",
                temperature=0.4,
                max_tokens=1024,
            )
            response_text = resp.content
        except Exception as e:
            await self.logger.awarning("vllm_fallback", error=str(e))
            # VLLM 不可用时 fallback 到通用模型
            resp = await self.llm.complete(
                messages=messages,
                backend="general",
                temperature=0.4,
                max_tokens=1024,
            )
            response_text = resp.content

        # 危机信号附加安全提示
        if has_crisis:
            response_text += _CRISIS_RESPONSE_SUFFIX
            await self.logger.awarning("crisis_detected", user_id=state.get("user_id"))

        state["response"] = response_text
        state["tool_calls"] = []
        return state

    async def stream(self, state: AgentState) -> AsyncGenerator[str, None]:
        user_message = self._get_user_message(state)
        has_crisis = self._detect_crisis(user_message)

        messages = build_mental_messages(
            user_message=user_message,
            short_term_memory=self._get_short_term_memory(state),
            user_profile=self._get_user_profile(state),
        )

        full_response = []
        try:
            backend = "vllm"
            async for token in self.llm.stream(
                messages=messages, backend=backend, temperature=0.4, max_tokens=1024
            ):
                full_response.append(token)
                yield token
        except Exception:
            backend = "general"
            async for token in self.llm.stream(
                messages=messages, backend=backend, temperature=0.4, max_tokens=1024
            ):
                full_response.append(token)
                yield token

        if has_crisis:
            for char in _CRISIS_RESPONSE_SUFFIX:
                full_response.append(char)
                yield char

        state["response"] = "".join(full_response)
        state["tool_calls"] = []
