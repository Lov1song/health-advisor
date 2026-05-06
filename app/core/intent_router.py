"""意图识别路由 — LLM-based 三分类"""

from __future__ import annotations

from app.core.state import AgentState
from app.llm.client import get_llm_client
from app.prompts.intent_prompts import build_intent_prompt
from app.utils.logger import get_logger

logger = get_logger("intent_router")

VALID_INTENTS = {"mental", "nutrition", "general"}
CONFIDENCE_THRESHOLD = 0.7  # 低于此阈值 fallback 到 general


async def classify_intent(
    message: str,
    context: str = "无",
) -> tuple[str, float, str]:
    """意图分类

    Returns:
        (intent, confidence, reasoning)
    """
    llm = get_llm_client()
    prompt_messages = build_intent_prompt(message=message, context=context)

    try:
        result = await llm.complete_json(
            messages=prompt_messages,
            temperature=0.1,  # 低温度保证分类稳定性
        )

        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        # 校验
        if intent not in VALID_INTENTS:
            await logger.awarn("invalid_intent", raw_intent=intent)
            intent = "general"
            confidence = 0.5

        # 低置信度 fallback
        if confidence < CONFIDENCE_THRESHOLD:
            await logger.ainfo(
                "low_confidence_fallback",
                original_intent=intent,
                confidence=confidence,
            )
            intent = "general"

        await logger.ainfo(
            "intent_classified",
            intent=intent,
            confidence=confidence,
            reasoning=reasoning[:100],
        )
        return intent, confidence, reasoning

    except Exception as e:
        await logger.aerror("intent_classification_failed", error=str(e))
        return "general", 0.5, f"分类失败: {e}"


async def classify_intent_node(state: AgentState) -> AgentState:
    """LangGraph 节点 — 意图分类

    从 state.messages 提取用户消息和上下文，执行分类。
    """
    # 提取最新用户消息
    user_message = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # 构建上下文 (短期记忆中最近 3 轮)
    memory = state.get("short_term_memory", [])
    if memory:
        context_lines = []
        for m in memory[-6:]:  # 最近 3 轮 (user+assistant)
            role = "用户" if m.get("role") == "user" else "助手"
            context_lines.append(f"{role}: {m.get('content', '')[:100]}")
        context = "\n".join(context_lines)
    else:
        context = "无"

    intent, confidence, reasoning = await classify_intent(user_message, context)

    state["intent"] = intent  # type: ignore
    state["intent_confidence"] = confidence
    return state
