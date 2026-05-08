"""意图识别路由 — 规则快路径 + LLM 兜底"""

from __future__ import annotations

from app.core.state import AgentState
from app.llm.client import get_llm_client
from app.prompts.intent_prompts import build_intent_prompt
from app.utils.logger import get_logger

logger = get_logger("intent_router")

VALID_INTENTS = {"mental", "nutrition", "general"}
CONFIDENCE_THRESHOLD = 0.7

# ---- 规则分类关键词 ----

_MENTAL_KW = frozenset([
    "焦虑", "抑郁", "失眠", "心理", "压力", "孤独", "自杀", "自残",
    "情绪", "心情", "烦恼", "恐惧", "悲伤", "绝望", "崩溃", "沮丧",
    "内耗", "心理咨询", "心理健康", "创伤", "应激", "恐慌",
])

_NUTRITION_KW = frozenset([
    "菜谱", "食谱", "营养", "减肥", "热量", "卡路里", "BMI", "饮食",
    "食材", "蛋白质", "脂肪", "碳水", "维生素", "矿物质", "吃什么",
    "怎么吃", "健身餐", "增肌", "减脂餐", "食物", "食品",
])


def _fast_classify(message: str) -> tuple[str, float] | None:
    """基于关键词的规则分类，返回 (intent, confidence) 或 None（交给 LLM）"""
    mental = sum(1 for kw in _MENTAL_KW if kw in message)
    nutrition = sum(1 for kw in _NUTRITION_KW if kw in message)

    if mental == 0 and nutrition == 0:
        return None
    if mental > nutrition:
        return "mental", min(0.75 + mental * 0.05, 0.95)
    if nutrition > mental:
        return "nutrition", min(0.75 + nutrition * 0.05, 0.95)
    return None  # 平局，交给 LLM 裁决


async def classify_intent(
    message: str,
    context: str = "无",
) -> tuple[str, float, str]:
    """意图分类 — 规则快路径优先，无法确定时 LLM 兜底

    Returns:
        (intent, confidence, reasoning)
    """
    # 快路径：规则分类，省去一次 LLM 调用 (~300-500ms)
    fast = _fast_classify(message)
    if fast:
        intent, confidence = fast
        await logger.ainfo("intent_fast_classified", intent=intent, confidence=confidence)
        return intent, confidence, "rule-based"

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
