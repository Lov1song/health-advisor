"""营养师 Agent — Prompt 模板"""

from app.prompts.system_prompts import NUTRITION_AGENT_SYSTEM


def build_nutrition_messages(
    user_message: str,
    short_term_memory: list[dict] | None = None,
    user_profile: dict | None = None,
    rag_context: list[dict] | None = None,
    kg_context: list[dict] | None = None,
    tool_results: list[dict] | None = None,
) -> list[dict]:
    """构建营养师 Agent 的多层 prompt

    层次结构:
    1. system: 角色设定
    2. system: 用户画像 (个性化约束: 过敏/慢性病/偏好)
    3. system: 近期对话摘要
    4. system: RAG 检索结果 (菜谱/知识)
    5. system: 工具调用结果 (BMI/TDEE 等)
    6. user:   当前用户消息
    """
    messages: list[dict] = []

    # Layer 1: 角色设定
    messages.append({"role": "system", "content": NUTRITION_AGENT_SYSTEM})

    # Layer 2: 用户画像
    if user_profile:
        profile_text = _format_nutrition_profile(user_profile)
        messages.append({
            "role": "system",
            "content": f"## 用户健康档案\n{profile_text}\n请严格结合以上信息，注意过敏原和饮食禁忌。",
        })

    # Layer 3: 短期记忆
    if short_term_memory:
        memory_lines = []
        for m in short_term_memory[-5:]:
            role = "用户" if m.get("role") == "user" else "助手"
            memory_lines.append(f"{role}: {m.get('content', '')[:200]}")
        messages.append({
            "role": "system",
            "content": f"## 近期对话\n" + "\n".join(memory_lines),
        })

    # Layer 4: RAG 检索上下文
    if rag_context:
        rag_text = _format_rag_context(rag_context)
        messages.append({
            "role": "system",
            "content": f"## 菜谱检索结果\n{rag_text}\n请基于以上检索结果推荐菜谱，并结合用户需求筛选。",
        })

    # Layer 5: 知识图谱上下文
    if kg_context:
        kg_text = _format_kg_context(kg_context)
        messages.append({
            "role": "system",
            "content": f"## 营养学知识\n{kg_text}\n请参考以上知识给出科学的营养建议。",
        })

    # Layer 6: 工具调用结果
    if tool_results:
        tool_text = _format_tool_results(tool_results)
        messages.append({
            "role": "system",
            "content": f"## 健康指标计算结果\n{tool_text}\n请解读以上指标并给出针对性建议。",
        })

    # Layer 7: 用户消息
    messages.append({"role": "user", "content": user_message})

    return messages


def _format_nutrition_profile(profile: dict) -> str:
    parts = []
    if profile.get("age"):
        parts.append(f"- 年龄: {profile['age']}")
    if profile.get("gender"):
        gender_map = {"male": "男", "female": "女"}
        parts.append(f"- 性别: {gender_map.get(profile['gender'], profile['gender'])}")
    if profile.get("height_cm"):
        parts.append(f"- 身高: {profile['height_cm']} cm")
    if profile.get("weight_kg"):
        parts.append(f"- 体重: {profile['weight_kg']} kg")
    if profile.get("allergies"):
        parts.append(f"- ⚠️ 过敏原: {', '.join(profile['allergies'])}")
    if profile.get("chronic_conditions"):
        parts.append(f"- ⚠️ 慢性病: {', '.join(profile['chronic_conditions'])}")
    if profile.get("dietary_preferences"):
        prefs = profile["dietary_preferences"]
        if isinstance(prefs, dict):
            pref_items = [f"{k}={v}" for k, v in prefs.items()]
            parts.append(f"- 饮食偏好: {', '.join(pref_items)}")
    return "\n".join(parts) if parts else "暂无详细信息"


def _format_rag_context(contexts: list[dict]) -> str:
    parts = []
    for i, ctx in enumerate(contexts[:5], 1):
        name = ctx.get("name", ctx.get("title", f"菜谱{i}"))
        desc = ctx.get("description", ctx.get("content", ""))[:300]
        score = ctx.get("score", "")
        score_str = f" (相关度: {score:.2f})" if isinstance(score, (int, float)) else ""
        parts.append(f"### {i}. {name}{score_str}\n{desc}")
    return "\n\n".join(parts) if parts else "未找到相关菜谱"


def _format_kg_context(contexts: list[dict]) -> str:
    parts = []
    for ctx in contexts[:5]:
        entity = ctx.get("entity", "")
        relation = ctx.get("relation", "")
        target = ctx.get("target", "")
        parts.append(f"- {entity} → {relation} → {target}")
    return "\n".join(parts) if parts else "无相关知识"


def _format_tool_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        tool = r.get("tool", "unknown")
        output = r.get("result", {})
        parts.append(f"**{tool}**: {output}")
    return "\n".join(parts) if parts else "无工具结果"
