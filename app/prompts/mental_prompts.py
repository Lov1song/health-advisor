"""心理咨询 Agent — Prompt 模板"""

from app.prompts.system_prompts import MENTAL_AGENT_SYSTEM


def build_mental_messages(
    user_message: str,
    short_term_memory: list[dict] | None = None,
    user_profile: dict | None = None,
) -> list[dict]:
    """构建心理咨询 Agent 的多层 prompt

    层次结构:
    1. system: 角色设定 + 安全规则
    2. system: 用户画像注入 (个性化)
    3. system: 近期对话摘要 (连续性)
    4. user:   当前用户消息
    """
    messages: list[dict] = []

    # Layer 1: 角色设定
    messages.append({"role": "system", "content": MENTAL_AGENT_SYSTEM})

    # Layer 2: 用户画像
    if user_profile:
        profile_text = _format_profile(user_profile)
        messages.append({
            "role": "system",
            "content": f"## 用户画像\n{profile_text}\n请结合以上信息提供个性化的心理支持。",
        })

    # Layer 3: 短期记忆
    if short_term_memory:
        memory_text = _format_memory(short_term_memory)
        messages.append({
            "role": "system",
            "content": f"## 近期对话记录\n{memory_text}\n请保持对话的连续性和一致性。",
        })

    # Layer 4: 用户当前消息
    messages.append({"role": "user", "content": user_message})

    return messages


def _format_profile(profile: dict) -> str:
    """格式化用户画像为可读文本"""
    parts = []
    if profile.get("age"):
        parts.append(f"- 年龄: {profile['age']}")
    if profile.get("gender"):
        gender_map = {"male": "男", "female": "女", "other": "其他"}
        parts.append(f"- 性别: {gender_map.get(profile['gender'], profile['gender'])}")
    if profile.get("emotional_baseline"):
        eb = profile["emotional_baseline"]
        if isinstance(eb, dict):
            for k, v in eb.items():
                parts.append(f"- {k}: {v}")
    if profile.get("chronic_conditions"):
        parts.append(f"- 慢性病史: {', '.join(profile['chronic_conditions'])}")
    return "\n".join(parts) if parts else "暂无详细信息"


def _format_memory(memories: list[dict]) -> str:
    """格式化近期对话记录"""
    lines = []
    for m in memories[-5:]:  # 最近 5 轮
        role = "用户" if m.get("role") == "user" else "助手"
        content = m.get("content", "")[:200]  # 截断过长内容
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "无历史对话"
