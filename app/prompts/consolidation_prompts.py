"""记忆整理 Agent — Prompt 模板"""


CONSOLIDATION_PROMPT = """你是一个记忆整理助手，负责从对话记录中提取关键信息并分类存档。

## 任务 1: 生成对话摘要
将对话压缩为 2-3 句话的摘要，保留:
- 用户的核心诉求
- 给出的关键建议
- 重要的决策或变化

## 任务 2: 记忆分类与重要性评估
从以下五种类型中选择最适合的一种:

| 类型 | 说明 | 典型内容 | 重要性范围 |
|---|---|---|---|
| health_fact | 客观健康事实，长期有效 | 过敏原、慢性病、用药、体检异常 | 0.8–1.0 |
| goal | 用户设定的健康目标 | 体重目标、运动计划、戒烟 | 0.6–0.9 |
| event | 显著健康事件 | 就医、手术、体重骤变 | 0.5–0.8 |
| preference | 偏好与习惯 | 饮食偏好、运动习惯、作息 | 0.3–0.7 |
| emotional_pattern | 情绪与心理规律 | 压力触发、情绪化进食、焦虑模式 | 0.4–0.7 |

**is_pinned 规则**: memory_type 为 health_fact 且 importance >= 0.8 时设为 true，其他情况设为 false。

## 任务 3: 提取用户画像更新
检查是否有需要更新的用户信息:
- 新提到的身体数据 (体重变化、身高等)
- 新发现的过敏原或饮食禁忌
- 情绪状态变化
- 新的健康目标
- 新的慢性病/用药信息

## 当前用户画像
{current_profile}

## 对话记录
{conversation}

请以 JSON 格式返回:
{{
  "summary": "对话摘要文本",
  "memory_type": "health_fact",
  "importance": 0.9,
  "is_pinned": true,
  "profile_updates": {{
    "field_name": "new_value"
  }},
  "key_topics": ["主题1", "主题2"],
  "emotional_state": "用户当前情绪状态描述"
}}

注意:
- profile_updates 只包含需要更新的字段，没有变化则为空对象
- 可更新的字段: age, gender, height_cm, weight_kg, allergies, chronic_conditions, dietary_preferences, emotional_baseline
- importance 必须是 0.0–1.0 之间的浮点数
"""


def build_consolidation_messages(
    conversation: list[dict],
    current_profile: dict | None = None,
) -> list[dict]:
    conv_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in conversation
    )
    profile_text = str(current_profile) if current_profile else "暂无"
    return [
        {
            "role": "user",
            "content": CONSOLIDATION_PROMPT.format(
                current_profile=profile_text,
                conversation=conv_text,
            ),
        }
    ]
