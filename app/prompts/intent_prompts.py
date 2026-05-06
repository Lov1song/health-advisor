"""意图识别 Prompt 模板"""


INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类器，需要将用户的健康咨询消息分类为以下三个类别之一。

## 类别定义

1. **mental** — 心理健康
   - 情绪问题: 焦虑、抑郁、烦躁、恐惧、孤独
   - 压力管理: 工作压力、学业压力、生活压力
   - 睡眠问题: 失眠、多梦、睡眠质量差
   - 人际关系: 家庭矛盾、社交困难、职场关系
   - 自我认知: 自卑、迷茫、缺乏动力
   - 心理创伤: 应激事件、丧失、重大变故

2. **nutrition** — 营养饮食
   - 饮食建议: 减脂饮食、增肌饮食、健康食谱
   - 菜谱推荐: 早餐/午餐/晚餐推荐、特定食材菜谱
   - 营养分析: 食物热量、营养成分、搭配方案
   - 健康指标: BMI 计算、基础代谢、每日热量需求
   - 特殊饮食: 糖尿病饮食、低盐饮食、素食搭配
   - 食品安全: 食物禁忌、过敏原、食材选购

3. **general** — 通用健康
   - 运动健身: 运动方案、健身指导、运动损伤
   - 疾病常识: 感冒、发烧、高血压、糖尿病科普
   - 养生保健: 中医养生、季节保健、作息建议
   - 用药指导: 常见药物、用药注意事项
   - 体检解读: 指标解读、异常应对
   - 其他不属于前两类的健康相关问题

## 分类规则
- 如果消息同时涉及多个类别，选择**最核心**的意图
- "失眠" 如果侧重情绪原因 → mental，如果问助眠食物 → nutrition
- 不确定时倾向 general（兜底类别）
- confidence 反映你对分类的确信程度 (0.0-1.0)

## 近期对话上下文
{context}

## 用户消息
{message}

请以 JSON 格式返回分类结果:
{{"intent": "mental|nutrition|general", "confidence": 0.0-1.0, "reasoning": "简要说明分类理由"}}
"""


def build_intent_prompt(message: str, context: str = "无") -> list[dict]:
    """构建意图识别的完整 messages"""
    return [
        {
            "role": "user",
            "content": INTENT_CLASSIFICATION_PROMPT.format(
                message=message,
                context=context,
            ),
        }
    ]
