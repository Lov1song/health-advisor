"""意图识别路由 — 单元测试

测试策略: mock LLM 调用，验证分类逻辑和 fallback 机制。
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.core.intent_router import classify_intent, CONFIDENCE_THRESHOLD


@pytest.fixture
def mock_llm():
    """Mock LLM client"""
    with patch("app.core.intent_router.get_llm_client") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_mental_intent(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "intent": "mental",
            "confidence": 0.92,
            "reasoning": "用户提到压力和失眠",
        }

        intent, confidence, reasoning = await classify_intent("我最近压力很大，总是失眠")
        assert intent == "mental"
        assert confidence >= CONFIDENCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_nutrition_intent(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "intent": "nutrition",
            "confidence": 0.88,
            "reasoning": "用户询问菜谱推荐",
        }

        intent, confidence, _ = await classify_intent("推荐一些低脂的晚餐食谱")
        assert intent == "nutrition"
        assert confidence >= CONFIDENCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_general_intent(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "intent": "general",
            "confidence": 0.85,
            "reasoning": "用户询问运动建议",
        }

        intent, confidence, _ = await classify_intent("每天应该运动多久")
        assert intent == "general"

    @pytest.mark.asyncio
    async def test_low_confidence_fallback(self, mock_llm):
        """低置信度 fallback 到 general"""
        mock_llm.complete_json.return_value = {
            "intent": "mental",
            "confidence": 0.5,  # 低于阈值
            "reasoning": "不确定",
        }

        intent, confidence, _ = await classify_intent("今天天气真好")
        assert intent == "general"  # fallback

    @pytest.mark.asyncio
    async def test_invalid_intent_fallback(self, mock_llm):
        """无效意图值 fallback 到 general"""
        mock_llm.complete_json.return_value = {
            "intent": "invalid_type",
            "confidence": 0.9,
            "reasoning": "test",
        }

        intent, _, _ = await classify_intent("测试消息")
        assert intent == "general"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, mock_llm):
        """LLM 调用失败 fallback 到 general"""
        mock_llm.complete_json.side_effect = Exception("API error")

        intent, confidence, _ = await classify_intent("测试消息")
        assert intent == "general"
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_context_passed(self, mock_llm):
        """验证上下文被正确传入"""
        mock_llm.complete_json.return_value = {
            "intent": "mental",
            "confidence": 0.9,
            "reasoning": "结合上下文",
        }

        await classify_intent("我还是很难受", context="用户: 我心情不好\n助手: 能告诉我发生了什么吗")

        # 检查 LLM 被调用且消息包含 context
        call_args = mock_llm.complete_json.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        prompt_text = messages[0]["content"]
        assert "我心情不好" in prompt_text
