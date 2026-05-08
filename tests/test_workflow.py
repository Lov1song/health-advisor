"""LangGraph 工作流 — 集成测试

测试策略: mock LLM 调用，验证工作流节点编排和状态流转。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.state import AgentState


class TestWorkflowNodes:
    """测试各工作流节点"""

    @pytest.mark.asyncio
    async def test_load_memory_initializes_state(self):
        from app.core.workflow import load_memory_node

        state: AgentState = {
            "user_id": "test-user",
            "session_id": "test-session",
            "messages": [{"role": "user", "content": "hello"}],
            "intent": "general",
            "intent_confidence": 0.0,
            "response": "",
        }

        result = await load_memory_node(state, {})
        assert result["short_term_memory"] == []
        assert result["long_term_memory"] == []
        assert result["user_profile"] == {}
        assert result["turn_count"] == 0

    @pytest.mark.asyncio
    async def test_save_memory_increments_turn(self):
        from app.core.workflow import save_memory_node

        state: AgentState = {
            "user_id": "test",
            "session_id": "test",
            "messages": [],
            "intent": "general",
            "intent_confidence": 0.0,
            "response": "test response",
            "turn_count": 4,
            "should_consolidate": False,
        }

        result = await save_memory_node(state, {})
        assert result["turn_count"] == 5
        assert result["should_consolidate"] is False

    @pytest.mark.asyncio
    async def test_save_memory_triggers_consolidation_at_10(self):
        from app.core.workflow import save_memory_node

        state: AgentState = {
            "user_id": "test",
            "session_id": "test",
            "messages": [],
            "intent": "general",
            "intent_confidence": 0.0,
            "response": "test",
            "turn_count": 9,  # 第 9 轮 → +1 = 10 → 触发
            "should_consolidate": False,
        }

        result = await save_memory_node(state, {})
        assert result["turn_count"] == 10
        assert result["should_consolidate"] is True


class TestRouting:
    """测试意图路由"""

    def test_route_mental(self):
        from app.core.workflow import route_by_intent

        state = {"intent": "mental"}
        assert route_by_intent(state) == "mental"

    def test_route_nutrition(self):
        from app.core.workflow import route_by_intent

        state = {"intent": "nutrition"}
        assert route_by_intent(state) == "nutrition"

    def test_route_general(self):
        from app.core.workflow import route_by_intent

        state = {"intent": "general"}
        assert route_by_intent(state) == "general"

    def test_route_default(self):
        from app.core.workflow import route_by_intent

        state = {}
        assert route_by_intent(state) == "general"

    def test_should_consolidate_true(self):
        from app.core.workflow import should_consolidate

        assert should_consolidate({"should_consolidate": True}) is True

    def test_should_consolidate_false(self):
        from app.core.workflow import should_consolidate

        assert should_consolidate({"should_consolidate": False}) is False
        assert should_consolidate({}) is False


class TestMentalAgent:
    """测试心理 Agent 的危机检测"""

    def test_crisis_detection(self):
        from app.agents.mental_agent import MentalAgent

        agent = MentalAgent()
        assert agent._detect_crisis("我不想活了") is True
        assert agent._detect_crisis("我想自杀") is True
        assert agent._detect_crisis("今天心情不好") is False
        assert agent._detect_crisis("我很开心") is False
