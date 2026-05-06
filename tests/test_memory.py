"""Phase 5 记忆系统 — 单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


class TestMemoryConsolidator:
    """记忆整理器测试"""

    @pytest.mark.asyncio
    async def test_consolidate_returns_summary(self):
        """正常整理返回结构化摘要"""
        mock_result = {
            "summary": "用户询问了减脂饮食建议，助手推荐了低热量高蛋白食谱。",
            "profile_updates": {"weight_kg": 72.5},
            "key_topics": ["减脂", "饮食"],
            "emotional_state": "积极主动",
        }

        with patch("app.memory.consolidator.get_llm_client") as mock_get:
            mock_llm = AsyncMock()
            mock_llm.complete_json.return_value = mock_result
            mock_get.return_value = mock_llm

            from app.memory.consolidator import MemoryConsolidator
            consolidator = MemoryConsolidator()
            result = await consolidator.consolidate(
                conversation=[
                    {"role": "user", "content": "帮我制定减脂食谱"},
                    {"role": "assistant", "content": "建议选择高蛋白低热量食物..."},
                ]
            )

        assert result["summary"] == mock_result["summary"]
        assert result["key_topics"] == ["减脂", "饮食"]

    @pytest.mark.asyncio
    async def test_consolidate_empty_conversation(self):
        """空对话直接返回空结果"""
        from app.memory.consolidator import MemoryConsolidator
        consolidator = MemoryConsolidator()
        result = await consolidator.consolidate(conversation=[])

        assert result["summary"] == ""
        assert result["profile_updates"] == {}

    @pytest.mark.asyncio
    async def test_consolidate_llm_error_fallback(self):
        """LLM 异常时返回空结果，不抛出异常"""
        with patch("app.memory.consolidator.get_llm_client") as mock_get:
            mock_llm = AsyncMock()
            mock_llm.complete_json.side_effect = Exception("API Error")
            mock_get.return_value = mock_llm

            from app.memory.consolidator import MemoryConsolidator
            consolidator = MemoryConsolidator()
            result = await consolidator.consolidate(
                conversation=[{"role": "user", "content": "test"}]
            )

        assert result["summary"] == ""
        assert result["profile_updates"] == {}


class TestMemoryManager:
    """记忆管理器测试"""

    @pytest.mark.asyncio
    async def test_load_short_term_returns_messages(self):
        """短期记忆按时间正序返回"""
        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_msg1.content = "你好"

        mock_msg2 = MagicMock()
        mock_msg2.role = "assistant"
        mock_msg2.content = "你好！有什么可以帮助你的？"

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_msg2, mock_msg1]
        mock_db.execute.return_value = mock_result

        from app.memory.manager import MemoryManager
        manager = MemoryManager()

        with patch("app.memory.manager.get_settings") as mock_settings:
            mock_settings.return_value.SHORT_TERM_WINDOW = 10
            messages = await manager.load_short_term(mock_db, uuid.uuid4())

        # DB 返回降序 [msg2(assistant), msg1(user)]，reversed 后 → [msg1(user), msg2(assistant)]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_load_long_term_returns_summaries(self):
        """长期记忆按时间正序返回"""
        mock_mem = MagicMock()
        mock_mem.summary = "上次对话摘要"
        mock_mem.key_topics = ["健康", "运动"]
        mock_mem.emotional_state = "平静"
        mock_mem.turn_start = 0
        mock_mem.turn_end = 10

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_mem]
        mock_db.execute.return_value = mock_result

        from app.memory.manager import MemoryManager
        manager = MemoryManager()

        with patch("app.memory.manager.get_settings") as mock_settings:
            mock_settings.return_value.LONG_TERM_TOP_K = 5
            memories = await manager.load_long_term(mock_db, uuid.uuid4())

        assert len(memories) == 1
        assert memories[0]["summary"] == "上次对话摘要"
        assert memories[0]["key_topics"] == ["健康", "运动"]

    @pytest.mark.asyncio
    async def test_save_long_term(self):
        """保存长期记忆测试"""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()  # add() 是同步方法，避免 coroutine warning

        from app.memory.manager import MemoryManager
        manager = MemoryManager()
        await manager.save_long_term(
            db=mock_db,
            user_id=uuid.uuid4(),
            summary="测试摘要",
            key_topics=["测试"],
            emotional_state="平静",
            turn_start=0,
            turn_end=10,
        )

        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()


class TestWorkflowMemoryIntegration:
    """工作流记忆集成测试"""

    @pytest.mark.asyncio
    async def test_run_workflow_accepts_long_term_memory(self):
        """run_workflow 接受 long_term_memory 参数"""
        from app.core.workflow import run_workflow

        with patch("app.core.workflow.get_workflow") as mock_get:
            mock_workflow = AsyncMock()
            mock_workflow.ainvoke.return_value = {
                "intent": "general",
                "intent_confidence": 0.9,
                "response": "测试回复",
                "tool_calls": [],
                "turn_count": 1,
                "should_consolidate": False,
            }
            mock_get.return_value = mock_workflow

            result = await run_workflow(
                user_id="user_123",
                session_id="session_456",
                user_message="你好",
                long_term_memory=[{"summary": "历史摘要", "key_topics": ["健康"]}],
            )

        assert result["response"] == "测试回复"
        # 验证 long_term_memory 被传入 initial_state
        call_args = mock_workflow.ainvoke.call_args
        initial_state = call_args.args[0] if call_args.args else call_args.kwargs.get("input")
        if initial_state:
            assert initial_state["long_term_memory"] == [{"summary": "历史摘要", "key_topics": ["健康"]}]
