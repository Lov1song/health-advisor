"""记忆整理器 — LLM 将短期对话压缩为结构化长期记忆"""

from __future__ import annotations

from app.llm.client import get_llm_client
from app.prompts.consolidation_prompts import build_consolidation_messages
from app.utils.logger import get_logger

logger = get_logger("memory.consolidator")

_VALID_TYPES = {"health_fact", "goal", "event", "preference", "emotional_pattern"}


class MemoryConsolidator:
    def __init__(self) -> None:
        self.llm = get_llm_client()

    async def consolidate(
        self,
        conversation: list[dict],
        current_profile: dict | None = None,
    ) -> dict:
        """整理对话，返回结构化记忆

        Returns:
            {
              "summary": str,
              "memory_type": str,        # health_fact | goal | event | preference | emotional_pattern
              "importance": float,       # 0.0–1.0
              "is_pinned": bool,
              "profile_updates": dict,
              "key_topics": list[str],
              "emotional_state": str,
            }
        """
        if not conversation:
            return _empty_result()

        messages = build_consolidation_messages(
            conversation=conversation,
            current_profile=current_profile,
        )
        try:
            result = await self.llm.complete_json(messages=messages, temperature=0.3)
            memory_type = result.get("memory_type", "event")
            if memory_type not in _VALID_TYPES:
                memory_type = "event"

            importance = float(result.get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))

            # health_fact 且 importance >= 0.8 时强制钉选
            is_pinned = (memory_type == "health_fact" and importance >= 0.8) or bool(result.get("is_pinned", False))

            await logger.ainfo(
                "consolidation_done",
                memory_type=memory_type,
                importance=importance,
                is_pinned=is_pinned,
                topics=result.get("key_topics", []),
            )
            return {
                "summary": result.get("summary", ""),
                "memory_type": memory_type,
                "importance": importance,
                "is_pinned": is_pinned,
                "profile_updates": result.get("profile_updates", {}),
                "key_topics": result.get("key_topics", []),
                "emotional_state": result.get("emotional_state", ""),
            }
        except Exception as e:
            await logger.aerror("consolidation_failed", error=str(e))
            return _empty_result()


def _empty_result() -> dict:
    return {
        "summary": "",
        "memory_type": "event",
        "importance": 0.5,
        "is_pinned": False,
        "profile_updates": {},
        "key_topics": [],
        "emotional_state": "",
    }
