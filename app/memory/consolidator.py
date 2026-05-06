"""记忆整理器 — LLM 将短期对话压缩为长期摘要"""

from __future__ import annotations

from app.llm.client import get_llm_client
from app.prompts.consolidation_prompts import build_consolidation_messages
from app.utils.logger import get_logger

logger = get_logger("memory.consolidator")


class MemoryConsolidator:
    """将最近 N 轮对话整理为长期记忆摘要"""

    def __init__(self) -> None:
        self.llm = get_llm_client()

    async def consolidate(
        self,
        conversation: list[dict],
        current_profile: dict | None = None,
    ) -> dict:
        """整理对话，返回结构化摘要

        Returns:
            {
              "summary": str,
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
            result = await self.llm.complete_json(
                messages=messages,
                temperature=0.3,
            )
            await logger.ainfo(
                "consolidation_done",
                topics=result.get("key_topics", []),
            )
            return result
        except Exception as e:
            await logger.aerror("consolidation_failed", error=str(e))
            return _empty_result()


def _empty_result() -> dict:
    return {
        "summary": "",
        "profile_updates": {},
        "key_topics": [],
        "emotional_state": "",
    }
