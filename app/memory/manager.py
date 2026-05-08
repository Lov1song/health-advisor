"""分层记忆管理器 — 短期 (DB 消息) + 长期 (摘要)"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import LongTermMemory, Message, UserProfile
from app.utils.logger import get_logger

logger = get_logger("memory.manager")


class MemoryManager:
    """管理用户对话记忆的加载与持久化"""

    async def load_short_term(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> list[dict]:
        """加载最近 SHORT_TERM_WINDOW 条消息"""
        window = get_settings().SHORT_TERM_WINDOW
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(window)
        )
        messages = result.scalars().all()
        return [{"role": m.role, "content": m.content} for m in reversed(messages)]

    async def load_long_term(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> list[dict]:
        """加载用户最近 LONG_TERM_TOP_K 条长期记忆摘要"""
        top_k = get_settings().LONG_TERM_TOP_K
        result = await db.execute(
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(top_k)
        )
        memories = result.scalars().all()
        return [
            {
                "summary": m.summary,
                "key_topics": m.key_topics or [],
                "emotional_state": m.emotional_state or "",
                "turn_range": f"{m.turn_start}-{m.turn_end}",
            }
            for m in reversed(memories)
        ]

    async def load_user_profile(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> dict:
        """加载用户健康档案"""
        result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        profile = result.scalar_one_or_none()
        if not profile:
            return {}
        return {
            "age": profile.age,
            "gender": profile.gender,
            "height_cm": profile.height_cm,
            "weight_kg": profile.weight_kg,
            "allergies": profile.allergies or [],
            "chronic_conditions": profile.chronic_conditions or [],
            "dietary_preferences": profile.dietary_preferences or {},
            "emotional_baseline": profile.emotional_baseline or {},
        }

    async def save_message(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        role: str,
        content: str,
        intent: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """持久化单条对话消息"""
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            metadata_=metadata or {},
        )
        db.add(msg)
        await db.flush()

    async def update_user_profile(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        updates: dict,
    ) -> None:
        """将 LLM 提取的 profile_updates 字段写回 UserProfile 表"""
        if not updates:
            return
        _allowed = {
            "age", "gender", "height_cm", "weight_kg",
            "allergies", "chronic_conditions", "dietary_preferences", "emotional_baseline",
        }
        result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        profile = result.scalar_one_or_none()
        if profile is None:
            profile = UserProfile(user_id=user_id)
            db.add(profile)

        changed = False
        for field, value in updates.items():
            if field in _allowed and value is not None:
                setattr(profile, field, value)
                changed = True

        if changed:
            await db.flush()
            await logger.ainfo("user_profile_updated", user_id=str(user_id), fields=list(updates))

    async def save_long_term(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        summary: str,
        key_topics: list[str] | None = None,
        emotional_state: str | None = None,
        turn_start: int = 0,
        turn_end: int = 0,
    ) -> None:
        """持久化一条长期记忆"""
        memory = LongTermMemory(
            user_id=user_id,
            summary=summary,
            key_topics=key_topics or [],
            emotional_state=emotional_state,
            turn_start=turn_start,
            turn_end=turn_end,
        )
        db.add(memory)
        await db.flush()
        await logger.ainfo(
            "long_term_memory_saved",
            user_id=str(user_id),
            turn_range=f"{turn_start}-{turn_end}",
        )
