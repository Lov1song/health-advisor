"""分层记忆管理器 — 短期 (DB 消息) + 长期 (语义检索 + 钉选事实)"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import LongTermMemory, Message, UserProfile
from app.memory.retriever import find_duplicate, search_similar, upsert_memory
from app.utils.logger import get_logger

logger = get_logger("memory.manager")


def _effective_importance(
    base: float,
    created_at: datetime,
    access_count: int,
    is_pinned: bool,
) -> float:
    """钉选记忆不衰减；普通记忆随时间衰减，被引用越多权重越高"""
    if is_pinned:
        return base
    age_days = max(0, (datetime.now(timezone.utc) - created_at).days)
    decay = max(0.1, 1.0 - 0.005 * age_days)
    boost = min(2.0, 1.0 + 0.1 * access_count)
    return base * decay * boost


def _fmt(m: LongTermMemory) -> dict:
    return {
        "summary": m.summary,
        "memory_type": m.memory_type or "event",
        "importance": m.importance or 0.5,
        "is_pinned": m.is_pinned or False,
        "key_topics": m.key_topics or [],
        "emotional_state": m.emotional_state or "",
        "turn_range": f"{m.turn_start}-{m.turn_end}",
    }


class MemoryManager:

    async def load_short_term(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> list[dict]:
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
        query_text: str | None = None,
    ) -> list[dict]:
        """混合检索：钉选事实始终注入 + 按当前问题语义匹配最相关记忆

        query_text 为 None 时降级为按时间倒序（向后兼容）。
        """
        cfg = get_settings()

        # ── 1. 始终加载所有钉选记忆（health_fact 类） ──────────────────
        pinned_rows = (
            await db.execute(
                select(LongTermMemory)
                .where(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.is_pinned == True,  # noqa: E712
                )
                .order_by(LongTermMemory.created_at.desc())
            )
        ).scalars().all()
        pinned_ids = {str(m.id) for m in pinned_rows}

        # ── 2. 语义检索非钉选记忆 ─────────────────────────────────────
        semantic_rows: list[LongTermMemory] = []
        if query_text:
            try:
                hits = await search_similar(
                    query_text=query_text,
                    user_id=str(user_id),
                    top_k=cfg.MEMORY_SEMANTIC_TOP_K,
                )
                # hits 是 ChromaDB embedding_id，与 PG 的 embedding_id 对应
                candidate_ids = [h["id"] for h in hits if h["id"] not in pinned_ids]
                score_map = {h["id"]: h["score"] for h in hits}

                if candidate_ids:
                    rows = (
                        await db.execute(
                            select(LongTermMemory).where(
                                LongTermMemory.user_id == user_id,
                                LongTermMemory.embedding_id.in_(candidate_ids),
                            )
                        )
                    ).scalars().all()

                    semantic_rows = sorted(
                        rows,
                        key=lambda m: score_map.get(m.embedding_id or "", 0.0)
                        * _effective_importance(
                            m.importance, m.created_at, m.access_count, m.is_pinned
                        ),
                        reverse=True,
                    )
            except Exception as e:
                await logger.aerror("semantic_retrieval_failed", error=str(e))

        # ── 3. 若语义和钉选都为空，降级为按时间倒序 ─────────────────────
        if not pinned_rows and not semantic_rows:
            fallback = (
                await db.execute(
                    select(LongTermMemory)
                    .where(LongTermMemory.user_id == user_id)
                    .order_by(LongTermMemory.created_at.desc())
                    .limit(cfg.LONG_TERM_TOP_K)
                )
            ).scalars().all()
            return [_fmt(m) for m in reversed(fallback)]

        # ── 4. 合并：钉选优先，语义结果补充至 LONG_TERM_TOP_K ──────────
        semantic_unique = [m for m in semantic_rows if str(m.id) not in pinned_ids]
        merged = list(pinned_rows) + semantic_unique
        top_n = merged[: cfg.LONG_TERM_TOP_K]

        # ── 5. 批量更新访问记录 ─────────────────────────────────────────
        if top_n:
            await db.execute(
                update(LongTermMemory)
                .where(LongTermMemory.id.in_([m.id for m in top_n]))
                .values(
                    access_count=LongTermMemory.access_count + 1,
                    last_accessed=datetime.now(timezone.utc),
                )
            )

        await logger.ainfo(
            "long_term_memory_loaded",
            user_id=str(user_id),
            pinned=len(pinned_rows),
            semantic=len(semantic_unique),
            total=len(top_n),
        )
        return [_fmt(m) for m in top_n]

    async def load_user_profile(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> dict:
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
        db.add(Message(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            metadata_=metadata or {},
        ))
        await db.flush()

    async def update_user_profile(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        updates: dict,
    ) -> None:
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
        memory_type: str = "event",
        importance: float = 0.5,
        is_pinned: bool = False,
        key_topics: list[str] | None = None,
        emotional_state: str | None = None,
        turn_start: int = 0,
        turn_end: int = 0,
    ) -> None:
        """去重写入：语义相似度 > 阈值则合并已有记忆，否则新建"""
        cfg = get_settings()

        # ── 去重检查 ────────────────────────────────────────────────────
        dup_embedding_id = await find_duplicate(
            text=summary,
            user_id=str(user_id),
            threshold=cfg.MEMORY_DEDUP_THRESHOLD,
        )

        if dup_embedding_id:
            existing = (
                await db.execute(
                    select(LongTermMemory).where(
                        LongTermMemory.embedding_id == dup_embedding_id,
                        LongTermMemory.user_id == user_id,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.summary = summary
                existing.importance = max(existing.importance, importance)
                existing.key_topics = key_topics or existing.key_topics
                existing.emotional_state = emotional_state or existing.emotional_state
                existing.turn_end = turn_end
                if is_pinned:
                    existing.is_pinned = True
                await db.flush()
                await upsert_memory(
                    memory_id=dup_embedding_id,
                    text=summary,
                    user_id=str(user_id),
                    memory_type=existing.memory_type,
                    importance=existing.importance,
                    is_pinned=existing.is_pinned,
                )
                await logger.ainfo(
                    "long_term_memory_merged",
                    user_id=str(user_id),
                    embedding_id=dup_embedding_id,
                )
                return

        # ── 新建记忆 ────────────────────────────────────────────────────
        memory = LongTermMemory(
            user_id=user_id,
            summary=summary,
            memory_type=memory_type,
            importance=importance,
            is_pinned=is_pinned,
            key_topics=key_topics or [],
            emotional_state=emotional_state,
            turn_start=turn_start,
            turn_end=turn_end,
        )
        db.add(memory)
        await db.flush()  # 获取 memory.id

        embedding_id = str(memory.id)
        await upsert_memory(
            memory_id=embedding_id,
            text=summary,
            user_id=str(user_id),
            memory_type=memory_type,
            importance=importance,
            is_pinned=is_pinned,
        )
        memory.embedding_id = embedding_id
        await db.flush()

        await logger.ainfo(
            "long_term_memory_saved",
            user_id=str(user_id),
            memory_type=memory_type,
            importance=importance,
            is_pinned=is_pinned,
            turn_range=f"{turn_start}-{turn_end}",
        )
