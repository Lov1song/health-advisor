"""聊天 API — 集成 LangGraph 多智能体工作流"""

import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.config import get_settings as _get_settings
from app.core.workflow import run_workflow
from app.agents.mental_agent import MentalAgent
from app.agents.nutrition_agent import NutritionAgent
from app.agents.general_agent import GeneralAgent
from app.core.intent_router import classify_intent
from app.db.models import ChatSession, Message, User, UserProfile
from app.db.postgres import async_session_factory
from app.memory import MemoryConsolidator, MemoryManager
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk, ToolCallInfo
from app.utils.logger import get_logger

router = APIRouter(tags=["chat"])
logger = get_logger("chat")

_memory_manager = MemoryManager()
_consolidator = MemoryConsolidator()

# Agent 单例 (用于 WebSocket 流式响应)
_agents = {
    "mental": MentalAgent(),
    "nutrition": NutritionAgent(),
    "general": GeneralAgent(),
}


# ============================================================
# 辅助函数
# ============================================================

async def _get_or_create_session(
    db: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID | None
) -> ChatSession:
    if session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session:
            return session

    session = ChatSession(user_id=user_id)
    db.add(session)
    await db.flush()
    return session


async def _save_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    intent: str | None = None,
    metadata: dict | None = None,
) -> Message:
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        metadata_=metadata or {},
    )
    db.add(msg)
    await db.flush()
    return msg


async def _load_recent_messages(db: AsyncSession, session_id: uuid.UUID, limit: int = 10) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def _load_user_profile(db: AsyncSession, user_id: uuid.UUID) -> dict:
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


# ============================================================
# REST 聊天接口
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """同步聊天 — 通过 LangGraph 工作流执行完整流程"""
    start_time = time.perf_counter()

    session = await _get_or_create_session(db, user.id, body.session_id)

    # 先加载历史上下文（不含本轮消息），再保存本轮用户消息
    short_term_memory = await _memory_manager.load_short_term(db, session.id)
    long_term_memory = await _memory_manager.load_long_term(db, user.id)
    user_profile = await _load_user_profile(db, user.id)

    await _save_message(db, session.id, "user", body.message)

    # 执行 LangGraph 工作流
    result = await run_workflow(
        user_id=str(user.id),
        session_id=str(session.id),
        user_message=body.message,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        user_profile=user_profile,
        turn_count=session.turn_count,
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # 保存助手回复
    await _save_message(
        db, session.id, "assistant", result["response"],
        intent=result["intent"],
        metadata={"intent_confidence": result["intent_confidence"], "latency_ms": elapsed_ms},
    )

    session.turn_count = result["turn_count"]
    session.last_active = datetime.now(timezone.utc)
    await db.flush()

    # 触发记忆整理 (每 CONSOLIDATION_INTERVAL 轮)
    if result.get("should_consolidate") and short_term_memory:
        try:
            consolidation = await _consolidator.consolidate(
                conversation=short_term_memory,
                current_profile=user_profile,
            )
            if consolidation.get("summary"):
                await _memory_manager.save_long_term(
                    db=db,
                    user_id=user.id,
                    summary=consolidation["summary"],
                    key_topics=consolidation.get("key_topics", []),
                    emotional_state=consolidation.get("emotional_state"),
                    turn_start=result["turn_count"] - 10,
                    turn_end=result["turn_count"],
                )
        except Exception as e:
            await logger.awarn("consolidation_skipped", error=str(e))

    tool_calls = [
        ToolCallInfo(
            tool=tc.get("tool", "unknown"),
            args=tc.get("args", {}),
            result=str(tc.get("result", "")),
        )
        for tc in result.get("tool_calls", [])
    ]

    return ChatResponse(
        session_id=session.id,
        intent=result["intent"],
        intent_confidence=result["intent_confidence"],
        response=result["response"],
        tool_calls=tool_calls,
        metadata={"latency_ms": elapsed_ms, "turn_count": result["turn_count"]},
    )


# ============================================================
# WebSocket 流式聊天
# ============================================================

@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """WebSocket 流式聊天

    流程: 认证 → 循环 { 接收消息 → 加载记忆 → 意图识别 → Agent 流式输出 → 持久化 → done }
    """
    await ws.accept()

    token = ws.query_params.get("token")
    if not token:
        await ws.send_json({"type": "error", "data": {"message": "缺少认证 token"}})
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    from app.api.deps import decode_access_token
    try:
        user_id_str = decode_access_token(token)
        user_id = uuid.UUID(user_id_str)
    except Exception:
        await ws.send_json({"type": "error", "data": {"message": "认证失败"}})
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await logger.ainfo("ws_connected", user_id=user_id_str)

    async with async_session_factory() as db:
        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                content = data.get("content", "")
                if not content.strip():
                    continue

                start_time = time.perf_counter()

                # 1) 加载会话 + 上下文
                session_id_raw = data.get("session_id")
                session_id = uuid.UUID(session_id_raw) if session_id_raw else None
                session = await _get_or_create_session(db, user_id, session_id)

                # 先加载历史，再保存本轮消息
                short_term_memory = await _memory_manager.load_short_term(db, session.id)
                long_term_memory = await _memory_manager.load_long_term(db, user_id)
                user_profile = await _load_user_profile(db, user_id)

                await _save_message(db, session.id, "user", content)

                # 2) 意图识别
                intent, confidence, _ = await classify_intent(content)
                await ws.send_json(
                    StreamChunk(type="intent", data={"intent": intent, "confidence": confidence}).model_dump()
                )

                # 3) 构建 state
                state = {
                    "user_id": user_id_str,
                    "session_id": str(session.id),
                    "messages": [{"role": "user", "content": content}],
                    "intent": intent,
                    "intent_confidence": confidence,
                    "short_term_memory": short_term_memory,
                    "long_term_memory": long_term_memory,
                    "user_profile": user_profile,
                    "tool_calls": [],
                    "rag_context": [],
                    "kg_context": [],
                    "turn_count": session.turn_count,
                    "should_consolidate": False,
                    "response": "",
                }

                # 4) Agent 流式输出
                agent = _agents.get(intent, _agents["general"])

                if intent == "nutrition":
                    await ws.send_json(
                        StreamChunk(type="tool", data={"tool": "planner", "status": "running"}).model_dump()
                    )

                full_response: list[str] = []
                try:
                    async for tok in agent.stream(state):
                        full_response.append(tok)
                        await ws.send_json(
                            StreamChunk(type="token", data={"content": tok}).model_dump()
                        )
                except Exception as e:
                    await logger.aerror("stream_error", error=str(e))
                    await ws.send_json(
                        StreamChunk(type="error", data={"message": "生成回复时出错"}).model_dump()
                    )
                    await db.rollback()
                    continue

                # 5) 持久化回复 + 更新会话
                response_text = "".join(full_response)
                await _save_message(db, session.id, "assistant", response_text, intent=intent)

                session.turn_count = (session.turn_count or 0) + 1
                session.last_active = datetime.now(timezone.utc)
                await db.flush()

                # 6) 记忆整理
                if session.turn_count % _get_settings().CONSOLIDATION_INTERVAL == 0 and short_term_memory:
                    try:
                        consolidation = await _consolidator.consolidate(short_term_memory, user_profile)
                        if consolidation.get("summary"):
                            await _memory_manager.save_long_term(
                                db=db,
                                user_id=user_id,
                                summary=consolidation["summary"],
                                key_topics=consolidation.get("key_topics", []),
                                emotional_state=consolidation.get("emotional_state"),
                                turn_start=session.turn_count - 10,
                                turn_end=session.turn_count,
                            )
                    except Exception as e:
                        await logger.awarn("ws_consolidation_skipped", error=str(e))

                await db.commit()

                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
                await ws.send_json(
                    StreamChunk(
                        type="done",
                        data={
                            "session_id": str(session.id),
                            "latency_ms": elapsed_ms,
                            "intent": intent,
                            "tool_calls": state.get("tool_calls", []),
                        },
                    ).model_dump()
                )

        except WebSocketDisconnect:
            await logger.ainfo("ws_disconnected", user_id=user_id_str)
        except Exception as e:
            await logger.aerror("ws_error", error=str(e))
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
