"""ORM 模型 — 用户、会话、消息"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.postgres import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    profile = relationship("UserProfile", back_populates="user", uselist=False, lazy="selectin")
    sessions = relationship("ChatSession", back_populates="user", lazy="dynamic")
    long_term_memories = relationship("LongTermMemory", back_populates="user", lazy="dynamic")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    allergies = Column(JSONB, default=list)
    chronic_conditions = Column(JSONB, default=list)
    dietary_preferences = Column(JSONB, default=dict)
    emotional_baseline = Column(JSONB, default=dict)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user = relationship("User", back_populates="profile")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(128), default="新对话")
    started_at = Column(DateTime(timezone=True), default=_utcnow)
    last_active = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    turn_count = Column(Integer, default=0)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", order_by="Message.created_at", lazy="dynamic")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    intent = Column(String(32), nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")


class LongTermMemory(Base):
    """长期记忆 — 每 CONSOLIDATION_INTERVAL 轮触发一次 LLM 整理，压缩为摘要"""

    __tablename__ = "long_term_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary = Column(Text, nullable=False)
    key_topics = Column(JSONB, default=list)
    emotional_state = Column(String(256), nullable=True)
    turn_start = Column(Integer, default=0)
    turn_end = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="long_term_memories")
