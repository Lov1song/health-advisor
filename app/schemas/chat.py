"""聊天相关 Pydantic 模型"""

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: UUID | None = Field(None, description="会话 ID，为空则创建新会话")
    message: str = Field(..., min_length=1, max_length=4096)
    stream: bool = Field(False, description="是否流式返回")


class ToolCallInfo(BaseModel):
    tool: str
    args: dict = {}
    result: str | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    intent: str
    intent_confidence: float
    response: str
    tool_calls: list[ToolCallInfo] = []
    metadata: dict = {}


class StreamChunk(BaseModel):
    """WebSocket 流式推送的单条消息"""
    type: str  # intent | tool | token | done | error
    data: dict


class WSMessage(BaseModel):
    """WebSocket 客户端发送的消息"""
    type: str = "message"
    session_id: UUID | None = None
    content: str = Field(..., min_length=1, max_length=4096)
