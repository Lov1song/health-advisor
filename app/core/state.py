"""LangGraph 工作流状态定义 — AgentState"""

from typing import Literal

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph StateGraph 状态

    所有节点共享此状态字典，通过读/写字段进行通信。
    """

    # ---- 用户上下文 ----
    user_id: str
    session_id: str

    # ---- 输入 ----
    messages: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]

    # ---- 意图识别 ----
    intent: Literal["mental", "nutrition", "general"]
    intent_confidence: float

    # ---- 记忆上下文 ----
    short_term_memory: list[dict]
    long_term_memory: list[dict]
    user_profile: dict

    # ---- Agent 执行 ----
    tool_calls: list[dict]
    rag_context: list[dict]
    kg_context: list[dict]

    # ---- 控制流 ----
    turn_count: int
    should_consolidate: bool

    # ---- 输出 ----
    response: str
