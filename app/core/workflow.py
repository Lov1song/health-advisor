"""LangGraph 主工作流 — 多智能体编排"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.general_agent import GeneralAgent
from app.agents.mental_agent import MentalAgent
from app.agents.nutrition_agent import NutritionAgent
from app.config import get_settings
from app.core.intent_router import classify_intent_node
from app.core.state import AgentState
from app.utils.logger import get_logger

logger = get_logger("workflow")

# ---- Agent 单例 ----
_mental_agent = MentalAgent()
_nutrition_agent = NutritionAgent()
_general_agent = GeneralAgent()


# ============================================================
# LangGraph 节点函数
# ============================================================

async def load_memory_node(state: AgentState) -> AgentState:
    """节点: 加载记忆上下文

    Phase 5 完整实现，当前初始化空记忆。
    """
    if "short_term_memory" not in state:
        state["short_term_memory"] = []
    if "long_term_memory" not in state:
        state["long_term_memory"] = []
    if "user_profile" not in state:
        state["user_profile"] = {}
    if "tool_calls" not in state:
        state["tool_calls"] = []
    if "rag_context" not in state:
        state["rag_context"] = []
    if "kg_context" not in state:
        state["kg_context"] = []
    if "turn_count" not in state:
        state["turn_count"] = 0

    await logger.ainfo("memory_loaded", user_id=state.get("user_id"))
    return state


async def mental_agent_node(state: AgentState) -> AgentState:
    """节点: 心理咨询 Agent"""
    await logger.ainfo("executing_agent", agent="mental")
    return await _mental_agent.run(state)


async def nutrition_agent_node(state: AgentState) -> AgentState:
    """节点: 营养师 Agent"""
    await logger.ainfo("executing_agent", agent="nutrition")
    return await _nutrition_agent.run(state)


async def general_agent_node(state: AgentState) -> AgentState:
    """节点: 通用健康 Agent"""
    await logger.ainfo("executing_agent", agent="general")
    return await _general_agent.run(state)


async def save_memory_node(state: AgentState) -> AgentState:
    """节点: 保存记忆

    Phase 5 完整实现，当前仅更新轮次计数。
    """
    state["turn_count"] = state.get("turn_count", 0) + 1
    interval = get_settings().CONSOLIDATION_INTERVAL
    state["should_consolidate"] = (state["turn_count"] % interval == 0)

    await logger.ainfo(
        "memory_saved",
        turn_count=state["turn_count"],
        should_consolidate=state["should_consolidate"],
    )
    return state


async def consolidate_memory_node(state: AgentState) -> AgentState:
    """节点: 记忆整理

    Phase 5 完整实现，当前仅记录日志。
    """
    await logger.ainfo("memory_consolidation_triggered", turn_count=state.get("turn_count"))
    # TODO: Phase 5 — 调用 consolidation agent 异步整理
    state["should_consolidate"] = False
    return state


# ============================================================
# 路由函数
# ============================================================

def route_by_intent(state: AgentState) -> str:
    """条件路由 — 根据意图分类结果分发到对应 Agent"""
    intent = state.get("intent", "general")
    return intent  # "mental" | "nutrition" | "general"


def should_consolidate(state: AgentState) -> bool:
    """条件路由 — 是否触发记忆整理"""
    return state.get("should_consolidate", False)


# ============================================================
# 工作流构建
# ============================================================

def build_workflow() -> StateGraph:
    """构建 LangGraph StateGraph 工作流

    流程:
    load_memory → classify_intent → [mental|nutrition|general] → save_memory → [consolidate?] → END
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("mental_agent", mental_agent_node)
    graph.add_node("nutrition_agent", nutrition_agent_node)
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("save_memory", save_memory_node)
    graph.add_node("consolidate_memory", consolidate_memory_node)

    # 入口
    graph.set_entry_point("load_memory")

    # 边: load_memory → classify_intent
    graph.add_edge("load_memory", "classify_intent")

    # 条件边: classify_intent → Agent (三选一)
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "mental": "mental_agent",
            "nutrition": "nutrition_agent",
            "general": "general_agent",
        },
    )

    # Agent → save_memory
    for agent_name in ["mental_agent", "nutrition_agent", "general_agent"]:
        graph.add_edge(agent_name, "save_memory")

    # 条件边: save_memory → consolidate 或 END
    graph.add_conditional_edges(
        "save_memory",
        should_consolidate,
        {
            True: "consolidate_memory",
            False: END,
        },
    )

    # consolidate → END
    graph.add_edge("consolidate_memory", END)

    return graph


# ---- 编译后的工作流单例 ----

_compiled_workflow = None


def get_workflow():
    """获取编译后的工作流 (懒加载单例)"""
    global _compiled_workflow
    if _compiled_workflow is None:
        graph = build_workflow()
        _compiled_workflow = graph.compile()
        logger.info("workflow_compiled")  # sync log at startup
    return _compiled_workflow


# ---- 便捷调用接口 ----

async def run_workflow(
    user_id: str,
    session_id: str,
    user_message: str,
    short_term_memory: list[dict] | None = None,
    long_term_memory: list[dict] | None = None,
    user_profile: dict | None = None,
    turn_count: int = 0,
) -> dict[str, Any]:
    """执行完整工作流，返回结果字典

    这是 Chat API 调用的主入口。
    """
    workflow = get_workflow()

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "messages": [{"role": "user", "content": user_message}],
        "short_term_memory": short_term_memory or [],
        "long_term_memory": long_term_memory or [],
        "user_profile": user_profile or {},
        "tool_calls": [],
        "rag_context": [],
        "kg_context": [],
        "turn_count": turn_count,
        "should_consolidate": False,
        "intent": "general",
        "intent_confidence": 0.0,
        "response": "",
    }

    # 执行工作流
    final_state = await workflow.ainvoke(initial_state)

    return {
        "intent": final_state.get("intent", "general"),
        "intent_confidence": final_state.get("intent_confidence", 0.0),
        "response": final_state.get("response", ""),
        "tool_calls": final_state.get("tool_calls", []),
        "turn_count": final_state.get("turn_count", 0),
    }
