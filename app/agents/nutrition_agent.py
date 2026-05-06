"""营养师 Agent — Agentic-RAG + Plan-Execute 模式"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from app.agents.base import BaseAgent
from app.core.state import AgentState
from app.prompts.nutrition_prompts import build_nutrition_messages
from app.rag.kg_retriever import query_nutrition_knowledge
from app.rag.recipe_retriever import search_recipes
from app.tools.health_calc import calculate_bmi, calculate_bmr, calculate_tdee
from app.utils.logger import get_logger

logger = get_logger("agent.nutrition")


# ---- Plan-Execute 任务类型 ----

PLAN_PROMPT = """分析用户的营养/饮食相关问题，拆解为需要执行的子任务。

可用工具:
1. **calculate_health** — 计算 BMI/BMR/TDEE (需要: 身高/体重/年龄/性别/活动水平)
2. **search_recipes** — 检索菜谱 (需要: 关键词/条件)
3. **query_knowledge** — 查询营养学知识图谱 (需要: 营养素/食材/症状)
4. **direct_answer** — 无需工具，直接基于知识回答

用户画像:
{profile}

用户消息:
{message}

请以 JSON 格式返回任务计划:
{{
  "tasks": [
    {{"type": "calculate_health|search_recipes|query_knowledge|direct_answer", "params": {{}}, "reason": "说明"}}
  ]
}}
"""


class NutritionAgent(BaseAgent):
    """营养师 Agent

    核心流程 (Plan-Execute):
    1. Planner: LLM 分析用户意图 → 拆解为子任务列表
    2. Executor: 并行执行子任务 (健康计算 / 菜谱检索 / 知识图谱)
    3. Synthesizer: 汇总所有结果 → 生成个性化回复
    """

    def __init__(self):
        super().__init__("nutrition")

    async def _plan(self, state: AgentState) -> list[dict]:
        """Step 1: 任务规划"""
        user_message = self._get_user_message(state)
        profile = self._get_user_profile(state)

        prompt = PLAN_PROMPT.format(
            profile=str(profile) if profile else "暂无",
            message=user_message,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            tasks = result.get("tasks", [])
            await self.logger.ainfo("plan_created", task_count=len(tasks))
            return tasks
        except Exception as e:
            await self.logger.awarn("plan_failed", error=str(e))
            return [{"type": "direct_answer", "params": {}, "reason": "规划失败，直接回答"}]

    async def _execute_task(self, task: dict, state: AgentState) -> dict:
        """执行单个子任务"""
        task_type = task.get("type", "direct_answer")
        params = task.get("params", {})
        profile = self._get_user_profile(state)

        if task_type == "calculate_health":
            return await self._execute_health_calc(params, profile)
        elif task_type == "search_recipes":
            return await self._execute_recipe_search(params)
        elif task_type == "query_knowledge":
            return await self._execute_kg_query(params)
        else:
            return {"type": "direct_answer", "result": None}

    async def _execute_health_calc(self, params: dict, profile: dict) -> dict:
        """执行健康指标计算 — 自动从用户档案补全缺失参数"""
        # 参数补全: 优先使用传入参数，缺失时从用户档案获取
        weight = params.get("weight_kg") or profile.get("weight_kg")
        height = params.get("height_cm") or profile.get("height_cm")
        age = params.get("age") or profile.get("age")
        gender = params.get("gender") or profile.get("gender")
        activity = params.get("activity_level", "moderate")

        results = {}

        if weight and height:
            bmi = calculate_bmi(float(weight), float(height))
            results["bmi"] = bmi.model_dump()

        if weight and height and age and gender:
            bmr = calculate_bmr(float(weight), float(height), int(age), gender)
            results["bmr"] = bmr.model_dump()

            tdee = calculate_tdee(float(weight), float(height), int(age), gender, activity)
            results["tdee"] = tdee.model_dump()

        return {"type": "calculate_health", "result": results}

    async def _execute_recipe_search(self, params: dict) -> dict:
        """执行菜谱检索 — ChromaDB 语义向量检索"""
        query = params.get("query", params.get("keywords", "健康菜谱"))
        allergen_exclude = params.get("allergen_exclude")
        await self.logger.ainfo("recipe_search", query=query)
        recipes = await search_recipes(query=query, allergen_exclude=allergen_exclude)
        return {"type": "search_recipes", "result": recipes}

    async def _execute_kg_query(self, params: dict) -> dict:
        """执行知识图谱查询 — Neo4j Cypher"""
        entity = params.get("entity", params.get("query", ""))
        await self.logger.ainfo("kg_query", entity=entity)
        triples = await query_nutrition_knowledge(entity=entity)
        return {"type": "query_knowledge", "result": triples}

    async def _execute_all(self, tasks: list[dict], state: AgentState) -> list[dict]:
        """Step 2: 并行执行所有子任务"""
        if not tasks:
            return []

        # 过滤 direct_answer 类型，不需要执行
        executable = [t for t in tasks if t.get("type") != "direct_answer"]
        if not executable:
            return []

        results = await asyncio.gather(
            *[self._execute_task(t, state) for t in executable],
            return_exceptions=True,
        )

        # 过滤异常
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                await self.logger.awarn("task_execution_failed", error=str(r))
            else:
                valid_results.append(r)

        return valid_results

    async def run(self, state: AgentState) -> AgentState:
        """同步执行: Plan → Execute → Synthesize"""
        user_message = self._get_user_message(state)

        # Step 1: Plan
        tasks = await self._plan(state)

        # Step 2: Execute (并行)
        task_results = await self._execute_all(tasks, state)

        # 分类结果
        rag_context = []
        kg_context = []
        tool_results = []
        for r in task_results:
            if r["type"] == "search_recipes":
                rag_context.extend(r.get("result", []))
            elif r["type"] == "query_knowledge":
                kg_context.extend(r.get("result", []))
            elif r["type"] == "calculate_health":
                tool_results.append({"tool": "health_calc", "result": r.get("result", {})})

        # 更新 state
        state["rag_context"] = rag_context
        state["kg_context"] = kg_context
        state["tool_calls"] = tool_results

        # Step 3: Synthesize — 汇总结果生成回复
        messages = build_nutrition_messages(
            user_message=user_message,
            short_term_memory=self._get_short_term_memory(state),
            user_profile=self._get_user_profile(state),
            rag_context=rag_context if rag_context else None,
            kg_context=kg_context if kg_context else None,
            tool_results=tool_results if tool_results else None,
        )

        resp = await self.llm.complete(
            messages=messages,
            backend="general",
            temperature=0.6,
            max_tokens=2048,
        )

        state["response"] = resp.content
        return state

    async def stream(self, state: AgentState) -> AsyncGenerator[str, None]:
        """流式执行: Plan → Execute → Stream Synthesize"""
        user_message = self._get_user_message(state)

        tasks = await self._plan(state)
        task_results = await self._execute_all(tasks, state)

        rag_context = []
        kg_context = []
        tool_results = []
        for r in task_results:
            if r["type"] == "search_recipes":
                rag_context.extend(r.get("result", []))
            elif r["type"] == "query_knowledge":
                kg_context.extend(r.get("result", []))
            elif r["type"] == "calculate_health":
                tool_results.append({"tool": "health_calc", "result": r.get("result", {})})

        state["rag_context"] = rag_context
        state["kg_context"] = kg_context
        state["tool_calls"] = tool_results

        messages = build_nutrition_messages(
            user_message=user_message,
            short_term_memory=self._get_short_term_memory(state),
            user_profile=self._get_user_profile(state),
            rag_context=rag_context if rag_context else None,
            kg_context=kg_context if kg_context else None,
            tool_results=tool_results if tool_results else None,
        )

        full_response = []
        async for token in self.llm.stream(
            messages=messages, backend="general", temperature=0.6, max_tokens=2048
        ):
            full_response.append(token)
            yield token

        state["response"] = "".join(full_response)
