"""Neo4j 异步客户端 — 知识图谱查询"""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("db.neo4j")

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        cfg = get_settings()
        _driver = AsyncGraphDatabase.driver(
            cfg.NEO4J_URI,
            auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
        )
    return _driver


async def run_query(cypher: str, params: dict | None = None) -> list[dict]:
    """执行 Cypher 查询，返回记录列表"""
    driver = get_driver()
    try:
        async with driver.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()
    except Exception as e:
        await logger.aerror("neo4j_query_failed", error=str(e), cypher=cypher[:200])
        return []


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
