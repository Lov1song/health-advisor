"""营养学知识图谱检索 — 基于 Neo4j

图谱 Schema:
  (:Food {name, aliases, calories_per_100g})
  (:Nutrient {name, unit, daily_recommended})
  (:Condition {name, description})
  (:Food)-[:CONTAINS {amount, unit}]->(:Nutrient)
  (:Food)-[:BENEFITS]->(:Condition)
  (:Food)-[:CONTRAINDICATED_FOR]->(:Condition)
  (:Nutrient)-[:TREATS]->(:Condition)
"""

from __future__ import annotations

from app.config import get_settings
from app.db.neo4j_client import run_query
from app.utils.logger import get_logger

logger = get_logger("rag.kg")


async def query_nutrition_knowledge(
    entity: str,
    max_hops: int | None = None,
) -> list[dict]:
    """查询与实体相关的营养学知识三元组

    Args:
        entity: 食材/营养素/症状名称
        max_hops: 最大跳数 (默认取配置 KG_MAX_HOPS)

    Returns:
        三元组列表 [{entity, relation, target, extra}]
    """
    cfg = get_settings()
    hops = max_hops or cfg.KG_MAX_HOPS

    # 一跳查询: entity → 直接关联节点
    one_hop_cypher = """
    MATCH (n)-[r]->(m)
    WHERE toLower(n.name) CONTAINS toLower($entity)
       OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS toLower($entity))
    RETURN n.name AS entity, type(r) AS relation, m.name AS target,
           properties(r) AS rel_props, labels(m) AS target_labels
    LIMIT 20
    """

    results = await run_query(one_hop_cypher, {"entity": entity})

    # 二跳查询: 若结果不足且 hops >= 2
    if len(results) < 3 and hops >= 2:
        two_hop_cypher = """
        MATCH (n)-[r1]->(mid)-[r2]->(m)
        WHERE toLower(n.name) CONTAINS toLower($entity)
        RETURN n.name AS entity,
               (type(r1) + '->' + mid.name + '->' + type(r2)) AS relation,
               m.name AS target,
               {} AS rel_props,
               labels(m) AS target_labels
        LIMIT 10
        """
        two_hop = await run_query(two_hop_cypher, {"entity": entity})
        results.extend(two_hop)

    if not results:
        await logger.ainfo("kg_no_results", entity=entity)
        return []

    triples = []
    for row in results:
        extra = {}
        rel_props = row.get("rel_props") or {}
        if rel_props.get("amount"):
            extra["amount"] = f"{rel_props['amount']} {rel_props.get('unit', '')}"
        triples.append({
            "entity": row.get("entity", ""),
            "relation": _translate_relation(row.get("relation", "")),
            "target": row.get("target", ""),
            "extra": extra,
        })

    await logger.ainfo("kg_query_done", entity=entity, results=len(triples))
    return triples


def _translate_relation(rel_type: str) -> str:
    _MAP = {
        "CONTAINS": "含有",
        "BENEFITS": "有益于",
        "CONTRAINDICATED_FOR": "禁忌于",
        "TREATS": "可缓解",
        "PAIRS_WITH": "适合搭配",
        "AVOID_WITH": "不宜同食",
    }
    return _MAP.get(rel_type, rel_type)


async def seed_nutrition_kg() -> None:
    """初始化示例营养知识图谱数据（仅首次运行）"""
    count_result = await run_query("MATCH (n) RETURN count(n) AS cnt")
    if count_result and count_result[0].get("cnt", 0) > 0:
        return

    await logger.ainfo("seeding_nutrition_kg")

    seed_cypher = """
    // 食物节点
    MERGE (apple:Food {name: '苹果'})
      SET apple.calories_per_100g = 52, apple.aliases = ['苹果']
    MERGE (spinach:Food {name: '菠菜'})
      SET spinach.calories_per_100g = 23
    MERGE (chicken:Food {name: '鸡胸肉'})
      SET chicken.calories_per_100g = 165
    MERGE (tofu:Food {name: '豆腐'})
      SET tofu.calories_per_100g = 76
    MERGE (salmon:Food {name: '三文鱼'})
      SET salmon.calories_per_100g = 208
    MERGE (oats:Food {name: '燕麦'})
      SET oats.calories_per_100g = 389
    MERGE (bitter_melon:Food {name: '苦瓜'})
      SET bitter_melon.calories_per_100g = 17

    // 营养素节点
    MERGE (vitC:Nutrient {name: '维生素C'}) SET vitC.unit = 'mg'
    MERGE (iron:Nutrient {name: '铁'}) SET iron.unit = 'mg'
    MERGE (protein:Nutrient {name: '蛋白质'}) SET protein.unit = 'g'
    MERGE (omega3:Nutrient {name: 'Omega-3'}) SET omega3.unit = 'g'
    MERGE (fiber:Nutrient {name: '膳食纤维'}) SET fiber.unit = 'g'
    MERGE (calcium:Nutrient {name: '钙'}) SET calcium.unit = 'mg'

    // 疾病/状态节点
    MERGE (diabetes:Condition {name: '糖尿病'})
    MERGE (hyper:Condition {name: '高血压'})
    MERGE (anemia:Condition {name: '贫血'})
    MERGE (obesity:Condition {name: '肥胖'})
    MERGE (heart:Condition {name: '心血管疾病'})

    // 食物-营养素关系
    MERGE (apple)-[:CONTAINS {amount: '4.6', unit: 'mg'}]->(vitC)
    MERGE (spinach)-[:CONTAINS {amount: '2.7', unit: 'mg'}]->(iron)
    MERGE (spinach)-[:CONTAINS {amount: '28', unit: 'mg'}]->(vitC)
    MERGE (chicken)-[:CONTAINS {amount: '31', unit: 'g'}]->(protein)
    MERGE (salmon)-[:CONTAINS {amount: '2.3', unit: 'g'}]->(omega3)
    MERGE (oats)-[:CONTAINS {amount: '10.6', unit: 'g'}]->(fiber)
    MERGE (tofu)-[:CONTAINS {amount: '17', unit: 'g'}]->(protein)
    MERGE (tofu)-[:CONTAINS {amount: '350', unit: 'mg'}]->(calcium)

    // 食物-疾病受益关系
    MERGE (bitter_melon)-[:BENEFITS]->(diabetes)
    MERGE (oats)-[:BENEFITS]->(diabetes)
    MERGE (spinach)-[:BENEFITS]->(anemia)
    MERGE (salmon)-[:BENEFITS]->(heart)
    MERGE (chicken)-[:BENEFITS]->(obesity)

    // 营养素-疾病关系
    MERGE (omega3)-[:TREATS]->(heart)
    MERGE (iron)-[:TREATS]->(anemia)
    MERGE (fiber)-[:TREATS]->(diabetes)
    MERGE (fiber)-[:TREATS]->(obesity)
    """
    await run_query(seed_cypher)
    await logger.ainfo("nutrition_kg_seeded")
