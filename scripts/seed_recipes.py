"""菜谱数据初始化脚本 — 向 ChromaDB 写入示例中餐菜谱

用法:
    python scripts/seed_recipes.py              # 写入 data/recipes.json（若存在）或内置示例
    python scripts/seed_recipes.py --force      # 强制清空重写
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.recipe_retriever import index_recipe
from app.utils.logger import get_logger

logger = get_logger("seed")

# 优先加载 LLM 生成的完整语料，回退到内置示例
_DATA_FILE = ROOT / "data" / "recipes.json"

RECIPES = [
    # ── 早餐 ──────────────────────────────────────────────
    {
        "id": "recipe_001",
        "name": "燕麦牛奶粥",
        "description": "燕麦与低脂牛奶熬制，富含β-葡聚糖，有助降低胆固醇，稳定血糖，适合减脂期和糖尿病患者早餐。",
        "ingredients": ["燕麦片", "低脂牛奶", "蜂蜜", "蓝莓"],
        "tags": ["早餐", "减脂", "糖尿病友好", "高纤维"],
        "calories": 280,
    },
    {
        "id": "recipe_002",
        "name": "全麦鸡蛋三明治",
        "description": "全麦面包配水煮蛋和生菜，蛋白质与复合碳水搭配，提供持久饱腹感，适合增肌或控体重人群。",
        "ingredients": ["全麦面包", "鸡蛋", "生菜", "番茄", "低脂奶酪"],
        "tags": ["早餐", "高蛋白", "增肌"],
        "calories": 320,
    },
    {
        "id": "recipe_003",
        "name": "蒸鸡蛋羹",
        "description": "嫩滑鸡蛋羹，消化吸收率高，老幼皆宜。加少量香油和生抽提鲜，低热量高蛋白。",
        "ingredients": ["鸡蛋", "温水", "香油", "生抽"],
        "tags": ["早餐", "易消化", "老幼皆宜", "低热量"],
        "calories": 120,
    },
    # ── 午餐/正餐 ─────────────────────────────────────────
    {
        "id": "recipe_004",
        "name": "清蒸鲈鱼",
        "description": "低脂高蛋白，富含DHA和Omega-3脂肪酸，有益心血管健康。葱姜去腥，蒸制保留营养。",
        "ingredients": ["鲈鱼", "生姜", "葱", "生抽", "料酒"],
        "tags": ["午餐", "高蛋白", "低脂", "心血管健康", "Omega-3"],
        "calories": 180,
    },
    {
        "id": "recipe_005",
        "name": "番茄鸡蛋面",
        "description": "经典家常面，番茄富含番茄红素，抗氧化。鸡蛋补充蛋白质，全面营养，制作简单。",
        "ingredients": ["面条", "番茄", "鸡蛋", "盐", "油", "葱"],
        "tags": ["午餐", "快手", "均衡营养"],
        "calories": 380,
    },
    {
        "id": "recipe_006",
        "name": "凉拌黄瓜",
        "description": "黄瓜爽脆，热量极低，每100g仅15千卡。加蒜末、醋和香油凉拌，开胃去腻，减脂首选。",
        "ingredients": ["黄瓜", "大蒜", "醋", "香油", "盐", "辣椒油"],
        "tags": ["凉菜", "减脂", "低热量", "开胃"],
        "calories": 50,
    },
    {
        "id": "recipe_007",
        "name": "西兰花炒鸡胸肉",
        "description": "高蛋白低脂经典增肌餐。鸡胸肉蛋白质含量高达31%，西兰花含维生素C和叶酸，搭配均衡。",
        "ingredients": ["鸡胸肉", "西兰花", "大蒜", "盐", "黑胡椒", "橄榄油"],
        "tags": ["午餐", "晚餐", "增肌", "高蛋白", "低脂", "健身餐"],
        "calories": 260,
    },
    {
        "id": "recipe_008",
        "name": "苦瓜炒鸡蛋",
        "description": "苦瓜含苦瓜素，有助调节血糖，适合糖尿病患者。鸡蛋补充蛋白质，去苦味可先焯水。",
        "ingredients": ["苦瓜", "鸡蛋", "盐", "油"],
        "tags": ["午餐", "糖尿病友好", "降糖", "苦味"],
        "calories": 150,
    },
    {
        "id": "recipe_009",
        "name": "豆腐海带汤",
        "description": "低热量高钙汤品，豆腐富含植物蛋白和钙，海带含碘和多糖，有助甲状腺健康，素食可选。",
        "ingredients": ["豆腐", "海带", "盐", "香油", "葱"],
        "tags": ["汤类", "低热量", "高钙", "素食", "补碘"],
        "calories": 80,
    },
    {
        "id": "recipe_010",
        "name": "菠菜猪肝汤",
        "description": "补铁补血经典食谱，猪肝含血红素铁，吸收率高；菠菜含非血红素铁和维生素C促进铁吸收。",
        "ingredients": ["猪肝", "菠菜", "姜丝", "盐", "料酒"],
        "tags": ["汤类", "补铁", "补血", "贫血调理"],
        "calories": 130,
    },
    # ── 晚餐/轻食 ─────────────────────────────────────────
    {
        "id": "recipe_011",
        "name": "蒸红薯",
        "description": "天然甜味，富含膳食纤维和β-胡萝卜素，GI值低于白米饭，适合减脂代餐和糖尿病患者。",
        "ingredients": ["红薯"],
        "tags": ["晚餐", "减脂", "代餐", "糖尿病友好", "高纤维"],
        "calories": 86,
    },
    {
        "id": "recipe_012",
        "name": "三文鱼牛油果沙拉",
        "description": "富含Omega-3和健康脂肪，促进心血管健康。牛油果提供单不饱和脂肪，饱腹感强。",
        "ingredients": ["三文鱼", "牛油果", "生菜", "番茄", "柠檬汁", "橄榄油"],
        "tags": ["晚餐", "轻食", "Omega-3", "心血管健康", "高营养"],
        "calories": 420,
    },
    {
        "id": "recipe_013",
        "name": "小米南瓜粥",
        "description": "养胃暖胃经典粥品，南瓜富含β-胡萝卜素，小米含B族维生素，适合胃不好人群和老年人。",
        "ingredients": ["小米", "南瓜", "清水"],
        "tags": ["早餐", "晚餐", "养胃", "老年人", "易消化"],
        "calories": 120,
    },
    {
        "id": "recipe_014",
        "name": "水煮牛肉",
        "description": "高蛋白低脂红肉，配大量蔬菜。牛肉含血红素铁和锌，适合缺铁性贫血和增肌人群。",
        "ingredients": ["牛肉", "豆芽", "西芹", "辣椒", "花椒", "豆瓣酱"],
        "tags": ["午餐", "晚餐", "高蛋白", "增肌", "补铁"],
        "calories": 350,
    },
    {
        "id": "recipe_015",
        "name": "杂粮饭",
        "description": "混合糙米、黑米、红豆、燕麦，GI值低，膳食纤维丰富，延缓血糖上升，适合血糖管理。",
        "ingredients": ["糙米", "黑米", "红豆", "燕麦"],
        "tags": ["主食", "减脂", "糖尿病友好", "高纤维", "全谷物"],
        "calories": 180,
    },
]


def _load_recipes() -> list[dict]:
    """优先从 data/recipes.json 加载，否则用内置示例"""
    if _DATA_FILE.exists():
        with open(_DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"从 {_DATA_FILE.name} 加载 {len(data)} 条食谱")
        return data
    return RECIPES


async def main() -> None:
    recipes = _load_recipes()
    print(f"开始写入 {len(recipes)} 条菜谱到 ChromaDB...")
    for i, recipe in enumerate(recipes, 1):
        await index_recipe(recipe)
        print(f"  [{i}/{len(recipes)}] {recipe['name']}")
    print("[OK] 菜谱数据写入完成")


if __name__ == "__main__":
    asyncio.run(main())
