"""LLM 批量生成健康食谱 → data/recipes.json

用法:
    python scripts/generate_recipes.py              # 全量生成 (~200 条)
    python scripts/generate_recipes.py --per-theme 10   # 每主题 10 条
    python scripts/generate_recipes.py --dry-run    # 只打印 prompt，不调用 API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path，使脚本可以 import app.*
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from openai import AsyncOpenAI

# ── 配置 ─────────────────────────────────────────────────────

OUTPUT_FILE = ROOT / "data" / "recipes.json"

# ID 起始偏移（前 15 条是手写示例 recipe_001~015）
ID_START = 16

# 可用标签（控制词汇，避免 LLM 乱造）
ALLOWED_TAGS = (
    "早餐 午餐 晚餐 汤类 凉菜 主食 加餐 "
    "减脂 增肌 控血糖 降血压 补铁 补钙 Omega-3 "
    "高蛋白 低热量 低脂 高纤维 "
    "糖尿病友好 心血管健康 贫血调理 老年人 素食 孕期 易消化 养胃 抗炎"
)

# 主题列表: (主题名, 生成侧重点说明)
THEMES = [
    ("减脂控体重",    "低热量(<350kcal)、高饱腹感、适合体脂管理，常见食材: 鸡胸肉、蔬菜、蒟蒻、魔芋"),
    ("糖尿病控血糖",  "低GI、富含膳食纤维，常见食材: 苦瓜、燕麦、杂粮、豆类、绿叶菜"),
    ("高蛋白增肌",    "每份蛋白质>20g，常见食材: 鸡胸肉、牛肉、三文鱼、豆腐、鸡蛋、虾"),
    ("心血管健康",    "低钠低饱和脂肪、富含Omega-3和抗氧化物，常见食材: 深海鱼、燕麦、坚果、橄榄油"),
    ("补铁补血贫血",  "富含血红素铁或促铁吸收食材，常见: 猪肝、牛肉、菠菜、黑木耳、红豆"),
    ("素食植物蛋白",  "纯植物性食材，蛋白来源: 豆腐、豆浆、鹰嘴豆、坚果、藜麦、全谷物"),
    ("老年人易消化",  "软烂好咀嚼、低盐低油、养胃，常见: 粥品、蒸蛋、豆腐、细面条、南瓜"),
    ("孕期哺乳营养",  "高叶酸、高钙、补铁，常见: 菠菜、奶制品、豆腐、三文鱼、猪肝、坚果"),
    ("抗炎提免疫",    "富含姜黄素、Omega-3、维生素C/E，常见: 姜黄、深色蔬果、蓝莓、绿茶、洋葱"),
    ("快手早餐主食",  "15分钟内完成，饱腹均衡，常见: 全麦面包、鸡蛋、燕麦、豆浆、馒头"),
]

# ── Prompt 模板 ───────────────────────────────────────────────

PROMPT_TEMPLATE = """请生成 {count} 道关于「{theme}」的中式健康食谱。

侧重点：{focus}

每道食谱的格式如下：
- name: 菜名（中文，简洁，4-12 字）
- description: 营养说明（2-3 句话，说明食材搭配逻辑和具体健康功效，数据尽量准确）
- ingredients: 主要食材列表（3-8 个中文食材名，不含调料）
- tags: 从以下标签中选 2-5 个: {tags}
- calories: 每人份千卡（整数，合理估算，误差不超过 20%）

约束：
1. 菜名不能与以下已有菜名重复：{existing}
2. 食谱要真实可操作（非虚构食材）
3. 热量估算合理（低热量 <300 kcal，中等 300-500 kcal，高热量 >500 kcal）

以 JSON 对象返回，格式：
{{"recipes": [
  {{"name": "...", "description": "...", "ingredients": [...], "tags": [...], "calories": 0}},
  ...
]}}
"""


# ── 生成函数 ─────────────────────────────────────────────────

async def generate_batch(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    focus: str,
    count: int,
    seen_names: set[str],
    dry_run: bool = False,
) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(
        count=count,
        theme=theme,
        focus=focus,
        tags=ALLOWED_TAGS,
        existing="、".join(list(seen_names)[:30]) or "无",
    )

    if dry_run:
        print(f"\n{'='*60}\n[DRY-RUN] 主题: {theme}\n{prompt[:400]}...\n")
        return []

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        recipes = data.get("recipes", [])

        # 过滤重名
        unique = []
        for r in recipes:
            name = r.get("name", "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique.append(r)

        print(f"  [OK] {theme}: 生成 {len(unique)} 条")
        return unique

    except Exception as e:
        print(f"  ✗ {theme}: 失败 — {e}")
        return []


async def run(per_theme: int = 20, dry_run: bool = False) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("GENERAL_MODEL", "deepseek-chat")

    if not api_key and not dry_run:
        print("错误: 未设置 DEEPSEEK_API_KEY，请在 .env 中配置或导出环境变量")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key or "dummy", base_url=base_url)

    # 加载已有菜谱（避免重名）
    seen_names: set[str] = set()
    existing_recipes: list[dict] = []

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            existing_recipes = json.load(f)
        seen_names = {r["name"] for r in existing_recipes}
        print(f"已有食谱: {len(existing_recipes)} 条，将在此基础上追加")
    else:
        # 从 seed_recipes.py 里读取已有手写示例
        try:
            from scripts.seed_recipes import RECIPES as _seed
            seen_names = {r["name"] for r in _seed}
            existing_recipes = list(_seed)
            print(f"从 seed_recipes.py 加载 {len(existing_recipes)} 条示例")
        except Exception:
            pass

    print(f"\n开始生成，每主题 {per_theme} 条，共 {len(THEMES)} 个主题...\n")

    # 批量生成（顺序执行，避免触发限速）
    new_recipes: list[dict] = []
    for theme, focus in THEMES:
        batch = await generate_batch(
            client, model, theme, focus, per_theme, seen_names, dry_run
        )
        new_recipes.extend(batch)
        if not dry_run:
            await asyncio.sleep(0.5)  # 限速缓冲

    if dry_run:
        print(f"\n[DRY-RUN] 结束，实际不写入文件")
        return

    # 分配 ID
    all_recipes = existing_recipes.copy()
    next_id = ID_START + len([r for r in all_recipes if r.get("id", "").startswith("recipe_")])

    for r in new_recipes:
        r["id"] = f"recipe_{next_id:03d}"
        next_id += 1
        all_recipes.append(r)

    # 写入
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_recipes, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 写入完成: {OUTPUT_FILE}")
    print(f"  合计: {len(all_recipes)} 条（新增 {len(new_recipes)} 条）")


# ── 入口 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 批量生成健康食谱")
    parser.add_argument("--per-theme", type=int, default=20, help="每个主题生成条数 (默认 20)")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用 API")
    args = parser.parse_args()

    asyncio.run(run(per_theme=args.per_theme, dry_run=args.dry_run))
