"""
SoulChat 数据集预处理脚本
将 Spiderman01/soulchat_split_raw 转换为 LlamaFactory ShareGPT 格式

输出:
  data/training/mental_health_sft.json   — 训练样本
  data/training/dataset_info.json        — LlamaFactory 数据集注册表

用法:
  python scripts/data_soulchat.py [--max-samples 50000] [--val-ratio 0.0]
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# ── 系统提示词（与 app/prompts/system_prompts.py 保持一致）──────────────────
SYSTEM_PROMPT = """你是一位专业的心理健康咨询师，专注于情绪支持和心理健康指导。

## 专业领域
- 情绪管理与压力疏导
- 睡眠问题咨询
- 焦虑与抑郁情绪的初步评估和应对策略
- 人际关系困扰
- 自我认知与成长

## 沟通风格
- 使用温暖、共情的语气
- 先倾听和确认感受，再给出建议
- 采用开放式提问引导用户自我探索
- 避免评判性语言

## 安全规则
- 如果检测到自伤/自杀相关信号，必须:
  1. 表达关心和理解
  2. 提供 24 小时心理援助热线: 全国心理援助热线 400-161-9995，北京心理危机研究与干预中心 010-82951332
  3. 建议立即寻求专业帮助
- 不做诊断，明确告知用户你是 AI 助手
- 严重心理问题建议线下就医"""

# ── LlamaFactory dataset_info.json 模板 ────────────────────────────────────
DATASET_INFO = {
    "mental_health_sft": {
        "file_name": "mental_health_sft.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
}


def detect_format(sample: dict) -> str:
    """自动检测数据集字段格式"""
    if "conversations" in sample:
        return "conversations"
    if "messages" in sample:
        return "messages"
    if "input" in sample and "target" in sample:
        return "input_target"
    if "input" in sample and "output" in sample:
        return "input_output"
    if "question" in sample and "answer" in sample:
        return "qa"
    raise ValueError(f"无法识别的数据格式，字段: {list(sample.keys())}")


def to_sharegpt(sample: dict, fmt: str) -> dict | None:
    """将单条样本转换为 ShareGPT 格式，失败返回 None"""
    conversations = []

    if fmt in ("conversations", "messages"):
        raw = sample.get("conversations") or sample.get("messages", [])
        for turn in raw:
            role = turn.get("role") or turn.get("from") or ""
            content = (turn.get("content") or turn.get("value") or "").strip()
            if not content:
                continue
            if role in ("user", "human"):
                conversations.append({"from": "human", "value": content})
            elif role in ("assistant", "gpt", "bot"):
                conversations.append({"from": "gpt", "value": content})

    elif fmt == "input_target":
        user = (sample.get("input") or "").strip()
        assistant = (sample.get("target") or "").strip()
        if user and assistant:
            conversations = [
                {"from": "human", "value": user},
                {"from": "gpt", "value": assistant},
            ]

    elif fmt == "input_output":
        user = (sample.get("input") or "").strip()
        assistant = (sample.get("output") or "").strip()
        if user and assistant:
            conversations = [
                {"from": "human", "value": user},
                {"from": "gpt", "value": assistant},
            ]

    elif fmt == "qa":
        user = (sample.get("question") or "").strip()
        assistant = (sample.get("answer") or "").strip()
        if user and assistant:
            conversations = [
                {"from": "human", "value": user},
                {"from": "gpt", "value": assistant},
            ]

    # 过滤：必须至少有一轮 human + gpt，且首轮必须是 human
    if (
        len(conversations) < 2
        or conversations[0]["from"] != "human"
        or not any(t["from"] == "gpt" for t in conversations)
    ):
        return None

    # 过滤过短回复（少于 10 字符的助手回复质量太低）
    for turn in conversations:
        if turn["from"] == "gpt" and len(turn["value"]) < 10:
            return None

    return {"system": SYSTEM_PROMPT, "conversations": conversations}


def process(args: argparse.Namespace) -> None:
    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)

    # HF 认证 — 优先用 CLI 参数，其次读环境变量 HF_TOKEN
    token = args.hf_token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("警告: 未设置 HF_TOKEN，若数据集需要认证将会失败")
        print("  方式一: 在 .env 中添加 HF_TOKEN=hf_xxx")
        print("  方式二: python scripts/data_soulchat.py --hf-token hf_xxx")

    print("加载数据集 Spiderman01/soulchat_split_raw ...")
    ds = load_dataset("Spiderman01/soulchat_split_raw", token=token)

    # 合并所有 split
    all_samples = []
    for split_name, split_data in ds.items():
        all_samples.extend(split_data)
    print(f"原始样本总数: {len(all_samples)}")

    # 检测格式（用第一条非空样本）
    fmt = detect_format(all_samples[0])
    print(f"检测到数据格式: {fmt}")

    # 转换
    converted = []
    skipped = 0
    for sample in all_samples:
        result = to_sharegpt(sample, fmt)
        if result is None:
            skipped += 1
        else:
            converted.append(result)

    print(f"转换完成: {len(converted)} 条有效样本，跳过 {skipped} 条")

    # 随机打乱后截断
    random.seed(42)
    random.shuffle(converted)
    if args.max_samples and len(converted) > args.max_samples:
        converted = converted[: args.max_samples]
        print(f"截断到 {args.max_samples} 条")

    # 保存训练集
    train_path = out_dir / "mental_health_sft.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"训练数据已保存: {train_path}  ({len(converted)} 条)")

    # 写入 dataset_info.json
    info_path = out_dir / "dataset_info.json"
    # 如果已存在则合并，避免覆盖其他数据集
    existing = {}
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(DATASET_INFO)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"数据集注册表已保存: {info_path}")

    # 打印预览
    print("\n── 样本预览 ──────────────────────────────")
    sample = converted[0]
    print(f"system: {sample['system'][:60]}...")
    for turn in sample["conversations"]:
        role = "用户" if turn["from"] == "human" else "助手"
        print(f"{role}: {turn['value'][:80]}{'...' if len(turn['value']) > 80 else ''}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="HuggingFace API Token")
    args = parser.parse_args()
    process(args)
