"""向量化模块 — 直接用 transformers 加载 BGE 模型，不依赖 sentence_transformers"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import torch
import torch.nn.functional as F

from app.utils.logger import get_logger

logger = get_logger("rag.embedder")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512 维中文小模型，首次运行自动下载


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoTokenizer, AutoModel
    logger.msg("loading_embedding_model", model=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _encode(texts: list[str]) -> list[list[float]]:
    tokenizer, model = _load_model()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        out = model(**inputs)
    # CLS token → L2 归一化
    emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
    return emb.tolist()


async def embed_text(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _encode, [text])
    return results[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode, texts)
