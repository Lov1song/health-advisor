"""向量化模块 — SentenceTransformer 单例"""

from __future__ import annotations

from functools import lru_cache

from app.utils.logger import get_logger

logger = get_logger("rag.embedder")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512 维中文小模型，首次运行自动下载


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    logger.msg("loading_embedding_model", model=MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    model = _load_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
