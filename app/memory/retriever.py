"""记忆向量存储 — ChromaDB 语义检索与去重"""

from __future__ import annotations

import chromadb

from app.config import get_settings
from app.rag.embedder import embed_text
from app.utils.logger import get_logger

logger = get_logger("memory.retriever")

COLLECTION_NAME = "long_term_memories"

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _client is None:
        cfg = get_settings()
        _client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


async def upsert_memory(
    memory_id: str,
    text: str,
    user_id: str,
    memory_type: str,
    importance: float,
    is_pinned: bool,
) -> None:
    """写入或更新记忆向量"""
    try:
        collection = _get_collection()
        embedding = await embed_text(text)
        collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "user_id": user_id,
                "memory_type": memory_type,
                "importance": importance,
                "is_pinned": 1 if is_pinned else 0,
            }],
        )
    except Exception as e:
        await logger.aerror("memory_upsert_failed", memory_id=memory_id, error=str(e))


async def search_similar(
    query_text: str,
    user_id: str,
    top_k: int = 8,
) -> list[dict]:
    """语义检索最相关记忆，返回 [{id, score}]"""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []

        query_embedding = await embed_text(query_text)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": {"$eq": user_id}},
            include=["distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {"id": doc_id, "score": round(1.0 - dist, 4)}
            for doc_id, dist in zip(results["ids"][0], results["distances"][0])
        ]
    except Exception as e:
        await logger.aerror("memory_search_failed", error=str(e))
        return []


async def find_duplicate(
    text: str,
    user_id: str,
    threshold: float = 0.85,
) -> str | None:
    """若存在语义相似度超过阈值的记忆则返回其 ID，否则返回 None"""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return None

        query_embedding = await embed_text(text)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"user_id": {"$eq": user_id}},
            include=["distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        score = 1.0 - results["distances"][0][0]
        return results["ids"][0][0] if score >= threshold else None
    except Exception as e:
        await logger.aerror("memory_dedup_check_failed", error=str(e))
        return None


async def delete_memory(memory_id: str) -> None:
    """从向量集合中删除记忆"""
    try:
        _get_collection().delete(ids=[memory_id])
    except Exception as e:
        await logger.aerror("memory_delete_failed", memory_id=memory_id, error=str(e))
