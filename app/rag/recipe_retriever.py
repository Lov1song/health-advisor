"""菜谱 RAG — ChromaDB 语义检索"""

from __future__ import annotations

import chromadb

from app.config import get_settings
from app.rag.embedder import embed_text
from app.utils.logger import get_logger

logger = get_logger("rag.recipe")

COLLECTION_NAME = "recipes"

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


async def search_recipes(
    query: str,
    top_k: int | None = None,
    allergen_exclude: list[str] | None = None,
) -> list[dict]:
    """按语义相似度检索菜谱，返回 top_k 条，带分数"""
    cfg = get_settings()
    k = top_k or cfg.RECIPE_TOP_K

    try:
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            await logger.awarn("recipe_collection_empty")
            return []

        query_embedding = embed_text(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 2, total),  # 多取一倍，后续过滤过敏原
            include=["documents", "metadatas", "distances"],
        )

        recipes: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""
            dist = results["distances"][0][i] if results["distances"] else 1.0
            score = round(1.0 - dist, 4)

            # 过敏原过滤
            if allergen_exclude:
                ingredients_raw = meta.get("ingredients", "")
                if any(a in ingredients_raw for a in allergen_exclude):
                    continue

            recipes.append({
                "id": doc_id,
                "name": meta.get("name", ""),
                "description": doc,
                "calories": meta.get("calories", 0),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "score": score,
            })
            if len(recipes) >= k:
                break

        await logger.ainfo("recipe_search_done", query=query[:50], results=len(recipes))
        return recipes

    except Exception as e:
        await logger.aerror("recipe_search_failed", error=str(e))
        return []


async def index_recipe(recipe: dict) -> None:
    """索引单条菜谱"""
    collection = _get_collection()
    doc_id = recipe.get("id", str(abs(hash(recipe["name"]))))
    ingredients = recipe.get("ingredients", [])
    text = (
        f"{recipe['name']}。"
        f"{recipe.get('description', '')}。"
        f"主要食材: {', '.join(ingredients)}。"
        f"标签: {', '.join(recipe.get('tags', []))}。"
    )
    embedding = embed_text(text)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "name": recipe.get("name", ""),
            "tags": ",".join(recipe.get("tags", [])),
            "calories": int(recipe.get("calories", 0)),
            "ingredients": ",".join(ingredients),
        }],
    )
