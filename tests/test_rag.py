"""Phase 3 RAG 系统 — 单元测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRecipeRetriever:
    """菜谱向量检索测试"""

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_empty_collection(self):
        """ChromaDB 集合为空时返回空列表"""
        with patch("app.rag.recipe_retriever._get_collection") as mock_get:
            mock_col = MagicMock()
            mock_col.count.return_value = 0
            mock_get.return_value = mock_col

            from app.rag.recipe_retriever import search_recipes
            results = await search_recipes("低脂早餐")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_recipes(self):
        """正常检索返回菜谱列表"""
        mock_results = {
            "ids": [["recipe_001", "recipe_002"]],
            "documents": [["燕麦粥描述", "三明治描述"]],
            "metadatas": [[
                {"name": "燕麦牛奶粥", "tags": "早餐,减脂", "calories": 280, "ingredients": "燕麦片,牛奶"},
                {"name": "全麦三明治", "tags": "早餐,高蛋白", "calories": 320, "ingredients": "全麦面包,鸡蛋"},
            ]],
            "distances": [[0.1, 0.3]],
        }

        with patch("app.rag.recipe_retriever._get_collection") as mock_get:
            mock_col = MagicMock()
            mock_col.count.return_value = 10
            mock_col.query.return_value = mock_results
            mock_get.return_value = mock_col

            with patch("app.rag.recipe_retriever.embed_text", return_value=[0.1] * 512):
                from app.rag.recipe_retriever import search_recipes
                results = await search_recipes("低脂早餐", top_k=2)

        assert len(results) == 2
        assert results[0]["name"] == "燕麦牛奶粥"
        assert results[0]["score"] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_allergen_filter(self):
        """过敏原过滤测试"""
        mock_results = {
            "ids": [["recipe_001", "recipe_002"]],
            "documents": [["燕麦粥", "坚果沙拉"]],
            "metadatas": [[
                {"name": "燕麦粥", "tags": "早餐", "calories": 280, "ingredients": "燕麦,牛奶"},
                {"name": "坚果沙拉", "tags": "轻食", "calories": 300, "ingredients": "腰果,花生,生菜"},
            ]],
            "distances": [[0.1, 0.2]],
        }

        with patch("app.rag.recipe_retriever._get_collection") as mock_get:
            mock_col = MagicMock()
            mock_col.count.return_value = 10
            mock_col.query.return_value = mock_results
            mock_get.return_value = mock_col

            with patch("app.rag.recipe_retriever.embed_text", return_value=[0.1] * 512):
                from app.rag.recipe_retriever import search_recipes
                results = await search_recipes("沙拉", allergen_exclude=["花生"])

        # 含花生的坚果沙拉被过滤掉
        names = [r["name"] for r in results]
        assert "坚果沙拉" not in names

    @pytest.mark.asyncio
    async def test_index_recipe(self):
        """索引菜谱测试"""
        with patch("app.rag.recipe_retriever._get_collection") as mock_get:
            mock_col = MagicMock()
            mock_get.return_value = mock_col

            with patch("app.rag.recipe_retriever.embed_text", return_value=[0.1] * 512):
                from app.rag.recipe_retriever import index_recipe
                await index_recipe({
                    "id": "test_001",
                    "name": "测试菜谱",
                    "description": "测试描述",
                    "ingredients": ["食材A", "食材B"],
                    "tags": ["测试"],
                    "calories": 200,
                })

        mock_col.upsert.assert_called_once()
        call_kwargs = mock_col.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["test_001"]


class TestKGRetriever:
    """知识图谱检索测试"""

    @pytest.mark.asyncio
    async def test_query_returns_triples(self):
        """正常查询返回三元组"""
        mock_neo4j_rows = [
            {"entity": "苹果", "relation": "CONTAINS", "target": "维生素C",
             "rel_props": {"amount": "4.6", "unit": "mg"}, "target_labels": ["Nutrient"]},
            {"entity": "苹果", "relation": "BENEFITS", "target": "心血管疾病",
             "rel_props": {}, "target_labels": ["Condition"]},
        ]

        with patch("app.rag.kg_retriever.run_query", new_callable=AsyncMock) as mock_run:
            # 第一次调用返回 2 条，len < 3 会触发二跳；第二次返回空阻止重复
            mock_run.side_effect = [mock_neo4j_rows, []]

            from app.rag.kg_retriever import query_nutrition_knowledge
            results = await query_nutrition_knowledge("苹果")

        assert len(results) == 2
        assert results[0]["entity"] == "苹果"
        assert results[0]["relation"] == "含有"
        assert results[0]["target"] == "维生素C"
        assert results[0]["extra"]["amount"] == "4.6 mg"

    @pytest.mark.asyncio
    async def test_query_empty_result(self):
        """无匹配结果时返回空列表"""
        with patch("app.rag.kg_retriever.run_query", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = []

            from app.rag.kg_retriever import query_nutrition_knowledge
            results = await query_nutrition_knowledge("未知食材xyz")

        assert results == []

    @pytest.mark.asyncio
    async def test_relation_translation(self):
        """关系类型中文翻译"""
        from app.rag.kg_retriever import _translate_relation

        assert _translate_relation("CONTAINS") == "含有"
        assert _translate_relation("BENEFITS") == "有益于"
        assert _translate_relation("CONTRAINDICATED_FOR") == "禁忌于"
        assert _translate_relation("UNKNOWN") == "UNKNOWN"
