"""应用配置 — 基于 pydantic-settings，自动从 .env 加载"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- 应用 ----
    APP_NAME: str = "health-advisor"
    DEBUG: bool = False

    # ---- 数据库 ----
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/health_advisor"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---- Neo4j ----
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # ---- 向量数据库 ----
    VECTOR_DB_TYPE: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # ---- LLM ----
    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_MODEL: str = "deepseek-8b-qlora"          # vLLM --served-model-name 必须与此一致
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    GENERAL_MODEL: str = "deepseek-chat"

    # ---- 记忆系统 ----
    SHORT_TERM_WINDOW: int = 10
    CONSOLIDATION_INTERVAL: int = 10
    LONG_TERM_TOP_K: int = 10
    MEMORY_SEMANTIC_TOP_K: int = 8
    MEMORY_DEDUP_THRESHOLD: float = 0.85

    # ---- RAG ----
    RECIPE_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    KG_MAX_HOPS: int = 2

    # ---- 认证 ----
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440


@lru_cache
def get_settings() -> Settings:
    return Settings()
