"""FastAPI 应用入口 — lifespan 管理 + 路由挂载"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.middleware import RequestLoggingMiddleware, register_exception_handlers
from app.api.v1.router import router as v1_router
from app.db.postgres import init_db, close_db
from app.db.redis_client import init_redis, close_redis
from app.db.neo4j_client import close_driver as close_neo4j
from app.utils.logger import setup_logging, get_logger

settings = get_settings()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期 — 启动时初始化资源，关闭时释放"""
    # ---- 启动 ----
    setup_logging()
    await logger.ainfo("starting", app=settings.APP_NAME)

    # 初始化数据库
    await init_db()
    await logger.ainfo("postgres_ready")

    # 初始化 Redis
    try:
        await init_redis()
        await logger.ainfo("redis_ready")
    except Exception as e:
        await logger.awarn("redis_unavailable", error=str(e))

    # 初始化 RAG (ChromaDB 菜谱 + Neo4j 知识图谱)
    try:
        from scripts.seed_recipes import _load_recipes
        from app.rag.recipe_retriever import seed_if_empty
        await seed_if_empty(_load_recipes())
        await logger.ainfo("chromadb_ready")
    except Exception as e:
        await logger.awarning("chromadb_seed_skipped", error=str(e))

    try:
        from app.rag.kg_retriever import seed_nutrition_kg
        await seed_nutrition_kg()
        await logger.ainfo("neo4j_ready")
    except Exception as e:
        await logger.awarning("neo4j_seed_skipped", error=str(e))

    await logger.ainfo("started", app=settings.APP_NAME)

    yield

    # ---- 关闭 ----
    await close_redis()
    await close_neo4j()
    await close_db()
    await logger.ainfo("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="多智能体健康咨询系统 API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # ---- 中间件 ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # ---- 异常处理 ----
    register_exception_handlers(app)

    # ---- 路由 ----
    app.include_router(v1_router)

    # ---- 健康检查 ----
    @app.get("/healthz", tags=["infra"])
    async def healthz():
        return {"status": "ok"}

    return app


app = create_app()
