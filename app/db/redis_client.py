"""Redis 异步客户端 — 用于短期记忆 + 缓存 + 分布式锁"""

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()

redis_pool: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    global redis_pool
    redis_pool = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        max_connections=50,
    )
    # 验证连接
    await redis_pool.ping()
    return redis_pool


async def get_redis() -> aioredis.Redis:
    if redis_pool is None:
        return await init_redis()
    return redis_pool


async def close_redis() -> None:
    global redis_pool
    if redis_pool:
        await redis_pool.close()
        redis_pool = None
