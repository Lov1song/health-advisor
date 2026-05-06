"""v1 路由聚合"""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.users import router as users_router

router = APIRouter(prefix="/api/v1")
router.include_router(users_router)
router.include_router(health_router)
router.include_router(chat_router)
