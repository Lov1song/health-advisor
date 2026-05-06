"""中间件 — 请求日志、全局异常处理"""

import time
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.exceptions import HealthAdvisorError

logger = structlog.get_logger("middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志 — 记录方法、路径、耗时"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid4())[:8]
        start = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        await logger.ainfo(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器"""

    @app.exception_handler(HealthAdvisorError)
    async def health_advisor_error_handler(_: Request, exc: HealthAdvisorError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(_: Request, exc: Exception):
        await logger.aerror("unhandled_exception", error=str(exc), type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务内部错误，请稍后重试"},
        )
