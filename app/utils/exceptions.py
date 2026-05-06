"""自定义异常体系"""


class HealthAdvisorError(Exception):
    """基础异常"""

    def __init__(self, message: str = "服务内部错误", status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(HealthAdvisorError):
    def __init__(self, message: str = "认证失败"):
        super().__init__(message=message, status_code=401)


class AuthorizationError(HealthAdvisorError):
    def __init__(self, message: str = "权限不足"):
        super().__init__(message=message, status_code=403)


class NotFoundError(HealthAdvisorError):
    def __init__(self, resource: str = "资源"):
        super().__init__(message=f"{resource}不存在", status_code=404)


class ValidationError(HealthAdvisorError):
    def __init__(self, message: str = "参数校验失败"):
        super().__init__(message=message, status_code=422)


class LLMError(HealthAdvisorError):
    """LLM 调用异常"""

    def __init__(self, message: str = "LLM 服务不可用"):
        super().__init__(message=message, status_code=503)


class MemoryError(HealthAdvisorError):
    """记忆系统异常"""

    def __init__(self, message: str = "记忆系统错误"):
        super().__init__(message=message, status_code=500)


class RAGError(HealthAdvisorError):
    """RAG 检索异常"""

    def __init__(self, message: str = "检索系统错误"):
        super().__init__(message=message, status_code=500)
