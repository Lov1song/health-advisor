"""用户相关 Pydantic 模型"""

from uuid import UUID

from pydantic import BaseModel, Field


# ---- 注册 / 登录 ----

class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, examples=["alice"])
    password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---- 用户档案 ----

class UserProfileUpdate(BaseModel):
    age: int | None = Field(None, ge=1, le=150)
    gender: str | None = Field(None, pattern=r"^(male|female|other)$")
    height_cm: float | None = Field(None, ge=30, le=300)
    weight_kg: float | None = Field(None, ge=5, le=500)
    allergies: list[str] | None = None
    chronic_conditions: list[str] | None = None
    dietary_preferences: dict | None = None
    emotional_baseline: dict | None = None


class UserProfileResponse(BaseModel):
    user_id: UUID
    age: int | None = None
    gender: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    allergies: list[str] = []
    chronic_conditions: list[str] = []
    dietary_preferences: dict = {}
    emotional_baseline: dict = {}

    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: UUID
    username: str
    profile: UserProfileResponse | None = None

    model_config = {"from_attributes": True}
