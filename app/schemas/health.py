"""健康指标计算 Pydantic 模型"""

from pydantic import BaseModel, Field


class BMIRequest(BaseModel):
    weight_kg: float = Field(..., ge=5, le=500, description="体重 (kg)")
    height_cm: float = Field(..., ge=30, le=300, description="身高 (cm)")


class BMIResponse(BaseModel):
    bmi: float
    category: str
    advice: str


class BMRRequest(BaseModel):
    weight_kg: float = Field(..., ge=5, le=500)
    height_cm: float = Field(..., ge=30, le=300)
    age: int = Field(..., ge=1, le=150)
    gender: str = Field(..., pattern=r"^(male|female)$")


class BMRResponse(BaseModel):
    bmr_kcal: float
    formula: str = "Mifflin-St Jeor"


class TDEERequest(BaseModel):
    weight_kg: float = Field(..., ge=5, le=500)
    height_cm: float = Field(..., ge=30, le=300)
    age: int = Field(..., ge=1, le=150)
    gender: str = Field(..., pattern=r"^(male|female)$")
    activity_level: str = Field(
        ...,
        pattern=r"^(sedentary|light|moderate|active|very_active)$",
        description="活动水平: sedentary/light/moderate/active/very_active",
    )


class TDEEResponse(BaseModel):
    bmr_kcal: float
    tdee_kcal: float
    activity_level: str
    activity_multiplier: float
    advice: str
