"""健康指标 API — BMI / BMR / TDEE"""

from fastapi import APIRouter

from app.schemas.health import (
    BMIRequest,
    BMIResponse,
    BMRRequest,
    BMRResponse,
    TDEERequest,
    TDEEResponse,
)
from app.tools.health_calc import calculate_bmi, calculate_bmr, calculate_tdee

router = APIRouter(prefix="/health", tags=["health"])


@router.post("/bmi", response_model=BMIResponse)
async def compute_bmi(body: BMIRequest):
    """计算 BMI 指数"""
    return calculate_bmi(body.weight_kg, body.height_cm)


@router.post("/bmr", response_model=BMRResponse)
async def compute_bmr(body: BMRRequest):
    """计算基础代谢率"""
    return calculate_bmr(body.weight_kg, body.height_cm, body.age, body.gender)


@router.post("/tdee", response_model=TDEEResponse)
async def compute_tdee(body: TDEERequest):
    """计算每日总能量消耗"""
    return calculate_tdee(
        body.weight_kg, body.height_cm, body.age, body.gender, body.activity_level
    )
