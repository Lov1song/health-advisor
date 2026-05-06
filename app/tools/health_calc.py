"""健康指标计算工具 — BMI / BMR / TDEE"""

from app.schemas.health import BMIResponse, BMRResponse, TDEEResponse

# ---- BMI ----

_BMI_RANGES = [
    (18.5, "偏瘦", "您的体重偏低，建议适当增加营养摄入，注意均衡饮食。"),
    (24.0, "正常", "您的体重在健康范围内，请继续保持良好的生活习惯。"),
    (28.0, "超重", "您的体重偏高，建议控制饮食热量并增加运动。"),
    (float("inf"), "肥胖", "建议尽快咨询专业医师，制定科学的减重方案。"),
]


def calculate_bmi(weight_kg: float, height_cm: float) -> BMIResponse:
    """计算 BMI 并给出分类和建议（中国标准）"""
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("身高和体重必须为正数")

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    for threshold, category, advice in _BMI_RANGES:
        if bmi < threshold:
            return BMIResponse(bmi=bmi, category=category, advice=advice)

    # fallback（不应到达）
    return BMIResponse(bmi=bmi, category="未知", advice="请咨询医师。")


# ---- BMR (Mifflin-St Jeor) ----

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> BMRResponse:
    """基础代谢率 — Mifflin-St Jeor 公式"""
    if gender == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    return BMRResponse(bmr_kcal=round(bmr, 1))


# ---- TDEE ----

_ACTIVITY_MULTIPLIERS = {
    "sedentary": (1.2, "久坐不动（办公室工作，几乎不运动）"),
    "light": (1.375, "轻度活动（每周运动 1-3 天）"),
    "moderate": (1.55, "中等活动（每周运动 3-5 天）"),
    "active": (1.725, "高度活动（每周运动 6-7 天）"),
    "very_active": (1.9, "极高活动（体力劳动或每天高强度训练）"),
}


def calculate_tdee(
    weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str
) -> TDEEResponse:
    """每日总能量消耗 = BMR × 活动系数"""
    bmr_result = calculate_bmr(weight_kg, height_cm, age, gender)
    multiplier, description = _ACTIVITY_MULTIPLIERS[activity_level]
    tdee = round(bmr_result.bmr_kcal * multiplier, 1)

    # 生成建议
    if tdee < 1500:
        advice = "您的每日能量消耗较低，注意不要过度节食。"
    elif tdee < 2000:
        advice = f"您的每日能量消耗约 {tdee} kcal，适合轻度控制饮食。"
    elif tdee < 2500:
        advice = f"您的每日能量消耗约 {tdee} kcal，保持均衡饮食即可。"
    else:
        advice = f"您的每日能量消耗较高（{tdee} kcal），注意补充足够的碳水和蛋白质。"

    return TDEEResponse(
        bmr_kcal=bmr_result.bmr_kcal,
        tdee_kcal=tdee,
        activity_level=activity_level,
        activity_multiplier=multiplier,
        advice=advice,
    )
