"""健康指标计算工具 — 单元测试"""

import pytest

from app.tools.health_calc import calculate_bmi, calculate_bmr, calculate_tdee


class TestBMI:
    def test_normal_weight(self):
        result = calculate_bmi(65, 175)
        assert result.bmi == 21.2
        assert result.category == "正常"

    def test_underweight(self):
        result = calculate_bmi(45, 170)
        assert result.bmi == 15.6
        assert result.category == "偏瘦"

    def test_overweight(self):
        result = calculate_bmi(80, 170)
        assert result.bmi == 27.7
        assert result.category == "超重"

    def test_obese(self):
        result = calculate_bmi(100, 170)
        assert result.bmi == 34.6
        assert result.category == "肥胖"

    def test_invalid_height(self):
        with pytest.raises(ValueError):
            calculate_bmi(70, 0)

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            calculate_bmi(-5, 170)


class TestBMR:
    def test_male(self):
        result = calculate_bmr(75, 175, 30, "male")
        # 10*75 + 6.25*175 - 5*30 + 5 = 750 + 1093.75 - 150 + 5 = 1698.75
        assert result.bmr_kcal == 1698.8
        assert result.formula == "Mifflin-St Jeor"

    def test_female(self):
        result = calculate_bmr(60, 165, 28, "female")
        # 10*60 + 6.25*165 - 5*28 - 161 = 600 + 1031.25 - 140 - 161 = 1330.25
        assert result.bmr_kcal == 1330.2


class TestTDEE:
    def test_sedentary(self):
        result = calculate_tdee(75, 175, 30, "male", "sedentary")
        assert result.activity_multiplier == 1.2
        expected_tdee = round(1698.8 * 1.2, 1)
        assert result.tdee_kcal == expected_tdee

    def test_active(self):
        result = calculate_tdee(60, 165, 28, "female", "active")
        assert result.activity_multiplier == 1.725
        assert result.tdee_kcal > result.bmr_kcal

    def test_has_advice(self):
        result = calculate_tdee(75, 175, 30, "male", "moderate")
        assert len(result.advice) > 0

    def test_invalid_activity(self):
        with pytest.raises(KeyError):
            calculate_tdee(75, 175, 30, "male", "invalid_level")
