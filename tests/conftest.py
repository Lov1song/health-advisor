"""测试公共 fixtures"""

import pytest


@pytest.fixture
def sample_user_profile():
    return {
        "age": 30,
        "gender": "male",
        "height_cm": 175.0,
        "weight_kg": 75.0,
        "allergies": ["花生"],
        "chronic_conditions": [],
        "dietary_preferences": {"vegetarian": False},
        "emotional_baseline": {"stress_level": "medium"},
    }
