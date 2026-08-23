"""Test decay: formula correctness, category half-life, retrieval boost"""
from cow.memory_engine.decay import calculate_strength
from cow.memory_engine.models import MemoryItem


class TestDecayFormula:
    def test_day_zero_is_confidence(self):
        item = MemoryItem(content="test", category="preference", confidence=0.8)
        s = calculate_strength(item, current_days=0)
        assert abs(s - 0.8) < 0.01

    def test_decay_over_time(self):
        item = MemoryItem(content="test", category="preference", confidence=0.8)
        s0 = calculate_strength(item, current_days=0)
        s30 = calculate_strength(item, current_days=30)
        assert s30 < s0
        assert s30 > 0  # 还没到 0

    def test_retrieval_boost(self):
        item = MemoryItem(content="test", category="preference", confidence=0.8, retrieval_count=5)
        s = calculate_strength(item, current_days=10)
        # 有 5 次检索加成，比没有加成的要高
        item_no_boost = MemoryItem(content="test", category="preference", confidence=0.8, retrieval_count=0)
        s_no = calculate_strength(item_no_boost, current_days=10)
        assert s > s_no

    def test_identity_last_longer_than_feeling(self):
        """identity(60天)比 feeling(10天)衰减慢"""
        from cow.memory_engine.config import CATEGORY_HALF_LIFE
        id_item = MemoryItem(content="test", category="identity", confidence=0.8,
                             half_life_days=CATEGORY_HALF_LIFE["identity"])
        feel_item = MemoryItem(content="test", category="feeling", confidence=0.8,
                               half_life_days=CATEGORY_HALF_LIFE["feeling"])
        s_id = calculate_strength(id_item, current_days=30)
        s_feel = calculate_strength(feel_item, current_days=30)
        assert s_id > s_feel, f"identity={s_id:.4f} should > feeling={s_feel:.4f}"
