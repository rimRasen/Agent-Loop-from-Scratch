import pytest

from agent_loop.model import cost_usd


def test_known_model_is_priced_per_million_tokens() -> None:
    # claude-opus-5-5: $4 in, $20 out per MTok
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.00)
    assert cost_usd("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.00)
    assert cost_usd("claude-opus-5-5", 1_000, 500) == pytest.approx(0.004 + 0.010)


def test_unknown_model_has_no_cost_rather_than_a_guess() -> None:
    assert cost_usd("claude-opus-5-5-typo", 1_000, 1_000) is None


def test_long_prompt_tier_applies_above_threshold_only() -> None:
    # claude-haiku-5-5: $0.10/$0.50 up to 100K input, $0.50/$2.50 beyond
    assert cost_usd("claude-haiku-5-5", 100_000, 0) == pytest.approx(0.01)
    assert cost_usd("claude-haiku-5-5", 100_001, 0) == pytest.approx(100_001 * 0.50 / 1e6)
    assert cost_usd("claude-haiku-5-5", 100_001, 1_000) == pytest.approx(
        (100_001 * 0.50 + 1_000 * 2.50) / 1e6
    )
