"""Calls the real API. Costs money. Run on purpose with: uv run pytest -m live"""

import os

import pytest

from agent_loop.model import AnthropicModel

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("AGENT_MODEL"), reason="AGENT_MODEL not set")
def test_real_call_round_trips() -> None:
    model = AnthropicModel.from_env()
    resp = model.complete(system="Reply with exactly: ok",
                          messages=[{"role": "user", "content": "ping"}], tools=[],
                          max_tokens=1024)
    assert resp.stop_reason == "end_turn"
    assert "ok" in resp.text().lower()
    assert resp.input_tokens > 0 and resp.output_tokens > 0
    assert resp.request_id
    assert resp.cost_usd() is not None
    assert resp.trace_attrs()["requested_model"] == model.model_id
