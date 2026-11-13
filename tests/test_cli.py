from __future__ import annotations

from dataclasses import fields

import pytest

from agent_loop.cli import ask_on_terminal, main
from agent_loop.loop import ExtensionRequest, Limits

REQ = ExtensionRequest("token_budget", current=100, proposed=200, tokens_used=120,
                       cost_usd=0.01, steps=3)


@pytest.mark.parametrize(("answer", "expected"),
                         [("y", True), ("YES\n", True), ("", False), ("n", False),
                          ("sure", False)])
def test_terminal_approver(answer: str, expected: bool) -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    assert ask_on_terminal(read)(REQ) is expected
    assert "from 100 to 200" in prompts[0]


def test_terminal_approver_treats_eof_as_no() -> None:
    def eof(_: str) -> str:
        raise EOFError

    assert ask_on_terminal(eof)(REQ) is False


def test_every_limit_has_a_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    help_text = capsys.readouterr().out
    for f in fields(Limits):
        assert "--" + f.name.replace("_", "-") in help_text


def test_run_without_agent_model_fails_cleanly(monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    assert main(["run", "hi", "--max-steps", "5"]) == 2
    assert "AGENT_MODEL" in capsys.readouterr().err


def test_invalid_limit_from_the_command_line_is_rejected(
        monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        main(["run", "hi", "--max-steps", "0"])
