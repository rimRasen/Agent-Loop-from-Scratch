from agent_loop.context import truncate


def test_short_text_is_unchanged() -> None:
    t = truncate("abc", 3)
    assert (t.text, t.truncated, t.original_chars) == ("abc", False, 3)


def test_long_text_keeps_head_and_tail() -> None:
    t = truncate("HEAD" + "-" * 100 + "TAIL", 8)
    assert t.truncated and t.original_chars == 108
    assert t.text.startswith("HEAD\n[") and t.text.endswith("]\nTAIL")
    assert "100 of 108 characters omitted" in t.text


def test_odd_limit_and_limit_of_one() -> None:
    assert truncate("abcdefg", 3).text.startswith("a\n")
    assert truncate("abcdefg", 3).text.endswith("\nfg")
    one = truncate("abcdefg", 1)
    assert one.text.endswith("\ng") and one.truncated
