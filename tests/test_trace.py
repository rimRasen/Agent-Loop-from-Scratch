import json
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_loop.trace import FIELDS, Span, TraceFormatError, Tracer, read_spans


def fake_clock(step: int = 10) -> Callable[[], int]:
    """Each call returns the previous value + step, so durations are exact."""
    t = [1_000]

    def now() -> int:
        t[0] += step
        return t[0]

    return now


def make_tracer(tmp_path: Path) -> Tracer:
    return Tracer(tmp_path / "run.jsonl", run_id="r1", clock=fake_clock())


def test_span_is_written_with_exactly_the_spec_fields(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr, tr.span("model", "call") as s:
        s.record_usage(input_tokens=120, output_tokens=30, cost_usd=0.0011)

    [line] = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    obj = json.loads(line)
    assert list(obj) == list(FIELDS)  # same keys, same order
    assert obj["run_id"] == "r1"
    assert obj["parent_id"] is None
    assert (obj["kind"], obj["name"]) == ("model", "call")
    assert (obj["input_tokens"], obj["output_tokens"], obj["cost_usd"]) == (120, 30, 0.0011)
    assert obj["status"] == "ok" and obj["error"] is None
    assert obj["end_ms"] - obj["start_ms"] == 10


def test_tool_span_has_null_tokens_not_zero(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr, tr.span("tool", "shell"):
        pass
    [span] = read_spans(tmp_path / "run.jsonl")
    assert span.input_tokens is None and span.output_tokens is None and span.cost_usd is None


def test_nested_spans_link_to_parent_and_are_written_child_first(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr:
        with tr.span("run", "root") as root:
            with tr.span("tool", "delegate") as tool:
                with tr.span("model", "child-call") as child:
                    pass
            with tr.span("model", "sibling") as sibling:
                pass

    spans = read_spans(tmp_path / "run.jsonl")
    assert [s.name for s in spans] == ["child-call", "delegate", "sibling", "root"]
    by_name = {s.name: s for s in spans}
    assert by_name["root"].parent_id is None
    assert by_name["delegate"].parent_id == root.span_id
    assert by_name["child-call"].parent_id == tool.span_id
    assert by_name["sibling"].parent_id == root.span_id  # stack popped after delegate closed
    assert len({root.span_id, tool.span_id, child.span_id, sibling.span_id}) == 4


def test_exception_is_recorded_and_still_propagates(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr:
        with pytest.raises(RuntimeError, match="boom"), tr.span("run", "root"):
            with tr.span("model", "call"):
                raise RuntimeError("boom")

    spans = read_spans(tmp_path / "run.jsonl")
    assert [(s.name, s.status, s.error) for s in spans] == [
        ("call", "error", "RuntimeError: boom"),
        ("root", "error", "RuntimeError: boom"),
    ]


def test_fail_marks_error_without_raising(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr, tr.span("tool", "shell") as s:
        s.fail("exit code 2")
    [span] = read_spans(tmp_path / "run.jsonl")
    assert (span.status, span.error) == ("error", "exit code 2")


def test_explicit_fail_message_is_kept_when_an_exception_follows(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr:
        with pytest.raises(ValueError), tr.span("tool", "shell") as s:
            s.fail("timeout after 5s")
            raise ValueError("secondary")
    [span] = read_spans(tmp_path / "run.jsonl")
    assert span.error == "timeout after 5s"


def test_span_is_on_disk_as_soon_as_it_ends(tmp_path: Path) -> None:
    tr = make_tracer(tmp_path)
    with tr.span("run", "root"):
        with tr.span("model", "first"):
            pass
        # Outer span still open, file still open: the finished child must already be readable.
        assert [s.name for s in read_spans(tmp_path / "run.jsonl")] == ["first"]
    tr.close()


def test_reader_skips_a_torn_final_line(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr, tr.span("model", "ok"):
        pass
    path = tmp_path / "run.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"run_id": "r1", "span_id": "ab')  # hard kill mid-write, no newline
    assert [s.name for s in read_spans(path)] == ["ok"]


def test_reader_rejects_corruption_before_the_last_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with make_tracer(tmp_path) as tr, tr.span("model", "ok"):
        pass
    good = path.read_text(encoding="utf-8")
    path.write_text("not json\n" + good, encoding="utf-8")
    with pytest.raises(TraceFormatError, match=":1: invalid JSON"):
        read_spans(path)


@pytest.mark.parametrize("mutate", [
    lambda o: o.pop("cost_usd"),
    lambda o: o.update(extra=1),
    lambda o: o.update(status="weird"),
])
def test_reader_rejects_lines_that_do_not_match_the_format(
    tmp_path: Path, mutate: Callable[[dict[str, object]], object]
) -> None:
    path = tmp_path / "run.jsonl"
    with make_tracer(tmp_path) as tr, tr.span("model", "ok"):
        pass
    obj = json.loads(path.read_text(encoding="utf-8"))
    mutate(obj)
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(TraceFormatError):
        read_spans(path)


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr, tr.span("model", "café ☕") as s:
        s.record_usage(1, 2, None)
        s.fail("non-ASCII error: ñ")
    [span] = read_spans(tmp_path / "run.jsonl")
    assert span == Span(run_id="r1", span_id=s.span_id, parent_id=None, kind="model",
                        name="café ☕", start_ms=1010, end_ms=1020, input_tokens=1,
                        output_tokens=2, cost_usd=None, status="error",
                        error="non-ASCII error: ñ", attrs={})


def test_two_runs_appending_to_one_file_keep_their_run_ids(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    for run_id in ("a", "b"):
        with Tracer(path, run_id=run_id, clock=fake_clock()) as tr, tr.span("run", "root"):
            pass
    assert [s.run_id for s in read_spans(path)] == ["a", "b"]


def test_attrs_round_trip_and_default_to_empty(tmp_path: Path) -> None:
    with make_tracer(tmp_path) as tr:
        with tr.span("run", "root") as run:
            with tr.span("tool", "shell") as tool:
                tool.set("truncated", True)
                tool.set("original_chars", 48213)
            run.set("stop_reason", "max_steps")
        with tr.span("model", "plain"):
            pass
    by_name = {s.name: s for s in read_spans(tmp_path / "run.jsonl")}
    assert by_name["shell"].attrs == {"truncated": True, "original_chars": 48213}
    assert by_name["root"].attrs == {"stop_reason": "max_steps"}
    assert by_name["plain"].attrs == {}


def test_unserialisable_attr_fails_at_the_call_site_not_when_the_span_closes(
    tmp_path: Path,
) -> None:
    with make_tracer(tmp_path) as tr:
        with pytest.raises(TypeError), tr.span("tool", "shell") as s:
            s.set("bad", object())
    [span] = read_spans(tmp_path / "run.jsonl")  # span still written, with the error recorded
    assert span.status == "error" and span.attrs == {}
