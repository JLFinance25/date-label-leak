"""Tests for src/run_gemini.py with fake replies and a fake key only. Nothing here talks to Google or
reads the real Keychain. Run: python3 -m pytest tests

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import argparse
import io
import json
import subprocess
import sys
import urllib.error
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_gemini  # noqa: E402
import run_models  # noqa: E402

FAKE_KEY = "FAKE-test-key-0123456789abcdef"
GOOD_REPLY = '{"forecast": [0.1, 0.2, 0.3]}'


# ---------- a pretend Google ----------

def google_reply(text=GOOD_REPLY, tokens_in=200, tokens_out=10, thinking=990, traffic="ON_DEMAND_FLEX"):
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": tokens_in, "candidatesTokenCount": tokens_out,
                          "thoughtsTokenCount": thinking, "totalTokenCount": tokens_in + tokens_out + thinking,
                          "trafficType": traffic},
        "modelVersion": "gemini-3.8-flash-test",
    }


def http_error(code, message):
    body = json.dumps({"error": {"code": code, "message": message, "status": "TEST"}}).encode()
    return urllib.error.HTTPError("https://example.invalid/", code, "error", {}, io.BytesIO(body))


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeGoogle:
    """Stands in for urllib's urlopen. plan(n) gives what the n-th request gets (1-based):
    a reply dict, or an exception to raise."""

    def __init__(self, plan=None):
        self.plan = plan or (lambda n: google_reply())
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        outcome = self.plan(len(self.requests))
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


@pytest.fixture
def google(monkeypatch):
    fake = FakeGoogle()
    monkeypatch.setattr(run_gemini, "_urlopen", fake)
    monkeypatch.setattr(run_gemini, "read_key", lambda: FAKE_KEY)
    monkeypatch.setattr(run_gemini, "_sleep", lambda seconds: None)
    return fake


def run_real(tmp_path, *extra, max_cost="100"):
    """A 'real model' run, but against the pretend Google above."""
    return run_gemini.main(["--model", "gemini-3.8-flash", "--pilot", "--max-cost", max_cost,
                            "--project", "test-project", "--out-dir", str(tmp_path), *extra])


def run_fake(tmp_path, *extra):
    return run_gemini.main(["--model", "fake", "--out-dir", str(tmp_path), *extra])


def records(tmp_path, model="gemini-3.8-flash", kind="forecasts"):
    path = tmp_path / model / f"{kind}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answered(recs):
    return [r for r in recs if not str(r["parse_error"] or "").startswith("request failed")]


# ---------- 3 replies per prompt ----------

def test_three_replies_per_prompt(tmp_path):
    assert run_fake(tmp_path, "--pilot", "--limit", "5") == 0
    recs = records(tmp_path, model="fake")
    assert len(recs) == 15
    by_prompt = {}
    for r in recs:
        by_prompt.setdefault(r["prompt_id"], []).append(r["reply_index"])
    assert len(by_prompt) == 5
    assert all(sorted(v) == [0, 1, 2] for v in by_prompt.values())


def test_quiz_three_replies(tmp_path):
    assert run_fake(tmp_path, "--quiz", "--limit", "2") == 0
    recs = records(tmp_path, model="fake", kind="quiz")
    assert len(recs) == 6
    assert {r["parsed"] for r in recs} <= {"rise", "fall", None}


def test_three_replies_with_several_workers(tmp_path, google):
    assert run_real(tmp_path, "--limit", "4", "--workers", "3") == 0
    recs = records(tmp_path)
    assert Counter((r["prompt_id"], r["reply_index"]) for r in recs) == Counter(
        {(r["prompt_id"], r["reply_index"]): 1 for r in recs})
    assert len(recs) == 12 and len(google.requests) == 12


# ---------- resume ----------

def test_resume_with_no_duplicates(tmp_path):
    assert run_fake(tmp_path, "--pilot", "--limit", "3") == 0
    assert run_fake(tmp_path, "--pilot", "--limit", "4") == 0
    recs = records(tmp_path, model="fake")
    keys = [(r["prompt_id"], r["reply_index"]) for r in recs]
    assert len(keys) == 21 and len(set(keys)) == 21

    # A crash after 2 of a prompt's 3 replies: the re-run adds only the missing third reply.
    path = tmp_path / "fake" / "forecasts.jsonl"
    lines = path.read_text().splitlines()
    lost = json.loads(lines[-1])
    path.write_text("\n".join(lines[:-1]) + "\n")
    assert run_fake(tmp_path, "--pilot", "--limit", "1") == 0
    recs = records(tmp_path, model="fake")
    keys = [(r["prompt_id"], r["reply_index"]) for r in recs]
    assert len(keys) == 21 and len(set(keys)) == 21
    assert (recs[-1]["prompt_id"], recs[-1]["reply_index"]) == (lost["prompt_id"], lost["reply_index"])


def test_nothing_left_to_do(tmp_path, capsys):
    assert run_fake(tmp_path, "--quiz") == 0
    assert run_fake(tmp_path, "--quiz") == 0
    assert len(records(tmp_path, model="fake", kind="quiz")) == 99 * 3
    assert "Nothing to do" in capsys.readouterr().out


# ---------- failures ----------

def test_busy_server_is_retried_then_answers(tmp_path, google):
    google.plan = lambda n: http_error(429, "Resource exhausted") if n <= 2 else google_reply()
    assert run_real(tmp_path, "--limit", "1") == 0
    recs = records(tmp_path)
    assert len(recs) == 3 and all(r["parse_ok"] for r in recs)
    assert recs[0]["tries"] == 3 and len(google.requests) == 5


def test_request_failures_are_saved_then_retried_on_rerun(tmp_path, google):
    google.plan = lambda n: http_error(503, "Service unavailable")
    assert run_real(tmp_path, "--limit", "2") == 0
    recs = records(tmp_path)
    assert len(recs) == 6 and not answered(recs)
    assert all(r["parse_error"].startswith("request failed: HTTP 503") for r in recs)
    assert all(r["tries"] == run_gemini.MAX_TRIES and r["cost_usd"] == 0 for r in recs)
    assert len(google.requests) == 6 * run_gemini.MAX_TRIES

    google.plan = lambda n: google_reply()
    assert run_real(tmp_path, "--limit", "2") == 0
    recs = records(tmp_path)
    good = answered(recs)
    assert len(recs) == 12 and len(good) == 6
    assert len({(r["prompt_id"], r["reply_index"]) for r in good}) == 6

    assert run_real(tmp_path, "--limit", "2") == 0  # the 2 prompts are complete; the next 2 are new
    assert len(answered(records(tmp_path))) == 12


def test_bad_key_stops_at_once(tmp_path, google, capsys):
    google.plan = lambda n: http_error(403, "Permission denied")
    assert run_real(tmp_path) == 2
    assert len(google.requests) == 1
    assert records(tmp_path) == []
    assert "HTTP 403" in capsys.readouterr().out


def test_many_failures_in_a_row_stop_the_run(tmp_path, google, capsys):
    google.plan = lambda n: http_error(500, "Internal error")
    assert run_real(tmp_path, "--limit", "10") == 2
    assert len(records(tmp_path)) == run_gemini.MAX_FAILURES_IN_A_ROW
    assert "in a row failed" in capsys.readouterr().out


def test_blocked_or_empty_reply_counts_as_a_parse_failure_not_a_retry(tmp_path, google):
    google.plan = lambda n: {"promptFeedback": {"blockReason": "OTHER"}, "usageMetadata": {"promptTokenCount": 200}}
    assert run_real(tmp_path, "--limit", "1") == 0
    recs = records(tmp_path)
    assert len(recs) == 3 and len(answered(recs)) == 3
    assert all(r["parse_error"] == "empty reply" and r["finish_reason"] == "OTHER" for r in recs)
    assert len(google.requests) == 3


# ---------- the cost guard ----------

PER_REPLY = (200 * 0.375 + (10 + 990) * 1.875) / 1e6  # google_reply()'s tokens at Flash's Flex prices


@pytest.mark.parametrize("workers", ["1", "3"])
def test_cost_guard_stops_before_the_limit(tmp_path, google, capsys, workers):
    assert run_real(tmp_path, "--workers", workers, max_cost="0.02") == 2
    spent = sum(r["cost_usd"] for r in records(tmp_path))
    assert 0 < spent <= 0.02
    assert spent + 2 * PER_REPLY > 0.02  # it didn't stop absurdly early either
    assert "--max-cost limit of $0.02" in capsys.readouterr().out

    # Re-running with the same limit makes no new calls: the money already spent still counts.
    calls = len(google.requests)
    assert run_real(tmp_path, max_cost="0.02") == 2
    assert len(google.requests) == calls

    # A higher limit carries on from where it stopped.
    assert run_real(tmp_path, "--limit", "3", max_cost="1") == 0
    assert len(google.requests) > calls


def test_limit_below_one_reply_makes_no_call(tmp_path, google):
    # Before any reply's cost is known, the guard assumes a generous first guess (about 1.1 cents).
    assert run_real(tmp_path, max_cost="0.005") == 2
    assert google.requests == []


def test_cost_arithmetic():
    # Google Cloud's published prices for 3.8 Flash until Dec 31, 2026 (per 1M tokens).
    assert run_gemini.PRICES["gemini-3.8-flash"] == {"standard": {"input": 0.75, "output": 3.75},
                                                     "flex": {"input": 0.375, "output": 1.875}}
    tokens = {"input": 1_000_000, "output": 0, "thinking": 0, "total": 0}
    assert run_gemini.cost_usd("gemini-3.8-flash", tokens, "ON_DEMAND") == pytest.approx(0.75)
    assert run_gemini.cost_usd("gemini-3.8-flash", tokens, "ON_DEMAND_FLEX") == pytest.approx(0.375)
    assert run_gemini.cost_usd("gemini-3.8-flash", tokens, None) == pytest.approx(0.75)  # unknown: assume standard
    tokens = {"input": 0, "output": 400_000, "thinking": 600_000, "total": 0}  # thinking billed as output
    assert run_gemini.cost_usd("gemini-3.8-flash", tokens, "ON_DEMAND") == pytest.approx(3.75)


def test_flex_asked_but_not_served_is_priced_as_standard(tmp_path, google, capsys):
    google.plan = lambda n: google_reply(traffic="ON_DEMAND")
    assert run_real(tmp_path, "--limit", "1") == 0
    assert all(r["cost_usd"] == pytest.approx(PER_REPLY * 2) for r in records(tmp_path))
    assert capsys.readouterr().out.count("Warning: asked for Flex") == 1


def test_old_prices_block_real_runs(tmp_path, google, monkeypatch):
    monkeypatch.setattr(run_gemini, "PRICES_GOOD_UNTIL", run_gemini.date(2020, 1, 1))
    with pytest.raises(SystemExit):
        run_real(tmp_path, "--limit", "1")
    assert google.requests == []


def test_max_cost_is_required_for_a_real_run(tmp_path, google):
    with pytest.raises(SystemExit):
        run_gemini.main(["--model", "gemini-3.8-flash", "--pilot", "--out-dir", str(tmp_path)])
    assert google.requests == []


def test_tokens_and_cost_are_saved(tmp_path, google):
    assert run_real(tmp_path, "--limit", "1") == 0
    r = records(tmp_path)[0]
    assert r["tokens"] == {"input": 200, "output": 10, "thinking": 990, "total": 1200}
    assert r["cost_usd"] == pytest.approx(PER_REPLY)
    assert r["model_version"] == "gemini-3.8-flash-test"


# ---------- the key ----------

def test_key_never_appears_in_output_or_saved_files(tmp_path, google, capsys):
    def plan(n):
        if n == 1:
            return google_reply()
        if n == 2:  # a server message that echoes the key
            return http_error(503, f"backend saw key {FAKE_KEY}")
        if n == 3:  # a network error that mentions the key
            return urllib.error.URLError(f"connection reset while sending {FAKE_KEY}")
        if n <= 6:
            return google_reply()
        return http_error(400, f"API key not valid: {FAKE_KEY}")
    google.plan = plan
    assert run_real(tmp_path, "--limit", "5") == 2

    out = capsys.readouterr()
    assert FAKE_KEY not in out.out and FAKE_KEY not in out.err
    assert "[key hidden]" in out.out  # the 400 message was printed, scrubbed
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text()
    # The key travels only in the request header, never in the address.
    for req in google.requests:
        assert req.get_header("X-goog-api-key") == FAKE_KEY
        assert FAKE_KEY not in req.full_url


def test_key_read_from_keychain(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=FAKE_KEY + "\n", stderr="")
    monkeypatch.setattr(run_gemini.subprocess, "run", fake_run)
    monkeypatch.setenv("USER", "someone")
    assert run_gemini.read_key() == FAKE_KEY
    assert seen["cmd"] == ["security", "find-generic-password", "-a", "someone", "-s", "gemini-platform-key", "-w"]


def test_missing_key_stops_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(run_gemini.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 44, stdout="", stderr="not found"))
    with pytest.raises(SystemExit, match="Nothing was sent to Google"):
        run_gemini.read_key()


def test_fake_mode_never_reads_the_key(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("fake mode must not read the key")
    monkeypatch.setattr(run_gemini, "read_key", boom)
    assert run_fake(tmp_path, "--pilot", "--limit", "1") == 0


# ---------- what is sent ----------

def test_request_matches_the_prereg_settings(tmp_path, google):
    assert run_real(tmp_path, "--limit", "1") == 0
    body = json.loads(google.requests[0].data)
    config = body["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == run_models.FORECAST_SCHEMA
    assert "temperature" not in json.dumps(body)  # PREREG 5.2: no temperature setting
    assert body["contents"][0]["parts"][0]["text"].startswith("Below are 24 consecutive monthly values")
    assert google.requests[0].full_url == ("https://aiplatform.googleapis.com/v1/projects/test-project/locations/"
                                           "global/publishers/google/models/gemini-3.8-flash:generateContent")
    assert records(tmp_path)[0]["settings"]["temperature"] == "not set"


def test_flex_by_default_and_standard_on_request(tmp_path, google):
    assert run_real(tmp_path, "--limit", "1") == 0
    req = google.requests[0]
    assert req.get_header("X-vertex-ai-llm-request-type") == "shared"
    assert req.get_header("X-vertex-ai-llm-shared-request-type") == "flex"
    assert records(tmp_path)[0]["traffic_type"] == "ON_DEMAND_FLEX"

    google.plan = lambda n: google_reply(traffic="ON_DEMAND")
    assert run_real(tmp_path / "std", "--limit", "1", "--tier", "standard") == 0
    req = google.requests[-1]
    assert req.get_header("X-vertex-ai-llm-shared-request-type") is None
    assert req.get_header("X-vertex-ai-llm-request-type") is None


def test_flex_waits_longer_between_retries(tmp_path, google, monkeypatch):
    waits = []
    monkeypatch.setattr(run_gemini, "_sleep", waits.append)
    google.plan = lambda n: http_error(429, "Resource exhausted") if n <= 2 else google_reply()
    assert run_real(tmp_path, "--limit", "1") == 0
    assert [int(w) for w in waits] == [30, 60]
    waits.clear()
    google.plan = lambda n: http_error(429, "Resource exhausted") if n in (6, 7) else google_reply()
    assert run_real(tmp_path / "std", "--limit", "1", "--tier", "standard") == 0
    assert [int(w) for w in waits] == [2, 4]


def test_quiz_uses_the_quiz_schema(tmp_path, google):
    google.plan = lambda n: google_reply('{"answer": "fall"}')
    assert run_real(tmp_path, "--quiz", "--limit", "1") == 0
    assert json.loads(google.requests[0].data)["generationConfig"]["responseJsonSchema"] == run_models.QUIZ_SCHEMA
    assert [r["parsed"] for r in records(tmp_path, kind="quiz")] == ["fall"] * 3


def test_prompt_selection_matches_run_models():
    for pilot, quiz, n in [(True, False, 440), (False, False, 4700), (False, True, 99)]:
        args = argparse.Namespace(series=20, pilot=pilot, quiz=quiz)
        assert len(run_gemini.select_prompts(args)) == n
    assert run_gemini.select_prompts is run_models.select_prompts


def test_reading_google_replies():
    resp = {
        "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": GOOD_REPLY}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 6, "totalTokenCount": 11},
    }
    got = run_gemini.read_response(resp)
    assert got["reply"] == GOOD_REPLY
    assert got["tokens"] == {"input": 5, "output": 6, "thinking": 0, "total": 11}
    assert run_gemini.read_response({})["reply"] is None


def test_addresses():
    assert run_gemini.endpoint_url("gemini-3.8-flash", None) == (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.8-flash:generateContent")
    assert run_gemini.endpoint_url("gemini-3.1-pro-preview", "my-proj") == (
        "https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global"
        "/publishers/google/models/gemini-3.1-pro-preview:generateContent")


def test_lock_file_blocks_a_second_copy(tmp_path):
    folder = tmp_path / "fake"
    folder.mkdir()
    (folder / "forecasts.lock").touch()
    assert run_fake(tmp_path, "--pilot", "--limit", "1") == 1
    assert not (folder / "forecasts.jsonl").exists()
