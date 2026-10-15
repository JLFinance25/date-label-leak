"""Send the study prompts to Gemini through Google Cloud's Gemini platform and save every reply
(PREREG.md 5.2, runs G1 to G4 in 5.3, and the deviation in section 8).

Settings fixed by PREREG 5.2: thinking level "low", the same JSON schemas as run_models.py, no
temperature setting, and 3 replies per prompt (D8: Gemini can't run at temperature 0, so the forecast
is the average of 3 replies). Each reply is its own record with a reply_index of 0, 1 or 2, appended
as soon as it arrives. Re-running the same command skips replies already saved and retries the ones
that failed as requests, so a crash or a sleeping Mac loses nothing.

Calls are ordinary online calls on Google's "global" endpoint. By default they ask for Flex PayGo,
which Google prices at half the standard rate, the same as batch jobs, in exchange for slower and more
throttled replies. --tier standard asks for the standard rate instead. Google says in each reply which
kind of traffic it served (usageMetadata.trafficType), and the cost estimate uses that.

Every reply records its token counts (input, output, thinking) and an estimated cost from Google
Cloud's published prices (PRICES below). --max-cost stops the run before the estimated cost of the
output file (earlier runs of the same command included) would pass the limit.

The API key is read from the macOS Keychain on every run (service "gemini-platform-key"). It is
never printed, logged or saved. It goes only into the request header, and any error text is
scrubbed of it before it is printed or saved.

Usage (Jayden runs these; each one spends the parent's Google Cloud credit, so each needs his yes):
  python3 src/run_gemini.py --project PROJECT_ID --pilot --max-cost 5          # G1 cost test: 440 x 3
  python3 src/run_gemini.py --project PROJECT_ID --max-cost 40                 # G2: 4,700 x 3
  python3 src/run_gemini.py --project PROJECT_ID --quiz --max-cost 2           # G2 quiz: 99 x 3
  python3 src/run_gemini.py --project PROJECT_ID --model gemini-3.1-pro-preview --pilot --max-cost 10  # G3
  python3 src/run_gemini.py --model fake --pilot             # test the loop with fake replies (no key, no Google)

Prompt selection is the same as run_models.py (it imports select_prompts and the two schemas from
there): first 20 series, the 440-prompt pilot with --pilot, the 99 quiz prompts with --quiz.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import argparse
import getpass
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from pathlib import Path

from read_reply import parse_forecast, parse_quiz
from run_models import FORECAST_SCHEMA, QUIZ_SCHEMA, ROOT, select_prompts

MODELS = ("gemini-3.8-flash", "gemini-3.1-pro-preview", "fake")
REPLIES_PER_PROMPT = 3
THINKING_LEVEL = "LOW"
KEYCHAIN_SERVICE = "gemini-platform-key"
HOST = "https://aiplatform.googleapis.com"  # the "global" endpoint; Flex PayGo works only there

# Flex PayGo: Google's page lists these two headers for "Flex only", and allows a request timeout of up
# to 30 minutes (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/flex-paygo,
# read Oct 12, 2026).
FLEX_HEADERS = {"X-Vertex-AI-LLM-Request-Type": "shared", "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
                "X-Server-Timeout": "1800"}
TIMEOUT_SECONDS = {"flex": 1860, "standard": 600}

# US dollars per 1 million tokens on the global endpoint, from
# https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing (read Oct 12, 2026).
# Thinking tokens are billed as output ("Text output (response and reasoning)"). Flex and batch share
# one price, half of standard. 3.8 Flash's prices double on Jan 1, 2027, so after PRICES_GOOD_UNTIL this
# script refuses real runs until the table is updated. Pro's prices are for prompts up to 200K tokens.
# "fake" uses Flash's prices so the tests exercise the cost guard with realistic numbers.
PRICES = {
    "gemini-3.8-flash": {"standard": {"input": 0.75, "output": 3.75}, "flex": {"input": 0.375, "output": 1.875}},
    "gemini-3.1-pro-preview": {"standard": {"input": 2.00, "output": 12.00}, "flex": {"input": 1.00, "output": 6.00}},
    "fake": {"standard": {"input": 0.75, "output": 3.75}, "flex": {"input": 0.375, "output": 1.875}},
}
PRICES_GOOD_UNTIL = date(2026, 12, 31)
FLEX_TRAFFIC = "ON_DEMAND_FLEX"
# Before any reply's cost is known, the cost guard assumes the next reply costs this much at the
# standard price: 300 tokens in and 3,000 out (thinking included), more than these prompts should need.
FIRST_GUESS_TOKENS = {"input": 300, "output": 3000, "thinking": 0}

# Retries for one call: a busy server (429), a timeout or a server error (5xx) is retried after waiting
# 2, 4, 8, 16 then 32 seconds on standard, or 30, 60, 120, 240 then 480 seconds on Flex (Google: for
# Flex, "do not retry aggressively"). Each wait gets up to 1 extra random second. After that the reply
# is saved as "request failed" and the next re-run tries it again. A bad key or bad request (400, 401,
# 403, 404) stops the run at once, because every other call would fail the same way. Google charges
# only for calls that return 200, so failed calls cost nothing.
MAX_TRIES = 6
WAIT_UNIT = {"standard": 1, "flex": 15}
FATAL_HTTP = {400, 401, 403, 404}
MAX_FAILURES_IN_A_ROW = 10

_urlopen = urllib.request.urlopen  # tests replace this with a fake; nothing else does
_sleep = time.sleep


class RequestError(Exception):
    """A call that got no model answer. The message has already been scrubbed of the key."""

    def __init__(self, message: str, status: int | None, retryable: bool):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class StopRun(Exception):
    """Something that would make every remaining call fail (bad key, bad request)."""


def read_key() -> str:
    """Read the API key from the macOS Keychain. Never print or return it anywhere else."""
    user = os.environ.get("USER") or getpass.getuser()
    cmd = ["security", "find-generic-password", "-a", user, "-s", KEYCHAIN_SERVICE, "-w"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        key = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        key = ""
    if not key:
        raise SystemExit(
            f"Couldn't read the API key from the macOS Keychain (service '{KEYCHAIN_SERVICE}', "
            f"account '{user}'). Nothing was sent to Google."
        )
    return key


def scrub(text: str, key: str | None) -> str:
    return text.replace(key, "[key hidden]") if key else text


def endpoint_url(model: str, project: str | None) -> str:
    """The generateContent address on the global endpoint. Without a project, the shorter address
    from Google's API-key quickstart."""
    if project is None:
        return f"{HOST}/v1/publishers/google/models/{model}:generateContent"
    return f"{HOST}/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"


def request_body(prompt: str, schema: dict) -> dict:
    """The settings PREREG 5.2 fixes. There is deliberately no temperature in here."""
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
        },
    }


def make_asker(model: str, project: str | None, tier: str, key: str):
    """Return ask(prompt, schema) -> Google's reply as a dict. The key stays inside this function."""
    url = endpoint_url(model, project)
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    if tier == "flex":
        headers.update(FLEX_HEADERS)

    def ask(prompt: str, schema: dict) -> dict:
        data = json.dumps(request_body(prompt, schema)).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with _urlopen(req, timeout=TIMEOUT_SECONDS[tier]) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RequestError(scrub(f"HTTP {e.code}: {google_error(e)}", key), e.code,
                               e.code not in FATAL_HTTP) from None
        except Exception as e:  # network trouble, timeout, or a reply that isn't JSON
            raise RequestError(scrub(f"{type(e).__name__}: {e}", key), None, True) from None

    return ask


def google_error(e: urllib.error.HTTPError) -> str:
    """Google's own explanation from an error reply, kept short."""
    try:
        err = json.loads(e.read()).get("error", {})
        text = f"{err.get('status', '')} {err.get('message', '')}".strip()
    except Exception:
        text = ""
    return (text or str(e.reason))[:300]


def read_response(resp: dict) -> dict:
    """Pull the reply text, finish reason, traffic type and token counts out of Google's reply."""
    candidates = resp.get("candidates") or []
    first = candidates[0] if candidates else {}
    parts = (first.get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    usage = resp.get("usageMetadata") or {}
    return {
        "reply": "".join(texts) if texts else None,
        "finish_reason": first.get("finishReason") or (resp.get("promptFeedback") or {}).get("blockReason"),
        "model_version": resp.get("modelVersion"),
        "traffic_type": usage.get("trafficType"),
        "tokens": {
            "input": usage.get("promptTokenCount", 0),
            "output": usage.get("candidatesTokenCount", 0),
            "thinking": usage.get("thoughtsTokenCount", 0),
            "total": usage.get("totalTokenCount", 0),
        },
        "usage_raw": usage,
    }


def cost_usd(model: str, tokens: dict, traffic_type: str | None) -> float:
    """Estimated dollars for one reply. Only traffic Google marks as Flex gets the Flex price."""
    price = PRICES[model]["flex" if traffic_type == FLEX_TRAFFIC else "standard"]
    return (tokens["input"] * price["input"] + (tokens["output"] + tokens["thinking"]) * price["output"]) / 1e6


def ask_fake(prompt: str, quiz: bool, tier: str, rng: random.Random) -> dict:
    """A fake reply shaped like Google's, including the kinds of messes real models produce."""
    if quiz:
        text = rng.choice(['{"answer": "rise"}', '{"answer": "fall"}', "I think it rose."])
    else:
        text = rng.choice([
            '{"forecast": [0.12, 0.10, 0.11]}',
            '```json\n{"forecast": [0.2, 0.15, 0.13]}\n```',
            'Sure! {"forecast": ["0.1%", "0.1%", "0.1%"]}',
            '{"forecast": [0.1, 0.1]}',
            "The index will probably stay flat.",
        ])
    tokens_in, tokens_out, thinking = len(prompt) // 4, len(text) // 4 + 1, rng.randint(50, 800)
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": tokens_in, "candidatesTokenCount": tokens_out,
                          "thoughtsTokenCount": thinking, "totalTokenCount": tokens_in + tokens_out + thinking,
                          "trafficType": FLEX_TRAFFIC if tier == "flex" else "ON_DEMAND"},
        "modelVersion": "fake",
    }


def call_with_retries(ask, prompt: str, schema: dict, tier: str) -> tuple[dict | None, str | None, int]:
    """Return (Google's reply, None, tries) or (None, "request failed: ...", tries). Raises StopRun."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return ask(prompt, schema), None, attempt
        except RequestError as e:
            if not e.retryable:
                raise StopRun(str(e)) from None
            if attempt == MAX_TRIES:
                return None, f"request failed: {e}", attempt
            _sleep(2 ** attempt * WAIT_UNIT[tier] + random.uniform(0, 1))
    raise AssertionError("unreachable")


def timed_call(ask, prompt: str, schema: dict, tier: str):
    t0 = time.time()
    resp, error, tries = call_with_retries(ask, prompt, schema, tier)
    return resp, error, tries, time.time() - t0


def out_path(args) -> Path:
    kind = "quiz" if args.quiz else "forecasts"
    base = Path(args.out_dir) if args.out_dir else ROOT / "data" / ("pilot" if args.pilot else "runs")
    return base / args.model / f"{kind}.jsonl"


def load_saved(path: Path) -> tuple[set[tuple[str, int]], float, float]:
    """Return (replies already saved, estimated dollars spent so far, the costliest single reply)."""
    done, spent, costliest = set(), 0.0, 0.0
    if not path.exists():
        return done, spent, costliest
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cost = record.get("cost_usd") or 0.0
            spent += cost
            costliest = max(costliest, cost)
            # A request that failed (timeout, server busy) is not a model answer, so it gets retried.
            if not str(record.get("parse_error") or "").startswith("request failed"):
                done.add((record["prompt_id"], record["reply_index"]))
    return done, spent, costliest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.8-flash", choices=MODELS,
                        help="Gemini model ID, or 'fake' to test the loop without Google")
    parser.add_argument("--series", type=int, default=20)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--quiz", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new prompts (for testing)")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="stop before the estimated cost of this output file passes this many dollars")
    parser.add_argument("--tier", choices=("flex", "standard"), default="flex",
                        help="Flex PayGo (half price, slower) or standard PayGo")
    parser.add_argument("--workers", type=int, default=1, help="calls in flight at once")
    parser.add_argument("--project", default=None, help="Google Cloud project ID (not secret)")
    parser.add_argument("--out-dir", default=None, help="save somewhere else (for testing)")
    args = parser.parse_args(argv)
    if args.model != "fake":
        if args.max_cost is None:
            parser.error("--max-cost is required for a real Gemini run (it spends the parent's credit)")
        if date.today() > PRICES_GOOD_UNTIL:
            parser.error(f"PRICES were checked for calls up to {PRICES_GOOD_UNTIL}; Google's prices change "
                         "after that. Update PRICES from Google's pricing page first.")

    prompts = select_prompts(args)
    path = out_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock file: stops two copies of this command writing the same file at once. If a crash leaves
    # a stale lock, delete the .lock file by hand.
    lock = path.with_suffix(".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Another run is already writing {path.name} (lock file {lock} exists). Wait for it to finish.",
              file=sys.stderr)
        return 1
    os.close(lock_fd)
    try:
        if args.model == "fake":
            rng = random.Random(0)
            ask = lambda prompt, schema: ask_fake(prompt, args.quiz, args.tier, rng)  # noqa: E731
        else:
            ask = make_asker(args.model, args.project, args.tier, read_key())
        return run(args, prompts, path, ask)
    finally:
        lock.unlink(missing_ok=True)


def run(args, prompts, path: Path, ask) -> int:
    done, spent, costliest = load_saved(path)
    jobs = [(row.prompt_id, row.prompt, k)
            for row in prompts.itertuples(index=False)
            for k in range(REPLIES_PER_PROMPT) if (row.prompt_id, k) not in done]
    if args.limit is not None:
        keep = set(list(dict.fromkeys(pid for pid, _, _ in jobs))[: args.limit])
        jobs = [job for job in jobs if job[0] in keep]
    first_guess = cost_usd(args.model, FIRST_GUESS_TOKENS, None)
    print(f"{args.model} ({args.tier}): {len(prompts)} prompts x {REPLIES_PER_PROMPT} replies selected, "
          f"{len(done)} replies already saved, {len(jobs)} calls to go -> {path}")
    print(f"  estimated cost so far ${spent:.4f}" + (f", limit ${args.max_cost:.2f}" if args.max_cost else ""))

    schema = QUIZ_SCHEMA if args.quiz else FORECAST_SCHEMA
    parse = parse_quiz if args.quiz else parse_forecast
    settings = {"thinking_level": THINKING_LEVEL, "temperature": "not set", "json_schema": True,
                "replies_per_prompt": REPLIES_PER_PROMPT, "tier": args.tier, "location": "global",
                "address": "project" if args.project else "no project"}
    calls = failures = parse_failures = in_a_row = 0
    warned_traffic = False
    stop_reason = None
    start = time.time()
    pending = {}
    queue = iter(jobs)
    workers = max(1, args.workers)

    def next_cost() -> float:
        return costliest if costliest > 0 else first_guess

    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            # Fill the free worker slots, unless the run is stopping or the money limit is near.
            while stop_reason is None and len(pending) < workers:
                if args.max_cost is not None and spent + (len(pending) + 1) * next_cost() > args.max_cost:
                    # With calls still in flight, wait for their real costs before deciding to stop.
                    if not pending:
                        stop_reason = (f"the next reply could pass the --max-cost limit of ${args.max_cost:.2f} "
                                       f"(estimated cost so far ${spent:.4f})")
                    break
                job = next(queue, None)
                if job is None:
                    break
                pending[pool.submit(timed_call, ask, job[1], schema, args.tier)] = job
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                prompt_id, _, k = pending.pop(future)
                try:
                    resp, error, tries, seconds = future.result()
                except StopRun as e:
                    stop_reason = stop_reason or (f"Google refused the request ({e}). Nothing more was sent. "
                                                  "Fix the cause first, because the same command would be refused again")
                    continue
                calls += 1
                if resp is not None:
                    got = read_response(resp)
                    parsed, parse_error = parse(got["reply"])
                    cost = cost_usd(args.model, got["tokens"], got["traffic_type"])
                    parse_failures += parsed is None
                    in_a_row = 0
                    if args.tier == "flex" and got["traffic_type"] != FLEX_TRAFFIC and not warned_traffic:
                        warned_traffic = True
                        print(f"  Warning: asked for Flex but Google served traffic type {got['traffic_type']}; "
                              "the cost estimate uses the standard price for those calls.", flush=True)
                else:
                    got = {"reply": None, "finish_reason": None, "model_version": None, "traffic_type": None,
                           "tokens": None, "usage_raw": None}
                    parsed, parse_error, cost = None, error, 0.0
                    failures += 1
                    in_a_row += 1
                    if in_a_row >= MAX_FAILURES_IN_A_ROW:
                        stop_reason = stop_reason or f"{in_a_row} requests in a row failed; try again later"
                spent += cost
                costliest = max(costliest, cost)
                record = {
                    "prompt_id": prompt_id,
                    "reply_index": k,
                    "model": args.model,
                    "reply": got["reply"],
                    "parsed": parsed,
                    "parse_ok": parsed is not None,
                    "parse_error": parse_error,
                    "finish_reason": got["finish_reason"],
                    "model_version": got["model_version"],
                    "traffic_type": got["traffic_type"],
                    "tokens": got["tokens"],
                    "usage_raw": got["usage_raw"],
                    "cost_usd": round(cost, 8),
                    "tries": tries,
                    "seconds": round(seconds, 3),
                    "settings": settings,
                    "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                if calls % 50 == 0 or calls == len(jobs):
                    rate = (time.time() - start) / calls
                    print(f"  {calls}/{len(jobs)} calls, {failures} request failures, {parse_failures} parse "
                          f"failures, ${spent:.4f} so far, {rate:.2f} s per call", flush=True)

    if stop_reason:
        print(f"Stopped early: {stop_reason}. Re-run the same command to carry on; saved replies are kept.")
        return 2
    if not jobs:
        print("Nothing to do: every selected reply is already saved.")
    else:
        print(f"Done: {calls} calls, {failures} request failures, {parse_failures} parse failures, "
              f"estimated cost of this file ${spent:.4f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
