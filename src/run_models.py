"""Send the study prompts to a local model through Ollama and save every reply (PLAN.md, D8 and D9).

Settings (D8): temperature 0, thinking off, a JSON schema for the reply, one reply per prompt.
Each reply is appended to a .jsonl file as soon as it arrives. Re-running the same command skips
prompts already saved, so a crash or a sleeping Mac loses nothing.

Usage (Jayden runs these; Claude does not run models):
  python3 src/run_models.py --model gemma4:e2b --pilot          # pilot: 20 dates, month labels, 20 series
  python3 src/run_models.py --model gemma4:e2b                  # full run: all prompts, first 20 series
  python3 src/run_models.py --model gemma4:e2b --series 50      # full run on 50 series (D8, if pilot is noisy)
  python3 src/run_models.py --model gemma4:e2b --quiz           # knowledge quiz (D10)
  python3 src/run_models.py --model fake --pilot                # test the loop with fake replies, no model

Pilot rule (fixed in advance): every 5th of the 99 dates (20 dates), month labels only, plus the
"month t" neutral and the no-date prompt, on the first 20 series. That's 440 prompts. Pilot replies go
to data/pilot/ and are not study data.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from read_reply import parse_forecast, parse_quiz

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OLLAMA_URL = "http://localhost:11434/api/chat"

OPTIONS = {"temperature": 0, "seed": 0}
FORECAST_SCHEMA = {
    "type": "object",
    "properties": {"forecast": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}},
    "required": ["forecast"],
}
QUIZ_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["rise", "fall"]}},
    "required": ["answer"],
}


def select_prompts(args) -> pd.DataFrame:
    if args.quiz:
        return pd.read_csv(PROCESSED / "quiz_prompts.csv")
    prompts = pd.read_csv(PROCESSED / "prompts.csv", keep_default_na=False)
    prompts = prompts[prompts["series_id"] < args.series]
    if args.pilot:
        dates = pd.read_csv(PROCESSED / "label_dates.csv")
        pilot_labels = set(dates["month_text"].iloc[::5])
        keep = (prompts["precision"] == "month") & (prompts["label_text"].isin(pilot_labels) | prompts["is_neutral"])
        keep |= prompts["precision"] == "none"
        prompts = prompts[keep]
    return prompts


def out_path(args) -> Path:
    safe_model = args.model.replace(":", "_").replace("/", "_")
    kind = "quiz" if args.quiz else "forecasts"
    base = Path(args.out_dir) if args.out_dir else ROOT / "data" / ("pilot" if args.pilot else "runs")
    folder = base / safe_model
    return folder / f"{kind}.jsonl"


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # A request that failed (timeout, server busy) is not a model answer, so it gets retried.
            if not str(record.get("parse_error") or "").startswith("request failed"):
                done.add(record["prompt_id"])
    return done


def ask_ollama(model: str, prompt: str, schema: dict) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": schema,
        "options": OPTIONS,
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["message"]["content"]


def ask_fake(prompt: str, quiz: bool, rng: random.Random) -> str:
    """Canned replies for testing the loop, including the kinds of messes real models produce."""
    if quiz:
        return rng.choice(['{"answer": "rise"}', '{"answer": "fall"}', "I think it rose."])
    return rng.choice([
        '{"forecast": [0.12, 0.10, 0.11]}',
        '```json\n{"forecast": [0.2, 0.15, 0.13]}\n```',
        'Sure! {"forecast": ["0.1%", "0.1%", "0.1%"]}',
        '{"forecast": [0.1, 0.1]}',
        "The index will probably stay flat.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name, or 'fake' to test the loop")
    parser.add_argument("--series", type=int, default=20)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--quiz", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new prompts (for testing)")
    parser.add_argument("--out-dir", default=None, help="save somewhere else (for testing)")
    args = parser.parse_args()

    prompts = select_prompts(args)
    path = out_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock file: stops two copies of this command writing the same file at once (it happened in the
    # first pilot check on Oct 12). If a crash leaves a stale lock, delete the .lock file by hand.
    lock = path.with_suffix(".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"Another run is already writing {path.name} (lock file {lock} exists). Wait for it to finish.")
    os.close(lock_fd)
    try:
        run(args, prompts, path)
    finally:
        lock.unlink(missing_ok=True)


def run(args, prompts: pd.DataFrame, path: Path) -> None:
    done = already_done(path)
    todo = prompts[~prompts["prompt_id"].isin(done)]
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{args.model}: {len(prompts)} prompts selected, {len(done)} already saved, {len(todo)} to go -> {path}")

    schema = QUIZ_SCHEMA if args.quiz else FORECAST_SCHEMA
    parse = parse_quiz if args.quiz else parse_forecast
    rng = random.Random(0)
    failures = 0
    start = time.time()

    with path.open("a") as f:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            t0 = time.time()
            try:
                if args.model == "fake":
                    reply, error = ask_fake(row.prompt, args.quiz, rng), None
                else:
                    reply, error = ask_ollama(args.model, row.prompt, schema), None
            except Exception as e:  # network or server error: record it, keep going
                reply, error = None, f"request failed: {e}"
            parsed, parse_error = parse(reply) if error is None else (None, error)
            failures += parsed is None
            record = {
                "prompt_id": row.prompt_id,
                "model": args.model,
                "reply": reply,
                "parsed": parsed,
                "parse_ok": parsed is not None,
                "parse_error": parse_error,
                "seconds": round(time.time() - t0, 3),
                "options": {**OPTIONS, "think": False},
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - start) / i
                print(f"  {i}/{len(todo)} done, {failures} parse failures, {rate:.2f} s per prompt", flush=True)

    if len(todo) == 0:
        print("Nothing to do: every selected prompt is already saved.")


if __name__ == "__main__":
    sys.exit(main())
