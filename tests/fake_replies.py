"""Make fake reply files with a planted leak of known size, for testing src/analyze.py (PREREG 6).

None of this is a model reply. Each fake forecast starts from the formula forecast for its series,
plus a random per-series offset (so the neutral forecast isn't exactly right), plus a planted shift
for real-date labels, in noise units:

    shift = leak x real future change + past_effect x real past change + random noise

At the year level, the 3 dates in a year share one prompt, so the planted shift uses the year's
average future and past change. The files have the same fields run_models.py and run_gemini.py save.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_prompts import NEUTRAL  # noqa: E402
from read_reply import parse_forecast, parse_quiz  # noqa: E402

N_SERIES = 20


def forecast_text(values) -> str:
    return json.dumps({"forecast": [round(float(v), 6) for v in values]})


def planted_forecasts(mat, leak=0.0, past_effect=0.0, noise=0.2, seed=0, quarter_leak=None, year_leak=None,
                      none_offset=0.0, covid_only_leak=0.0, base_spread=0.5) -> dict:
    """Return {prompt_id: forecast average} for every study prompt (series 0 to 19)."""
    rng = np.random.default_rng(seed)
    series = mat["series"].set_index("series_id")
    dates = mat["dates"]
    quarter_leak = leak if quarter_leak is None else quarter_leak
    year_leak = leak if year_leak is None else year_leak
    fc = {}
    for s in range(N_SERIES):
        row = series.loc[s]
        sd = row["noise_sd"]
        base = row[["f1", "f2", "f3"]].mean() + rng.normal(0, base_spread) * sd
        for level in ("month", "quarter", "year"):
            fc[f"s{s:02d}|{level}|{NEUTRAL[level]}"] = base
        fc[f"s{s:02d}|none|none"] = base + none_offset * sd
        for level, level_leak in (("month", leak), ("quarter", quarter_leak)):
            for d in dates.itertuples():
                shift = level_leak * d.real_future_3m_pct + past_effect * d.real_past_3m_pct + rng.normal(0, noise)
                if d.covid_window:
                    shift += covid_only_leak * d.real_future_3m_pct
                fc[f"s{s:02d}|{level}|{getattr(d, level + '_text')}"] = base + shift * sd
        for year, grp in dates.groupby("year_text"):
            shift = (year_leak * grp["real_future_3m_pct"].mean() + past_effect * grp["real_past_3m_pct"].mean()
                     + rng.normal(0, noise))
            fc[f"s{s:02d}|year|{year}"] = base + shift * sd
    return fc


def write_forecasts(path: Path, model: str, mat, fc: dict, replies=1, reply_noise=0.0, seed=0, overrides=None,
                    skip=(), extra_records=()) -> None:
    """Write a forecasts.jsonl. With replies=3, each record gets a reply_index like run_gemini.py.

    overrides: {prompt_id: [reply text, ...]} replaces the generated replies for that prompt.
    skip: prompt_ids left out (an unfinished run). extra_records: raw records appended at the end.
    """
    rng = np.random.default_rng(seed + 1000)
    series = mat["series"].set_index("series_id")
    overrides = overrides or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid in mat["prompts"]["prompt_id"]:
            if pid in skip:
                continue
            row = series.loc[int(pid[1:3])]
            formula = [row["f1"], row["f2"], row["f3"]]
            if pid in overrides:
                texts = overrides[pid]
            else:
                texts = []
                for _ in range(replies):
                    target = fc[pid] + (rng.normal(0, reply_noise) * row["noise_sd"] if reply_noise else 0.0)
                    delta = target - np.mean(formula)
                    texts.append(forecast_text([v + delta for v in formula]))
            for k, text in enumerate(texts):
                f.write(json.dumps(forecast_record(pid, model, text, k if replies > 1 else None)) + "\n")
        for record in extra_records:
            f.write(json.dumps(record) + "\n")


def forecast_record(pid, model, text, reply_index=None, error=None) -> dict:
    parsed, parse_error = (None, error) if error else parse_forecast(text)
    record = {"prompt_id": pid, "model": model, "reply": text, "parsed": parsed,
              "parse_ok": parsed is not None, "parse_error": parse_error}
    if reply_index is not None:
        record["reply_index"] = reply_index
    return record


def quiz_record(pid, model, text, reply_index=None, error=None) -> dict:
    parsed, parse_error = (None, error) if error else parse_quiz(text)
    record = {"prompt_id": pid, "model": model, "reply": text, "parsed": parsed,
              "parse_ok": parsed is not None, "parse_error": parse_error}
    if reply_index is not None:
        record["reply_index"] = reply_index
    return record


def write_quiz(path: Path, model: str, answers: dict, replies=1) -> None:
    """answers: {quiz prompt_id: [reply text, ...]} (one text per reply)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid, texts in answers.items():
            for k, text in enumerate(texts):
                f.write(json.dumps(quiz_record(pid, model, text, k if replies > 1 else None)) + "\n")


def answer(word: str) -> str:
    return json.dumps({"answer": word})
