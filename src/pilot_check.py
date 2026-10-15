"""Summarize the pilot runs: failures, noise and speed (PLAN.md, D8). Not the leak test.

On purpose, this does NOT compare forecast shifts with what the real economy did. The pilot exists to
measure noise and failures so PREREG.md can set the series count and thresholds. Looking at the leak
itself before PREREG is committed would let the results shape the plan.

Reads data/pilot/<model>/forecasts.jsonl. Keeps the first saved answer per prompt (the Oct 12 check ran
twice at once). Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "data" / "pilot"
SERIES_FILE = ROOT / "data" / "processed" / "series.csv"
WILD = 10  # noise units (D8)


def load(model_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (model_dir / "forecasts.jsonl").open() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[~df["parse_error"].fillna("").str.startswith("request failed")]
    df = df.drop_duplicates("prompt_id", keep="first")
    parts = df["prompt_id"].str.split("|", expand=True)
    df["series_id"] = parts[0].str[1:].astype(int)
    df["precision"], df["label"] = parts[1], parts[2]
    return df


def summarize(model: str, df: pd.DataFrame, series: pd.DataFrame) -> dict:
    s = series.set_index("series_id")
    ok = df[df["parse_ok"]].copy()
    ok["fc_avg"] = ok["parsed"].apply(np.mean)
    ok["formula_avg"] = ok["series_id"].map(s[["f1", "f2", "f3"]].mean(axis=1))
    ok["noise_sd"] = ok["series_id"].map(s["noise_sd"])
    ok["deviation"] = (ok["fc_avg"] - ok["formula_avg"]) / ok["noise_sd"]
    wild = ok["deviation"].abs() > WILD

    # Does the forecast just repeat numbers from the end of the series?
    def copies(row):
        tail = [s.loc[row.series_id, f"v{t}"] for t in range(21, 25)]
        return any(abs(row.parsed[0] - v) < 1e-9 for v in tail)

    # Shift = real-date forecast minus "month t" forecast, same series (noise units). Spread only.
    kept = ok[~wild]
    neutral = kept[kept["label"] == "month t"].set_index("series_id")["fc_avg"]
    real = kept[(kept["precision"] == "month") & (kept["label"] != "month t")].copy()
    real["shift"] = (real["fc_avg"] - real["series_id"].map(neutral)) / real["noise_sd"]
    per_date = real.groupby("label")["shift"].mean()

    return {
        "model": model,
        "answers": len(df),
        "parse failures": int((~df["parse_ok"]).sum()),
        "wild (>10 units)": int(wild.sum()),
        "median s/prompt": round(df["seconds"].median(), 2),
        "mean deviation from formula": round(ok.loc[~wild, "deviation"].mean(), 2),
        "first value copied from series": f"{ok.apply(copies, axis=1).mean():.0%}",
        "share of shifts exactly 0": f"{(real['shift'].abs() < 1e-9).mean():.0%}",
        "shift sd (single answers)": round(real["shift"].std(), 3),
        "shift sd (date average over 20 series)": round(per_date.std(), 3),
    }


def main() -> None:
    series = pd.read_csv(SERIES_FILE)
    out = [summarize(d.name, load(d), series) for d in sorted(PILOT.iterdir()) if (d / "forecasts.jsonl").exists()]
    table = pd.DataFrame(out).set_index("model").T
    print(table.to_string())


if __name__ == "__main__":
    main()
