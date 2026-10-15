"""The leak analysis, as PREREG.md section 6 describes it (6.1 to 6.7).

For each model folder (data/runs/<model>/forecasts.jsonl and quiz.jsonl) this:
  6.1  decides which replies count: the first reply per prompt (the first 3 for Gemini), with failed
       requests left out; parse failures (src/read_reply.py) and wild replies (more than 10 noise units
       from the formula) are counted and dropped; a series whose neutral reply was dropped loses its
       real-date replies at that level too, and those are counted. Failure rates are reported per blur
       level and per date, and on the 27 "fall" dates vs the 72 "rise" dates.
  6.2  computes each shift (real-date forecast minus neutral forecast, in noise units), the date shift
       (average over series), and the slope b of date shift on the real future 3-month change, holding
       the real past 3-month change fixed (measure B), plus measure A (future only).
  6.3  gives b a 95% range from 10,000 bootstrap re-draws of the dates (seed 20261012). For the
       headline model at month level: leak = yes if b > 0 and the whole range is above 0.
  6.4  words the headline result with the PREREG table (smallest slope worth caring about: 0.03).
  6.5  runs the shuffle placebo, the COVID check (without Jan and May 2020) and the rank check
       (partial Spearman correlation); robust = both checks' ranges above 0.
  6.6  describes the blur curve (month, quarter, year; year re-draws whole years), the "month t" vs
       no-date gap, and the size curve.
  6.7  scores the knowledge quiz with balanced accuracy and a bootstrap range; plus the exploratory
       slopes on dates the model answered right and wrong.

Where PREREG leaves room, this file does the following (listed for Jayden in WORKLOG.md):
  - Every resampling (bootstrap, placebo, quiz) uses its own fresh generator with seed 20261012
    (numpy default_rng). PREREG fixes that seed only for the headline bootstrap.
  - 95% ranges are numpy percentiles (linear interpolation) of the re-drawn values.
  - The rank check re-ranks inside each bootstrap re-draw; tied values share the average rank.
  - The quiz is scored two ways, with failures dropped and with failures counted wrong. PREREG doesn't
    say which, so both are reported and neither is picked here.
  - The neutral vs no-date gap is reported both signed and as an average size (absolute value).
  - The placebo, COVID and rank checks run at month level for every model. Only the headline model's
    are the pre-registered checks; the others are descriptive.
  - Gemini: a reply saved twice under the same reply_index keeps its first copy, then the first 3
    replies per prompt are kept. A Gemini quiz with one readable reply out of 3 uses that reply.
  - A run with missing replies is refused unless --allow-incomplete is given.

Safety (PREREG 6 and 7): this refuses to read pilot replies at all, refuses data/runs/ unless
--study-data is given and this file, read_reply.py and build_prompts.py are committed in git, and
checks the data files against the SHA-256 fingerprints in PREREG 4.6 before reading any replies.

Usage:
  python3 src/analyze.py --study-data            # the real analysis (only once PREREG says it's time)
  python3 -m pytest tests/test_analyze.py        # the planted-leak tests on fake replies

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from build_prompts import NEUTRAL
from read_reply import parse_forecast, parse_quiz

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "data" / "runs"
PILOT = ROOT / "data" / "pilot"
RESULTS = ROOT / "results"

N_SERIES = 20  # PREREG 4.1: series 0 to 19 for every model
LEVELS = ("month", "quarter", "year")
WILD = 10  # noise units (PREREG 6.1 rule 3)
N_BOOT = 10_000  # PREREG 6.3
SEED = 20261012  # PREREG 6.3
WORTH_CARING = 0.03  # PREREG 6.4
HEADLINE = "gemini-3.8-flash"  # PREREG 5.2, decision (e)
FALLBACK = "gemma4:12b"  # only after the switch is logged in PREREG section 8
# PREREG 6.6 size order. Gemma 4 31B (API) goes between 12B and Flash once its model ID is checked and
# logged in PREREG section 8; until then any model not listed here goes last.
SIZE_ORDER = ("gemma4:e2b", "gemma4:e4b", "gemma4:12b", "gemini-3.8-flash", "gemini-3.1-pro-preview")
CODE_FILES = ("src/analyze.py", "src/read_reply.py", "src/build_prompts.py")

# PREREG 4.6 (taken Oct 12, 2026). If any of these change, it's a deviation.
FINGERPRINTS = {
    "data/raw/INDPRO.csv": "1a916926060d09b970880153da76d36040505666dbe686ae2abda1ff7eba6279",
    "data/processed/series.csv": "bc60991dd61b6a8dd6cae2e608029cad1570013100251cd7233702d31bd7aef8",
    "data/processed/label_dates.csv": "1afad4bb875ed089ee8e52173d23710db92b8f9cb96ed0e899fa6e126663d25f",
    "data/processed/prompts.csv": "d3cca4726a39545c44ec95948610b7d5ff87e2849e6cd287ffdecbd7bc70d065",
    "data/processed/quiz_prompts.csv": "144a43ec4e26a05e814363c4431b3d498f03d10c3da5213b2780342393a470dd",
}

WORDING = {  # PREREG 6.4
    "big": "leak, at least the size worth caring about",
    "unclear": "leak; unclear whether it is big enough to matter",
    "small": "leak, but smaller than worth caring about",
    "none": "no leak worth caring about",
    "cant_tell": "can't tell",
    "away": "the forecast bends away from what happened; reported as unexpected",
}


class IncompleteRun(Exception):
    """A run file is missing replies for some study prompts."""


# ---------- materials ----------

def load_materials(processed: Path = PROCESSED) -> dict:
    series = pd.read_csv(processed / "series.csv")
    dates = pd.read_csv(processed / "label_dates.csv", parse_dates=["label_date"],
                        dtype={"month_text": str, "quarter_text": str, "year_text": str})
    prompts = pd.read_csv(processed / "prompts.csv", keep_default_na=False)
    quiz = pd.read_csv(processed / "quiz_prompts.csv")
    return {
        "series": series[series["series_id"] < N_SERIES].reset_index(drop=True),
        "dates": dates,
        "prompts": prompts.loc[prompts["series_id"] < N_SERIES,
                               ["prompt_id", "series_id", "precision", "label_text", "is_neutral"]].reset_index(drop=True),
        "quiz": quiz[["prompt_id", "label_date", "truth"]],
    }


def check_fingerprints(root: Path = ROOT) -> list[str]:
    """Return the files whose SHA-256 doesn't match PREREG 4.6 (an empty list means all match)."""
    return [name for name, digest in FINGERPRINTS.items()
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]


# ---------- 6.1 which replies count ----------

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}, line {n} isn't valid JSON ({e}). Fix or remove that line first.")
    return records


def counted_replies(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Rule 1: drop failed requests, then keep the first reply per prompt (the first 3 for Gemini)."""
    df = pd.DataFrame(records, columns=None if records else ["prompt_id", "reply", "parse_error"])
    if "parse_error" not in df.columns:
        df["parse_error"] = None
    failed = df["parse_error"].fillna("").astype(str).str.startswith("request failed")
    df = df[~failed]
    k = 3 if "reply_index" in df.columns else 1
    before = len(df)
    if k == 3:
        df = df.drop_duplicates(["prompt_id", "reply_index"], keep="first")
    df = df.groupby("prompt_id", sort=False).head(k).reset_index(drop=True)
    counts = {"records": len(records), "request_failed": int(failed.sum()),
              "duplicates_dropped": before - len(df), "replies_per_prompt": k}
    return df, counts


def split_prompt_ids(ids: pd.Series) -> pd.DataFrame:
    parts = ids.str.split("|", n=2, expand=True)
    return pd.DataFrame({"series_id": parts[0].str[1:].astype(int), "precision": parts[1], "label_text": parts[2]},
                        index=ids.index)


def score_forecast_replies(kept: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Rules 2 to 4, one reply at a time: parse it, then check it against the wild limit."""
    out = pd.concat([kept[["prompt_id"]], split_prompt_ids(kept["prompt_id"])], axis=1)
    parsed = [parse_forecast(r) for r in kept["reply"]]
    out["forecast"] = [p for p, _ in parsed]
    out["parse_reason"] = [reason for _, reason in parsed]
    s = series.set_index("series_id")
    formula_avg = s[["f1", "f2", "f3"]].mean(axis=1)
    out["reply_avg"] = [np.mean(p) if p is not None else np.nan for p in out["forecast"]]
    out["deviation"] = (out["reply_avg"] - out["series_id"].map(formula_avg)) / out["series_id"].map(s["noise_sd"])
    out["status"] = np.select([out["forecast"].isna(), out["deviation"].abs() > WILD],
                              ["parse_failure", "wild"], "ok")
    return out


def prompt_forecasts(scored: pd.DataFrame, prompts: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per study prompt: the forecast is the average of the replies that survived (rule 4)."""
    flags = scored.assign(parse_fail=scored["status"] == "parse_failure", wild=scored["status"] == "wild",
                          good=scored["status"] == "ok")
    g = flags.groupby("prompt_id")
    agg = pd.DataFrame({"n_replies": g.size(), "n_parse_fail": g["parse_fail"].sum(), "n_wild": g["wild"].sum(),
                        "n_good": g["good"].sum()})
    agg["fc_avg"] = flags[flags["good"]].groupby("prompt_id")["reply_avg"].mean()
    table = prompts.set_index("prompt_id").join(agg)
    for col in ("n_replies", "n_parse_fail", "n_wild", "n_good"):
        table[col] = table[col].fillna(0).astype(int)
    table["missing"] = (k - table["n_replies"]).clip(lower=0)
    return table


def level_shifts(table: pd.DataFrame, series: pd.DataFrame, level: str) -> pd.DataFrame:
    """6.2 shift for each (series, real label) at one level, plus why a shift is missing (rule 5)."""
    sd = series.set_index("series_id")["noise_sd"]
    rows = table[table["precision"] == level]
    neutral = rows[rows["is_neutral"]].set_index("series_id")["fc_avg"]
    real = rows[~rows["is_neutral"]].reset_index()[["prompt_id", "series_id", "label_text", "fc_avg"]]
    real["neutral_fc"] = real["series_id"].map(neutral)
    real["shift"] = (real["fc_avg"] - real["neutral_fc"]) / real["series_id"].map(sd)
    real["shift_status"] = np.select([real["fc_avg"].isna(), real["neutral_fc"].isna()],
                                     ["own_dropped", "no_reference"], "ok")
    return real


def date_shifts(shifts: pd.DataFrame, dates: pd.DataFrame, level: str) -> pd.DataFrame:
    """The date shift (average over series with both replies) next to each date's real outcomes.
    At year level the 3 dates in a year share one prompt, so they share one date shift."""
    per_label = shifts.groupby("label_text").agg(date_shift=("shift", "mean"), n_series=("shift", "count"))
    d = pd.DataFrame({
        "label_date": dates["label_date"],
        "label_text": dates[f"{level}_text"],
        "year": dates["label_date"].dt.year,
        "future": dates["real_future_3m_pct"],
        "past": dates["real_past_3m_pct"],
        "covid": dates["covid_window"].astype(bool),
        "truth": np.where(dates["real_future_3m_pct"] > 0, "rise", "fall"),
    })
    d = d.join(per_label, on="label_text")
    d["n_series"] = d["n_series"].fillna(0).astype(int)
    return d


# ---------- 6.2 and 6.3 slopes and the bootstrap ----------

def ols_slopes(Y, F, P=None):
    """Slopes for many fits at once (one fit per row). Returns (b, c); c is None without P."""
    Y, F = np.atleast_2d(Y).astype(float), np.atleast_2d(F).astype(float)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Fc = F - F.mean(axis=1, keepdims=True)
    s11, s1y = (Fc * Fc).sum(axis=1), (Fc * Yc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if P is None:
            return s1y / s11, None
        P = np.atleast_2d(P).astype(float)
        Pc = P - P.mean(axis=1, keepdims=True)
        s22, s12, s2y = (Pc * Pc).sum(axis=1), (Fc * Pc).sum(axis=1), (Pc * Yc).sum(axis=1)
        det = s11 * s22 - s12 ** 2
        return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


def fit_b(y, f, p) -> np.ndarray:
    """Measure B: date shift = a + b x future + c x past. Returns [a, b, c]."""
    b, c = ols_slopes(y, f, p)
    b, c = b[0], c[0]
    return np.array([np.mean(y) - b * np.mean(f) - c * np.mean(p), b, c])


def fit_a(y, f) -> np.ndarray:
    """Measure A: date shift = a + b x future. Returns [a, b]."""
    b = ols_slopes(y, f)[0][0]
    return np.array([np.mean(y) - b * np.mean(f), b])


def bootstrap_indices(clusters, n_boot: int, rng) -> np.ndarray:
    """Re-draw whole clusters with replacement. A cluster is one date, or one year at year level."""
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    if len({len(m) for m in groups}) != 1:
        raise ValueError("every cluster must have the same number of dates")
    members = np.array(groups)
    draws = rng.integers(0, len(members), size=(n_boot, len(members)))
    return members[draws].reshape(n_boot, -1)


def _range(values) -> tuple[float, float]:
    if np.isnan(values).all():  # e.g. every date shift is exactly 0, so no correlation exists
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def slope_with_range(d: pd.DataFrame, measure: str = "B", cluster_col: str | None = None, n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """Slope b (and c for measure B) with its 95% bootstrap range over dates (or whole years)."""
    d = d.dropna(subset=["date_shift"])
    y, f, p = (d[c].to_numpy(float) for c in ("date_shift", "future", "past"))
    clusters = d[cluster_col].to_numpy() if cluster_col else np.arange(len(d))
    idx = bootstrap_indices(clusters, n_boot, np.random.default_rng(seed))
    if measure == "B":
        a, b, c = fit_b(y, f, p)
        boot_b, boot_c = ols_slopes(y[idx], f[idx], p[idx])
    else:
        (a, b), c = fit_a(y, f), None
        boot_b, boot_c = ols_slopes(y[idx], f[idx])
    lo, hi = _range(boot_b)
    out = {"measure": measure, "a": float(a), "b": float(b), "lo": lo, "hi": hi,
           "c": None if c is None else float(c), "n_dates": len(d), "n_clusters": int(len(np.unique(clusters))),
           "n_boot": n_boot, "seed": seed, "bad_draws": int(np.isnan(boot_b).sum())}
    if boot_c is not None:
        out["c_lo"], out["c_hi"] = _range(boot_c)
    return out


def wording_for_range(lo: float, hi: float, threshold: float = WORTH_CARING) -> str:
    """PREREG 6.4's table, read from the 95% range."""
    if hi < 0:
        return WORDING["away"]
    if lo > threshold:
        return WORDING["big"]
    if lo > 0:
        return WORDING["unclear"] if hi >= threshold else WORDING["small"]
    return WORDING["none"] if hi < threshold else WORDING["cant_tell"]


def headline_verdict(fit: dict) -> dict:
    """PREREG 6.3: leak = yes if b is above 0 and the whole 95% range is above 0."""
    return {"leak": bool(fit["b"] > 0 and fit["lo"] > 0), "wording": wording_for_range(fit["lo"], fit["hi"])}


# ---------- 6.5 checks ----------

def shuffle_placebo(d: pd.DataFrame, real_b: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Shuffle which dates the (future, past) pairs go with, keeping each pair together, and refit."""
    d = d.dropna(subset=["date_shift"])
    y, f, p = (d[c].to_numpy(float) for c in ("date_shift", "future", "past"))
    perm = np.random.default_rng(seed).permuted(np.tile(np.arange(len(d)), (n_boot, 1)), axis=1)
    b, _ = ols_slopes(y[None, :], f[perm], p[perm])
    return {"mean": float(np.mean(b)), "median": float(np.median(b)), "sd": float(np.std(b)),
            "share_at_least_real": float(np.mean(b >= real_b)), "real_b": float(real_b), "n_shuffles": n_boot,
            "seed": seed}


def covid_check(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    out = slope_with_range(d[~d["covid"]], "B", n_boot=n_boot, seed=seed)
    out["dropped"] = [f"{t:%Y-%m-%d}" for t in d.loc[d["covid"], "label_date"]]
    return out


def _corr(a, b):
    ac, bc = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):  # a column with no spread has no correlation
        return (ac * bc).sum(axis=-1) / np.sqrt((ac ** 2).sum(axis=-1) * (bc ** 2).sum(axis=-1))


def partial_spearman(y, f, p):
    """Rank all three, take the past-change ranks out of the other two, and correlate what's left."""
    ry, rf, rp = (rankdata(x, axis=-1) for x in (y, f, p))
    r_yf, r_yp, r_fp = _corr(ry, rf), _corr(ry, rp), _corr(rf, rp)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r_yf - r_yp * r_fp) / np.sqrt((1 - r_yp ** 2) * (1 - r_fp ** 2))


def rank_check(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    d = d.dropna(subset=["date_shift"])
    y, f, p = (d[c].to_numpy(float) for c in ("date_shift", "future", "past"))
    idx = bootstrap_indices(np.arange(len(d)), n_boot, np.random.default_rng(seed))
    boot = partial_spearman(y[idx], f[idx], p[idx])
    lo, hi = _range(boot)
    return {"rho": float(partial_spearman(y, f, p)), "lo": lo, "hi": hi, "n_dates": len(d), "n_boot": n_boot,
            "seed": seed, "bad_draws": int(np.isnan(boot).sum())}


# ---------- 6.1 rule 6 failure report and 6.6 neutral vs none ----------

def _rate(part, whole) -> float | None:
    return float(part / whole) if whole else None


def failure_report(table: pd.DataFrame, shifts: dict, dates_by_level: dict, k: int) -> tuple[dict, pd.DataFrame]:
    total_replies = int(table["n_replies"].sum())
    parse, wild = int(table["n_parse_fail"].sum()), int(table["n_wild"].sum())
    report = {
        "replies": total_replies, "parse_failures": parse, "wild": wild,
        "parse_failure_rate": _rate(parse, total_replies), "wild_rate": _rate(wild, total_replies),
        "dropped_rate": _rate(parse + wild, total_replies),
        "prompts_by_good_replies": {str(n): int((table["n_good"] == n).sum()) for n in range(k + 1)},
        "by_level": {}, "rise_vs_fall": {},
    }
    by_date = []
    for level in LEVELS:
        rows = table[table["precision"] == level]
        real, neutral = rows[~rows["is_neutral"]], rows[rows["is_neutral"]]
        no_ref = shifts[level]["shift_status"] == "no_reference"
        report["by_level"][level] = {
            "real": {"replies": int(real["n_replies"].sum()), "parse_failures": int(real["n_parse_fail"].sum()),
                     "wild": int(real["n_wild"].sum()),
                     "parse_failure_rate": _rate(real["n_parse_fail"].sum(), real["n_replies"].sum()),
                     "wild_rate": _rate(real["n_wild"].sum(), real["n_replies"].sum()),
                     "prompts": len(real), "no_reference": int(no_ref.sum()),
                     "no_reference_rate": _rate(no_ref.sum(), len(real))},
            "neutral": {"replies": int(neutral["n_replies"].sum()),
                        "parse_failures": int(neutral["n_parse_fail"].sum()), "wild": int(neutral["n_wild"].sum())},
        }
        per_label = real.groupby("label_text").agg(replies=("n_replies", "sum"), parse_failures=("n_parse_fail", "sum"),
                                                   wild=("n_wild", "sum"), prompts=("n_replies", "size"))
        per_label["no_reference"] = shifts[level].assign(nr=no_ref).groupby("label_text")["nr"].sum()
        d = dates_by_level[level][["label_date", "label_text", "truth"]].join(per_label, on="label_text")
        d.insert(0, "level", level)
        d["label_date"] = d["label_date"].dt.strftime("%Y-%m-%d")
        d["parse_failure_rate"] = d["parse_failures"] / d["replies"]
        d["wild_rate"] = d["wild"] / d["replies"]
        d["no_reference_rate"] = d["no_reference"] / d["prompts"]
        by_date.append(d)
        report["rise_vs_fall"][level] = {
            truth: {"dates": len(g), "parse_failure_rate": float(g["parse_failure_rate"].mean()),
                    "wild_rate": float(g["wild_rate"].mean()), "no_reference_rate": float(g["no_reference_rate"].mean())}
            for truth, g in d.groupby("truth")
        }
    none = table[table["precision"] == "none"]
    report["by_level"]["none"] = {"replies": int(none["n_replies"].sum()),
                                  "parse_failures": int(none["n_parse_fail"].sum()), "wild": int(none["n_wild"].sum())}
    # PREREG 6.1 rule 6 asks for the rise/fall comparison when the failure rate is above 5%. It is
    # always computed above; this flag says whether PREREG requires it to be reported.
    report["rise_fall_check_required"] = bool((report["parse_failure_rate"] or 0) > 0.05)
    return report, pd.concat(by_date, ignore_index=True)


def neutral_vs_none(table: pd.DataFrame, series: pd.DataFrame) -> dict:
    """6.6: the gap between each neutral-label forecast and the no-date forecast, in noise units."""
    sd = series.set_index("series_id")["noise_sd"]
    none = table[table["precision"] == "none"].set_index("series_id")["fc_avg"]
    out = {}
    for level in LEVELS:
        neutral = table[(table["precision"] == level) & table["is_neutral"]].set_index("series_id")["fc_avg"]
        gap = ((neutral - none) / sd).dropna()
        out[level] = {"mean_gap": float(gap.mean()) if len(gap) else None,
                      "mean_abs_gap": float(gap.abs().mean()) if len(gap) else None, "n_series": len(gap)}
    return out


# ---------- 6.7 knowledge quiz ----------

def quiz_answers(kept: pd.DataFrame) -> pd.DataFrame:
    """One answer per quiz prompt: the majority of its readable replies. A tie or no readable reply fails."""
    words = [parse_quiz(r)[0] for r in kept["reply"]]
    rows = []
    for pid, group in pd.Series(words, index=kept["prompt_id"].to_numpy()).groupby(level=0, sort=False):
        readable = [w for w in group if w is not None]
        rise, fall = readable.count("rise"), readable.count("fall")
        winner = "rise" if rise > fall else "fall" if fall > rise else None
        rows.append({"prompt_id": pid, "answer": winner, "readable": len(readable), "replies": len(group)})
    return pd.DataFrame(rows, columns=["prompt_id", "answer", "readable", "replies"]).astype({"answer": object})


def _balanced(correct, rise):
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_rise = (correct * rise).sum(axis=-1) / rise.sum(axis=-1)
        acc_fall = (correct * ~rise).sum(axis=-1) / (~rise).sum(axis=-1)
    return (acc_rise + acc_fall) / 2, acc_rise, acc_fall


def score_quiz(answers: pd.DataFrame, quiz: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    q = quiz.merge(answers, on="prompt_id", how="left")
    q["failed"] = q["answer"].isna()
    q["correct"] = (q["answer"] == q["truth"]) & ~q["failed"]
    out = {"dates": len(q), "failures": int(q["failed"].sum())}
    for mode, use in (("failures_dropped", q[~q["failed"]]), ("failures_wrong", q)):
        correct, rise = use["correct"].to_numpy(float), (use["truth"] == "rise").to_numpy()
        balanced, acc_rise, acc_fall = _balanced(correct, rise)
        idx = bootstrap_indices(np.arange(len(use)), n_boot, np.random.default_rng(seed))
        lo, hi = _range(_balanced(correct[idx], rise[idx])[0])
        out[mode] = {"n_dates": len(use), "plain_accuracy": float(correct.mean()),
                     "balanced_accuracy": float(balanced), "rise_accuracy": float(acc_rise),
                     "fall_accuracy": float(acc_fall), "lo": lo, "hi": hi, "knows": bool(lo > 0.5)}
    out["table"] = q
    return out


def quiz_split_slopes(month: pd.DataFrame, q: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Exploratory (labeled so): month-level slope on dates the quiz got right vs wrong."""
    graded = q[~q["failed"]].assign(label_date=lambda x: pd.to_datetime(x["label_date"]))
    out = {"exploratory": True}
    for name, dates in (("right", graded.loc[graded["correct"], "label_date"]),
                        ("wrong", graded.loc[~graded["correct"], "label_date"])):
        d = month[month["label_date"].isin(set(dates))]
        if d["date_shift"].notna().sum() >= 5:
            out[name] = slope_with_range(d, "B", n_boot=n_boot, seed=seed)
        else:
            out[name] = {"n_dates": int(d["date_shift"].notna().sum()), "b": None, "note": "too few dates to fit"}
    return out


# ---------- one model ----------

def analyze_model(folder: Path, mat: dict, n_boot: int = N_BOOT, seed: int = SEED,
                  allow_incomplete: bool = False) -> dict:
    folder = Path(folder)
    result = {"folder": folder.name, "model": folder.name, "levels": None, "quiz": None, "tables": {}}
    month = None
    forecasts = folder / "forecasts.jsonl"
    if forecasts.exists():
        records = read_jsonl(forecasts)
        result["model"] = records[0].get("model", folder.name) if records else folder.name
        kept, counts = counted_replies(records)
        inside = kept["prompt_id"].isin(set(mat["prompts"]["prompt_id"]))
        counts["outside_study"] = int((~inside).sum())
        kept = kept[inside].reset_index(drop=True)
        k = counts["replies_per_prompt"]
        table = prompt_forecasts(score_forecast_replies(kept, mat["series"]), mat["prompts"], k)
        counts["missing_replies"] = int(table["missing"].sum())
        if counts["missing_replies"] and not allow_incomplete:
            raise IncompleteRun(f"{forecasts}: {counts['missing_replies']} replies are missing. Finish the run first "
                                "(or pass --allow-incomplete to analyse it anyway).")
        shifts = {lv: level_shifts(table, mat["series"], lv) for lv in LEVELS}
        dshift = {lv: date_shifts(shifts[lv], mat["dates"], lv) for lv in LEVELS}
        failures, by_date = failure_report(table, shifts, dshift, k)
        levels = {}
        for lv in LEVELS:
            cluster = "year" if lv == "year" else None
            levels[lv] = {m: slope_with_range(dshift[lv], m, cluster, n_boot, seed) for m in ("B", "A")}
            levels[lv]["dates_without_shift"] = int(dshift[lv]["date_shift"].isna().sum())
        month = dshift["month"]
        checks = {"placebo": shuffle_placebo(month, levels["month"]["B"]["b"], n_boot, seed),
                  "covid": covid_check(month, n_boot, seed), "rank": rank_check(month, n_boot, seed)}
        checks["robust"] = bool(checks["covid"]["lo"] > 0 and checks["rank"]["lo"] > 0)
        result.update({"replies_per_prompt": k, "counts": counts, "failures": failures, "levels": levels,
                       "checks": checks, "neutral_vs_none": neutral_vs_none(table, mat["series"])})
        result["tables"] = {"prompts": table, "shifts": shifts, "date_shifts": dshift, "failures_by_date": by_date}

    quiz_file = folder / "quiz.jsonl"
    if quiz_file.exists():
        records = read_jsonl(quiz_file)
        if result["levels"] is None and records:
            result["model"] = records[0].get("model", folder.name)
        kept, counts = counted_replies(records)
        kept = kept[kept["prompt_id"].isin(set(mat["quiz"]["prompt_id"]))].reset_index(drop=True)
        answers = quiz_answers(kept)
        per_prompt = mat["quiz"][["prompt_id"]].merge(answers, how="left")
        counts["missing_replies"] = int((counts["replies_per_prompt"] - per_prompt["replies"].fillna(0)).clip(lower=0).sum())
        if counts["missing_replies"] and not allow_incomplete:
            raise IncompleteRun(f"{quiz_file}: {counts['missing_replies']} replies are missing. Finish the run first "
                                "(or pass --allow-incomplete to analyse it anyway).")
        score = score_quiz(answers, mat["quiz"], n_boot, seed)
        q = score.pop("table")
        score["counts"] = counts
        if month is not None:
            score["exploratory_slopes"] = quiz_split_slopes(month, q, n_boot, seed)
        result["quiz"] = score
        result["tables"]["quiz"] = q
    return result


# ---------- all models, safety and output ----------

def size_order(models) -> list[str]:
    known = [m for m in SIZE_ORDER if m in models]
    return known + [m for m in models if m not in SIZE_ORDER]


def _inside(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


def uncommitted_files() -> list[str]:
    """The analysis code files with changes git hasn't committed (untracked counts as uncommitted)."""
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *CODE_FILES],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return ["(git status failed)"]
    return [line[3:] for line in out.stdout.splitlines()]


def git_head() -> str | None:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def check_safe_to_run(folders: list[Path], study_data: bool) -> None:
    if any(_inside(f, PILOT) for f in folders):
        raise SystemExit("Refusing to analyse pilot replies: PREREG 6 and 7 say pilot replies are never compared "
                         "with real outcomes.")
    if any(_inside(f, RUNS) for f in folders):
        if not study_data:
            raise SystemExit(f"{RUNS} holds study replies. Analysing them compares model shifts with real outcomes, "
                             "which PREREG 6 allows only once the headline model and all its settings are fixed in "
                             "PREREG.md. When that's true, re-run with --study-data.")
        dirty = uncommitted_files()
        if dirty:
            raise SystemExit("PREREG 6: the analysis code must be committed before it touches study data. "
                             f"Commit these first: {', '.join(dirty)}")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    return obj


def headline_section(results: dict, headline: str) -> dict:
    r = results.get(headline)
    if r is None or r["levels"] is None:
        return {"model": headline, "available": False,
                "note": "No forecasts for the headline model in this folder, so the headline test wasn't run."}
    fit = r["levels"]["month"]["B"]
    verdict = headline_verdict(fit)
    return {"model": headline, "available": True, "b": fit["b"], "lo": fit["lo"], "hi": fit["hi"],
            "n_dates": fit["n_dates"], "leak": verdict["leak"], "wording": verdict["wording"],
            "robust": r["checks"]["robust"], "covid": r["checks"]["covid"], "rank": r["checks"]["rank"],
            "placebo": r["checks"]["placebo"]}


def _fmt(fit: dict | None) -> str:
    if not fit or fit.get("b") is None:
        return "n/a"
    return f"{fit['b']:+.4f} [{fit['lo']:+.4f}, {fit['hi']:+.4f}]"


def report_text(summary: dict) -> str:
    h = summary["headline"]
    lines = ["# Leak analysis (PREREG.md section 6)", "",
             f"Code commit: {summary['code_commit']}. Bootstrap: {summary['n_boot']} draws, seed {summary['seed']}.",
             "Slopes are noise units of forecast shift per percentage point of real 3-month change, with 95% ranges.",
             "", "## Headline test (6.3)", ""]
    if h["available"]:
        lines += [f"Model {h['model']}, month labels, measure B, {h['n_dates']} dates: b = {_fmt(h)}.",
                  f"Leak: **{'yes' if h['leak'] else 'no'}**. Wording (6.4): {h['wording']}.",
                  f"Robust (6.5): {'yes' if h['robust'] else 'no'}. COVID check: {_fmt(h['covid'])}; "
                  f"rank check: rho = {h['rank']['rho']:+.3f} [{h['rank']['lo']:+.3f}, {h['rank']['hi']:+.3f}]; "
                  f"placebo mean {h['placebo']['mean']:+.4f}, share at least as large {h['placebo']['share_at_least_real']:.4f}."]
    else:
        lines.append(h["note"])
    lines += ["", "## Size curve (6.6, descriptive)", "", "| Model | month b [95% range] |", "|---|---|"]
    lines += [f"| {row['model']} | {_fmt(row)} |" for row in summary["size_curve"]]
    lines += ["", "## Each model", ""]
    for name, r in summary["models"].items():
        lines.append(f"### {name}")
        if r["levels"] is not None:
            f = r["failures"]
            lines += [f"Replies {f['replies']}: parse failures {f['parse_failures']} ({f['parse_failure_rate']:.2%}), "
                      f"wild {f['wild']} ({f['wild_rate']:.2%}). Prompts by good replies: {f['prompts_by_good_replies']}.",
                      "", "| Level | B: b [range] | A: b [range] | no reference |", "|---|---|---|---|"]
            for lv in LEVELS:
                lines.append(f"| {lv} | {_fmt(r['levels'][lv]['B'])} | {_fmt(r['levels'][lv]['A'])} | "
                             f"{f['by_level'][lv]['real']['no_reference']} |")
            gap = r["neutral_vs_none"]["month"]
            lines.append("")
            lines.append(f"'month t' minus no date: mean {gap['mean_gap']:+.4f}, mean size {gap['mean_abs_gap']:.4f} "
                         "noise units." if gap["n_series"] else "'month t' minus no date: n/a.")
            c = r["checks"]
            lines.append(f"Checks (descriptive unless this is the headline model): COVID {_fmt(c['covid'])}; rank rho "
                         f"{c['rank']['rho']:+.3f} [{c['rank']['lo']:+.3f}, {c['rank']['hi']:+.3f}]; placebo mean "
                         f"{c['placebo']['mean']:+.4f}.")
        if r["quiz"] is not None:
            for mode in ("failures_dropped", "failures_wrong"):
                s = r["quiz"][mode]
                lines.append(f"Quiz ({mode.replace('_', ' ')}): balanced {s['balanced_accuracy']:.3f} "
                             f"[{s['lo']:.3f}, {s['hi']:.3f}], plain {s['plain_accuracy']:.3f}, knows: {s['knows']}.")
        lines.append("")
    return "\n".join(lines)


def write_outputs(results: dict, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    (out_dir / "report.md").write_text(report_text(summary))
    for name, r in results.items():
        safe = name.replace(":", "_").replace("/", "_")
        if "failures_by_date" in r["tables"]:
            r["tables"]["failures_by_date"].to_csv(out_dir / f"failures_by_date_{safe}.csv", index=False)
            pd.concat([d.assign(level=lv) for lv, d in r["tables"]["date_shifts"].items()]).to_csv(
                out_dir / f"date_shifts_{safe}.csv", index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=str(RUNS), help="folder with one subfolder per model")
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap draws (PREREG: 10,000)")
    parser.add_argument("--headline", choices=(HEADLINE, FALLBACK), default=HEADLINE,
                        help=f"use {FALLBACK} only after the switch is logged in PREREG section 8")
    parser.add_argument("--study-data", action="store_true", help="allow reading data/runs/ (study replies)")
    parser.add_argument("--allow-incomplete", action="store_true", help="analyse runs with missing replies")
    args = parser.parse_args(argv)

    runs = Path(args.runs_dir)
    folders = [runs] + ([d for d in sorted(runs.iterdir()) if d.is_dir()] if runs.is_dir() else [])
    check_safe_to_run(folders, args.study_data)
    bad = check_fingerprints()
    if bad:
        raise SystemExit(f"These files don't match their PREREG 4.6 fingerprints: {', '.join(bad)}. That's a deviation.")
    models = [d for d in folders[1:] if (d / "forecasts.jsonl").exists() or (d / "quiz.jsonl").exists()]
    if not models:
        raise SystemExit(f"No model folders with forecasts.jsonl or quiz.jsonl in {runs}.")

    mat = load_materials()
    results = {}
    for folder in models:
        try:
            r = analyze_model(folder, mat, args.n_boot, SEED, args.allow_incomplete)
        except IncompleteRun as e:
            raise SystemExit(str(e))
        results[r["model"]] = r
    order = size_order(list(results))
    summary = {
        "code_commit": git_head(), "n_boot": args.n_boot, "seed": SEED, "worth_caring": WORTH_CARING,
        "headline": headline_section(results, args.headline),
        "size_curve": [{"model": m, **results[m]["levels"]["month"]["B"]} for m in order if results[m]["levels"]],
        "models": {m: {k: v for k, v in results[m].items() if k != "tables"} for m in order},
    }
    write_outputs(results, summary, Path(args.out_dir))
    print(report_text(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
