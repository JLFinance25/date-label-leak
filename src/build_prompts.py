"""Build every prompt for the study, plus the real-world outcome for each date (PLAN.md, D2 to D5, D10).

Reads data/processed/series.csv (from make_series.py) and data/raw/INDPRO.csv. Writes:
  data/processed/label_dates.csv   the 99 dates, their label texts, and what really happened
  data/processed/prompts.csv       one row per (series, label) forecast prompt
  data/processed/quiz_prompts.csv  the knowledge quiz, one row per date

Prompt wording approved by Jayden on Oct 12, 2026. Written by Claude (AI) at Jayden's request.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SERIES_FILE = ROOT / "data" / "processed" / "series.csv"
RAW_FILE = ROOT / "data" / "raw" / "INDPRO.csv"
OUT_DIR = ROOT / "data" / "processed"

HISTORY = 24
FIRST_LABEL, LAST_LABEL = "1990-01-01", "2022-09-01"
# D7: dates whose past or future 3-month window touches Feb to Jun 2020 are dropped in the COVID check.
COVID_START, COVID_END = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01")

PROMPT = (
    "Below are 24 consecutive monthly values of a US economic activity index, "
    "shown as percent change from the previous month: {values}. "
    "{label_sentence}"
    "Forecast the next 3 monthly values. "
    'Reply only with JSON in this form: {{"forecast": [x, y, z]}}'
)
LABEL_SENTENCE = "The last value is for {label}. "

QUIZ = (
    "In the 3 months after {month}, did US industrial production rise or fall? "
    'Reply only with JSON in this form: {{"answer": "rise"}} or {{"answer": "fall"}}'
)

NEUTRAL = {"month": "month t", "quarter": "quarter t", "year": "year t"}


def label_texts(date: pd.Timestamp) -> dict[str, str]:
    return {
        "month": f"{date:%B %Y}",
        "quarter": f"{date.year} Q{date.quarter}",
        "year": f"{date.year}",
    }


def build_label_dates() -> pd.DataFrame:
    """Every 4th month (Jan, May, Sep) from Jan 1990 to Sep 2022, with the real 3-month changes."""
    raw = pd.read_csv(RAW_FILE, parse_dates=["observation_date"]).set_index("observation_date")["INDPRO"]
    dates = pd.date_range(FIRST_LABEL, LAST_LABEL, freq="4MS")
    rows = []
    for d in dates:
        level_now = raw[d]
        future = (raw[d + pd.DateOffset(months=3)] / level_now - 1) * 100
        past = (level_now / raw[d - pd.DateOffset(months=3)] - 1) * 100
        # Months whose changes are used: past window t-2..t, future window t+1..t+3.
        window_start, window_end = d - pd.DateOffset(months=2), d + pd.DateOffset(months=3)
        covid = window_start <= COVID_END and window_end >= COVID_START
        rows.append({
            "label_date": d.date(),
            **{f"{k}_text": v for k, v in label_texts(d).items()},
            "real_future_3m_pct": round(future, 4),
            "real_past_3m_pct": round(past, 4),
            "covid_window": covid,
        })
    return pd.DataFrame(rows)


def format_values(row: pd.Series) -> str:
    return ", ".join(f"{row[f'v{t + 1}']:.2f}" for t in range(HISTORY))


def build_prompts(series: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    # Distinct label texts per blur level (year texts repeat 3 times, so they are deduplicated).
    labels = []
    for precision in ("month", "quarter", "year"):
        for text in dict.fromkeys(dates[f"{precision}_text"]):
            labels.append((precision, text, False))
        labels.append((precision, NEUTRAL[precision], True))
    labels.append(("none", "", True))

    rows = []
    for _, s in series.iterrows():
        values = format_values(s)
        for precision, text, is_neutral in labels:
            sentence = LABEL_SENTENCE.format(label=text) if text else ""
            rows.append({
                "prompt_id": f"s{int(s['series_id']):02d}|{precision}|{text or 'none'}",
                "series_id": int(s["series_id"]),
                "precision": precision,
                "label_text": text,
                "is_neutral": is_neutral,
                "prompt": PROMPT.format(values=values, label_sentence=sentence),
            })
    return pd.DataFrame(rows)


def build_quiz(dates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt_id": "quiz|" + dates["month_text"],
        "label_date": dates["label_date"],
        "truth": (dates["real_future_3m_pct"] > 0).map({True: "rise", False: "fall"}),
        "prompt": [QUIZ.format(month=m) for m in dates["month_text"]],
    })


def main() -> None:
    series = pd.read_csv(SERIES_FILE)
    dates = build_label_dates()
    prompts = build_prompts(series, dates)
    quiz = build_quiz(dates)

    assert prompts["prompt_id"].is_unique
    dates.to_csv(OUT_DIR / "label_dates.csv", index=False)
    prompts.to_csv(OUT_DIR / "prompts.csv", index=False)
    quiz.to_csv(OUT_DIR / "quiz_prompts.csv", index=False)

    per_series = len(prompts) // len(series)
    print(f"{len(dates)} label dates ({dates['covid_window'].sum()} touch the COVID window: "
          f"{', '.join(dates.loc[dates['covid_window'], 'month_text'])})")
    print(f"{per_series} prompts per series: {prompts.groupby('precision').size().to_dict()} "
          f"(divide by {len(series)} series)")
    print(f"{len(prompts)} prompts for all {len(series)} series; {per_series * 20} for the first 20")
    print(f"{len(quiz)} quiz prompts ({(quiz['truth'] == 'rise').sum()} rise, {(quiz['truth'] == 'fall').sum()} fall)")
    print()
    print("Example (series 0, September 2008):")
    print(prompts.loc[prompts["prompt_id"] == "s00|month|September 2008", "prompt"].iloc[0])
    print()
    print("Example (series 0, no date):")
    print(prompts.loc[prompts["prompt_id"] == "s00|none|none", "prompt"].iloc[0])


if __name__ == "__main__":
    main()
