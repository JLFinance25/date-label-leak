"""Describe real US Industrial Production, so the made-up series can match it (PLAN.md, D1).

Reads data/raw/INDPRO.csv (never edits it) and prints the average monthly percent change,
its standard deviation, and the biggest drop, for Jan 1990 to Dec 2022 with and without 2020.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "INDPRO.csv"

START, END = "1990-01-01", "2022-12-01"


def monthly_pct_change() -> pd.Series:
    """Monthly percent change of the index, indexed by month."""
    df = pd.read_csv(RAW_FILE, parse_dates=["observation_date"])
    level = df.set_index("observation_date")["INDPRO"]
    return level.pct_change() * 100


def describe(changes: pd.Series, label: str) -> None:
    print(f"{label}: {len(changes)} months")
    print(f"  average monthly change: {changes.mean():.3f}%")
    print(f"  standard deviation:     {changes.std():.3f}%")
    print(f"  first-order autocorrelation (how much one month carries into the next): {changes.autocorr():.3f}")


def main() -> None:
    changes = monthly_pct_change().loc[START:END]
    describe(changes, "Jan 1990 to Dec 2022")

    no_2020 = changes[changes.index.year != 2020]
    print()
    describe(no_2020, "Same window without 2020")

    worst = changes.idxmin()
    typical = no_2020.abs().median()
    print()
    print(f"Biggest drop: {changes.min():.2f}% in {worst:%B %Y}")
    print(f"Typical month size (median absolute change, no 2020): {typical:.2f}%")
    print(f"So the biggest drop is about {abs(changes.min()) / typical:.0f} times a typical month.")


if __name__ == "__main__":
    main()
