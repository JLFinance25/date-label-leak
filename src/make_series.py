"""Make the made-up AR(1) series and their correct forecasts (PLAN.md, D1 and D8).

Each series is monthly percent change following
    next = MEAN + phi * (this - MEAN) + noise,   noise ~ Normal(0, noise_sd)
The correct h-month-ahead forecast is
    MEAN + phi**h * (last - MEAN)
which uses only the numbers and phi. No date goes in, so it cannot depend on a date label.

Settings come from the real data (src/real_stats.py on INDPRO, Jan 1990 to Dec 2022 without 2020,
pulled Oct 12, 2026): average +0.135% a month, standard deviation 0.660%.
Phi values 0.2, 0.5, 0.8 were chosen by Jayden on Oct 12 (option B). noise_sd is set so every
series wobbles as much as the real index overall: noise_sd = TARGET_SD * sqrt(1 - phi^2).

We make 50 series with a fixed seed. The study uses the first 20; the local models use up to 50
if the pilot shows high noise (D8). Because the seed is fixed, the first 20 never change.

Writes data/processed/series.csv. Also runs a check: simulate many futures from each series'
last value and confirm their average matches the formula.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "processed" / "series.csv"

SEED = 20261012
N_SERIES = 50
HISTORY = 24        # months shown to the model
HORIZON = 3         # months it forecasts
BURN_IN = 200       # months simulated and thrown away so the start point doesn't matter
DECIMALS = 2        # values are shown to the model rounded to 2 decimals

MEAN = 0.135
TARGET_SD = 0.660
PHIS = (0.2, 0.5, 0.8)


def noise_sd(phi: float) -> float:
    return TARGET_SD * np.sqrt(1 - phi**2)


def simulate(phi: float, n_months: int, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(BURN_IN + n_months)
    x[0] = MEAN
    shocks = rng.normal(0, noise_sd(phi), size=len(x))
    for t in range(1, len(x)):
        x[t] = MEAN + phi * (x[t - 1] - MEAN) + shocks[t]
    return x[BURN_IN:]


def formula_forecast(last: float, phi: float) -> list[float]:
    return [MEAN + phi**h * (last - MEAN) for h in range(1, HORIZON + 1)]


def make_series() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N_SERIES):
        phi = PHIS[i % len(PHIS)]
        values = np.round(simulate(phi, HISTORY, rng), DECIMALS)
        # The forecast starts from the rounded last value, because that is what the model sees.
        forecast = formula_forecast(values[-1], phi)
        row = {"series_id": i, "phi": phi, "mean": MEAN, "noise_sd": round(noise_sd(phi), 4)}
        row.update({f"v{t + 1}": v for t, v in enumerate(values)})
        row.update({f"f{h + 1}": round(f, 4) for h, f in enumerate(forecast)})
        rows.append(row)
    return pd.DataFrame(rows)


def check_formula(df: pd.DataFrame, n_futures: int = 200_000) -> None:
    """Simulate many futures from each last value; their average should match the formula."""
    rng = np.random.default_rng(SEED + 1)
    worst = 0.0
    for _, row in df.iterrows():
        phi, sd = row["phi"], noise_sd(row["phi"])
        x = np.full(n_futures, row[f"v{HISTORY}"])
        for h in range(1, HORIZON + 1):
            x = MEAN + phi * (x - MEAN) + rng.normal(0, sd, size=n_futures)
            simulated_avg = x.mean()
            # Allowed gap: 5 standard errors of the simulated average.
            allowed = 5 * x.std() / np.sqrt(n_futures)
            gap = abs(simulated_avg - row[f"f{h}"])
            worst = max(worst, gap / allowed)
            assert gap < allowed, (
                f"series {row['series_id']} h={h}: simulated {simulated_avg:.4f} vs formula {row[f'f{h}']:.4f}"
            )
    print(f"Formula check passed for {len(df)} series x {HORIZON} months "
          f"(largest gap was {worst:.2f} of the allowed 5 standard errors).")


def main() -> None:
    df = make_series()
    check_formula(df)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    values = df[[f"v{t + 1}" for t in range(HISTORY)]]
    print(f"Wrote {len(df)} series to {OUT_FILE.relative_to(ROOT)}")
    print(f"Across all values: average {values.values.mean():.3f}%, standard deviation {values.values.std():.3f}% "
          f"(real index: {MEAN}%, {TARGET_SD}%)")
    print()
    print("Series 0 as the model will see it:")
    print("  " + ", ".join(f"{v:.2f}" for v in values.iloc[0]))
    print(f"  phi {df.loc[0, 'phi']}, correct forecast: "
          + ", ".join(f"{df.loc[0, f'f{h + 1}']:.3f}" for h in range(HORIZON)))


if __name__ == "__main__":
    main()
