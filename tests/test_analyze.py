"""Tests for src/analyze.py (PREREG.md section 6), using fake replies only.

PREREG 6 requires the analysis code to be checked on made-up replies with a planted leak of known size
before it touches study data: it must recover the planted slope and find about 0 when nothing is
planted. Nothing here reads data/runs/ or data/pilot/. The fake replies are built from the real
outcomes in data/processed/label_dates.csv on purpose, so the planted leak has the real outcomes'
shape (COVID dates included). That is not peeking: no model made these replies.

Run: python3 -m pytest tests
Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze  # noqa: E402
from fake_replies import (answer, forecast_record, forecast_text, planted_forecasts, quiz_record,  # noqa: E402
                          write_forecasts, write_quiz)


@pytest.fixture(scope="module")
def mat():
    return analyze.load_materials()


def run_model(tmp_path, mat, fc, model="gemma4:12b", n_boot=2000, allow_incomplete=False, quiz=None, **write_kw):
    folder = tmp_path / model.replace(":", "_")
    write_forecasts(folder / "forecasts.jsonl", model, mat, fc, **write_kw)
    if quiz is not None:
        write_quiz(folder / "quiz.jsonl", model, quiz, replies=write_kw.get("replies", 1))
    return analyze.analyze_model(folder, mat, n_boot=n_boot, seed=analyze.SEED, allow_incomplete=allow_incomplete)


def formula(mat, s):
    row = mat["series"].set_index("series_id").loc[s]
    return [row["f1"], row["f2"], row["f3"]], row["noise_sd"]


def shifted(mat, s, units):
    """A reply whose average is `units` noise units from series s's formula average."""
    f, sd = formula(mat, s)
    return forecast_text([v + units * sd for v in f])


def truth_answers(mat, flip=()):
    """Quiz replies that are right on every date except the dates in `flip`."""
    out = {}
    for row in mat["quiz"].itertuples():
        word = row.truth if row.prompt_id not in flip else ("fall" if row.truth == "rise" else "rise")
        out[row.prompt_id] = [answer(word)]
    return out


# ---------- materials ----------

def test_study_files_match_prereg_fingerprints():
    assert analyze.check_fingerprints() == []


def test_materials_are_the_4700_study_prompts(mat):
    assert len(mat["prompts"]) == 4700
    assert mat["prompts"]["series_id"].max() == 19
    assert len(mat["dates"]) == 99 and len(mat["quiz"]) == 99
    assert (mat["quiz"]["truth"] == "rise").sum() == 72


# ---------- 6.1 which replies count ----------

def test_first_reply_per_prompt_is_kept():
    pid = "s00|month|month t"
    records = [forecast_record(pid, "m", forecast_text([1, 2, 3])), forecast_record(pid, "m", forecast_text([4, 5, 6]))]
    kept, counts = analyze.counted_replies(records)
    assert list(kept["reply"]) == [forecast_text([1, 2, 3])]
    assert counts["duplicates_dropped"] == 1 and counts["replies_per_prompt"] == 1


def test_request_failures_are_not_answers():
    pid = "s00|month|month t"
    records = [forecast_record(pid, "m", None, error="request failed: timed out"),
               forecast_record(pid, "m", forecast_text([1, 2, 3]))]
    kept, counts = analyze.counted_replies(records)
    assert list(kept["reply"]) == [forecast_text([1, 2, 3])]
    assert counts["request_failed"] == 1 and counts["duplicates_dropped"] == 0


def test_gemini_keeps_the_first_three_replies():
    pid = "s00|month|month t"
    records = [forecast_record(pid, "g", forecast_text([k, k, k]), k) for k in (2, 0, 1)]
    records.append(forecast_record(pid, "g", forecast_text([9, 9, 9]), 0))  # saved twice
    kept, counts = analyze.counted_replies(records)
    assert counts["replies_per_prompt"] == 3
    assert sorted(kept["reply_index"]) == [0, 1, 2]
    assert forecast_text([9, 9, 9]) not in set(kept["reply"])
    assert counts["duplicates_dropped"] == 1


def test_parse_failures_and_wild_replies_are_counted_and_dropped(mat):
    pids = ["s00|month|month t", "s00|month|January 1990", "s00|month|May 1990", "s00|month|September 1990",
            "s00|month|January 1991"]
    texts = ["The index will rise.", shifted(mat, 0, 10.5), shifted(mat, 0, -10.5), shifted(mat, 0, 9.5),
             '{"forecast": [1, 2]}']
    kept, _ = analyze.counted_replies([forecast_record(p, "m", t) for p, t in zip(pids, texts)])
    scored = analyze.score_forecast_replies(kept, mat["series"]).set_index("prompt_id")
    assert list(scored["status"]) == ["parse_failure", "wild", "wild", "ok", "parse_failure"]
    assert scored.loc["s00|month|September 1990", "deviation"] == pytest.approx(9.5, abs=1e-4)


def test_gemini_forecast_is_the_average_of_surviving_replies(tmp_path, mat):
    fc = planted_forecasts(mat, noise=0.0)
    f, sd = formula(mat, 0)
    overrides = {
        "s00|month|month t": ["garbage", shifted(mat, 0, 1.0), shifted(mat, 0, 2.0)],
        "s00|quarter|quarter t": ["garbage", "more garbage", shifted(mat, 0, 50.0)],  # none survive
    }
    res = run_model(tmp_path, mat, fc, model="gemini-3.8-flash", replies=3, overrides=overrides)
    table = res["tables"]["prompts"]
    assert table.loc["s00|month|month t", "fc_avg"] == pytest.approx(np.mean(f) + 1.5 * sd, abs=1e-5)
    assert table.loc["s00|month|month t", "n_good"] == 2
    assert np.isnan(table.loc["s00|quarter|quarter t", "fc_avg"])
    assert res["failures"]["prompts_by_good_replies"] == {"0": 1, "1": 0, "2": 1, "3": 4698}
    assert res["failures"]["parse_failures"] == 3 and res["failures"]["wild"] == 1


def test_dropped_neutral_drops_that_series_at_that_level(tmp_path, mat):
    fc = planted_forecasts(mat, leak=0.1)
    res = run_model(tmp_path, mat, fc, overrides={"s03|month|month t": ["no idea"]})
    by_level = res["failures"]["by_level"]
    assert by_level["month"]["real"]["no_reference"] == 99
    assert by_level["month"]["neutral"]["parse_failures"] == 1
    assert by_level["quarter"]["real"]["no_reference"] == 0
    month = res["tables"]["date_shifts"]["month"]
    assert (month["n_series"] == 19).all()


def test_failure_rates_per_level_and_per_date(tmp_path, mat):
    fc = planted_forecasts(mat)
    overrides = {f"s{s:02d}|month|September 2008": ["not json"] for s in range(5)}
    overrides["s07|year|2008"] = ["not json"]
    res = run_model(tmp_path, mat, fc, overrides=overrides)
    month = res["failures"]["by_level"]["month"]["real"]
    assert month["parse_failures"] == 5 and month["replies"] == 20 * 99
    assert month["parse_failure_rate"] == pytest.approx(5 / (20 * 99))
    by_date = res["tables"]["failures_by_date"].set_index(["level", "label_date"])
    assert by_date.loc[("month", "2008-09-01"), "parse_failure_rate"] == pytest.approx(0.25)
    assert by_date.loc[("month", "2008-05-01"), "parse_failure_rate"] == 0
    # The 3 dates of 2008 share one year prompt, so each shows its failure.
    for d in ("2008-01-01", "2008-05-01", "2008-09-01"):
        assert by_date.loc[("year", d), "parse_failure_rate"] == pytest.approx(1 / 20)


def test_failures_compared_on_rise_and_fall_dates(tmp_path, mat):
    fc = planted_forecasts(mat)
    fall = mat["dates"][mat["dates"]["real_future_3m_pct"] <= 0]
    overrides = {f"s00|month|{t}": ["not json"] for t in fall["month_text"]}
    res = run_model(tmp_path, mat, fc, overrides=overrides)
    month = res["failures"]["rise_vs_fall"]["month"]
    assert month["fall"]["parse_failure_rate"] == pytest.approx(1 / 20)
    assert month["rise"]["parse_failure_rate"] == 0
    assert month["fall"]["dates"] == 27 and month["rise"]["dates"] == 72


def test_unfinished_run_is_refused_unless_allowed(tmp_path, mat):
    fc = planted_forecasts(mat)
    with pytest.raises(analyze.IncompleteRun):
        run_model(tmp_path, mat, fc, skip={"s00|month|May 1990"})
    res = run_model(tmp_path, mat, fc, skip={"s00|month|May 1990"}, allow_incomplete=True)
    assert res["counts"]["missing_replies"] == 1


def test_replies_outside_the_study_series_are_ignored(tmp_path, mat):
    fc = planted_forecasts(mat)
    extra = [forecast_record("s25|month|month t", "m", forecast_text([0, 0, 0]))]
    res = run_model(tmp_path, mat, fc, extra_records=extra)
    assert res["counts"]["outside_study"] == 1
    assert res["failures"]["replies"] == 4700


# ---------- 6.2 shift and slope ----------

def test_slope_fits_match_numpy_least_squares():
    rng = np.random.default_rng(1)
    f, p = rng.normal(size=99), rng.normal(size=99)
    y = 0.3 + 0.2 * f - 0.1 * p + rng.normal(size=99)
    ones = np.ones(99)
    assert analyze.fit_b(y, f, p) == pytest.approx(np.linalg.lstsq(np.column_stack([ones, f, p]), y)[0])
    assert analyze.fit_a(y, f) == pytest.approx(np.linalg.lstsq(np.column_stack([ones, f]), y)[0])


def test_shift_is_measured_in_noise_units(tmp_path, mat):
    fc = planted_forecasts(mat, noise=0.0)
    neutral = fc["s00|month|month t"]
    f, sd = formula(mat, 0)
    text = forecast_text([v - np.mean(f) + neutral + 0.5 * sd for v in f])
    res = run_model(tmp_path, mat, fc, overrides={"s00|month|September 2008": [text]})
    shifts = res["tables"]["shifts"]["month"].set_index(["series_id", "label_text"])
    assert shifts.loc[(0, "September 2008"), "shift"] == pytest.approx(0.5, abs=1e-4)
    month = res["tables"]["date_shifts"]["month"].set_index("label_date")
    assert month.loc[pd.Timestamp("2008-09-01"), "date_shift"] == pytest.approx(0.5 / 20, abs=1e-5)
    assert month.loc[pd.Timestamp("2008-05-01"), "date_shift"] == pytest.approx(0, abs=1e-5)


def test_year_dates_share_one_shift_but_keep_their_own_outcomes(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.1))
    year = res["tables"]["date_shifts"]["year"]
    assert len(year) == 99
    assert (year.groupby("year")["date_shift"].nunique() == 1).all()
    assert year.groupby("year")["future"].nunique().max() == 3


def test_planted_leak_is_recovered_within_its_range(tmp_path, mat):
    fc = planted_forecasts(mat, leak=0.10, past_effect=0.05, noise=0.2, seed=1)
    res = run_model(tmp_path, mat, fc, n_boot=10_000)
    fit = res["levels"]["month"]["B"]
    assert fit["n_dates"] == 99 and fit["n_boot"] == 10_000
    assert fit["lo"] < 0.10 < fit["hi"]
    assert fit["lo"] > 0
    assert fit["b"] == pytest.approx(0.10, abs=0.01)
    assert fit["c"] == pytest.approx(0.05, abs=0.01)


def test_planted_leak_is_recovered_from_three_noisy_gemini_replies(tmp_path, mat):
    fc = planted_forecasts(mat, leak=0.10, noise=0.2, seed=2)
    res = run_model(tmp_path, mat, fc, model="gemini-3.8-flash", replies=3, reply_noise=0.3, n_boot=10_000)
    fit = res["levels"]["month"]["B"]
    assert fit["lo"] < 0.10 < fit["hi"] and fit["lo"] > 0


def test_nothing_planted_gives_about_zero(tmp_path, mat):
    # With no leak, a 95% range still misses 0 about 1 time in 20 by chance. Seed 3 is such a case
    # (b = 0.0025, range just above 0), so this checks b is about 0 and far below 0.03; the next test
    # checks how often the range misses 0 over many no-leak datasets.
    for seed in (3, 15, 16):
        res = run_model(tmp_path, mat, planted_forecasts(mat, noise=0.2, seed=seed), n_boot=10_000)
        fit = res["levels"]["month"]["B"]
        assert abs(fit["b"]) < 0.01
        assert fit["hi"] < analyze.WORTH_CARING
        assert analyze.wording_for_range(fit["lo"], fit["hi"]) != analyze.WORDING["big"]


def test_false_alarm_rate_with_nothing_planted_is_about_5_percent(mat):
    d = mat["dates"]
    f, p = d["real_future_3m_pct"].to_numpy(), d["real_past_3m_pct"].to_numpy()
    rng = np.random.default_rng(123)
    reps, misses, false_leaks = 400, 0, 0
    for r in range(reps):
        y = rng.normal(0, 0.2 / np.sqrt(20), size=99)  # date shifts from 20 series of pure noise
        fit = analyze.slope_with_range(pd.DataFrame({"date_shift": y, "future": f, "past": p}), "B",
                                       n_boot=2000, seed=r)
        misses += not (fit["lo"] <= 0 <= fit["hi"])
        false_leaks += analyze.headline_verdict(fit)["leak"]
    assert 0.02 <= misses / reps <= 0.09
    assert false_leaks / reps <= 0.05


def test_reacting_to_the_past_is_not_counted_as_a_leak(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, past_effect=0.10, noise=0.2, seed=4))
    fit_b, fit_a = res["levels"]["month"]["B"], res["levels"]["month"]["A"]
    assert fit_b["lo"] < 0 < fit_b["hi"]
    assert fit_b["c"] == pytest.approx(0.10, abs=0.01)
    # Measure A leaves the past out, so it picks up 0.10 x (slope of past on future).
    d = mat["dates"]
    past_on_future = np.cov(d["real_future_3m_pct"], d["real_past_3m_pct"])[0, 1] / d["real_future_3m_pct"].var()
    assert fit_a["b"] == pytest.approx(0.10 * past_on_future, abs=0.01)
    assert fit_a["c"] is None


# ---------- 6.3 and 6.4 bootstrap and headline ----------

def test_bootstrap_is_reproducible_with_its_seed(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.05, seed=5))
    d = res["tables"]["date_shifts"]["month"]
    one = analyze.slope_with_range(d, "B", n_boot=3000, seed=analyze.SEED)
    two = analyze.slope_with_range(d, "B", n_boot=3000, seed=analyze.SEED)
    other = analyze.slope_with_range(d, "B", n_boot=3000, seed=1)
    assert (one["lo"], one["hi"]) == (two["lo"], two["hi"])
    assert (one["lo"], one["hi"]) != (other["lo"], other["hi"])
    assert analyze.N_BOOT == 10_000 and analyze.SEED == 20261012


def test_year_bootstrap_redraws_whole_years():
    years = np.repeat(np.arange(1990, 2023), 3)
    idx = analyze.bootstrap_indices(years, n_boot=200, rng=np.random.default_rng(0))
    assert idx.shape == (200, 99)
    for row in idx:
        counts = np.bincount(row, minlength=99).reshape(33, 3)
        assert (counts == counts[:, :1]).all()  # a year's 3 dates are always drawn together


def test_bootstrap_range_covers_the_true_slope_about_95_percent_of_the_time(mat):
    d = mat["dates"]
    f, p = d["real_future_3m_pct"].to_numpy(), d["real_past_3m_pct"].to_numpy()
    rng = np.random.default_rng(7)
    reps, hits = 200, 0
    for r in range(reps):
        y = 0.05 * f + 0.02 * p + rng.normal(0, 0.05, size=99)
        frame = pd.DataFrame({"date_shift": y, "future": f, "past": p})
        fit = analyze.slope_with_range(frame, "B", n_boot=1000, seed=r)
        hits += fit["lo"] <= 0.05 <= fit["hi"]
    assert 0.88 <= hits / reps <= 0.99


def test_headline_verdict_needs_b_and_the_whole_range_above_zero():
    assert analyze.headline_verdict({"b": 0.05, "lo": 0.01, "hi": 0.09})["leak"] is True
    assert analyze.headline_verdict({"b": 0.05, "lo": -0.01, "hi": 0.09})["leak"] is False
    assert analyze.headline_verdict({"b": -0.01, "lo": 0.001, "hi": 0.09})["leak"] is False


def test_wording_follows_the_prereg_table():
    w = analyze.wording_for_range
    assert w(0.04, 0.09) == "leak, at least the size worth caring about"
    assert w(0.01, 0.05) == "leak; unclear whether it is big enough to matter"
    assert w(0.01, 0.02) == "leak, but smaller than worth caring about"
    assert w(-0.01, 0.02) == "no leak worth caring about"
    assert w(-0.01, 0.05) == "can't tell"
    assert w(-0.05, -0.01) == "the forecast bends away from what happened; reported as unexpected"


# ---------- 6.5 checks ----------

def test_shuffle_placebo_centers_on_zero(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.10, seed=6), n_boot=10_000)
    placebo = res["checks"]["placebo"]
    assert placebo["n_shuffles"] == 10_000
    assert abs(placebo["mean"]) < 0.005
    assert placebo["share_at_least_real"] < 0.001
    null = run_model(tmp_path, mat, planted_forecasts(mat, seed=7))["checks"]["placebo"]
    assert abs(null["mean"]) < 0.005
    assert 0.01 < null["share_at_least_real"] < 0.99


def test_covid_check_drops_january_and_may_2020(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.10, seed=8))
    covid = res["checks"]["covid"]
    assert covid["n_dates"] == 97
    assert covid["dropped"] == ["2020-01-01", "2020-05-01"]
    assert covid["lo"] < 0.10 < covid["hi"]


def test_a_leak_planted_only_on_the_covid_dates_is_flagged_as_not_robust(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, covid_only_leak=0.10, noise=0.2, seed=9))
    assert res["levels"]["month"]["B"]["b"] > 0.01
    assert res["checks"]["covid"]["lo"] < 0 < res["checks"]["covid"]["hi"]
    assert res["checks"]["robust"] is False


def test_partial_spearman_matches_correlating_rank_residuals():
    rng = np.random.default_rng(3)
    f, p = rng.normal(size=99), rng.normal(size=99)
    y = 0.4 * f + 0.6 * p + rng.normal(size=99)
    ry, rf, rp = rankdata(y), rankdata(f), rankdata(p)
    design = np.column_stack([np.ones(99), rp])
    res_y = ry - design @ np.linalg.lstsq(design, ry)[0]
    res_f = rf - design @ np.linalg.lstsq(design, rf)[0]
    assert analyze.partial_spearman(y, f, p) == pytest.approx(np.corrcoef(res_y, res_f)[0, 1])


def test_rank_check_and_robust_flag_on_a_real_leak(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.10, seed=10))
    assert res["checks"]["rank"]["rho"] > 0 and res["checks"]["rank"]["lo"] > 0
    assert res["checks"]["robust"] is True


# ---------- 6.6 blur, size, neutral vs none ----------

def test_blur_curve_has_a_slope_per_level_and_years_are_resampled_whole(tmp_path, mat):
    fc = planted_forecasts(mat, leak=0.10, quarter_leak=0.05, year_leak=0.0, seed=11)
    levels = run_model(tmp_path, mat, fc)["levels"]
    assert levels["month"]["B"]["b"] == pytest.approx(0.10, abs=0.01)
    assert levels["quarter"]["B"]["b"] == pytest.approx(0.05, abs=0.01)
    assert levels["year"]["B"]["lo"] < 0 < levels["year"]["B"]["hi"]
    assert levels["month"]["B"]["n_clusters"] == 99 and levels["year"]["B"]["n_clusters"] == 33


def test_neutral_label_vs_no_date_gap(tmp_path, mat):
    res = run_model(tmp_path, mat, planted_forecasts(mat, noise=0.0, none_offset=0.3))
    gap = res["neutral_vs_none"]["month"]
    assert gap["mean_gap"] == pytest.approx(-0.3, abs=1e-4)
    assert gap["mean_abs_gap"] == pytest.approx(0.3, abs=1e-4)
    assert gap["n_series"] == 20


def test_size_curve_order():
    names = ["gemini-3.8-flash", "something-new", "gemma4:12b", "gemma4:e2b", "gemini-3.1-pro-preview"]
    assert analyze.size_order(names) == ["gemma4:e2b", "gemma4:12b", "gemini-3.8-flash", "gemini-3.1-pro-preview",
                                         "something-new"]


# ---------- 6.7 knowledge quiz ----------

def quiz_kept(answers, replies=1):
    records = [quiz_record(pid, "m", t, k if replies > 1 else None) for pid, texts in answers.items()
               for k, t in enumerate(texts)]
    return analyze.counted_replies(records)[0]


def test_quiz_perfect_model_knows_the_history(mat):
    score = analyze.score_quiz(analyze.quiz_answers(quiz_kept(truth_answers(mat))), mat["quiz"], n_boot=2000,
                               seed=analyze.SEED)
    main = score["failures_dropped"]
    assert main["balanced_accuracy"] == 1.0 and main["plain_accuracy"] == 1.0 and main["knows"] is True


def test_quiz_always_rise_is_a_coin_flip(mat):
    answers = {pid: [answer("rise")] for pid in mat["quiz"]["prompt_id"]}
    score = analyze.score_quiz(analyze.quiz_answers(quiz_kept(answers)), mat["quiz"], n_boot=2000, seed=analyze.SEED)
    main = score["failures_dropped"]
    assert main["plain_accuracy"] == pytest.approx(72 / 99)
    assert main["balanced_accuracy"] == pytest.approx(0.5)
    assert main["knows"] is False


def test_gemini_quiz_uses_the_majority_and_ties_fail():
    cases = {
        "quiz|A": [answer("rise"), answer("rise"), answer("fall")],
        "quiz|B": [answer("rise"), answer("fall"), "garbage"],  # 1-to-1 tie
        "quiz|C": ["garbage", "garbage", "garbage"],
        "quiz|D": [answer("fall"), "garbage", "garbage"],  # one readable reply
        "quiz|E": [answer("fall"), answer("fall"), "garbage"],
    }
    got = analyze.quiz_answers(quiz_kept(cases, replies=3)).set_index("prompt_id")["answer"]
    assert got["quiz|A"] == "rise" and got["quiz|D"] == "fall" and got["quiz|E"] == "fall"
    assert pd.isna(got["quiz|B"]) and pd.isna(got["quiz|C"])


def test_quiz_failures_are_reported_both_dropped_and_counted_wrong(mat):
    fall = list(mat["quiz"].loc[mat["quiz"]["truth"] == "fall", "prompt_id"])
    answers = truth_answers(mat)
    for pid in fall[:10]:
        answers[pid] = ["garbage"]
    score = analyze.score_quiz(analyze.quiz_answers(quiz_kept(answers)), mat["quiz"], n_boot=2000, seed=analyze.SEED)
    assert score["failures"] == 10
    assert score["failures_dropped"]["n_dates"] == 89
    assert score["failures_dropped"]["balanced_accuracy"] == pytest.approx(1.0)
    assert score["failures_wrong"]["n_dates"] == 99
    assert score["failures_wrong"]["balanced_accuracy"] == pytest.approx((1 + 17 / 27) / 2)


def test_exploratory_slopes_on_dates_answered_right_and_wrong(tmp_path, mat):
    flip = set(mat["quiz"]["prompt_id"].iloc[::3])
    res = run_model(tmp_path, mat, planted_forecasts(mat, leak=0.10, seed=12), quiz=truth_answers(mat, flip))
    split = res["quiz"]["exploratory_slopes"]
    assert split["exploratory"] is True
    assert split["right"]["n_dates"] == 66 and split["wrong"]["n_dates"] == 33
    assert split["right"]["b"] == pytest.approx(0.10, abs=0.02)


# ---------- safety and the command line ----------

def test_refuses_pilot_replies():
    for folder in (analyze.PILOT, analyze.PILOT / "gemma4_e2b"):
        with pytest.raises(SystemExit, match="pilot"):
            analyze.main(["--runs-dir", str(folder)])


def test_study_data_needs_the_study_flag():
    with pytest.raises(SystemExit, match="--study-data"):
        analyze.main(["--runs-dir", str(analyze.RUNS)])


def test_study_data_needs_the_analysis_code_committed(monkeypatch):
    monkeypatch.setattr(analyze, "uncommitted_files", lambda: ["src/analyze.py"])
    with pytest.raises(SystemExit, match="commit"):
        analyze.main(["--runs-dir", str(analyze.RUNS), "--study-data"])


def test_end_to_end_on_fake_runs(tmp_path, mat):
    runs, out = tmp_path / "runs", tmp_path / "out"
    flash = planted_forecasts(mat, leak=0.10, seed=13)
    write_forecasts(runs / "gemini-3.8-flash" / "forecasts.jsonl", "gemini-3.8-flash", mat, flash, replies=3,
                    reply_noise=0.2)
    write_quiz(runs / "gemini-3.8-flash" / "quiz.jsonl", "gemini-3.8-flash",
               {pid: texts * 3 for pid, texts in truth_answers(mat).items()}, replies=3)
    write_forecasts(runs / "gemma4_12b" / "forecasts.jsonl", "gemma4:12b", mat, planted_forecasts(mat, seed=14))
    assert analyze.main(["--runs-dir", str(runs), "--out-dir", str(out), "--n-boot", "2000"]) == 0

    summary = json.loads((out / "analysis.json").read_text())
    head = summary["headline"]
    assert head["model"] == "gemini-3.8-flash" and head["available"] is True
    assert head["leak"] is True and head["robust"] is True
    assert head["wording"] == "leak, at least the size worth caring about"
    assert [row["model"] for row in summary["size_curve"]] == ["gemma4:12b", "gemini-3.8-flash"]
    assert summary["models"]["gemini-3.8-flash"]["quiz"]["failures_dropped"]["knows"] is True
    assert summary["models"]["gemma4:12b"]["quiz"] is None
    assert (out / "report.md").exists()
    assert (out / "failures_by_date_gemini-3.8-flash.csv").exists()
    assert (out / "date_shifts_gemma4_12b.csv").exists()


def test_missing_headline_model_is_reported_not_guessed(tmp_path, mat):
    runs, out = tmp_path / "runs", tmp_path / "out"
    write_forecasts(runs / "gemma4_12b" / "forecasts.jsonl", "gemma4:12b", mat, planted_forecasts(mat, leak=0.1))
    assert analyze.main(["--runs-dir", str(runs), "--out-dir", str(out), "--n-boot", "500"]) == 0
    head = json.loads((out / "analysis.json").read_text())["headline"]
    assert head["model"] == "gemini-3.8-flash" and head["available"] is False
