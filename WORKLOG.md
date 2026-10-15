# Work log (public version)

Who did what, by date. This is a trimmed copy of the project's working log: research decisions, runs and results only.
Because this repository is a fresh single-commit copy, commit ids that appear in some files (for example `0182df1`)
refer to the project's original history, which is not included here.

**Credit:** Code, analysis and text drafted by Claude (AI); directed, checked and approved by Jayden Ling.
Jayden chose the question, made each design decision below, approved the pre-registration before any study run,
and approved the final write-up. No study run happened before the pre-registration was committed; only small pilots
did, and pilot data is not study data.

## 2026-10-12: plan, data, code, pre-registration
- Jayden decided the design (`PREREG.md` sections 1 to 4): made-up AR(1) series of monthly percent changes; 99 real
  month labels (every 4th month, January 1990 to September 2022); US Industrial Production as the real outcome; a
  blur dial (month, quarter, year, none); the leak measured as a slope with the past 3 months held fixed; one
  declared yes/no test plus COVID, rank and shuffle checks.
- Downloaded Industrial Production (FRED) and the NBER business-cycle table; terms logged in `data/raw/SOURCES.md`.
- Real Industrial Production, 1990 to 2022 without 2020: average +0.135% a month, standard deviation 0.660%,
  autocorrelation 0.14. April 2020 was -13.2%, about 34 times a typical month (`src/real_stats.py`).
- Generated 50 AR(1) series (phi 0.2, 0.5, 0.8; seed 20261012; the formula check passed). The study uses the first 20.
  Built the prompts: 4,700 forecast prompts per model, plus a 99-question knowledge quiz (72 rise, 27 fall).
- Pilot (440 prompts per Gemma model, no comparison with real outcomes): 0 parse failures. Gemma 4 E2B copied a
  value from the series about a third of the time, and about half its shifts were exactly 0. A Gemini 3.8 Flash cost
  test (1,320 replies) had 0 failures and 9 "wild" replies (more than 10 noise units off); all 9 forecast a 12 to 17%
  drop by month 3. We did not look up which labels they were and changed no plan because of them.
- Claude drafted `PREREG.md`; Jayden decided each open point and approved it as a whole, and it was committed before
  any study run. His decisions: 20 series for every model; 0.03 as the smallest slope worth caring about; the
  prediction as drafted; Gemini 3.8 Flash as the headline model.
- Two logged changes, both about how Gemini was reached and billed and both made before any study data existed
  (`PREREG.md` section 8): the Google Cloud Gemini platform on a free-trial credit instead of Google AI Studio, and
  Flex pricing instead of batch jobs.

## 2026-10-12 to 13: study runs
- Gemma 4 E2B, E4B and 12B on a Mac: 4,700 forecasts and 99 quiz replies each; 0 parse failures, 0 failed requests.
- Gemini 3.8 Flash: 14,100 forecast replies (4,700 prompts x 3 replies) and 297 quiz replies; 0 failed requests, 0
  parse failures, 6 wild replies. Estimated cost about $7.23 from token counts, not checked against the Google bill
  (UNVERIFIED). The forecast run was stopped partway and restarted; it resumed with no duplicates and no lost replies.
- Claude checked completeness and failures only before the analysis; nothing was compared with real outcomes.

## 2026-10-13: analysis
- Ran the pre-registered analysis (`src/analyze.py`, 10,000 bootstrap draws). Headline test, Gemini 3.8 Flash with
  month labels: slope +0.1135 [+0.0831, +0.2135], above the 0.03 bar; leak = yes. Robust to dropping the COVID dates
  (+0.18), the rank check (rho +0.65) and the shuffle placebo (about 0). Gemini scores 94% on the knowledge quiz.
- Quarter labels still leak (+0.079); year labels cannot tell (+0.011 [-0.002, +0.062]). The Gemma models show no leak,
  but they do not know the history either (quiz about 50%). Results are in `results/main/`; a Gemma-only run done
  earlier is in `results/gemma_only/`.
- Claude drafted `writeup/REPORT.md` and `README.md`.

## 2026-10-15: fact-check and final text
- Claude checked every number in the report and README against `results/main/`, `PREREG.md` and `data/raw/SOURCES.md`:
  all matched. Wording problems found and fixed (approved by Jayden): a sentence that overstated what the date label
  proves; a note that all 6 dropped wild replies are on the January 2020 label; the total cost (about $8 including the
  cost and connection tests); a short "predicted vs happened" paragraph (P1, P2, P5 held; P3 did not; P4 went the
  other way); and a prior-work paragraph that no longer claims to be first.
- One final writing pass on the report and README (wording only; numbers, claims and labels unchanged). Jayden approved
  the final text. Not done: Google's terms on republishing the raw Gemini replies (UNVERIFIED).
