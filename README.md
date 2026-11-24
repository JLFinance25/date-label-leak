# Date Label Leak

**Does a date label alone let an AI "remember" what happened next?**

## Abstract
**Objective.** Test whether a real month label alone ("January 2020") pulls an AI model's forecast toward what the US
economy actually did next, even when the numbers it sees are made up.
**Data and method.** 20 made-up AR(1) monthly series, each shown with 99 real month labels (every 4th month, Jan 1990
to Sep 2022) at three precision levels (month, quarter, year) and with a neutral label. Four models: Gemini 3.8 Flash
(`gemini-3.8-flash`, Google Cloud, queried Oct 12-13, 2026) and Gemma 4 E2B, E4B and 12B (Ollama, run locally). The
leak is the slope of the forecast shift against the real 3-month change in US Industrial Production (FRED INDPRO),
holding the past 3 months fixed, with a 10,000-draw bootstrap. The test and its pass bar (0.03) were pre-registered.
**Findings.** Gemini 3.8 Flash leaks: slope +0.11 [95% range +0.08, +0.21], about a sixth of a full leak, and it
survived all three planned checks. Quarter labels still leak (+0.08 [+0.05, +0.18]); year labels shrink it about
tenfold (+0.01 [-0.00, +0.06], can't tell). The Gemma models show no leak but also don't know the history (quiz near
chance). In real units, the January 2020 label alone moved Gemini's forecast down by about 0.75 percentage points per
forecast month.
**What's new.** Earlier tests kept real economic numbers, so the date and familiar numbers were tangled; here the
numbers are made up, so only the date can carry the leak, and the leak is measured at three date precisions.
**Practical use.** Anyone back-testing an AI forecaster on past data should not assume hiding names is enough: a
month or quarter label can leak; test with year labels or with periods after the model's training cutoff.

Researchers who test AI forecasters on past data sometimes hide company names. One example is Glasserman and Lin
(arXiv 2309.17322), who anonymized company names in headlines. Such tests may still keep the dates, and how often
they do is UNVERIFIED.

## The question
This project shows an AI the same made-up numbers with different real month labels ("May 2009", "January 2020") and
checks whether its forecast leans toward what the US economy actually did next. The numbers carry no real history and
only the date label changes, so any shift comes from what the model associates with that date.

**Status: results final; not peer-reviewed.**

## Results (pre-registered test)
The "shift" is the forecast with a real date label minus the forecast with the neutral label "month t". A "noise
unit" is the size of a series' typical random monthly wobble.
- **Gemini 3.8 Flash leaks.** Its shift follows what US Industrial Production really did over the next 3 months:
  slope +0.11 [95% range +0.08, +0.21] noise units per percentage point, holding the past 3 months fixed. That is
  above the pre-registered "worth caring about" bar of 0.03, and it survived all three planned checks (dropping the COVID dates, a rank-based version, and shuffling
  which outcome goes with which date).
- **Quarter labels still leak** (+0.08 [+0.05, +0.18]). Year-only labels shrink the slope about tenfold, but the
  study can't rule out a leak worth caring about there (+0.01 [-0.00, +0.06]).
- **The three small Gemma 4 models (E2B, E4B, 12B) don't leak, but they also don't know the history** (their quiz
  scores are near chance), so their result says little.

- **In real units** (Claude's arithmetic on `results/main/date_shifts_gemini-3.8-flash.csv`, converting with the
  median noise size of the 20 series, 0.57 percentage points of monthly change; an approximation, since each series
  has its own noise size): with the "January 2020" label, Gemini's forecast fell about 0.75 percentage points per
  forecast month (1.31 noise units); with "May 2009" it rose about 0.32 points (0.56 noise units). Across all 99
  month labels, the date alone moved the forecast by about 0.13 points per month on average (0.22 noise units),
  about a fifth of one month's random wobble.

## What's new
The closest work we know (abstracts read Nov 15, 2026): Fake Date Tests (Eliseev and Seleznev, arXiv 2601.07992)
moves the dates on real macro forecasting prompts and finds lookahead bias in every model tested; HindsightBench (Jia,
arXiv 2607.18867) runs a date-only arm on a real, vintage-correct macro panel across 15 models; The Memorization
Problem (Lopez-Lira, Tang and Zhu, arXiv 2504.14765) shows models recall exact economic values and rebuild masked
dates. **What this study adds:** the numbers are made up, so the correct forecast comes from a formula and only the
date label can carry outside knowledge; and the leak is measured as a slope at three date precisions (month,
quarter, year). Details of these papers beyond their abstracts are UNVERIFIED.

## Who this is for
Research groups that back-test AI forecasters on historical economic or financial data. The decision it informs:
whether hiding names is enough, or whether dates must be blurred to the year or the test moved after the model's
training cutoff. To reuse it you need your model's API or local weights, the scripts in `src/`, and about 4,800
prompts per model (3 replies each for a model that can't run at temperature 0; about $8 of cloud credit for Gemini
3.8 Flash here). Run it once per model version before trusting a back-test.

Full plain-English write-up: [`writeup/REPORT.md`](writeup/REPORT.md). Numbers: `results/main/`.

## What I did
I chose the question and the design below. The code does the following.

1. `src/make_series.py` makes 50 AR(1) series (each month pulled partway back to its average, plus random luck); the
   study uses the first 20. The right forecast comes from a formula, so it can't depend on a date.
2. `src/build_prompts.py` builds the prompts: 99 real month labels (every 4th month, Jan 1990 to Sep 2022) at three
   blur levels (month, quarter, year), plus neutral labels, plus a 99-question history quiz.
3. `src/run_models.py` (Gemma via Ollama, temperature 0, thinking off) and `src/run_gemini.py` (Gemini, 3 replies per
   prompt) save every reply as JSON lines in `data/runs/`.
4. `src/analyze.py` runs the analysis exactly as `PREREG.md` section 6 says: shifts, the slope with a 10,000-draw
   bootstrap over dates, the checks, the blur and quiz results. It was tested first on fake replies with a planted
   leak (`tests/`).

I locked the plan and the one yes/no test in `PREREG.md`, and it was committed before any study run. Changes after
that are logged in its section 8.

## Reproduce
```
python3 -m pytest tests/                 # tests on fake replies, no model needed
python3 src/analyze.py --study-data --out-dir results/main   # re-runs the analysis on the saved replies
```
Re-running the models needs Ollama with the exact model IDs in `PREREG.md` 5.1, and a Google Cloud project for Gemini.

## Data sources
- Board of Governors of the Federal Reserve System (US), Industrial Production: Total Index [INDPRO], retrieved from
  FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO (pulled Oct 12, 2026).
- NBER business-cycle dates (context only): https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions. Source: NBER.

| Dataset | Link | Size | Licence or terms |
|---|---|---|---|
| FRED INDPRO (`data/raw/INDPRO.csv`) | https://fred.stlouisfed.org/series/INDPRO | 1,292 monthly rows (Jan 1919 to Aug 2026), 24 KB; pulled Oct 12, 2026 | Public Domain: Citation Requested |
| NBER cycle dates (`data/raw/NBER_business_cycle_dates.xlsx`) | https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions | 21 KB; pulled Oct 12, 2026 | Copy allowed with attribution ("Source: NBER") |
| Model replies (`data/runs/`) | this repo | 14,100 Gemini and 14,100 Gemma forecast replies, plus quiz replies | Gemma 4: Apache 2.0. Gemini output: Google Cloud Service Specific Terms (read Oct 16, 2026) say Google claims no ownership of output and ban using it to build a competing model |
| Made-up series (`data/processed/series.csv`) | this repo | 50 series (first 20 used) | made by `src/make_series.py` |

The MIT license covers the code and the write-up. The INDPRO and NBER data files in `data/raw/` keep their own terms (INDPRO: public domain, citation requested; NBER: free to copy with attribution to NBER), listed in `data/raw/SOURCES.md`.

## Acknowledgments and AI use
I chose the question and made the design decisions: the made-up series, the real month labels, the blur levels, the
real-outcome measure, the one yes/no test and the three checks, which models to run, 20 series for every model, the
0.03 "worth caring about" bar, and Gemini 3.8 Flash as the headline model. I approved the pre-registration (`PREREG.md`)
on Oct 12, 2026, before any study run, and chose the MIT license. Claude (Anthropic's AI assistant) wrote the code and
drafted the text under my direction. I started some model runs from my own terminal; Claude ran the rest under my direction. On Oct 15, 2026 I read and
approved the final text of this README and the report. The numbers were checked against the result files by Claude,
not re-audited line by line by me.

Models tested: Gemini 3.8 Flash (`gemini-3.8-flash`, stable, Google Cloud, thinking "low", queried Oct 12-13, 2026);
Gemma 4 E2B, E4B and 12B (Ollama IDs 45d11505b91d, eb2177adf764, c7597fc90b86; nvfp4; Ollama 0.40.3; Apache 2.0).
My project log (kept privately; the public `WORKLOG.md` is a trimmed summary) records Claude Sonnet 5.5 and Claude
Opus 5.5 for the Oct 15, 2026 polish passes and Claude Opus 5.5 for the Nov 15, 2026 draft fixes; the Claude model used
in earlier sessions is not recorded (UNVERIFIED).

No teacher, mentor or other adult helped with this work; all help came from the AI tools named above.

## Related papers

More pre-registered studies by Jayden Ling are on Zenodo: [all papers](https://zenodo.org/search?q=creators.name%3A%22Ling%2C%20Jayden%22). Flagship: *When do operations-research models beat simple baselines on public data? 38 pre-specified tests*, [doi:10.5281/zenodo.24280952](https://doi.org/10.5281/zenodo.24280952). This repo has no Zenodo record of its own yet. Sibling repos: [number-guard](https://github.com/JLFinance25/number-guard), [phantom-answers](https://github.com/JLFinance25/phantom-answers), [date-label-leak](https://github.com/JLFinance25/date-label-leak), [february-cliff](https://github.com/JLFinance25/february-cliff), [chronology-pairs](https://github.com/JLFinance25/chronology-pairs).
