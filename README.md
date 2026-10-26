# Date Label Leak

**Does a date label alone let an AI "remember" what happened next?**

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

The MIT license covers the code and the write-up. The INDPRO and NBER data files in `data/raw/` keep their own terms (INDPRO: public domain, citation requested; NBER: free to copy with attribution to NBER), listed in `data/raw/SOURCES.md`.

## Acknowledgments and AI use
I chose the question and made the design decisions: the made-up series, the real month labels, the blur levels, the
real-outcome measure, the one yes/no test and the three checks, which models to run, 20 series for every model, the
0.03 "worth caring about" bar, and Gemini 3.8 Flash as the headline model. I approved the pre-registration (`PREREG.md`)
on Oct 12, 2026, before any study run, and chose the MIT license. Claude (Anthropic's AI assistant) wrote the code and
drafted the text under my direction. I started some model runs from my own terminal; Claude ran the rest under my direction. On Oct 15, 2026 I read and
approved the final text of this README and the report. The numbers were checked against the result files by Claude,
not re-audited line by line by me.
