# Does a date label alone leak the future into an AI forecast?

**Status: final text approved by Jayden Ling (Oct 15, 2026).**
**Credit:** Code, analysis and this text drafted by Claude (AI); directed and checked by Jayden Ling.

## The short answer
Yes, for the one model that knows the history. When Gemini 3.8 Flash saw made-up numbers labeled with a real
month (like "January 2020"), its forecast leaned toward what US industry actually did in the 3 months after that
month.

The made-up numbers carry no real history, and the only thing that changes between each real-date prompt and its
neutral version ("month t") is the date label. So the shift comes from what the model associates with that date. This study can't fully separate memory of
what happened next from knowledge of what the time was like (see "What this study can't tell you").

The three small Gemma 4 models showed no leak, but they also didn't know the history, so their result says little.

## What we did
1. **Made-up numbers.** Code generated 20 monthly series that follow a simple rule (AR(1): each month is pulled
   partway back to its average, plus random luck). Because the code makes the numbers, the correct forecast comes
   from a formula and can't depend on any date.
2. **Real date labels.** Each series was shown to each AI with 99 different real month labels (every 4th month,
   January 1990 to September 2022), and once with a neutral label ("month t"). The AI forecast the next 3 months.
3. **The shift.** For each date, the shift is the forecast with the real date minus the forecast with "month t",
   averaged over the 20 series. It is measured in "noise units": the size of a series' typical random monthly
   wobble (median 0.57 percentage points; `data/processed/series.csv`).
4. **The leak.** The leak is the slope of the shift against what US Industrial Production really did over the 3
   months after each date, holding fixed what it did over the 3 months before. Holding the past fixed matters: a
   model is allowed to know the past, and monthly changes are mildly persistent (autocorrelation 0.14,
   `src/real_stats.py`, run Oct 12, 2026, on INDPRO 1990 to 2022 without 2020).
5. **Locked in advance.** The design, the one yes/no test and the "smallest leak worth caring about" (0.03) were
   pre-registered and committed before any study run (`PREREG.md`, approved by Jayden Oct 12, 2026). Two logged
   changes were made before any study data was seen, both about how Gemini was reached and billed (PREREG
   section 8).

## Results
All numbers below come from `results/main/`, made by `src/analyze.py` (code state at commit 0182df1, Oct 13, 2026).
The 95% ranges in brackets come from 10,000 bootstrap re-draws of the dates (seed 20261012).

**The pre-registered test (Gemini 3.8 Flash, exact-month labels): leak = yes.**
- **Slope: +0.11 [+0.08, +0.21]** noise units per percentage point of real 3-month change. The whole range is above
  the 0.03 bar, so by the pre-registered wording it is "a leak, at least the size worth caring about".
- **Size.** On our own yardstick (PREREG 6.4), a model that fully used the future would score about 0.64, so this
  is roughly a sixth of a full leak.
- **It held up under every planned check.**
  - Dropping the COVID dates: +0.18 [+0.12, +0.24].
  - A rank-based version (less sensitive to extreme months): correlation +0.65 [+0.50, +0.77].
  - Shuffling which outcome goes with which date (10,000 times): the slope averages about 0, and none of the
    10,000 shuffles was as large as the real slope.
- **It knows the history.** Asked "did US industrial production rise or fall in the 3 months after [date]?", it
  scored 94% balanced accuracy [87%, 99%] (50% would be guessing).

**Two dates that show it** (`results/main/date_shifts_gemini-3.8-flash.csv`; real changes from Industrial
Production, FRED, pulled Oct 12, 2026; revised data as of that date, not what was first published):
- **January 2020.** The index's last 3 months were nearly flat (-0.4%); the next 3 months fell 16.3% (COVID). With
  the same made-up numbers and the label "January 2020", the forecast dropped by 1.3 noise units.
- **May 2009.** The last 3 months had fallen 3.4%; the next 3 rose 1.9%. With the label "May 2009", the forecast
  went *up* by 0.6 noise units, against the recent past, toward the recovery that came next.

**Blurring the date (descriptive, not a separate yes/no test):**

| Label shown | Slope [95% range] | Pre-registered wording |
|---|---|---|
| Month ("September 2008") | +0.11 [+0.08, +0.21] | leak, at least worth caring about |
| Quarter ("2008 Q3") | +0.08 [+0.05, +0.18] | leak, at least worth caring about |
| Year ("2008") | +0.01 [-0.00, +0.06] | can't tell |

Blurring to the quarter didn't remove the leak. Blurring to the year shrank the best-guess slope about tenfold, but
its range still reaches above 0.03, so this study can't say the year-only leak is too small to matter.

**The small models (Gemma 4 E2B, E4B, 12B, run free on a Mac):**

| Model | Month-level slope | Knows the history? (quiz, balanced accuracy) |
|---|---|---|
| Gemma 4 E2B | +0.002 [-0.004, +0.005] | no: 52% [43%, 62%] |
| Gemma 4 E4B | +0.001 [-0.004, +0.003] | no: 55% [48%, 61%] |
| Gemma 4 12B | +0.004 [-0.005, +0.020] | no: 48% [42%, 53%] |
| Gemini 3.8 Flash | +0.11 [+0.08, +0.21] | yes: 94% [87%, 99%] |

None of the Gemma models leaked, and none knew the history either. A model can't leak a future it doesn't know.

**Data quality.**
- Every reply was readable: 0 parse failures in 14,100 Gemma and 14,100 Gemini forecast replies.
- 6 Gemini replies (0.04%) were "wild" (more than 10 noise units from the formula) and were dropped, as planned.
  All 6 were on the January 2020 label (3 at month level, 3 at quarter level;
  `results/main/failures_by_date_gemini-3.8-flash.csv`).
- Note (our reasoning, not a tested result): the saved January 2020 replies show spring-2020-shaped drops, so the
  rule probably removed some of the most leak-shaped replies. That would make the reported slope cautious rather
  than inflated. We did not check which replies exactly crossed the 10-unit cut.

**What we predicted vs what happened** (predictions P1 to P5, `PREREG.md` section 2, written before any study data;
checked here against `results/main/report.md`):
- P1 (Gemini 3.8 Flash leaks): holds (+0.11 [+0.08, +0.21]).
- P2 (year slope smaller than month slope): holds (+0.01 vs +0.11).
- P3 (among the Gemma models the slope grows with size, E2B about 0): does not hold. All three slopes are near 0
  (+0.0015, +0.0011, +0.0039) with ranges that include 0, and there is no clear growth.
- P4 (dropping COVID dates makes the slope smaller): wrong in direction. The slope went up, to +0.18.
- P5 (Gemini knows the history, E2B is near a coin flip): holds (94% vs 52%).

## What this means
Suppose you test an AI forecaster on past data, hiding company names but keeping the month. In this one model, the
date alone then moved the forecast toward what happened next, though part may be knowledge of the time rather than
memory of what followed. In this test, a quarter label was not blurry enough.

## What this study can't tell you
- **Remembering the future vs. knowing the mood of the time.** We held the index's own past fixed, but not other
  news known *at* the label month. Part of the pull could come from that, not from memory of what came later. The
  January 2020 and May 2009 examples point to the future, but the design can't fully separate the two. Either way,
  outside knowledge got into a forecast of made-up numbers through the date.
- **One model leaked.** Only one large model was tested. Whether bigger models leak more can't be answered here:
  the models that didn't leak also didn't know the history, and they come from a different product line. Gemini
  3.8 Flash's size isn't public (UNVERIFIED).
- **Settings.** Gemini can't be set to its single most likely answer (temperature 0), so each Gemini forecast is
  the average of 3 replies, with "thinking" set to low (PREREG 5.2).
- **Prior work.**
  - The closest study we found is Fake Date Tests (Eliseev and Seleznev, 2026, arXiv 2601.07992). It kept real US
    data (CPI, GDP, interest rates) and moved the forecast and cutoff dates to February 2030; none of the three
    models tested passed. Because the numbers were real, the date and familiar numbers were tangled together.
    Here the numbers are made up.
  - Newer close relatives: HindsightBench (arXiv 2607.18867, Jul 2026), which has a date-only arm on a macro panel
    and calls the existence of this kind of leak "settled" (whether any arm uses made-up numbers is UNVERIFIED);
    NumLeak (arXiv 2605.30393); and Detecting Lookahead Bias in LLM Forecasts (arXiv 2512.23847).
  - The Memorization Problem (Lopez-Lira, Tang and Zhu, arXiv 2504.14765) supports the mechanism.
  - Glasserman and Lin (arXiv 2309.17322) is cited only for anonymizing company names; whether they kept dates is
    UNVERIFIED.
  - To our knowledge (a search on Oct 15, 2026), we did not find a study that put real date labels at several
    precision levels on made-up numbers and measured a slope. One search can't show nobody did it.
  - Details of these papers beyond their abstracts are second-hand (UNVERIFIED); read them before citing.

## Sources
- Board of Governors of the Federal Reserve System (US), Industrial Production: Total Index [INDPRO], retrieved from
  FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO (pulled Oct 12, 2026; public
  domain, citation requested).
- NBER US business-cycle dates (context only): https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
  (pulled Oct 12, 2026). Source: NBER.
- Models: Gemma 4 E2B, E4B, 12B (Ollama; IDs in `PREREG.md` 5.1), and Gemini 3.8 Flash on Google Cloud's Gemini
  platform (Oct 12 to 13, 2026). Cost: about $8 of cloud credit in total. The main run is estimated at $7.23 from
  token counts and not checked against the Google bill (UNVERIFIED); the rest was a cost test and a connection test.
