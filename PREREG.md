# Pre-registration: Date Label Leak

**Status: APPROVED by Jayden Ling on Oct 12, 2026, and committed before any study run.** Drafted by Claude (AI) on
Oct 12, 2026, from the approved `PLAN.md` (decisions D0 to D12) and the code in `src/`. Jayden made every open
decision on Oct 12 (section 9). Any change from here on is logged in section 8.

**Credit:** Drafted by Claude (AI); directed, checked and approved by Jayden Ling on Oct 12, 2026.

**Code this file points at:** git commit `e5af9057cd7bcb269ba47ce8542e0f5e21ce89c8` (Oct 12, 2026: code, data and pilot snapshot).

---

## 1. The question

Researchers who test AI forecasters on past data often hide company or series names, but they usually keep the
dates. **Does a real calendar date label by itself push a model's forecast toward what really happened next, even
when the numbers are made up and have no real history?**

Two follow-up questions, described but not tested yes/no:
- **Blur:** does the effect shrink as the date gets vaguer (exact month, then quarter, then year)?
- **Size:** do bigger models show more of it?

## 2. Prediction (drafted by Claude; approved as drafted by Jayden Oct 12, 2026)

Drafted by Claude. These are guesses made before any study data exists. They are written down so we can't move
the goalposts later. Only P1 is a yes/no test (section 6.3). The others are described, not tested.

- **P1 (headline).** The headline model (section 4) leaks: its month-level slope is above 0, and the 95% range is
  above 0. We guess it is also at least the smallest size worth caring about (0.03, section 6.4).
- **P2 (blur).** For each model that leaks, the year-level slope is smaller than the month-level slope.
- **P3 (size).** Among the Gemma 4 models, the slope grows with size, and E2B's slope is about 0. In the pilot, E2B
  often copied a number from the series, and half its shifts were exactly 0.
- **P4 (COVID).** Dropping the two dates whose windows touch Feb to Jun 2020 makes the headline slope smaller, but
  it stays above 0.
- **P5 (quiz).** The Gemini models know what happened after most dates (balanced quiz score clearly above 50%).
  Gemma 4 E2B is close to a coin flip.

Why we guess this (a guess, not a known fact): big models have read a lot about recessions and know which months
were bad. A label like "September 2008" may pull their forecast down even though the numbers say nothing about 2008.
Small models mostly copy or extend the numbers and ignore the label.

## 3. The answer key: why the date can't change the right answer

Each made-up series follows a simple rule called AR(1). Every month, the value is pulled partway back toward its
average, then a random shock is added:

    next month = average + phi x (this month - average) + random shock

`phi` (between 0 and 1) says how much of this month's distance from the average carries into next month. The shock
is random noise with an average of 0.

The best possible forecast h months ahead is therefore:

    forecast(h) = average + phi^h x (last value - average)

**In plain words:** the best guess is that the gap from the average shrinks by the factor phi every month. The future
shocks are pure luck with an average of 0, so the best guess for them is 0, and they drop out. What's left uses
only three things: the average, phi and the last value. **No date goes into the formula**, so the right answer is
identical whether the label says "September 2008", "May 1995" or "month t". If a model's forecast moves when only
the date label changes, the label caused that move. If the move lines up with what the real economy did next,
the label leaked the future.

`src/make_series.py` checks the formula by simulating 200,000 futures from each series' last value. The average
of those futures matches the formula for all 50 series and all 3 months.

## 4. Materials

### 4.1 Made-up series (`src/make_series.py`, output `data/processed/series.csv`)
- 50 AR(1) series of monthly percent changes, random seed 20261012. Series i uses phi = 0.2, 0.5, 0.8 in turn
  (0.2 for i = 0, 3, 6, ...).
- Average 0.135% a month and overall spread (standard deviation) 0.660%. Both match real US Industrial Production,
  Jan 1990 to Dec 2022 without 2020 (`src/real_stats.py`). The noise size is set so every series has that same
  spread: noise sd = 0.660 x sqrt(1 - phi^2), which is 0.647, 0.572 and 0.396 for the three phi values.
- 200 warm-up months are thrown away. Then 24 months are kept and shown to the model, rounded to 2 decimals.
  The answer key starts from the *rounded* last value, because that's what the model sees.
- **Series used (decided by Jayden Oct 12, 2026):** **the first 20 (series 0 to 19) for every model**, local and
  Gemini, so every model answers exactly the same 4,700 prompts. Series 20 to 49 are not used.

### 4.2 Dates (`src/build_prompts.py`, output `data/processed/label_dates.csv`)
- Every 4th month from January 1990 to September 2022 (January, May and September of each year): **99 dates**. It's
  a fixed rule, so no date was hand-picked. Because of the 4-month gaps, no two dates' 3-month windows overlap.
- For each date t, from INDPRO (section 4.5):
  - **real future change** = percent change in the index from month t to month t+3;
  - **real past change** = percent change from month t-3 to month t.
- **COVID window:** a date is flagged if any month whose change it uses (t-2 to t+3) falls between Feb and Jun 2020.
  That flags two dates: **January 2020** (future -16.3%) and **May 2020** (future +11.6%, past -15.2%).

### 4.3 Blur levels and neutral labels (D5)
Only the last value gets a date. The levels:

| Level | Real label example | Neutral label | Distinct real labels |
|---|---|---|---|
| month | "September 2008" | "month t" | 99 |
| quarter | "2008 Q3" | "quarter t" | 99 |
| year | "2008" | "year t" | 33 (each year covers 3 dates) |
| none | (no date sentence) | none | none |

At every level, the outcome is the real future change for the true month. The model just sees less of the date.

### 4.4 Prompt wording (approved by Jayden Oct 12; `src/build_prompts.py`)

> Below are 24 consecutive monthly values of a US economic activity index, shown as percent change from the previous
> month: {24 values}. The last value is for {label}. Forecast the next 3 monthly values. Reply only with JSON in this
> form: {"forecast": [x, y, z]}

With no date, the sentence "The last value is for {label}." is left out. That gives 235 prompts per series
(month 100, quarter 100, year 34, none 1), or **4,700 for 20 series** (11,750 for all 50).

**Knowledge quiz (D10), one per date, 99 per model:**

> In the 3 months after {Month Year}, did US industrial production rise or fall? Reply only with JSON in this form:
> {"answer": "rise"} or {"answer": "fall"}

The right answer is "rise" if the real future change is above 0. That's 72 rises and 27 falls.

### 4.5 Real-world data
- **INDPRO** (US Industrial Production, Total Index, monthly, seasonally adjusted; Federal Reserve G.17 via FRED).
  Vintage: FRED's data as downloaded **Oct 12, 2026** (last updated Sep 18, 2026). These are today's revised
  numbers, not first-published ones (D4). This file is never re-downloaded or edited, even after FRED's
  Oct 16 release. Details and terms: `data/raw/SOURCES.md`.
- NBER business-cycle dates: for charts and context only, not for any test.

### 4.6 Fingerprints of the exact files (SHA-256, taken Oct 12, 2026)
If any of these change, it's a deviation (section 8).

| File | SHA-256 |
|---|---|
| data/raw/INDPRO.csv | 1a916926060d09b970880153da76d36040505666dbe686ae2abda1ff7eba6279 |
| data/processed/series.csv | bc60991dd61b6a8dd6cae2e608029cad1570013100251cd7233702d31bd7aef8 |
| data/processed/label_dates.csv | 1afad4bb875ed089ee8e52173d23710db92b8f9cb96ed0e899fa6e126663d25f |
| data/processed/prompts.csv | d3cca4726a39545c44ec95948610b7d5ff87e2849e6cd287ffdecbd7bc70d065 |
| data/processed/quiz_prompts.csv | 144a43ec4e26a05e814363c4431b3d498f03d10c3da5213b2780342393a470dd |

## 5. Models, settings and machine

### 5.1 Local models (Ollama; IDs from `data/raw/SOURCES.md`, checked with `ollama list` Oct 12, 2026)

| Model | Ollama ID | Size on disk | Parameters (`ollama show`) | Knowledge cutoff |
|---|---|---|---|---|
| gemma4:e2b | 45d11505b91d | 7.5 GB | 5.2B total ("effective 2B") | January 2025 |
| gemma4:e4b | eb2177adf764 | 9.5 GB | 8.1B total ("effective 4B") | January 2025 |
| gemma4:12b | c7597fc90b86 | 7.7 GB | 12.4B | January 2025 |

Cutoff source: the Gemma 4 model card gives one date for the whole family: "a cutoff date of January 2025"
(https://ai.google.dev/gemma/docs/core/model_card_4, read Oct 12, 2026). The last real outcome month in this study
is December 2022, so every model's training data should cover it.

All three use nvfp4 quantization (a compression that stores the model's numbers with fewer bits). The license is
Apache 2.0, according to `ollama show --license`.

**Settings** (`src/run_models.py`; the same for every local model, forecasts and quiz):
- temperature 0 (always the model's single most likely next word), seed 0;
- thinking off (`think: false`; Gemma 4 has thinking on by default);
- the reply is forced into a JSON schema: exactly 3 numbers for forecasts, or "rise"/"fall" for the quiz;
- one reply per prompt; Ollama's default context length (each prompt is 61 to 68 words);
- Ollama 0.40.3. Don't update Ollama or re-download a model until all runs are done. Before and after each run,
  check `ollama list` shows the IDs above.

**Machine (decided by Jayden Oct 12, 2026):** **Jayden's Mac** (Apple M5 Pro, 24 GB) for all Gemma runs, because:
1. Gemma 4 isn't on the PC, and pulling it there could get a different version under the same name;
2. the pilot ran on the Mac;
3. size comparisons should come from one machine with one set of settings.

Only one model run at a time, and no other project's Ollama jobs on the Mac during a run (runs are scheduled
through Jayden's shared run queue, marked Mac only).

Expected time (an estimate from pilot speed, not measured): about 1.5 hours for all three Gemma models on 20 series
plus a few minutes for the quizzes.

### 5.2 Gemini models (API, through a parent-owned account)
Facts from Google's pages, read Oct 12, 2026:

| | Gemini 3.8 Flash | Gemini 3.1 Pro | Gemma 4 31B (API) |
|---|---|---|---|
| Model ID | `gemini-3.8-flash` | `gemini-3.1-pro-preview` | [to check before its run; logged in section 8] |
| Status | Stable | **Preview** (Google can change it) | |
| Knowledge cutoff | March 2026; some topics only January 2025 | not stated on its card; it says it's based on Gemini 3 Pro, whose cutoff is January 2025 (UNVERIFIED for 3.1 Pro itself) | January 2025 |
| Thinking | can't be turned off: low, medium or high | can't be turned off: low, medium or high (UNVERIFIED for "low"; the helper read it, Claude didn't) | |
| Temperature | **ignored** (Google deprecated it from Gemini 3.6 on) | Google's Gemini 3 guide says keep it at 1.0 (UNVERIFIED wording) | |
| JSON schema replies | supported | supported | |
| Batch API (half price) | supported | supported | not checked |
| Paid price per 1M tokens, batch (input / output incl. thinking) | $0.375 / $1.875 (doubles Jan 1, 2027) | $1.00 / $6.00 | free tier only |

Sources: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash ,
https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview ,
https://deepmind.google/models/model-cards/gemini-3-8-flash/ ,
https://deepmind.google/models/model-cards/gemini-3-1-pro/ ,
https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6 , https://ai.google.dev/gemini-api/docs/pricing .

**What this changes:**
1. Neither Gemini model can run at temperature 0, so D8's rule applies: **3 replies per prompt at the default
   temperature, and the forecast is their average.** That triples the cost and gives slightly noisier replies.
2. Neither can turn thinking off. Gemini runs use the **lowest thinking level, "low"**. So the Gemini models "think"
   before answering and the Gemma models don't. The size curve mixes those two, and the report says so.
3. Pro is a preview model, so the same ID could give different answers later.

**Settings rule (fixed now):** same prompts and first 20 series; thinking "low"; no temperature setting (or 1.0 if
Pro needs one); JSON schema; Batch API; 3 replies per prompt. Any Gemini setting that can only be fixed once the
Gemini code exists gets written into section 8 with a date, **before** any leak analysis on any model.

**Rough cost** (Claude's arithmetic on the batch prices above, not measured): about 14,400 calls per model
(4,799 prompts x 3), each about 200 tokens in (a token is a word-piece; numbers take several). The unknown is how
many hidden thinking tokens "low" uses. Assuming 100 to 1,500 per reply: **Flash about $4 to $42; Pro about $13 to
$135.** Pro alone could pass the $50 cap (D11).

**Headline model (decided by Jayden Oct 12, 2026): Gemini 3.8 Flash.** It's stable, cheaper, and its cutoff covers
every date. Pro becomes an optional extra point on the size curve, only if a small measured cost test shows room
under $50. That way, the headline no longer waits on Pro's cost test. This changes D7, which said the biggest
model (Pro). The headline is now the biggest model we can reliably afford, and Flash is probably smaller than Pro.

**Fallback (fixed now):** if no parent-owned Gemini account is ready by Oct 26, 2026 (the planned start of the full
runs), the headline moves to Gemma 4 12B. The switch is decided and logged in section 8 before any leak analysis.

**Money (D11):** a parent owns the account and the API key (Jayden confirmed Oct 12) and sets a $50 cap. Confirm
the cap is set before the first Gemini call. Before the full run, a small run on the 440 pilot prompts measures the
real cost. Its replies are never compared with real outcomes.

### 5.3 Runs this PREREG covers
Once this file is committed, only the runs below are covered. Each uses the code at the commit named at the top
of this file, the settings in 5.1, and series 0 to 19. Each joins the shared run queue as Mac only. Before and
after each run, `ollama list` must show the model ID from 5.1. Hours are estimates from pilot speed (section 7),
rounded up, not measured on the full run.

| Run | Model | What | Prompts | Command | Output | Hours |
|---|---|---|---|---|---|---|
| R1 | gemma4:e2b | forecasts | 4,700 | `python3 src/run_models.py --model gemma4:e2b` | data/runs/gemma4_e2b/forecasts.jsonl | 0.3 |
| R2 | gemma4:e4b | forecasts | 4,700 | `python3 src/run_models.py --model gemma4:e4b` | data/runs/gemma4_e4b/forecasts.jsonl | 0.5 |
| R3 | gemma4:12b | forecasts | 4,700 | `python3 src/run_models.py --model gemma4:12b` | data/runs/gemma4_12b/forecasts.jsonl | 1.0 |
| R4 | gemma4:e2b | quiz | 99 | `python3 src/run_models.py --model gemma4:e2b --quiz` | data/runs/gemma4_e2b/quiz.jsonl | 0.1 |
| R5 | gemma4:e4b | quiz | 99 | `python3 src/run_models.py --model gemma4:e4b --quiz` | data/runs/gemma4_e4b/quiz.jsonl | 0.1 |
| R6 | gemma4:12b | quiz | 99 | `python3 src/run_models.py --model gemma4:12b --quiz` | data/runs/gemma4_12b/quiz.jsonl | 0.1 |

The pilot timing (section 7) is the timing pilot for R1 to R6. By approving this file, Jayden confirms he has seen
it.

**Not covered: each needs Jayden's own yes in chat.** These use the parent's Gemini account, and the code for them
isn't written yet.

| Run | Model | What | Why it needs his yes |
|---|---|---|---|
| G1 | gemini-3.8-flash | cost test: 440 pilot prompts x 3 replies | costs money |
| G2 | gemini-3.8-flash | forecasts (4,700 x 3) and quiz (99 x 3): the headline run | costs money |
| G3 | gemini-3.1-pro-preview | cost test, then maybe forecasts and quiz | costs money; optional |
| G4 | Gemma 4 31B (API free tier) | forecasts and quiz | new code; model ID not yet checked |

## 6. Analysis

The leak analysis code is written and committed before it touches study data. It's checked on made-up replies with
a **planted leak of known size**: the code must recover the planted slope, and it must find about 0 when nothing is
planted. Pilot replies are never compared with real outcomes. **No model's shifts are compared with real outcomes
until the headline model and all its settings are fixed in this file.**

### 6.1 Which replies count (D8)
1. **First reply per prompt.** If a prompt_id was saved twice, keep the first. For Gemini, keep the first 3. A
   network or server error ("request failed") isn't a model answer: it's retried and not counted.
2. **Parse failures** (anything `src/read_reply.py` can't read as exactly 3 finite numbers) are counted and dropped,
   with **no retries**. The rule: take the first {...} block in the reply. It's the same for every model and label.
3. **Wild forecasts:** if the average of the 3 forecast values is more than **10 noise units** from the formula's
   average, the reply is dropped and counted. A noise unit is the series' noise sd.
4. **Gemini's 3 replies:** rules 2 and 3 apply to each reply on its own. The forecast is the average of the replies
   that survive. If none survive, the prompt counts as failed. Report how many prompts had 1, 2 or 3 good replies.
5. If the neutral reply for a series and level is dropped, that series has no reference for that level. So its
   real-date replies at that level are dropped too, and counted.
6. Report the failure and wild rates per model, per blur level and per date. If a model's failure rate is above 5%,
   also compare the rate on the 27 "fall" dates with the 72 "rise" dates. Failures that cluster on bad-economy dates
   could fake or hide a leak.

### 6.2 The shift and the slope (D6)
- **Shift** for one series, one date and one level =
  (average of the 3 forecast values with the real label - the same with that level's neutral label) / the series'
  noise sd. It's in noise units.
- **Date shift** = the average shift over all the series that have both replies.
- **Main measure (B):** fit a straight line across the 99 dates,

      date shift = a + b x real future change + c x real past change

  The leak is **b**: noise units of forecast shift per percentage point of real 3-month change, holding the past
  change fixed. The past is held fixed because knowing the past is fair: by September 2008, Lehman had already
  collapsed, and downturns tend to keep going. Without this, "knows the past" would look like "knows the future".
- **Measure A**, reported next to B: the same line without the past change.
- **Year level:** the 3 dates in one year share one prompt, so they share one date shift. Each still gets its own
  outcome.

### 6.3 The one headline test (D7)
- **Model:** the headline model (section 5.2, decision (e)), month labels, measure B, all 99 dates.
- **Range:** a bootstrap with 10,000 re-draws of the 99 dates (pick 99 dates at random with replacement, refit,
  repeat), random seed 20261012. The 95% range is the 2.5th to 97.5th percentile of the 10,000 slopes.
- **Leak = yes** if b is above 0 **and** the whole 95% range is above 0. Otherwise, no.
- This is the only yes/no claim. Everything else is described with its range, so we don't have 15 tries at finding
  one lucky result.

### 6.4 Smallest slope worth caring about (decided by Jayden Oct 12, 2026)
**b = 0.03 noise units per percentage point.**
- **The yardstick:** if a model fully "knew" that industry fell 3% over the next 3 months and forecast the series to
  fall the same way, its average month would drop by 1 percentage point. On these series, a full leak works out to
  a slope of about **0.64**. So 0.03 is about **one-twentieth of a full leak**. (The yardstick is our own
  construction, not a standard measure.)
- **In practice:** for September 2008 (real future change -3.1%), 0.03 means a forecast pulled down by about 0.1
  noise units. For January 2020 (-16.3%), about 0.5 noise units.
- **Can the study see it?** Based on the pilot's noise (section 7) and the real outcomes without the COVID dates,
  the noisiest Gemma model on 20 series could reliably detect a slope of about 0.018. That's below 0.03. This is
  an estimate. The Gemini models' noise isn't measured yet.

How the headline result is described, using the 95% range:

| 95% range | Wording |
|---|---|
| all above 0.03 | leak, at least the size worth caring about |
| above 0, but includes 0.03 | leak; unclear whether it is big enough to matter |
| above 0, all below 0.03 | leak, but smaller than worth caring about |
| includes 0, all below 0.03 | no leak worth caring about |
| includes 0, reaches above 0.03 | can't tell |
| all below 0 | the forecast bends *away* from what happened; reported as unexpected |

### 6.5 Checks (D7)
- **Shuffle placebo.** Shuffle which dates the (future, past) outcome pairs go with, 10,000 times. Keep each pair
  together, so the past and future of one date stay matched. Refit each time. The shuffled slopes should center
  near 0. Report the share of shuffled slopes at least as large as the real one.
- **COVID check.** Refit without January 2020 and May 2020, with a bootstrap over the 97 dates.
- **Rank check.** Use ranks instead of values: rank all three variables, take out the past-change ranks from the
  other two, and correlate what's left (a partial Spearman correlation), with a bootstrap range.
- The headline counts as **robust** if the COVID check and the rank check both have 95% ranges above 0. If either
  doesn't, the report says plainly that the leak depends on those two dates or on a few extreme values. The yes/no
  in 6.3 doesn't change.

### 6.6 Blur curve and size curve (descriptive only)
- **Blur:** slope b with its 95% range at month, quarter and year, for each model. For year, the bootstrap re-draws
  whole years (33), not single dates, because the 3 dates in a year share one prompt.
- **Neutral vs none:** the average gap between the "month t" forecast and the no-date forecast. This checks that the
  neutral sentence itself doesn't move the forecast much.
- **Size:** month-level slope b with its range for Gemma 4 E2B, E4B and 12B, then Gemma 4 31B, Gemini Flash and
  Gemini Pro (if run).
- None of these are declared findings; they're described. P2 and P3 get checked against them in words.

### 6.7 Knowledge quiz (D10)
- For Gemini, the quiz answer is the majority of its readable replies out of 3. A 1-to-1 tie (one reply failed), or
  no readable reply, counts as a failure.
- Score each model on balanced accuracy: the average of its accuracy on the 72 rise dates and on the 27 fall dates.
  A model that always says "rise" scores 73% plain accuracy but exactly 50% balanced, so balanced accuracy is the
  fair score. Report plain accuracy too.
- A model **knows** the history if its balanced accuracy's 95% bootstrap range (10,000 re-draws of dates) is above
  50%.
- What the quiz is for: if a model doesn't leak but does know the history, that's "knows but doesn't use it". If it
  doesn't know the history, a no-leak result says little about it.
- Exploratory, labeled as such: the slope on the dates the model answered correctly vs wrongly.

## 7. Pilot (done Oct 12, 2026; pilot data is not study data)

**Pilot rule (fixed before it ran):** every 5th of the 99 dates (20 dates), month labels only, plus "month t" and
no-date, on series 0 to 19. That's 440 prompts per model, on the Mac, with the settings in 5.1.
Summary from `src/pilot_check.py` (re-run Oct 12):

| | Gemma 4 E2B | Gemma 4 E4B | Gemma 4 12B |
|---|---|---|---|
| Replies | 440 | 440 | 440 |
| Parse failures | 0 | 0 | 0 |
| Wild (>10 noise units) | 0 | 0 | 0 |
| Median seconds per prompt | 0.20 | 0.31 | 0.66 |
| First value copied from the series | 33% | 0% | 0% |
| Shifts exactly 0 | 52% | 44% | 18% |
| Spread (sd) of date shifts, averaged over 20 series | 0.032 | 0.028 | 0.069 |

**What the pilot was not used for:** on purpose, nobody compared pilot shifts with what the economy really did. The
pilot measured only noise, failures and speed. That way the leak couldn't shape this plan.

## 8. Deviations

Any change to this file after it's committed gets a dated line here: what changed, why, and whether any study
data had been seen. Changes are logged, never hidden or quietly rewritten.

| Date | What changed | Why | Study data seen? |
|---|---|---|---|
| Oct 12, 2026 | Gemini runs (G1 to G4) reach the same models through **Google Cloud's Gemini platform** (formerly Vertex AI) instead of AI Studio's Gemini API. They're paid from the $300 Google Cloud free-trial credit on a parent-owned account. Same model IDs, prompts, settings (thinking "low", JSON replies, 3 replies per prompt, no temperature setting) and batch jobs. Nothing else changes. | Google excludes AI Studio's Gemini API from the free-trial credit, but its Gemini platform is covered (https://ai.google.dev/gemini-api/docs/billing , https://cloud.google.com/products/gemini-enterprise-agent-platform , read Oct 12, 2026). On the platform, gemini-3.8-flash is generally available, with JSON replies, the low/medium/high thinking settings and batch jobs at the same batch price (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash). Decided by Jayden in chat. | No: no study run has happened. |
| Oct 12, 2026 | Gemini runs (G1 to G4) use **Flex PayGo** instead of batch jobs: ordinary online calls on the platform's global endpoint with Google's Flex request headers (`src/run_gemini.py`). Same model IDs, prompts, settings (thinking "low", JSON replies, 3 replies per prompt, no temperature setting) and price: Google lists Flex and batch at one shared price, half of standard. Each reply records the traffic type Google served. A reply not served as Flex is costed at the standard price. Nothing else changes. | Batch jobs need a command-line Google sign-in and a Cloud Storage bucket. The platform's API keys work only for ordinary calls (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference , https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/authentication). Flex gives ordinary calls the batch price, on the global endpoint only (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/flex-paygo , https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing , all read Oct 12, 2026). A connection test (1 pilot prompt x 3 replies, estimated $0.0027) confirmed that Google served Flex with the key. Decided by Jayden in chat. | No: only the connection test's 3 replies to one pilot prompt, not compared with any real outcome. |

## 9. Decisions made while drafting

| # | Decision | Claude's recommendation | Jayden |
|---|---|---|---|
| (a) | Series on the local models: 20 or 50 | 20, the same as Gemini | **20 (Oct 12)** |
| (b) | Smallest slope worth caring about | 0.03 noise units per point | **0.03 (Oct 12)** |
| (c) | Machine for the Gemma runs | the Mac | **Mac, settings as listed (Oct 12)** |
| (d) | The prediction (section 2) | as drafted | **approved as drafted (Oct 12)** |
| (e) | Headline model, and confirm a parent owns the Gemini account | Gemini 3.8 Flash (Pro optional); Gemma 4 12B if no account by Oct 26 | **Flash, 12B fallback (Oct 12)**; Jayden says a parent already owns the account (Oct 12) |
