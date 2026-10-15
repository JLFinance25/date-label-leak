# Data Sources

One entry per download or dataset: link, date, what it is, and its license or terms. Files here are never edited by hand.

The NBER business-cycle dates and one official monthly activity index will be logged here with their terms, plus any model downloads.

## INDPRO.csv (downloaded Oct 12, 2026 by Claude, with Jayden's OK)
- What: US Industrial Production, Total Index. Monthly, seasonally adjusted, index 2017=100. Jan 1919 to Aug 2026
  (1,292 months). Source: Board of Governors of the Federal Reserve System, G.17 release, via FRED.
- Link: https://fred.stlouisfed.org/series/INDPRO (file: https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO)
- Vintage: FRED's current data as of Oct 12, 2026 (last updated Sep 18, 2026; next release Oct 16, 2026). These are today's
  revised numbers, not first-published ones (plan D4).
- Terms: FRED tags this series "Public Domain: Citation Requested". Citation: Board of Governors of the Federal Reserve
  System (US), Industrial Production: Total Index [INDPRO], retrieved from FRED, Federal Reserve Bank of St. Louis;
  https://fred.stlouisfed.org/series/INDPRO

## NBER_business_cycle_dates.xlsx (downloaded Oct 12, 2026 by Claude, with Jayden's OK)
- What: NBER's official US business-cycle peak and trough dates (the Business Cycle Dating Committee table).
- Link: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
  (file: https://www.nber.org/sites/default/files/2023-03/BCDC_spreadsheet_for_website.xlsx, renamed on save)
- Terms: the page says "Permission to copy is granted, provided attribution of source is given." Cite as "Source: NBER".


## Gemma 4 models via Ollama (downloaded Oct 12, 2026 by Claude, with Jayden's OK; stored by Ollama, not in this folder)
- What: `gemma4:e2b` (Ollama ID 45d11505b91d, 7.5 GB), `gemma4:e4b` (eb2177adf764, 9.5 GB), `gemma4:12b`
  (c7597fc90b86, 7.7 GB). All are nvfp4 quantization, Ollama 0.40.3, on Jayden's Mac (M5 Pro, 24 GB).
  `ollama show` reports total parameters as 5.2B, 8.1B and 12.4B. "E2B" and "E4B" are Google's "effective" sizes.
- Link: https://ollama.com/library/gemma4 ; Google docs: https://ai.google.dev/gemma/docs/core
- License: `ollama show --license` prints the Apache License.
- Defaults to change in the study code: **thinking is ON by default** and **temperature defaults to 1**. Plan D8 needs
  temperature 0, and thinking should be off (or the choice locked in PREREG).
