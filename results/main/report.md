# Leak analysis (PREREG.md section 6)

Code commit: 0182df1077b64dd683a695c21b4133938a7e36b0. Bootstrap: 10000 draws, seed 20261012.
Slopes are noise units of forecast shift per percentage point of real 3-month change, with 95% ranges.

## Headline test (6.3)

Model gemini-3.8-flash, month labels, measure B, 99 dates: b = +0.1135 [+0.0831, +0.2135].
Leak: **yes**. Wording (6.4): leak, at least the size worth caring about.
Robust (6.5): yes. COVID check: +0.1812 [+0.1168, +0.2372]; rank check: rho = +0.653 [+0.503, +0.765]; placebo mean -0.0001, share at least as large 0.0000.

## Size curve (6.6, descriptive)

| Model | month b [95% range] |
|---|---|
| gemma4:e2b | +0.0015 [-0.0039, +0.0046] |
| gemma4:e4b | +0.0011 [-0.0037, +0.0028] |
| gemma4:12b | +0.0039 [-0.0054, +0.0199] |
| gemini-3.8-flash | +0.1135 [+0.0831, +0.2135] |

## Each model

### gemma4:e2b
Replies 4700: parse failures 0 (0.00%), wild 0 (0.00%). Prompts by good replies: {'0': 0, '1': 4700}.

| Level | B: b [range] | A: b [range] | no reference |
|---|---|---|---|
| month | +0.0015 [-0.0039, +0.0046] | +0.0011 [-0.0049, +0.0045] | 0 |
| quarter | +0.0027 [-0.0022, +0.0048] | +0.0023 [-0.0002, +0.0051] | 0 |
| year | +0.0001 [-0.0086, +0.0048] | -0.0001 [-0.0087, +0.0053] | 0 |

'month t' minus no date: mean -0.0359, mean size 0.1575 noise units.
Checks (descriptive unless this is the headline model): COVID +0.0004 [-0.0049, +0.0057]; rank rho -0.004 [-0.202, +0.196]; placebo mean -0.0000.
Quiz (failures dropped): balanced 0.521 [0.434, 0.616], plain 0.657, knows: False.
Quiz (failures wrong): balanced 0.521 [0.434, 0.616], plain 0.657, knows: False.

### gemma4:e4b
Replies 4700: parse failures 0 (0.00%), wild 0 (0.00%). Prompts by good replies: {'0': 0, '1': 4700}.

| Level | B: b [range] | A: b [range] | no reference |
|---|---|---|---|
| month | +0.0011 [-0.0037, +0.0028] | +0.0008 [-0.0036, +0.0031] | 0 |
| quarter | -0.0003 [-0.0013, +0.0022] | -0.0005 [-0.0012, +0.0022] | 0 |
| year | +0.0012 [+0.0001, +0.0043] | +0.0010 [+0.0001, +0.0057] | 0 |

'month t' minus no date: mean +0.2432, mean size 0.3247 noise units.
Checks (descriptive unless this is the headline model): COVID -0.0005 [-0.0050, +0.0034]; rank rho -0.105 [-0.321, +0.115]; placebo mean -0.0000.
Quiz (failures dropped): balanced 0.546 [0.477, 0.610], plain 0.374, knows: False.
Quiz (failures wrong): balanced 0.546 [0.477, 0.610], plain 0.374, knows: False.

### gemma4:12b
Replies 4700: parse failures 0 (0.00%), wild 0 (0.00%). Prompts by good replies: {'0': 0, '1': 4700}.

| Level | B: b [range] | A: b [range] | no reference |
|---|---|---|---|
| month | +0.0039 [-0.0054, +0.0199] | +0.0025 [-0.0033, +0.0214] | 0 |
| quarter | +0.0013 [-0.0052, +0.0139] | +0.0010 [-0.0041, +0.0136] | 0 |
| year | +0.0011 [-0.0041, +0.0105] | +0.0010 [-0.0033, +0.0138] | 0 |

'month t' minus no date: mean +0.0223, mean size 0.2646 noise units.
Checks (descriptive unless this is the headline model): COVID +0.0027 [-0.0093, +0.0130]; rank rho -0.006 [-0.216, +0.198]; placebo mean -0.0000.
Quiz (failures dropped): balanced 0.484 [0.424, 0.532], plain 0.283, knows: False.
Quiz (failures wrong): balanced 0.484 [0.424, 0.532], plain 0.283, knows: False.

### gemini-3.8-flash
Replies 14100: parse failures 0 (0.00%), wild 6 (0.04%). Prompts by good replies: {'0': 0, '1': 1, '2': 4, '3': 4695}.

| Level | B: b [range] | A: b [range] | no reference |
|---|---|---|---|
| month | +0.1135 [+0.0831, +0.2135] | +0.1073 [+0.0818, +0.2350] | 0 |
| quarter | +0.0791 [+0.0486, +0.1786] | +0.0749 [+0.0502, +0.1923] | 0 |
| year | +0.0107 [-0.0021, +0.0617] | +0.0095 [-0.0014, +0.0708] | 0 |

'month t' minus no date: mean +0.0261, mean size 0.0719 noise units.
Checks (descriptive unless this is the headline model): COVID +0.1812 [+0.1168, +0.2372]; rank rho +0.653 [+0.503, +0.765]; placebo mean -0.0001.
Quiz (failures dropped): balanced 0.938 [0.871, 0.993], plain 0.960, knows: True.
Quiz (failures wrong): balanced 0.938 [0.871, 0.993], plain 0.960, knows: True.
