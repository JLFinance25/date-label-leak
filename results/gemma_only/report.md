# Leak analysis (PREREG.md section 6)

Code commit: 3ceabb05f479986cb051924ae703feba0c6aea14. Bootstrap: 10000 draws, seed 20261012.
Slopes are noise units of forecast shift per percentage point of real 3-month change, with 95% ranges.

## Headline test (6.3)

No forecasts for the headline model in this folder, so the headline test wasn't run.

## Size curve (6.6, descriptive)

| Model | month b [95% range] |
|---|---|
| gemma4:e2b | +0.0015 [-0.0039, +0.0046] |
| gemma4:e4b | +0.0011 [-0.0037, +0.0028] |
| gemma4:12b | +0.0039 [-0.0054, +0.0199] |

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
