"""Read model replies (PLAN.md, D8).

A forecast reply must be JSON like {"forecast": [x, y, z]} with exactly 3 finite numbers.
A quiz reply must be JSON like {"answer": "rise"} or {"answer": "fall"}.
Anything else is a parse failure. No retries and no guessing: the failure is recorded with its reason.

To be fair to models that wrap JSON in a code fence or add a sentence, we take the first {...} block
in the reply. That rule is the same for every model and every label, and it gets locked in PREREG.md.

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import json
import math
import re

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


def _first_json_object(text: str):
    if text is None:
        return None, "empty reply"
    for match in _JSON_BLOCK.finditer(text):
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError:
            continue
    return None, "no JSON object found"


def parse_forecast(text: str) -> tuple[list[float] | None, str | None]:
    """Return (forecast, None) on success or (None, reason) on failure."""
    obj, err = _first_json_object(text)
    if err:
        return None, err
    if not isinstance(obj, dict) or "forecast" not in obj:
        return None, "no 'forecast' key"
    values = obj["forecast"]
    if not isinstance(values, list) or len(values) != 3:
        return None, "forecast is not a list of 3"
    out = []
    for v in values:
        # bool is a subclass of int in Python, so rule it out explicitly.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                v = float(str(v).strip().rstrip("%"))
            except ValueError:
                return None, "forecast value is not a number"
        if not math.isfinite(v):
            return None, "forecast value is not finite"
        out.append(float(v))
    return out, None


def parse_quiz(text: str) -> tuple[str | None, str | None]:
    obj, err = _first_json_object(text)
    if err:
        return None, err
    if not isinstance(obj, dict) or "answer" not in obj:
        return None, "no 'answer' key"
    answer = str(obj["answer"]).strip().lower()
    if answer not in ("rise", "fall"):
        return None, "answer is not rise or fall"
    return answer, None
