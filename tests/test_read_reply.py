"""Tests for src/read_reply.py using fake replies. Run: python3 -m pytest tests  (or python3 tests/test_read_reply.py)

Written by Claude (AI) at Jayden's request (see WORKLOG.md, Oct 12, 2026).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from read_reply import parse_forecast, parse_quiz  # noqa: E402

GOOD = [
    ('{"forecast": [0.1, -0.2, 0.3]}', [0.1, -0.2, 0.3]),
    ('```json\n{"forecast": [1, 2, 3]}\n```', [1.0, 2.0, 3.0]),
    ('Here you go: {"forecast": ["0.1%", "0.2", 0.3]} hope that helps', [0.1, 0.2, 0.3]),
    ('{"forecast":[0.12,0.10,0.11]}', [0.12, 0.10, 0.11]),
]
BAD = [
    ("", "no JSON object found"),
    (None, "empty reply"),
    ("The index will rise.", "no JSON object found"),
    ('{"prediction": [1, 2, 3]}', "no 'forecast' key"),
    ('{"forecast": [1, 2]}', "forecast is not a list of 3"),
    ('{"forecast": [1, 2, 3, 4]}', "forecast is not a list of 3"),
    ('{"forecast": [1, "up", 3]}', "forecast value is not a number"),
    ('{"forecast": [1, true, 3]}', "forecast value is not a number"),
    ('{"forecast": [1, NaN, 3]}', "forecast value is not finite"),
    ('{"forecast": 0.5}', "forecast is not a list of 3"),
]


def test_good_forecasts():
    for text, expected in GOOD:
        parsed, err = parse_forecast(text)
        assert err is None, (text, err)
        assert parsed == expected, (text, parsed)


def test_bad_forecasts():
    for text, reason in BAD:
        parsed, err = parse_forecast(text)
        assert parsed is None, text
        assert err == reason, (text, err)


def test_quiz():
    assert parse_quiz('{"answer": "rise"}') == ("rise", None)
    assert parse_quiz('{"answer": " FALL "}') == ("fall", None)
    assert parse_quiz('{"answer": "flat"}') == (None, "answer is not rise or fall")
    assert parse_quiz("It rose.") == (None, "no JSON object found")


if __name__ == "__main__":
    test_good_forecasts()
    test_bad_forecasts()
    test_quiz()
    print("All reply-reader tests passed.")
