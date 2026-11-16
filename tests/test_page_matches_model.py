"""The published page shows what the current model computes on the full data.

docs/index.html carries a copy of every coach's numbers from the last full refit (about 40
minutes). If the model code or the data changes and the page is not rebuilt, the page shows
stale numbers while every other test passes. This refits the timeout model (about a minute)
and compares every coach on the page; substitutions take an hour, so they are left to the
benchmark. Skipped on a fresh clone until the data is downloaded and built.
"""

import json
import re
from pathlib import Path

import duckdb
import pytest

from coachmodel.scout_report import task_summary

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "coach.duckdb"

pytestmark = pytest.mark.skipif(not DATABASE_PATH.exists(), reason="full database not built")


def test_every_coachs_timeout_numbers_on_the_page_match_a_fresh_fit():
    published = (ROOT / "docs" / "index.html").read_text()
    page = json.loads(re.search(r"^const DATA = (.*);$", published, re.M).group(1))
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        fresh, _, _ = task_summary(connection, "timeout")
    stale = []
    for coach in page["coaches"]:
        for key in ("ratio", "low", "high", "tendency_low", "tendency_high"):
            shown, computed = coach["timeouts"][key], fresh.loc[coach["name"], key]
            if abs(shown - computed) > 0.0005:  # a visitor sees whole percents; this is far finer
                stale.append(f"{coach['name']} {key}: page {shown:.4f}, model {computed:.4f}")
    assert len(page["coaches"]) > 40
    assert stale == [], "rebuild the page (python -m coachmodel.scout_report):\n" + "\n".join(stale[:10])
