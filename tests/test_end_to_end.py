"""End to end: saved play-by-play -> SQL pipeline -> model fits -> page -> a click in Chrome.

Every other test checks one piece with stand-ins around it. This one runs the whole path a
visitor depends on with nothing faked in between, on the three fixture games, and checks that
the card shows what the models computed. It catches one step handing the next stale, renamed
or flipped data; it cannot say whether the model is right (the held-out checks do that) or
whether people understand the page (a classmate does that).

With three games the fits are weak but fast, and no coach has enough close games for the
foul-trouble audit or the likely-moves card, so the page's "not enough" states are checked too.
Writing this test found three bugs that full data had hidden: a team seen only on the road had
no name (the model fit crashed), the audit crashed instead of auditing nobody, and the page
builder read player names from wherever it was run.
"""

from pathlib import Path

import pytest
from chrome import PHONE, in_chrome, needs_chrome, open_page

from coachmodel import scout_report

FIXTURES = Path(__file__).parent / "fixtures" / "data"
pytestmark = needs_chrome


@pytest.fixture(scope="module")
def built_page(database, tmp_path_factory):
    data = scout_report.compute(database, min_games=1, players_path=FIXTURES / "raw" / "players.json")
    data["current_coaches"] = scout_report.current_coaches(FIXTURES / "head_coaches.csv")
    path = tmp_path_factory.mktemp("end_to_end") / "index.html"
    path.write_text(scout_report.render(data))
    return path.as_uri(), data


def more_or_fewer(ratio: float, noun: str) -> str:
    """The card's headline, written independently of the page's JavaScript."""
    change = round(abs(ratio - 1) * 100)
    if change == 0:
        return f"Makes about as many {noun} as an average coach in the same situations."
    return f"Makes {change}% {'more' if ratio > 1 else 'fewer'} {noun} than an average coach in the same situations."


def test_the_pipeline_puts_the_fixture_coaches_on_the_page(built_page):
    _, data = built_page
    # The one regular-season fixture game is Pacers at Knicks (2024-25).
    assert sorted(coach["name"] for coach in data["coaches"]) == ["Rick Carlisle", "Tom Thibodeau"]
    assert data["games"] == 1
    for coach in data["coaches"]:
        assert 0 < coach["timeouts"]["low"] < coach["timeouts"]["ratio"] < coach["timeouts"]["high"]
        assert 0 < coach["substitutions"]["low"] < coach["substitutions"]["ratio"] < coach["substitutions"]["high"]


def test_clicking_each_team_shows_the_numbers_the_models_computed(built_page, database, tmp_path):
    url, data = built_page
    # The models' own numbers, fit again here, so a mix-up anywhere in the page builder shows.
    model_ratio = {task: scout_report.task_summary(database, task)[0]["ratio"] for task in ("timeout", "substitution")}
    runs = [open_page(url, PHONE)]
    for coach in data["coaches"]:
        runs.append({"run": f"""
            document.querySelector('.coach-box[data-coach="{coach['name']}"]').click();
            const text = id => document.getElementById(id)?.querySelector('.headline')?.textContent;
            return {{name: document.querySelector('.coach-head h2').textContent,
                    timeouts: text('timeouts-detail'), substitutions: text('substitutions-detail'),
                    card: document.getElementById('card').textContent}};"""})
    shown = in_chrome(runs, tmp_path)
    for coach, card in zip(data["coaches"], shown):
        assert card["name"] == coach["name"]
        assert card["timeouts"] == more_or_fewer(model_ratio["timeout"][coach["name"]], "timeout calls")
        assert card["substitutions"] == more_or_fewer(model_ratio["substitution"][coach["name"]], "substitutions")
        # Too little data for these two: the page says so instead of breaking.
        assert "Not enough close-game stoppages to rank his moves." in card["card"]
        assert "Not enough close games to audit his foul-trouble decisions" in card["card"]
