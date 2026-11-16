"""The scout page: data is embedded safely, and the page's own JavaScript, run against a
tiny fake browser, shows the sentences a classmate reads."""

import json
import math
import os
import re
import shutil
import subprocess

import pandas as pd
import pytest

from coachmodel import scout_report
from coachmodel.scout_report import latest_close_game, render, without_missing

FAKE_BROWSER = """
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this._text = ""; this.listeners = {}; }
  // Like a real browser: values become strings, and node.attributes[name] is an Attr object, not
  // the value. (Code that compared node.attributes["data-coach"] to a name passed here while it
  // was broken in every real browser.)
  setAttribute(key, value) { this.attributes[key] = {name: key, value: String(value)}; }
  getAttribute(key) { return key in this.attributes ? this.attributes[key].value : null; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children = []; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  scrollIntoView() { scrolledTo.push(this); }
  set textContent(value) { this._text = value; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(" "); }
}
const scrolledTo = [];
const nodes = {"data-line": new Node("p"), "current-coaches": new Node("div"), "former-coaches": new Node("div"),
               "former-group": new Node("details"), "former-summary": new Node("summary"),
               "picker": new Node("div"), "show-all": new Node("button"),
               "card": new Node("article")};
nodes.picker.setAttribute("class", "picker");  // as in the page's HTML
function clickCoach(name) {
  [...nodes["current-coaches"].children, ...nodes["former-coaches"].children]
    .find(box => box.getAttribute("data-coach") === name).listeners.click();
}
globalThis.document = {createElement: tag => new Node(tag), getElementById: id => nodes[id]};
globalThis.history = {replaceState() {}};
globalThis.location = {hash: process.env.PAGE_HASH || ""};
"""


def coach(name, timeout_ratio, foul_trouble, likely_moves=None, team="DEN", seasons="2021-22 to 2024-25", games=300):
    return {"name": name, "games": games,
            "teams": [{"team": team, "games": games, "seasons": seasons,
                       "timeout_ratio": timeout_ratio, "substitution_ratio": None}],
            "timeouts": {"ratio": timeout_ratio, "low": timeout_ratio - 0.3, "high": timeout_ratio + 0.3,
                         "tendency_low": timeout_ratio - 0.06, "tendency_high": timeout_ratio + 0.06, "decisions": 900},
            "substitutions": {"ratio": 0.9, "low": 0.6, "high": 1.2, "tendency_low": 0.95, "tendency_high": 1.05,
                              "decisions": 5000},
            "likely_moves": likely_moves,
            "foul_trouble": foul_trouble}


MALONE_MOVES = {"season": "2024-25", "stoppages": 412, "moves": [
    {"label": "Call a timeout", "kind": "timeout", "probability": 0.21, "low": 0.17, "high": 0.26,
     "league": 0.15, "differs": True},
    {"label": "Take out Aaron Gordon", "kind": "substitution", "probability": 0.12, "low": 0.09, "high": 0.15,
     "league": 0.11, "differs": False},
    {"label": "Take out Russell Westbrook", "kind": "substitution", "probability": 0.004, "low": 0.003,
     "high": 0.006, "league": 0.004, "differs": False},
]}


DATA = {
    "seasons": "2021-22 to 2025-26", "games": 6150, "audited_coaches": 25,
    "close_game": {"max_margin": 5, "max_seconds_left": 360},
    "league_foul_trouble": {"Q1 2 fouls": 0.2, "Q3 4 fouls": 0.23, "Q4 5 fouls": 0.66},
    "coaches": [
        coach("Michael Malone", 1.53, {"close_games": 142, "q1_2_fouls": 0.08, "q3_4_fouls": 0.09, "q4_5_fouls": 0.86,
                                       "cost_per_starter_game": -0.00114, "low": -0.0015, "high": -0.0006,
                                       "season_wins": -0.47, "value_over_bench": 6.7, "cost_rank": 1},
              MALONE_MOVES),
        coach("Wes Unseld Jr.", 0.71, None),
        coach("Joe Mazzulla", 1.0, None, team="BOS", seasons="2022-23 to 2025-26", games=250),
    ],
    "current_coaches": {"BOS": "Joe Mazzulla"},
}


def run_page(html: str, hash_value: str = "", then: str = "console.log(nodes.card.textContent)") -> str:
    script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    result = subprocess.run(["node", "-e", FAKE_BROWSER + script + "\n" + then],
                            capture_output=True, text=True, check=True, env={**os.environ, "PAGE_HASH": hash_value})
    return result.stdout


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")


def test_missing_numbers_become_null_and_script_tags_cannot_break_out():
    data = {"coaches": [{"name": "</script><b>", "ratio": math.nan}]}
    html = render(data)
    assert "__SCOUT_DATA__" not in html
    assert "</script><b>" not in html
    assert without_missing(data)["coaches"][0]["ratio"] is None
    assert json.loads(json.dumps(without_missing(data), allow_nan=False))


@needs_node
def test_opens_on_the_coach_with_most_games_and_answers_both_questions():
    card = run_page(render(DATA))
    assert "Michael Malone" in card
    assert "Makes 53% more timeout calls than an average coach in the same situations." in card
    assert "His foul-trouble benching costs his team about 0.47 wins a season." in card
    # "Clearly different" follows the usual-habit range, not the wider one-season range.
    assert "Clearly more than average · the most of 3 coaches" in card  # timeouts
    assert "Not clearly different from average" in card  # substitutions: usual habit crosses average


@needs_node
def test_link_to_a_coach_and_missing_audit_message():
    card = run_page(render(DATA), "#Wes%20Unseld%20Jr.")
    assert "Makes 29% fewer timeout calls" in card
    assert "Not enough close games to audit" in card
    assert "Not enough close-game stoppages to rank his moves." in card


@needs_node
def test_likely_moves_lead_the_card_and_flag_only_coaches_who_differ():
    card = run_page(render(DATA))
    assert card.index("Most likely next moves") < card.index("Makes 53% more timeout calls")
    assert "within 5 points, last 6 minutes of the 4th" in card
    assert "Call a timeout" in card and "21%" in card and "Range 17% to 26% · average coach 15%" in card
    assert card.count("often than the league") == 1  # only the timeout move, where his habit clearly differs
    assert "more often than the league" in card
    assert "Range 0.3% to 0.6%" in card  # small chances keep a decimal instead of rounding to 0%
    assert "NaN" not in card


def test_close_game_uses_the_two_settings_and_each_coachs_latest_season(monkeypatch):
    rows = pd.DataFrame({
        "coach_name": ["A", "A", "A", "A", "A", "B"],
        "season": ["2025-26", "2025-26", "2025-26", "2025-26", "2024-25", "2024-25"],
        "is_fourth_or_overtime": [1, 1, 1, 1, 1, 1], "is_overtime": [0, 0, 0, 1, 0, 0],
        "period_seconds_left": [300, 400, 300, 100, 300, 300],
        "abs_score_margin": [5, 2, 6, 0, 1, 3],
    })
    # close; too early; too far apart; overtime; A's older season; B's latest season
    assert latest_close_game(rows).tolist() == [True, False, False, False, False, True]
    monkeypatch.setattr(scout_report, "CLOSE_GAME_MAX_MARGIN", 6)
    assert latest_close_game(rows).tolist()[2]


@needs_node
def test_coaches_are_picked_by_team_with_former_coaches_kept_apart():
    picker = run_page(render(DATA), then='console.log(nodes["current-coaches"].textContent + " | " + '
                                         'nodes["former-coaches"].textContent)')
    current, former = picker.split(" | ")
    assert "Boston Celtics Joe Mazzulla · 250 games" in current
    assert "Malone" not in current
    assert "Denver Nuggets Michael Malone · last coached 2024-25" in former
    assert "Mazzulla" not in former


@needs_node
def test_clicking_a_team_box_opens_its_coach_and_marks_the_box():
    output = run_page(render(DATA), then='clickCoach("Joe Mazzulla"); console.log(nodes.card.textContent); '
                      'console.log(nodes["current-coaches"].children[0].getAttribute("aria-pressed"))')
    assert output.startswith("Joe Mazzulla")
    assert output.strip().endswith("true")


def test_current_coach_is_the_one_hired_last_in_the_latest_season(tmp_path):
    table = tmp_path / "head_coaches.csv"
    table.write_text("season,team,coach_name,from_date,note\n"
                     "2024-25,POR,Old Coach,2024-10-01,opening coach\n"
                     "2025-26,POR,Chauncey Billups,2025-10-01,opening coach\n"
                     "2025-26,POR,Tiago Splitter,2025-10-23,interim\n"
                     "2025-26,BOS,Joe Mazzulla,2025-10-01,opening coach\n")
    assert scout_report.current_coaches(table) == {"BOS": "Joe Mazzulla", "POR": "Tiago Splitter"}


def card_with_moves(*moves):
    """The page's card for one coach whose likely moves are (probability, league, differs)."""
    data = {**DATA, "coaches": [coach("Joe Mazzulla", 1.0, None, team="BOS", seasons="2025-26", likely_moves={
        "season": "2025-26", "stoppages": 300, "moves": [
            {"label": f"Move {index}", "kind": "timeout", "probability": probability, "low": probability - 0.02,
             "high": probability + 0.02, "league": league, "differs": differs}
            for index, (probability, league, differs) in enumerate(moves, start=1)]})]}
    return run_page(render(data))


@needs_node
def test_a_gap_under_2_points_reads_about_the_same_as_the_league():
    card = card_with_moves((0.090, 0.099, True))  # 0.9 points below the league
    assert "about the same as the league" in card
    assert "less often than the league" not in card


@needs_node
def test_a_gap_of_2_points_or_more_keeps_its_direction():
    card = card_with_moves((0.09, 0.15, True), (0.12, 0.10, True))  # 6 points below; exactly 2 above
    assert "less often than the league" in card
    assert "more often than the league" in card
    assert "about the same as the league" not in card


@needs_node
def test_moves_where_the_coach_does_not_clearly_differ_stay_untagged():
    card = card_with_moves((0.09, 0.15, False))
    assert "than the league" not in card and "as the league" not in card


@needs_node
def test_former_coaches_start_hidden_behind_a_heading_with_their_count():
    output = run_page(render(DATA), "#Joe%20Mazzulla",
                      then='console.log(nodes["former-summary"].textContent + " | " + Boolean(nodes["former-group"].open))')
    assert output.strip() == "Former coaches (2) | false"


@needs_node
def test_a_link_to_a_former_coach_opens_the_former_coaches_section():
    output = run_page(render(DATA), "#Michael%20Malone", then='console.log(Boolean(nodes["former-group"].open))')
    assert output.strip() == "true"


@needs_node
def test_report_opens_with_full_team_name_and_three_stat_tiles():
    card = run_page(render(DATA))
    assert card.startswith("Michael Malone Denver Nuggets 2021-22 to 2024-25 · 300 games")
    assert "Timeouts +53% more timeouts than an average coach A real habit" in card
    assert "Subs −10% fewer subs than an average coach Within normal range" in card
    assert "Foul trouble −0.47 wins a season from benching players in foul trouble A real effect" in card


@needs_node
def test_moves_show_this_coach_next_to_an_average_coach():
    card = run_page(render(DATA))
    assert "This coach" in card and "Average coach" in card
    assert "What do the ranges mean?" in card


@needs_node
def test_picking_a_team_shrinks_the_grid_to_a_strip_and_show_all_brings_it_back():
    output = run_page(render(DATA), then='console.log(nodes.picker.getAttribute("class")); clickCoach("Joe Mazzulla"); '
                      'console.log(nodes.picker.getAttribute("class")); nodes["show-all"].listeners.click(); '
                      'console.log(nodes.picker.getAttribute("class"))')
    assert output.split("\n")[:3] == ["picker", "picker compact", "picker"]


@needs_node
@pytest.mark.parametrize("link", ["#Joe%20Mazzulla", "#Wes%20Unseld%20Jr."])  # a current and a former coach
def test_a_shared_link_scrolls_to_the_card_and_shrinks_the_grid_like_a_tap(link):
    output = run_page(render(DATA), link, then='console.log(scrolledTo.includes(nodes.card)); '
                      'console.log(nodes.picker.getAttribute("class"))')
    assert output.split("\n")[:2] == ["true", "picker compact"]


@needs_node
def test_opening_the_page_without_a_link_stays_on_the_full_grid():
    output = run_page(render(DATA), then='console.log(scrolledTo.length); console.log(nodes.picker.getAttribute("class"))')
    assert output.split("\n")[:2] == ["0", "picker"]
