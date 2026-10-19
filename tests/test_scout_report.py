"""The scout page: data is embedded safely, and the page's own JavaScript, run against a
tiny fake browser, shows the sentences a classmate reads."""

import json
import math
import os
import re
import shutil
import subprocess

import pytest

from coachmodel.scout_report import render, without_missing

FAKE_BROWSER = """
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this._text = ""; this.listeners = {}; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children = []; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  set textContent(value) { this._text = value; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(" "); }
}
const nodes = {"data-line": new Node("p"), "coach": new Node("select"), "card": new Node("article")};
globalThis.document = {createElement: tag => new Node(tag), getElementById: id => nodes[id]};
globalThis.history = {replaceState() {}};
globalThis.location = {hash: process.env.PAGE_HASH || ""};
"""


def coach(name, timeout_ratio, foul_trouble):
    return {"name": name, "games": 300,
            "teams": [{"team": "DEN", "games": 300, "seasons": "2021-22 to 2024-25",
                       "timeout_ratio": timeout_ratio, "substitution_ratio": None}],
            "timeouts": {"ratio": timeout_ratio, "low": timeout_ratio - 0.06, "high": timeout_ratio + 0.06, "decisions": 900},
            "substitutions": {"ratio": 0.9, "low": 0.85, "high": 0.95, "decisions": 5000},
            "foul_trouble": foul_trouble}


DATA = {
    "seasons": "2021-22 to 2025-26", "games": 6150, "audited_coaches": 25,
    "league_foul_trouble": {"Q1 2 fouls": 0.2, "Q3 4 fouls": 0.23, "Q4 5 fouls": 0.66},
    "coaches": [
        coach("Michael Malone", 1.53, {"close_games": 142, "q1_2_fouls": 0.08, "q3_4_fouls": 0.09, "q4_5_fouls": 0.86,
                                       "cost_per_starter_game": -0.00114, "low": -0.0015, "high": -0.0006,
                                       "season_wins": -0.47, "value_over_bench": 6.7, "cost_rank": 1}),
        coach("Wes Unseld Jr.", 0.71, None),
    ],
}


def run_page(html: str, hash_value: str = "") -> str:
    script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    result = subprocess.run(["node", "-e", FAKE_BROWSER + script + "\nconsole.log(nodes.card.textContent)"],
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


@needs_node
def test_link_to_a_coach_and_missing_audit_message():
    card = run_page(render(DATA), "#Wes%20Unseld%20Jr.")
    assert "Makes 29% fewer timeout calls" in card
    assert "Not enough close games to audit" in card
    assert "NaN" not in card
