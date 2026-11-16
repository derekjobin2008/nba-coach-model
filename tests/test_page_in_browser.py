"""The scout page in a real (headless) Chrome, at phone and laptop widths, in both themes.

The other page tests run the page's JavaScript against a small fake browser, which cannot see
layout, styles or scrolling. These are regression tests for bugs only a real browser showed:
the picked team box not highlighted, the page scrolling sideways (or, now, a line cut off) on a phone, a shared link
leaving the card below the team grid, and a picked former coach missing from the pinned strip.
They use the page's real data (taken from docs/index.html) with the current template, and skip
when Chrome or Node.js is not installed.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from coachmodel.scout_report import render

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).with_name("chrome_driver.mjs")
PHONE = {"width": 375, "height": 812}
LAPTOP = {"width": 1280, "height": 800}


def find_chrome() -> str | None:
    candidates = [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]
    return next((path for path in candidates if path and Path(path).exists()), None)


CHROME = find_chrome()
pytestmark = pytest.mark.skipif(CHROME is None or shutil.which("node") is None,
                                reason="needs Google Chrome (or CHROME=path) and Node.js")


@pytest.fixture(scope="module")
def page_url(tmp_path_factory):
    """The current template filled with the published page's data, saved as a local file."""
    published = (ROOT / "docs" / "index.html").read_text()
    data = json.loads(re.search(r"^const DATA = (.*);$", published, re.M).group(1))
    path = tmp_path_factory.mktemp("page") / "index.html"
    path.write_text(render(data))
    return path.as_uri(), data


def in_chrome(steps: list[dict], tmp_path: Path) -> list:
    """Run the steps in Chrome; returns each "run" step's value. Fails on any error in the page."""
    steps_path = tmp_path / "steps.json"
    steps_path.write_text(json.dumps(steps))
    result = subprocess.run(["node", str(DRIVER), CHROME, str(steps_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    *values, errors = json.loads(result.stdout)
    assert errors["pageErrors"] == []
    return values


def open_page(url: str, size: dict, dark: bool = False, link: str = "") -> dict:
    return {"open": url + (f"#{link}" if link else ""), **size, "dark": dark}


SHOWN = "const shown = node => { const box = node.getBoundingClientRect(); return box.width > 0 && box.height > 0; };"
PICKED = "[...document.querySelectorAll('.coach-box')].filter(box => box.getAttribute('aria-pressed') === 'true')"


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_no_coach_scrolls_sideways_or_gets_cut_off_on_a_phone(page_url, tmp_path, dark):
    url, data = page_url
    # Click through every coach. A line too wide for a phone either makes the page scroll
    # sideways or, since the card hides what overflows it, gets cut off at the card's edge.
    [problems] = in_chrome([open_page(url, PHONE, dark), {"run": """
        const problems = [];
        for (const box of document.querySelectorAll('.coach-box')) {
          box.click();
          await new Promise(resolve => setTimeout(resolve, 20));
          if (document.documentElement.scrollWidth > innerWidth) problems.push(`${box.dataset.coach}: page scrolls sideways`);
          const card = document.getElementById('card').getBoundingClientRect();
          for (const node of document.querySelectorAll('#card *')) {
            const edge = node.getBoundingClientRect();
            if (edge.width && (edge.right > card.right + 1 || edge.left < card.left - 1)) {
              problems.push(`${box.dataset.coach}: cut off: ${node.className || node.tagName} "${node.textContent.slice(0, 40)}"`);
              break;
            }
          }
        }
        return [document.querySelectorAll('.coach-box').length, problems];"""}], tmp_path)
    assert problems == [len(data["coaches"]), []]


def test_clicking_a_team_box_highlights_that_box_only(page_url, tmp_path):
    url, data = page_url
    team = sorted(data["current_coaches"])[3]
    name = data["current_coaches"][team]
    [picked] = in_chrome([open_page(url, LAPTOP), {"run": f"""
        document.querySelector('.coach-box[data-coach="{name}"]').click();
        const picked = {PICKED};
        return [picked.map(box => box.dataset.coach), getComputedStyle(picked[0]).borderTopWidth,
                document.querySelector('.coach-head h2').textContent];"""}], tmp_path)
    assert picked == [[name], "2px", name]


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_a_shared_link_lands_on_the_card_with_the_strip_pinned_above(page_url, tmp_path, dark):
    url, data = page_url
    name = data["current_coaches"]["CHI"]
    [landed] = in_chrome([open_page(url, PHONE, dark, link=name.replace(" ", "%20")), {"run": f"""
        {SHOWN}
        await new Promise(resolve => setTimeout(resolve, 700));
        const picker = document.getElementById('picker').getBoundingClientRect();
        const card = document.getElementById('card').getBoundingClientRect();
        const picked = {PICKED};
        return {{picker: document.getElementById('picker').className, pickerTop: picker.top,
                cardOnScreen: card.top >= picker.bottom - 1 && card.top < innerHeight / 2,
                picked: picked.map(box => box.dataset.coach), pickedShown: picked.every(shown)}};"""}], tmp_path)
    assert landed == {"picker": "picker compact", "pickerTop": 0, "cardOnScreen": True,
                      "picked": [name], "pickedShown": True}


def test_a_picked_former_coach_shows_in_the_strip_and_nowhere_else(page_url, tmp_path):
    url, data = page_url
    current = set(data["current_coaches"].values())
    former = next(coach["name"] for coach in data["coaches"] if coach["name"] not in current)
    strip = f"""
        {SHOWN}
        await new Promise(resolve => setTimeout(resolve, 100));
        return {{former: [...document.querySelectorAll('#former-coaches .coach-box')].filter(shown).map(box => box.dataset.coach),
                heading: shown(document.getElementById('former-summary'))}};"""
    [former_picked, current_picked, all_shown] = in_chrome([
        open_page(url, PHONE, link=former.replace(" ", "%20")), {"run": strip},
        open_page(url, PHONE, link=data["current_coaches"]["CHI"].replace(" ", "%20")), {"run": strip},
        {"run": "document.getElementById('show-all').click();" + strip},
    ], tmp_path)
    assert former_picked == {"former": [former], "heading": False}
    assert current_picked == {"former": [], "heading": False}
    assert all_shown["heading"] is True  # "Show all" brings back the former coaches' heading
