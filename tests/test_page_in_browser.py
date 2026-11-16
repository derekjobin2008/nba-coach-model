"""The scout page in a real (headless) Chrome, at phone and laptop widths, in both themes.

The other page tests run the page's JavaScript against a small fake browser, which cannot see
layout, styles or scrolling. These are regression tests for bugs only a real browser showed:
the picked team box not highlighted, the page scrolling sideways (or, now, a line cut off) on a phone, a shared link
leaving the card below the team grid, and a picked former coach missing from the pinned strip.
They use the page's real data (taken from docs/index.html) with the current template, and skip
when Chrome or Node.js is not installed.
"""

import json
import re
from pathlib import Path

import pytest
from chrome import LAPTOP, PHONE, in_chrome, needs_chrome, open_page

from coachmodel.scout_report import render

ROOT = Path(__file__).resolve().parents[1]
pytestmark = needs_chrome


@pytest.fixture(scope="module")
def page_url(tmp_path_factory):
    """The current template filled with the published page's data, saved as a local file."""
    published = (ROOT / "docs" / "index.html").read_text()
    data = json.loads(re.search(r"^const DATA = (.*);$", published, re.M).group(1))
    path = tmp_path_factory.mktemp("page") / "index.html"
    path.write_text(render(data))
    return path.as_uri(), data


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


def test_changing_the_coach_in_the_address_bar_switches_the_card(page_url, tmp_path):
    url, data = page_url
    current = set(data["current_coaches"].values())
    former = next(coach["name"] for coach in data["coaches"] if coach["name"] not in current)
    [switched] = in_chrome([open_page(url, PHONE, link=data["current_coaches"]["CHI"].replace(" ", "%20")), {"run": f"""
        location.hash = {json.dumps("#" + former.replace(" ", "%20"))};
        await new Promise(resolve => setTimeout(resolve, 300));
        return [document.querySelector('.coach-head h2').textContent,
                {PICKED}.map(box => box.dataset.coach), document.getElementById('picker').className];"""}], tmp_path)
    assert switched == [former, [former], "picker compact"]


@pytest.mark.parametrize("section", ["timeouts-detail", "substitutions-detail"])
def test_every_coach_shows_as_a_visible_dot_in_the_where_he_stands_chart(page_url, tmp_path, section):
    url, data = page_url
    [chart] = in_chrome([open_page(url, PHONE), {"run": f"""
        const dots = [...document.querySelectorAll('#{section} .field-strip .other')].map(dot => dot.getBoundingClientRect());
        const strip = document.querySelector('#{section} .field-strip').getBoundingClientRect();
        let hidden = 0;
        dots.forEach((a, i) => dots.slice(i + 1).forEach(b => {{ if (Math.hypot(a.x - b.x, a.y - b.y) < 7) hidden++; }}));
        const outside = dots.filter(d => d.top < strip.top || d.bottom > strip.bottom || d.left < strip.left - 4 || d.right > strip.right + 4).length;
        return [dots.length, hidden, outside, document.querySelectorAll('#{section} .field-strip .me').length];"""}], tmp_path)
    assert chart == [len(data["coaches"]), 0, 0, 1]
