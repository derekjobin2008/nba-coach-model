"""The scout page's colors stay readable in both themes.

Regression tests for two reported bugs: the dark-mode "more/less often than the league" tag
fell to a 4.38 contrast ratio (blue text on its dark blue pill), and the "This coach" bar
vanished in dark mode on dark team colors (navy on navy). Contrast is the WCAG ratio: at least
4.5 for text, at least 3 for a shape such as a bar.
"""

import re

import pytest

from coachmodel.scout_report import TEMPLATE_PATH

TEMPLATE = TEMPLATE_PATH.read_text()
TEXT_MINIMUM = 4.5
SHAPE_MINIMUM = 3.0


def variables(block_start: str) -> dict[str, str]:
    """The --name: #hex colors declared in the CSS block that starts with block_start."""
    start = TEMPLATE.index(block_start)
    block = TEMPLATE[start:TEMPLATE.index("}", start)]
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\b", block))


LIGHT = variables(":root {")
DARK = variables(':root[data-theme="dark"] {')
DARK_FROM_SYSTEM = variables(':root:not([data-theme="light"]) {')
TEAM_COLORS = dict(re.findall(r'([A-Z]{3}): "(#[0-9A-Fa-f]{6})"',
                              TEMPLATE[TEMPLATE.index("const TEAM_COLORS"):TEMPLATE.index("};", TEMPLATE.index("const TEAM_COLORS"))]))


def luminance(hex_color: str) -> float:
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# Text color on the backgrounds it is drawn on (see the CSS rules using each variable).
TEXT_PAIRS = [
    ("ink", "bg"), ("ink", "card"), ("ink", "accent-soft"),  # body text, cards, picked box and verdict
    ("muted", "bg"), ("muted", "card"), ("muted", "accent-soft"),  # notes, labels, the picked box's coach line
    ("accent", "bg"), ("accent", "card"), ("accent", "accent-soft"),  # section titles, links, the league tag
]


def test_both_ways_into_dark_mode_use_the_same_colors():
    # The palette is written twice (phone set to dark, and an explicit dark theme); they must not drift apart.
    assert DARK == DARK_FROM_SYSTEM
    assert set(LIGHT) == set(DARK)


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("text, background", TEXT_PAIRS)
def test_text_is_readable(theme_name, text, background):
    theme = LIGHT if theme_name == "light" else DARK
    ratio = contrast(theme[text], theme[background])
    assert ratio >= TEXT_MINIMUM, f"{theme_name}: {text} on {background} is {ratio:.2f}"


@pytest.mark.parametrize("team", sorted(TEAM_COLORS))
def test_white_coach_name_is_readable_on_every_team_color(team):
    assert contrast("#ffffff", TEAM_COLORS[team]) >= TEXT_MINIMUM


def test_this_coach_bar_shows_on_the_card_in_both_themes():
    assert len(TEAM_COLORS) == 30
    # Light mode draws the bar in the team color on a white card.
    assert min(contrast(color, LIGHT["card"]) for color in TEAM_COLORS.values()) >= SHAPE_MINIMUM
    # Dark mode cannot use team colors (most are dark), so the bar switches to the accent color.
    assert re.search(r':root\[data-theme="dark"\] \.card \{ --team-bar: var\(--accent\); \}', TEMPLATE)
    assert re.search(r':root:not\(\[data-theme="light"\]\) \.card \{ --team-bar: var\(--accent\); \}', TEMPLATE)
    assert contrast(DARK["accent"], DARK["card"]) >= SHAPE_MINIMUM
