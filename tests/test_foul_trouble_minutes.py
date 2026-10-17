import pandas as pd
import pytest

from coachmodel.foul_trouble_minutes import early_trouble, minutes_change, weighted_change


def game(person_id, first_half, second_half, fouls_after_q1=0, fouls_at_half=0):
    return {"person_id": person_id, "first_half_minutes": first_half, "second_half_minutes": second_half,
            "fouls_after_q1": fouls_after_q1, "fouls_at_half": fouls_at_half, "fouled_out": False}


def test_early_trouble_follows_the_coaches_rule():
    rows = pd.DataFrame([game(1, 18, 18, 2, 2), game(1, 18, 18, 1, 3), game(1, 18, 18, 1, 2)])
    assert early_trouble(rows).tolist() == [True, True, False]


def test_minutes_made_up_after_halftime_show_up_as_second_half_gain():
    # Usually 18 + 18. In his trouble game he sat 4 first-half minutes and got 3 back.
    rows = pd.DataFrame([game(1, 18, 18), game(1, 18, 18), game(1, 14, 21, fouls_after_q1=2)])
    change = weighted_change(minutes_change(rows))
    assert change["first_half_minutes"] == pytest.approx(-4)
    assert change["second_half_minutes"] == pytest.approx(3)
    assert change["total_minutes"] == pytest.approx(-1)


def test_players_are_compared_with_themselves():
    # A 34-minute star never in trouble and a 20-minute big who often is: the big's
    # trouble games are compared with his own 20, not the star's 34.
    rows = pd.DataFrame([game(1, 17, 17)] * 5 + [game(2, 10, 10)] * 3
                        + [game(2, 10, 10, fouls_after_q1=2)] * 2)
    change = minutes_change(rows)
    assert list(change.index) == [2]
    assert weighted_change(change)["total_minutes"] == pytest.approx(0)
