"""The model benchmark: its verdict rule, its scoring, and the pass line on recorded results."""

import numpy as np
import pandas as pd
import pytest

from coachmodel.baseline import TASKS, split_by_date
from coachmodel.benchmark import LADDER, RESULTS_PATH, TEST_FROM, ladder_log_odds, score_ladder, verdict
from coachmodel.coach_effects import load_task_rows


def test_a_gain_whose_range_includes_zero_is_no_measurable_difference():
    assert verdict(0.0006, 0.0016) == "better"
    assert verdict(-0.00007, 0.0009) == "no measurable difference"  # +0.04% with this range is not a help
    assert verdict(-0.002, -0.001) == "worse"


def synthetic_test_rows(rng, games=300, rows_per_game=40):
    """Rows whose true chance is known, so a model that knows it must beat one that does not."""
    truth = rng.uniform(0.02, 0.4, games * rows_per_game)
    return (pd.DataFrame({"game_id": np.repeat(np.arange(games), rows_per_game),
                          "called_timeout": (rng.random(len(truth)) < truth).astype(int)}),
            np.log(truth / (1 - truth)))


def test_a_model_that_knows_more_is_better_and_a_copy_is_no_different():
    rng = np.random.default_rng(3)
    test, true_log_odds = synthetic_test_rows(rng)
    flat = np.full(len(test), np.log(test["called_timeout"].mean() / (1 - test["called_timeout"].mean())))
    noisy = true_log_odds + rng.normal(0, 1.0, len(test))
    predictions = {"league-wide rate": flat, "coach average": flat, "league model": noisy,
                   "coach model": true_log_odds, "season-weighted": true_log_odds}
    table = score_ladder(test, predictions, "called_timeout", np.random.default_rng(0)).set_index("model")
    assert list(table.index) == LADDER
    assert table.loc["coach model", "verdict"] == "better"  # the truth beats a noisy guess
    assert table.loc["season-weighted", "verdict"] == "no measurable difference"  # an exact copy
    assert table.loc["coach average", "gain"] == 0


def test_every_rung_fits_and_predicts_on_the_fixture_games(database):
    rows = load_task_rows(database, TASKS["timeout"])
    train, test = (part.reset_index(drop=True) for part in split_by_date(rows, "2024-10-01"))
    assert len(train) and len(test)
    predictions = ladder_log_odds(train, test, "timeout")
    assert list(predictions) == LADDER
    assert all(np.isfinite(values).all() and len(values) == len(test) for values in predictions.values())


recorded = pytest.mark.skipif(not RESULTS_PATH.exists(), reason="no benchmark results recorded yet")


@recorded
def test_every_recorded_run_used_the_same_frozen_test():
    # A model that only looks better because the test games changed would show up here.
    results = pd.read_csv(RESULTS_PATH)
    assert set(results["test_from"]) == {TEST_FROM}
    assert results.groupby("task")[["test_games", "test_rows"]].nunique().max().max() == 1


@recorded
def test_the_latest_coach_model_beats_the_league_model():
    results = pd.read_csv(RESULTS_PATH)
    for task, runs in results.groupby("task"):
        latest = runs[(runs["date"] == runs["date"].iloc[-1]) & (runs["commit"] == runs["commit"].iloc[-1])]
        coach_model = latest.set_index("model").loc["coach model"]
        assert coach_model["compared_with"] == "league model"
        assert coach_model["verdict"] == "better", f"{task}: coach model gain {coach_model['gain']:+.3%}"
