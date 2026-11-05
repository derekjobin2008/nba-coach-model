"""Does weighting this season's games more help? A replay of 2025-26 in date order.

The forecast so far treats a coach's habit as one number across every season. Coaches
change, so this adds a "this coach, this season" effect on top, partially pooled toward
his long-run habit with the season-to-season drift already measured (SEASON_DRIFT_SD).
Its weight grows on its own as the season's games accumulate: after a few games it
barely moves, after half a season it carries most of what is new. Past seasons always
stay in the fit.

The replay walks through 2025-26 a month at a time. Before each month both forecasts
are refit on the training seasons plus every 2025-26 game already played, then scored
on that month's games:
    without season weighting   coach and coach-team effects (and players for subs)
    with season weighting      the same plus the coach-season effect
The league model is fit once on the training seasons, so it never sees 2025-26, and both
forecasts see exactly the same games; the only difference is the weighting.

Usage:
    python -m coachmodel.season_replay --task timeout
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from coachmodel.baseline import TASKS, build_model, calibration_table, split_by_date
from coachmodel.build_db import DATABASE_PATH
from coachmodel.coach_effects import (
    CHOSEN_TAUS, LEVEL_COLUMNS, GroupEffects, bootstrap_improvement, load_task_rows, pooled_log_odds,
    row_log_loss, season_weighted_taus,
)

TEST_FROM = "2025-10-01"


def monthly_replay(rows: pd.DataFrame, offset: np.ndarray, test_from: str, task_name: str) -> pd.DataFrame:
    """Log-odds for every test row from fits that use only games before its month."""
    label = TASKS[task_name].label
    is_test = (rows["game_date"] >= pd.Timestamp(test_from)).to_numpy()
    months = rows["game_date"].dt.to_period("M").to_numpy()
    predictions = pd.DataFrame(index=np.flatnonzero(is_test), columns=["without", "with"], dtype=float)
    for month in sorted(set(months[is_test])):
        known = ~is_test | (months < month)
        scored = is_test & (months == month)
        known_rows = rows[known].reset_index(drop=True)
        for name, with_season in (("without", False), ("with", True)):
            taus = season_weighted_taus(task_name) if with_season else CHOSEN_TAUS[task_name]
            model = GroupEffects(known_rows, {level: LEVEL_COLUMNS[level] for level in taus}).fit(
                offset[known], known_rows[label].to_numpy(), known_rows, taus)
            predictions.loc[np.flatnonzero(scored), name] = offset[scored] + model.effects_for(rows[scored])
        print(f"  {month}: scored {scored.sum():,} rows", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task = TASKS[args.task]

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = load_task_rows(connection, task)
    rows = rows[rows["season_type"] == "Regular Season"].reset_index(drop=True)
    train, _ = split_by_date(rows, TEST_FROM)
    offset = pooled_log_odds(build_model(task).fit(train, train[task.label]), rows)

    print(f"=== {args.task}: replaying 2025-26 a month at a time ===")
    predictions = monthly_replay(rows, offset, TEST_FROM, args.task)
    test = rows.loc[predictions.index]
    labels = test[task.label].to_numpy()
    losses = {name: row_log_loss(labels, predictions[name].to_numpy()) for name in predictions}
    improvement = 1 - losses["with"].sum() / losses["without"].sum()
    low, high = np.percentile(bootstrap_improvement(test["game_id"].to_numpy(), losses["without"], losses["with"],
                                                    np.random.default_rng(124)), [2.5, 97.5])
    print(f"\nlog loss without season weighting {losses['without'].mean():.5f}, "
          f"with {losses['with'].mean():.5f}")
    print(f"improvement {improvement:+.3%} (95% range from resampling games {low:+.3%} to {high:+.3%})")
    months = test["game_date"].dt.to_period("M").to_numpy()
    print("by month (improvement):  " + "  ".join(
        f"{month}: {1 - losses['with'][months == month].sum() / losses['without'][months == month].sum():+.2%}"
        for month in sorted(set(months))))
    for name in ("without", "with"):
        table = calibration_table(labels, expit(predictions[name].to_numpy()))
        under = (table["observed_rate"] > table["mean_predicted"]).sum()
        print(f"\ncalibration, {name} season weighting ({under} of {len(table)} bins under the observed rate):")
        print("   predicted " + " ".join(f"{v:6.2%}" for v in table["mean_predicted"]))
        print("   observed  " + " ".join(f"{v:6.2%}" for v in table["observed_rate"]))


if __name__ == "__main__":
    main()
