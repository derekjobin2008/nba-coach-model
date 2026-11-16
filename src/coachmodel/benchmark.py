"""The model benchmark: one frozen test that every version of the models is scored on.

So that "better" always means the same thing:
    split   fit on every game before 2025-10-01, score on every 2025-26 game after it. The
            test games are never used for fitting or for choosing settings.
    score   log loss at the next stoppage (lower is better), with a 95% range on each gain
            from resampling whole games.
    ladder  each model must beat the one below it:
                league-wide rate   one number for everyone
                coach average      each coach's own training rate, ignoring the game state
                league model       the game state, no coach effects
                coach model        + partially pooled coach and coach-team effects (and
                                   players, for subs): the page's habit numbers
                season-weighted    + a this-coach-this-season effect: the page's likely moves
    verdict a gain whose 95% range excludes zero is "better" (or "worse"); a range that
            includes zero is "no measurable difference", never "better".

Every run appends its rows, with the date and commit, to reports/benchmark.csv, so a
regression or an improvement shows in the file instead of in someone's summary.
tests/test_benchmark.py fails if the latest coach model no longer beats the league model.

Usage:
    python -m coachmodel.benchmark --task timeout        # about a minute
    python -m coachmodel.benchmark --task substitution   # about an hour
"""

import argparse
import datetime
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import logit

from coachmodel.baseline import TASKS, build_model, split_by_date
from coachmodel.build_db import DATABASE_PATH, REPO_ROOT
from coachmodel.coach_effects import (
    CHOSEN_TAUS, LEVEL_COLUMNS, GroupEffects, bootstrap_improvement, load_task_rows, pooled_log_odds, row_log_loss,
    season_weighted_taus,
)
from coachmodel.forecast_check import coach_average_log_odds

TEST_FROM = "2025-10-01"  # frozen: changing it starts a new benchmark, not a new row
RESULTS_PATH = REPO_ROOT / "reports" / "benchmark.csv"
LADDER = ["league-wide rate", "coach average", "league model", "coach model", "season-weighted"]


def verdict(low: float, high: float) -> str:
    """Derek's rule: only a range that excludes zero counts as a difference."""
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "no measurable difference"


def ladder_log_odds(train: pd.DataFrame, test: pd.DataFrame, task_name: str) -> dict[str, np.ndarray]:
    """Each rung's prediction (log-odds) for every test row, fit on the training rows only."""
    task = TASKS[task_name]
    label = task.label
    pooled = build_model(task).fit(train, train[label])
    train_offset, league = pooled_log_odds(pooled, train), pooled_log_odds(pooled, test)
    predictions = {
        "league-wide rate": np.full(len(test), logit(train[label].mean())),
        "coach average": coach_average_log_odds(train, test, label),
        "league model": league,
    }
    for name, taus in (("coach model", CHOSEN_TAUS[task_name]), ("season-weighted", season_weighted_taus(task_name))):
        columns = {level: LEVEL_COLUMNS[level] for level in taus}
        effects = GroupEffects(train, columns).fit(train_offset, train[label].to_numpy(), train, taus)
        predictions[name] = league + effects.effects_for(test)
    return predictions


def score_ladder(test: pd.DataFrame, predictions: dict[str, np.ndarray], label: str, rng) -> pd.DataFrame:
    """Log loss per rung, and each rung's gain over the rung below with a 95% range."""
    labels, games = test[label].to_numpy(), test["game_id"].to_numpy()
    losses = {name: row_log_loss(labels, predictions[name]) for name in LADDER}
    rows = []
    for below, name in zip([None] + LADDER[:-1], LADDER):
        row = {"model": name, "log_loss": losses[name].mean()}
        if below:
            low, high = np.percentile(bootstrap_improvement(games, losses[below], losses[name], rng), [2.5, 97.5])
            row.update(compared_with=below, gain=1 - losses[name].sum() / losses[below].sum(),
                       low=low, high=high, verdict=verdict(low, high))
        rows.append(row)
    return pd.DataFrame(rows)


def commit_id() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task = TASKS[args.task]

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = load_task_rows(connection, task)
    train, test = (part.reset_index(drop=True) for part in split_by_date(rows, TEST_FROM))
    table = score_ladder(test, ladder_log_odds(train, test, args.task), task.label, np.random.default_rng(124))
    table.insert(0, "task", args.task)
    table.insert(0, "commit", commit_id())
    table.insert(0, "date", datetime.date.today().isoformat())
    table["test_from"], table["test_games"], table["test_rows"] = TEST_FROM, test["game_id"].nunique(), len(test)

    print(f"{args.task}: fit on {train['game_id'].nunique():,} games before {TEST_FROM}, "
          f"scored on {test['game_id'].nunique():,} games ({len(test):,} rows)\n")
    for row in table.itertuples():
        line = f"  {row.model:17s} log loss {row.log_loss:.5f}"
        if isinstance(row.compared_with, str):
            line += f"   vs {row.compared_with:16s} {row.gain:+.3%} ({row.low:+.3%} to {row.high:+.3%})  {row.verdict}"
        print(line)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False, float_format="%.6g")
    print(f"\nappended to {RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
