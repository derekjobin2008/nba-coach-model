"""Does the forecast beat simple baselines on games it was never trained on?

Before the scout page predicts what a coach will do next, the prediction has to beat:
    coach average   that coach's own rate from the training seasons, ignoring the game
                    state (one number per coach)
    league model    the pooled model: the game state, no coach effects
The forecast is the league model plus partially pooled coach and coach-team effects (and
player effects for substitutions), with the spreads chosen earlier by cross-validation.
Trained on 2021-22 to 2024-25, scored on every 2025-26 game, at the next stoppage, the
moment a coach can act. Improvements in log loss come with 95% ranges from resampling
games.

For substitutions the page shows a ranked list of who comes out, so the ranking is also
checked: at stoppages where the coach took at least one player out, how often the
forecast's top-ranked player was one of them. The coach average gives every player the
same chance, so it does no better than picking at random among the five.

Usage:
    python -m coachmodel.forecast_check --task timeout
    python -m coachmodel.forecast_check --task substitution
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from coachmodel.baseline import TASKS, build_model, split_by_date
from coachmodel.build_db import DATABASE_PATH
from coachmodel.coach_effects import (
    CHOSEN_TAUS, LEVEL_COLUMNS, GroupEffects, bootstrap_improvement, load_task_rows, pooled_log_odds, row_log_loss,
)

TEST_FROM = "2025-10-01"


def coach_average_log_odds(train: pd.DataFrame, test: pd.DataFrame, label: str) -> np.ndarray:
    """Each coach's own training-season rate; coaches new in the test season get the league rate."""
    league_rate = train[label].mean()
    rates = train.groupby("coach_name")[label].mean()
    probability = test["coach_name"].map(rates).fillna(league_rate).clip(1e-6, 1 - 1e-6)
    return logit(probability.to_numpy())


def top_ranked_hit_rate(test: pd.DataFrame, probability: np.ndarray, label: str) -> float:
    """Share of team-stoppages with a sub where the highest-probability player was subbed."""
    frame = test[["game_id", "stoppage_action_id", "team_id", label]].assign(probability=probability)
    with_sub = frame.groupby(["game_id", "stoppage_action_id", "team_id"])[label].transform("max") == 1
    frame = frame[with_sub]
    top = frame.loc[frame.groupby(["game_id", "stoppage_action_id", "team_id"])["probability"].idxmax()]
    return float(top[label].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task, taus = TASKS[args.task], CHOSEN_TAUS[args.task]
    rng = np.random.default_rng(124)

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = load_task_rows(connection, task)
    train, test = (part.reset_index(drop=True) for part in split_by_date(rows, TEST_FROM))
    labels = test[task.label].to_numpy()

    pooled = build_model(task).fit(train, train[task.label])
    train_offset, league = pooled_log_odds(pooled, train), pooled_log_odds(pooled, test)
    columns = {level: LEVEL_COLUMNS[level] for level in taus}
    effects = GroupEffects(train, columns).fit(train_offset, train[task.label].to_numpy(), train, taus)
    predictions = {
        "coach average": coach_average_log_odds(train, test, task.label),
        "league model": league,
        "forecast": league + effects.effects_for(test),
    }
    losses = {name: row_log_loss(labels, log_odds) for name, log_odds in predictions.items()}

    print(f"task: {args.task}; trained on {train['season'].min()} to {train['season'].max()} "
          f"({train['game_id'].nunique():,} games), tested on {test['season'].min()} "
          f"({test['game_id'].nunique():,} games, {len(test):,} {'team' if args.task == 'timeout' else 'player'}-stoppages)")
    print("\nlog loss on the unseen season (lower is better):")
    for name, loss in losses.items():
        print(f"  {name:14s} {loss.mean():.4f}")
    print("\nforecast improvement in log loss (95% range from resampling games):")
    for baseline in ("coach average", "league model"):
        improvement = 1 - losses["forecast"].sum() / losses[baseline].sum()
        low, high = np.percentile(bootstrap_improvement(test["game_id"].to_numpy(), losses[baseline],
                                                        losses["forecast"], rng), [2.5, 97.5])
        print(f"  vs {baseline:14s} {improvement:+.2%}   ({low:+.2%} to {high:+.2%})")

    if args.task == "substitution":
        print("\nranked list: when the coach took someone out, was the top-ranked player one of them?")
        for name in ("league model", "forecast"):
            print(f"  {name:14s} {top_ranked_hit_rate(test, expit(predictions[name]), task.label):.1%}")
        players_per_sub = test.groupby(["game_id", "stoppage_action_id", "team_id"])[task.label].agg(["size", "sum"])
        players_per_sub = players_per_sub[players_per_sub["sum"] > 0]
        print(f"  coach average  {(players_per_sub['sum'] / players_per_sub['size']).mean():.1%} "
              "(every player tied: the same as picking at random)")


if __name__ == "__main__":
    main()
