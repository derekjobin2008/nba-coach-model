"""Are the coach-effect ranges honest? Rolling-origin check on seasons the model never saw.

For each test season (2022-23 through 2025-26), the models are fit on the seasons
before it. Each coach who also coached in the training seasons gets:
    b_train, se_train  his coach effect and standard error from the training seasons
    b_test, se_test    his coach effect estimated from the test season alone, holding the
                       league model and his training-era team and player effects fixed
Then z = (b_test - b_train) / sqrt(se_train^2 + se_test^2). If the uncertainty is honest,
z behaves like a standard normal: about 95% of |z| below 1.96 and a standard deviation
of about 1. A standard deviation of 2 means the ranges are half as wide as they should be.

Each standard-error method is checked (see coach_standard_errors). Whatever spread is
left after the best method is read as real season-to-season change in coaches ("drift"),
estimated by the method of moments: the variance of (b_test - b_train) beyond what the
standard errors explain. For the check, each season's drift is estimated from the other
seasons only (cross-fitting), so no season sets its own range width.

Usage:
    python -m coachmodel.range_check --task timeout
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from coachmodel.baseline import TASKS, build_model
from coachmodel.build_db import DATABASE_PATH
from coachmodel.coach_effects import (
    CHOSEN_TAUS, LEVEL_COLUMNS, GroupEffects, coach_standard_errors, load_task_rows, pooled_log_odds,
)

TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
METHODS = ["conditional", "joint", "clustered"]
MIN_TEST_DECISIONS = 30  # a coach needs this many decisions in the test season to be checked


def one_season_effect(offset: np.ndarray, labels: np.ndarray, games: np.ndarray) -> tuple[float, float, float]:
    """One coach's effect estimated from one season by maximum likelihood (Newton's
    method on a single log-odds shift), with an independent-decisions standard error and
    a game-clustered one."""
    effect = 0.0
    for _ in range(50):
        probability = expit(offset + effect)
        step = np.sum(labels - probability) / np.sum(probability * (1 - probability))
        effect += step
        if abs(step) < 1e-10:
            break
    probability = expit(offset + effect)
    information = np.sum(probability * (1 - probability))
    per_game = pd.Series(labels - probability).groupby(games).sum().to_numpy()
    return effect, 1 / np.sqrt(information), np.sqrt(np.sum(per_game ** 2)) / information


def check_season(rows: pd.DataFrame, task_name: str, test_season: str) -> pd.DataFrame:
    task, taus = TASKS[task_name], CHOSEN_TAUS[task_name]
    seasons = sorted(rows["season"].unique())
    train = rows[rows["season"] < test_season].reset_index(drop=True)
    test = rows[rows["season"] == test_season].reset_index(drop=True)
    pooled = build_model(task).fit(train, train[task.label])
    train_offset, test_offset = pooled_log_odds(pooled, train), pooled_log_odds(pooled, test)
    train_labels = train[task.label].to_numpy()
    columns = {level: LEVEL_COLUMNS[level] for level in taus}
    model = GroupEffects(train, columns).fit(train_offset, train_labels, train, taus)
    standard_errors = {method: coach_standard_errors(model, train, train_offset, train_labels, taus, method)
                       for method in METHODS}

    others = [level for level in columns if level != "coach"]
    test_without_coach = test_offset + model.effects_for(test, others)
    results = []
    for coach, part in test.groupby("coach_name"):
        labels = part[task.label].to_numpy()
        if coach not in model.effects["coach"] or labels.sum() < MIN_TEST_DECISIONS:
            continue
        effect, se_independent, se_clustered = one_season_effect(
            test_without_coach[part.index], labels, part["game_id"].to_numpy())
        results.append({
            "season": test_season, "coach": coach, "b_train": model.effects["coach"][coach], "b_test": effect,
            "se_test_independent": se_independent, "se_test_clustered": se_clustered,
            **{f"se_train_{method}": standard_errors[method][coach] for method in METHODS},
        })
    print(f"  {test_season}: trained on {seasons[0]} to {train['season'].max()}, {len(results)} coaches checked",
          flush=True)
    return pd.DataFrame(results)


def honesty(pairs: pd.DataFrame, method: str, drift_variance=0.0) -> dict:
    """drift_variance: one number, or one per row (cross-fitted)."""
    test_column = "se_test_clustered" if method == "clustered" else "se_test_independent"
    spread = np.sqrt(pairs[f"se_train_{method}"] ** 2 + pairs[test_column] ** 2 + drift_variance)
    z = (pairs["b_test"] - pairs["b_train"]) / spread
    return {"inside 95% range": float(np.mean(np.abs(z) < 1.96)), "sd of z": float(np.std(z, ddof=1))}


def drift_variance(pairs: pd.DataFrame) -> float:
    """Season-to-season change in a coach's true effect (method of moments): how much
    more b_test - b_train varies than the clustered standard errors explain."""
    gap = pairs["b_test"] - pairs["b_train"]
    explained = pairs["se_train_clustered"] ** 2 + pairs["se_test_clustered"] ** 2
    return float(max(np.mean(gap ** 2) - np.mean(explained), 0.0))


def cross_fitted_drift(pairs: pd.DataFrame) -> np.ndarray:
    """Drift for each row estimated only from the other test seasons, so a season's
    check never uses its own data to set its width."""
    return pairs["season"].map({season: drift_variance(pairs[pairs["season"] != season])
                                for season in pairs["season"].unique()}).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = load_task_rows(connection, TASKS[args.task])
    rows = rows[rows["season_type"] == "Regular Season"]
    print(f"rolling-origin check for {args.task}:")
    pairs = pd.concat([check_season(rows, args.task, season) for season in TEST_SEASONS], ignore_index=True)

    print(f"\n{len(pairs)} coach-seasons; honest ranges have about 95% inside and a z spread of about 1.0\n")
    table = pd.DataFrame({method: honesty(pairs, method) for method in METHODS}).T
    drift = drift_variance(pairs)
    table.loc["clustered + drift (cross-fitted)"] = honesty(pairs, "clustered", cross_fitted_drift(pairs))
    print(table.to_string(formatters={"inside 95% range": "{:.0%}".format, "sd of z": "{:.2f}".format}))
    typical = pairs[["se_train_conditional", "se_train_joint", "se_train_clustered"]].median()
    print(f"\nmedian training standard error (log-odds): conditional {typical['se_train_conditional']:.3f}, "
          f"joint {typical['se_train_joint']:.3f}, clustered {typical['se_train_clustered']:.3f}")
    print(f"season-to-season drift in a coach's effect: sd {np.sqrt(drift):.3f} log-odds "
          f"(about {np.expm1(np.sqrt(drift)):.0%} in his rate)")


if __name__ == "__main__":
    main()
