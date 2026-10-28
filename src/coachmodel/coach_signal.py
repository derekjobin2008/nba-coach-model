"""How strong is the coach part of the forecast? Three checks before it leads the page.

1. Size: each coach's long-run tendency against the league model, with its 95% range
   (from the scout report's cached results), and how many clearly differ from average.
2. Split-half reliability: each coach's games are split at random into two halves, and
   his coach effect is fit separately on each (league model fixed). If habits are real,
   the halves agree across coaches. The correlation between halves is a reliability; the
   Spearman-Brown formula 2r / (1 + r) gives the reliability of an effect fit on all his
   games.
3. Calibration on the unseen 2025-26 season: predicted vs observed by decile, and per
   coach, (observed - expected) / sqrt(expected) for the league model and the forecast.
   If the league model misses particular coaches and the coach effects fix that, the
   spread of this per-coach z shrinks toward 1 (what chance alone gives; game-to-game
   correlation pushes it a little above 1 even for a perfect model).

Usage:
    python -m coachmodel.coach_signal --task timeout
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from coachmodel.baseline import TASKS, build_model, calibration_table, split_by_date
from coachmodel.build_db import DATABASE_PATH, REPO_ROOT
from coachmodel.coach_effects import CHOSEN_TAUS, LEVEL_COLUMNS, GroupEffects, load_task_rows, pooled_log_odds

CACHE_PATH = REPO_ROOT / "data" / "scout_data.json"
PAGE_KEYS = {"timeout": "timeouts", "substitution": "substitutions"}
MIN_DECISIONS_PER_HALF = 40  # a coach needs this many decisions in each half to be compared
TEST_FROM = "2025-10-01"


def fit_effects(rows: pd.DataFrame, offset: np.ndarray, task_name: str) -> GroupEffects:
    taus = CHOSEN_TAUS[task_name]
    columns = {level: LEVEL_COLUMNS[level] for level in taus}
    return GroupEffects(rows, columns).fit(offset, rows[TASKS[task_name].label].to_numpy(), rows, taus)


def size_table(task_name: str) -> pd.DataFrame:
    """Each coach's long-run tendency (rate ratio vs the league model) and its range."""
    data = json.loads(CACHE_PATH.read_text())
    key = PAGE_KEYS[task_name]
    return pd.DataFrame([{"coach": coach["name"], "ratio": coach[key]["ratio"],
                          "low": coach[key]["tendency_low"], "high": coach[key]["tendency_high"]}
                         for coach in data["coaches"]]).set_index("coach")


def split_half(rows: pd.DataFrame, offset: np.ndarray, task_name: str, rng) -> pd.DataFrame:
    """Coach effects fit separately on two random halves of every coach's games."""
    label = TASKS[task_name].label
    games = rows["game_id"].unique()
    half_of_game = dict(zip(games, rng.integers(0, 2, len(games))))
    half = rows["game_id"].map(half_of_game).to_numpy()
    effects = {}
    for part in (0, 1):
        chosen = half == part
        part_rows = rows[chosen].reset_index(drop=True)
        model = fit_effects(part_rows, offset[chosen], task_name)
        decisions = part_rows.groupby("coach_name")[label].sum()
        effects[part] = pd.Series(model.effects["coach"])[decisions >= MIN_DECISIONS_PER_HALF]
    return pd.DataFrame({"half A": effects[0], "half B": effects[1]}).dropna()


def per_coach_z(test: pd.DataFrame, probability: np.ndarray, label: str) -> pd.Series:
    totals = pd.DataFrame({"coach": test["coach_name"], "observed": test[label], "expected": probability}).groupby("coach").sum()
    return (totals["observed"] - totals["expected"]) / np.sqrt(totals["expected"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task = TASKS[args.task]
    rng = np.random.default_rng(124)

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = load_task_rows(connection, task)
    rows = rows[rows["season_type"] == "Regular Season"].reset_index(drop=True)

    sizes = size_table(args.task)
    clear = sizes[(sizes["low"] > 1) | (sizes["high"] < 1)]
    print(f"=== {args.task} ===")
    print(f"1. size: {len(sizes)} coaches; {len(clear)} clearly differ from the league model "
          f"(usual-habit range excludes average). Spread of rate ratios: middle 50% "
          f"{sizes['ratio'].quantile(0.25):.2f} to {sizes['ratio'].quantile(0.75):.2f}, "
          f"full {sizes['ratio'].min():.2f} to {sizes['ratio'].max():.2f}")
    strongest = clear.assign(distance=np.abs(np.log(clear["ratio"]))).sort_values("distance", ascending=False)
    print("   strongest habits: " + ", ".join(f"{name} {row.ratio:.2f}x" for name, row in strongest.head(8).iterrows()))

    league = build_model(task).fit(rows, rows[task.label])
    halves = split_half(rows, pooled_log_odds(league, rows), args.task, rng)
    r = halves["half A"].corr(halves["half B"])
    agree = np.sign(halves["half A"]) == np.sign(halves["half B"])
    strong = halves[(halves.abs() > np.log(1.10)).all(axis=1)]
    print(f"\n2. split-half: {len(halves)} coaches with {MIN_DECISIONS_PER_HALF}+ decisions in each half; "
          f"correlation between halves {r:.2f}, reliability on all his games {2 * r / (1 + r):.2f}; "
          f"same direction in both halves {agree.mean():.0%}")
    print(f"   coaches 10%+ from average in both halves: {len(strong)}, same direction in "
          f"{(np.sign(strong['half A']) == np.sign(strong['half B'])).mean():.0%} of them")

    train, test = (part.reset_index(drop=True) for part in split_by_date(rows, TEST_FROM))
    pooled = build_model(task).fit(train, train[task.label])
    league_test = pooled_log_odds(pooled, test)
    effects = fit_effects(train, pooled_log_odds(pooled, train), args.task)
    forecast = expit(league_test + effects.effects_for(test))
    labels = test[task.label].to_numpy()
    table = calibration_table(labels, forecast)
    print("\n3. calibration on 2025-26 (forecast), 10 equal-size bins:")
    print("   predicted " + " ".join(f"{v:6.2%}" for v in table["mean_predicted"]))
    print("   observed  " + " ".join(f"{v:6.2%}" for v in table["observed_rate"]))
    print("   per coach, (observed - expected) / sqrt(expected), spread across coaches "
          "(about 1 = only chance):")
    for name, probability in (("league model", expit(league_test)), ("forecast", forecast)):
        z = per_coach_z(test, probability, task.label)
        print(f"     {name:13s} sd {z.std():.2f}; coaches beyond +-2: {(z.abs() > 2).sum()} of {len(z)}")


if __name__ == "__main__":
    main()
