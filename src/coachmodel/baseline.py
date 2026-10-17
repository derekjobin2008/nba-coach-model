"""Pooled (league-wide) baselines for the two prediction tasks: will this team's coach
call a timeout at this stoppage, and will this on-court player be subbed out at it?
These are the generic models a coach-specific model must beat.

Usage:
    python -m coachmodel.baseline --task timeout --test-from 2025-03-01
    python -m coachmodel.baseline --task substitution --test-from 2025-03-01
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

MODELS_SQL_DIR = REPO_ROOT / "sql" / "models"


@dataclass(frozen=True)
class Task:
    features_sql: Path
    label: str
    categorical: list[str]
    # Splines for features whose effect is not linear, e.g. timeouts are rare early in
    # a quarter and common near its end.
    spline: list[str]
    numeric: list[str]
    flags: list[str]

    @property
    def features(self) -> list[str]:
        return self.categorical + self.spline + self.numeric + self.flags


GAME_STATE_FLAGS = ["is_fourth_or_overtime", "is_overtime", "is_last_two_minutes", "is_late_and_close", "is_home"]

TASKS = {
    "timeout": Task(
        features_sql=MODELS_SQL_DIR / "timeout_features.sql",
        label="called_timeout",
        categorical=["stoppage_cause"],
        spline=["period_seconds_left"],
        numeric=["score_margin", "abs_score_margin", "game_seconds_elapsed", "opponent_run",
                 "timeouts_remaining", "team_fouls_in_period", "opponent_fouls_in_period", "rest_days"],
        flags=GAME_STATE_FLAGS + ["team_in_penalty", "opponent_in_penalty", "has_ball"],
    ),
    "substitution": Task(
        features_sql=MODELS_SQL_DIR / "substitution_features.sql",
        label="subbed_out",
        categorical=["stoppage_cause"],
        # Fatigue: the chance of a sub rises with stint length, then falls for the
        # players (often stars) who are left in.
        spline=["period_seconds_left", "stint_minutes", "minutes_played"],
        numeric=["score_margin", "abs_score_margin", "game_seconds_elapsed", "opponent_run", "player_fouls"],
        flags=GAME_STATE_FLAGS + ["has_mandatory_timeout", "is_in_foul_trouble"],
    ),
}


def load_features(connection: duckdb.DuckDBPyConnection, task: Task) -> pd.DataFrame:
    frame = connection.sql(task.features_sql.read_text()).df()
    frame[task.flags + [task.label]] = frame[task.flags + [task.label]].astype(int)
    return frame


def split_by_date(frame: pd.DataFrame, test_from: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole games on or after test_from are held out, so no game is in both sets."""
    is_test = frame["game_date"] >= pd.Timestamp(test_from)
    return frame[~is_test], frame[is_test]


def build_model(task: Task) -> Pipeline:
    features = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), task.categorical),
        ("splines", SplineTransformer(n_knots=6, degree=3), task.spline),
        ("numeric", StandardScaler(), task.numeric),
        ("flags", "passthrough", task.flags),
    ])
    return Pipeline([
        ("features", features),
        ("logistic", LogisticRegression(max_iter=2000)),
    ])


def calibration_table(labels: np.ndarray, predicted: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Group predictions into equal-size bins and compare the average prediction with
    how often the event actually happened in each bin."""
    frame = pd.DataFrame({"label": labels, "predicted": predicted})
    frame["bin"] = pd.qcut(frame["predicted"], q=bins, labels=False, duplicates="drop")
    return (
        frame.groupby("bin")
        .agg(rows=("label", "size"), mean_predicted=("predicted", "mean"), observed_rate=("label", "mean"))
        .reset_index(drop=True)
    )


def evaluate(train_labels: np.ndarray, test_labels: np.ndarray, predicted: np.ndarray) -> dict:
    """Log loss of the model against the no-information baseline: predicting the
    training set's overall rate for every row."""
    constant_rate = train_labels.mean()
    return {
        "test_rows": len(test_labels),
        "test_rate": test_labels.mean(),
        "model_log_loss": log_loss(test_labels, predicted, labels=[0, 1]),
        "constant_log_loss": log_loss(test_labels, np.full(len(test_labels), constant_rate), labels=[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--test-from", required=True, help="hold out games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task = TASKS[args.task]

    with duckdb.connect(str(args.database), read_only=True) as connection:
        frame = load_features(connection, task)
    train, test = split_by_date(frame, args.test_from)
    model = build_model(task).fit(train, train[task.label])
    predicted = model.predict_proba(test)[:, 1]

    results = evaluate(train[task.label].to_numpy(), test[task.label].to_numpy(), predicted)
    print(f"task: {args.task}")
    print(f"train: {len(train):,} rows ({train['game_id'].nunique():,} games), "
          f"test: {results['test_rows']:,} rows ({test['game_id'].nunique():,} games)")
    print(f"{task.label} rate in test: {results['test_rate']:.2%}")
    print(f"log loss, constant rate: {results['constant_log_loss']:.4f}")
    print(f"log loss, pooled model:  {results['model_log_loss']:.4f}  "
          f"({1 - results['model_log_loss'] / results['constant_log_loss']:.1%} better)")
    print("\ncalibration (test set, 10 equal-size bins):")
    print(calibration_table(test[task.label].to_numpy(), predicted).to_string(
        formatters={"mean_predicted": "{:.2%}".format, "observed_rate": "{:.2%}".format}))


if __name__ == "__main__":
    main()
