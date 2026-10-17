"""Win probability: the chance the home team wins from a game state (margin, time left,
possession). Phase 3 measures every coaching decision in win probability.

The model is logistic regression on a few features built from one idea: a lead is worth
more the less time is left. With t = share of the game still to play, the outcome is
roughly margin + (points still to be scored), and the spread of the points still to be
scored grows like sqrt(t). So the features scale by 1 / sqrt(t):
    margin / sqrt(t)        a lead, measured in "remaining-time standard deviations"
    ball / sqrt(t)          having the ball (+1 home, -1 away) is worth about a point
    sqrt(t)                 home-court edge accrues over the time still to play
The result is smooth and monotone in margin and time, which the optimization in Phase 3
needs. Team strength is not included yet.

Usage:
    python -m coachmodel.win_probability --test-from 2025-03-01
    python -m coachmodel.win_probability --test-from 2025-03-01 --chart 0042300237
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from coachmodel.baseline import calibration_table, evaluate, split_by_date
from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

FEATURES_SQL = REPO_ROOT / "sql" / "models" / "win_probability_features.sql"
REGULATION_SECONDS = 2880
# A few seconds of slack keeps 1 / sqrt(t) finite at the buzzer.
TIME_SLACK_SECONDS = 5


def model_features(home_margin, seconds_left, home_ball) -> np.ndarray:
    time_share = (np.asarray(seconds_left, dtype=float) + TIME_SLACK_SECONDS) / REGULATION_SECONDS
    scale = 1 / np.sqrt(time_share)
    return np.column_stack([
        np.asarray(home_margin, dtype=float) * scale,
        np.asarray(home_ball, dtype=float) * scale,
        np.sqrt(time_share),
    ])


class WinProbability:
    def fit(self, frame: pd.DataFrame):
        features = model_features(frame["home_margin"], frame["seconds_left"], frame["home_ball"])
        # No penalty: three features and millions of rows, nothing to regularize.
        self.model = LogisticRegression(C=np.inf, max_iter=1000).fit(features, frame["home_won"])
        return self

    def home_win_probability(self, home_margin, seconds_left, home_ball) -> np.ndarray:
        return self.model.predict_proba(model_features(home_margin, seconds_left, home_ball))[:, 1]


def load_rows(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = connection.sql(FEATURES_SQL.read_text()).df()
    frame["home_won"] = frame["home_won"].astype(int)
    return frame


SANITY_CHECKS = [
    # (description, home margin, seconds left, home has the ball?, expectation)
    ("tip-off, home team", 0, 2880, None, "about 55-60%"),
    ("home up 10, 2:00 left, away ball", 10, 120, False, "95% or more"),
    ("home down 10, 2:00 left, home ball", -10, 120, True, "5% or less"),
    ("tied, 0:24 left, home ball (last shot)", 0, 24, True, "a little above 50%"),
    ("tied, 0:24 left, away ball", 0, 24, False, "a little below 50%"),
    ("home up 3, 0:10 left, away ball", 3, 10, False, "high, but a 3 ties it"),
]


def plot_game(connection: duckdb.DuckDBPyConnection, model: WinProbability, game_id: str, path: Path) -> None:
    """Save a chart of the home team's win probability through one game, with each
    coach's timeouts marked."""
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    rows = load_rows(connection)
    rows = rows[rows["game_id"] == game_id].sort_values("action_id")
    home, away = connection.sql(f"SELECT home_team, away_team FROM games WHERE game_id = '{game_id}'").fetchone()
    timeouts = connection.sql(f"""
        SELECT d.game_seconds_elapsed, d.is_home FROM decision_events AS d
        WHERE d.game_id = '{game_id}' AND d.event_type = 'timeout' AND NOT d.likely_mandatory
    """).df()
    elapsed_minutes = (REGULATION_SECONDS - rows["seconds_left"].where(rows["period"] <= 4, 0)) / 60
    probability = model.home_win_probability(rows["home_margin"], rows["seconds_left"], rows["home_ball"])

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(elapsed_minutes, probability, color="#2f6db5", linewidth=1.6)
    axis.axhline(0.5, color="#999999", linewidth=0.8)
    for _, timeout in timeouts.iterrows():
        axis.axvline(timeout["game_seconds_elapsed"] / 60, color="#c0392b" if timeout["is_home"] else "#7f8c8d",
                     linewidth=0.8, linestyle="--")
    for quarter_end in (12, 24, 36):
        axis.axvline(quarter_end, color="#dddddd", linewidth=0.8, zorder=0)
    axis.set(xlim=(0, 48), ylim=(0, 1), xlabel="game minute",
             ylabel=f"{home} win probability",
             title=f"{away} @ {home}: {home} win probability "
                   f"(dashed: coach timeouts, red = {home}, grey = {away})")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-from", required=True, help="hold out games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--chart", action="append", default=[], metavar="GAME_ID",
                        help="also save a win probability chart for this game to reports/")
    args = parser.parse_args()

    connection = duckdb.connect(str(args.database), read_only=True)
    frame = load_rows(connection)
    frame = frame[frame["season_type"] == "Regular Season"]
    train, test = split_by_date(frame, args.test_from)
    model = WinProbability().fit(train)
    for game_id in args.chart:
        chart_path = REPO_ROOT / "reports" / f"win_probability_{game_id}.png"
        plot_game(connection, model, game_id, chart_path)
        print(f"saved {chart_path.relative_to(REPO_ROOT)}")
    connection.close()
    predicted = model.home_win_probability(test["home_margin"], test["seconds_left"], test["home_ball"])

    results = evaluate(train["home_won"].to_numpy(), test["home_won"].to_numpy(), predicted)
    print(f"train: {train['game_id'].nunique():,} games ({len(train):,} plays), "
          f"test: {test['game_id'].nunique():,} games ({len(test):,} plays)")
    print(f"home win rate in test: {test.drop_duplicates('game_id')['home_won'].mean():.1%}")
    print(f"log loss, constant rate: {results['constant_log_loss']:.4f}")
    print(f"log loss, win probability model: {results['model_log_loss']:.4f}")
    print("\ncalibration (test plays, 10 equal-size bins):")
    print(calibration_table(test["home_won"].to_numpy(), predicted).to_string(
        formatters={"mean_predicted": "{:.1%}".format, "observed_rate": "{:.1%}".format}))

    print("\nsanity checks (home win probability):")
    for description, margin, seconds, home_ball, expectation in SANITY_CHECKS:
        if home_ball is None:  # tip-off: average over who wins the jump
            probability = model.home_win_probability([margin] * 2, [seconds] * 2, [1, -1]).mean()
        else:
            probability = model.home_win_probability([margin], [seconds], [1 if home_ball else -1])[0]
        print(f"  {description:42s} {probability:6.1%}   (expect {expectation})")


if __name__ == "__main__":
    main()
