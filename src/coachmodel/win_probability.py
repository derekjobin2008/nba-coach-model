"""Win probability: the chance the home team wins from a game state (margin, time left,
possession). Phase 3 measures every coaching decision in win probability.

The model is logistic regression on a few features built from one idea: a lead is worth
more the less time is left. With t = share of the game still to play, the outcome is
roughly margin + (points still to be scored), and the spread of the points still to be
scored grows like sqrt(t). So the features scale by 1 / sqrt(t):
    margin / sqrt(t)        a lead, measured in "remaining-time standard deviations"
    ball / sqrt(t)          having the ball (+1 home, -1 away) is worth about a point
    sqrt(t)                 home-court edge accrues over the time still to play
    gap * sqrt(t)           the better team's edge (rating gap, points per game) accrues
                            the same way: gap * t points still to come, scaled by 1/sqrt(t)
The result is smooth and monotone in margin and time, which the optimization in Phase 3
needs.

Team ratings use only games already played (no leakage): a team's point differential so
far this season plus K_GAMES games' worth of its previous season's average (times
PRIOR_WEIGHT, since teams regress toward average between seasons), divided by games
played plus K_GAMES. K_GAMES and PRIOR_WEIGHT are fit by predicting game margins on the
training seasons.

Each season's own home edge (+1.6 to +2.6 points per game) is not modeled: it cannot be
predicted from the season's earlier games (2022-23: +3.18 in its first half, +1.82 in its
second), and a version that tried was always fit to ignore them.

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


K_GAMES_GRID = [3, 5, 10, 15, 20, 30]
PRIOR_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def model_features(home_margin, seconds_left, home_ball, rating_gap=0.0) -> np.ndarray:
    time_share = (np.asarray(seconds_left, dtype=float) + TIME_SLACK_SECONDS) / REGULATION_SECONDS
    scale = 1 / np.sqrt(time_share)
    rating_gap = np.broadcast_to(np.asarray(rating_gap, dtype=float), time_share.shape)
    return np.column_stack([
        np.asarray(home_margin, dtype=float) * scale,
        np.asarray(home_ball, dtype=float) * scale,
        np.sqrt(time_share),
        rating_gap * np.sqrt(time_share),
    ])


def team_rating(games_before, margin_before, previous_season_margin, k_games, prior_weight) -> np.ndarray:
    """Points per game better than average, from earlier games and last season."""
    prior = prior_weight * np.nan_to_num(np.asarray(previous_season_margin, dtype=float))
    return (np.asarray(margin_before, dtype=float) + k_games * prior) / (np.asarray(games_before, dtype=float) + k_games)


def rating_gap(rows: pd.DataFrame, k_games: float, prior_weight: float) -> np.ndarray:
    home = team_rating(rows["home_games_before"], rows["home_margin_before"], rows["home_previous_season_margin"],
                       k_games, prior_weight)
    away = team_rating(rows["away_games_before"], rows["away_margin_before"], rows["away_previous_season_margin"],
                       k_games, prior_weight)
    return np.nan_to_num(home - away)


def fit_rating_weights(games: pd.DataFrame) -> tuple[float, float]:
    """K_GAMES and PRIOR_WEIGHT that best predict final margins (home edge + rating gap,
    least squares) on one row per game."""
    margin = games["final_home_margin"].to_numpy(dtype=float)
    home_edge = float(np.mean(margin))
    errors = {}
    for k_games in K_GAMES_GRID:
        for prior_weight in PRIOR_WEIGHT_GRID:
            residual = margin - home_edge - rating_gap(games, k_games, prior_weight)
            errors[(k_games, prior_weight)] = np.mean(residual ** 2)
    return min(errors, key=errors.get)


class WinProbability:
    def __init__(self, use_ratings: bool = True):
        self.use_ratings = use_ratings

    def gap(self, frame: pd.DataFrame) -> np.ndarray:
        return rating_gap(frame, self.k_games, self.prior_weight) if self.use_ratings else 0.0

    def fit(self, frame: pd.DataFrame):
        if self.use_ratings:
            self.k_games, self.prior_weight = fit_rating_weights(frame.drop_duplicates("game_id"))
        features = model_features(frame["home_margin"], frame["seconds_left"], frame["home_ball"], self.gap(frame))
        # No penalty: four features and millions of rows, nothing to regularize.
        self.model = LogisticRegression(C=np.inf, max_iter=1000).fit(features, frame["home_won"])
        return self

    def home_win_probability(self, home_margin, seconds_left, home_ball, rating_gap=0.0) -> np.ndarray:
        """rating_gap: home rating minus away rating (points per game); 0 = even teams."""
        return self.model.predict_proba(model_features(home_margin, seconds_left, home_ball, rating_gap))[:, 1]

    def predict_rows(self, frame: pd.DataFrame) -> np.ndarray:
        return self.home_win_probability(frame["home_margin"], frame["seconds_left"], frame["home_ball"],
                                         self.gap(frame))


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
    probability = model.predict_rows(rows)

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


def calibration_noise(frame: pd.DataFrame, predicted: np.ndarray, rng, draws: int = 200) -> np.ndarray:
    """Chance spread (sd) of each calibration bin's observed win rate, from resampling
    whole games: a game's outcome is shared by all its plays, so a bin holds a few
    hundred independent outcomes, not hundreds of thousands."""
    bins = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    outcomes = frame["home_won"].to_numpy()
    by_game = list(frame.groupby("game_id").indices.values())
    rates = []
    for _ in range(draws):
        rows = np.concatenate([by_game[i] for i in rng.integers(0, len(by_game), len(by_game))])
        rates.append(pd.Series(outcomes[rows]).groupby(bins[rows]).mean().reindex(range(10)).to_numpy())
    return np.nanstd(rates, axis=0)


def leave_one_season_out(frame: pd.DataFrame, use_ratings: bool, rng) -> pd.DataFrame:
    """Fit on all other seasons, score each season: log loss, the worst calibration error
    among the middle bins (3rd to 8th of 10), and that error in units of chance spread
    (about 2.6 or more would be more than chance among six bins)."""
    results = []
    for season in sorted(frame["season"].unique()):
        train, test = frame[frame["season"] != season], frame[frame["season"] == season]
        predicted = WinProbability(use_ratings).fit(train).predict_rows(test)
        table = calibration_table(test["home_won"].to_numpy(), predicted)
        error = (table["observed_rate"] - table["mean_predicted"]).to_numpy()
        noise = calibration_noise(test, predicted, rng)
        results.append({"season": season,
                        "log loss": evaluate(train["home_won"].to_numpy(), test["home_won"].to_numpy(), predicted)["model_log_loss"],
                        "worst middle bin": float(np.abs(error[2:8]).max()),
                        "in noise units": float(np.abs(error[2:8] / noise[2:8]).max())})
    return pd.DataFrame(results).set_index("season")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-from", default="2025-10-01", help="hold out games on or after this date (default: 2025-26)")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--chart", action="append", default=[], metavar="GAME_ID",
                        help="also save a win probability chart for this game to reports/")
    parser.add_argument("--by-season", action="store_true",
                        help="leave-one-season-out check, with and without team ratings")
    args = parser.parse_args()

    if args.by_season:
        with duckdb.connect(str(args.database), read_only=True) as connection:
            frame = load_rows(connection)
        frame = frame[frame["season_type"] == "Regular Season"]
        rng = np.random.default_rng(124)
        without = leave_one_season_out(frame, use_ratings=False, rng=rng)
        with_ratings = leave_one_season_out(frame, use_ratings=True, rng=rng)
        table = without.join(with_ratings, lsuffix=" (no ratings)", rsuffix=" (ratings)")
        print("leave-one-season-out: fit on the other seasons, score this one")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        return

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
    predicted = model.predict_rows(test)

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
