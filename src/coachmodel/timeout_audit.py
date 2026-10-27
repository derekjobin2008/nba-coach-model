"""Per-coach timeout audit: what does each coach's timeout spending cost at the end of
close games?

The decision: when the opponent is on a run before the final minutes, spend a timeout
or save it. coachmodel.timeout_value measured both sides:
    spending one during a run   about 0 points over the next 3 minutes (-0.10, range -0.18
                                to -0.01), so no benefit
    saving one for the end      about +2 points of win probability per usable timeout at
                                2:00 of a close game (range includes 0)
Only 2 timeouts may be used in the last 3 minutes, so a saved timeout matters only if
spending it would leave fewer than 2. Backward induction over (timeouts left, time left)
is then immediate: spending gains nothing and can only lose end-game timeouts, so the
optimal policy never spends one that could leave the team under 2 at the end.

The audit measures each coach against that policy: how often he reaches 2:00 of a game
within 6 points with fewer than 2 usable timeouts, and the cost:
    wins per season = timeouts short per close game x value of a timeout x close games
                      per season
The 95% range combines resampling his close games with drawing the timeout's value from
its own measured range. Timeouts are also used for reasons this does not model (a play
late in a quarter, an injury, avoiding a violation), so this is an upper bound on what
run timeouts cost.

Usage:
    python -m coachmodel.timeout_audit
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT
from coachmodel.timeout_value import END_GAME_SQL, LAST_THREE_MINUTES_LIMIT, end_game_value

MIN_CLOSE_GAMES = 40
DRAWS = 2000

RUN_TIMEOUTS_SQL = """
    -- Coach timeouts called during an opponent run of 6+ before the final 3 minutes, per
    -- regular-season game coached.
    SELECT c.coach_name,
           count(*) FILTER (WHERE s.called_timeout AND s.opponent_run >= 6
                            AND NOT (s.period = 4 AND s.period_seconds_left < 180) AND s.period <= 4)
               / count(DISTINCT s.game_id) AS run_timeouts_per_game
    FROM stoppages AS s
    JOIN game_coaches AS c USING (game_id, team_id)
    JOIN games AS g USING (game_id)
    WHERE g.season_type = 'Regular Season'
    GROUP BY c.coach_name
"""


def shortfall(timeouts_left: pd.Series) -> np.ndarray:
    """Usable timeouts missing at 2:00 (0, 1, or 2)."""
    return (LAST_THREE_MINUTES_LIMIT - timeouts_left.clip(upper=LAST_THREE_MINUTES_LIMIT)).to_numpy(dtype=float)


def audit(end_game: pd.DataFrame, value: float, value_se: float, run_timeouts: pd.Series, rng) -> pd.DataFrame:
    value_draws = rng.normal(value, value_se, DRAWS)
    rows = []
    for coach, games in end_game.groupby("coach_name"):
        if len(games) < MIN_CLOSE_GAMES:
            continue
        missing = shortfall(games["timeouts_left"])
        close_per_season = len(games) / games["season"].nunique()
        resampled = missing[rng.integers(0, len(missing), (DRAWS, len(missing)))].mean(axis=1)
        wins = resampled * value_draws * close_per_season
        rows.append({
            "coach": coach, "close games": len(games),
            "short at 2:00": float(np.mean(missing > 0)),
            "run timeouts per game": float(run_timeouts.get(coach, np.nan)),
            "wins per season": float(missing.mean() * value * close_per_season),
            "low": float(np.percentile(wins, 2.5)), "high": float(np.percentile(wins, 97.5)),
        })
    return pd.DataFrame(rows).set_index("coach").sort_values("wins per season", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    from coachmodel.win_probability import WinProbability, load_rows

    with duckdb.connect(str(args.database), read_only=True) as connection:
        end_game = connection.sql(END_GAME_SQL.read_text()).df()
        run_timeouts = connection.sql(RUN_TIMEOUTS_SQL).df().set_index("coach_name")["run_timeouts_per_game"]
        win_rows = load_rows(connection)
    win_model = WinProbability().fit(win_rows[win_rows["season_type"] == "Regular Season"])
    measured = end_game_value(end_game, win_model).loc["usable timeouts"]
    table = audit(end_game[end_game["coach_name"].notna()], measured["estimate"], measured["se"],
                  run_timeouts, np.random.default_rng(124))

    print(f"value of a usable timeout at 2:00 of a close game: {measured['estimate']:+.3f} win probability "
          f"(95% range {measured['low']:+.3f} to {measured['high']:+.3f})")
    print(f"{len(table)} coaches with {MIN_CLOSE_GAMES}+ close games; cost = wins per season lost to arriving at "
          "2:00 with fewer than 2 usable timeouts (an upper bound)\n")
    formats = {"short at 2:00": "{:.0%}".format, "run timeouts per game": "{:.2f}".format,
               "wins per season": "{:.3f}".format, "low": "{:.3f}".format, "high": "{:.3f}".format}
    print(table.head(6).to_string(formatters=formats))
    print("   ...")
    print(table.tail(4).to_string(header=False, formatters=formats))
    excluding_zero = table[table["low"] > 0]
    print(f"\n{len(excluding_zero)} of {len(table)} coaches have a range that excludes zero. "
          f"Largest cost {table['wins per season'].max():.2f} wins per season "
          f"(foul-trouble audit's largest: about 0.47).")
    correlation = table[["run timeouts per game", "short at 2:00"]].corr().iloc[0, 1]
    print(f"correlation between run timeouts per game and being short at 2:00, across coaches: {correlation:.2f}")


if __name__ == "__main__":
    main()
