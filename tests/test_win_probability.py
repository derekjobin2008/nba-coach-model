import numpy as np

from coachmodel.win_probability import WinProbability, load_rows, model_features

PACERS_KNICKS_GAME_6 = "0042400306"
KNICKS, PACERS = 1610612752, 1610612754


def test_shots_are_taken_by_the_team_with_the_ball(database):
    consistent_share = database.sql("""
        WITH before_each_play AS (
            SELECT pbp.action_type, pbp.team_id,
                   lag(possession.team_with_ball) OVER (PARTITION BY pbp.game_id ORDER BY pbp.action_id) AS had_ball
            FROM pbp JOIN possession USING (game_id, action_id)
        )
        SELECT avg((had_ball = team_id)::INT) FROM before_each_play
        WHERE action_type IN ('Made Shot', 'Missed Shot')
    """).fetchone()[0]
    assert consistent_share > 0.99


def test_game_6_opening_possessions(database):
    # Brunson scored the first basket (11:43, Q1), so the Knicks had the ball to start,
    # and the Pacers had it right after his layup.
    first_two = database.sql(f"""
        SELECT p.team_with_ball FROM possession AS p JOIN pbp USING (game_id, action_id)
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND (pbp.action_id = 1 OR pbp.description LIKE 'Brunson 4%Layup (2 PTS)%')
        ORDER BY action_id
    """).fetchall()
    assert first_two == [(KNICKS,), (PACERS,)]


def test_features_stay_finite_at_the_buzzer():
    assert np.isfinite(model_features([3, -3, 0], [0, 0, 0], [1, -1, 1])).all()


def test_win_probability_behaves_like_basketball(database):
    model = WinProbability().fit(load_rows(database))
    # More margin, more win probability.
    margins = np.arange(-15, 16)
    by_margin = model.home_win_probability(margins, np.full(len(margins), 600), np.ones(len(margins)))
    assert np.all(np.diff(by_margin) > 0)
    # The same lead is worth more as the clock runs down.
    early, late = model.home_win_probability([6, 6], [2400, 30], [-1, -1])
    assert late > early > 0.5


def test_team_ratings_use_only_earlier_games():
    import duckdb
    from pathlib import Path
    connection = duckdb.connect()
    connection.execute("""
        CREATE TABLE games AS SELECT * FROM (VALUES
            ('g1', '2024-25', 'Regular Season', DATE '2024-10-22', 1, 2, 110, 100),
            ('g2', '2024-25', 'Regular Season', DATE '2024-10-24', 1, 3, 90, 95),
            ('g3', '2024-25', 'Regular Season', DATE '2024-10-26', 3, 1, 100, 120)
        ) AS t(game_id, season, season_type, game_date, home_team_id, away_team_id, home_points, away_points)
    """)
    sql = (Path(__file__).parents[1] / "sql" / "12_team_rating_inputs.sql").read_text()
    connection.execute(sql)
    team_1 = connection.sql("""
        SELECT game_id, games_before, margin_before FROM team_rating_inputs WHERE team_id = 1 ORDER BY game_id
    """).fetchall()
    # Before g3, team 1 has played g1 (+10) and g2 (-5); g3's own result (+20) is never used.
    assert team_1 == [("g1", 0, 0), ("g2", 1, 10), ("g3", 2, 5)]


def test_rating_weights_find_a_real_last_season_effect():
    import pandas as pd
    from coachmodel.win_probability import fit_rating_weights
    rng = np.random.default_rng(0)
    n = 5000
    strength_home, strength_away = rng.normal(0, 5, n), rng.normal(0, 5, n)
    games = pd.DataFrame({
        # Early-season games: little current-season evidence, last season informative.
        "home_games_before": 2, "away_games_before": 2,
        "home_margin_before": 2 * (strength_home + rng.normal(0, 12, n)),
        "away_margin_before": 2 * (strength_away + rng.normal(0, 12, n)),
        "home_previous_season_margin": strength_home + rng.normal(0, 2, n),
        "away_previous_season_margin": strength_away + rng.normal(0, 2, n),
        "final_home_margin": 2 + strength_home - strength_away + rng.normal(0, 12, n),
    })
    k_games, prior_weight = fit_rating_weights(games)
    assert prior_weight > 0 and k_games >= 5
