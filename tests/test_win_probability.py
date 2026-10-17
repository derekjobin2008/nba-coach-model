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
