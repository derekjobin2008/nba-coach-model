-- Win probability training rows: the game state after every play, from the home team's
-- side, labeled with whether the home team won. Every play is a row, so long games
-- count more; that matches how the model is used (at any moment of any game).
SELECT
    s.game_id, g.season, g.season_type, g.game_date,
    s.action_id, s.period, s.period_seconds_left,
    s.score_margin AS home_margin,
    -- Time left until the final buzzer of regulation, or of the current overtime.
    CASE WHEN s.period <= 4 THEN 2880 - s.game_seconds_elapsed ELSE s.period_seconds_left END AS seconds_left,
    s.period > 4 AS is_overtime,
    CASE WHEN s.has_ball THEN 1 ELSE -1 END AS home_ball,
    g.home_points > g.away_points AS home_won
FROM team_state AS s
JOIN games AS g USING (game_id)
WHERE s.is_home
  AND s.has_ball IS NOT NULL
