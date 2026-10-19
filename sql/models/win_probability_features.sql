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
    g.home_points > g.away_points AS home_won,
    -- The final margin is never a model input: it only fits the team-rating weights on
    -- training games (how well pre-game ratings predicted results).
    g.home_points - g.away_points AS final_home_margin,
    -- Pre-game strength inputs for both teams (earlier games only; see 12_team_rating_inputs).
    home.games_before AS home_games_before, home.margin_before AS home_margin_before,
    home.previous_season_margin AS home_previous_season_margin,
    away.games_before AS away_games_before, away.margin_before AS away_margin_before,
    away.previous_season_margin AS away_previous_season_margin
FROM team_state AS s
JOIN games AS g USING (game_id)
LEFT JOIN team_rating_inputs AS home ON home.game_id = s.game_id AND home.team_id = g.home_team_id
LEFT JOIN team_rating_inputs AS away ON away.game_id = s.game_id AND away.team_id = g.away_team_id
WHERE s.is_home
  AND s.has_ball IS NOT NULL
