-- What is a timeout saved for the end worth? One row per team per close game: the state
-- at the 2:00 mark of the 4th quarter (the last play at or before it), timeouts left for
-- both teams, and whether the team won. Regular season, within 6 points at 2:00.
WITH at_two_minutes AS (
    SELECT game_id, team_id, max(action_id) AS action_id
    FROM team_state
    WHERE period = 4 AND period_seconds_left >= 120
    GROUP BY ALL
)
SELECT
    s.game_id, s.team_id, s.opponent_team_id, g.season, s.is_home, s.score_margin,
    coalesce(s.has_ball, false) AS has_ball,
    s.timeouts_remaining AS timeouts_left,
    opponent.timeouts_remaining AS opponent_timeouts_left,
    (s.is_home AND g.home_points > g.away_points) OR (NOT s.is_home AND g.away_points > g.home_points) AS won,
    own.games_before AS team_games_before, own.margin_before AS team_margin_before,
    own.previous_season_margin AS team_previous_season_margin,
    opp.games_before AS opponent_games_before, opp.margin_before AS opponent_margin_before,
    opp.previous_season_margin AS opponent_previous_season_margin
FROM at_two_minutes AS a
JOIN team_state AS s USING (game_id, team_id, action_id)
JOIN team_state AS opponent
  ON opponent.game_id = s.game_id AND opponent.action_id = s.action_id AND opponent.team_id = s.opponent_team_id
JOIN games AS g ON g.game_id = s.game_id
LEFT JOIN team_rating_inputs AS own ON own.game_id = s.game_id AND own.team_id = s.team_id
LEFT JOIN team_rating_inputs AS opp ON opp.game_id = s.game_id AND opp.team_id = s.opponent_team_id
WHERE g.season_type = 'Regular Season' AND abs(s.score_margin) <= 6
