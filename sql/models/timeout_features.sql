-- Features for the timeout task: one row per stoppage per team, from information the
-- coach has when play stops. Excludes stoppages with a scorer-forced (TV) timeout and
-- teams with no regulation timeouts left, where there is no decision to make.
SELECT
    s.game_id, s.team_id, g.season, g.season_type, g.game_date, coaches.coach_name,
    s.stoppage_cause,
    greatest(least(s.score_margin, 25), -25) AS score_margin,
    abs(greatest(least(s.score_margin, 25), -25)) AS abs_score_margin,
    s.period_seconds_left,
    s.game_seconds_elapsed,
    s.period >= 4 AS is_fourth_or_overtime,
    s.period > 4 AS is_overtime,
    -- The clock stops after made baskets in the last 2 minutes of the 4th and OT, and
    -- close late games are when coaches manage possessions.
    s.period >= 4 AND s.period_seconds_left <= 120 AS is_last_two_minutes,
    s.period >= 4 AND s.period_seconds_left <= 180 AND abs(s.score_margin) <= 6 AS is_late_and_close,
    least(s.opponent_run, 15) AS opponent_run,
    -- Overtime gives each team fresh timeouts under different rules: use 2 as a stand-in.
    coalesce(s.timeouts_remaining, 2) AS timeouts_remaining,
    s.team_fouls_in_period,
    s.opponent_fouls_in_period,
    s.team_in_penalty,
    s.opponent_in_penalty,
    s.is_home,
    -- Who has the ball after the play that stopped the game (inferred possession).
    coalesce(s.has_ball, false) AS has_ball,
    coalesce(least(s.rest_days, 4), 1) AS rest_days,
    s.called_timeout
FROM stoppages AS s
JOIN games AS g USING (game_id)
LEFT JOIN game_coaches AS coaches USING (game_id, team_id)
WHERE NOT s.has_mandatory_timeout
  AND coalesce(s.timeouts_remaining, 2) > 0
