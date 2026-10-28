-- Features for the substitution task: one row per player on the court at each
-- stoppage, from information the coach has when play stops. Player identity is left
-- out of the pooled baseline's features: who the player is belongs to the
-- hierarchical model.
SELECT
    p.game_id, p.stoppage_action_id, p.team_id, p.person_id, g.season, g.season_type, g.game_date, coaches.coach_name,
    s.stoppage_cause,
    greatest(least(s.score_margin, 25), -25) AS score_margin,
    abs(greatest(least(s.score_margin, 25), -25)) AS abs_score_margin,
    s.period_seconds_left,
    s.game_seconds_elapsed,
    s.period >= 4 AS is_fourth_or_overtime,
    s.period > 4 AS is_overtime,
    s.period >= 4 AND s.period_seconds_left <= 120 AS is_last_two_minutes,
    s.period >= 4 AND s.period_seconds_left <= 180 AND abs(s.score_margin) <= 6 AS is_late_and_close,
    least(s.opponent_run, 15) AS opponent_run,
    -- A TV timeout is set by the clock rules, not the coach, and is a natural time to sub.
    s.has_mandatory_timeout,
    s.is_home,
    least(p.player_fouls, 6) AS player_fouls,
    -- Common rule of thumb: more fouls than the quarter number is foul trouble.
    p.player_fouls > s.period AS is_in_foul_trouble,
    p.minutes_played,
    p.stint_minutes,
    p.subbed_out
FROM stoppage_players AS p
JOIN stoppages AS s USING (game_id, stoppage_action_id, team_id)
JOIN games AS g USING (game_id)
LEFT JOIN game_coaches AS coaches USING (game_id, team_id)
