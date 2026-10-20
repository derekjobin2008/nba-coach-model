-- Does a timeout stop a run? One row per team per stoppage while the opponent is on a
-- run, with what happened over the next 3 minutes of game clock.
--
-- Three kinds of stoppage are compared:
--   coach timeout   this team's coach called one
--   TV timeout      the scorer forced one (set by the clock rules, so it falls during
--                   runs essentially at random: a break no coach chose)
--   no timeout      play went on
-- Regular season, regulation, at least 3 minutes left in the period (so the window stays
-- inside it), and not the 4th quarter's last 3 minutes (end-of-game timeouts are a
-- different decision). Team strength comes from team_rating_inputs (earlier games only).
WITH candidates AS (
    SELECT
        s.game_id, s.team_id, s.opponent_team_id, s.stoppage_action_id, g.season,
        s.period, s.period_seconds_left, s.game_seconds_elapsed, s.score_margin,
        s.opponent_run, coalesce(s.has_ball, false) AS has_ball, s.stoppage_cause, s.is_home,
        s.team_fouls_in_period, s.opponent_fouls_in_period, coalesce(s.timeouts_remaining, 0) AS timeouts_remaining,
        CASE WHEN s.called_timeout THEN 'coach timeout'
             WHEN s.has_mandatory_timeout THEN 'TV timeout'
             ELSE 'no timeout' END AS stoppage_kind
    FROM stoppages AS s
    JOIN games AS g USING (game_id)
    WHERE g.season_type = 'Regular Season'
      AND s.period <= 4
      AND s.period_seconds_left >= 180
      AND s.opponent_run >= 6
      -- A TV timeout and this coach's own timeout at the same stoppage: leave it out of
      -- both groups.
      AND NOT (s.called_timeout AND s.has_mandatory_timeout)
),
margins AS (
    SELECT game_id, team_id, game_seconds_elapsed AS t, action_id, score_margin
    FROM team_state
)
SELECT
    c.*,
    -- Margin 3 minutes of game clock later (latest play at or before then), minus now.
    later.score_margin - c.score_margin AS next_3_minutes,
    own.games_before AS team_games_before, own.margin_before AS team_margin_before,
    own.previous_season_margin AS team_previous_season_margin,
    opp.games_before AS opponent_games_before, opp.margin_before AS opponent_margin_before,
    opp.previous_season_margin AS opponent_previous_season_margin
FROM candidates AS c
ASOF JOIN margins AS later
    ON later.game_id = c.game_id AND later.team_id = c.team_id AND c.game_seconds_elapsed + 180 >= later.t
LEFT JOIN team_rating_inputs AS own ON own.game_id = c.game_id AND own.team_id = c.team_id
LEFT JOIN team_rating_inputs AS opp ON opp.game_id = c.game_id AND opp.team_id = c.opponent_team_id
