-- Per-player inputs for the foul-trouble MDP: how often a player fouls per minute on the
-- court, and the minutes he typically plays when he starts a competitive game (decided by
-- 10 points or fewer), which becomes his minutes cap.
WITH player_games AS (
    SELECT
        s.game_id, s.person_id,
        sum(s.end_seconds - s.start_seconds) / 60 AS minutes,
        bool_or(s.period = 1 AND s.start_action_id = 0) AS started
    FROM stints AS s
    GROUP BY ALL
),
fouls_per_game AS (
    SELECT game_id, person_id, count(*) AS fouls FROM personal_fouls GROUP BY ALL
)
SELECT
    pg.person_id,
    sum(pg.minutes) AS minutes,
    sum(coalesce(f.fouls, 0)) AS fouls,
    count(*) FILTER (WHERE pg.started AND abs(g.home_points - g.away_points) <= 10) AS close_starts,
    median(pg.minutes) FILTER (WHERE pg.started AND abs(g.home_points - g.away_points) <= 10) AS typical_close_minutes
FROM player_games AS pg
JOIN games AS g USING (game_id)
LEFT JOIN fouls_per_game AS f USING (game_id, person_id)
WHERE g.season_type = 'Regular Season'
GROUP BY pg.person_id
