-- Do starters benched for early foul trouble get their minutes back later? One row per
-- starter per competitive regular-season game (decided by 10 points or fewer): minutes in
-- each half, fouls at the end of Q1 and at halftime, and whether he fouled out.
WITH starters AS (
    SELECT DISTINCT s.game_id, s.person_id
    FROM stints AS s JOIN games AS g USING (game_id)
    WHERE s.period = 1 AND s.start_action_id = 0
      AND g.season_type = 'Regular Season' AND abs(g.home_points - g.away_points) <= 10
),
minutes AS (
    SELECT
        s.game_id, s.person_id,
        sum(greatest(least(s.end_seconds, 1440) - s.start_seconds, 0)) / 60 AS first_half_minutes,
        sum(greatest(s.end_seconds - greatest(s.start_seconds, 1440), 0)) / 60 AS second_half_minutes
    FROM stints AS s JOIN starters USING (game_id, person_id)
    WHERE s.period <= 4
    GROUP BY ALL
),
fouls AS (
    SELECT
        f.game_id, f.person_id,
        count(*) FILTER (WHERE p.period = 1) AS fouls_after_q1,
        count(*) FILTER (WHERE p.period <= 2) AS fouls_at_half,
        max(f.player_fouls) >= 6 AS fouled_out
    FROM personal_fouls AS f JOIN pbp AS p USING (game_id, action_id)
    GROUP BY ALL
)
SELECT
    m.game_id, m.person_id, m.first_half_minutes, m.second_half_minutes,
    coalesce(f.fouls_after_q1, 0) AS fouls_after_q1,
    coalesce(f.fouls_at_half, 0) AS fouls_at_half,
    coalesce(f.fouled_out, false) AS fouled_out
FROM minutes AS m LEFT JOIN fouls AS f USING (game_id, person_id)
