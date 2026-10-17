-- How coaches actually handle starters' foul trouble: for starters in competitive games
-- (decided by 10 points or fewer), seconds spent with each foul count in each game minute,
-- and how many of those seconds they were on the court, per game and team. Summed over
-- games this is a coach's (or the league's) revealed policy: on-court share by game
-- minute and foul count, which the MDP's optimal policy is compared against.
WITH starters AS (
    SELECT DISTINCT s.game_id, s.team_id, s.person_id
    FROM stints AS s JOIN games AS g USING (game_id)
    WHERE s.period = 1 AND s.start_action_id = 0
      AND g.season_type = 'Regular Season' AND abs(g.home_points - g.away_points) <= 10
),
foul_times AS (
    SELECT f.game_id, f.person_id, f.player_fouls, p.game_seconds_elapsed AS t
    FROM personal_fouls AS f JOIN pbp AS p USING (game_id, action_id)
),
foul_intervals AS (
    -- [start, end) of the time a starter had a given number of fouls, from tip-off to
    -- the end of regulation.
    SELECT s.game_id, s.team_id, s.person_id, 0 AS fouls_now,
           0.0 AS t_from, coalesce(min(ft.t), 2880) AS t_to
    FROM starters AS s LEFT JOIN foul_times AS ft USING (game_id, person_id)
    GROUP BY ALL
    UNION ALL
    SELECT ft.game_id, s.team_id, ft.person_id, ft.player_fouls,
           ft.t, coalesce(lead(ft.t) OVER (PARTITION BY ft.game_id, ft.person_id ORDER BY ft.t), 2880)
    FROM foul_times AS ft JOIN starters AS s USING (game_id, person_id)
),
game_minutes AS (
    SELECT range AS game_minute FROM range(48)
),
time_in_state AS (
    -- Seconds of each game minute a starter spent with each foul count...
    SELECT i.game_id, i.team_id, i.person_id, i.fouls_now, m.game_minute,
           greatest(least(i.t_to, (m.game_minute + 1) * 60) - greatest(i.t_from, m.game_minute * 60), 0) AS seconds
    FROM foul_intervals AS i JOIN game_minutes AS m
      ON i.t_from < (m.game_minute + 1) * 60 AND i.t_to > m.game_minute * 60
),
on_court_in_state AS (
    -- ...and how many of those seconds he was on the court.
    SELECT t.game_id, t.person_id, t.fouls_now, t.game_minute,
           sum(greatest(least(s.end_seconds, i.t_to, (t.game_minute + 1) * 60)
                        - greatest(s.start_seconds, i.t_from, t.game_minute * 60), 0)) AS seconds_on
    FROM time_in_state AS t
    JOIN foul_intervals AS i USING (game_id, person_id, fouls_now)
    JOIN stints AS s ON s.game_id = t.game_id AND s.person_id = t.person_id
    GROUP BY ALL
)
SELECT
    t.game_id, t.team_id, t.game_minute, least(t.fouls_now, 5) AS fouls_now,
    sum(t.seconds) AS seconds,
    sum(coalesce(o.seconds_on, 0)) AS seconds_on
FROM time_in_state AS t
LEFT JOIN on_court_in_state AS o USING (game_id, person_id, fouls_now, game_minute)
WHERE t.seconds > 0 AND t.fouls_now <= 5
GROUP BY ALL
