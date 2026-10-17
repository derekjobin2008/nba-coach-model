-- What changes while a player is on the court in foul trouble? For each player, quarter,
-- and current foul count: minutes on the court, fouls he committed, and points his team
-- scored and allowed during those minutes. Comparing each player's foul-trouble minutes
-- with his own low-foul minutes in the same quarter separates the effect of foul trouble
-- from who gets into foul trouble (foul-prone players, mostly big men).
--
-- Regular season, quarters 1-4, excluding the last 2 minutes of the 4th, where
-- intentional fouling and end-of-game possessions would dominate.
WITH foul_times AS (
    SELECT f.game_id, f.person_id, f.player_fouls, p.game_seconds_elapsed AS t
    FROM personal_fouls AS f JOIN pbp AS p USING (game_id, action_id)
),
players AS (
    SELECT DISTINCT game_id, person_id, team_id FROM stints
),
foul_intervals AS (
    -- [start, end) of the time a player had each foul count.
    SELECT p.game_id, p.person_id, p.team_id, 0 AS fouls_now,
           0.0 AS t_from, coalesce(min(ft.t), 2880) AS t_to
    FROM players AS p LEFT JOIN foul_times AS ft USING (game_id, person_id)
    GROUP BY ALL
    UNION ALL
    SELECT ft.game_id, ft.person_id, p.team_id, ft.player_fouls,
           ft.t, coalesce(lead(ft.t) OVER (PARTITION BY ft.game_id, ft.person_id ORDER BY ft.t), 2880)
    FROM foul_times AS ft JOIN players AS p USING (game_id, person_id)
),
quarters AS (
    SELECT q AS quarter, (q - 1) * 720.0 AS q_from,
           CASE WHEN q = 4 THEN 2880.0 - 120 ELSE q * 720.0 END AS q_to
    FROM range(1, 5) AS r(q)
),
on_court_pieces AS (
    -- Stretches on the court with a fixed foul count inside one quarter.
    SELECT
        i.game_id, i.person_id, i.team_id, least(i.fouls_now, 5) AS fouls_now, q.quarter,
        greatest(s.start_seconds, i.t_from, q.q_from) AS piece_from,
        least(s.end_seconds, i.t_to, q.q_to) AS piece_to
    FROM foul_intervals AS i
    JOIN stints AS s ON s.game_id = i.game_id AND s.person_id = i.person_id
    JOIN quarters AS q ON q.q_from < least(s.end_seconds, i.t_to) AND q.q_to > greatest(s.start_seconds, i.t_from)
    WHERE least(s.end_seconds, i.t_to, q.q_to) > greatest(s.start_seconds, i.t_from, q.q_from)
),
team_scores AS (
    -- Each team's score after every play, to read off points scored within a piece.
    SELECT t.game_id, t.team_id, p.game_seconds_elapsed AS t, t.action_id,
           CASE WHEN t.is_home THEN p.score_home ELSE p.score_away END AS team_score,
           CASE WHEN t.is_home THEN p.score_away ELSE p.score_home END AS opponent_score
    FROM team_state AS t JOIN pbp AS p USING (game_id, action_id)
),
pieces_with_scores AS (
    SELECT
        piece.*,
        coalesce(at_end.team_score, 0) - coalesce(at_start.team_score, 0) AS team_points,
        coalesce(at_end.opponent_score, 0) - coalesce(at_start.opponent_score, 0) AS opponent_points
    FROM on_court_pieces AS piece
    -- Score at the piece's start and end (latest play at or before each time). Plays at
    -- the exact moment a piece starts belong to the piece that ended then: free throws
    -- from the foul that raised his count count against the count he fouled with, not
    -- the new one.
    ASOF LEFT JOIN team_scores AS at_start
        ON at_start.game_id = piece.game_id AND at_start.team_id = piece.team_id AND piece.piece_from >= at_start.t
    ASOF LEFT JOIN team_scores AS at_end
        ON at_end.game_id = piece.game_id AND at_end.team_id = piece.team_id AND piece.piece_to >= at_end.t
),
fouls_committed AS (
    -- A foul committed while he had f fouls is the one that took him to f + 1.
    SELECT f.person_id, least(f.player_fouls - 1, 5) AS fouls_now,
           CASE WHEN p.game_seconds_elapsed < 720 THEN 1 WHEN p.game_seconds_elapsed < 1440 THEN 2
                WHEN p.game_seconds_elapsed < 2160 THEN 3 ELSE 4 END AS quarter,
           count(*) AS fouls
    FROM personal_fouls AS f JOIN pbp AS p USING (game_id, action_id)
    JOIN games AS g ON g.game_id = f.game_id
    WHERE g.season_type = 'Regular Season' AND p.period <= 4
      AND NOT (p.period = 4 AND p.period_seconds_left <= 120)
    GROUP BY ALL
)
SELECT
    p.person_id, p.quarter, p.fouls_now,
    sum(p.piece_to - p.piece_from) / 60 AS minutes,
    sum(p.team_points) AS team_points,
    sum(p.opponent_points) AS opponent_points,
    coalesce(any_value(f.fouls), 0) AS fouls
FROM pieces_with_scores AS p
JOIN games AS g USING (game_id)
LEFT JOIN fouls_committed AS f USING (person_id, fouls_now, quarter)
WHERE g.season_type = 'Regular Season'
GROUP BY p.person_id, p.quarter, p.fouls_now
