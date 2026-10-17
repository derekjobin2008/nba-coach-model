-- Lineup segments: stretches of a game when the same ten players are on the court, with
-- the points each team scored. The input to regularized adjusted plus-minus (RAPM).
--
-- Every stint start or end is a boundary; between two consecutive boundaries nobody
-- checks in or out. Regular season, regulation and overtime. Segments where either side
-- does not have exactly five known players (an unresolved substitution) are dropped.
WITH boundaries AS (
    SELECT DISTINCT game_id, start_seconds AS t FROM stints
    UNION
    SELECT DISTINCT game_id, end_seconds FROM stints
),
segments AS (
    SELECT game_id, t AS seg_from, lead(t) OVER (PARTITION BY game_id ORDER BY t) AS seg_to
    FROM boundaries
),
lineups AS (
    -- The players on the court for each segment, home side and away side.
    SELECT
        seg.game_id, seg.seg_from, seg.seg_to,
        list(s.person_id ORDER BY s.person_id) FILTER (WHERE s.team_id = g.home_team_id) AS home_players,
        list(s.person_id ORDER BY s.person_id) FILTER (WHERE s.team_id = g.away_team_id) AS away_players
    FROM segments AS seg
    JOIN stints AS s
      ON s.game_id = seg.game_id AND s.start_seconds <= seg.seg_from AND s.end_seconds >= seg.seg_to
    JOIN games AS g ON g.game_id = seg.game_id
    WHERE seg.seg_to > seg.seg_from AND g.season_type = 'Regular Season'
    GROUP BY ALL
),
scores AS (
    SELECT game_id, game_seconds_elapsed AS t, action_id, score_home, score_away FROM pbp
)
SELECT
    l.game_id, l.seg_from, l.seg_to,
    (l.seg_to - l.seg_from) / 60 AS minutes,
    coalesce(at_end.score_home, 0) - coalesce(at_start.score_home, 0) AS home_points,
    coalesce(at_end.score_away, 0) - coalesce(at_start.score_away, 0) AS away_points,
    l.home_players, l.away_players
FROM lineups AS l
-- Score at the segment's start and end. As in the foul-trouble effects, plays at the
-- exact boundary belong to the segment that ends there (a substitution happens after
-- the free throws or basket logged at the same clock).
ASOF LEFT JOIN scores AS at_start ON at_start.game_id = l.game_id AND l.seg_from >= at_start.t
ASOF LEFT JOIN scores AS at_end ON at_end.game_id = l.game_id AND l.seg_to >= at_end.t
WHERE len(l.home_players) = 5 AND len(l.away_players) = 5
