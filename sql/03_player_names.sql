-- Lookup from the way a name is written in a game to that player's id, per team and game.
--
-- Substitution text names the incoming player only by a short form ("Toppin",
-- "Ja. Green", "Pöltl"), and the form changes when two teammates share a last name.
-- Each name form gets a source rank, and we keep only the best-ranked, unambiguous id:
--   1. the outgoing side of a substitution ("FOR Ja. Green"), which comes with the id,
--      so it is written exactly as incoming names are;
--   2. the initial form ('J. Green') the feed attaches to every player action;
--   3. the bare last name ('Green').
-- About 99% of incoming players match; the rest never appear again in that game.
CREATE OR REPLACE TABLE player_names AS
WITH candidate_names AS (
    SELECT game_id, team_id, person_id,
           regexp_extract(description, ' FOR (.+)$', 1) AS name_form, 1 AS source_rank
    FROM pbp WHERE action_type = 'Substitution'
    UNION ALL
    SELECT game_id, team_id, person_id, player_name_initial, 2
    FROM pbp WHERE person_id IS NOT NULL AND team_id IS NOT NULL AND player_name_initial IS NOT NULL
    UNION ALL
    SELECT game_id, team_id, person_id, player_name, 3
    FROM pbp WHERE person_id IS NOT NULL AND team_id IS NOT NULL AND player_name IS NOT NULL
),
ids_per_name AS (
    SELECT game_id, team_id, name_form, source_rank,
           count(DISTINCT person_id) AS id_count,
           any_value(person_id) AS person_id
    FROM candidate_names
    GROUP BY game_id, team_id, name_form, source_rank
)
SELECT game_id, team_id, name_form, person_id, source_rank
FROM ids_per_name
-- QUALIFY filters on a window function, like HAVING does for GROUP BY:
-- keep the best (lowest) source rank for each name form...
QUALIFY row_number() OVER (PARTITION BY game_id, team_id, name_form ORDER BY source_rank) = 1
    -- ...and only if that best source points to exactly one player.
    AND id_count = 1;

-- Tier 4, for incoming names tiers 1-3 missed: the feed sometimes disambiguates with a
-- longer first-name prefix ("Je. Green" for Jeff Green, while the initial form is the
-- ambiguous "J. Green") or a suffix ("Jemison" vs "Jemison III"). Match those against
-- full names of everyone who appeared for that team in any game of that season, again
-- keeping only unambiguous matches.
INSERT INTO player_names
WITH players AS (
    SELECT
        PERSON_ID AS person_id,
        strip_accents(split_part(DISPLAY_LAST_COMMA_FIRST, ', ', 1)) AS last_name,
        strip_accents(split_part(DISPLAY_LAST_COMMA_FIRST, ', ', 2)) AS first_name
    FROM read_json('data/raw/players.json')
),
season_rosters AS (
    SELECT DISTINCT games.season, pbp.team_id, pbp.person_id
    FROM pbp JOIN games USING (game_id)
    WHERE pbp.person_id IS NOT NULL AND pbp.team_id IS NOT NULL
),
unmatched_names AS (
    SELECT DISTINCT
        pbp.game_id, games.season, pbp.team_id,
        regexp_extract(pbp.description, 'SUB: (.+) FOR ', 1) AS name_form
    FROM pbp JOIN games USING (game_id)
    WHERE pbp.action_type = 'Substitution'
      AND NOT EXISTS (
          SELECT 1 FROM player_names AS known
          WHERE known.game_id = pbp.game_id AND known.team_id = pbp.team_id
            AND known.name_form = regexp_extract(pbp.description, 'SUB: (.+) FOR ', 1))
),
name_parts AS (
    -- "Je. Green" -> prefix 'Je', last 'Green'; "Jemison" -> prefix '', last 'Jemison'.
    SELECT
        *,
        CASE WHEN name_form LIKE '%. %' THEN split_part(name_form, '. ', 1) ELSE '' END AS first_prefix,
        strip_accents(CASE WHEN name_form LIKE '%. %'
                           THEN substr(name_form, strpos(name_form, '. ') + 2)
                           ELSE name_form END) AS last_part
    FROM unmatched_names
),
candidates AS (
    SELECT n.game_id, n.team_id, n.name_form, players.person_id
    FROM name_parts AS n
    JOIN season_rosters AS roster ON roster.season = n.season AND roster.team_id = n.team_id
    JOIN players ON players.person_id = roster.person_id
    WHERE starts_with(players.first_name, n.first_prefix)
      -- Equal last names, or equal up to a suffix on either side ("Jemison III").
      AND (players.last_name = n.last_part
           OR starts_with(players.last_name, n.last_part || ' ')
           OR starts_with(n.last_part, players.last_name || ' '))
)
SELECT game_id, team_id, name_form, any_value(person_id) AS person_id, 4 AS source_rank
FROM candidates
GROUP BY game_id, team_id, name_form
HAVING count(DISTINCT person_id) = 1;
