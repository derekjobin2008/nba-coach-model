# NBA Coach Decision Model

Models opposing NBA coaches' in-game decisions (timeouts, substitutions) from public
play-by-play, to forecast what a bench will do in a given game state and to compare
coaches' revealed policies with a win-probability-optimal policy.

## Layout

| Folder | What goes there |
|---|---|
| `data/raw/` | Raw API pulls (git-ignored, rebuildable) |
| `sql/` | SQL transformations, run in order against DuckDB |
| `src/coachmodel/` | Python: ingestion, model fitting, OR solvers |
| `notebooks/` | Exploration and hand-validation of games |
| `tests/` | pytest tests |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Download data and build the database

```bash
# Download one season of play-by-play (~1,230 games, ~20-30 min; reruns skip cached games)
PYTHONPATH=src .venv/bin/python -m coachmodel.ingest --season 2024-25
# Specific playoff games
PYTHONPATH=src .venv/bin/python -m coachmodel.ingest --season 2023-24 --season-type Playoffs --game-id 0042300237
# Run sql/*.sql in order into data/coach.duckdb
PYTHONPATH=src .venv/bin/python -m coachmodel.build_db
```

Tables, built in this order:

| Table | One row per |
|---|---|
| `games` | game (home/away teams, final score) |
| `pbp` | play-by-play action (clock parsed, score carried forward) |
| `player_names` | way a name is written in a game, mapped to a player id |
| `decision_events` | coach timeout or substitution (view `lineup_changes` groups subs per stoppage) |
| `stints` | stretch a player spends on the court |
| `personal_fouls` | personal foul, with the player's and team's running counts |
| `possession` | play: which team has the ball after it (inferred) |
| `team_state` | play per team: margin, possession, fouls, penalty, opponent run, timeouts left, rest |
| `decision_state` | decision, with the game state known at that moment |
| `stoppages` | dead-ball stoppage per team, labeled `called_timeout` (Phase 2 timeout task) |
| `stoppage_players` | on-court player per stoppage, labeled `subbed_out` (Phase 2 substitution task) |
| `team_rating_inputs` | team per game: point differential in earlier games and last season's average |
| `game_coaches` | team per game, with its head coach |

Head coaches come from `data/head_coaches.csv`: each team's opening coach per season plus
every in-season change (firing, interim, leave). Add a row there when a coach changes.
The roster endpoint is not used as the source (it reports a post-season snapshot that
includes off-season hires) but is checked against the table in `tests/test_full_data.py`.

The NBA stats site stops answering after a few hundred quick requests, so the downloader
paces itself and waits out blocks (up to 15 minutes) instead of skipping games.

## Models

```bash
# Pooled baselines: train on games before the date, evaluate on the rest
PYTHONPATH=src .venv/bin/python -m coachmodel.baseline --task timeout --test-from 2025-03-01
PYTHONPATH=src .venv/bin/python -m coachmodel.baseline --task substitution --test-from 2025-03-01
# Win probability (home team), with optional charts saved to reports/
PYTHONPATH=src .venv/bin/python -m coachmodel.win_probability --chart 0042300237
# Season-by-season check, with and without team ratings
PYTHONPATH=src .venv/bin/python -m coachmodel.win_probability --by-season
# Does foul trouble change how a player plays? (each player vs his own low-foul baseline)
PYTHONPATH=src .venv/bin/python -m coachmodel.foul_trouble_effects
# Phase 3: foul-trouble MDP (backward induction), optimal vs coaches' observed policy
PYTHONPATH=src .venv/bin/python -m coachmodel.foul_trouble
# Player values: regularized adjusted plus-minus from lineup segments
PYTHONPATH=src .venv/bin/python -m coachmodel.player_value
# Are the coach-effect ranges honest? Rolling-origin check (timeouts ~1 minute, substitutions ~1 hour)
PYTHONPATH=src .venv/bin/python -m coachmodel.range_check --task timeout
# Per-coach audit: what each coach's foul-trouble handling costs (about 3 minutes)
PYTHONPATH=src .venv/bin/python -m coachmodel.foul_trouble_audit
# Do starters benched for early foul trouble get their minutes back?
PYTHONPATH=src .venv/bin/python -m coachmodel.foul_trouble_minutes
# First claim: pooled model + partially pooled head-coach effects, scored on held-out games
PYTHONPATH=src .venv/bin/python -m coachmodel.coach_effects --task timeout --test-from 2025-03-01
```

Feature queries live in `sql/models/` (read by the model code, not part of the build).

## Scout report page

```bash
# Fit the models on every game and build docs/index.html (about 15 minutes)
PYTHONPATH=src .venv/bin/python -m coachmodel.scout_report
# Restyle only: rebuild the page from the cached results in data/scout_data.json
PYTHONPATH=src .venv/bin/python -m coachmodel.scout_report --render
```

The page is one self-contained file (the data is embedded) served by GitHub Pages from
`docs/` in the public repository: https://derekjobin2008.github.io/nba-coach-model/

## Public copy

`scripts/publish_public.sh "message"` pushes a clean snapshot of `main` (without the course
files `.claude/`, `.claude-sessions/`, `.githooks/`, `CLAUDE.md`) to the git remote named
`public`. Each run adds one commit to the public repository's own history.

## Test

```bash
.venv/bin/pytest
```

Tests build the pipeline from three saved games in `tests/fixtures/`, so they need no download.
`tests/test_full_data.py` also checks the full database against public facts (e.g. a fired
coach's record) and skips until the data is downloaded and built.
