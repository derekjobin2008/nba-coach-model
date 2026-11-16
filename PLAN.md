# Project Plan

<!-- Claude keeps this file current; see "Keeping PLAN.md" in CLAUDE.md. -->

## The App

A model of opposing NBA coach behavior. Most analytics model players; Derek wants to model
coaches. Public play-by-play logs every substitution and timeout, so each coach's decision
policy can be learned from it. Outputs: (1) a forecast of what the other bench will do in a
given game state, (2) an audit of where a coach departs from a win-probability-optimal policy.
For Derek (Systems Engineering & Design, operations research concentration) to build OR,
data analytics, and SQL skills, and eventually to show NBA teams.

Phases:
1. Data and decision events: play-by-play → DuckDB → decision events table with reconstructed
   game state (margin, time, period, fouls, opponent run, minutes, timeouts left, rest, home/away),
   validated against hand-checked games.
2. Baseline claim: predict "timeout at next stoppage" and "player subbed out"; pooled model vs.
   hierarchical coach effects; evaluate on a held-out season (log loss, calibration).
3. OR layer: timeout/sub decisions as an MDP / optimal stopping problem; compare revealed vs.
   optimal policy in win probability; later a pre-game advance-scout report.

## Done

- 2026-10-11: Repo scaffolded (data/, sql/, src/coachmodel/, notebooks/, tests/), virtualenv + requirements.
- 2026-10-11: Checked nba_api feasibility (see Decisions/Problems).

- 2026-10-11: Ingestion (`coachmodel.ingest`): caches raw game lists + PlayByPlayV3 JSON per game, with retries and pauses.
- 2026-10-11: SQL pipeline (`coachmodel.build_db`): `games`, `pbp` (clock parsed, score carried forward with window functions), `decision_events` (timeouts with timeouts-used count and a likely-mandatory flag; subs), `lineup_changes` view. 6 pytest tests on two fixture games.
- 2026-10-11: Derek hand-checked both validation games: timeouts (team, clock) match the broadcast record, and stint minutes match ESPN's box score (Haliburton 36.5 -> 37, Brunson 34.4 -> 34).

- 2026-10-11: Incoming-player IDs for substitutions (`player_names` lookup, 4 tiers; 99.98% resolved on 2024-25). Tier 4 matches first-name prefix + last name against each team's season roster using the all-players list (`data/raw/players.json`).
- 2026-10-11: `decision_state` (sql/05): per decision, own/opponent team fouls and penalty, outgoing player's personal fouls, opponent unanswered run, timeouts remaining (regulation), rest days. Built with ASOF JOINs so each row only sees earlier plays. Team-foul counts match the feed's T# on all 12,083 numbered fouls. 11 tests.
- 2026-10-11: `stints` (sql/05): on-court stretches per player, with period starters inferred (the feed never lists them). 99.3% of team-games sum to exactly 5 x game length; misses trace to the few unresolved entrants. `decision_state` gains outgoing player's minutes so far and current stint length. 13 tests.
- 2026-10-11: Derek confirmed from the broadcast: the flagged Q1 timeouts in MIN@DEN G7 were TV (mandatory) timeouts, and the Q4 3:05 pair was two coach timeouts at one stoppage (neither flagged, correctly).
- 2026-10-11: `team_state` (state after every play, per team) shared by `decision_state` and the new `stoppages` table: one row per dead-ball stoppage per team, labeled `called_timeout`, with `has_mandatory_timeout`. 7,914 of 7,914 coach timeouts land on a labeled stoppage. Timeout rate rises from ~1% (no opponent run) to ~13% (7-10 point run). Validation games: 8 coach + 3 TV (MIN@DEN), 8 + 2 (NYK@IND). 16 tests.
- 2026-10-11: Derek ran a query on `stoppages`: trailing teams call more timeouts (4.3% down 10+ vs 2.6% up 10+). "Tied" looked low (2.05%) because ties cluster in Q1; in Q4/OT tied stoppages have the highest rate (7.1% vs 5.4%), a Simpson's paradox and a reminder that one-way tables mislead.
- 2026-10-11: `stoppage_players` (sql/09): each on-court player at each stoppage, labeled `subbed_out`, with fouls, minutes, stint length. 99.95% of team-stoppages have exactly 5 players; 99.7% of subs land on a labeled row. Sub rate climbs with fouls (13% at 4 fouls, 39% at 5, Q1-Q3) and stint length. 19 tests.
- 2026-10-11: Derek tried `stoppage_players` via the foul-trouble question; his hypothesis held: a 4-foul player is subbed at 13.2% of Q1-Q3 stoppages vs 4.5% in Q4/OT (5 fouls: 38.7% vs 7.2%); heavy-minutes players with 4 fouls in close games: 18.5% vs 3.0%.
- 2026-10-11: `game_coaches` (sql/10): head coach per team per game, from the roster endpoint plus a hand-kept `data/coach_changes.csv` of in-season changes (ASOF join on game date). 20 fixture tests + full-data checks of fired coaches' records.
- 2026-10-12: Fixed a label leak in `stoppages` (see Problems). Bench actions are now assigned to the dead ball they respond to (`bench_action_stoppages`); 84% of coach timeouts and 90% of subs land on a stoppage, the rest are live-ball actions left out of these tasks. Timeout rate: 0.8% with no opponent run, ~13% during an 8-point run. 21 fixture tests.
- 2026-10-12: Pooled baseline timeout model (`coachmodel.baseline`, features in `sql/models/timeout_features.sql`): logistic regression with stoppage cause, clock spline, margin, run, fouls, timeouts left. Temporary split on 2024-25 (train before 2025-03-01, 861 games; test after, 344 games): log loss 0.0960 vs 0.1215 for a constant rate (21% better); calibrated within ~1 point per bin, slightly high in the middle bins. 25 tests. Derek reproduced it; with the split moved to 2025-01-15 (560 train / 645 test games) it is still 21.6% better.
- 2026-10-12: Pooled baseline for the substitution task (`--task substitution`, features in `sql/models/substitution_features.sql`): fouls, foul trouble, minutes and stint-length splines, TV-timeout flag, game state; no player identity. Same 2024-25 split: log loss 0.1125 vs 0.1373 constant (18.1% better), calibrated within ~0.7 points per bin. `stoppage_cause` moved into the `stoppages` table so both tasks share it. 28 tests. Derek has seen the results but not run it himself yet.
- 2026-10-12: Preview of the first claim: per-team actual vs expected (pooled model) on the held-out 2024-25 games varies far beyond noise. Subs: NYK 0.61x (Thibodeau's short rotation), GSW 1.41x, MEM 1.50x. Timeouts: UTA 0.39x, DEN 1.67x, IND 1.63x. The pooled model is right on average and wrong per bench.
- 2026-10-12: Downloads recovered: 2021-22, 2024-25 complete; 2022-23 1,228/1,230; 2023-24 in progress.
- 2026-10-12: Head coaches now come from a hand-kept `data/head_coaches.csv` (150 opening coaches + 14 in-season changes); the roster endpoint turned out to include off-season hires (see Problems) and is now a cross-check test. Fired coaches' 2024-25 records match (Brown 13-18, Jenkins 44-29, Malone 47-32).
- 2026-10-12: **First claim holds (preliminary).** `coachmodel.coach_effects`: pooled model + one partially pooled effect per head coach (random intercept, penalized likelihood, tau by 5-fold CV over games). Trained on 2021-22, 2022-23, partial 2023-24, and 2024-25 before March; tested on 2024-25 from 2025-03-01. Timeouts: log loss 0.0961 -> 0.0954 (0.72%, 95% bootstrap 0.32% to 1.12%); median error in a coach's timeout total 21% -> 14%. Subs: 0.21% (0.08% to 0.34%); count error 16% -> 12%. Effects: timeouts Mazzulla 0.57x to Carlisle 1.89x odds; subs Thibodeau 0.67x, Budenholzer 1.50x. 35 tests. Derek has not run it yet.
- 2026-10-12: Nested coach + coach-team stint effects (Derek's choice). Trained with 601 games of 2023-24 now in. Timeouts: tau_coach 0.2, tau_stint 0.2; 0.68% better than pooled (0.29% to 1.08%), same as coach-only; coach-total error 22% -> 15%. Timeout style partly travels (Vogel 1.78x LAL / 1.73x PHX, Nurse 0.66x / 0.72x) and partly not (Bickerstaff 0.78x CLE / 1.10x DET). Subs: tau_coach 0.1, tau_stint 0.1; 0.29% better (0.18% to 0.42%), better than coach-only (0.21%); count error 16% -> 12%; sub style travels closely (Rivers 0.89x / 0.89x, Vogel 1.14x / 1.15x). Popovich and Mitch Johnson nearly equal on the same Spurs roster: coach vs roster confound, which player effects would address. 37 tests.
- 2026-10-12: Possession inference (`sql/06_possession.sql`): team with the ball after every play, from made/missed shots, rebounds, turnovers, fouls, last free throws; 99.9% consistent with who takes the next shot (the only "miss" is offensive-foul turnovers, which repeat the foul's change). `team_state.has_ball`.
- 2026-10-12: Win probability model (`coachmodel.win_probability`), Phase 3's yardstick: logistic regression on margin/sqrt(t), ball/sqrt(t), sqrt(t). Log loss 0.487 vs 0.692 constant. Leave-one-season-out calibration within ~4.6 points per bin (late-game bins within ~1.6); the residual tracks each season's home edge (54-58%). Sanity checks pass (tip-off 56.1%, up 10 with 2:00 left 98.6%, tied last shot 59.8% vs 44.0%). Charts for both validation games in `reports/`; Derek checked them against the games. 42 tests.
- 2026-10-12: Player effects for substitutions (tau_player 0.4 by CV): improvement over pooled rises from 0.29% to 0.69% (95% 0.43% to 0.94%). Stars are rarely pulled (Gilgeous-Alexander 0.16x, Durant 0.22x, Booker 0.32x, Doncic 0.34x); bench bigs and minutes-managed players often (McGee 2.50x, Isaac 2.67x). Coach effects after removing roster: Thibodeau stays 0.82x (his short rotation is style, not roster); Rivers 0.93x -> 1.01x and Udoka 0.94x -> 1.02x (their low sub rates were their stars); Mike Brown 1.13x -> 1.25x.
- 2026-10-12: Coaches' revealed foul-trouble policy (`sql/models/foul_trouble_coach_policy.sql`): starters' on-court share collapses once fouls exceed the quarter (Q1 2 fouls 20%, Q2 3 fouls 17%, Q3 4 fouls 23%) and the rule vanishes in Q4 (5 fouls: 66%).
- 2026-10-12: **Phase 3 foul-trouble MDP** (`coachmodel.foul_trouble`): one starter, state (minutes left, fouls, minutes played, margin), play/sit each minute, solved by backward induction; win at the buzzer as the reward. Inputs from data: foul rate 0.076/min (2.7 per 36), margin noise 2.70/min (calibrated to the win probability model), minutes cap = coaches' rotation without foul trouble (Derek: cap in the context of the player's typical minutes). Results for a starter 10 pts/48 better than his backup: optimal 63.82%, coaches 63.15%, same rotation ignoring fouls 63.30%. Coaches' benching halves foul-outs (2.6% -> 1.5%) but costs 0.3 min and 0.15 WP points per game; optimal gains ~0.67 points per game (~0.55 wins per season per such starter). Simulation of 200k games per policy matches the DP. 48 tests. Derek has not run it yet.
- 2026-10-12: Finding from the MDP: with a constant foul rate and constant player value, *when* a starter plays barely matters (foul-out risk depends on minutes played, not their timing, and a point counts the same early or late); only total minutes and margin-aware timing do. So the expected sanity behaviors "sit early with a high foul rate" and "play more late" did not appear; the model is nearly indifferent, for that reason.
- 2026-10-12: Foul-trouble effects (`coachmodel.foul_trouble_effects`), each player vs his own low-foul minutes in the same quarter (removes the big-man selection). Fouls: 0.55x in foul trouble (95% over players 0.51 to 0.59); Q4 with 5 fouls 0.51x. Derek's observation that opponents target a player in foul trouble may be real, but the net is far fewer fouls (caution dominates). Value: no measurable drop; Q4 5-foul points scored and allowed both 1.00x; overall points allowed 1.018x and scored 1.036x move together (pace late in quarters), net +2 per 48. Measured foul-rate drop now feeds the MDP: foul-outs halve (ignoring fouls 1.3%), audit unchanged (coaches' benching -0.16 WP points per game for a 10-point starter; optimal +0.68). 52 tests.
- 2026-10-12: **Per-coach foul-trouble audit** (`coachmodel.foul_trouble_audit`): each coach's revealed on-court shares by quarter and foul count (pooled toward the league with 10 minutes of prior per cell), run through the MDP for a starter worth 6 pts/48, compared with his own rotation ignoring fouls (so rotation size does not count), cap = his rotation's minutes, 95% range from 200 game resamples. 25 coaches with 100+ close games: 24 lose win probability to their foul handling, 22 with ranges excluding zero. Largest cost Spoelstra (-0.18 WP points per starter-game, ~-0.75 wins per season over 5 similar starters; starters on court 16% in Q1 with 2 fouls); smallest Daigneault (~0, starters on court 45% in Q1 with 2 fouls) and Mosley. 55 tests. Derek has not run it yet.
- 2026-10-13: Minutes-recovery check (`coachmodel.foul_trouble_minutes`): starters with early foul trouble (2+ fouls in Q1 or 3+ by half) vs their own other close games, excluding foul-outs: first half -1.90 min, second half -0.32 (95% -0.52 to -0.12), whole game -2.22. None of the minutes come back (the extra fouls get them benched again in Q3), and they still foul out more (6.6% vs 1.6%). The MDP's assumption that benched minutes are lost holds, so the audit stands. 58 tests.
- 2026-10-13: Player values (`coachmodel.player_value`): RAPM by ridge regression on 129k lineup segments (`sql/models/lineup_segments.sql`, points match box scores within 0.1 per game), segments under 30 s downweighted, ridge strength by 5-fold CV over games (beats a no-player model on held-out games, +0.75%). Top: Jokic +11.9 per 48, Gilgeous-Alexander +8.6, Giannis +8.5, Kawhi +7.8, Curry +7.2. Starters over bench per coach: median 1.9 per 48 (range -1.9 to 6.7), likely understated by ridge shrinkage.
- 2026-10-13: Per-coach audit now uses each coach's own starter-over-bench value (floored at 0). Largest cost Malone (-0.11 WP points per starter-game, ~-0.47 wins; starters +6.7 over bench, benched earliest); Daigneault still the only coach at or above zero. 19 of 25 ranges exclude zero; rank agreement with the fixed 6-point version 0.65. Thibodeau, Billups, Unseld near 0 because their starters barely beat their benches by RAPM (treat with caution). 62 tests.
- 2026-10-13: All five seasons downloaded (2021-22 and 2024-25 complete; 2022-23, 2023-24, 2025-26 missing game numbers 602 and 723 each, to investigate) with head-coach roster files. Full database rebuilt (8.5M player-stoppage rows). Roster cross-check passes for every season it can verify; Prunty's 2023-24 interim games check out. 63 tests, none skipped.
- 2026-10-13: **Real test of the first claim: it holds on the full held-out 2025-26 season** (train 2021-22 to 2024-25, 4,918 games; test 1,228 games). Pooled baselines: timeouts 22.9% better than a constant rate, subs 16.9%. Coach effects over pooled: timeouts +0.55% (95% 0.40% to 0.70%), coach-total error 15% -> 11%; subs (coach + stint + player) +1.28% (1.17% to 1.39%), +0.63% without players, coach-total error 21% -> 14%. Coaches in new jobs (timeouts): carry-over helped Borrego (-29% -> -25% error) and hurt Mike Brown at NYK (-3% -> +7%); unseen coaches (Ott, Splitter) get the league average. Derek has not run it yet.
- 2026-10-13: The "missing" games 602 and 723 in three seasons were ordinary games that hit the API quota (each season's run hit it at about the same request count); retried, all five seasons now have 1,230 of 1,230 games.
- 2026-10-13: Possession (`has_ball`) added to the timeout features: 2025-26 log loss 0.0976 -> 0.0975 (22.9% -> 23.0% better than constant). Almost no gain because the stoppage cause already implies possession (after an opponent's basket you have the ball); only team rebounds split (timeout rate 2.2% with the ball vs 0.95% without).
- 2026-10-19: **Bench Brain scout page live** at https://derekjobin2008.github.io/nba-coach-model/ (`coachmodel.scout_report`, GitHub Pages from `docs/` of the public repo). Pick one of 49 head coaches: timeout and substitution tendencies vs an average coach in the same situations with 95% ranges (Laplace), foul-trouble shares vs league and the audit's cost with its range (27 audited coaches), and a by-team table for coaches who moved. Checked at 375 px wide; Derek opened it and read both answers for Nick Nurse (29% fewer timeouts, fewest of 49; foul-trouble benching ~0.10 wins a season). 67 tests, two of which run the page's own JavaScript.
- 2026-10-19: **Honest ranges check** (`coachmodel.range_check`). Finish line (agreed with Derek): on seasons the model never saw, 95% ranges contain about 95% of coaches. Rolling-origin design (fit on earlier seasons, predict each of 2022-23 to 2025-26; 108 timeout and 112 substitution coach-seasons), z = (next-season effect - estimate) / combined standard error. The page's original ranges were overconfident: timeouts 67% inside (z sd 1.90), substitutions 36% (z sd 5.04). Joint Laplace (coach vs team vs player uncertainty) fixes most of it (87% / 67%); game clustering adds almost nothing; the rest is real season-to-season change in coaches (sd 0.105 log-odds, ~11%, for timeouts; 0.176, ~19%, for substitutions). With all three, cross-fitted: **94% / 96% inside, z sd 0.97 / 1.01**. The page now shows two ranges per coach: his long-run tendency (joint + clustered; decides "clearly different from average": 22 of 49 coaches for timeouts, 12 of 49 for substitutions) and a single season (adds drift; the range that passed the check). 71 tests.
- 2026-10-19: Derek moved on from the cleaner range layout without the 5-second phone check ("keep going").
- 2026-10-19: **Team strength in win probability** (`sql/12_team_rating_inputs.sql`, `coachmodel.win_probability --by-season`). Pre-game rating per team from earlier games only (window frame ends one row before the game) plus last season: (margin so far + 10 games x half of last season's margin per game) / (games + 10), weights fit on game margins. Leave-one-season-out log loss improves every season (e.g. 2025-26 0.475 -> 0.437). Finish line was "middle calibration bins within 2 points", which turned out to be below the noise floor (each bin holds a few hundred game outcomes, chance sd about 2 points); corrected to "within chance": worst middle bin 1.3 to 2.3 chance-sd in every season (lower than without ratings in every season). The foul-trouble MDP's margin noise moves from 2.70 to 2.85; the per-coach audit shrinks about 6% (Spoelstra -0.50 -> -0.47 wins) with an identical ranking of all 27 coaches, so it is robust to this change. Page rebuilt. 73 tests.
- 2026-10-19: **Timeout values** (`coachmodel.timeout_value`, `sql/models/timeout_run_effects.sql`, `timeout_endgame.sql`). Finish line (Derek agreed): each value with a 95% range. (a) During an opponent run of 6+ (82,729 stoppages), a coach timeout is worth -0.10 points over the next 3 minutes vs similar stoppages without one (-0.18 to -0.01, game-clustered), adjusting for run, score, time, possession, cause, fouls, timeouts left, team strength; TV timeouts (forced by the clock, a natural comparison) -0.17 (-0.48 to +0.15); runs of 8+ -0.10, 10+ -0.04, both ranges include 0; run length predicts nothing (no momentum). Raw comparisons were misleading because coaches call timeouts with the ball (98% vs 82%), worth +0.88 points. (b) One extra usable timeout at 2:00 of a game within 6 points: +2.1 win probability points (-1.3 to +5.5; few teams have fewer than 2, so the range is wide); the win probability model's coefficient there is 1.06 (well calibrated). 75 tests. Derek has not run it yet.
- 2026-10-26: **Per-coach timeout audit** (`coachmodel.timeout_audit`). Finish line (Derek agreed): per coach, how often his timeout use leaves him short at the end of close games and the cost in wins per season, with a 95% range carrying both measured values' uncertainty. Since a run timeout is worth ~0 and only 2 timeouts may be used in the last 3 minutes, the optimal policy is immediate: never spend one that could leave you under 2. League: 12% of teams are short at 2:00 of a game within 6. 42 coaches with 40+ close games: most short Vogel 32% and Carlisle 31% (~0.22 wins per season, ranges -0.12 to +0.6), least Mazzulla 1% and McMillan 0%. Run timeouts per game and being short correlate 0.50 across coaches, but no coach's range excludes zero: timeout habits cost at most about half what foul-trouble habits do. Upper bound (timeouts have other uses). 78 tests. Derek has not run it yet.
- 2026-10-27: Derek completed the CS 124 lesson "Discuss Ideas, Not Implementations" (all four steps).
- 2026-10-27: **Forecast checks for a predictive scout page** (Derek's request: pick a coach and game state, see likely moves with confidence; must beat baselines on unseen games before any page work). `coachmodel.forecast_check`, 2025-26 held out: timeouts beat the coach's own average by 23.3% and the league model by 0.55% (0.40 to 0.70); subs 17.8% and 1.28% (1.17 to 1.40). Derek pointed out the coach-average baseline ignores game state, so the league model is the comparison that matters. `coachmodel.coach_signal`: coach habits are reliable (split-half correlation 0.95 timeouts, 0.94 subs; every coach 10%+ from average in both halves points the same way) but concentrated: 22 of 49 coaches clearly differ for timeouts (middle half 0.87x to 1.16x), 12 of 49 for subs (0.94x to 1.10x). Ranked list of who comes out: top pick right 52.8% (league model 49.6%, random 28.5%). 2025-26 subs are underpredicted in every calibration bin (coaches subbed more that season). Recommendation to Derek: lead the page with the game-state forecast and flag only coaches who clearly differ; first add a current-season league-rate adjustment. 80 tests. Waiting on Derek's decision.

- 2026-10-28: Derek completed the CS 124 lesson "Correct It, Out Loud, Early" (all four steps).
- 2026-10-28: **Season weighting** (`coachmodel.season_replay`). A coach-season effect on top of each coach's long-run habit, partially pooled with the measured drift as its spread, so this season counts more as its games accumulate and past seasons always stay in. Replay of 2025-26 a month at a time (refit on earlier games only, same games for both forecasts): timeouts +0.04% log loss (-0.007% to +0.09%): **no measurable difference**; subs +0.11% (+0.06% to +0.16%): a small measured gain, though October got worse (-0.39%, not yet explained). Sub calibration bins under the observed rate: 5 of 10 without, 1 of 10 with, so the league-wide season shift is not needed (Derek's condition). Kept for both tasks: for subs because the replay shows a gain, for timeouts because the drift-based design is sound, not because the replay showed it works.
- 2026-10-28: **"Most likely next moves" card** leads each coach's report: his top 3 moves at the next stoppage of a close game (within 5 points, last 6 minutes of the 4th, his latest season, playoffs included), each with a 95% range and the average coach's chance; a tag only where his usual habit clearly differs from the league. Close-game numbers are two constants in `scout_report.py`. Existing page numbers unchanged (largest change 0.0006).
- 2026-11-03: Derek completed the CS 124 lesson "Validate What It Built" (all four steps).
- 2026-11-03: **Team-first coach picker** (classmate feedback: team names next to coaches, boxes instead of a dropdown, easier for non-fans). 30 boxes, one per team, with the full team name, current coach (from `data/head_coaches.csv`, so interims Splitter and Borrego) and his games; clicking highlights the box and scrolls to his report; links like `#Steve%20Kerr` still work.
- 2026-11-03: **League tag threshold** (Derek's red-green change): a tagged move within 2 percentage points of the league reads "about the same as the league" (16 of 44 tagged moves); 2 or more keeps "more/less often". The tag has its own row so it never forces sideways scrolling.
- 2026-11-04: Derek completed the CS 124 lesson "Say What You Saw" (all four steps).
- 2026-11-04: **Former coaches collapsed** into a "Former coaches (19)" section that opens on click, or by itself from a link to a former coach. Derek tried the page and said it works. 90 tests.
- 2026-11-05: **Redesign, first piece**: team-colored coach header, three stat tiles (timeouts, subs, foul trouble) with a plain-words meaning and a chip, "this coach" vs "average coach" bars, a "What do the ranges mean?" explainer, and after a tap the team grid shrinks to a strip pinned at the top with "Show all teams and former coaches".
- 2026-11-07: **Arena night colors and new fonts** (Derek chose from three sketches; the old fonts "felt too basic"): navy and slate with an electric-blue accent; Barlow Condensed for titles and numbers, Inter for sentences. Every text color checked at 4.5 contrast or more in both themes.
- 2026-11-10: Derek completed the CS 124 lesson "Say What You Saw" (step 3, wanting graphs under Timeouts and Substitutions) and "Ask What Would Change" (2026-11-11).
- 2026-11-11: **Shared links land on the card** (Derek's idea from the lesson): a link like `#Billy%20Donovan` now opens like a tap (strip pinned, page scrolled to his card). Tests written red first; checked in a phone-sized browser. 96 tests. Derek asked to commit these without his own look (2026-11-16).
- 2026-11-16: A picked former coach shows in the pinned strip (his box, with his name). Not yet tried by Derek.
- 2026-11-16: **Regression tests for every reported page bug**, each seen failing with the old bug put back: real-browser tests in headless Chrome (no install: the Chrome already on the laptop, through its DevTools protocol), a color-contrast test, and a fake browser that stores attributes like a real one. 152 tests, about 16 seconds.
- 2026-11-16: **End-to-end test** (`tests/test_end_to_end.py`, Derek's partner's suggestion): the three saved games run through the SQL pipeline, the model fits and the page builder, then Chrome clicks each coach and the card's timeout and substitution sentences must match a fresh model fit; the "not enough data" states must show instead of errors. Seen failing with timeouts and subs swapped in the page builder, and with "more" and "fewer" flipped on the page. Writing it found three bugs that full data hid (see Problems). 156 tests.
- 2026-11-16: **Model benchmark** (`coachmodel.benchmark`, Derek's partner's suggestion): every rung scored on the frozen 2025-26 test, results appended to `reports/benchmark.csv`, tests that fail if the latest coach model stops beating the league model or a run used different test games (each seen failing on a planted bad row). First results, gain over the rung below: timeouts, coach model vs league model +0.55% (+0.41% to +0.69%), season-weighted vs coach model no measurable difference (+0.008%, -0.006% to +0.021%); substitutions, coach model +1.28% (+1.17% to +1.39%), season-weighted **worse**, -0.31% (-0.36% to -0.27%).
- 2026-11-16: `tests/test_page_matches_model.py`: the published page's timeout numbers for all 49 coaches match a fresh fit (fails, naming the coach, if the page is stale; seen failing with one number nudged 1%).
- 2026-11-16: Typing or pasting a different coach into the address bar of an open page now switches to him like a tap (it did nothing until a reload). 163 tests.
- 2026-11-16: **Range coverage pass line**: the page's timeout ranges contained 94% of coaches' next seasons (108 coach-seasons, z spread 0.97); recorded in `reports/range_check.csv`, test seen failing on a planted 67% row. Substitutions still to run (about an hour).

## Next

1. **A more engaging page, continued** (Derek, 2026-11-04: look like ESPN-style analytics but more dynamic; easy for non-fans, still useful to fans). Done check: a classmate who does not follow the NBA names one thing a coach does differently within about 10 seconds, and a fan finds a takeaway without opening Method. Open: a former coach's box is missing from the pinned strip (found 2026-11-11); graphs under Timeouts and Substitutions (Derek, 2026-11-10; which question the graph answers is still his to choose); let the visitor change the score and time and watch the moves update.
2. **Game prediction** (classmate feedback, Derek wants it): predict this season's games from the model. Needs its own baseline test (for example the better record wins) before it goes on the page. Head-to-head coach comparison was also suggested.
   Derek's full vision (2026-11-05): enter two teams for an upcoming game, pull injuries and lineups plus history, estimate the winner, and accept Kalshi or Polymarket prices to say whether a bet is worth taking. Agreed caveats: live data needs a backend or a scheduled rebuild; the model must beat market prices after fees on past games before the page calls any bet worth it; the page shows the model's edge and expected value, not how much to stake.
3. **Build the validation plan below**, starting with the pass-line tests and the end-to-end test. Decide first whether "close game" should include playoffs.
4. **Season weighting for substitutions: decide** (Derek's call). The benchmark (fit only on seasons before 2025-26, so no current-season games) found it measurably worse for subs, -0.31%; the monthly replay (current-season games added month by month) found +0.11% overall but -0.39% in October, the one month with no current-season games either. So both agree: it hurts until this season's games arrive, then helps. A likely reason, not yet tested: the model treats each season's deviation as independent around a fixed habit, so with no games yet it predicts last season's quirks vanish, while real habits carry over from one season to the next. Test: how strongly a coach's season effect in one season predicts the next. Options: keep it (the page's likely-moves card uses a coach's latest season with its games already in), model the drift as carrying over, or switch the card to the coach model.
5. **Keep making the models statistically stronger** (Derek, 2026-10-19: the page is not the end). Candidates: a fully Bayesian check on a subsample; whether drift differs by coach; the foul-trouble open questions below.
6. Foul-trouble audit, open questions: starters may be worth more in close late games; RAPM shrinkage understates starter-over-bench gaps (a box-score prior would help); test Derek's targeting idea directly (do opponents attack a player in foul trouble?).
7. Timeouts remaining in overtime and the 4th-quarter limits (max 4 in Q4, max 2 after 3:00).

## Validation plan

(2026-11-10, from Derek's questions on how to know it works without taking anyone's word for it.) "It works" is four claims, each with its own check; pass lines are written before results are seen.

1. **It runs.** Exists: 152 pytest tests: pipeline on three fixture games, page logic in Node with a fake page, page colors, and real-browser tests in headless Chrome at phone and laptop widths in both themes (2026-11-16; driven through Chrome's DevTools protocol by `tests/chrome_driver.mjs`, so nothing extra is installed; they skip without Chrome). To add: CI (GitHub Actions) running `pytest` on every push.
2. **The page shows what the model computed.** Exists (2026-11-16): the end-to-end test below, on the three fixture games, and `tests/test_page_matches_model.py`, which refits the timeout model on the full database (15 seconds) and fails if any coach's number on the published page is stale. Substitutions are too slow to refit in a test (an hour); the benchmark covers them.
3. **The model holds up on unseen games.** Exists: the model benchmark below (2026-11-16) with pass-line tests and a results log; run by hand: `forecast_check`, `range_check`, `season_replay`. Range coverage is a pass line too (2026-11-16): `range_check` appends to `reports/range_check.csv` and a test fails outside 90% to 99% (Claude's threshold: two binomial standard deviations around 95% with about 100 to 150 coach-seasons; Derek can change it). To add: forecasts logged for upcoming games and scored after; break-it-on-purpose checks that each test can fail; one number re-derived independently by a separate agent from the raw SQL.
4. **People get something from it.** To add: a simulated skimming non-fan (an agent with only the link) as a rehearsal; then Derek's done check with a real classmate, one fan and one non-fan: a non-fan names one thing a coach does differently within about 10 seconds, and a fan finds a takeaway without opening Method.

**End-to-end feature integration test** (Derek's partner's suggestion, 2026-11-10). One test that runs the whole path a real visitor depends on, with nothing faked in between:
saved play-by-play for the three fixture games -> `build_db` (all the SQL) -> coach-effect fits -> `scout_report` writes the page -> a real browser opens it -> click a team box -> read the card.
Pass lines: the card shows that team's coach; each probability on the card equals the model's own number for that coach (to the shown rounding); a coach with too few games shows the "not enough" message instead of crashing; no browser errors. It catches what the separate tests cannot: a step handing the next one stale, renamed or flipped data (the kind of bug where every piece passes on its own). It misses whether the model is right (claim 3) and whether people understand the page (claim 4). Three games make the fits weak but fast (seconds, not the 40-minute full refit); a second version runs on the full database and skips when the data is not downloaded, like `tests/test_full_data.py`. Needs `scout_report` to accept an output path so the test writes to a temporary folder, not `docs/`.

**Model benchmark** (Derek's partner's suggestion, 2026-11-10, read as a fixed benchmark for the models, the way machine-learning models are compared on a standard test set). One frozen setup that every model version is scored on, so "better" always means the same thing:
- Frozen split: train on seasons before 2025-26, test on 2025-26 (regular season; playoffs as a separate, smaller set once decided). The test games never change and are never used for fitting or for choosing settings.
- Fixed tasks and scores: timeout at a stoppage, sub-out at a stoppage; log loss (main score), calibration by bin, and a game-resampled 95% range on each gain.
- A ladder of baselines every new model must climb: the league-wide rate; the coach's own average; the pooled league model; the current coach model. A new idea (season weighting, a Bayesian fit, a tree model) enters the table only if it beats the current best and its range excludes zero; otherwise it is recorded as no measurable difference.
- A results table kept in the repo (one row per model version: date, commit, scores, ranges), written by one command, so improvements and regressions are visible over time without trusting anyone's summary.
- Pass line as a test: the current model must still beat the league model on the benchmark, or the test fails.
It catches: models that only look better because the test set or the score changed, and a quiet regression.

**Regression tests for reported bugs** (Derek's partner's suggestion, 2026-11-10). Rule from now on: every bug Derek, a classmate or Claude reports gets a test that fails on the old code and passes on the fix (red, then green), named after what was seen, and listed under Problems next to the bug. A test only counts if it was seen failing first. Past bugs, checked 2026-11-10:
- Covered: unresolved substitution names, the 4th roster tier (`test_garbage_time_entrants_resolved_from_season_roster`, `test_every_substitution_names_both_players`); NaN written as invalid JSON (`test_missing_numbers_become_null_and_script_tags_cannot_break_out`); the 2-point league tag (red-green when built).
- Done 2026-11-16, each seen red with the old bug put back, then green: the team box not highlighted (the fake browser now stores attributes like a real one, and `tests/test_page_in_browser.py` checks it in Chrome); a line too wide for a phone (now caught as the page scrolling sideways or, since the redesign's card hides overflow, a line cut off at the card's edge, for all 49 coaches in both themes); the dark-mode tag contrast and the dark-mode bar (`tests/test_page_colors.py`: every text pair at 4.5 or more in both themes, white on all 30 team colors, the bar at 3 or more); a shared link leaving the card below the grid; a picked former coach missing from the strip.
- Found while writing them, not fixed (Derek's call): the gray "Average coach" bar is 2.56 against the white card in light mode, under the 3 usually asked of shapes; it always has its percentage beside it.
It catches: an old bug coming back when nearby code changes. It misses: bugs nobody has reported yet, and a test written so it can never fail (hence seeing it fail first). It misses: overfitting to the benchmark itself after many tries (a fresh holdout, such as 2026-27 games scored as they are played, guards against this) and whether the findings are true about basketball.

## Decisions

- 2026-10-11: Python + DuckDB. Derek wanted DuckDB or Postgres; DuckDB chosen because it is a
  single file with no server, fast for analytics, and has full CTE/window-function SQL. Postgres
  stays an option if a multi-user web app needs it later.
- 2026-10-11: Data source is nba_api PlayByPlayV3. PlayByPlayV2 now returns empty JSON
  (nba_api issue #591), and the cdn.nba.com live JSON returned 403. pbpstats is built on those
  older feeds, so it is likely unusable for fresh pulls (not tested).

- 2026-10-11: Seasons: train on 2021-22 to 2024-25, hold out 2025-26. Starts after the
  COVID-shortened schedules, which would distort rest days and other features. Derek agreed.
- 2026-10-11: Python only downloads; all parsing happens in SQL (DuckDB reads the raw JSON
  directly), so Derek gets SQL practice and the raw data stays reproducible.
- 2026-10-11: Mandatory timeouts are flagged (`likely_mandatory`), not dropped, because the
  rule-based guess is unverified and dropping would hide mistakes.
- 2026-10-11: Coach's challenges are excluded from timeouts (a different decision).
- 2026-10-11: State features moved into one `team_state` table (every play x both teams)
  so decisions and stoppages share one definition instead of two copies of the SQL.
- 2026-10-11 (revised 2026-10-12): Stoppage = all dead-ball plays at one game clock (made
  shots, fouls, FTs, violations, jump balls, replays, non-steal turnovers, team rebounds).
  Timeouts and subs belong to the previous dead ball if within 4 s with no live play in
  between (threshold read off the gap distribution). State is read at the stoppage's first
  play. Live-ball timeouts (16%, e.g. after the team's own rebound) are not modeled yet;
  they could become a separate "possession start" opportunity later.
- 2026-10-11: Head coaches: roster endpoint + hand-kept change list (Derek chose this over
  the endpoint alone, which only reports each season's final coach, so e.g. Mike Brown's
  2024-25 games would be credited to Doug Christie). Changes researched from each season's
  Wikipedia page and news coverage; 16 coach stints across the five seasons. Dates are each
  new coach's first game where it differs from the announcement (Snyder 2023-02-28, Rivers
  2024-01-29: Prunty coached 3 games in between).
- 2026-10-11: Foul-trouble benching is a strong Phase 3 candidate (an optimal-stopping
  problem); Derek's view: coaches keep starters in late because there is no later game time
  to save them for, but it is situational, which is what coach-level effects will measure.
- 2026-10-12: Baseline modeling in scikit-learn (Derek agreed); feature queries stay in SQL.
  statsmodels considered (better coefficient summaries) but the hierarchical step will
  likely use a Bayesian tool anyway. Finish line agreed with Derek: beat a constant-rate
  log loss on held-out games, and a calibration table that tracks observed rates.
- 2026-10-12: First-claim test fits coach effects as a random intercept by penalized
  maximum likelihood (scipy, written out so the math is visible) on top of the pooled
  model's log-odds, rather than a full Bayesian model, to get an early answer. Derek's
  agreed finish line: lower held-out log loss for both tasks with a bootstrap range above
  zero. Real coaches were used instead of teams once coach data was available.
- 2026-10-12: Coach effects are nested: coach tendency + coach-team stint adjustment, each
  partially pooled (Derek chose this over one effect per coach or per stint, so the model
  can say whether style travels). A coach at a new team starts from his career tendency.
  Kept penalized maximum likelihood rather than MCMC: millions of rows make sampling slow,
  and the most-likely effects are what forecasting needs; uncertainty intervals later.
- 2026-10-12: Win probability is a small parametric logistic model, not boosted trees:
  Phase 3's dynamic programming needs a smooth function that is monotone in margin and
  time. Possession inferred from play types (Derek agreed) rather than left out, since a
  late possession is worth several percent.
- 2026-10-12: Substitution player effects: coach and stint spreads fixed at 0.1 (their
  cross-validated values without players), only the player spread searched; the full
  3-level grid ran over an hour on 4 million rows and was stopped.
- 2026-10-12: Foul-trouble MDP solved on a per-minute grid with margins clipped at +-30, a
  normal margin move each minute, and a tie at the buzzer worth half a win. All policies get
  the same minutes budget (the coaches' rotation without foul trouble, 32 minutes) so the
  comparison is about decisions, not one policy simply playing him more. Player value is a
  sweep (3/6/10 points per 48) until it can be estimated.
- 2026-10-12: Foul-trouble effects measured within player and within quarter rather than
  pooled, because foul-prone players are the ones at high foul counts. The 4th quarter is
  used for the MDP's foul-rate multipliers since coaches rarely bench there (least
  selection). Free throws at the moment of a foul are credited to the foul count it was
  committed with (first version credited them to the next count, which faked a defensive
  collapse after every foul).
- 2026-10-12: Per-coach audit compares each coach with himself (actual policy vs his own
  rotation with fouls ignored), not with the optimal policy, so differences in how many
  minutes coaches give starters do not count as foul handling. Season wins scale the
  per-starter number by 5 starters x 82 games, which overstates it if only some starters
  get into foul trouble.
- 2026-10-13: Player value is RAPM (ridge on lineup segments) rather than raw on/off,
  because raw on/off credits a player for his teammates. Pooled across seasons (one value
  per player) for stability. Value over backup = starters vs the same coach's non-starters,
  minutes-weighted; floored at 0 in the audit.
- 2026-10-16: Derek wants a public copy on his own GitHub account (to show teams). The
  course repo's history contains his reporting link and session transcripts, so it cannot
  be made public or mirrored. `scripts/publish_public.sh` publishes clean snapshots without
  the course files, as a separate history; the course repo stays the main one.
- 2026-10-19: Scout page is one static HTML file generated by Python with the results
  embedded, hosted on GitHub Pages from `docs/` of the public repo: no server or database,
  opens from a link or from disk. The generated page is committed (an exception to "no
  build output in git") because Pages serves it from the repo. Coach effects on the page
  carry 95% ranges from a Laplace approximation, so a coach with little data visibly
  shows a wide range (Derek wants the work grounded in statistics).
- 2026-10-19: Page ranges are season forecasts (joint + clustered standard error plus
  season-to-season drift), not the uncertainty of a coach's career average, because a
  scout uses them to predict this season. Drift is a constant per task from the rolling
  check, stored in `coach_effects.SEASON_DRIFT_SD`, rather than refit on every page build
  (the substitution check takes about an hour). Sandwich errors were first computed as
  H^-1 B H^-1, which drops the prior's share of uncertainty and came out narrower than
  the joint ones; corrected to H^-1 + H^-1 (B - A) H^-1.
- 2026-10-19: Season home edge not modeled: it varies (+1.6 to +2.6 points per game) but a
  season's early games do not predict it (2022-23: +3.18 first half, +1.82 second half),
  and a version that estimated it from earlier games was always fit to ignore them, so it
  was removed. I set the first win-probability finish line (within 2 points) without
  checking the noise floor; lesson: measure what chance alone produces before setting a
  calibration target.
- 2026-10-19: Timeout effect measured by regression adjustment with game-clustered errors
  rather than raw before/after (runs end on their own; coaches time timeouts with the
  ball), with TV timeouts as a quasi-random comparison group since the clock rules, not
  the coach, place them. End-game value uses a linear probability model with the win
  probability prediction as a control so the coefficient reads directly as win
  probability per timeout.
- 2026-10-26: The timeout MDP was solved analytically rather than coded as a grid: with
  spending worth at most zero and only the last 2 timeouts usable at the end, backward
  induction gives "never spend one that could leave you under 2" directly, so the audit
  measures shortfall at 2:00 against that policy.
- 2026-10-11: Validation games chosen by Derek: 2025 ECF Game 6 IND-NYK and 2024 WCSF Game 7 MIN-DEN.

- 2026-10-11: Game state uses ASOF JOIN (latest row at or before the decision) rather than
  correlated subqueries or pandas: one pass, readable, and it structurally prevents leakage.
- 2026-10-11: Team fouls are counted from foul rows, not read from the T# text, because the
  feed stops printing T# once a team reaches the penalty (shows PN instead).

- 2026-10-28: Derek led the page with the coach's most likely moves and asked for this season to count more, weighted up as games accumulate, without dropping past seasons. He rejected adding a league-wide season shift in the same step: "we will not know which one improved the log loss." It was held until the replay showed it was not needed.
- 2026-10-28: Season weight comes from the already-measured drift (a coach-season level with spread 0.105 / 0.176) rather than a hand-picked schedule; Derek liked that it is not "a schedule picked by hand." Close game: within 5 points, last 6 minutes of the 4th (Derek accepted; kept in one place).
- 2026-10-28: Derek corrected the reporting rule: a gain whose range includes zero is "no measurable difference", not "a help".
- 2026-11-03: First change from classmate feedback is the team-first picker, before game prediction: every visitor hits the picker, and a non-fan knows "the Lakers", not the coach's name. Current coaches only in the main grid; a coach who moved appears once, under his current team; no search box.
- 2026-11-03: Tag cutoff of 2 percentage points is Derek's call ("a gap under 2 points ... about the same"); he wants someone who knows basketball to say whether it feels right. Rounded to a tenth of a point so 12% vs 10% counts as exactly 2.

## Problems

- 2026-10-11: PlayByPlayV3 timeouts all have subType "Regular" (plus "Coach Challenge"), so
  mandatory timeouts are not labeled and must be inferred from the rules (first stoppage after
  the under-7:00 / under-3:00 marks with no timeout yet). Open.
- 2026-10-11: PlayByPlayV3 substitution rows carry only the outgoing player's personId; the
  incoming player appears only as a short name in the description ("SUB: Portis FOR Lopez"),
  written differently from playerName ("Ja. Green", "Pöltl" vs "Poeltl"). Resolved without
  extra downloads: the outgoing side of every sub pairs the description's name form with an id,
  with playerNameI and last name as fallbacks (sql/03_player_names.sql). That left 0.9%
  unresolved, which mattered for lineups: all 43 team-periods with 6 inferred starters had one.
  Cause: names like "Je. Green" (Jeff) whose initial form "J. Green" is ambiguous, and suffixes
  ("Jemison" vs "Jemison III"). Fixed with a 4th tier against season rosters; 0.02% remain.
- 2026-10-11: nba_api data frames contain NaN, and json.dumps writes it as invalid JSON. Save
  with DataFrame.to_json, which writes null.
- 2026-11-03: The clicked team box was not highlighted in a real browser although the tests passed: the tests' fake browser stores attributes as strings, a real one as objects. Found by driving the page in a browser; fixed by tracking boxes by coach name. Lesson: page tests in the fake browser can pass while the page is wrong.
- 2026-11-03: Keeping the longer tag on one line made the page scroll sideways on narrow screens; fixed by giving the tag its own row in each move.
- 2026-11-03: Derek could not open the page to check it: the browser pane cannot open local files. Fixed by opening it in his own browser and serving `docs/` on the local network for his phone.
- 2026-11-16: The first real-browser tests took 66 seconds, about 10 per test. The guess was slow Google Fonts; timing showed the page loaded in 0.3 s and the rest was the driver's own 10-second load timeout, still pending and keeping Node alive after the last step. Cancelling it brought the run to 14 seconds.
- 2026-11-16: Headless Chrome will not open a window narrower than 500 px, so a plain `--window-size=375,…` cannot test a phone; the driver sets the exact size through the DevTools protocol instead.
- 2026-11-16: The first end-to-end run, on three saved games, crashed three times on things the full data never shows. (1) Team names came only from home teams, so Minnesota (only away in the fixtures) had none and the model fit crashed sorting a missing name; now home and away teams are both used (identical map on the full data). (2) The foul-trouble audit fit player values before checking whether any coach had the 100 close games it needs, and crashed on one game; it now returns no audited coaches, which the page already shows as "Not audited". (3) The page builder read player names from `data/raw/players.json` relative to the folder it was run from; the path is now explicit. Each has a test.
- 2026-11-16: The end-to-end test's first version compared the page with the page builder's own output, so a mix-up inside the builder (timeouts shown as subs) still passed. Found by putting that mix-up in on purpose; it now compares against a fresh model fit.
