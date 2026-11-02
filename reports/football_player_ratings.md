# Football player ratings and the line-up surprise (2 Nov 2026, development only)

**Bottom line: the ratings look right, but by kick-off the market has priced the line-up fully, and team news as we can
measure it is a small part of the early-to-close move.** The surprise in the confirmed eleven (its rating less the
expected eleven's) explains 0.7% of the move from the early price to the close out of sample (permuted floor 0.02%). It
adds nothing to the closing price's forecast of the result: a log-loss gain of -0.06 per thousand, 90% interval -0.28 to
+0.15. Any line-up edge would have to sit in the minutes between the team sheet and the market's reaction, which only
the line-up collector's timestamps and the recorder's 2-minute books can show. Holdout (2025/26 on) not read.

Figures: `reports/football_runs/player_ratings_dev.json`, from `python -m football.player_ratings` on the API-Football
history (49,206 matches of the 22 divisions, 1 Aug 2020 to 2 Nov 2026, fetched into this session through
`sources/api_football_history.py`; the same counts as the S3 build, run 41466369877, plus one match played since) and
football-data's gold table. Code: `football/player_ratings.py`; tests: `tests/test_football_player_ratings.py`
(recovery of true ratings, stints at substitutions, red cards and own goals, and a rewrite of the last day's closes
leaving that day's surprise unchanged).

## The ratings

| Rating | What it measures | Fit | Players rated (Nov 2026) |
|---|---|---|---|
| Market plus-minus (MPM) | A player's share of his side's closing implied supremacy (log goal-rate units), against the average starter of his pool | Ridge on the starting elevens, home advantage a competition, decayed (half-life about 300 days), refitted monthly on earlier matches | 15,679 |
| Goals plus-minus (GPM) | Goal difference per 90 minutes while he was on the pitch, against the average player | Ridge on 312,555 stints (cut at every substitution and red card), refitted monthly | 17,828 |

Points per goal: 0.64. A side's points a match rise by 0.64 for each goal of goal difference a match, over its season,
in the 22 divisions (`points_per_goal`). So a GPM rating of +0.2 over 3,000 minutes (33 matches) is about 6.7 goals
added, or about 4.3 points.

Highest-rated in November 2026, as a face-validity check, not a finding:

- MPM: Diogo Costa, C. McGregor, L. Rota, Pepê, V. Pavlidis, S. Hezze, H. Kane, M. Akanji, Tomás Araújo, David Raya,
  E. Haaland, J. Butland.
- GPM: H. Kane, G. Til, S. Dest, Mauro Júnior, M. Olise, C. McGregor, B. Mechele, D. Upamecano, Carlos Forbs,
  J. Kimmich, Diogo Costa, Lamine Yamal.

## The line-up surprise against the market (2021/22 to 2024/25, walk-forward by season)

Linked to the gold table: 47,982 of 49,206 API-Football matches (97.5%). With features and a surprise: 46,869.

| Test | Rows | Surprise (MPM) | Surprise (GPM) | XI rating (MPM) | Team features | Team features + surprise |
|---|---|---|---|---|---|---|
| Move: out-of-sample R2 vs mean | 23,417 | +0.67% | +0.70% | +0.29% | +2.44% | +4.10% |
| Move: permuted floor | | +0.02% | +0.02% | -0.03% | -0.53% | -0.59% |
| Result, gain over the open (per thousand) | 30,520 | -0.06 (-0.27 to +0.16) | -0.01 (-0.24 to +0.20) | +0.02 (-0.20 to +0.22) | | |
| Result, gain over the close (per thousand) | 30,520 | -0.06 (-0.28 to +0.15) | -0.05 (-0.28 to +0.16) | 0.00 (-0.22 to +0.21) | | |

## What this means

- **The close prices the line-up.** Nothing in the eleven, rated either way, improves on the closing price.
- **Team news, as measured, is a small part of the move.** The surprise adds about 1.7 points of R2 to the team features'
  forecast of the early-to-close move. Most of the move is something else.
- **Our expected eleven is crude.** It is the last five matches' starters, so it treats known injuries and suspensions
  as surprises, which the market's early price already knew. A better expectation (the injury list, from
  `api_football_lineups` once it runs) would sharpen the surprise. That might raise its share of the move, but cannot
  change the close result.
- **Where an edge could still be:** in the minutes after the team sheet, before the books settle. That needs the time
  each line-up first appeared (the collector) against the recorder's 2-minute books, both waiting on the paper-models
  runner.

Data notes: 40 player rows repeat a player within a match (0.002%; starters are de-duplicated before fitting); 715
substitutions could not be resolved from the events; 207 players have no API-Football id.
