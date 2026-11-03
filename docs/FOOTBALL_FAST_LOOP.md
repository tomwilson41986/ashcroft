# Football: test a feature or a model in about a minute

The slow parts of a football evaluation are the same every time: the Dixon-Coles prices, the market ratings, the
team, xG and player features (about 25 minutes together, measured below). The lab builds them once into a cached
matrix. After that, an experiment reads the matrix, adds feature blocks in seconds and scores its variants
walk-forward. Code: `football/lab.py`, `football/blocks/`. Tests: `tests/test_football_lab.py`.

| Step | Before | Now |
|---|---|---|
| Build every feature | Every run, 25-40 min, plus a 13-min bootstrap | Once (`lab cache`); a code change rebuilds only its stage |
| Add a feature | Edit a feature module, rebuild | One file in `football/blocks/`, no rebuild |
| Leak test | Written by hand per feature | Automatic for every block (`pytest tests/test_football_lab.py`) |
| Test several ideas | One per run | Many arms and models in one `lab run` call |
| Record what was tried | By hand | Every run appended to `reports/football_runs/lab_ledger.jsonl` |

Measured timings for the steps are in section 5.

## 1. Build the cache (once, and after new data)

```bash
python -m football.lab cache --root <sources folder> --history-root <api-football tables> --cache .football_cache
```

Stages, each saved as `<stage>_<key>.parquet`:

1. `dc` - Dixon-Coles prices, each fitted before the match's collection day.
2. `team` - team features.
3. `books` - the early bookmaker prices.
4. `xg` - xG features.
5. `players` - line-up surprise.

A stage's key is a hash of its own code, its inputs (file contents) and the keys of the stages it reads. Editing
`football/xg_features.py` rebuilds `xg` and nothing else; new results data rebuilds everything that reads it.
`manifest.json` lists the keys and how long each stage took.

## 2. Add a feature: one file in `football/blocks/`

A block has:

- `FEATURES`: the columns it adds, all with one prefix (e.g. `rest_`).
- `build(m) -> DataFrame`: `match_id` plus those columns, computed from the cached matrix `m`.

The helpers in `football/blocks/_util.py` give the per-side table (`sides`) and windows of any per-match measure,
lagged to the collection day (`lagged`, `home_minus_away`). The rule: only what was known at the early price may enter
(opening prices, the fixture list, results and closes of matches before the collection day).

`pytest tests/test_football_lab.py` checks every block, automatically and by name. It rewrites the results and closes
of the last collection window and requires that window's features not to move. A block that fails is leaking. The
test suite also checks that the check catches a deliberately leaky block.

Blocks so far:

| Block | What it adds |
|---|---|
| `rest_days` | rest days and congestion |
| `venue_form` | home form at home, away form away |
| `market_state` | margin, favourite, draw, implied total, Betfair's early gap |
| `ah_consistency` | the handicap line against the 1X2 |
| `book_gaps` | soft books against the sharp open, and dispersion |
| `avg_anchor` | the market average's book against Betfair's, the maximum and Bet365 (Avg from 2019/20, Betbrain's average before) |

## 3. Run an experiment

```bash
python -m football.lab run --cache .football_cache \
  --arm base=team_all --arm rest=team_all+rest_days --arm mkt=team_all+market_state+book_gaps \
  --model ridge --model stack --note "rest days and market state"
```

- **Arm:** `name=part+part`. A part is a feature family (`team_all`, `xg_core`, `players`, ...), a block name, or a
  single column.
- **Models:** `ridge`, `gbm` (LightGBM with the league), `stack` (their mean).
- **Target:** the move of the home side's chance from the early price to the close, same-book rows, walk-forward by
  season (`--from 2016 --to 2024`, trained from `--train-from 2013`). `--subset xg` restricts to matches with xG
  windows.
- **Output, per arm and model:**
  - R2 against the training mean, and a permuted floor;
  - the standard error across folds;
  - the move caught by the top tenth of forecasts, and its right-way share;
  - the trade on Betfair's own 2024/25 prices, costed a tick off fair and at a symmetric book (`--no-cost` skips it).
- **Held back:** the seasons from 2025/26 were read for earlier pre-registered rules. They are left out unless
  `--include-spent` is given, and a result that uses them is development evidence only.

## 4. The protocol: screen fast, confirm once

1. **Screen in the lab:** minutes, development seasons only. Compare arms by R2 against the floor, top-tenth move, and
   fold standard error. An idea that does not beat its base by more than about one fold standard error is noise.
2. **Write down the rule:** for an idea worth trading, record the arm, model and threshold in a pre-registration file
   before any new data is read.
3. **Confirm forward:** on the recorder's books with sizes (`research/open_close/football_paper_trial.py`). That is
   the only test that counts for money.
4. **Ledger:** every screen lands in the ledger. Reading it before a new idea avoids re-testing what failed. Its
   length is the count of forks, which a final result should be read against.

## 5. Measured timings (3 Nov 2026, this session's 4-core container)

| Step | Time |
|---|---|
| Cold cache build, all five stages | 22 min 10 s (Dixon-Coles 225 s, team 305 s, books 8 s, xG 134 s, players 652 s) |
| Warm cache check, nothing changed | 4 s |
| One arm, ridge | 2-5 s |
| One arm, LightGBM or stack | 14-25 s |
| Load the matrix and build all blocks (once a run) | about 90 s |
| 7 arms x 1 model | 1 min 46 s |
| 7 arms x 3 models | 6 min 24 s (21 walk-forward fits over 9 seasons, with Betfair costing) |

A change to one feature module rebuilds only its stage; a new block needs no rebuild at all.

## 6. The first screens (3 Nov 2026, development; `reports/football_runs/lab_ledger.jsonl`)

Early-to-close move of the home chance, walk-forward 2016/17-2024/25, 60,805 matches:

| Arm | Ridge R2 | Stack R2 | Stack top tenth | Stack right way |
|---|---|---|---|---|
| Team features | 4.22% | 4.70% | 4.50% | 65.1% |
| + rest days | 4.22% | 4.72% | 4.63% | 65.9% |
| + venue form | 4.20% | 4.67% | 4.42% | 65.1% |
| + market state | 4.34% | 5.06% | 4.80% | 65.7% |
| + handicap vs 1X2 | 4.28% | 5.07% | 4.70% | 66.4% |
| + book gaps | 4.49% | 5.13% | 4.74% | 66.0% |
| **+ all five blocks** | **4.84%** | **5.57%** | **4.99%** | **66.9%** |

Fold standard errors are about 0.3-0.4 points, so all five together (+0.87 on the stack) is about two to three
standard errors; rest days and venue form add nothing. The stack beats ridge, and ridge beats LightGBM, in every arm.

On the xG subset (6,377 matches, 2024/25-2026/27, including the spent seasons: development only), the stack goes:

| Stack features | R2 | Top tenth |
|---|---|---|
| Team features | 3.72% | 3.85% |
| + the five blocks | 5.36% | 4.54% |
| + xG | 5.61% | 4.98% |

**Line-up features are refused for an early-price entry.** Adding the player features (the line-up surprise and the
eleven's rating) takes the R2 to 9.12% and the top tenth to 6.19%. But the eleven is public only at the team sheet,
about an hour before kick-off and days after the early price, so a trade at the early price cannot use them. What it
shows is that team news is a large part of the move, which is the case for the line-up-window strategy (entry after
the sheets, at the price then, measured on the recorder's books). The lab now refuses these features unless
`--entry lineup` is given.

**On the Betfair costing:** the stacks show +1.5% to +1.9% on 2024/25 at a symmetric book (base stack +1.53%, 90%
+0.83 to +2.36, 1,007 bets). That season was the one 29 screened configurations were read on. The same costing gave the
earlier linear rule +0.88% on 2024/25 and -0.17% on the holdout. Read it as a reason to pre-register and paper-trade,
not as an edge.

## 7. From a screen to a registered rule (3 Nov 2026)

The first rule through the protocol is `reports/preregistration_football_stack.md`. Code:
`research/open_close/football_stack_trial.py`.

- **Arm:** the stack arm with the best R2 on the 2024/25 fold. That is the one fold whose same-book rows are Betfair's,
  the market traded. Team features alone: 4.18%. With the five blocks: 3.56%. Avg-anchored: 2.87%.
- **Threshold:** chosen by `sweep`: 0.04.
- **Models:** fitted once and frozen by `freeze`, with SHA-256 in the file. `score` refuses models whose hashes differ.

The blocks win on the pooled Pinnacle seasons but not on Betfair's. A block that reads Pinnacle's open describes a book
this rule does not trade, and Pinnacle's early prices are stale from July 2025.

The resting-order variant is a second registration on the same signals (`reports/preregistration_football_stack_resting.md`,
`research/open_close/football_stack_resting.py`). It posts one tick inside the spread and is filled only when a later
recorded book proves the order would have matched. No history holds the books between the early price and the
close, so it has no development figure at all.
