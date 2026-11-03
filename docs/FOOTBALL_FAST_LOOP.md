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

- `FEATURES`: the columns it adds, all with one prefix (e.g. `sch_`).
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
| `schedule` | rest days and congestion |
| `venue_form` | home form at home, away form away |
| `market_state` | margin, favourite, draw, implied total, Betfair's early gap |
| `ah_consistency` | the handicap line against the 1X2 |
| `book_gaps` | soft books against the sharp open, and dispersion |

## 3. Run an experiment

```bash
python -m football.lab run --cache .football_cache \
  --arm base=team_all --arm sched=team_all+schedule --arm mkt=team_all+market_state+book_gaps \
  --model ridge --model stack --note "schedule and market state"
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

## 5. Measured timings

_Filled in from the first runs (2 Nov 2026): see below._
