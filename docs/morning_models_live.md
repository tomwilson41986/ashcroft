# The morning models, live (4 Nov 2026)

racing². Real orders on the owner's Betfair account, only with the repository variable `MORNING_LIVE=yes`.

## The owner's decisions (4 Nov 2026)

| Asked | Decided |
|---|---|
| "Create two live models; Steamers, Quick Returners. Set these up and run to both win and place." | Two models, each with its own ledger, order references and day limit |
| Place leg: place SP, win only, or the 10:00 place book | **Win only.** At the place SP neither model has a place edge in the 2022-26 test (below) |
| Real orders now, or paper until the gate | **"Live now, half stakes"**: GBP50 to win, at most GBP100 a bet, GBP2,000 a day per model |
| The live trader's horses | **Left out**: no re-back (the overlap study, `research/strat/results_2026-11-04/paper_overlap_study_1104.txt`) |
| Timing | **No order before 10:00 UK**, so the live trader's morning prices are its own |

This runs live before the paper gate in `docs/paper_trader_morning_edges.md` has been read. That is the owner's call,
against the house rule (CLAUDE.md) that a new model stays on paper until it passes a pre-registered gate. The rule
was put to the owner before building, along with the fill check below.

## What the evidence says, and what it does not

| | Steamer (CLEAN, top 5%) | Quick return (back in 1-4 days) | Source |
|---|---|---|---|
| Research 2022-26, win A/E at the morning price | 1.155 (16,562 runners) | 1.148 (5,353), p 6e-07 | `strat_i_steamers_1104.txt`, `strat_e_quick_return_1104.txt` |
| Research 2022-26, win A/E at BSP | 1.032 (p 0.07) | 1.055 | `strat_steamers_rules.csv`, `strat_quick_return_rules.csv` |
| Research 2022-26, place A/E at the place SP | 1.007 (p 0.49), ROI -2.4% | 1.013 (p 0.40), ROI -0.6% | same |
| Replay on the recorded books, 30 Sep to 2 Nov, live trader's horses left out: win bets, A/E vs fills, P&L at 5% | 243, 1.231, +GBP815 | 77, 1.351, +GBP441 | `paper_overlap_study_1104.txt` |
| Same replay: share of the to-win stake filled, median price vs morning WAP | 76%, -4.5% | 70%, -5.2% | same |

The replay misses the fill check set before the paper test: 80% of the stake filled and no worse than 3% under the
morning WAP. On 33 days its win A/E figures are not significant (chi-square p 0.16 and 0.21).

## How it trades

| Item | Value | Where |
|---|---|---|
| Selections | From the 06:00 card (`s3://$BUCKET/racecards/<day>_<hhmm>.csv`) and earlier days in the HRB database, built as the research built them | `morning/select.py`, `research/paper/morning_features.py` |
| Steamer model | The frozen CLEAN LightGBM (fitted 2018 to 31 Oct 2026), threshold 0.0559 (training top 5%) | `s3://$BUCKET/paper/morning_edges/model/` |
| Quick return | `days_since_lr` 1-4, flat / AW / hurdle / chase, 5+ runners | `research/paper/morning_features.py rule_qr` |
| Selection job | 06:45 UTC, GitHub's runner, no Betfair key; writes `s3://$BUCKET/morning/selections/<day>.csv` | `.github/workflows/morning-models.yml` (select) |
| Trade job | 08:05 UTC on the morning runner; waits for 10:00 UK | same (trade) |
| Order | One limit back a selection, FILL_OR_KILL, at the Betfair tick at or above 95% of the best back, so it takes the best price and walks the ladder down to 95% of it; anything not matched at once is cancelled | `morning/live.py` |
| Stake | GBP50 / (price - 1), at most GBP100; under GBP1 not placed (`below_min`) | |
| Day limit | GBP2,000 matched a day per model (`day_limit`) | |
| No book at 10:00 | Tried each minute to 10:15 UK, then dropped (`no_book`) | |
| Race off within 2 minutes | Not bet (`too_late`) | |
| The live trader's horses | Its ledger (`trading/live/<day>/ledger.csv`) read before every round: any horse it has matched a back on is skipped (`skipped_live`) | |
| A horse both models select | Two bets, the steamer's first | |
| References | `mes<mmdd><selection>` (steamer), `meq<mmdd><selection>` (quick return), never the live trader's `ash<mmdd>`, so it never adopts or lays these | `trading/session.py sync` |
| Settle | 22:50 UTC: Betfair's settled bets under our references (live), or the closed books (paper). P&L at 2% and 5% commission | `morning/live.py settle` |
| Ledgers | `s3://$BUCKET/morning/<live\|paper>/<day>/ledger.csv` (written after every order) and `summary.json` | |
| Stop | S3 object `morning/STOP` or `trading/STOP`: no new orders within a minute. Cancel the run to stop at once | |

What it does not touch: `live-trade.yml`, `auto_trade.py`, `trading/` (imported, unchanged), `betfair_recorder.py`,
`model/`. It runs on its own runner, never the live horse trader's, the paper models' or football's
(`tests/test_workflow_guards.py`).

## Before the first live morning (the owner)

1. Merge the PR: schedules run from the default branch only.
2. Register the runner on the UK server (one command, as the live runner's user):
   `bash scripts/setup_runner.sh morning`. The data and football runners' jobs have been queued since 3 and 4 Nov,
   so they are not registered either and cannot stand in.
3. Set the repository variable `MORNING_LIVE=yes`. Without it the same job runs on paper against the live books,
   which is worth one morning first to see the fills.
4. Check the account can carry it: up to GBP4,000 a day across the two models, beside the live trader's own limit.

## The selection check (the card against the research's rows)

`research/queries/morning_live_select_check_1104.py` rebuilds each day from 27 Oct to 2 Nov from that day's 06:00 card
and compares it with the selections the research's method makes from the results table. The results are in
`research/strat/results_2026-11-04/morning_live_select_check_1104.txt`. Some differences cannot be avoided: the card
cannot know a horse withdrawn after 06:00 (it has no book at 10:00 and is not bet), and a field size on the card
still includes the later non-runners.

As run on 4 Nov 2026 (27 Oct to 2 Nov, the horses that ran):

| Model | Research's selections | Found from the card | Card-only | Withdrawn after 06:00 (not bet) |
|---|---|---|---|---|
| Quick return | 27 | 27 (100%) | 0 | 9 |
| Steamer | 90 | 80 (89%) | 3 | 13 |

The steamer's 10 misses and 3 extras sit near its threshold. The card's field size and OR rank still count the
horses withdrawn later, where the research's rows (from the results) did not. Two cards a week name some races
only by the date ("Fri 30th Oct 2026"); such a race takes the code most of its runners last ran under, or is not
traded. Today's card (4 Nov) gave 15 selections: 4 quick return, 11 steamer.

