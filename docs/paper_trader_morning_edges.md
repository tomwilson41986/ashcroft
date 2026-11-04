# Paper trader: the morning edges (spec, 4 Nov 2026)

> 4 Nov 2026, later: the owner put the steamer and quick-return models live, WIN only, at half stakes, before this
> gate (`docs/morning_models_live.md`). The paper trader keeps running as the record of the place leg and the
> Timeform improvers.

racing². Paper only: no order reaches Betfair. Built from `reports/strategy_research_2026-11-04.md` (PR #231).
Follows the house rule (CLAUDE.md, 2 Nov 2026): nothing here touches the live UK/IE model, `live-trade.yml`,
`auto_trade.py` or `trading/`; it runs beside them and must pass a gate fixed before the first forward day.

## 1. What it trades

Three strategies, frozen as written here. Each has its own paper account, ledger and gate.

| Id | Selection (all known before the price is taken) | Market and price | When |
|---|---|---|---|
| `qr_am` | Quick return: GB/IE flat, AW, hurdle or chase runner whose last run was 1-4 days ago (`days_since_lr` on the 06:00 card) | Back WIN and back PLACE at the recorded morning book | 10:00 UK mark |
| `steam_am` | Steamer model, CLEAN features: predicted log(morning price / BSP) at or above the frozen threshold (the training set's top 5%) | Back WIN and back PLACE at the recorded morning book | 10:00 UK mark |
| `tf_pb_bsp` | Timeform improver: Flat/AW runner whose last run was a career-best Timeform performance rating by 5lb or more | Back WIN and PLACE at BSP | at the off (SP) |

**No order before 10:00 UK** (the owner, 4 Nov 2026: "Ensure it doesn't bet before 10:00 so it doesnt take prices
from the live trader"). The session does nothing at any earlier minute (`ENTRY_UK` in `research/paper/morning_edges.py`,
held by `tests/test_morning_edges.py`); a selection with no book at 10:00 is retried each minute to 10:15, then
dropped. And **a horse the live trader has backed that day is left alone**, at any time and in both markets (its
ledger, `trading/live/<day>/ledger.csv`), so the paper trader never competes with it for a price or a later top-up.
Counted as `skipped_live`.

Exclusions (all three): races of 4 runners or fewer (no place market); a runner with no back price in the book at the
mark; a non-runner (void, as on the exchange: its stake is returned and the other bets take Betfair's reduction
factor).

## 2. Staking

To win a set amount, as asked: **stake = GBP100 / (price - 1)** in each market, so a winner pays GBP100 before
commission and a placed horse pays GBP100 in the place market.

| Limit | Value |
|---|---|
| Per selection | GBP100 to win in WIN, GBP100 to win in PLACE |
| Largest stake on one bet | GBP200 (caps the to-win stake on short prices) |
| Day limit (staked, per strategy) | GBP4,000 |
| Paper bank (per strategy) | GBP5,000, as the other paper agents |

Commission: the ledger records P&L at 2% (the desk's rate) and at 5% (the research's rate). The gate reads 5%.

## 3. Filling an order on paper (the point of the test)

The research priced every bet at Betfair's **morning volume-weighted price** from the price files. That is an
average, not a fill. The paper trader fills against the **recorded book** instead:

1. At the mark (10:00 UK), read the runner's WIN and PLACE books from the recorder (`betfair_live/<day>/`, the
   snapshot nearest the mark, within 5 minutes, `betfair_recorder.py CLOCK_MARKS`).
2. Take the best back price and walk down the ladder, filling at each level up to the size shown, until the stake is
   filled or the next price is under 95% of the best. An unfilled remainder is left unfilled and recorded.
3. Use the shared book (`research/arms/core.py run_agents`), so these agents and the existing desk agents cannot all
   take the same money. They step after `live_rule` (the live trader's orders came first).

Every bet also records the price-file morning WAP for the same runner, so the gap between "what research assumed"
and "what the book gave" is measured on every bet.

## 4. The steamer model

| Item | Value |
|---|---|
| Script | `research/queries/done/strat_i_steamers_1104.py`, CLEAN feature set |
| Training data | race_results + betfair_prices, runners with a morning price (GBP50+ matched), 2018-01-01 to 2026-10-31 |
| Target | log(morning WAP / BSP) |
| Learner | LightGBM regressor: 600 trees, learning rate 0.03, 63 leaves, min child 400, subsample 0.8, colsample 0.8, seed 0 |
| Features | code, handicap, class, field size, age, OR and its rank, jockey's allowance, days since last run, last run's finish, market rank, BSP and steam, runs held, trainer change, jockey change, headgear on/off, month, hour, trainer and jockey earlier strike rate, A/E vs BSP and mean steam (earlier days only) |
| Threshold | the 95th percentile of the fitted model's predictions on its own training rows |
| Freeze | save the booster, the feature list, the threshold and the training cut-off to `s3://$PRED_BUCKET/paper/morning_edges/model/` before day 1; never refit during the test |

None of the features uses the morning price or the morning volume (those were in the FULL model and were dropped:
they are measured over the same window as the price backed). Everything is known from the 06:00 card and earlier
days.

## 5. Daily run

| When | Step | Source |
|---|---|---|
| 06:40 UTC daily | Refresh the Timeform cache (current and previous month) | Railway `APIData_Table2` -> `s3://$PRED_BUCKET/paper/morning_edges/timeform/` |
| same run, each pending day | Selections from the card view; replay 10:00-10:15 UK against the recorded WIN and PLACE books; settle; `tf_pb_bsp` at BSP | `horse_racing.db`, `betfair_live/<day>/`, Betfair price files, `trading/live/<day>/ledger.csv` |

What makes the test honest is that a selection reads nothing the 06:00 card did not have: the day's results and
prices are blanked before the features are built, and the check runs on every job.

## 6. Ledger

`s3://$PRED_BUCKET/paper/morning_edges/`:

* `model/booster.txt`, `model/meta.json`: the frozen steamer model and its threshold, cut-off and features;
* `selections/<day>.csv`: one row per (strategy, runner), the rule's inputs and (steamer) the prediction;
* `bets/<day>.csv`: per bet or skip: strategy, market, time, best back, size asked, filled price and size, the
  price-file morning WAP, BSP, result, P&L before commission; `skipped_live` and `no_book` rows;
* `days/<day>.json`: per strategy and market: selections, bets, skipped_live, no_book, staked, asked, wins, expected
  (sum of 1/filled price), P&L at 2% and 5%, median gap to the morning WAP.

Overlap: every horse the live trader backed is counted as `skipped_live`, so the
promotion decision can see how much is already being traded.

## 7. The gate (fixed now, read once per strategy, promotion only on the owner's word)

Read when a strategy has **1,000 settled win bets** and at least **60 racing days**, whichever comes later. Expected
pace from 2022-26: `steam_am` about 10 a day (about 100 days), `qr_am` about 3 a day (about 330 days), `tf_pb_bsp`
about 20 a day on the Flat (about 60 days).

A strategy passes only if all hold, at 5% commission:

1. **Win A/E against the filled price > 1**, chi-square p < 0.05;
2. **Place A/E against the filled place price > 1**;
3. **Win + place P&L positive**, with a day-block bootstrap 90% interval wholly above zero;
4. **Fills**: at least 80% of the to-win stake filled at the mark, and the median filled price no worse than 3%
   under the price-file morning WAP (if the WAP is unreachable, the research figure does not apply);
5. **No month** with win A/E under 0.85 on 100+ bets.

Stop early (pause the strategy, tell the owner) if, after 300+ bets, win A/E against filled prices is under 0.90, or
the paper drawdown passes GBP3,000.

## 8. What was built (4 Nov 2026)

| Piece | Where | Notes |
|---|---|---|
| Features and rules | `research/paper/morning_features.py` | the rules, the CLEAN features, `card_view` (the day's results and prices blanked), the trainer |
| Trader | `research/paper/morning_edges.py` | selections, the 10:00 session on the shared replay book (`research/arms/core.run_agents`), settlement, ledger, gate; `--train` freezes the model |
| Workflow | `.github/workflows/paper-morning-edges.yml` | daily 06:40 UTC on GitHub's runner; no Betfair key; dispatch inputs `days`, `backfill_from`, `train` |
| Setup run | `research/queries/paper_morning_edges_setup_1104.py` | trains and freezes the model, fills the Timeform cache, replays 30 Sep onward (the fill check) |
| Stop | `s3://$PRED_BUCKET/paper/morning_edges/STOP` | stops new selections |
| Dashboard | the traders dashboard, one row per strategy | day P&L, running A/E vs fills, fill ratio, WAP gap |
| Tests | `tests/test_morning_edges.py` | no order before 10:00 UK (and 10:00 UK is 09:00 UTC in summer); the live trader's horse skipped; to-win staking, the ladder walk and the GBP200 cap; settlement with reduction factors; card view; no order code; the workflow carries no Betfair key |

Selections are computed when the day is replayed (after its price file is archived), from the card's facts and
earlier days only: every result and price column of the day is blanked first (`card_view`), and each job checks that
the predictions it bets on equal those from the blanked view.

## 9. Before day 1

- [ ] Replay `qr_am` and `steam_am` on the books already recorded (30 Sep 2026 onward): fill ratio and WAP gap. If
      the median fill is more than 3% under the WAP, stop and report before running forward.
- [x] Measure overlap with the live trader on the same days (30 Sep to 2 Nov, 33 days;
      `research/strat/results_2026-11-04/paper_overlap_study_1104.txt`). Steamer WIN, P&L at 5%: the live trader's
      horses left out 243 bets, A/E 1.231 vs fills, +GBP815 (ROI +25.6%); the live trader's horses re-backed 148 bets,
      A/E 1.091 (chi2 p 0.64), +GBP181 (+7.0%); re-backed only with predicted steam still to come 116 bets, A/E 1.101,
      +GBP196 (+9.2%). PLACE loses on every variant. On those horses the live trader had already backed at an average
      10.44 against our 8.55 at 10:00, and came back after 10:00 109 times (GBP1,093 matched). Decision: the live
      trader's horses stay out of the gated strategies (the default, `skipped_live`).
- [ ] Freeze the model and write this document's version (commit SHA) into the ledger.
- [ ] Confirm the price files' morning WAP window (the research used it as "the morning price"; its exact cut-off
      time is not recorded in this repository).

## 10. Research figures this is testing (2022 to 3 Nov 2026, out of sample)

| Strategy | Runners | Win A/E vs morning WAP | Place A/E | Win+place, level | Win+place, to win GBP100 |
|---|---|---|---|---|---|
| `qr_am` | 5,353 | 1.148 (p 6e-07) | 1.060 | +8.2% | +3.4% |
| `steam_am` (top 5%) | 16,562 | 1.155 | 1.076 | +12.8% | +5.8% |
| `tf_pb_bsp` (vs BSP) | 31,083 | 1.042 (p 0.001) | 1.020 | +0.5% | - |

Sources: `research/strat/results_2026-11-04/` (`strat_e_quick_return_1104.txt`, `strat_i_steamers_1104.txt`,
`strat_f_timeform_morning_1104.txt`).
