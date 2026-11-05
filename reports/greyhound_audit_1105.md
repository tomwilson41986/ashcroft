# Greyhound system audit, 5 Nov 2026

The owner, 4 Nov 2026: "Perform a full review of our approach, stack, model, features, trading agent. What can we
improve? Do a full audit and very open mind." Read-only: three independent code audits (model and features; trading
agents and controls; data, stack and operations) and the edge search of 4 Nov (`reports/greyhound_edge_search.md`).
Every finding cites the code or the run behind it. Nothing was changed.

## Verdict

1. **The core hypothesis is falsified.** "A form model prices dogs better than the market" does not hold. The
   pre-race model adds about nothing to the BSP: its weight in the Benter blend is about 0 (`reports/greyhound_model.md`
   sections 4, 11, 18-20). No limit-on-close rule against our price passed in 2025-26 (0 of 160; research-query
   41891640383). Across 9 strategy families, over 20,000 rules were tried, at fills we could really get, and none
   passed (`reports/greyhound_edge_search.md`).
2. **The closing-price forecast is real, but its information is already in the tradable book.** At 1 minute out it
   beats the mid: RMSE 0.144 against 0.158 (closing-price run 41884995327). With the place market added it scores
   0.172 against 0.176 (historic data, run 41892329688). But at the books' real prices every rule built on it lost
   (run 41897170252).
3. **The paper results have twice been artefacts of the fill model.** First, the early lay: paper won while live lost.
   Second, trade to the SP: +9.5% to +16.5% on a tick-through fill became -1% to -4% at the real book price. Any rule
   scored with the optimistic replay is suspect.
4. **Real money is exposed with weaker controls than the strategy's risk needs.** See A1 to A5.

## A. Safety: the live early lay (decisions for the owner)

| # | Finding | Evidence | Recommendation |
|---|---|---|---|
| A1 | The early lay places real orders by default. Its forward-test gate has not passed. There is no loss limit. | `paper-models.yml:126` falls back to `'live'` (asserted by `tests/test_workflow_guards.py:247`). `early_lay.py:62` sets `daily_loss_limit: None`. The `laytest.py:50-52` gate is 20 days and 1,000 trades. `greyhound_model.md:1235`: "none is measured". 4 Nov live: -GBP21.48 on GBP194.61 matched (running table, greyhound-live run 41881343717). | Switch to paper (repo variable `GREYHOUND_EARLY_LAY=paper`) until the gate passes on the queue-aware replay. Make paper the default and live an explicit opt-in. Set a hard daily loss limit. |
| A2 | The trader exits at `--until` and leaves matched lays open. | `early_lay.py:586-600`: `run()` ends with no cancel and no green-up. It watches races 60 minutes ahead (`:65`), so races after 21:30 are laid and then abandoned. A cancelled job leaves them open too (4 Nov runs were cancelled after 157 and 196 minutes). | Stop new lays `watch_minutes` before `--until`. Close every position before exit. Add a SIGTERM/atexit handler that cancels all `el-` orders. |
| A3 | The kill switch is slow and fails open. | `live.py:605-616` caches the STOP check for 60 s. `:623-627`: an S3 error reads as "not stopped". Resting lays are cancelled only for markets still watched (`early_lay.py:396-397`). The test checks only that no new lays go in. No drill logs found. | Check every poll, treat errors as stopped, cancel by order reference. Add a test that resting lays are cancelled. Drill and log it. |
| A4 | Restart recovery depends on the runner's local ledger. | `early_lay.py:300-313` resumes from the local file only. `green()` skips markets not in `priced` (`:481-483`). | On start, rebuild state from Betfair's current orders (`el-` references), with S3 as a fallback. |
| A5 | There is no pre-trade risk layer. The two traders can cancel each other's orders. | `early_lay.py:180,200` and `quote_mid.py:200,227` call the exchange's private methods. The early lay cancels every order on a market (`:195-200`), so it would pull quote_mid's live quotes. There is no per-market exposure cap, funds check or order-rate limit. Commission is not deducted from locked P&L. | One gate in front of every order: stake, per-market and daily exposure, price collar, order rate, funds. Cancel by reference only. Deduct commission. |
| A6 | Nothing alerts or reconciles. | A crashed trader does not fail the job (`paper-models.yml:161-168`). `scripts/session_watchdog.py` does not cover greyhounds. Settled bets are never read in `greyhound/`. Live lays have no CLV. | Watchdog with alerts when the trader's or recorder's status is more than 5 minutes stale. Reconcile daily against Betfair's cleared orders. |

## B. Measurement: what the numbers can be trusted for

| # | Finding | Evidence | Recommendation |
|---|---|---|---|
| B1 | No untouched hold-out. One test window from 1 Jun 2026 was reused by about 52 experiments and chose the live recipe. There is no multiple-testing correction. | `fastloop.py:56`; `research/greyhound/experiments/`; `live.py:61-76`. | Seal a future hold-out, read it once. Report deflated or best-of-K bounds. |
| B2 | Headline model numbers use features a morning card cannot know: the day's weigh-in, going, prize, handicap. | `metrics.py:232-233`, `hrb.py:107,148-149`. Batch 7's `best_rescaled` was `features: all` (RMSE 0.4086), cited at `live.py:62`; card-safe was 0.4127. | Report card-safe only. Label weigh-in runs as ceilings. |
| B3 | Paper is scored with an optimistic fill model. | `early_lay.py:653-697,842` ("the best case"). The queue-aware replay exists only in research (`research/greyhound/queue_fill.py`); its own fill may still be optimistic: traded volume counts both sides, and a penny traded through counts as a full fill (`greyhound/queue_fill.py:91-97`). | Score every paper book with the queue-aware replay. Validate it against live fills before trusting it. |
| B4 | The Benter test is a per-runner logistic regression, not a conditional logit. | `model.py:158-173`. | Use the race-level conditional logit (`fit_clogit` exists). |
| B5 | No trial registry. | Specs are committed for some batches only. Results sit in CI artifacts and prose, with no commit SHA (`fastloop.py:686`). | Append-only registry: spec, SHA, cache hash, data's last day, scores. Count trials. |
| B6 | Training and live differ. | The live card takes the trainer from the last run (`card.py:78-79`). The research cache keeps rows from 2020-06 only (`fastloop.py:104-105`), while live computes on everything (`live.py:181`). The simulator shares one RNG across races (`sim.py:91`). Hosted runners use Python 3.11; `pyproject` and the paper runner use 3.12. | Same frames, a per-race seed, the catalogue's trainer, one Python. Extend `card_check` to compare research and live. |
| B7 | Reproducibility. | The cache hash covers code only, not the data or every module (`fastloop.py:49-51,81-88`). Dependencies are `>=` with no lockfile. Fits are stopped by wall-clock time (`:170-182`). | Hash the data snapshot and all modules. Add a lockfile. Stop fits by round count. |

## C. Strategy: where an edge could still be (an open mind)

The market is efficient against everything we know before the off: form, our model, the morning price, track and
trap bias, the place-against-win relation. What is left is execution and information speed, and those need data we
do not hold.

1. **Record the Exchange Stream ourselves, starting now.** It is free on the live key and gives the full ladder,
   traded volume by price and every change, for every GB greyhound win and place market. Each day recorded is a day of
   honest backtest for queue-aware rules. That beats buying PRO history for the strategies we would trade. The
   current recorders poll the best 3 levels: every 60 s on the delayed key (about 90 s stale), and every 5 s on the
   live key from 15 minutes out only (`book_record.py:36`).
2. **Make execution the research subject.** The live lay lost because its fills were adversely selected: lays matched
   when dogs were shortening. A market-making or rebate-aware strategy is valuable only if its fill model is proven on
   recorded queue data. Build the backtester first (stream replay with queue position, cancellations and both-sided
   volume), validate it against our own live fills, then test strategies.
3. **Speed of information before the off.** Reserves and non-runners, trap changes, the going, kennel news, the
   weigh-in: no feed gives the weigh-in before the race (checked 4 Nov), and markets move on these in the final
   minutes. Our edge at 1 minute out comes from the place market and the book's total. A stream-fed forecast acted on
   within seconds, as passive orders, is the honest version of the trade-to-SP idea.
4. **Thin markets.** Australian place markets and forecast/each-way markets are less efficient than GB win markets.
   The AU steamer lay was the only family near passing, at the chance rate. First, confirm the account can bet AU
   markets.
5. **Costs.** Know the account's actual commission rate and the Betfair charges that apply. Many marginal rules flip
   sign between 2% and 5%. "No data found" for the account's rate in the code.
6. **Stop optimising the pre-race model for trading.** Keep it as one feature of the closing forecast. Its RMSE gains,
   such as the weight history, do not reach the P&L while the BSP absorbs them.

## D. Stack and operations

| # | Finding | Evidence | Recommendation |
|---|---|---|---|
| D1 | One self-hosted runner, one job, six background processes. The live early lay is among them. | `paper-models.yml:58,111-168`, `set +e`. Logs only at job end. | Run traders and recorders as supervised services on a persistent host (systemd or a container), not as a GitHub Actions job. Keep Actions for batch work. |
| D2 | Queue starvation: 35 of 85 scheduled greyhound-live runs were cancelled on 4 Nov. | One concurrency group (`greyhound-live.yml:41-43`). Each predict rebuilds every feature (`live.py:217-222`, 32-35 min). | A concurrency group per task. Cache the day's features incrementally. |
| D3 | Storage: growing CSV day files re-gzipped every 15 minutes. Local copies preferred over S3. Bad lines skipped silently. No schema version. | `betfair_recorder.py:211-215,326-331,771-779`. | Parquet partitioned by day, `schema_version`, a read path that counts skipped lines. |
| D4 | Data gaps. | No win PPWAP in the greyhound price files. Morning WAP for 14.7% of runners. GBGB only from 2018 (`gbgb.py:27`). Historic BASIC only Jan-Sep 2026. Place books only from 3 Nov. AU markets not recorded live. | Backfill GBGB to 2012. Record the stream (C1). Import further BASIC months if they are free. |
| D5 | Code health. | Dependencies unpinned. No push or pull-request test workflow. 502 files in `research/queries/done`. Duplicated helpers. `CLAUDE.md` does not describe `greyhound/`. | Lockfile and one Python. pytest on push. A shared book-reading module. A greyhound architecture section. |
| D6 | Secrets and permissions. | One AWS key pair for every job. The Betfair certificate stays on the runner (`paper-models.yml:84-97`). | Scope the credentials per job. Remove the certificate after each run. |

## Order of work

1. Owner decisions: early lay to paper (A1); a daily loss limit; whether to pursue AU (C4).
2. Safety fixes, before any further live use: A2 to A6.
3. Start recording the Exchange Stream on a persistent host (C1, D1).
4. Measurement fixes: B1 to B7, and the queue-aware replay as the only paper scorer (B3).
5. Research on the stream data: a fill model validated against our own fills, then execution strategies and the
   closing forecast at seconds out (C2, C3).
6. Operations: D2 to D6.

Sources: the three code audits of 5 Nov (file and line references above), `reports/greyhound_model.md`,
`reports/greyhound_edge_search.md`, and the GitHub Actions runs cited.
