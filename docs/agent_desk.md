# racing² agent desk

An autonomous, self-improving trading team across horse racing, greyhounds, football, tennis and cricket on Betfair.
Several validated strategies may run at once on one market. The team reviews its own results and improves.

## The owner's decisions, 10 Nov 2026

1. **Fully autonomous.** Agents promote and scale strategies inside one global bankroll and day loss the owner sets.
   The owner reviews after the fact.
2. **The 9 Nov freeze is lifted for the agents.** Every test is still written in `reports/research_ledger.jsonl`
   before its run and read against the deflated Sharpe (`research/overfit.py`, at least 0.95, every trial counted).
3. **Agents run as Claude Code scheduled tasks.** No API bill. The order path stays deterministic Python.

## Where we start

| Sport | State | Source |
|---|---|---|
| Horse racing | One validated edge: the morning back closed at the SP, +3.54% net CLV (90% +2.86% to +4.19%) on 12,389 out-of-sample trades | `reports/preregistration_clv_oos.md:32` |
| Horse racing | Leads only: lays as a second book +GBP827 over 21 days (0.44% of liability); a late-price model near the off +2.9% CLV on 4.8 bets a day | `TRADING.md:1787-1790,1814-1820` |
| Greyhounds | "The core hypothesis is falsified"; over 20,000 rules, none passed; paper twice flattered by the fill model | `reports/greyhound_audit_1105.md:10-21` |
| Football | "No strategy is profitable yet" | `reports/football_model_rethink.md:678` |
| Tennis | Match data only; no tennis price of any kind held | `reports/open_close/data.md:213` |
| Cricket | Census only: match odds a median GBP1.67m a day; no price history, no model | `reports/open_close/markets.md:30-31` |

## Design

```
 LLM agents (scheduled Claude Code sessions: git and a results branch only; no Betfair key, no AWS write)
   Researcher (one a sport)                 Analyst (nightly; weekly report)
      | PR: strategy code + registry entry + ledger line     | findings and new ideas -> desk/backlog/<sport>.jsonl
      v                                                      v
 ------ the repository is the team's memory: desk/strategies/  research_ledger.jsonl  desk/backlog/  playbook ------
      | CI (desk-ci.yml, deterministic): tests, workflow guards, registry, ledger append-only, gate hashes
      v                    deterministic Python on the UK runners
   Gatekeeper: promote / scale / demote by pre-registered arithmetic
   Strategy -> Intent -> desk/risk.py -> exchange          desk/warden.py: the account against the envelope -> STOP
                          ^ the envelope: DESK_* repository variables (the owner's alone)
```

Principles:

1. **An LLM never places, sizes or promotes.** Agents write hypotheses and code. Strategies emit intents. Only
   `desk/risk.py` turns an intent into an order. Only the gatekeeper changes a stage or a stake.
2. **The envelope is out of the agents' reach.** It lives in repository variables. Unset or malformed means paper.
3. **Real fills are the truth test.** Scaling reads Betfair's cleared orders by strategy reference, never a
   strategy's own ledger or a paper fill model.
4. **Forward days only.** A backtest earns paper and nothing more; a gate reads days after its registration. The gate
   is hash-pinned when registered; an amendment is a new strategy.
5. **The live horse trader is not refactored.** New strategies run in their own processes under their own reference
   prefix; the horse trader adopts only `ash...` orders (`trading/session.py:634-637`).

## Strategy lifecycle

| Stage | Money | Moves up when |
|---|---|---|
| research | none | ledger line and registry entry merged before the run; walk-forward at real book prices; interval above zero; deflated Sharpe at least 0.95 with every trial in the sport counted |
| paper | none | the registered number of forward days (at least 30, `research/overfit.py MIN_OBS`), read once, the same test |
| pilot | the pilot budget in the envelope | live fills and CLV inside the tolerance set from paper; no reconciliation break |
| scaled | one stake step at a time | the realised live interval above zero at the current step; the next step fits the envelope |

Down moves are automatic: a strategy's day loss halts its day; pilot CLV below paper by the tolerance demotes it and
marks that fill model invalid; repeated demotion retires it. Step sizes derive from the envelope; no figure is set here.

## Several strategies on one market

Each strategy has its own prefix and process. `desk/risk.py` sees every desk strategy's positions on the market, so
a second book near the off knows what the morning book already holds; it caps the market's worst case for the
strategy and for the desk together. Horse racing's first candidates are in `desk/backlog/horse_racing.jsonl`:
the late book, the lay book, a subset book, the stake floor at long prices; resting orders wait for the horse stream.

## The agents

| Role | Cadence | Does |
|---|---|---|
| Researcher (one a sport) | daily | top backlog item -> ledger line and registry entry (with its stated chance of passing) -> code -> queue the run -> read the result -> record it |
| Analyst | nightly; weekly report | per strategy: Betfair's record against the ledger, CLV, slippage, live against paper; new ideas to the backlog; past ideas scored against their stated chances; the playbook updated with what failed and why; the owner's racing² report |

Review is deterministic CI, not a second agent's approval: one bot identity cannot approve itself.

## Phases

| Phase | What | Who |
|---|---|---|
| 0a Lock-down | The checklist below | The owner |
| 0b Control layer | `desk/` registry, envelope, risk kernel, accounting, warden (alert only), ledger checks, CI | This change |
| Now, in parallel | Tennis and cricket prices recorded (forward gates need 30 periods); the BASIC Historic Data plans for both | The owner, then the data jobs |
| 1 | Horse multi-strategy: strategy interface, desk exchange (limit lays, cancel by reference), replay on recorded books; the late, lay and subset books in replay then paper | Build |
| 2 | Gatekeeper and the agent loop on horses: results queue, two scheduled tasks, the warden with `--enforce` | Build, then the owner switches |
| 3 | Greyhounds and football: the registered trials become registry entries; execution-led ideas only | Agents |
| 4 | Tennis and cricket models once a season of prices is held | Agents |

## Phase 1 so far (11 Nov 2026)

- `desk/strategy.py`: a strategy is a pure function from a `MarketView` (the market, its book, each runner, the
  closing model's view on demand) to `Intent`s; `on_fill` tells it what matched.
- `desk/session.py` `DeskSession`: several strategies on one market, every intent through the kernel in paper mode,
  each order under the strategy's own reference (`<prefix><mmdd>-...`), settled at the BSP from Betfair's record.
  It runs on the replay's shared book beside the live rule (`research/arms/core.py run_agents`). Paper only.
- `desk/algos/lay_book.py`: the lay arm ported as the desk's first strategy, registered at the research stage
  (`desk/strategies/horse_racing/horse-lay-book.json`, gate pinned) and in the ledger (`desk-lay-book-1111`).
  On the arms' test day it reproduces the arm's fills and score exactly (`tests/test_desk_session.py`).
- The first replay (`research/queries/done/desk_lay_book_1111.py`, run 42972505001) was **not read**: the port's
  fills differed from the arm's on all 29 days. Cause: the port sized every lay at the start of the minute against
  the same day-liability room, where the arm sizes each as it sends it; and its check compared the port under the
  registered limits with an arm that has none. Fixed with `Strategy.revise` (each order sized as it is sent) and a
  test of the case; the ledger holds the result line.
- The second replay (`research/queries/done/desk_lay_book_1112.py`, run 42988481613) was **not read** either: the port
  matched the arm exactly on 14 of 29 days and by a penny on most of the rest. Cause: the kernel's pence floor lost a
  penny on some exact stakes in floating point (1.13 became 1.12). Fixed and tested. Under the registered limits the
  0.2 price collar was the only limit that bound (38 refusals, all on the walk up the lay ladder).
- `research/queries/desk_lay_book_1113.py` (ledger `desk-lay-book-1113`, registered before the run): 1112's design
  and reading unchanged. If its parity fails again, the work stops for a report rather than a fourth fix.
- Not yet: a live desk exchange (phase 2, behind the pilot stage); the late and subset books (they need a model fitted
  on late prices and a new book construction first: `desk/backlog/horse_racing.jsonl`).

## 0a Lock-down: the owner's checklist before anything autonomous bets

Found in review, 10 Nov 2026, verified in the workflow files:

- [ ] **Branch workflows hold secrets.** A push to `claude/happy-ritchie-*` runs that branch's own workflow file on
      the UK runner with the Betfair login (`live-odds.yml:13,82-99`). Move the Betfair and AWS write secrets into a
      GitHub Environment limited to the default branch.
- [ ] **A research query holds the AWS key.** `research-query.yml:61-62` gives any pushed query the key that also
      writes and deletes STOP objects (`greyhound-stop.yml`). Give research a read-only key.
- [ ] **Branch protection.** Require `desk-ci.yml` and code-owner review (`.github/CODEOWNERS`) on the default branch.
- [ ] **STOP resume behind approval.** `greyhound-stop.yml resume` behind an Environment with a required reviewer.
- [ ] **The horse and morning kill switches fail open** (`auto_trade.py:141-145`, `morning/live.py:84-88,326-330`): a
      separate change to `auto_trade.py`, the owner's to approve.
- [ ] **The envelope:** `DESK_BANKROLL`, `DESK_DAILY_LOSS`, `DESK_SPORT_SHARE`, `DESK_RESERVE`, `DESK_PILOT_BUDGET`,
      `DESK_MAX_MARKET_EXPOSURE` (`desk/envelope.py`).
- [ ] **The warden's runner:** `DESK_RUNNER`, an existing UK label such as `data`; never the horse racing runner.
- [ ] **The repository is public:** every strategy, gate and ledger line the agents write is readable by anyone.

## What exists in 0b

```bash
python -m desk.registry --list      # the inventory: 8 traders, every one that can place an order
python -m desk.registry --check     # ids, prefixes (unique, never ash for a new one), gates, limits
python -m desk.trials --check --base origin/claude/scrape-horse-racing-results-vaHKc
python -m desk.warden --out out/warden.json    # read-only; exit 2 on a breach, 3 on a read error
```

The warden reads only `listCurrentOrders` and `listClearedOrders` (both already used by the live traders, here for
the first time without market ids) and `getAccountFunds`. Betfair's account statement (`getAccountStatement`) is not
read yet: its fields could not be checked against Betfair's documentation on 10 Nov 2026 (the reference pages refused
the fetch), so it waits for phase 2.

## The high-frequency track (the owner, 11 Nov 2026)

"What about if we were to execute hundreds or thousands of trades per market? Both sides of the line. Developing a
model to predict the next price move ... Kalman, Markov chain ... particularly <5 minutes to the off."

What the repo already shows, before any of it is built:
- Quoting both sides around the mid lost in both sports (horses -GBP17,554 over 22 days, research ledger line 590;
  greyhounds every margin, `reports/greyhound_model.md:1230-1236`): resting orders fill when the price runs through them.
- Short-term taker rules lost (greyhound stream lab 0 of 86, `reports/greyhound_edge_search.md:213`; horse small book
  near the off 0 of 54, ledger 663).
- Betfair's transaction charge for high order counts is unverified (`reports/open_close/markets_sources.md:16`): it must
  be read off the account's charges page before any money moves.

Stages, each gated on the last:
1. **Prediction** (`research/micro/`): can the next mid move be predicted from the order book in the last 300 s?
   Baseline, queue-imbalance logistic, Markov chain, Kalman fair price, LightGBM; walk-forward by day; sealed days.
   First test: `research/queries/micro_next_move_1111.py` (ledger `micro-next-move-1111`), greyhounds; horses once 10
   days of horse stream are held (from 12 Nov).
2. **Simulation**: a queue-aware, latency-aware simulator built on `research/greyhound/stream_replay.py`, counting every
   order and cancel; model-skewed market making, one-sided passive, and a taker control; after commission, the Expert
   Fee and the transaction charge.
3. **A penny fill probe** (the owner's approval and budget): the simulator's fills checked against real ones.
4. **Build**: a stream-fed trader with the order stream, batched orders, a dead-man cancel, on a supervised host.
