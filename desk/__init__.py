"""The desk: the control layer for an autonomous, multi-strategy, multi-sport trading team (docs/agent_desk.md).

The owner, 10 Nov 2026, planning the agent team: fully autonomous inside one bankroll and daily-loss cap the owner
sets; the agents may search every sport again (the 9 Nov freeze lifted for them), every test still written in the
research ledger before its run and read against the deflated Sharpe; the agents run as Claude Code scheduled tasks.

What this package holds, and what it does not:

- ``envelope``  the owner's limits, read from repository variables; absent or malformed means paper only.
- ``registry``  the algorithm inventory: one JSON file a strategy under ``desk/strategies/<sport>/``, every trader
                that can place an order included, each with its own order-reference prefix and kill switch.
- ``risk``      the one path from a strategy's intent to an order: deterministic limits, none an LLM can change.
- ``account``   pure accounting on Betfair's own record: orders to strategies by reference, worst case a market.
- ``warden``    the account-level watchdog over every trader (read-only on the delayed key); it writes the S3
                STOP objects the traders already honour only when run with ``--enforce``.
- ``trials``    the research ledger kept append-only, new lines checked, trials counted for the deflated Sharpe.

Nothing in the live horse trader (``auto_trade.py``, ``trading/``, ``model/``) imports this package
(tests/test_desk.py holds that line); it is governed from outside, by the warden.
"""

SPORTS = ("horse_racing", "greyhounds", "football", "tennis", "cricket")
