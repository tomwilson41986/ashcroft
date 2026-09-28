"""Automated trading on the Betfair Exchange win markets from the model's prices.

pricing   the fair probability of each runner: the model's, the market's, or the two pooled
staking   how much to back: level, stake-to-win-a-set-amount, single-runner and race-level Kelly
strategy  which runners to back in a race, and at what price at worst
risk      the limits every order passes before it is sent
exchange  the Betfair API (live orders) and a paper exchange that fills against the live book

Paper is the default everywhere: nothing reaches the exchange unless the trading mode is
"live" and every credential is present (see TRADING.md).
"""
