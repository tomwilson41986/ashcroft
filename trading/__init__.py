"""Paper trading on the Betfair Exchange win markets from the model's prices.

pricing   the fair probability of each runner: the model's, the market's, or the two pooled
staking   how much to back: level, stake-to-win-a-set-amount, single-runner and race-level Kelly
strategy  which runners to back in a race, and at what price at worst
risk      the limits every simulated order passes
exchange  read-only Betfair market data, and a paper exchange that fills against the live book

Nothing here places an order: the Betfair client reads the catalogue and prices only, and every
fill is simulated. Live execution is not built; TRADING.md says what it would need.
"""
