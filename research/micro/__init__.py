"""Market microstructure in the last minutes before the off (read-only research; the owner, 11 Nov 2026: "Developing a
model to predict the next price move ... Kalman, markov chain, or whatever high frequency traders use ... applied to
Betfair Horse Racing and Greyhound markets. Particularly <5 minutes to the off").

Stage 1 of docs/agent_desk.md's high-frequency track: can the next move be predicted at all, before any order?

- ``events``    the recorded Exchange Stream to one row per runner per book change in the last minutes
- ``features``  the high-frequency feature set and the next-move targets, with no look-ahead
- ``models``    the baseline, queue-imbalance logistic, Markov chain, Kalman fair price and gradient boosting

Probability conventions, used throughout: a price p is the probability 1/p. The money offered to back at the best
back price (``atb``) is the ask side (a backer buys probability at 1/back); the money offered to lay at the best lay
price (``atl``) is the bid side (a layer sells it at 1/lay). An "up" move is the probability rising: the price
shortening.
"""
