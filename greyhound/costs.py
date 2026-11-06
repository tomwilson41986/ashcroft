"""The commission Betfair charges this account on greyhound markets: one figure for every greyhound trader, model and
study (the owner, 6 Nov 2026: "check betfair to see what commission we pay on greyhounds. Then set this as
standard").

Betfair charges commission on a market's net winnings, never on a loss. ``COMMISSION`` is the owner's figure for the
account (6 Nov 2026: "it's 2%"); research/greyhound/commission_check.py (greyhound-commission.yml) reads what Betfair
charges, as a cross-check. ``SOURCE`` says where the figure came from. The environment variable GREYHOUND_COMMISSION
overrides it for a run (a rate between 0 and 1).
"""

from __future__ import annotations

import os

COMMISSION = 0.02
SOURCE = "the owner, 6 Nov 2026: \"it's 2%\" (the account's commission on greyhounds)"


def commission(env=None) -> float:
    """The rate on a market's net winnings: GREYHOUND_COMMISSION if set to a rate in [0, 1), else ``COMMISSION``."""
    env = os.environ if env is None else env
    try:
        v = float(str(env.get("GREYHOUND_COMMISSION", "")).strip())
        return v if 0 <= v < 1 else COMMISSION
    except ValueError:
        return COMMISSION
