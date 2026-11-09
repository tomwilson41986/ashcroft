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
SOURCE = ("the owner, 6 Nov 2026: \"it's 2%\" (the account's commission on greyhounds); confirmed by Betfair, "
          "greyhound-commission.yml run 42353884684 (7 Nov 2026): GB greyhound WIN and PLACE markets' base rate 2.0% "
          "(no discount allowed), 2.008% of net winnings actually charged on 148 settled greyhound markets (7 Oct - "
          "4 Nov 2026)")
#: Australian greyhound markets (the same run): base rate 8.0%, discount allowed; the account's discount rate is 6%, so
#: about 8% x (1 - 0.06) = 7.5% on AU winnings. Not the standard: a study or trader of AU markets must use this instead.
AU_BASE_RATE = 0.08
ACCOUNT_DISCOUNT = 0.06


def commission(env=None) -> float:
    """The rate on a market's net winnings: GREYHOUND_COMMISSION if set to a rate in [0, 1), else ``COMMISSION``."""
    env = os.environ if env is None else env
    try:
        v = float(str(env.get("GREYHOUND_COMMISSION", "")).strip())
        return v if 0 <= v < 1 else COMMISSION
    except ValueError:
        return COMMISSION
