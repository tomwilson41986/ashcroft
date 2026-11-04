"""Is the GB greyhound place SP mispriced against the win market? (research/greyhound/place_value.py; read-only)
Walk-forward from 2021 on Betfair's greyhound price files: the place SP's calibration, Harville and a LightGBM place
model against it, and every back/lay rule at the place SP with a 90% interval by day. Then the look-ahead upper bound
(the win SP in place of the win PPWAP)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from research.greyhound import place_value as pv     # noqa: E402

pv.main(["--years", "2016-2026", "--first-test", "2021", "--out", "out/greyhound_place_value.json"])
print("\n==== ORACLE (win SP, look-ahead upper bound) ====")
pv.main(["--years", "2019-2026", "--first-test", "2023", "--win-price", "bsp", "--out", "out/greyhound_place_value_oracle.json"])
