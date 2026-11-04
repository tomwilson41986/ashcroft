"""The place SP against the win market's close (read-only). 1) The price files: how fast the edge of backing at the
place SP where Harville on the win SP clears it decays as the win SP is known less exactly (a sensitivity test: the
win SP with a normal error of sigma in log price; sigma 0.144 is the closing model's RMSE at 1 minute out, batch 7).
2) The recorded books: the same rule with the win market's projected SP / mid at 120, 60, 30 and 10 s before the off,
settled at the place SP (research/greyhound/place_live.py)."""
import os
import sys

sys.path.insert(0, os.getcwd())
from sources.common import Store                       # noqa: E402
from research.greyhound import place_value as pv, place_live as pl   # noqa: E402

store = Store()
j = pv.load(store, range(2019, 2027))
print("sigma | rule (2% commission, test 2023-26) | bets | ROI% | 90% by day")
for sigma in (0.0, 0.03, 0.06, 0.10, 0.144, 0.20):
    r = pv.run(store, range(2019, 2027), 2023, "bsp", noise=sigma, j=j)
    for x in r["rules"]:
        if x["comm"] == 0.02 and x["side"] == "back" and x["edge>"] in (0.05, 0.10, 0.20) and x.get("roi_stake%") is not None:
            print(f"{sigma} | {x['p']} back edge>{x['edge>']} | {x['bets']} | {x['roi_stake%']} | {x['ci90_stake%']}")
print("\n==== recorded books ====")
pl.main(["--from", "2026-10-06", "--out", "out/greyhound_place_live.json"])
