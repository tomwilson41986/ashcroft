"""The public's R-squared by segment: where does the Betfair SP predict winners least well? (read-only; ledger
`public-r2-1112`; from the owner's reading of 12 Nov 2026: the logarithmic public R-squared, as Benter used it)

For a set of races: R2 = 1 - sum ln(p_winner) / sum ln(1 / runners), over the winners, p the winner's share of the
Betfair SP book. 0: the price says no more than the field size; higher: the public predicts well there. A segment
whose R2 sits below the baseline year after year is one where the market is easier to beat, and where a model's own
view deserves more weight.

race_results, 1 Jan 2022 - 31 Mar 2026 (the locked holdout from 1 Apr 2026 is not read), races with a winner and at
least 90% of runners priced. Segments: code, handicap or not, runners, class, month, track. For each: the R2, its
gap to the baseline of the same period, and the gap in each of two halves (2022-23, 2024-26).
Registered reading: the gaps persist if the rank correlation of the tracks' gaps between the two halves (tracks with
150 races in each) is above 0.30 and the handicap/non-handicap and code gaps keep their sign. A pass makes the
segment's R2 over earlier days a feature for the loop and a weight for the closing model; nothing else follows.
"""
import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query(
    "SELECT race_date, track, race_time, race_type, race_name, race_class, surface_type, number_of_runners, bfsp, "
    "placing_numerical FROM race_results WHERE race_date >= '2022-01-01' AND race_date <= '2026-03-31'", con)
for c in ("bfsp", "placing_numerical", "number_of_runners"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["day"] = d.race_date.str[:10]
d["race"] = d.day + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
ok = d.bfsp > 1
d["inv"] = np.where(ok, 1 / d.bfsp.where(ok, 1), 0.0)
g = d.groupby("race")
d["p"] = d.inv / g.inv.transform("sum")
d["rows"], d["priced"] = g.bfsp.transform("size"), g.inv.transform(lambda s: (s > 0).sum())
W = d[(d.placing_numerical == 1) & (d.priced >= 0.9 * d.rows) & (d.rows >= 2) & (d.p > 0)].drop_duplicates("race").copy()
W["lp"], W["lf"] = np.log(W.p), np.log(1 / W.rows)
t = (W.race_type.fillna("") + " " + W.race_name.fillna("")).str.lower()
W["hcap"] = np.where(t.str.contains(r"handicap|nursery|h'cap|hcap"), "handicap", "non-handicap")
rt = W.race_type.fillna("").str.lower()
aw = W.surface_type.fillna("").str.lower().str.contains("aw|all|poly|tapeta|fibre|standard")
W["code"] = np.where(rt.str.contains("chase"), "chase", np.where(rt.str.contains("hurdle"), "hurdle",
                     np.where(rt.str.contains("nh|bumper"), "bumper", np.where(aw, "flat aw", "flat turf"))))
W["runners"] = pd.cut(W.rows, [1, 5, 8, 11, 15, 99], labels=["2-5", "6-8", "9-11", "12-15", "16+"])
W["class"] = W.race_class.astype(str).str.extract(r"(\d)")[0].fillna("none")
W["month"] = W.day.str[5:7]
W["half"] = np.where(W.day < "2024-01-01", "2022-23", "2024-26")
r2 = lambda x: 1 - x.lp.sum() / x.lf.sum()  # noqa: E731
base = {h: r2(x) for h, x in W.groupby("half")}
base["all"] = r2(W)
print(f"races {len(W):,} ({W.day.min()} .. {W.day.max()}); baseline R2 {base['all']:.4f}; by half {base['2022-23']:.4f}, {base['2024-26']:.4f}; "
      f"by year {W.groupby(W.day.str[:4]).apply(r2).round(4).to_dict()}")


def seg(col, min_races=300):
    rows = []
    for k, x in W.groupby(col, observed=True):
        if len(x) < min_races:
            continue
        a, b = x[x.half == "2022-23"], x[x.half == "2024-26"]
        rows.append({col: k, "races": len(x), "R2": r2(x), "gap": r2(x) - base["all"],
                     "gap 2022-23": r2(a) - base["2022-23"] if len(a) >= 100 else np.nan,
                     "gap 2024-26": r2(b) - base["2024-26"] if len(b) >= 100 else np.nan, "n1": len(a), "n2": len(b)})
    return pd.DataFrame(rows)


keep = {}
for col in ("code", "hcap", "runners", "class", "month"):
    s = seg(col)
    keep[col] = s
    print(f"\nby {col}")
    print(s.drop(columns=["n1", "n2"]).round(4).to_string(index=False))
tr = seg("track", 300)
tr = tr[(tr.n1 >= 150) & (tr.n2 >= 150)].sort_values("gap")
print(f"\nby track ({len(tr)} with 150 races in each half): the ten the public predicts worst, and best")
print(tr.head(10).drop(columns=["n1", "n2"]).round(4).to_string(index=False))
print(tr.tail(10).drop(columns=["n1", "n2"]).round(4).to_string(index=False))
rho = tr["gap 2022-23"].rank().corr(tr["gap 2024-26"].rank())
# what a gap of this size is by chance: tracks shuffled within half
rng = np.random.default_rng(1112)
null = []
for _ in range(300):
    z = W.assign(track=rng.permutation(W.track.to_numpy()))
    a = z[z.half == "2022-23"].groupby("track").apply(r2)
    b = z[z.half == "2024-26"].groupby("track").apply(r2)
    j = a.index.intersection(b.index).intersection(pd.Index(tr.track))
    null.append(a[j].rank().corr(b[j].rank()))
signs = all((keep[c]["gap 2022-23"] * keep[c]["gap 2024-26"] > 0).all() for c in ("code", "hcap"))
print(f"\ntracks: rank correlation of the gap between halves {rho:+.3f}; shuffled tracks give {np.mean(null):+.3f} "
      f"(95th percentile {np.percentile(null, 95):+.3f}); code and handicap gaps keep their sign: {signs}")
print("verdict:", "PASS" if (rho > 0.30 and signs) else "does not pass")
