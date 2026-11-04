"""Strategy study E (4 Nov 2026, read-only): the quick return, backed win and place, to win a set amount.

The one signal that held its sign in every code in both periods (studies A, 4 Nov): a horse back within 1-4 days of
its last run. Here it is tested as a strategy, as the owner asked: large samples, chi-square, positive WAX against the
BSP, positive in the place market, both markets backed, staked to win a set amount.

  1. CONTROL: every runner by code at BSP and at the morning prices (what an ordinary horse does morning to off)
  2. RULE FAMILY: DSLR 1-4 pooled and by code, refined by last run (placed / same code / handicap)
  3. CAPACITY: how much was matched in the morning on these horses
  4. STAKING (2022 to Nov 2026, chosen before on 2010-21): every selection backed to win GBP100 in the win market
     and GBP100 in the place market, at the morning price where GBP50+ was matched in the morning, else at BSP; and
     at BSP only. By year, the day curve, worst drawdown and longest run of losing days.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
TW = 100.0

df = L.load()
g = df.groupby("horse_name", sort=False)
df["prev_date"] = g["date"].shift(1)
df["prev_fin"] = g["fin"].shift(1)
df["prev_code"] = g["code"].shift(1)
df["prev_won"] = g["won"].shift(1)
df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, (df["date"] - df["prev_date"]).dt.days)
d = L.priced(df).copy()

L.show("1. CONTROL: every runner by code", L.tt(d, ["code"], 1000))
L.show("1b. CONTROL: every runner, DSLR 1-4 against the rest", L.tt(d.assign(q=d["dslr"] <= 4), ["q"], 1000))

q = d["dslr"].between(1, 4)
main = d["code"].isin(["flat", "aw", "hurdle", "chase"])
rows = [
    L.rule("ALL codes: DSLR 1-4", d, q & main),
    L.rule("ALL codes: DSLR 1-3", d, d["dslr"].between(1, 3) & main),
    L.rule("ALL codes: DSLR 1-4, same code as last time", d, q & main & (d["prev_code"] == d["code"])),
    L.rule("ALL codes: DSLR 1-4, placed last time", d, q & main & (d["prev_fin"] <= 3)),
    L.rule("ALL codes: DSLR 1-4, handicap", d, q & main & d["hcap"]),
    L.rule("ALL codes: DSLR 1-4, BSP under 11", d, q & main & (d["bsp"] < 11)),
    L.rule("flat+AW: DSLR 1-4", d, q & d["code"].isin(["flat", "aw"])),
    L.rule("flat+AW: DSLR 1-4, BSP under 11", d, q & d["code"].isin(["flat", "aw"]) & (d["bsp"] < 11)),
    L.rule("jumps: DSLR 1-4", d, q & d["code"].isin(["hurdle", "chase"])),
    L.rule("jumps: DSLR 1-4, BSP under 11", d, q & d["code"].isin(["hurdle", "chase"]) & (d["bsp"] < 11)),
] + [L.rule(f"{c}: DSLR 1-4", d, q & (d["code"] == c)) for c in ("flat", "aw", "hurdle", "chase")]
print("\n=== 2. THE QUICK RETURN (train 2010-21 | t_ test 2022-26) ===")
print(L.gate.__doc__); print(L.gate_am.__doc__)
L.rules_table(rows)

# ---- 3. capacity --------------------------------------------------------------------------------------------------
s = d[q & main & (d["date"] >= "2018-01-01")]
print("\n=== 3. CAPACITY: GBP matched in the morning on DSLR 1-4 selections, 2018+ ===")
print("selections", len(s), "| with a morning price", int((s["morningwap"] > 1).sum()),
      "| morning volume quantiles", s["morning_vol"].quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
for v in (50, 100, 250, 500):
    print(f"  share with GBP{v}+ matched in the morning: {(s['morning_vol'] >= v).mean():.3f}")
print("  stake to win GBP100 at the morning price, median", round(float((TW / (s["morningwap"] - 1)).median()), 2))

# ---- 4. staking ---------------------------------------------------------------------------------------------------
def stake_book(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per selection: stakes and P&L in each market, to win TW in each."""
    x = x.copy()
    use_am = (mode == "am") & (x["morningwap"] > 1) & (x["morning_vol"] >= L.MIN_AM_VOL)
    wp = np.where(use_am, x["morningwap"], x["bsp"])
    use_pam = use_am & (x["p_morningwap"] > 1) & (x["p_morning_vol"] >= L.MIN_AM_VOL / 5)
    pp = np.where(use_pam, x["p_morningwap"], x["pbsp"])
    x["w_stake"] = TW / (wp - 1)
    x["w_pnl"] = np.where(x["won"] == 1, TW * (1 - L.COMMISSION), -x["w_stake"])
    has_p = pp > 1
    x["p_stake"] = np.where(has_p, TW / (pp - 1), 0.0)
    x["p_pnl"] = np.where(~has_p, 0.0, np.where(x["placed"] == 1, TW * (1 - L.COMMISSION), -x["p_stake"]))
    x["at_am"] = use_am
    return x


def report(name: str, x: pd.DataFrame) -> None:
    x = x.copy()
    x["pnl"] = x["w_pnl"] + x["p_pnl"]
    day = x.groupby("date")["pnl"].sum()
    cur = day.cumsum()
    dd = (cur - cur.cummax()).min()
    losing = (day < 0).astype(int)
    runs = losing.groupby((losing == 0).cumsum()).sum().max()
    yr = x.groupby("year").agg(bets=("pnl", "size"), staked=("w_stake", "sum"), p_staked=("p_stake", "sum"),
                               win_pnl=("w_pnl", "sum"), place_pnl=("p_pnl", "sum"))
    yr["staked"] += yr.pop("p_staked")
    yr["pnl"] = yr["win_pnl"] + yr["place_pnl"]
    yr["roi%"] = 100 * yr["pnl"] / yr["staked"]
    st = x["w_stake"].sum() + x["p_stake"].sum()
    print(f"\n--- {name} ---")
    print(f"bets {len(x)} | at the morning price {x['at_am'].mean():.2f} | staked GBP{st:,.0f} | P&L GBP{x['pnl'].sum():,.0f} "
          f"(win {x['w_pnl'].sum():,.0f}, place {x['p_pnl'].sum():,.0f}) | ROI {100 * x['pnl'].sum() / st:.2f}% | "
          f"worst drawdown GBP{dd:,.0f} | longest run of losing days {runs} | days {len(day)}, "
          f"winning days {(day > 0).mean():.2f}")
    print(yr.round(1).to_string())


print("\n=== 4. STAKING: to win GBP100 in the win market AND GBP100 in the place market, each selection ===")
for nm, m in (("ALL codes DSLR 1-4", q & main), ("flat+AW DSLR 1-4", q & d["code"].isin(["flat", "aw"])),
              ("jumps DSLR 1-4", q & d["code"].isin(["hurdle", "chase"])),
              ("ALL codes DSLR 1-4, BSP under 11", q & main & (d["bsp"] < 11))):
    for period, lo, hi in (("TEST 2022-26", "2022-01-01", "2099"), ("TRAIN 2018-21 (morning prices held)", "2018-01-01", "2021-12-31")):
        x = d[m & (d["race_date"] >= lo) & (d["race_date"] <= hi)]
        report(f"{nm} | {period} | morning price where GBP50+ matched, else BSP", stake_book(x, "am"))
        report(f"{nm} | {period} | BSP only", stake_book(x, "bsp"))
