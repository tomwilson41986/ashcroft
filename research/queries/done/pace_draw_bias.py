"""Draw, early position and race shape: what each does to a runner, by course and trip, and whether the price knows.

The owner's asks of 25 Sep (with the pace papers): certain tracks have draw and
position biases -- check them by draw, draw segment and early position, in NFP,
beaten lengths and the rest; certain race shapes suit certain styles (a lone
leader dictates, a closer wants a hot pace); forecast the shape and see how it
would benefit each runner.

On the real database, development window only (2021-01-01 .. 2026-03-31; 2019-20
feed the style projections their history). Every cell reports:
    runners, races, win %, IV   wins / the wins a random runner would get (sum 1/N):
                                the impact value of the pace handicappers
    NFP                         mean normalised finishing position (0.5 = average)
    lbs_c                       pounds better than the race's average runner
    A/E                         wins / the BSP's expected wins: 1.00 = the market
                                priced it; > 1 = it underrated these runners
    z                           (wins - expected) / its binomial standard error
Positions come from the in-running comments (model/race_shape.py: rs_class 0 led,
1 prominent, 2 mid-division, 3 held up). "Realized" is where the runner actually
raced -- hindsight, the size of the prize if it could be forecast; "projected" is
the pre-race forecast from the horse's earlier runs (p_lead..p_rear), which is
what a model can use. Race shape the same two ways: realized from how many led
and raced prominently, forecast from the field's projected chances of leading.

Writes out/pace_draw_*.csv for the owner's workbook.
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
t0 = time.time()

FROM, WARM, UNTIL = "2021-01-01", "2019-01-01", "2026-04-01"
OUT = Path("out")
OUT.mkdir(exist_ok=True)

from model.race_shape import add_race_shape_features, race_code, race_key  # noqa: E402
from model.blocks.draw_v2 import going_group  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query(f"""
    SELECT race_date, race_time, track, horse_name, number_of_runners, placing_numerical, total_dst_bt,
           dist_furlongs, race_type, surface_type, comment, stall, stall_positioning, going_description, bfsp
    FROM race_results WHERE race_date >= '{WARM}' AND race_date < '{UNTIL}'""", conn)
d["race_date"] = pd.to_datetime(d["race_date"])
d = d.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)
print(f"{len(d):,} rows {d.race_date.min().date()}..{d.race_date.max().date()} loaded in {time.time() - t0:.0f}s")
d, _ = add_race_shape_features(d)
print(f"race shape built in {time.time() - t0:.0f}s")
d = d[d["race_date"] >= FROM].reset_index(drop=True)

# ---- the outcomes -----------------------------------------------------------------
rk = race_key(d)
race = pd.factorize(rk)[0]
n = pd.to_numeric(d["number_of_runners"], errors="coerce").fillna(pd.Series(race).map(pd.Series(race).value_counts()))
pos = pd.to_numeric(d["placing_numerical"], errors="coerce")
has_result = pos.gt(0).groupby(race).transform("any")
d = d[has_result.to_numpy()].reset_index(drop=True)
rk = race_key(d)
race = pd.factorize(rk)[0]
n = pd.to_numeric(d["number_of_runners"], errors="coerce").to_numpy(dtype=float)
pos = pd.to_numeric(d["placing_numerical"], errors="coerce").to_numpy(dtype=float)
d["won"] = (pos == 1).astype(float)
d["rid"] = race
d["rand"] = 1.0 / np.where(n > 0, n, np.nan)
bsp = pd.to_numeric(d["bfsp"], errors="coerce")
p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
d["p_mkt"] = (p / p.groupby(race).transform("sum")).to_numpy()
d["nfp"] = d["rs_nfp"]
d["lbs_c"] = d["rs_lbs_c"]
d["code"] = race_code(d).to_numpy()
dist = pd.to_numeric(d["dist_furlongs"], errors="coerce")
d["trip"] = pd.cut(dist, [0, 6.5, 8.5, 12.5, 99], labels=["5-6f", "7f-1m", "1m1f-1m4f", "1m4f+"]).astype(str)
flat = d["code"].isin(["flat", "aw"])
stall = pd.to_numeric(d["stall"], errors="coerce")
rel = (stall - 1) / (n - 1)
d["draw"] = np.where(flat & (stall > 0) & (n >= 6) & (stall <= n),
                     np.select([rel <= 1 / 3, rel <= 2 / 3], ["1 low", "2 mid"], "3 high"), "")
d["going"] = going_group(d).to_numpy()
cls = {0.0: "1 led", 1.0: "2 prominent", 2.0: "3 mid", 3.0: "4 held up"}
d["pos_real"] = d["rs_class"].map(cls).fillna("")
proj = d[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy(dtype=float)
ok = np.isfinite(proj).all(1) & (d["style_n_eff"].to_numpy(dtype=float) >= 1)
d["pos_proj"] = np.where(ok, np.array(["1 led", "2 prominent", "3 mid", "4 held up"])[np.nan_to_num(proj).argmax(1)], "")

# realized shape: how many led and how many raced on or near the pace
rc = d["rs_class"]
n_lead = (rc == 0).groupby(race).transform("sum")
n_front = (rc <= 1).groupby(race).transform("sum")
parsed = rc.notna().groupby(race).transform("mean")
d["shape_real"] = np.select(
    [parsed < 0.6, n_lead == 0, (n_lead == 1) & (n_front <= 2), n_lead >= 2],
    ["", "no clear leader", "lone leader", "contested lead"], "led, pressed")
# forecast shape: the engine's (shape_bin 0 no natural leader, 1 one, 2 contested)
d["shape_fc"] = d["shape_bin"].map({0.0: "fc: no natural leader", 1.0: "fc: one likely leader",
                                    2.0: "fc: contested"}).fillna("")
print(f"{len(d):,} runners in {d['rid'].nunique():,} races "
      f"{d.race_date.min().date()}..{d.race_date.max().date()}; comments parsed for {d.rs_class.notna().mean():.1%}; "
      f"projection held for {(d.pos_proj != '').mean():.1%}")


def cells(frame: pd.DataFrame, by: list[str], min_runners: int = 200) -> pd.DataFrame:
    f = frame[frame["p_mkt"].notna()]
    grp = f.groupby(by, observed=True)
    t = pd.DataFrame({
        "runners": grp.size(),
        "races": grp["rid"].nunique(),
        "wins": grp["won"].sum(),
        "rand": grp["rand"].sum(),
        "exp": grp["p_mkt"].sum(),
        "var": (f["p_mkt"] * (1 - f["p_mkt"])).groupby([f[b] for b in by], observed=True).sum(),
        "NFP": grp["nfp"].mean(),
        "lbs_c": grp["lbs_c"].mean(),
    })
    t = t[t["runners"] >= min_runners]
    t["win%"] = 100 * t["wins"] / t["runners"]
    t["IV"] = t["wins"] / t["rand"]
    t["A/E"] = t["wins"] / t["exp"]
    t["z"] = (t["wins"] - t["exp"]) / np.sqrt(t["var"])
    return t[["runners", "races", "win%", "IV", "NFP", "lbs_c", "A/E", "z"]].round(
        {"win%": 1, "IV": 2, "NFP": 3, "lbs_c": 2, "A/E": 3, "z": 1})


def show(title: str, t: pd.DataFrame, name: str, head: int | None = None) -> None:
    t.to_csv(OUT / f"pace_draw_{name}.csv")
    print(f"\n## {title}  [{len(t)} cells -> out/pace_draw_{name}.csv]")
    print((t if head is None else t.head(head)).to_string())


fl = d[d["draw"] != ""]
# 1. draw
show("1. Draw third by code and trip (flat and all-weather, 6+ runners)", cells(fl, ["code", "trip", "draw"]), "draw_code_trip")
show("1b. Draw third by code, trip and going", cells(fl, ["code", "trip", "going", "draw"], 500), "draw_going")
sp = d[flat & d["stall_positioning"].notna() & (d["draw"] != "")].copy()
sp["stalls"] = sp["stall_positioning"].astype(str).str.strip().str.lower()
show("1c. Draw third by where the stalls were placed (flat, 5-6f and 7f-1m)",
     cells(sp[sp.trip.isin(["5-6f", "7f-1m"])], ["trip", "stalls", "draw"], 500), "draw_stalls")
cd = cells(fl, ["track", "trip", "draw"], 150).reset_index()
wide = cd.pivot_table(index=["track", "trip"], columns="draw", values=["NFP", "lbs_c", "IV", "A/E", "runners"])
wide.columns = [f"{a}_{b.split()[1]}" for a, b in wide.columns]
wide = wide.dropna(subset=["NFP_low", "NFP_high"])
wide["NFP_low_minus_high"] = (wide["NFP_low"] - wide["NFP_high"]).round(3)
wide["lbs_low_minus_high"] = (wide["lbs_c_low"] - wide["lbs_c_high"]).round(2)
wide["AE_low_minus_high"] = (wide["A/E_low"] - wide["A/E_high"]).round(3)
wide = wide.sort_values("NFP_low_minus_high", key=lambda s: -s.abs())
show("1d. Draw by course and trip: low against high (sorted by the NFP gap; A/E gap near 0 = the price knows)",
     wide[["runners_low", "runners_high", "IV_low", "IV_high", "NFP_low_minus_high", "lbs_low_minus_high",
           "A/E_low", "A/E_high", "AE_low_minus_high"]], "draw_course", head=40)

# 2. early position
ep = d[d["pos_real"] != ""]
show("2. Where the runner actually raced (hindsight), by code and trip", cells(ep, ["code", "trip", "pos_real"]), "pos_real_code")
pj = d[d["pos_proj"] != ""]
show("2b. Where it was projected to race (before the off), by code and trip", cells(pj, ["code", "trip", "pos_proj"]), "pos_proj_code")
cr = cells(ep[ep.pos_real == "1 led"], ["track", "code", "trip"], 150).sort_values("IV", ascending=False)
show("2c. Leaders' impact value by course and trip (hindsight): where front-running pays most", cr, "leaders_course", head=30)
cp = cells(pj[pj.pos_proj == "1 led"], ["track", "code", "trip"], 150).sort_values("A/E", ascending=False)
show("2d. Projected leaders by course and trip: A/E against the price", cp, "proj_leaders_course", head=30)

# 3. draw x position
dp = d[(d["draw"] != "") & (d["pos_real"] != "")]
show("3. Draw third x where it raced (flat, 5-6f and 7f-1m)", cells(dp[dp.trip.isin(["5-6f", "7f-1m"])],
     ["trip", "draw", "pos_real"], 500), "draw_x_pos")

# 4. race shape x style
sh = d[(d["shape_real"] != "") & (d["pos_real"] != "")]
show("4. Realized shape x where it raced (hindsight): who each shape suits", cells(sh, ["code", "shape_real", "pos_real"], 300),
     "shape_real_x_pos")
fc = d[(d["shape_fc"] != "") & (d["pos_proj"] != "")]
show("4b. Forecast shape x projected position (before the off): what a model can use", cells(fc, ["code", "shape_fc", "pos_proj"], 300),
     "shape_fc_x_proj")
lone = d[(d["shape_fc"] == "fc: one likely leader") & (d["lead_rank"] == 1)]
show("4c. The one likely leader in a race forecast to have one, by code and trip", cells(lone, ["code", "trip"], 200),
     "lone_leader")

# 5. how well the shape is forecast
x = d[(d["pos_real"] != "") & d["p_lead"].notna()]
led = (x["rs_class"] == 0).to_numpy()
pl = x["p_lead"].to_numpy(dtype=float)
order = np.argsort(pl)
ranks = np.empty(len(pl)); ranks[order] = np.arange(1, len(pl) + 1)
auc = (ranks[led].sum() - led.sum() * (led.sum() + 1) / 2) / (led.sum() * (~led).sum())
print(f"\n## 5. Forecasting the pace\n   p_lead against who actually led: AUC {auc:.3f} over {len(x):,} runners; "
      f"base rate {led.mean():.3f}")
bins = pd.cut(x["p_lead"], [0, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0])
cal = x.assign(led=led).groupby(bins, observed=True).agg(runners=("led", "size"), p_lead=("p_lead", "mean"), led=("led", "mean"))
print(cal.round(3).to_string())
top = x[x["lead_rank"] == 1]
print(f"   the most likely leader led in {(top.rs_class == 0).mean():.1%} of {len(top):,} races")
rr = x.groupby(race_key(x)).agg(exp=("shape_exp_leaders", "first"), real=("rs_class", lambda s: (s == 0).sum()))
print(f"   the field's expected leaders against those who led: correlation {rr.exp.corr(rr.real):.3f} over {len(rr):,} races")
ct = pd.crosstab(x.groupby(race_key(x))["shape_fc"].first(), x.groupby(race_key(x))["shape_real"].first(), normalize="index")
print("   forecast shape (rows) against the shape the race had (columns), share of races:")
print(ct.round(3).to_string())
ct.to_csv(OUT / "pace_draw_shape_confusion.csv")
print(f"\n{time.time() - t0:.0f}s")
