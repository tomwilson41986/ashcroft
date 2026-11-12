"""The owner's five new builds scored directly with the framework's metrics (read-only; descriptive; ledger
`new-blocks-scored-1112`; the owner, 12 Nov 2026: "score all of these with our framework metrics: normalised
metrics, uplift, full suite").

The loop asks whether a block improves the BFSP forecast. This asks the plainer question of each reading itself:
do horses it rates high run better, and better than their price says? The cached matrix, the blocks built on it as
the loop builds them (every feature from earlier days only), scored on runs from 30 Sep 2023 to 31 Mar 2026 (the
locked holdout from 1 Apr 2026 is not read). For each group of runners:

    n        runners                     win%     winners
    exp%     the BSP's chances (normalised within the race)
    A/E      winners over the BSP's chances      A-E/run  winners less chances, per runner (uplift on the price)
    NFP      normalised finishing position (1 won, 0 last; a non-finisher 0)
    s        NFP less the position the BSP rank implied (the normalised uplift)
    fig+     the run's time figure less the horse's own decayed mean figure, in pounds

Descriptive: no pass or fail, nothing fitted. Groups are fifths of a reading within the scored rows, or flags.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
pd.set_option("display.width", 220)
from model import blocks, feature_cache  # noqa: E402
from model.blocks import time_figure as _tf  # noqa: E402
from model.race_shape import asof_decayed_mean, day_index, race_key  # noqa: E402

NAMES = ["market_rating", "expected_time", "connections_rating", "gear", "placement", "form_state"]
need = {"raceid", "race_date", "race_time", "track", "horse_name", "bfsp", "placing_numerical", "number_of_runners"}
for nme in NAMES + ["pace_v2", "time_figure"]:
    need |= set(getattr(blocks.load(nme) if hasattr(blocks, "load") else __import__(f"model.blocks.{nme}", fromlist=["x"]),
                        "READS", []))
import pyarrow.parquet as pq  # noqa: E402

mat = None
for start in ("2018-01-01", "2021-01-01"):
    data, _ = feature_cache._paths(".feature_cache", feature_cache.cache_key("horse_racing.db", start))
    if data.exists():
        have = set(pq.read_schema(data).names)
        mat = pd.read_parquet(data, columns=[c for c in need if c in have])
        print(f"matrix from {start}: {len(mat):,} rows; columns asked and not held: {sorted(c for c in need if c not in have)[:20]}")
        break
if mat is None:
    raise SystemExit("no data found: no cached matrix on this runner")
mat, feats = blocks.attach(mat, NAMES)
day = day_index(mat)
race = pd.Series(pd.factorize(race_key(mat), sort=True)[0], index=mat.index)
pos = pd.to_numeric(mat.placing_numerical, errors="coerce").to_numpy(float)
n = pd.to_numeric(mat.number_of_runners, errors="coerce").to_numpy(float)
bsp = pd.to_numeric(mat.bfsp, errors="coerce").to_numpy(float)
priced = np.isfinite(bsp) & (bsp > 1)
inv = pd.Series(np.where(priced, 1 / np.where(priced, bsp, 1), 0.0), index=mat.index)
mat["p"] = (inv / inv.groupby(race).transform("sum")).where(priced)
rk = pd.Series(np.where(priced, bsp, np.nan), index=mat.index).groupby(race).rank(method="average").to_numpy()
npr = pd.Series(priced.astype(float), index=mat.index).groupby(race).transform("sum").to_numpy()
with np.errstate(invalid="ignore", divide="ignore"):
    mat["nfp"] = np.where(np.isfinite(pos) & (pos >= 1) & (n > 1), np.clip((n - pos) / (n - 1), 0, 1), np.where(n > 1, 0.0, np.nan))
    mat["s"] = mat.nfp - np.where(priced & (npr > 1), (npr - rk) / (npr - 1), np.nan)
mat["won"] = (pos == 1).astype(float)
fig = np.asarray(_tf.figures(mat)["fig"], dtype=float)
h = pd.factorize(mat.horse_name.astype(str).str.lower().str.strip(), sort=True)[0]
own, _ = asof_decayed_mean(h, day, fig, h, day, halflife_days=365.0)
mat["fig"], mat["figp"] = fig, fig - own
d = pd.to_datetime(mat.race_date, errors="coerce")
S = mat[(d >= "2023-09-30") & (d <= "2026-03-31") & mat.p.notna()].copy()
print(f"scored: {len(S):,} runners, {S.raceid.nunique() if 'raceid' in S else race[S.index].nunique():,} races, "
      f"{d[S.index].min():%Y-%m-%d} .. {d[S.index].max():%Y-%m-%d}")


def table(g, label):
    t = g.agg(n=("won", "size"), win=("won", "mean"), exp=("p", "mean"), nfp=("nfp", "mean"), s=("s", "mean"),
              figp=("figp", "mean"))
    t["A/E"] = t.win / t.exp
    t["A-E/run"] = t.win - t.exp
    t = t.assign(win=t.win * 100, exp=t.exp * 100)[["n", "win", "exp", "A/E", "A-E/run", "nfp", "s", "figp"]]
    t.columns = ["n", "win%", "exp%", "A/E", "A-E/run", "NFP", "s", "fig+"]
    print(f"\n{label}")
    print(t.round({"win%": 2, "exp%": 2, "A/E": 3, "A-E/run": 4, "NFP": 4, "s": 4, "fig+": 2}).to_string())


def fifths(col, label):
    if col not in S or S[col].notna().sum() < 5000:
        print(f"\n{label}: no data found ({col})")
        return
    q = pd.qcut(S[col], 5, labels=["lowest", "2", "3", "4", "highest"], duplicates="drop")
    table(S.groupby(q, observed=True), f"{label} [{col}, fifths]")


def flags(cols, label):
    rows = {}
    for c in cols:
        if c in S:
            rows[c] = S[S[c] == 1]
    rows["(all scored)"] = S
    t = pd.concat({k: v.assign(_g=k) for k, v in rows.items()}).groupby("_g", sort=False)
    table(t, label)


print("\n================ 1. EXPECTED TIME")
ok = S.et_pred.notna() & S.fig.notna()
e, base = S.fig[ok] - S.et_pred[ok], S.fig[ok] - (S.fig[ok] - S.figp[ok])
print(f"forecast against the figure run: {ok.sum():,} runs; correlation {np.corrcoef(S.et_pred[ok], S.fig[ok])[0, 1]:.3f}; "
      f"mean absolute error {e.abs().mean():.2f} lb; the horse's own decayed mean as the forecast: "
      f"{base.abs().mean():.2f} lb on {base.notna().sum():,} runs with one")
table(S[S.et_rank.notna()].assign(r=S.et_rank.clip(upper=6)).groupby("r"), "by the forecast's rank in the race (6 = sixth or worse)")
fifths("et_uplift", "what today's conditions, connections and set-up add to the horse's own level")
fifths("et_win", "the forecast's share of the race")
print("\n================ 2. CONNECTIONS RATING")
for c, lab in (("cr_tr_s", "trainer, against the price"), ("cr_jk_s", "jockey, against the price"), ("cr_sum_s", "trainer + jockey, against the price"),
               ("cr_tr_f", "trainer, against the horse's level"), ("cr_jk_f", "jockey, against the horse's level"),
               ("cr_jk_new_s", "the jockey's record on rides taken over")):
    fifths(c, lab)
ch = S[S.cr_jk_change == 1]
if len(ch):
    q = pd.qcut(ch.cr_jk_delta_s, 5, labels=["much weaker", "2", "3", "4", "much stronger"], duplicates="drop")
    table(ch.groupby(q, observed=True), "a change of jockey: the new rider's rating less the old one's [cr_jk_delta_s]")
flags(["cr_jk_change"], "jockey change, all")
print("\n================ 3. GEAR")
flags(["gr_first", "gr_first_blnk", "gr_first_ckpc", "gr_first_vsor", "gr_first_hood", "gr_first_tt", "gr_more", "gr_less",
       "gr_off", "gr_tt_on", "gr_tt_off"], "gear moves")
table(S.groupby("gr_sev"), "by severity worn today (0 none, 1 hood/eyeshield, 2 cheekpieces, 3 visor, 4 blinkers)")
for c, lab in (("gr_tr_first_s", "first-time gear: the yard's record with it"), ("gr_tr_more_s", "more severe gear: the yard's record with it")):
    sub = S[S[c] != 0]
    if len(sub) > 2000:
        q = pd.qcut(sub[c], 3, labels=["yard poor with it", "middle", "yard good with it"], duplicates="drop")
        table(sub.groupby(q, observed=True), f"{lab} [{c}, thirds]")
print("\n================ 4. PLACEMENT AND INTENT")
S["prize_up"], S["prize_down"] = (S.pl_prize_vs_usual >= np.log(1.5)).astype(float), (S.pl_prize_vs_usual <= np.log(2 / 3)).astype(float)
S["mark_drop5"], S["trip_up"], S["trip_down"] = (S.pl_mark_d3 <= -5).astype(float), (S.pl_trip_change >= np.log(1.2)).astype(float), \
    (S.pl_trip_change <= -np.log(1.2)).astype(float)
flags(["prize_up", "prize_down", "mark_drop5", "pl_repeat", "trip_up", "trip_down", "pl_first_trip"], "placement moves")
for c, lab in (("pl_tr_prize_down_s", "dropped in prize: the yard's record"), ("pl_tr_mark_drop_s", "mark dropped 5 lb: the yard's record"),
               ("pl_tr_repeat_s", "back for a race it ran in before: the yard's record")):
    sub = S[S[c] != 0]
    if len(sub) > 2000:
        q = pd.qcut(sub[c], 3, labels=["yard poor with it", "middle", "yard good with it"], duplicates="drop")
        table(sub.groupby(q, observed=True), f"{lab} [{c}, thirds]")
fifths("pl_cycle_s", "the horse's own record at this point of a campaign")
print("\n================ 5. FORM AS A STATE")
fifths("fs_p_form", "Markov chain: the probability the horse is in form")
fifths("tk_z", "Kalman rating on the time figure, within the race")
fifths("mr_z", "for scale: the market rating within the race")
