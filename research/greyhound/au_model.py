"""The Australian greyhound model against Betfair's own prices (the owner, 17 Nov 2026: "Build a model for Australia";
ledger greyhound-au-model-1117).

The feasibility study (research/greyhound/au_feasibility.py, ledger greyhound-au-feasibility-1116) found no Betfair
SP for Australia in the archived price files and scored against the API's last back price: not feasible. Betfair's
Historic Data bundle the owner downloaded (sources/betfair_imports.json ``greyhound_basic_2026_01_09``: Jan-Sep 2026,
GB/AU/NZ, built to ``betfair_historic/markets_greyhound_racing_<year>.parquet`` by betfair_historic.py) holds each
Australian WIN market's runners with the BSP, the last price traded before the off (``ltp_close``) and the result.

**The model** (as the feasibility study, research/greyhound/au_feasibility.py ``features``), plus figures on one scale
across tracks (the owner, 17 Nov: "We need to standardize / normalise to use consistently across tracks"):
- the time figure less the meeting's going (each winner against its grade's par, shrunk: greyhound/pit/speed.py
  ``eb_variant``), linked across tracks by the dogs that move (``speed.link_offsets``);
- the first sectional the same way, from the track, trip and box par (greyhound/pit/secfig.py ``figures``), and each
  run's sectional against the field it met. Columns ``lt_*`` (linked time) and ``sx_*`` (linked sectional): last,
  mean of 3, best of 3, recency-weighted, against today's field.
LightGBM win model fitted on the ``TRAIN_YEARS`` before the test year (the test year 2026), the race normalised.

**Scored on Betfair's prices** (Australian WIN markets of the Historic Data, races where every runner has the price):
the market price is the BSP, else (a market without a BSP) the last price traded before the off. The blend (a
conditional logit on logit(model) and logit(market)) is fitted on the test year's first months (before
``SPLIT``) and read after it:
1. Benter's test on the whole test year (the model's coefficient beside the market's, z);
2. the blend's log-loss against the market's after ``SPLIT``, paired by day (2,000 draws);
3. a value rule backing at the market price where the blend's expected value after ``COMMISSION`` exceeds an edge
   (0, 0.05, 0.10, 0.20; price at most ``MAX_PRICE``): the edge chosen on the months before ``SPLIT``, read after.
Read-only, S3 only; writes out/au_model/{report.json, report.md, pred.parquet}.

    python -m research.greyhound.au_model --first-year 2016 --test-year 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.pit import secfig, speed
from greyhound.pit.speed import earlier
from research.greyhound import au_feasibility as af

log = logging.getLogger(__name__)

TRAIN_YEARS = 5
SPLIT = "2026-05-01"
COMMISSION = 0.05        # the gate's rate; 2% also reported (Betfair's rate on Australian greyhound markets for this
                         # account not verified: reported both ways)
MAX_PRICE = 30.0
EDGES = (0.0, 0.05, 0.10, 0.20)


# ---- Betfair's prices ------------------------------------------------------------------------------------------------

def load_historic(store, years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        t = store.get_parquet(f"betfair_historic/markets_greyhound_racing_{y}.parquet")
        if t is None or not len(t):
            continue
        t = t[(t.country.astype(str) == "AU") & (t.market_type.astype(str) == "WIN")]
        parts.append(t[["market_id", "venue", "market_time", "runner_name", "runner_status", "won", "bsp", "ltp_close"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def join_prices(api: pd.DataFrame, bf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each API runner's Betfair BSP and close: the dog's name (Betfair's '1. Name' less its number), the venue and the
    UTC minute of the scheduled off."""
    a = api.copy()
    t = pd.to_datetime(a.start_utc, utc=True, errors="coerce")
    a["utc_min"] = t.dt.strftime("%Y-%m-%d %H:%M")
    a["dk"], a["tk"] = af.norm(a.dog_name), af.norm(a.track)
    if bf is None or not len(bf):
        a["bf_bsp"], a["bf_close"], a["bf_won"] = np.nan, np.nan, np.nan
        return a, {"historic_runners": 0, "note": "no data found: no Australian WIN markets in betfair_historic"}
    b = bf.copy()
    b["utc_min"] = pd.to_datetime(b.market_time, utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    b["dk"] = af.norm(b.runner_name.astype(str).str.replace(r"^\s*\d+\.\s*", "", regex=True))
    b["tk"] = af.norm(b.venue)
    b = b[b.runner_status.astype(str) != "REMOVED"].drop_duplicates(["dk", "tk", "utc_min"], keep="last")
    m = a.merge(b[["dk", "tk", "utc_min", "bsp", "ltp_close", "won", "market_id"]]
                .rename(columns={"bsp": "bf_bsp", "ltp_close": "bf_close", "won": "bf_won"}),
                on=["dk", "tk", "utc_min"], how="left")
    notes = {"api_runners": int(len(a)), "historic_runners": int(len(b)), "matched": int(m.market_id.notna().sum()),
             "match_share_of_historic": round(float(m.market_id.notna().sum() / max(len(b), 1)), 4),
             "with_bsp": int(pd.to_numeric(m.bf_bsp, errors="coerce").gt(1).sum()),
             "with_close": int(pd.to_numeric(m.bf_close, errors="coerce").gt(1).sum()),
             "result_agrees": round(float((m.bf_won.dropna() == (pd.to_numeric(m.position, errors="coerce") == 1)
                                           [m.bf_won.notna()].astype(float)).mean()), 4) if m.bf_won.notna().any() else None,
             "historic_from": str(b.utc_min.min())[:10], "historic_to": str(b.utc_min.max())[:10]}
    return m, notes


# ---- figures on one scale ----------------------------------------------------------------------------------------------

def linked(d: pd.DataFrame) -> pd.DataFrame:
    """The linked time and sectional figures, a column each per run (``lt_run``, ``sx_run``, ``sx_rel``)."""
    r = pd.DataFrame({"race_id": d.race_id.to_numpy(), "dog_id": d.dog_id.to_numpy(), "race_date": d.race_date.to_numpy(),
                      "track": d.track.astype(str).to_numpy(), "distance_m": d.distance_m.to_numpy(),
                      "trap": d.trap.to_numpy(), "position": d.pos.to_numpy(), "sectional": d.sectional_s.to_numpy(),
                      "race_class": d.grade.astype(str).to_numpy() if "grade" in d else "",
                      "meeting_id": (d.track.astype(str) + d.race_date.astype(str)).to_numpy()})
    sx = secfig.figures(r)
    # the time figure (the feasibility study's ``fig``: every runner's own time against the winners' standard)
    day = pd.to_datetime(d.race_date)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    month = (day.dt.year * 12 + day.dt.month).to_numpy(float)
    key = (d.track.astype(str) + "|" + pd.to_numeric(d.distance_m, errors="coerce").round().astype("Int64").astype(str)).to_numpy()
    fig = d.fig.to_numpy(float)
    win = (d.pos == 1).to_numpy() & np.isfinite(fig)
    grade = r.race_class.to_numpy()
    base = pd.DataFrame({"race_date": d.race_date.to_numpy(), "_key": key, "_grade": grade,
                         "_pw": np.where(win, fig, 0.0), "_gw": win.astype(float)})
    from greyhound.pit import core
    gp = core.prior_window(base, ["_key", "_grade"], ["_pw", "_gw"], speed.STD_DAYS)
    par = np.where(gp._gw >= speed.MIN_GRADE, gp._pw / gp._gw.replace(0, np.nan), 0.0)
    early = np.asarray(day < pd.Timestamp(speed.K_BEFORE), bool)
    if not early.any():
        early = dn <= np.nanquantile(dn, 0.2)
    v, k = speed.eb_variant(r.meeting_id.to_numpy(), fig - par, win, early)
    FV = fig - v
    ok = np.isfinite(FV)
    dog = pd.to_numeric(d.dog_id, errors="coerce").to_numpy(float)
    months = np.unique(month[np.isfinite(month)])
    offs = speed.link_offsets(dog, dn, key, FV, month, ok, months)
    off = np.array([offs.get(m, {}).get(kk, 0.0) for m, kk in zip(month, key)])
    out = pd.DataFrame({"lt_run": FV + off, "sx_run": sx["L"], "sx_rel": sx["rel"]}, index=d.index)
    out.attrs["notes"] = {**sx["notes"], "k_time_going": round(k, 3), "time_offset_months": len(offs)}
    return out


def linked_features(d: pd.DataFrame, lk: pd.DataFrame) -> pd.DataFrame:
    dog = pd.to_numeric(d.dog_id, errors="coerce").to_numpy(float)
    dn = ((pd.to_datetime(d.race_date) - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    race = d.race_id.to_numpy()
    f = pd.DataFrame(index=d.index)
    for c, src in (("lt", "lt_run"), ("sx", "sx_run")):
        v = lk[src].to_numpy(float)
        f[f"{c}_last"] = earlier(dog, dn, v, "last")
        f[f"{c}_mean3"] = earlier(dog, dn, v, "mean", n=3)
        f[f"{c}_best3"] = earlier(dog, dn, v, "max", n=3)
        f[f"{c}_ewm"] = earlier(dog, dn, v, "ewm")
        f[f"{c}_vs"] = af._vs(race, f[f"{c}_ewm"].to_numpy(float))
        f[f"{c}_rank"] = pd.Series(f[f"{c}_ewm"].to_numpy(float)).groupby(race).rank(ascending=False).to_numpy()
    f["sx_rel_ewm"] = earlier(dog, dn, lk.sx_rel.to_numpy(float), "ewm")
    f["sx_rel_last"] = earlier(dog, dn, lk.sx_rel.to_numpy(float), "last")
    return f.astype(np.float32)


# ---- the study ---------------------------------------------------------------------------------------------------------

def value_rule(p, price, won, day, edge, commission) -> dict:
    ev = p * (price - 1.0) * (1 - commission) - (1 - p)
    sel = (ev > edge) & (price <= MAX_PRICE)
    if not sel.any():
        return {"edge": edge, "bets": 0}
    ret = np.where(won[sel] == 1, (price[sel] - 1.0) * (1 - commission), -1.0)
    g = pd.DataFrame({"d": day[sel], "r": ret}).groupby("d").r.agg(["sum", "size"])
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(g), (af.DRAWS, len(g)))
    roi = g["sum"].to_numpy()[idx].sum(1) / g["size"].to_numpy()[idx].sum(1)
    daily = (g["sum"] / g["size"]).to_numpy()
    return {"edge": edge, "bets": int(sel.sum()), "roi%": round(100 * float(ret.mean()), 2),
            "roi_90": [round(100 * float(np.quantile(roi, 0.05)), 2), round(100 * float(np.quantile(roi, 0.95)), 2)],
            "mean_price": round(float(price[sel].mean()), 2), "days": int(len(g)),
            "daily_sharpe": round(float(daily.mean() / daily.std()), 4) if len(daily) > 1 and daily.std() > 0 else None}


def run(store, first_year: int, test_year: int, out_dir: Path) -> dict:
    import lightgbm as lgb
    api = af.load_api(store, first_year)
    if not len(api):
        return {"note": "no data found: no GreyhoundAPI Australian results in the store"}
    api = api[api.position.notna()]
    d, jn = join_prices(api, load_historic(store, [test_year]))
    d = af.figures(d)
    d = d[d.won.notna()].reset_index(drop=True)
    X0, cols0 = af.features(d)
    lk = linked(d)
    X1 = linked_features(d, lk)
    X = pd.concat([X0, X1], axis=1)
    cols = list(X.columns)
    d["year"] = d.race_date.str[:4].astype(int)
    out = {"join": jn, "linked_notes": lk.attrs.get("notes"), "rows": int(len(d)), "races": int(d.race_id.nunique()),
           "features": cols, "from": str(d.race_date.min()), "to": str(d.race_date.max())}
    tr = (d.year < test_year) & (d.year >= test_year - TRAIN_YEARS)
    te = d.year == test_year
    if tr.sum() < 10_000 or te.sum() < 1_000:
        out["note"] = "too few rows"
        return out
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X[tr], d.won[tr])
    t = d[te].copy()
    t["p_model"] = af.race_norm(m.predict_proba(X[te])[:, 1], t.race_id.to_numpy())
    bsp = pd.to_numeric(t.bf_bsp, errors="coerce").where(lambda x: x > 1)
    close = pd.to_numeric(t.bf_close, errors="coerce").where(lambda x: x > 1)
    t["price"] = bsp.fillna(close)
    t["price_is_bsp"] = bsp.notna()
    g = t.groupby("race_id")
    full = g.price.transform(lambda s: s.notna().all()) & (g.won.transform("sum") == 1)
    s = t[full].copy()
    out["test_rows_priced"] = int(len(s))
    out["test_races_priced"] = int(s.race_id.nunique())
    out["price_bsp_share"] = round(float(s.price_is_bsp.mean()), 4) if len(s) else None
    if len(s) < 2000:
        out["note"] = "no data found: too few test races with a Betfair price for every runner"
        return out
    rid = s.race_id.to_numpy()
    s["p_mkt"] = af.race_norm(1.0 / s.price.to_numpy(float), rid)
    Xb = np.column_stack([af._logit(s.p_model.to_numpy()), af._logit(s.p_mkt.to_numpy())])
    b, se = af.clogit(Xb, s.won.to_numpy(float), rid)
    out["benter"] = {"coef_model": round(float(b[0]), 4), "z_model": round(float(b[0] / se[0]), 2),
                     "coef_market": round(float(b[1]), 4), "z_market": round(float(b[1] / se[1]), 2)}
    pre = (s.race_date < SPLIT).to_numpy()
    post = ~pre
    out["split"] = {"before": int(pre.sum()), "after": int(post.sum()), "at": SPLIT}
    if pre.sum() < 2000 or post.sum() < 2000:
        out["note"] = "too few priced rows on one side of the split"
        return out
    bp, _ = af.clogit(Xb[pre], s.won.to_numpy(float)[pre], rid[pre])
    s["p_blend"] = af.apply_clogit(Xb, bp, rid)
    sp = s[post]
    out["logloss_blend_vs_market_after"] = af.paired(sp.race_date.to_numpy(), af.ll(sp.p_blend.to_numpy(), sp.won.to_numpy()),
                                                      af.ll(sp.p_mkt.to_numpy(), sp.won.to_numpy()))
    out["logloss_model_vs_market_after"] = af.paired(sp.race_date.to_numpy(), af.ll(sp.p_model.to_numpy(), sp.won.to_numpy()),
                                                      af.ll(sp.p_mkt.to_numpy(), sp.won.to_numpy()))
    vr = {}
    for c in (COMMISSION, 0.02):
        before = [value_rule(s.p_blend.to_numpy()[pre], s.price.to_numpy(float)[pre], s.won.to_numpy()[pre],
                             s.race_date.to_numpy()[pre], e, c) for e in EDGES]
        after = [value_rule(sp.p_blend.to_numpy(), sp.price.to_numpy(float), sp.won.to_numpy(), sp.race_date.to_numpy(), e, c)
                 for e in EDGES]
        best = max((v for v in before if v["bets"]), key=lambda v: v["roi%"], default=None)
        vr[f"{c:.2f}"] = {"before": before, "after": after, "chosen_edge": best["edge"] if best else None}
    out["value_rule"] = vr
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    out["top_features"] = {k: int(v) for k, v in imp.head(15).items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    s[["race_id", "dog_id", "race_date", "track", "trap", "won", "price", "price_is_bsp", "p_model", "p_mkt", "p_blend"]] \
        .to_parquet(out_dir / "pred.parquet", index=False)
    return out


def report_md(r: dict) -> str:
    L = ["# Australian greyhound model against Betfair's prices", ""]
    if "join" in r:
        L += [f"The join to Betfair's Historic Data: {r['join']}.", ""]
    if r.get("linked_notes"):
        L += [f"Linked figures: {r['linked_notes']}.", ""]
    if "note" in r:
        return "\n".join(L + [r["note"]]) + "\n"
    L.append(f"{r['rows']:,} runners in {r['races']:,} races ({r['from']} to {r['to']}); test year priced: "
             f"{r['test_races_priced']:,} races ({r['price_bsp_share']} of runners at the BSP, the rest at the last "
             f"traded price).")
    bt = r["benter"]
    L += ["", f"Benter: model {bt['coef_model']} (z {bt['z_model']}), market {bt['coef_market']} (z {bt['z_market']}).",
          f"After {r['split']['at']}: blend log-loss {r['logloss_blend_vs_market_after']['a']} against the market's "
          f"{r['logloss_blend_vs_market_after']['b']}, {r['logloss_blend_vs_market_after']['diff']:+} "
          f"(z {r['logloss_blend_vs_market_after']['z']}); model alone {r['logloss_model_vs_market_after']['a']}.", ""]
    for c, v in r["value_rule"].items():
        L += [f"Value rule, commission {c} (edge chosen before the split: {v['chosen_edge']}):", "",
              "| Edge | Bets before | Return before % | Bets after | Return after % | 90% after | Mean price after |",
              "|---|---|---|---|---|---|---|"]
        for b0, a0 in zip(v["before"], v["after"]):
            L.append(f"| {b0['edge']} | {b0.get('bets', 0):,} | {b0.get('roi%', '-')} | {a0.get('bets', 0):,} | "
                     f"{a0.get('roi%', '-')} | {a0.get('roi_90', '-')} | {a0.get('mean_price', '-')} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-year", type=int, default=2016)
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/au_model")
    a = ap.parse_args(argv)
    o = Path(a.out)
    r = run(Store(root=a.store), a.first_year, a.test_year, o)
    o.mkdir(parents=True, exist_ok=True)
    (o / "report.json").write_text(json.dumps(r, indent=1, default=str))
    md = report_md(r)
    (o / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
