"""One day's greyhound results read end to end (the owner, 11 Nov 2026: "do an exhaustive diagnostic on our greyhound
results today. Do we have gaps or issues with our model?"). Read-only: S3 in, a report out; no Betfair call.

What it reads: the day's live predictions (greyhound/live/predictions/<day>.csv), the settled books (the BSP and
the winner, greyhound/fav_back.closing), the early lay's ledgers (live, paper and the paper variants), their status
and reconciliation, and the favourite back's paper ledger.

What it reports: coverage (settled races with no price from us, and why it can tell); the price against the SP
(RMSE of log, bias, by price band, track, hour, field, the worst misses; the win chance against the SP's); the early
lay as traded (what was laid, what matched, by price band, track and hour; the laid price against the SP for the
lays that matched and the ones that did not; how each was closed; the refusals and errors); live against paper on
the same lays; the trader's gaps.

    python -m research.greyhound.day_diagnostic --date 2026-11-11 --out out/greyhound_day
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 1001.0]
num = lambda v: pd.to_numeric(v, errors="coerce")                # noqa: E731


def _csv(store, key: str) -> pd.DataFrame | None:
    raw = store.get(key)
    if not raw:
        return None
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, on_bad_lines="skip")


def _md(df: pd.DataFrame, floats: int = 3) -> list[str]:
    if df is None or not len(df):
        return ["no data found"]
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floats)
    L = ["| " + " | ".join(str(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    L += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
          for r in d.itertuples(index=False)]
    return L


def band(x: pd.Series) -> pd.Series:
    return pd.cut(x, BANDS, right=False, labels=[f"{a:g}-{b:g}" if b < 1000 else f"{a:g}+" for a, b in
                                                 zip(BANDS[:-1], BANDS[1:])])


def price_table(j: pd.DataFrame, by: str) -> pd.DataFrame:
    g = j.groupby(by, observed=True)
    return pd.DataFrame({"runners": g.size(), "rmse_log": np.sqrt(g.err2.mean()), "bias_log": g.err.mean(),
                         "rmse_raw": np.sqrt(g.err2_raw.mean()) if "err2_raw" in j else np.nan}).reset_index()


def price_section(preds: pd.DataFrame, close: pd.DataFrame) -> tuple[list[str], dict, pd.DataFrame]:
    L, out = ["## 2. The model's price against the Betfair SP"], {}
    p = preds.assign(selection_id=num(preds.selection_id))
    j = p.merge(close, on=["market_id", "selection_id"], how="inner")
    j = j[(num(j.pred_bsp) > 1) & (j.bsp > 1)].copy()
    if not len(j):
        return L + ["", "no data found (no priced runner has an SP yet)"], out, j
    j["err"] = np.log(num(j.pred_bsp)) - np.log(j.bsp)
    j["err2"] = j.err ** 2
    if "pred_bsp_raw" in j:
        j["err2_raw"] = (np.log(num(j.pred_bsp_raw)) - np.log(j.bsp)) ** 2
    j["field"] = j.groupby("market_id").selection_id.transform("size")
    j["hour"] = j.race_time.astype(str).str[:2]
    j["bsp_band"] = band(j.bsp)
    j["pred_band"] = band(num(j.pred_bsp))
    q = 1.0 / j.bsp
    j["q_sp"] = q / q.groupby(j.market_id).transform("sum")
    pm = num(j.p_model)
    j["p_n"] = pm / pm.groupby(j.market_id).transform("sum")
    one = j.groupby("market_id").won.transform("sum") == 1
    w = j[one & (j.won == 1)]
    rk_m = j.groupby("market_id").p_n.rank(ascending=False, method="first")
    rk_s = j.groupby("market_id").q_sp.rank(ascending=False, method="first")
    rho = j.assign(a=np.log(num(j.pred_bsp)), b=np.log(j.bsp)).groupby("market_id")[["a", "b"]].apply(
        lambda g: g.a.corr(g.b, method="spearman") if len(g) > 2 else np.nan)
    out = {"runners": int(len(j)), "races": int(j.market_id.nunique()),
           "rmse_log_bsp": round(float(np.sqrt(j.err2.mean())), 4), "bias_log": round(float(j.err.mean()), 4),
           "rmse_log_bsp_unstretched": round(float(np.sqrt(j.err2_raw.mean())), 4) if "err2_raw" in j else None,
           "within_race_rank_corr_mean": round(float(rho.mean()), 3),
           "model_fav_is_sp_fav%": round(100 * float(((rk_m == 1) & (rk_s == 1)).sum() / max((rk_s == 1).sum(), 1)), 1),
           "sp_fav_won%": round(100 * float(j[rk_s == 1].won.mean()), 1),
           "model_fav_won%": round(100 * float(j[rk_m == 1].won.mean()), 1),
           "logloss_model": round(float(-np.log(w.p_n.clip(1e-6)).mean()), 4) if len(w) else None,
           "logloss_sp": round(float(-np.log(w.q_sp.clip(1e-6)).mean()), 4) if len(w) else None,
           "races_with_a_result": int(one.groupby(j.market_id).first().sum())}
    L += ["", "The test's figure for this model (1 Jun to 5 Nov 2026, reports/greyhound_model.md section 36): RMSE of "
          "log(BSP) 0.3396, 0.3366 with the per-track stretch. Today:", "", "```", json.dumps(out, indent=1), "```"]
    for by, name in (("bsp_band", "By the SP"), ("pred_band", "By our price"), ("track", "By track"),
                     ("hour", "By hour (UK)"), ("field", "By field size"), ("trap", "By trap")):
        if by in j:
            L += ["", f"**{name}**", ""] + _md(price_table(j, by))
    # is the forecast too flat or too steep: the slope of log(BSP) on log(our price) within the race
    c = j.assign(a=np.log(num(j.pred_bsp)), b=np.log(j.bsp))
    c["a"] -= c.groupby("market_id").a.transform("mean")
    c["b"] -= c.groupby("market_id").b.transform("mean")
    slope = float((c.a * c.b).sum() / (c.a ** 2).sum()) if (c.a ** 2).sum() > 0 else np.nan
    out["slope_log_bsp_on_log_price_within_race"] = round(slope, 3)
    L += ["", f"Slope of log(BSP) on log(our price) within the race: **{slope:.3f}** (1 is right; above 1 our prices "
          "are too flat, the market spreads the field wider than we do)."]
    worst = j.assign(abs=j.err.abs()).sort_values("abs", ascending=False).head(20)
    cols = [c_ for c_ in ("race_time", "track", "trap", "dog_name", "dog", "pred_bsp", "bsp", "err", "won") if c_ in worst]
    L += ["", "**The 20 worst misses** (err above 0: we priced it too long)", ""] + _md(worst[cols])
    return L, out, j


def coverage_section(preds, close, markets) -> tuple[list[str], dict]:
    L = ["## 1. Coverage: did every race get a price"]
    settled = set(close.market_id) if close is not None else set()
    priced = set(preds.market_id) if preds is not None else set()
    out = {"settled_markets": len(settled), "markets_priced": len(priced),
           "settled_and_priced": len(settled & priced), "settled_not_priced": len(settled - priced),
           "priced_runners": int(len(preds)) if preds is not None else 0}
    if preds is not None and close is not None and len(close):
        p = preds.assign(selection_id=num(preds.selection_id))
        ran = close[close.market_id.isin(priced)]
        m = ran.merge(p[["market_id", "selection_id"]].assign(_p=1), on=["market_id", "selection_id"], how="left")
        out["runners_with_sp_in_priced_markets"] = int(len(ran))
        out["of_them_not_priced"] = int(m._p.isna().sum())
        extra = p[p.market_id.isin(settled)].merge(close[["market_id", "selection_id"]].assign(_c=1),
                                                    on=["market_id", "selection_id"], how="left")
        out["priced_runners_with_no_sp (non-runners, reserves)"] = int(extra._c.isna().sum())
    L += ["", "```", json.dumps(out, indent=1), "```"]
    miss = sorted(settled - priced)
    if miss and markets is not None and len(markets):
        mk = markets.assign(market_id=markets.market_id.astype(str)).drop_duplicates("market_id")
        mk = mk[mk.market_id.isin(miss)]
        keep = [c for c in ("market_id", "venue", "event_name", "market_name", "market_start_time", "start_utc",
                            "country") if c in mk]
        L += ["", f"**Settled races with no price from us ({len(miss)})**", ""] + _md(mk[keep].head(80))
        if "venue" in mk:
            L += ["", "By venue:", ""] + _md(mk.groupby("venue").size().rename("races").reset_index())
    elif miss:
        L += ["", f"{len(miss)} settled races had no price; the markets file gave no names."]
    return L, out


def lay_frame(led: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """One row a lay sent: its price, what matched, how it closed, its locked P&L, the SP and the result."""
    led = led.assign(selection_id=num(led.selection_id))
    lays = led[led.action == "lay"].copy()
    if not len(lays):
        return lays
    fin = led[led.action.isin(["green", "green_part", "sp_close", "hold"])]
    g = fin.groupby(["market_id", "selection_id"])
    agg = pd.DataFrame({"lay_matched": g.lay_matched.apply(lambda s: num(s).max()),
                        "locked": g.locked_pnl.apply(lambda s: num(s).sum(min_count=1)),
                        "closed": g.action.last(),
                        "green_secs": g.seconds_to_off.apply(lambda s: num(s).min())}).reset_index()
    k = lays.sort_values("time_utc").groupby(["market_id", "selection_id"])
    first = k.first().reset_index()
    first["lays_sent"] = k.size().to_numpy()
    first["asked"] = k["size"].apply(lambda s: num(s).sum()).to_numpy()
    first = first.drop(columns=[c for c in ("lay_matched", "locked", "closed", "green_secs", "bsp", "won") if c in first])
    d = first.merge(agg, on=["market_id", "selection_id"], how="left")
    if close is not None and len(close):
        d = d.merge(close, on=["market_id", "selection_id"], how="left")
    else:
        d["bsp"], d["won"] = np.nan, np.nan
    for c in ("price", "f_bsp", "mid", "size", "seconds_to_off"):
        d[c] = num(d[c])
    d["lay_matched"] = num(d.lay_matched).fillna(0.0)
    d["is_matched"] = d.lay_matched > 0
    d["liab_matched"] = d.lay_matched * (d.price - 1)
    d["liab_asked"] = d.asked * (d.price - 1)
    d["drift_log"] = np.log(d.bsp / d.price)                    # above 0: it drifted past our lay price (good)
    d["edge_asked"] = np.log(d.f_bsp / d.price)                 # the forecast against the price laid
    d["mid_vs_sp"] = np.log(d.bsp / d.mid)
    # laid at P, closed at the SP B, a unit of liability: stake 1/(P-1); green profit = stake x (1 - P/B)
    d["theory_pnl"] = d.liab_matched / (d.price - 1) * (1 - d.price / d.bsp)
    d["theory_all"] = d.liab_asked / (d.price - 1) * (1 - d.price / d.bsp)
    d["price_band"] = band(d.price)
    d["hour"] = d.race_time.astype(str).str[:2]
    return d


def lay_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    g = d.groupby(by, observed=True)
    m = d[d.is_matched].groupby(by, observed=True)
    t = pd.DataFrame({"lays": g.size(), "matched": g.is_matched.sum(), "fill%": 100 * g.is_matched.mean(),
                      "liab_asked": g.liab_asked.sum(), "liab_matched": g.liab_matched.sum(),
                      "locked": g.locked.sum(), "drift_all": g.drift_log.mean(),
                      "drift_matched": m.drift_log.mean(), "theory_matched": m.theory_pnl.sum(),
                      "theory_if_all_filled": g.theory_all.sum()}).reset_index()
    t["locked_%liab"] = 100 * t.locked / t.liab_matched.replace(0, np.nan)
    return t


def early_lay_section(store, day: date, mode: str, close) -> tuple[list[str], dict, pd.DataFrame | None]:
    from greyhound import early_lay as el
    L = [f"## Early lay: {mode}"]
    led = _csv(store, f"{el.PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    st = store.get(f"{el.PREFIX}/{mode}/{day:%Y-%m-%d}/status.json")
    out = {"status": json.loads(st) if st else None}
    if led is None or not len(led):
        return L + ["", "no ledger found"], out, None
    out["rows"] = int(len(led))
    out["actions"] = {str(k): int(v) for k, v in led.action.value_counts().items()}
    out["first_row_utc"], out["last_row_utc"] = str(led.time_utc.min()), str(led.time_utc.max())
    t = pd.to_datetime(led.time_utc, utc=True, errors="coerce").dropna().sort_values()
    gaps = t.diff().dt.total_seconds().div(60)
    out["longest_gap_between_rows_min"] = round(float(gaps.max()), 1) if len(gaps) > 1 else None
    big = gaps[gaps > 20]
    out["gaps_over_20_min"] = [f"{t.loc[i] - pd.Timedelta(minutes=float(v)):%H:%M} to {t.loc[i]:%H:%M} UTC"
                               for i, v in big.items()][:12]
    if "error" in led:
        e = led[led.error.notna() & (led.error.astype(str) != "")]
        why = e.assign(why=e.error.astype(str).str.replace(r"[0-9.]+", "#", regex=True).str[:90])
        out["reasons"] = {f"{a}: {w}": int(n) for (a, w), n in why.groupby(["action", "why"]).size().sort_values(
            ascending=False).head(25).items()}
    L += ["", "```", json.dumps(out, indent=1, default=str), "```"]
    d = lay_frame(led, close)
    if not len(d):
        return L + ["", "no lay sent"], out, d
    s = d[d.bsp.notna()]
    tot = {"dogs_laid": int(len(d)), "races": int(d.market_id.nunique()), "lays_sent": int(d.lays_sent.sum()),
           "liability_asked": round(float(d.liab_asked.sum()), 2), "dogs_matched": int(d.is_matched.sum()),
           "fill_rate%": round(100 * float(d.is_matched.mean()), 1),
           "liability_matched": round(float(d.liab_matched.sum()), 2),
           "locked_pnl": round(float(d.locked.sum()), 2),
           "locked_%_of_matched_liability": round(100 * float(d.locked.sum() / d.liab_matched.sum()), 2)
           if d.liab_matched.sum() else None,
           "closed_how": {str(k): int(v) for k, v in d.closed.value_counts(dropna=False).items()},
           "with_an_sp": int(len(s)),
           "mean_log(SP/lay price), all lays": round(float(s.drift_log.mean()), 4) if len(s) else None,
           "mean_log(SP/lay price), matched": round(float(s[s.is_matched].drift_log.mean()), 4)
           if s.is_matched.any() else None,
           "mean_log(SP/lay price), unmatched": round(float(s[~s.is_matched].drift_log.mean()), 4)
           if (~s.is_matched).any() else None,
           "share_drifted_past_lay_price%, matched": round(100 * float((s[s.is_matched].drift_log > 0).mean()), 1)
           if s.is_matched.any() else None,
           "share_drifted_past_lay_price%, unmatched": round(100 * float((s[~s.is_matched].drift_log > 0).mean()), 1)
           if (~s.is_matched).any() else None,
           "mean_log(SP/mid when laid), all": round(float(s.mid_vs_sp.mean()), 4) if len(s) else None,
           "theory_pnl_matched_closed_at_sp": round(float(s.theory_pnl.sum()), 2),
           "theory_pnl_if_every_lay_filled": round(float(s.theory_all.sum()), 2),
           "median_seconds_to_off_at_lay": float(d.seconds_to_off.median()) if d.seconds_to_off.notna().any() else None,
           "lay_winners_matched": int(s[s.is_matched].won.sum()) if len(s) else None}
    out["totals"] = tot
    L += ["", "**Totals** (a lay matched when any of it matched by the green-up; drift = log(SP / lay price), above 0 "
          "the dog went out past our lay; theory = the lay closed at the SP, before commission)", "", "```",
          json.dumps(tot, indent=1), "```"]
    for by, name in (("price_band", "By lay price"), ("track", "By track"), ("hour", "By hour (UK)"),
                     ("trap", "By trap"), ("closed", "By how it closed")):
        if by in d:
            L += ["", f"**{name}**", ""] + _md(lay_table(d, by), 2)
    d["edge_band"] = pd.cut(d.edge_asked, [-9, 0.15, 0.25, 0.4, 0.6, 9], labels=["<0.15", "0.15-0.25", "0.25-0.4",
                                                                                "0.4-0.6", "0.6+"])
    L += ["", "**By the forecast's margin over the lay price, log(forecast SP / lay price)**", ""] + \
         _md(lay_table(d, "edge_band"), 2)
    return L, out, d


def live_vs_paper(live: pd.DataFrame | None, paper: pd.DataFrame | None) -> tuple[list[str], dict]:
    L = ["## Live against paper on the same lays"]
    if live is None or paper is None or not len(live) or not len(paper):
        return L + ["", "no data found (one of the two ledgers has no lay)"], {}
    k = ["market_id", "selection_id"]
    j = live[k + ["price", "is_matched", "liab_matched", "locked", "bsp", "drift_log", "time_utc"]].merge(
        paper[k + ["price", "is_matched", "liab_matched", "locked", "time_utc"]], on=k, how="outer",
        suffixes=("_live", "_paper"), indicator=True)
    both = j[j._merge == "both"]
    f = lambda s: bool(s) if pd.notna(s) else False               # noqa: E731
    out = {"lays_live": int(len(live)), "lays_paper": int(len(paper)), "in_both": int(len(both)),
           "live_only": int((j._merge == "left_only").sum()), "paper_only": int((j._merge == "right_only").sum()),
           "same_price_in_both%": round(100 * float((both.price_live.round(2) == both.price_paper.round(2)).mean()), 1)
           if len(both) else None,
           "in_both: matched live": int(both.is_matched_live.map(f).sum()),
           "in_both: filled on paper": int(both.is_matched_paper.map(f).sum()),
           "in_both: paper filled, live not": int((both.is_matched_paper.map(f) & ~both.is_matched_live.map(f)).sum()),
           "in_both: live matched, paper not": int((both.is_matched_live.map(f) & ~both.is_matched_paper.map(f)).sum()),
           "in_both: locked live": round(float(both.locked_live.sum()), 2),
           "in_both: locked paper": round(float(both.locked_paper.sum()), 2)}
    pf = both[both.is_matched_paper.map(f) & ~both.is_matched_live.map(f)]
    if len(pf):
        out["paper-only fills: mean log(SP/lay price)"] = round(float(pf.drift_log.mean()), 4)
        out["paper-only fills: paper locked"] = round(float(pf.locked_paper.sum()), 2)
    bm = both[both.is_matched_live.map(f)]
    if len(bm):
        out["live fills: mean log(SP/lay price)"] = round(float(bm.drift_log.mean()), 4)
    L += ["", "```", json.dumps(out, indent=1), "```"]
    return L, out


def run(store, day: date, out: Path, s3=None) -> dict:
    from betfair_recorder import _read_day_csv
    from greyhound import early_lay as el, fav_back as fb
    res = {"day": f"{day}", "read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    L = [f"# racing² - greyhound day diagnostic, {day} (read {res['read_utc']})", "",
         "Read-only, from S3. Every figure is computed from the files named in each section; where a file is missing "
         "the section says no data found."]
    preds = _csv(store, f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    try:
        close = fb.closing(day, s3=s3)
    except Exception as exc:
        close = None
        L += ["", f"The settled books could not be read: {str(exc)[:200]}"]
    markets = None
    for name in ("markets_greyhound_live.csv", "markets_greyhound.csv"):
        try:
            markets = _read_day_csv(day, name, s3=s3)
        except Exception:
            markets = None
        if markets is not None and len(markets):
            break
    if preds is not None:
        res["predictions"] = {"rows": int(len(preds)), "markets": int(preds.market_id.nunique()),
                              "columns": list(preds.columns)[:40],
                              **{c: str(preds[c].iloc[0]) for c in ("model_objective", "oos_prices", "model", "trained",
                                                                    "fitted_utc", "predicted_utc") if c in preds}}
        L += ["", "Predictions file: ", "", "```", json.dumps(res["predictions"], indent=1), "```"]
    else:
        L += ["", "Predictions file: no data found"]
    try:
        sec, res["coverage"] = coverage_section(preds, close, markets)
        L += [""] + sec
    except Exception as exc:
        L += ["", f"## 1. Coverage: failed ({str(exc)[:200]})"]
    try:
        if preds is not None and close is not None:
            sec, res["price"], _ = price_section(preds, close)
            L += [""] + sec
    except Exception as exc:
        L += ["", f"## 2. Price: failed ({str(exc)[:200]})"]
    frames = {}
    modes = ["live", "paper"] + [f"paper-{n}" for n in getattr(el, "PAPER_VARIANTS", {})]
    for mode in modes:
        try:
            sec, res[f"early_lay_{mode}"], frames[mode] = early_lay_section(store, day, mode, close)
            if mode in ("live", "paper") or frames[mode] is not None:
                L += [""] + (sec if mode in ("live", "paper") else sec[:1] + ["", "```", json.dumps(
                    res[f"early_lay_{mode}"].get("totals", {}), indent=1), "```"])
        except Exception as exc:
            L += ["", f"## Early lay {mode}: failed ({str(exc)[:200]})"]
    try:
        sec, res["live_vs_paper"] = live_vs_paper(frames.get("live"), frames.get("paper"))
        L += [""] + sec
    except Exception as exc:
        L += ["", f"## Live against paper: failed ({str(exc)[:200]})"]
    rec = store.get(f"{el.PREFIX}/live/{day:%Y-%m-%d}/reconcile.json")
    L += ["", "## Betfair's own settled bets against the ledger", "",
          "```\n" + rec.decode()[:3000] + "\n```" if rec else "no data found (the reconciliation runs the next morning)"]
    try:
        fl = _csv(store, f"{fb.PREFIX}/{fb.MODE}/{day:%Y-%m-%d}/ledger.csv")
        if fl is not None and close is not None:
            s = fb.settle_frame(fl, close)
            res["fav_back_paper"] = fb.summary(s)
            L += ["", "## The favourite back on paper", "", "```", json.dumps(res["fav_back_paper"], indent=1), "```"]
        else:
            L += ["", "## The favourite back on paper", "", "no data found"]
    except Exception as exc:
        L += ["", f"## The favourite back: failed ({str(exc)[:200]})"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    for name, d in frames.items():
        if d is not None and len(d):
            d.to_csv(out.parent / f"{out.name}_lays_{name}.csv", index=False)
    return res


def main(argv=None) -> int:
    from zoneinfo import ZoneInfo
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", help="YYYY-MM-DD (default today, UK)")
    ap.add_argument("--out", default="out/greyhound_day")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else datetime.now(ZoneInfo("Europe/London")).date()
    res = run(Store(), day, Path(a.out))
    print(json.dumps({k: res.get(k) for k in ("day", "coverage", "price")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
