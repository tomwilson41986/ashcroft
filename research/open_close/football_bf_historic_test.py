"""The football model tested on Betfair's own prices: the owner's Historic Data soccer bundle (Aug-Sep 2026).

racing² football research, 4 Nov 2026 (the owner: "just use this data to test our model on"). Betfair's BASIC
stream files carry the last traded price (LTP) each minute of every market. For each match-odds market this reads
the LTP standing at our collection moment (Friday 17:00 UK for Friday-Monday matches, Tuesday 13:00 UK for
Tuesday-Thursday: when the model's features are fixed) and the last LTP before the market turned in play (the close),
links the market to our match (``football.data.betfair_map``) and scores the model's forecast of the move between
them:

- **the frozen model** (``models/football/stack_v1``, the pre-registered stack on team features, fitted on 2013-2024,
  hashes checked): its pre-registered threshold rule;
- **walk-forward stacks** fitted only on matches before ``CUTOFF`` (1 Aug 2026) for the candidate's features and with
  short form added: every trade with positive expected value after costs (``football_improve.by_ev``, the forecast
  calibrated on the season before), and the threshold rule.

Costs: the entry and the close are traded prices, not quotes, so each end is taken a tick worse than the LTP (back a
tick under, lay a tick over), and 2% commission on a market's net win; at the LTP itself (no spread) as the upper
bound. The chance at each moment is the LTPs' implied chances normalised to one.

Caveat: these matches were in the 2025/26-to-date development reads (on football-data's two Betfair snapshots, not
these prices); the frozen model's fit ends in 2024, so for it the period is out of sample.

    python -m research.open_close.football_bf_historic_test --tar soccer.tar --cache <lab cache> \\
        --root <sources root with football/gold> --out reports/football_runs/bf_historic_test.json
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CUTOFF = pd.Timestamp("2026-08-01")
COLL_HOUR = {"fri": 17, "tue": 13}          # UK clock (football-data's collection, matches.php)
FROZEN = "models/football/stack_v1"
MAX_AGE_H = 6.0                              # a last trade older than this at the collection moment is too stale
COMMISSION = 0.02


def collection_instant(market_time_utc: pd.Timestamp) -> pd.Timestamp:
    """The collection moment for a market starting at ``market_time_utc``: its collection day (Friday for Friday to
    Monday matches, Tuesday for Tuesday to Thursday) at 17:00 / 13:00 UK, in UTC."""
    from football.team_features import collection_day
    uk = market_time_utc.tz_convert("Europe/London")
    day = collection_day([uk.tz_localize(None).normalize()]).iloc[0]
    hour = COLL_HOUR["tue"] if day.dayofweek == 1 else COLL_HOUR["fri"]
    return (day + pd.Timedelta(hours=hour)).tz_localize("Europe/London").tz_convert("UTC")


def parse_match_odds(raw: bytes) -> list[dict]:
    """One stream file -> one row a runner of each MATCH_ODDS market in it: the definition, the LTP standing at the
    collection moment and its age, the first LTP after it, the first and the last LTP before the market went in play."""
    text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
    if b'"MATCH_ODDS"' not in text:
        return []
    mk: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        pt = pd.Timestamp(msg.get("pt", 0), unit="ms", tz="UTC")
        for mc in msg.get("mc") or []:
            mid = mc.get("id")
            st = mk.setdefault(mid, {"defn": None, "coll": None, "inplay": False, "first": {}, "close": {},
                                     "at": {}, "at_t": {}, "after": {}})
            d = mc.get("marketDefinition")
            if d:
                st["defn"] = d
                if st["coll"] is None and d.get("marketTime"):
                    st["coll"] = collection_instant(pd.Timestamp(d["marketTime"]).tz_convert("UTC")
                                                    if pd.Timestamp(d["marketTime"]).tzinfo
                                                    else pd.Timestamp(d["marketTime"], tz="UTC"))
                if d.get("inPlay"):
                    st["inplay"] = True
            if st["defn"] is None or st["defn"].get("marketType") != "MATCH_ODDS" or st["inplay"]:
                continue
            for rc in mc.get("rc") or []:
                if "ltp" not in rc:
                    continue
                sid, p = int(rc["id"]), float(rc["ltp"])
                st["first"].setdefault(sid, p)
                st["close"][sid] = p
                if st["coll"] is not None:
                    if pt <= st["coll"]:
                        st["at"][sid], st["at_t"][sid] = p, pt
                    else:
                        st["after"].setdefault(sid, p)
    rows = []
    for mid, st in mk.items():
        d = st["defn"]
        if not d or d.get("marketType") != "MATCH_ODDS":
            continue
        for r in d.get("runners") or []:
            sid = int(r["id"])
            age = ((st["coll"] - st["at_t"][sid]).total_seconds() / 3600) if sid in st["at_t"] else np.nan
            rows.append({"market_id": mid, "event_id": d.get("eventId"), "event_name": d.get("eventName"),
                         "country": d.get("countryCode"), "market_time": d.get("marketTime"),
                         "market_type": "MATCH_ODDS", "selection_id": sid, "runner_name": r.get("name"),
                         "handicap": r.get("hc"), "won": {"WINNER": 1, "LOSER": 0}.get(r.get("status")),
                         "coll_utc": str(st["coll"]), "ltp_coll": st["at"].get(sid), "coll_age_h": age,
                         "ltp_after_coll": st["after"].get(sid), "ltp_first": st["first"].get(sid),
                         "ltp_close": st["close"].get(sid), "ltp_t_60": None, "ltp_t_1": None, "bsp": None})
    return rows


def _safe(raw: bytes) -> list[dict] | None:
    try:
        return parse_match_odds(raw)
    except Exception:
        return None


def extract(tar_path: str, workers: int = 8, batch: int = 4000) -> tuple[pd.DataFrame, dict]:
    """Every match-odds market in the bundle (read straight from the tar, in batches across processes)."""
    rows, n, bad = [], 0, 0
    with tarfile.open(tar_path) as tf, ProcessPoolExecutor(workers) as pool:
        buf = []

        def flush():
            nonlocal bad
            for got in pool.map(_safe, buf, chunksize=64):
                if got is None:
                    bad += 1
                else:
                    rows.extend(got)
            buf.clear()
        for m in tf:
            if not m.isfile():
                continue
            buf.append(tf.extractfile(m).read())
            n += 1
            if len(buf) >= batch:
                flush()
                log.info("%d files read, %d runner rows", n, len(rows))
        flush()
    df = pd.DataFrame(rows)
    if len(df):          # a market can sit in its own file and its event's: keep the reading with the most prices
        df["_n"] = df[["ltp_coll", "ltp_after_coll", "ltp_close"]].notna().sum(axis=1)
        df = df.sort_values("_n").drop_duplicates(["market_id", "selection_id"], keep="last").drop(columns="_n")
    return df, {"files": n, "unreadable": bad}


def panel(mo: pd.DataFrame, m: pd.DataFrame, aliases) -> tuple[pd.DataFrame, dict]:
    """The match-odds rows linked to our matches: one row a match with the entry and closing LTP of home, draw and
    away, the entry being the LTP standing at the collection moment (traded within ``MAX_AGE_H`` hours before it),
    else the first traded after it."""
    from football.data import betfair_map
    g = m[["match_id", "country", "match_date", "kickoff_utc", "home_team", "away_team", "home_team_id",
           "away_team_id"]].copy()
    g["match_date"] = pd.to_datetime(g["match_date"]).dt.strftime("%Y-%m-%d")
    mp, _al, review = betfair_map(g, mo, aliases)
    extra = mo[["market_id", "selection_id", "ltp_coll", "coll_age_h", "ltp_after_coll", "coll_utc"]]
    mp = mp.merge(extra, on=["market_id", "selection_id"], how="left")
    mp["entry"] = np.where(mp["coll_age_h"] <= MAX_AGE_H, mp["ltp_coll"], mp["ltp_after_coll"])
    mp = mp[mp["outcome"].isin(["H", "D", "A"])]
    w = mp.pivot_table(index="match_id", columns="outcome", values=["entry", "ltp_close"], aggfunc="first")
    w.columns = [f"{a}_{b.lower()}" for a, b in w.columns]
    w = w.reset_index()
    stale = mp.groupby("match_id")["coll_age_h"].max()
    w["entry_from_after"] = w["match_id"].map(stale.gt(MAX_AGE_H) | stale.isna())
    info = {"events": int(mo["event_id"].nunique()), "events_linked": int(mp["betfair_event_id"].nunique()),
            "events_for_review": int(len(review)), "matches": int(len(w))}
    return w, info


def _move_cols(p: pd.DataFrame) -> pd.DataFrame:
    """Normalised implied chances at entry and close, and the realised log chance moves home and away."""
    p = p.copy()
    for t, pre in (("entry", "qe"), ("ltp_close", "qc")):
        inv = 1 / p[[f"{t}_h", f"{t}_d", f"{t}_a"]]
        s = inv.sum(axis=1)
        for o in "hda":
            p[f"{pre}_{o}"] = (1 / p[f"{t}_{o}"]) / s
    p["yb_h"], p["yb_a"] = np.log(p["qc_h"] / p["qe_h"]), np.log(p["qc_a"] / p["qe_a"])
    return p


def trades(p: pd.DataFrame, pred_h, pred_a, side_h, side_a, cost: str) -> pd.DataFrame:
    """One row a trade: back first (side +1) or lay first (-1) at the entry LTP, closed at the closing LTP, a tick
    worse at each end (``cost`` 'tick') or at the LTP (``cost`` 'ltp'); profit on a GBP1 back stake (the lay first
    trade's liability is price - 1)."""
    from research.open_close.backtest import tick_over, tick_under
    out = []
    for leg, pred, side in (("h", pred_h, side_h), ("a", pred_a, side_a)):
        o, c = p[f"entry_{leg}"].to_numpy(float), p[f"ltp_close_{leg}"].to_numpy(float)
        side = np.asarray(side)
        if cost == "tick":
            ob = np.array([tick_under(x) for x in o])
            cl = np.array([tick_over(x) for x in c])
            ol = np.array([tick_over(x) for x in o])
            cb = np.array([tick_under(x) for x in c])
        else:
            ob, cl, ol, cb = o, c, o, c
        prof = np.where(side > 0, ob / cl - 1, 1 - ol / cb)
        keep = side != 0
        out.append(pd.DataFrame({"match_id": p["match_id"].to_numpy()[keep], "leg": leg, "side": side[keep],
                                 "pred": np.asarray(pred)[keep], "profit": prof[keep],
                                 "right_way": (np.sign(p[f"yb_{leg}"].to_numpy()[keep]) == side[keep]),
                                 "league": p["fd_code"].to_numpy()[keep], "week": p["week"].to_numpy()[keep]}))
    return pd.concat(out, ignore_index=True)


def summary(t: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bets, return on GBP1 after 2% commission on each market's net win, its 90% bootstrap interval by match."""
    if t.empty:
        return {"bets": 0}
    net = t.groupby("match_id")["profit"].sum()
    n_bets = t.groupby("match_id").size()
    pnl = np.where(net > 0, net * (1 - COMMISSION), net)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (n_boot, len(pnl)))
    roi_b = pnl[idx].sum(1) / n_bets.to_numpy()[idx].sum(1)
    return {"bets": int(len(t)), "matches": int(len(net)), "roi": float(pnl.sum() / n_bets.sum()),
            "roi_lo90": float(np.quantile(roi_b, 0.05)), "roi_hi90": float(np.quantile(roi_b, 0.95)),
            "right_way": float(t["right_way"].mean()), "weeks": int(t["week"].nunique())}


def ev_sides(pred, cal: dict, entry, cost: str):
    """``football_improve.by_ev`` on LTPs: back first when exp(E[y]) - 1 after commission beats the round trip's cost
    at the entry price with no move, lay first in the mirror case; 0 for no trade."""
    from research.open_close.backtest import tick_over, tick_under
    ey = cal["a"] + cal["b"] * np.asarray(pred, float)
    e = np.asarray(entry, float)
    if cost == "tick":
        cb = 1 - np.array([tick_under(x) for x in e]) / np.array([tick_over(x) for x in e])
        cl = np.array([tick_over(x) for x in e]) / np.array([tick_under(x) for x in e]) - 1
    else:
        cb = cl = np.zeros(len(e))
    net = lambda g: np.where(g > 0, g * (1 - COMMISSION), g)          # noqa: E731
    evb, evl = net(np.expm1(ey)) - cb, net(np.expm1(-ey)) - cl
    return np.where((evb > 0) & (evb >= evl), 1, np.where(evl > 0, -1, 0)), np.maximum(evb, evl)


def run(tar: str, cache: str, root: str, n_boot: int = 2000, mo_path: str | None = None) -> dict:
    from football import lab
    from football.reference import load_aliases
    from research.open_close import football_improve as fi
    from research.open_close import football_stack_trial as st
    from sources.common import Store
    out = {"note": __doc__.split("\n\n")[0], "cutoff": str(CUTOFF.date()), "max_age_h": MAX_AGE_H}
    if mo_path and Path(mo_path).exists():
        mo = pd.read_parquet(mo_path)
    else:
        mo, out["extract"] = extract(tar)
        if mo_path:
            mo.to_parquet(mo_path, index=False)
    out["match_odds_rows"] = int(len(mo))
    m = lab.latest(cache)
    aliases = load_aliases(extra=Store(root=root).get_parquet("football/gold/team_aliases.parquet"))
    p, out["link"] = panel(mo, m, aliases)
    p = _move_cols(p.dropna(subset=[f"{t}_{o}" for t in ("entry", "ltp_close") for o in "hda"]))
    out["link"]["matches_with_prices"] = int(len(p))
    results = {}
    # the frozen, pre-registered model (its file hashes checked against the pre-registration first)
    from research.open_close.football_exchange import read_prereg
    d0, _ = st._matrix(cache)
    dz = d0[d0["match_id"].isin(p["match_id"])].drop_duplicates("match_id")
    pr = p.merge(dz[["match_id", "fd_code", "date"]], on="match_id")
    pr["week"] = pd.to_datetime(pr["date"]).dt.to_period("W-THU").astype(str)
    out["link"]["matches_scored"] = int(len(pr))
    out["link"]["dates"] = [str(pd.to_datetime(pr["date"]).min().date()), str(pd.to_datetime(pr["date"]).max().date())]
    out["link"]["leagues"] = pr["fd_code"].astype(str).value_counts().to_dict()
    rule, _digest = read_prereg(st.PREREG)
    st.verify_models(FROZEN, rule["model_sha256"])
    fz = pr[["match_id"]].merge(st.predict(FROZEN, dz), on="match_id", how="left")
    th = float(rule.get("threshold", 0.04))
    ph, pa = fz["chance_h"].to_numpy(float), fz["chance_a"].to_numpy(float)
    fr = {"threshold": th, "move_fit": _fit(pr, ph, pa)}
    for cost in ("tick", "ltp"):
        t = trades(pr, ph, pa, np.where(ph > th, 1, np.where(ph < -th, -1, 0)),
                   np.where(pa > th, 1, np.where(pa < -th, -1, 0)), cost)
        fr[cost] = {"all": summary(t, n_boot), "outside_top5": summary(t[~t["league"].isin(fi.TOP5)], n_boot)}
    results["frozen_stack_v1"] = fr
    # walk-forward stacks fitted before the cutoff
    for arm, spec in (("both", fi.EV_ARMS["both"]), ("form_short", fi.EV_ARMS["form_short"])):
        d, cols = fi.load(cache, spec)
        tr_h = d[d["same_book"] & (d["season"] >= 2013) & (pd.to_datetime(d["date"]) < CUTOFF) & d["y_h"].notna()]
        tr_a = d[d["same_book"] & (d["season"] >= 2013) & (pd.to_datetime(d["date"]) < CUTOFF) & d["y_a"].notna()]
        te = d.set_index("match_id").loc[pr["match_id"]].reset_index()
        ph = lab.MODELS["stack"](tr_h, te, cols, tr_h["y_h"].to_numpy(float))
        pa = lab.MODELS["stack"](tr_a, te, cols, tr_a["y_a"].to_numpy(float))
        cal = fi.calibration(d, cols, 2025)                                  # the 2024 season's out-of-sample fit
        r = {"calibration": cal, "move_fit": _fit(pr, ph, pa)}
        for cost in ("tick", "ltp"):
            sh, evh = ev_sides(ph, cal, pr["entry_h"], cost)
            sa, eva = ev_sides(pa, cal, pr["entry_a"], cost)
            t = trades(pr, ph, pa, sh, sa, cost)
            r[f"ev_{cost}"] = {"all": summary(t, n_boot), "outside_top5": summary(t[~t["league"].isin(fi.TOP5)], n_boot)}
            for th in (0.03, 0.04, 0.05):
                t = trades(pr, ph, pa, np.where(ph > th, 1, np.where(ph < -th, -1, 0)),
                           np.where(pa > th, 1, np.where(pa < -th, -1, 0)), cost)
                r[f"threshold_{th}_{cost}"] = {"all": summary(t, n_boot),
                                               "outside_top5": summary(t[~t["league"].isin(fi.TOP5)], n_boot)}
        results[arm] = r
        log.info("%s: %s", arm, {k: v for k, v in r.items() if k.startswith("ev_")})
    out["results"] = results
    return out


def _fit(p: pd.DataFrame, ph, pa) -> dict:
    """How the forecast lines up with Betfair's realised move: correlation, slope, sign agreement of the top tenth."""
    y = np.r_[p["yb_h"].to_numpy(float), p["yb_a"].to_numpy(float)]
    f = np.r_[np.asarray(ph, float), np.asarray(pa, float)]
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    top = np.abs(f) >= np.quantile(np.abs(f), 0.9)
    return {"rows": int(len(y)), "corr": float(np.corrcoef(f, y)[0, 1]), "slope": float(np.polyfit(f, y, 1)[0]),
            "top_decile_right_way": float(np.mean(np.sign(f[top]) == np.sign(y[top]))),
            "mean_abs_move": float(np.mean(np.abs(y)))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--match-odds", default=None, help="parquet of the extracted match-odds rows (read or written)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(a.tar, a.cache, a.root, a.boot, a.match_odds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
