"""The innings-break probe: the chase model against Betfair at the one in-play point BASIC data can place without
ball timestamps.

    python -m cricket.model.innings_break --root data/cricket/store --work data/cricket/eval

Betfair's BASIC files hold every market of a cricket event and each market's status changes. The first-innings
markets (1ST_INNINGS_RUNS, 1_INNINGS_RUNS_LINE, 1_INNING_20_OVR_LINE) suspend when the first innings ends; the
chase's 2_INNING_6_OVR_LINE suspends after its sixth over. The rules, fixed before any result was seen (4 Nov 2026):

- **the break** is the earliest in-play suspension of a first-innings market;
- **the market's price** is each Match Odds runner's last traded price standing 5 minutes after the break, which
  must have traded inside the 10 minutes before that point, overround removed. It is the last traded price: BASIC
  has no ladder;
- **the model's price** is the chase from 0 for 0 against the real target, using only what is known at the break:
  the chasing XI's first two listed players open, the bowling is the side's experience-weighted average, the model
  is the walk-forward one fitted before 1 January 2026 (cricket/model/prematch.py), raw (the Benter regression is
  its calibration);
- **checks that the break was found**: minutes from the break to the chase's 6-over suspension (a T20 break plus six
  overs is roughly 35-60), and the market's move across the break against the first-innings total less the model's
  pre-match median (a market that has seen the innings moves with it).
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricket.model import simulator as sim
from cricket.model.benter import logistic, market_table, match_markets, same_side
from cricket.model.calibration import logit
from cricket.model.data import load
from cricket.model.features import build
from cricket.model.metrics import log_loss, paired_bootstrap
from cricket.model.outcomes import extras_runs_distribution
from cricket.model.prematch import start_states
from sources.common import Store

FIRST_INNINGS = ("1ST_INNINGS_RUNS", "1_INNINGS_RUNS_LINE", "1_INNING_20_OVR_LINE")
CHASE_6 = "2_INNING_6_OVR_LINE"
AFTER_MIN = 5.0
FRESH_MIN = 10.0
RAW = "betfair_historic/raw/cricket/"


def _t(ms) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)


def read_event(files: list[bytes]) -> dict:
    """Every market of an event: its type, its in-play suspensions, and Match Odds' last traded prices in time."""
    mk: dict[str, dict] = {}
    for raw in files:
        text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
        for line in text.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            pt = _t(msg.get("pt", 0))
            for mc in msg.get("mc") or []:
                s = mk.setdefault(mc["id"], {"type": None, "suspended_inplay": [], "ltp": []})
                d = mc.get("marketDefinition")
                if d:
                    s["type"] = d.get("marketType")
                    if d.get("inPlay") and d.get("status") == "SUSPENDED":
                        s["suspended_inplay"].append(pt)
                for rc in mc.get("rc") or []:
                    if "ltp" in rc:
                        s["ltp"].append((pt, int(rc["id"]), float(rc["ltp"])))
    return mk


def break_price(mk: dict, mo_id: str) -> dict:
    first = [min(s["suspended_inplay"]) for s in mk.values() if s["type"] in FIRST_INNINGS and s["suspended_inplay"]]
    if not first:
        return {"why": "no first-innings market with an in-play suspension"}
    brk = min(first)
    six = [min(s["suspended_inplay"]) for s in mk.values() if s["type"] == CHASE_6 and s["suspended_inplay"]]
    at = brk + dt.timedelta(minutes=AFTER_MIN)
    mo = mk.get(mo_id)
    if mo is None:
        return {"why": "Match Odds market not in the event's files", "break": brk}
    out = {"break": brk, "price_at": at,
           "to_chase_6_min": (min(six) - brk).total_seconds() / 60 if six and min(six) > brk else None}
    for sel in {s for _, s, _ in mo["ltp"]}:
        before = [(t, p) for t, s, p in mo["ltp"] if s == sel and t <= at]
        if not before or (at - before[-1][0]).total_seconds() / 60 > FRESH_MIN:
            out["why"] = "no Match Odds trade inside 10 minutes of the price time"
            return out
        out[sel] = before[-1][1]
    return out


def chase_batch(pre_batch: sim.Batch, targets: np.ndarray) -> sim.Batch:
    """The chase at 0 for 0 from a first-innings start batch: its chasing openers, queue and bowling (pre-chase
    information only)."""
    S = len(pre_batch)
    return sim.Batch(innings=np.full(S, 2), runs=np.zeros(S, int), wickets=np.zeros(S, int), legal=np.zeros(S, int),
                     max_balls=pre_batch.max_balls2, target=targets.astype(float), bsw=np.zeros(S, int),
                     lag_r=np.zeros((S, 6)), lag_w=np.zeros((S, 6)), striker=pre_batch.open2[:, 0].copy(),
                     non=pre_batch.open2[:, 1].copy(), bowler=pre_batch.team_bowl2.copy(), queue=pre_batch.queue2,
                     team_bowl=pre_batch.team_bowl2, ctx=pre_batch.ctx)


def run(store: Store, work: Path, paths: int, log=print) -> dict:
    t = load(store)
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    betfair = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    pre = store.get_parquet("cricket/eval/prematch.parquet")
    j, counts = match_markets(market_table(betfair), pre)
    j = j[j.year == 2026]
    ev = betfair[betfair.market_id.isin(j.market_id)][["market_id", "event_id"]].drop_duplicates()
    j = j.merge(ev, on="market_id")
    runners = betfair[betfair.market_id.isin(j.market_id)][["market_id", "selection_id", "runner_name"]]
    pre_i = pre.set_index("match_id")
    raw_keys = list(store.listing(RAW))
    rows, why = [], {}
    for r in j.itertuples():
        files = [store.get(k) for k in raw_keys if f"/{r.event_id}/" in k]
        bp = break_price(read_event(files), r.market_id)
        if "why" in bp:
            why[bp["why"]] = why.get(bp["why"], 0) + 1
            continue
        rr = runners[runners.market_id == r.market_id]
        bat = pre_i.loc[r.match_id, "bat_first"]
        sel_bat = [s for s, n in zip(rr.selection_id, rr.runner_name) if same_side(n, bat)]
        sel_chase = [s for s in rr.selection_id if s not in sel_bat]
        if len(sel_bat) != 1 or len(sel_chase) != 1:
            why["runner mapping"] = why.get("runner mapping", 0) + 1
            continue
        q_chase, q_bat = 1 / bp[sel_chase[0]], 1 / bp[sel_bat[0]]
        rows.append({"match_id": r.match_id, "market_id": r.market_id, "event_name": r.event_name,
                     "break_utc": bp["break"].isoformat(), "to_chase_6_min": bp["to_chase_6_min"],
                     "p_market_chase": q_chase / (q_chase + q_bat), "p_close_chase": 1 - r.p_market,
                     "chase_won": 1 - int(r.bat_won), "p_pre_model_chase": 1 - r.p_model})
    b = pd.DataFrame(rows)
    log(f"{len(j)} matched 2026 markets; break priced for {len(b)}; not priced: {why}")
    # the model at the break, from the 2026 walk-forward model
    f, br, bw, prior = build(t, since="2012-01-01", pop_before="2026-01-01")
    booster = xgb.Booster()
    booster.load_model(work / "prematch_model_2026.ubj")
    ex_v, ex_p = extras_runs_distribution(f[pd.to_datetime(f.start_date) < "2026-01-01"])
    s = start_states(f, 2026)
    s = s[s.match_id.isin(b.match_id)].set_index("match_id").loc[b.match_id].reset_index()
    ids = set(s.match_id)
    pb = sim.prepare(s, t["deliveries"][t["deliveries"].match_id.isin(ids)],
                     t["players"][t["players"].match_id.isin(ids)], br, bw, prior)
    totals = t["innings"][t["innings"].innings == 1].set_index("match_id")["runs"]
    b["first_total"] = b.match_id.map(totals).to_numpy()
    b["p_model_chase"] = sim.win_probability(booster, chase_batch(pb, b.first_total.to_numpy() + 1), paths,
                                             ex_v, ex_p, seed=2026)
    b["pre_q50"] = b.match_id.map(pre_i["total_q50"]).to_numpy()
    return {"rows": b, "counts": counts, "not_priced": why, "matched_2026": int(len(j))}


def analyse(b: pd.DataFrame) -> dict:
    y = b.chase_won.to_numpy(dtype=float)
    X = np.c_[logit(b.p_model_chase), logit(b.p_market_chase)]
    coef, se = logistic(X, y)
    move = logit(b.p_market_chase) - logit(b.p_close_chase)
    surprise = b.first_total - b.pre_q50
    gap = b.to_chase_6_min.dropna()
    return {"n": int(len(b)),
            "market": log_loss(y, b.p_market_chase), "model": log_loss(y, b.p_model_chase),
            "close": log_loss(y, b.p_close_chase),
            "model_minus_market": paired_bootstrap(y, b.p_model_chase, b.p_market_chase, b.match_id),
            "w_model": coef[1], "se_model": se[1], "w_market": coef[2], "se_market": se[2],
            "corr_model_market": float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]),
            "corr_move_surprise": float(np.corrcoef(move, surprise)[0, 1]),
            "gap_median": float(gap.median()) if len(gap) else None,
            "gap_range": (float(gap.min()), float(gap.max())) if len(gap) else None, "gap_n": int(len(gap))}


def report_md(res: dict, a: dict | None, source: str, sens: dict | None = None) -> str:
    L = ["# Cricket - innings-break probe: the chase model against Betfair's price at the break", "",
         f"Source: the owner's Betfair BASIC bundle (`{source}`), Cricsheet, the walk-forward 2026 model. Built by "
         "`python -m cricket.model.innings_break`. Rules fixed before the results: see the module.", "",
         f"Matched 2026 markets: {res['matched_2026']}; break priced: {a['n'] if a else 0}; not priced: "
         + ("; ".join(f"{k} {v}" for k, v in res["not_priced"].items()) or "none") + ".", ""]
    if not a:
        return "\n".join(L + ["No data found: no break could be priced.", ""])
    b = a["model_minus_market"]
    L += ["## Was the break found?", "",
          f"- Minutes from the break to the chase's 6-over suspension: median {a['gap_median']:.0f}, range "
          f"{a['gap_range'][0]:.0f}-{a['gap_range'][1]:.0f} ({a['gap_n']} events with that market).",
          f"- The market's move across the break (logit, close to break) against the first-innings total less the "
          f"model's pre-match median: correlation {a['corr_move_surprise']:+.2f} (a market that has seen the innings "
          "moves against the chasing side as the total beats expectation: negative).",
          f"- Model and market at the break, correlation of the logits: {a['corr_model_market']:.2f}.", "",
          "## Chase won: probabilities at the break", "",
          "| probability | log loss |", "|---|---:|",
          f"| Betfair at the break (last traded, overround removed) | {a['market']:.4f} |",
          f"| model, chase from 0 for 0 against the real target | {a['model']:.4f} |",
          f"| for reference: Betfair's pre-match close | {a['close']:.4f} |", "",
          f"Model minus market: {b['diff']:+.4f} (95% interval {b['lo']:+.4f} to {b['hi']:+.4f}, {b['groups']} "
          "matches).", "",
          "| Benter term | weight | s.e. | z |", "|---|---:|---:|---:|",
          f"| logit(model p) | {a['w_model']:.3f} | {a['se_model']:.3f} | {a['w_model'] / a['se_model']:.2f} |",
          f"| logit(market p) | {a['w_market']:.3f} | {a['se_market']:.3f} | {a['w_market'] / a['se_market']:.2f} |",
          "", f"The model adds to the price at the break (z above 1.96): "
          f"**{'yes' if a['w_model'] / a['se_model'] > 1.96 else 'no'}**.", ""]
    if sens:
        sb = sens["model_minus_market"]
        L += ["## Sensitivity (not the pre-registered result): breaks 25-75 minutes before the chase's 6-over "
              "suspension", "",
              f"{sens['n']} events. Betfair at the break {sens['market']:.4f}, model {sens['model']:.4f}; model minus "
              f"market {sb['diff']:+.4f} (95% interval {sb['lo']:+.4f} to {sb['hi']:+.4f}). Benter weight on the "
              f"model {sens['w_model']:.3f} (s.e. {sens['se_model']:.3f}).", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--work", default="data/cricket/eval")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--out", default="reports/cricket")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    res = run(store, Path(a.work), a.paths, log=lambda m: print(m, flush=True))
    an = analyse(res["rows"]) if len(res["rows"]) >= 20 else None
    g = res["rows"][res["rows"].to_chase_6_min.between(25, 75)]
    sens = analyse(g) if len(g) >= 20 else None
    md = report_md(res, an, store.where(RAW), sens)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "innings_break_probe.md").write_text(md)
    store.put_parquet("cricket/eval/innings_break_probe.parquet", res["rows"])
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
