"""The ante-post resting-order paper test (ledger antepost-paper-1123). Paper only: no order call, nothing to Betfair.

The owner, 23 Nov 2026: "set up the paper test of resting orders". Every rule that crossed the spread lost
(reports/antepost_data.md s.4.8: the trade-out -22.5%, its round trip about 14-21% of the price). This test rests the
orders inside the spread instead, on the daily ante-post books the recorder keeps (betfair_recorder.py --antepost,
one snapshot a day at about 21:30-22:00 UTC, s3://<capture bucket>/betfair_live/<day>/books_antepost.csv.gz).

It is a replay of the recorded snapshots, day by day, and decides on each day from that day's snapshot and the
earlier ones only, so a replay at any later date gives what a daily paper trader would have done. Only orders placed
on snapshots from FORWARD_START count; the days before it are development, reported apart.

Each snapshot day, for GB/IE ante-post horses in an open market, active, priced (a last traded price under 1.01 is
Betfair's marked-out marker, never a price), with a best back and a real best lay (under 1000):

  entry   between 60 and 14 days out, P(ran) >= P_MIN from the frozen turn-up model (SNAP_FEATURES, trained on the
          Historic Data panel to the day before FORWARD_START): rest a GBP1 back one tick under the best lay (at
          the best lay when the spread is one tick). It fills when a later snapshot shows the horse traded above
          that price, its last traded price having changed and the market's matched money having risen since the order
          was placed (a layer had to take our price first; the delayed key gives no runner volume).
          Unfilled after ORDER_DAYS, or once 14 days out, it is cancelled.
  exit    once filled, rest a lay at the entry price * (1 - X), rounded to the tick: it fills when a later snapshot
          shows the horse traded below it, with a new trade since the back filled (as for the entry). At 3 days out or less, an
          open position is closed by taking the best lay (crossing the spread once). A horse marked out, removed or
          in a market no longer open first is stuck and settles: laid at the SP if it ran, the stake lost if not,
          0 if Betfair removed it (void).
  return  an equal green of entry / exit - 1 per GBP1, 2% off a win (the ante-post and race-day markets netted).

    python -m antepost.paper --panel turnup_panel.parquet --books recorded_books.parquet \\
        --catalogue recorded_catalogue.parquet --db horse_racing.db --out out/antepost_paper
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from antepost import ev
from antepost import phase0 as p0
from antepost import turnup as tu

log = logging.getLogger(__name__)
FORWARD_START = "2026-11-24"     # snapshot days from here count; before it, development
READ_ON = "2027-07-01"           # the registered reading: once, on or after this day (antepost-paper-1123)
READ_AGAIN = "2027-12-31"        # once more, only if fewer than MIN_TRADES had closed by READ_ON
P_MIN, X = 0.6, 0.30             # the turn-up bar and the exit target: the 2025 choice in antepost-tradeout-1123
ENTRY_MAX, ENTRY_MIN, EXIT_AT = 60.0, 14.0, 3.0      # days out
ORDER_DAYS = 7
MIN_TRADES = 100
N_TRIALS = 409                   # the ledger's 375 horse_racing trials with this registration, plus the 34 grid
                                 # rules of the 1119 and 1123 EV and trade-out runs
SNAP_FEATURES = ["days_out", "log_ltp", "has_price", "share", "price_rank", "n_listed", "n_priced", "handicap", "jumps",
                 "festival", "ireland", "month", "days_since_run", "runs_365", "ran_14", "marked_out"]
TICKS = [(2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5), (30.0, 1.0), (50.0, 2.0),
         (100.0, 5.0), (1000.0, 10.0)]


# ----------------------------------------------------------------------------------------------------------------
# Betfair's price ladder
# ----------------------------------------------------------------------------------------------------------------

def tick_down(p: float) -> float:
    """One tick below p on Betfair's ladder (never under 1.01)."""
    for top, step in TICKS:
        if p <= top + 1e-9:
            return round(max(p - step, 1.01), 2)
    return 990.0


def to_tick(p: float) -> float:
    """The nearest price on Betfair's ladder at or below p (at least 1.01)."""
    p = max(min(p, 1000.0), 1.01)
    lo = 1.0
    for top, step in TICKS:
        if p <= top:
            return round(lo + np.floor((p - lo) / step + 1e-9) * step, 2)
        lo = top
    return 1000.0


# ----------------------------------------------------------------------------------------------------------------
# The frozen turn-up model and the snapshot rows
# ----------------------------------------------------------------------------------------------------------------

def train_frozen(panel: pd.DataFrame):
    """The turn-up LightGBM on SNAP_FEATURES (the ones a daily snapshot can give), on every panel race before
    FORWARD_START; rounds from early stopping on the last 15% of races by date, then refit on all of them."""
    import lightgbm as lgb
    d = panel[pd.to_datetime(panel["race_date"]) < FORWARD_START].copy()
    dates = d["race_date"].drop_duplicates().sort_values()
    cut = dates.iloc[int(len(dates) * 0.85)]
    fit, stop = d[d["race_date"] < cut], d[d["race_date"] >= cut]
    dtr = lgb.Dataset(fit[SNAP_FEATURES], tu.y_index(fit["y"]))
    dva = lgb.Dataset(stop[SNAP_FEATURES], tu.y_index(stop["y"]), reference=dtr)
    b = lgb.train(tu.LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = int(b.best_iteration or b.current_iteration())
    model = lgb.train(tu.LGB_PARAMS, lgb.Dataset(d[SNAP_FEATURES], tu.y_index(d["y"])), num_boost_round=rounds)
    text = model.model_to_string()
    return model, {"rows": int(len(d)), "races": int(d["market_id"].nunique()), "rounds": rounds,
                   "race_dates": [str(d["race_date"].min()), str(d["race_date"].max())],
                   "sha256": hashlib.sha256(text.encode()).hexdigest()}, text


def snapshot_rows(books: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """One row a GB/IE horse a snapshot day, with the turn-up features a snapshot can give (the horse's own runs are
    added by turnup.add_recent_runs)."""
    b = books.copy()
    for c in ("back1", "lay1", "last_traded", "runner_total_matched", "market_total_matched", "back1_size", "lay1_size"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["selection_id"] = pd.to_numeric(b["selection_id"], errors="coerce").astype("Int64")
    cat = catalogue.copy()
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce").astype("Int64")
    cat = cat.sort_values("recorded_utc").drop_duplicates(["market_id", "selection_id"], keep="last")
    d = b.merge(cat[["market_id", "selection_id", "market_name", "event_name", "country", "market_start_utc",
                     "runner_name"]], on=["market_id", "selection_id"], how="left")
    d = d[d["country"].isin(["GB", "IE"])].copy()
    d["t"] = pd.to_datetime(d["polled_utc"], utc=True)
    d["days_out"] = (pd.to_datetime(d["market_start_utc"], utc=True) - d["t"]).dt.total_seconds() / 86400
    d["marked_out"] = (d["last_traded"] < p0.MIN_PRICE).astype(int)
    ltp = d["last_traded"].where(d["last_traded"] >= p0.MIN_PRICE)
    d["log_ltp"] = np.log(ltp)
    d["has_price"] = ltp.notna().astype(int)
    active = (d["runner_status"] == "ACTIVE") & (d["status"] == "OPEN")
    keys = [d["market_id"], d["day"]]
    d["n_listed"] = active.astype(int).groupby(keys).transform("sum")
    d["n_priced"] = (active & (d["has_price"] == 1)).astype(int).groupby(keys).transform("sum")
    inv = np.exp(-d["log_ltp"]).where(active)
    d["share"] = inv / inv.groupby([d["market_id"], d["day"]]).transform("sum")
    d["price_rank"] = d["log_ltp"].where(active).groupby([d["market_id"], d["day"]]).rank(pct=True)
    name, ev_ = d["market_name"].fillna(""), d["event_name"].fillna("")
    d["handicap"] = name.str.contains(tu.HANDICAP).astype(int)
    d["jumps"] = name.str.contains(tu.JUMPS).astype(int)
    d["festival"] = (~ev_.str.contains(r"\d{1,2}(?:st|nd|rd|th)\s", regex=True)).astype(int)
    d["ireland"] = (d["country"] == "IE").astype(int)
    d["month"] = pd.to_datetime(d["market_start_utc"], utc=True).dt.month
    from betfair_prices import normalise_horse
    d["name_norm"] = d["runner_name"].map(normalise_horse)
    d["active"] = active
    return d


# ----------------------------------------------------------------------------------------------------------------
# The paper replay
# ----------------------------------------------------------------------------------------------------------------

def traded_since(r, o) -> bool:
    """A trade on this horse since the order (or the fill): its last traded price changed and the market's matched
    money rose. The delayed key's runner volume never moves (0 of 572 eligible horse-days rose day to day on the
    recorded books to 22 Nov 2026), so the horse's own last traded price is the evidence."""
    if not r["has_price"] or pd.isna(o.get("ltp_at")):
        return False
    changed = abs(float(np.exp(r["log_ltp"])) - o["ltp_at"]) > 1e-6
    rose = pd.notna(r.get("market_total_matched")) and pd.notna(o.get("matched_at")) and \
        r["market_total_matched"] > o["matched_at"]
    return bool(changed and rose)


def replay(snap: pd.DataFrame) -> pd.DataFrame:
    """Every order the rules would have placed, day by day, and what became of it."""
    days = sorted(snap["day"].unique())
    by_day = {dy: g.set_index(["market_id", "selection_id"]) for dy, g in snap.groupby("day")}
    orders, positions, done = {}, {}, []
    for dy in days:
        s = by_day[dy]
        # fills of earlier orders, read on this snapshot
        for key in list(orders):
            o = orders[key]
            r = s.loc[key] if key in s.index else None
            if r is None or not r["active"] or r["marked_out"]:
                done.append({**o, "status": "cancelled", "closed_day": dy})
                del orders[key]
                continue
            traded = traded_since(r, o)
            if traded and r["has_price"] and np.exp(r["log_ltp"]) > o["price"] + 1e-6:      # traded through
                positions[key] = {**o, "filled_day": dy, "matched_at": r["market_total_matched"],
                                  "ltp_at": float(np.exp(r["log_ltp"])), "target": to_tick(o["price"] * (1 - X))}
                del orders[key]
            elif (pd.Timestamp(dy) - pd.Timestamp(o["placed_day"])).days >= ORDER_DAYS or r["days_out"] <= ENTRY_MIN:
                done.append({**o, "status": "cancelled", "closed_day": dy})
                del orders[key]
        # exits of filled positions
        for key in list(positions):
            q = positions[key]
            if q["filled_day"] == dy:
                continue
            r = s.loc[key] if key in s.index else None
            if r is None or not r["active"] or r["marked_out"]:
                done.append({**q, "status": "stuck", "closed_day": dy})
                del positions[key]
                continue
            traded = traded_since(r, q)
            if traded and r["has_price"] and np.exp(r["log_ltp"]) < q["target"] - 1e-6:     # traded through
                done.append({**q, "status": "target", "exit_price": q["target"], "closed_day": dy})
                del positions[key]
            elif r["days_out"] <= EXIT_AT:
                if pd.notna(r["lay1"]) and r["lay1"] < 1000:
                    done.append({**q, "status": "crossed", "exit_price": float(r["lay1"]), "closed_day": dy})
                else:
                    done.append({**q, "status": "stuck", "closed_day": dy})
                del positions[key]
        # new resting backs
        busy = set(orders) | set(positions) | {(d["market_id"], d["selection_id"]) for d in done}
        cand = s[s["active"] & (s["has_price"] == 1) & (s["marked_out"] == 0) & s["back1"].notna() & s["lay1"].notna()
                 & (s["lay1"] < 1000) & (s["days_out"] <= ENTRY_MAX) & (s["days_out"] > ENTRY_MIN)
                 & (s["p_ran"] >= P_MIN)]
        for key, r in cand.iterrows():
            if key in busy:
                continue
            inside = tick_down(float(r["lay1"]))
            price = inside if inside > r["back1"] else float(r["lay1"])
            orders[key] = {"market_id": key[0], "selection_id": key[1], "placed_day": dy, "price": price,
                           "p_ran": float(r["p_ran"]), "days_out": float(r["days_out"]),
                           "matched_at": r["market_total_matched"], "ltp_at": float(np.exp(r["log_ltp"])),
                           "back1": float(r["back1"]),
                           "lay1": float(r["lay1"])}
    for o in orders.values():
        done.append({**o, "status": "resting"})
    for q in positions.values():
        done.append({**q, "status": "open"})
    cols = ["market_id", "selection_id", "placed_day", "price", "p_ran", "days_out", "matched_at", "ltp_at", "back1", "lay1",
            "status", "filled_day", "target", "exit_price", "closed_day"]
    return pd.DataFrame(done).reindex(columns=cols)


def settle(trades: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    """Returns per GBP1: closed trades from their green, stuck ones from the race (laid at the SP if it ran, -1 if not,
    0 if Betfair removed the horse); stuck trades on races not yet known stay unsettled."""
    t = trades.merge(outcome, on=["market_id", "selection_id"], how="left")
    ret = np.full(len(t), np.nan)
    closed = t["status"].isin(["target", "crossed"]).to_numpy()
    ret[closed] = ev.net(t.loc[closed, "price"].to_numpy() / t.loc[closed, "exit_price"].to_numpy() - 1)
    stuck = (t["status"] == "stuck").to_numpy()
    void = stuck & (t["removed"] == True).to_numpy()                                        # noqa: E712
    ran = stuck & (t["label"] == "ran").to_numpy() & ~void
    lost = stuck & t["label"].isin(["no_run", "ran_elsewhere"]).to_numpy() & ~void
    ret[void] = 0.0
    ret[ran] = ev.net(t.loc[ran, "price"].to_numpy() / pd.to_numeric(t.loc[ran, "bsp"], errors="coerce").to_numpy() - 1)
    ret[lost] = -1.0
    t["ret"] = ret
    return t


def outcomes(books: pd.DataFrame, catalogue: pd.DataFrame, db: str | None) -> pd.DataFrame:
    """Each recorded horse's race-day outcome (antepost/phase0.py race_day_join), and whether Betfair removed it."""
    cat = catalogue.sort_values("recorded_utc").drop_duplicates(["market_id", "selection_id"], keep="last").copy()
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce").astype("Int64")
    past = cat[pd.to_datetime(cat["market_start_utc"], utc=True) < pd.Timestamp.now(tz="UTC")]
    if past.empty:
        return pd.DataFrame(columns=["market_id", "selection_id", "label", "bsp", "removed"])
    mk = past.drop_duplicates("market_id").rename(columns={"market_start_utc": "market_time"})[
        ["market_id", "market_time", "event_id", "event_name", "country"]]
    rn = past[["market_id", "selection_id", "runner_name"]].copy()
    j, _ = p0.race_day_join(mk, rn, db)
    last = books.sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"], keep="last").copy()
    last["selection_id"] = pd.to_numeric(last["selection_id"], errors="coerce").astype("Int64")
    j = j.merge(last[["market_id", "selection_id", "runner_status"]], on=["market_id", "selection_id"], how="left")
    j["removed"] = j["runner_status"] == "REMOVED"
    for c in ("label", "bsp"):            # no database: nothing known of the races yet
        if c not in j:
            j[c] = np.nan
    return j[["market_id", "selection_id", "label", "bsp", "removed"]]


def summarise(t: pd.DataFrame) -> dict:
    out = {"orders": int(len(t)), "by_status": t["status"].value_counts().to_dict() if len(t) else {}}
    filled = t[t["status"].isin(["target", "crossed", "stuck", "open"])] if len(t) else t
    out["filled"] = int(len(filled))
    out["fill_rate"] = round(len(filled) / max(1, len(t[t["status"] != "resting"])), 4) if len(t) else None
    s = t[t["ret"].notna()] if len(t) else t
    out["settled_trades"] = int(len(s))
    if len(s):
        out["roi"] = round(float(s["ret"].mean()), 4)
        out["races"] = int(s["market_id"].nunique())
        out["roi_by_status"] = {k: [int(len(g)), round(float(g["ret"].mean()), 4)] for k, g in s.groupby("status")}
        if s["market_id"].nunique() >= 2:
            out["interval_90"] = tu.race_bootstrap(s["ret"].to_numpy(), s["market_id"].to_numpy())
    return out


def reading(fwd: pd.DataFrame, today: str) -> dict:
    """The registered reading (ledger antepost-paper-1123): once on or after READ_ON, again only on or after
    READ_AGAIN if fewer than MIN_TRADES had settled by READ_ON."""
    if today < READ_ON:
        return {"verdict": "NOT YET", "read_on": READ_ON}
    s = fwd[fwd["ret"].notna()] if len(fwd) else fwd
    if len(s) < MIN_TRADES:
        return {"verdict": "NOT ENOUGH TRADES" if today < READ_AGAIN else "FAIL", "settled_trades": int(len(s))}
    from research.overfit import dsr_check
    lo = tu.race_bootstrap(s["ret"].to_numpy(), s["market_id"].to_numpy())[0]
    dsr = dsr_check(s.groupby("market_id")["ret"].sum().to_numpy(), N_TRIALS)
    ok = lo > 0 and dsr["passed"]
    return {"verdict": "PASS" if ok else "FAIL", "settled_trades": int(len(s)), "roi_interval_low": lo, "dsr": dsr}


def run(panel, books, catalogue, db, today: str | None = None) -> tuple[dict, pd.DataFrame, str]:
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model, meta, text = train_frozen(panel)
    snap = snapshot_rows(books, catalogue)
    snap, rr = tu.add_recent_runs(snap, db)
    p = model.predict(snap[SNAP_FEATURES])
    snap["p_ran"] = p[:, 1]
    trades = replay(snap)
    trades = settle(trades, outcomes(books, catalogue, db)) if len(trades) else trades.assign(ret=np.nan)
    dev = trades[trades["placed_day"] < FORWARD_START] if len(trades) else trades
    fwd = trades[trades["placed_day"] >= FORWARD_START] if len(trades) else trades
    rep = {"model": meta, "recent_runs": rr, "snapshot_days": [str(snap["day"].min()), str(snap["day"].max())],
           "snapshot_rows": int(len(snap)), "forward_start": FORWARD_START,
           "rules": {"p_min": P_MIN, "x": X, "entry_days_out": [ENTRY_MIN, ENTRY_MAX], "exit_at": EXIT_AT,
                     "order_days": ORDER_DAYS},
           "development_in_sample": summarise(dev), "forward": summarise(fwd), "reading": reading(fwd, today)}
    return rep, trades, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--books", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--db", default="horse_racing.db")
    ap.add_argument("--out", default="out/antepost_paper")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep, trades, text = run(pd.read_parquet(a.panel), pd.read_parquet(a.books), pd.read_parquet(a.catalogue), a.db)
    rep["run_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    trades.to_parquet(out / "paper_trades.parquet", index=False)
    (out / "turnup_frozen.txt").write_text(text)
    (out / "antepost_paper.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
