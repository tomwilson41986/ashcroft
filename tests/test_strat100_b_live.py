"""The b_live half of the 100-strategy harness on TOY data (no real figure is read or made here).

A toy database with the data map's schema (betfair_live_markets, betfair_live_marks, live_orders, betfair_prices,
race_results) and a toy S3 (predictions, raw one-minute books, the other-place books) are built in a temporary
folder; data_b.load_b_live() reads them as it reads the real ones (the pivot, the joins, the day pass over the raw
books, the replays through research/arms on a toy day). Then: every strategy returns a valid bets frame the core
can score; the look-ahead tests (a later column changed must not change a selection); the arithmetic of the shared
pieces on hand-made rows. Every price here is a random draw or a hand-made number.
"""
import gzip
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.strat100 import core, run
from research.strat100 import data_b as B
from research.strat100 import strategies_b_live as S

ROOT = Path(__file__).resolve().parents[1]
BY_ID = {s.id: s for s in S.STRATEGIES}
SPEC_IDS = [f"S{n:03d}" for n in range(1, 53)]
SEL = ["race_date", "race_key", "selection", "side", "stake", "entry_price"]
EM_DASH = chr(8212)
DAYS = [f"2026-10-{d:02d}" for d in range(1, 13)] + ["2026-10-20", "2026-10-21", "2026-11-02", "2026-11-03",
                                                     "2026-11-17"]
REPLAY_DAYS = ["2026-10-12", "2026-11-03"]
NR_DAY, FREE_DAY, STALE_DAY, TBP_DAY = "2026-10-21", "2026-11-02", "2026-11-03", "2026-11-03"
CLOSE = {1: [3.2, 4.9, 6.5, 8.7, 9.8, 13.0, 21.0], 2: [3.2, 5.0, 6.5, 8.5, 9.5, 12.5, 16.0, 10.0]}
OFF = {1: "14:00", 2: "17:30"}
MARK_AT = {"clock_08:00": "08:00", "clock_09:00": "09:00", "clock_10:00": "10:00", "clock_11:00": "11:00",
           "clock_12:00": "12:00"}
OFF_MARKS = {"off_120": 120, "off_60": 60, "off_30": 30, "off_15": 15, "off_10": 10, "off_5": 5, "off_3": 3, "off_1": 1}


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class World:
    """A toy racing world: 17 days, two races a day (a WIN and a PLACE market each), a price path a runner with a
    planted role, the marks and the raw polls read off the same paths."""

    def __init__(self, seed: int = 5):
        self.rng = np.random.default_rng(seed)
        self.cat, self.marks, self.raw, self.orders, self.prices, self.preds = [], [], defaultdict(list), [], [], []
        self.runners = {}                     # (day, event, i) -> dict
        self.other_books, self.other_markets = defaultdict(list), defaultdict(list)
        for k, day in enumerate(DAYS):
            for ev in (1, 2):
                self._race(k, day, ev)
        self._orders()

    # ---- one runner's price at a time -----------------------------------------------------------------------
    def price(self, r: dict, t: pd.Timestamp) -> float:
        b, off, day = r["bsp"], r["off"], r["day"]
        mins = (off - t).total_seconds() / 60.0
        role = r["role"]
        if role == "steam_early":                      # 1.45 x the close, coming in between 07:30 and 10:30 UK
            a, z = B.uk_time(day, "07:30"), B.uk_time(day, "10:30")
            f = min(1.0, max(0.0, (t - a) / (z - a)))
            p = b * 1.45 ** (1.0 - f)
        elif role == "drifter":                        # 0.72 x the close in the morning, out all day
            a = B.uk_time(day, "05:15")
            f = min(1.0, max(0.0, (t - a) / (off - a)))
            p = b * 0.72 ** (1.0 - f)
        elif role == "volume_drifter":                 # jumps out at 08:30 UK
            p = b * (0.85 if t < B.uk_time(day, "08:30") else 0.97 if mins > 90 else 1.0)
        elif role == "late_steam":                     # comes in 45 minutes out
            p = b * (1.10 if mins > 45 else 1.0)
        elif role == "late_drift":
            p = b * (0.90 if mins > 45 else 1.0)
        elif role == "last_steam":                     # comes in 20 minutes out
            p = b * (1.12 if mins > 20 else 1.0)
        else:
            p = b * r["m"] ** max(0.0, min(1.0, mins / 500.0))
        if r.get("removed_at") is None and r.get("field_cut") is not None and t >= r["field_cut"][0] and not r.get("stale"):
            p = 1.0 + (p - 1.0) * (1.0 - r["field_cut"][1] / 100.0)       # the field shortens when a runner goes
        return max(1.02, p)

    def quote(self, r: dict, t: pd.Timestamp) -> dict:
        p = self.price(r, t)
        morning = t < B.uk_time(r["day"], "10:30")
        s = (0.12 if r["role"] == "wide" else 0.03) if morning else 0.02
        back, lay = round(p * (1 - s / 2), 2), round(p * (1 + s / 2), 2)
        z = r["sizes"]
        return {"back1": back, "back1_size": z[0], "back2": round(back * 0.98, 2), "back2_size": z[1],
                "back3": round(back * 0.96, 2), "back3_size": z[2], "lay1": lay, "lay1_size": z[3],
                "lay2": round(lay * 1.02, 2), "lay2_size": z[4], "lay3": round(lay * 1.04, 2), "lay3_size": z[5],
                "last_traded": round(p / 1.20, 2) if (r["role"] == "wide" and morning) else round(p, 2)}

    @staticmethod
    def matched(day: str, t: pd.Timestamp) -> float:
        for hhmm, v in (("08:30", 1000.0), ("09:30", 1300.0), ("10:30", 1500.0)):
            if t < B.uk_time(day, hhmm):
                return v
        return 20000.0

    # ---- one race -------------------------------------------------------------------------------------------
    def _race(self, k: int, day: str, ev: int) -> None:
        rng = self.rng
        n = len(CLOSE[ev])
        bsp = np.array(CLOSE[ev]) * np.exp(rng.normal(0, 0.03, n))
        bsp = bsp * (1.0 / bsp).sum()                                   # a book of 1 at the close
        off = B.uk_time(day, OFF[ev])
        start = off.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        win_id, place_id, event_id = f"1.{100 + k}{ev}1", f"1.{100 + k}{ev}2", f"{9000 + k * 10 + ev}"
        roles = ["fav", "steam_early", "drifter", "volume_drifter", "late_steam", "late_drift", "wide", "last_steam"][:n]
        order = rng.choice(n, size=n, replace=False, p=(1 / bsp) / (1 / bsp).sum())      # the finishing order
        places = 2 if n < 8 else 3
        q = (1 / bsp) / (1 / bsp).sum()
        p_place = core.place_prob(q, places)
        removed_i = 6 if (day == NR_DAY and ev == 1) else None
        t_rem = B.uk_time(day, "09:20")
        for i in range(n):
            sid = 100000 + k * 100 + ev * 10 + i
            name = f"Toy Horse {k} {ev} {i}"
            r = {"day": day, "ev": ev, "i": i, "sid": sid, "name": name, "bsp": float(bsp[i]), "off": off,
                 "role": roles[i], "m": float(rng.uniform(0.97, 1.06)), "win_id": win_id, "place_id": place_id,
                 "sizes": [float(x) for x in rng.integers(25, 120, 6)], "won": int(order[0] == i),
                 "placed": int(i in order[:places]), "place_bsp": float(1.0 / p_place[i] * (1.08 if i == 0 and k % 2 == 0 else 1.0)),
                 "removed_at": t_rem if removed_i == i else None,
                 "field_cut": (t_rem, 6.0) if (removed_i is not None and removed_i != i) else None,
                 "stale": removed_i is not None and i == 5,           # one runner's book does not follow the removal
                 "af": 6.0 if removed_i == i else 2.0 if (day == FREE_DAY and ev == 2 and i == 6)
                 else 26.0 if (day == STALE_DAY and ev == 2 and i == 3) else round(100.0 / float(bsp[i]), 1)}
            model = {"steam_early": 0.95, "drifter": 1.10}.get(r["role"], float(np.exp(rng.normal(0, 0.06))))
            r["pred"] = r["bsp"] * model
            fc = r["bsp"] * (1.0 if r["role"] in ("steam_early", "drifter") else float(np.exp(rng.normal(0, 0.08))))
            self.runners[(day, ev, i)] = r
            for mtype, mid, mname in (("WIN", win_id, "1m Hcap"), ("PLACE", place_id, "To Be Placed")):
                self.cat.append({"race_date": day, "market_id": mid, "market_type": mtype, "market_name": mname,
                                 "market_start_utc": start, "venue": f"Toytown{ev}", "country": "GB" if ev == 1 else "IE",
                                 "event_id": event_id, "race_type": "Hurdle" if ev == 1 else "Flat",
                                 "selection_id": sid, "runner_name": name, "official_rating": "70.0",
                                 "days_since_last_run": "3.0" if i == 4 else "20.0",
                                 "wearing": "cheekpieces" if i == 2 else None,
                                 "forecast_num": str(round((fc - 1.0) * 4)), "forecast_den": "4.0",
                                 "horse_norm": name.lower().replace(" ", ""), "race_results_id": None})
            if r["removed_at"] is None:
                self.preds.append({"date": day, "venue": f"Toytown{ev}", "race_time": off.tz_convert("Europe/London").strftime("%H:%M"),
                                   "market_id": win_id, "selection_id": sid, "runner_name": name,
                                   "predicted_bfsp": round(r["pred"], 3)})
                self.prices.append({"country": "uk", "market_type": "win", "race_date": day, "selection_id": sid,
                                    "win_lose": float(r["won"]), "bsp": r["bsp"], "ppwap": r["bsp"],
                                    "ipmax": 1000.0 if not r["won"] else r["bsp"] * 1.5,
                                    "ipmin": 1.01 if r["won"] else r["bsp"] * (0.45 if i % 2 else 0.8), "pp_vol": 5000.0})
                self.prices.append({"country": "uk", "market_type": "place", "race_date": day, "selection_id": sid,
                                    "win_lose": float(r["placed"]), "bsp": r["place_bsp"], "ppwap": r["place_bsp"],
                                    "ipmax": 50.0, "ipmin": 1.01, "pp_vol": 900.0})
        field = [self.runners[(day, ev, i)] for i in range(n)]

        def book_rows(t: pd.Timestamp, source: str = "trader"):
            live = [r for r in field if r["removed_at"] is None or t < r["removed_at"]]
            for r in field:
                gone = r["removed_at"] is not None and t >= r["removed_at"]
                qd = {} if gone else self.quote(r, t)
                row = {"polled_utc": _iso(t), "source": source, "market_id": win_id, "status": "OPEN", "inplay": 0,
                       "number_of_active_runners": len(live), "market_total_matched": self.matched(day, t),
                       "selection_id": r["sid"], "runner_status": "REMOVED" if gone else "ACTIVE",
                       "adjustment_factor": r["af"], "removal_utc": _iso(r["removed_at"]) if gone else None,
                       "sp_actual": None, "number_of_winners": 1, **qd}
                yield r, row, len(live)

        # the raw polls: every 5 minutes 05:15-13:00 UK
        t = B.uk_time(day, "05:15")
        while t <= B.uk_time(day, "13:00"):
            self.raw[day] += [row for _, row, _ in book_rows(t)]
            t += pd.Timedelta(minutes=5)
        # the marks: the win book off the same paths, the place book off the win price
        times = {m: B.uk_time(day, h) for m, h in MARK_AT.items()}
        times.update({m: off - pd.Timedelta(minutes=v) for m, v in OFF_MARKS.items()})
        times["last"] = off - pd.Timedelta(seconds=30)
        for mark, t in times.items():
            rows = list(book_rows(t))
            pq = np.array([1.0 / self.price(r, t) if row["runner_status"] == "ACTIVE" else 0.0 for r, row, _ in rows])
            for (r, row, n_live), pp in zip(rows, self._place_prices(rows, pq, places, t)):
                base = {"race_date": day, "selection_id": r["sid"], "mark": mark, "polled_utc": _iso(t),
                        "minutes_to_off": (off - t).total_seconds() / 60.0, "market_status": "OPEN",
                        "runner_status": row["runner_status"], "active_runners": n_live, "bsp": None,
                        "market_total_matched": row["market_total_matched"]}
                self.marks.append({**base, "market_id": win_id, **{c: row.get(c) for c in B.LEVELS + ["last_traded"]}})
                self.marks.append({**base, "market_id": place_id, **pp})
        for r in field:
            gone = r["removed_at"] is not None
            fin = {"race_date": day, "selection_id": r["sid"], "mark": "final", "polled_utc": _iso(off + pd.Timedelta(hours=2)),
                   "minutes_to_off": -120.0, "market_status": "CLOSED", "active_runners": 0, "market_total_matched": 0.0}
            self.marks.append({**fin, "market_id": win_id, "bsp": None if gone else r["bsp"],
                               "runner_status": "REMOVED" if gone else "WINNER" if r["won"] else "LOSER"})
            self.marks.append({**fin, "market_id": place_id, "bsp": None if gone else r["place_bsp"],
                               "runner_status": "REMOVED" if gone else "WINNER" if r["placed"] else "LOSER"})
        if day == TBP_DAY and ev == 1:
            self._tbp(day, field, off, event_id, start, q)

    def _place_prices(self, rows, pq, places, t):
        live = pq > 0
        p = np.zeros(len(pq))
        p[live] = core.place_prob(pq[live] / pq[live].sum(), min(places, int(live.sum()) - 1))
        out = []
        for (r, row, _), pl in zip(rows, p):
            if row["runner_status"] != "ACTIVE":
                out.append({})
                continue
            fair = 1.0 / pl
            if r["role"] == "last_steam":                      # the place price does not follow the late win steam
                pr = [self.price(x, r["off"] - pd.Timedelta(minutes=60)) for x, _, _ in rows]
                q0 = np.array([1.0 / v for v in pr])
                fair = 1.0 / core.place_prob(q0 / q0.sum(), places)[r["i"]]
            if r["role"] == "wide":
                fair *= 1.12                                   # a place offer over the win-implied price
            back, lay = round(fair * 0.99, 3), round(fair * 1.01, 3)
            out.append({"back1": back, "back1_size": 60.0, "back2": round(back * 0.99, 3), "back2_size": 30.0,
                        "back3": None, "back3_size": None, "lay1": lay, "lay1_size": 55.0, "lay2": round(lay * 1.01, 3),
                        "lay2_size": 30.0, "lay3": None, "lay3_size": None, "last_traded": round(fair, 3)})
        return out

    def _tbp(self, day, field, off, event_id, start, q) -> None:
        mid, t = "1.777", off - pd.Timedelta(minutes=5)
        p3 = core.place_prob(q, 3)
        for r, pl in zip(field, p3):
            self.other_markets[day].append({"market_id": mid, "market_type": "OTHER_PLACE", "market_name": "3 TBP",
                                            "market_start_utc": start, "venue": "Toytown1", "country": "GB",
                                            "event_id": event_id, "selection_id": r["sid"], "runner_name": r["name"]})
            fair = 1.0 / pl
            back = round(fair * (1.15 if r["i"] == 3 else 0.97), 3)             # one offer over the win-implied price
            lay = round(fair * (0.80 if r["i"] == 5 else 1.03), 3)              # one lay under it
            base = {"source": "recorder", "market_id": mid, "inplay": 0, "selection_id": r["sid"], "number_of_winners": 3}
            self.other_books[day].append({**base, "polled_utc": _iso(t), "status": "OPEN", "runner_status": "ACTIVE",
                                          "number_of_active_runners": len(field), "back1": back, "back1_size": 14.0,
                                          "lay1": lay, "lay1_size": 9.0})
            self.other_books[day].append({**base, "polled_utc": _iso(off + pd.Timedelta(hours=1)), "status": "CLOSED",
                                          "source": "final", "number_of_active_runners": 0,
                                          "runner_status": "WINNER" if r["i"] in (0, 3, 4) else "LOSER"})

    # ---- the live trader's own ledger -----------------------------------------------------------------------
    def _orders(self) -> None:
        seq = 0
        for day in DAYS:
            for ev in (1, 2):
                for i, plan in ((1, (("06:00", 40.0, 0.09), ("06:30", 30.0, 0.05), ("08:10", 25.0, 0.08))),
                                (6, (("06:05", 20.0, 0.04), ("07:00", 15.0, 0.035)))):
                    r = self.runners[(day, ev, i)]
                    if day == NR_DAY and ev == 1 and i == 1:
                        plan = plan + (("09:40", 20.0, 0.06),)          # a top-up after the day's removal
                    fills = []
                    for hhmm, stake, edge in plan:
                        t = B.uk_time(day, hhmm)
                        px = self.quote(r, t)["back1"]
                        seq += 1
                        cut = 0.94 if (r["field_cut"] is not None and t < r["field_cut"][0]) else 1.0
                        fills.append((stake, px * cut, px))
                        self.orders.append({"race_date": day, "seq": seq, "ts": _iso(t), "mode": "live", "event": "back",
                                            "market_id": r["win_id"], "selection_id": str(r["sid"]), "runner": r["name"],
                                            "minutes_to_off": str(round((r["off"] - t).total_seconds() / 60, 1)),
                                            "edge": str(edge), "best_back": str(px), "matched": str(stake),
                                            "avg_price": str(px), "status": "SUCCESS", "p_model": "0.1", "p_market": "0.09"})
                    seq += 1
                    gone = r["removed_at"] is not None
                    m = sum(f[0] for f in fills)
                    self.orders.append({"race_date": day, "seq": seq, "ts": _iso(r["off"] + pd.Timedelta(minutes=6)),
                                        "mode": "live", "event": "settle", "market_id": r["win_id"],
                                        "selection_id": str(r["sid"]), "runner": r["name"], "matched": str(m),
                                        "avg_price": str(round(sum(f[0] * f[1] for f in fills) / m, 4)),
                                        "result": "REMOVED" if gone else "WINNER" if r["won"] else "LOSER",
                                        "bsp": "" if gone else str(round(r["bsp"], 4)), "pnl": "1.0"})

    # ---- as the sources hold them ---------------------------------------------------------------------------
    def write_db(self, path: Path) -> None:
        con = sqlite3.connect(path)
        pd.DataFrame(self.cat).to_sql("betfair_live_markets", con, index=False)
        pd.DataFrame(self.marks).to_sql("betfair_live_marks", con, index=False)
        pd.DataFrame(self.orders).reindex(columns=B.ORDER_COLS).to_sql("live_orders", con, index=False)
        pd.DataFrame(self.prices).to_sql("betfair_prices", con, index=False)
        rr = []
        for (day, ev, i), r in self.runners.items():
            if i == 4:                                   # one earlier run: a second career run on the card
                rr.append({"race_date": "2026-09-01", "horse_name": r["name"], "race_type": "Handicap Chase",
                           "race_name": "Toy Handicap Chase", "headgear": ""})
            if i == 2:                                   # two earlier runs, never in a handicap or headgear
                rr += [{"race_date": d, "horse_name": r["name"], "race_type": "Maiden", "race_name": "Toy Maiden",
                        "headgear": ""} for d in ("2026-08-01", "2026-09-05")]
        pd.DataFrame(rr).to_sql("race_results", con, index=False)
        con.commit()
        con.close()

    def s3(self) -> dict:
        def gz(rows):
            return gzip.compress(pd.DataFrame(rows).to_csv(index=False).encode())
        out = {}
        for day in DAYS:
            out[f"predictions/{day}.csv"] = pd.DataFrame([p for p in self.preds if p["date"] == day]).to_csv(index=False).encode()
            out[f"betfair_live/{day}/books_trader.csv.gz"] = gz(self.raw[day])
            cat = [dict(c, recorded_utc=f"{day}T05:00:00Z") for c in self.cat if c["race_date"] == day]
            out[f"betfair_live/{day}/markets.csv.gz"] = gz(cat)
            if self.other_books[day]:
                out[f"betfair_live/{day}/books_other_place.csv.gz"] = gz(self.other_books[day])
                out[f"betfair_live/{day}/markets_other_place.csv.gz"] = gz(self.other_markets[day])
        return out

    def toy_day(self, day: str):
        """The day as research/arms/core.load_day gives it, from the toy raw books (the replay's input)."""
        from trading.exchange import Book, Market, Quote
        if day not in REPLAY_DAYS:
            return None
        raw = B.type_raw(pd.DataFrame(self.raw[day]).assign(market_id=lambda d: d["market_id"].astype(str)))
        markets, polls, settled = [], defaultdict(list), {}
        for ev in (1, 2):
            field = [r for (d, e, _), r in self.runners.items() if d == day and e == ev]
            m = Market(field[0]["win_id"], f"Toytown{ev}", "GB" if ev == 1 else "IE", field[0]["off"].to_pydatetime(),
                       name="1m Hcap", runners={r["sid"]: r["name"] for r in field})
            markets.append(m)
            for r in field:
                if r["removed_at"] is None:
                    settled[(m.market_id, r["sid"])] = (r["bsp"], "WINNER" if r["won"] else "LOSER")
            for t, g in raw[raw["market_id"].eq(m.market_id)].groupby("t", sort=True):
                rs = {int(x.selection_id): Quote(int(x.selection_id), status=x.runner_status,
                                                 back=[(getattr(x, f"back{i}"), getattr(x, f"back{i}_size")) for i in (1, 2, 3)
                                                       if getattr(x, f"back{i}") > 1],
                                                 lay=[(getattr(x, f"lay{i}"), getattr(x, f"lay{i}_size")) for i in (1, 2, 3)
                                                      if getattr(x, f"lay{i}") > 1], last_traded=x.last_traded)
                      for x in g.itertuples(index=False)}
                polls[m.market_id].append((t.to_pydatetime(), Book(m.market_id, "OPEN", False, 1000.0, rs)))
            d = date.fromisoformat(day)
            closed = {s: Quote(s, status=st, bsp=b) for (mm, s), (b, st) in settled.items() if mm == m.market_id}
            polls[m.market_id].append((datetime(d.year, d.month, d.day, 23, 0, tzinfo=timezone.utc),
                                       Book(m.market_id, "CLOSED", False, None, closed, True)))
        preds = pd.DataFrame([p for p in self.preds if p["date"] == day])[["market_id", "selection_id", "predicted_bfsp"]]
        rem = B.removals_of(raw)
        return SimpleNamespace(day=date.fromisoformat(day), kind="win", markets=markets, polls=dict(polls),
                               poll_times={k: [t for t, _ in v] for k, v in polls.items()},
                               starts={m.market_id: m.start for m in markets}, real_fills=defaultdict(list), preds=preds,
                               settled=settled, removed=rem[["market_id", "selection_id", "af", "rt"]], places={},
                               ledger=pd.DataFrame(), last_poll=None, provisional=False)

    def steam(self, db, days):
        rows = [{"race_date": r["day"], "race_time": {1: "2:00", 2: "5:30"}[r["ev"]], "track": f"Toytown{r['ev']}",
                 "horse_name": r["name"], "steam_score": 0.09 if r["i"] in (1, 4) else 0.01, "steam_field": 7.0}
                for r in self.runners.values()]
        return pd.DataFrame(rows), {"steam_p80": 0.03, "steam_p95": 0.08, "steam_cutoff": B.STEAM_CUTOFF}


@pytest.fixture(scope="module")
def world():
    return World()


def _load(world, tmp, monkeypatch_env: dict):
    """data_b.load_b_live() on the toy database and the toy S3 (a MonkeyPatch the caller undoes)."""
    mp = pytest.MonkeyPatch()
    store = world.s3()
    db = tmp / "toy.db"
    if not db.exists():
        world.write_db(db)
    mp.setattr(B, "_get", lambda key: store.get(key))
    mp.setattr(B, "load_replay_day", world.toy_day)
    mp.setattr(B, "steam_scores", world.steam)
    mp.setattr(S, "DRAWS_CLV", 400)
    env = {"STRAT100_DB": str(db), "STRAT100_WORK": str(tmp / "work"), "STRAT100_B_JOBS": "1", "STRAT100_B_DRAWS": "300",
           **monkeypatch_env}
    for k, v in env.items():
        mp.setenv(k, v)
    try:
        return B.load_b_live()
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def frames(world, tmp_path_factory):
    return _load(world, tmp_path_factory.mktemp("b_live"), {})


@pytest.fixture(autouse=True)
def _few_draws(monkeypatch):
    monkeypatch.setattr(S, "DRAWS_CLV", 400)


def build(sid, frames):
    out = BY_ID[sid].build(frames)
    bets, days = out if isinstance(out, tuple) else (out, frames["days"])
    return bets, days


def sel(bets: pd.DataFrame) -> pd.DataFrame:
    if len(bets) == 0:
        return pd.DataFrame(columns=SEL)
    return bets[SEL].sort_values(SEL[:4], kind="stable").reset_index(drop=True).round(6)


def changed(frames, **over) -> dict:
    """The frames with some replaced (copies: the shared frames are never changed in place)."""
    return {**frames, **over}


# ---- the loader -------------------------------------------------------------------------------------------------

def test_the_loader_reads_the_toy_sources_and_says_what_it_holds(frames, world):
    r = frames["runners"]
    assert frames["days"] == DAYS and set(r["market_type"]) == {"WIN", "PLACE"}
    assert len(r) == len(world.cat) and not r.duplicated(B.KEY).any()                 # one row a runner a market
    for tag in B.TAGS.values():
        assert f"{tag}_back1" in r.columns and f"{tag}_t" in r.columns
    win = r[r["market_type"].eq("WIN")]
    gone = win["final_status"].eq("REMOVED")
    assert gone.sum() == 1 and win.loc[~gone, "bsp"].gt(1).all() and win.loc[~gone, "predicted_bfsp"].gt(1).all()
    assert win["F"].gt(1).all() and r.loc[r["market_type"].eq("PLACE"), "predicted_bfsp"].isna().all()
    meta = frames["meta"]
    assert meta["days_books"] == DAYS and meta["days_replay"] == REPLAY_DAYS and meta["days_preds"] == DAYS
    assert "DELAYED" in meta["delayed_key"] and not meta["day_errors"]
    assert len(frames["removals"]) == 1 and frames["removals"]["af"].tolist() == [6.0]      # the toy raw books are WIN only
    assert frames["removals"]["rt"].iloc[0] == B.uk_time(NR_DAY, "09:20")
    assert set(frames["replay"]["id"]) <= set(S.REPLAY_ARMS) and "S001" in set(frames["replay"]["id"])
    assert set(frames["replay"]["race_date"]) <= set(REPLAY_DAYS)


def test_the_marks_pivot_keeps_each_marks_book_under_its_tag(frames, world):
    r = world.runners[("2026-10-05", 1, 1)]
    row = frames["runners"].set_index(B.KEY).loc[(r["win_id"], r["sid"])]
    t = B.uk_time("2026-10-05", "09:00")
    assert row["c09_back1"] == pytest.approx(world.quote(r, t)["back1"]) and row["c09_t"] == t
    assert row["o15_mto"] == pytest.approx(15.0) and row["bsp"] == pytest.approx(r["bsp"])
    b = B.book(frames["runners"], "c09")
    assert {"back1", "lay1", "t", "mto", "n_active", "n_rows"} <= set(b.columns) and (b["n_rows"] == b["n_active"]).all()
    assert not any(c.startswith("c09_") for c in b.columns)


def test_a_missing_source_is_no_data_found_not_a_crash(world, tmp_path, monkeypatch):
    monkeypatch.setenv("STRAT100_DB", str(tmp_path / "absent.db"))
    with pytest.raises(B.NoData, match="no data found"):
        B.load_b_live()
    bare = _load(world, tmp_path, {"STRAT100_B_RAW": "0", "STRAT100_B_STEAMER": "0", "STRAT100_B_DAYS": "2026-10-01..2026-10-03"})
    assert bare["days"] == DAYS[:3] and bare["meta"]["days_books"] == [] and len(bare["replay"]) == 0
    assert any("no data found" in p for p in bare["meta"]["problems"])
    for sid in ("S001", "S011", "S023", "S031", "S036", "S052"):          # nothing invented where the source is missing
        bets, days = build(sid, bare)
        assert len(bets) == 0 and days == []
    bets, days = build("BASE-B", bare)                                   # the real fills need no raw book
    assert len(bets) and days == DAYS[:3] and bets["cut_factor"].between(0, 1, inclusive="right").all()


def test_the_day_pass_in_two_processes_and_from_its_cache_gives_the_same_orders(world, tmp_path, frames):
    env = {"STRAT100_B_JOBS": "2", "STRAT100_B_DAYS": "2026-10-11..2026-10-12", "STRAT100_B_STEAMER": "0"}
    a = _load(world, tmp_path, env)
    assert a["meta"]["days_books"] == ["2026-10-11", "2026-10-12"] and a["meta"]["days_replay"] == ["2026-10-12"]
    assert len(list((tmp_path / "work" / "b_live").glob("day_2026-10-1*.pkl"))) == 2
    b = _load(world, tmp_path, env)                                       # read back from the cache
    want = frames["replay"][frames["replay"]["race_date"].eq("2026-10-12")].reset_index(drop=True)
    cols = ["id", "race_key", "selection", "side", "stake", "entry_price", "exit_price", "cut_factor", "ts"]
    for got in (a, b):
        pd.testing.assert_frame_equal(got["replay"][cols].reset_index(drop=True), want[cols], check_dtype=False)
    core.score(BY_ID["S011"], core.empty_bets(), [], n_trials=486, draws=100)      # nothing to read is not a crash


def test_fills_are_cut_by_the_removals_after_them_and_reconciled_to_betfairs_settled_price(frames, world):
    f = frames["fills"]
    assert f["first"].sum() == len(DAYS) * 4 and (~f["first"]).sum() == len(DAYS) * 6 + 1   # 2 horses a race, 3 + 2 orders
    r = world.runners[(NR_DAY, 1, 1)]
    g = f[f["market_id"].eq(r["win_id"]) & f["selection_id"].eq(r["sid"])].sort_values("ts")
    t_rem = B.uk_time(NR_DAY, "09:20")
    assert (g["ts"] < t_rem).tolist() == [True, True, True, False]
    assert g["cut_raw"].tolist() == pytest.approx([0.94] * 3 + [1.0])        # only the fills matched before the removal
    assert g["cut_factor"].tolist() == pytest.approx([0.94] * 3 + [1.0], abs=1e-3)   # and Betfair's settled price agrees
    other = f[~f["market_id"].eq(r["win_id"])]
    assert (other["cut_factor"] == 1.0).all()
    gone = world.runners[(NR_DAY, 1, 6)]
    base, _ = build("BASE-B", frames)
    void = base[base["selection"].eq(str(gone["sid"]))]
    assert len(void) == 2 and void["void"].all()                                            # a removed horse's backs


def test_cut_factor_takes_only_other_runners_removed_later_and_each_side_has_its_floor():
    rem = pd.DataFrame({"market_id": ["m", "m", "m", "x"], "selection_id": [1, 2, 3, 9], "af": [10.0, 2.0, 20.0, 50.0],
                        "rt": pd.to_datetime(["2026-10-05T09:00Z", "2026-10-05T09:30Z", "2026-10-05T07:00Z",
                                              "2026-10-05T09:00Z"], utc=True)})
    ts = pd.to_datetime(["2026-10-05T08:00Z"] * 3 + ["2026-10-05T09:10Z"], utc=True)
    every = B.cut_factor(rem, ["m", "m", "m", "m"], [5, 1, 5, 5], ts)             # K6's words: every factor (a back)
    assert every.tolist() == pytest.approx([0.90 * 0.98, 0.98, 0.90 * 0.98, 0.98])   # never its own, never an earlier one
    floor = B.cut_factor(rem, ["m", "m", "m", "m"], [5, 1, 5, 5], ts, min_af=B.BETFAIR_MIN_AF)
    assert floor.tolist() == pytest.approx([0.90, 1.0, 0.90, 1.0])                # Betfair's rule: under 2.5% cuts nothing
    assert B.min_af_for("back") == 0.0 and B.min_af_for("lay") == 2.5             # the conservative reading a side
    assert B.cut_factor(None, ["m"], [5], ts[:1]).tolist() == [1.0]


def test_a_back_is_cut_by_every_factor_and_a_lay_only_from_betfairs_floor():
    """K6 as read here: the literal rule for a back (harsher than Betfair's), Betfair's 2.5% floor for a lay (the
    literal rule would credit a layer with a cut Betfair does not make) and for S024, whose premise it is."""
    rem = pd.DataFrame({"market_id": ["m"], "selection_id": [9], "af": [2.0],
                        "rt": pd.to_datetime(["2026-10-05T10:00Z"], utc=True)})
    x = pd.DataFrame({"market_id": ["m"], "selection_id": [1], "t": pd.to_datetime(["2026-10-05T09:00Z"], utc=True)})
    fr = {"removals": rem}
    assert S._cut(fr, x).tolist() == pytest.approx([0.98]) and S._cut(fr, x, "lay").tolist() == [1.0]
    assert S._cut(fr, x, min_af=B.BETFAIR_MIN_AF).tolist() == [1.0]
    b = S._bets(x.assign(race_date="2026-10-05", bsp=5.0, final_status="LOSER"), "back", 6.0, 10.0, cut=S._cut(fr, x))
    settled = core.settle(core.check_bets(b))
    assert settled["pnl"].iloc[0] == pytest.approx(10.0 * (6.0 * 0.98 / 5.0 - 1.0))


# ---- the strategies ---------------------------------------------------------------------------------------------

def test_the_registry_holds_every_specified_strategy_once():
    ids = [s.id for s in S.STRATEGIES]
    assert len(ids) == len(set(ids)) and set(SPEC_IDS) <= set(ids)
    assert set(ids) - set(SPEC_IDS) == {"BASE-B", "S008-BASE", "S024-OTHER"}
    for s in S.STRATEGIES:
        assert s.dataset == "b_live" and s.tested_before and s.build.__doc__
        assert s.paired is None or (s.paired in BY_ID and BY_ID[s.paired].benchmark)
    found, problems = run.discover()
    assert not any("b_live" in k for k in problems) and {s.id for s in found} >= set(ids)
    assert set(S.REPLAY_ARMS) == {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S010", "S013",
                                  "S014", "S017", "S018", "S019", "S020", "S037"}


@pytest.fixture(scope="module")
def built(frames):
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "DRAWS_CLV", 400)
    try:
        before = {k: (v.shape, tuple(v.columns)) for k, v in frames.items() if isinstance(v, pd.DataFrame)}
        copy = {k: v.copy() for k, v in frames.items() if isinstance(v, pd.DataFrame)}
        out = {s.id: build(s.id, frames) for s in S.STRATEGIES}
        assert before == {k: (v.shape, tuple(v.columns)) for k, v in frames.items() if isinstance(v, pd.DataFrame)}
        for k, v in copy.items():                              # no strategy changed a shared frame in place
            pd.testing.assert_frame_equal(frames[k], v, obj=k)
        return out
    finally:
        mp.undo()


@pytest.mark.parametrize("sid", [s.id for s in S.STRATEGIES])
def test_every_strategy_returns_a_bets_frame_the_core_scores(sid, built, frames):
    bets, days = built[sid]
    assert isinstance(bets, pd.DataFrame) and set(days) <= set(frames["days"])
    b = core.check_bets(bets, BY_ID[sid])
    assert set(b["race_date"]) <= set(days) and int(b["unsettled"].sum()) == 0
    live = b[~b["void"]]
    assert (live["stake"] >= 2.0 - 1e-9).all() or sid in ("BASE-B",) or BY_ID[sid].entry == S.NO_HAIRCUT
    assert not live.duplicated(["race_date", "race_key", "selection", "side", "ts"] if "ts" in live else SEL[:4]).any()
    r = core.score(BY_ID[sid], bets, days, n_trials=486, draws=200)
    assert r["error"] is None and r["verdict"] in ("not read", "not read: too few", "FAIL", "PASS", "PASS (not blind)")
    if BY_ID[sid].side != "both" and len(live):
        assert set(live["side"]) == {BY_ID[sid].side}


def test_the_planted_runners_are_found(built):
    """The toy world plants a runner for most rules; a rule that finds none of them has lost its way."""
    for sid in ("S001", "S002", "S006", "S007", "S008", "S008-BASE", "S011", "S012", "S013", "S017", "S020", "BASE-B",
                "S015", "S016", "S021", "S022", "S023", "S025", "S026", "S027", "S028", "S029", "S030", "S031", "S032",
                "S033", "S034", "S035", "S036", "S037", "S038", "S039", "S040", "S041", "S042", "S043", "S044", "S046",
                "S047", "S048", "S049", "S050", "S051", "S052", "S024", "S024-OTHER"):
        assert len(built[sid][0]) > 0, sid


def test_the_replayed_arms_differ_from_the_rule_only_as_registered(built):
    base, no_top, bar8, bar0, lad = (built[s][0] for s in ("S001", "S013", "S003", "S002", "S017"))
    horse = ["race_date", "race_key", "selection"]
    assert base.duplicated(horse).any() and not no_top.duplicated(horse).any()            # top-ups, and none
    assert base.groupby(horse).size().max() <= 4                                          # a first back and 3 top-ups
    assert base["edge"].min() >= 0.03 - 1e-9 and bar8["edge"].min() >= 0.08 - 1e-9 and bar0["edge"].min() >= -1e-9
    assert len(bar8) <= len(base) <= len(bar0)
    assert (base.groupby(horse)["stake"].sum() <= 200.0 + 1e-6).all()
    assert (base.groupby("race_date")["stake"].sum() <= 4000.0 + 1e-6).all()
    assert set(built["S020"][0].groupby(horse)["stake"].sum().round(2)) <= {40.0} | set(
        built["S020"][0].groupby(horse)["stake"].sum().round(2)[lambda s: s < 40.0])
    lays = built["S037"][0]
    assert set(lays["side"]) == {"lay"} and (lays["entry_price"] <= 10.0).all() and not lays.duplicated(horse).any()
    t9 = built["S009"][0]
    assert len(t9) and (pd.to_datetime(t9["ts"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%H:%M") <= "06:15").all()
    t10 = built["S010"][0]
    late = pd.to_datetime(t10["ts"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%H:%M") > "10:00"
    assert (t10.loc[late, "minutes_to_off"] >= 360).all()
    assert len(lad) and len(lad) >= len(base) - 5


def test_base_b_variants_only_drop_or_cut_the_real_fills(built, frames):
    base = built["BASE-B"][0]
    total = base["stake"].sum()
    for sid in ("S015", "S016", "S021", "S022", "S023"):
        b = built[sid][0]
        assert b["stake"].sum() <= total + 1e-9, sid
        j = b.merge(base, on=["race_date", "race_key", "selection", "ts"], suffixes=("", "_base"))
        assert len(j) == len(b) and (j["stake"] <= j["stake_base"] + 1e-9).all() and (j["entry_price"] == j["entry_price_base"]).all()
    assert (built["S022"][0].groupby(["race_date", "race_key"])["stake"].sum() <= 400.0 + 1e-9).all()
    f = frames["fills"]
    assert len(built["S015"][0]) == int((f["first"] | (f["edge"] >= 0.07)).sum())
    held = built["S025"][0]
    assert set(held["exit"]) == {"result"} and held["stake"].sum() == pytest.approx(total)
    cond = built["S026"][0].merge(base, on=["race_date", "race_key", "selection", "ts"], suffixes=("", "_b"))
    p_cut = np.maximum(1.01, cond["entry_price"] * cond["cut_factor"])
    live = ~cond["void"]
    assert (cond.loc[live, "exit"].eq("bsp") == (cond.loc[live, "exit_price_b"] <= p_cut[live])).all()


def test_the_in_running_lay_fills_only_through_the_target_by_a_tick(built, frames):
    b = built["S027"][0]
    p = frames["prices"][frames["prices"]["market_type"].eq("win")]
    j = b[~b["void"]].assign(selection_id=lambda d: d["selection"].astype(int)).merge(p, on=["race_date", "selection_id"])
    lay_at = 1.0 + (j["bsp"] - 1.0) * 0.5
    hit = (j["ipmin"] <= np.asarray(core.tick_under(lay_at.to_numpy()), float)).to_numpy()
    assert (j["exit"].eq("inplay").to_numpy() == hit).all() and hit.any() and (~hit).any()
    assert j.loc[hit, "exit_price"].to_numpy() == pytest.approx(lay_at[hit].to_numpy())


def test_the_ladder_and_the_day_limit():
    b = pd.DataFrame({"back1": [10.0, 10.0, 10.0], "back1_size": [30.0, 30.0, 1.0], "back2": [9.8, 9.8, 9.8],
                      "back2_size": [20.0, 20.0, 0.5], "back3": [9.6, 9.6, np.nan], "back3_size": [50.0, 50.0, np.nan]})
    stake, price = S.ladder(b, [60.0, 60.0, 60.0], [True, False, True], [True, True, True])
    assert stake.tolist() == pytest.approx([60.0, 30.0, 0.0])                 # three levels; level 2 refused; under GBP2
    assert price[0] == pytest.approx((30 * 10 + 20 * 9.8 + 10 * 9.6) / 60) and price[1] == 10.0
    x = pd.DataFrame({"race_date": ["d"] * 3 + ["e"], "ev": [0.05, 0.09, 0.07, 0.01]})
    got = S.day_limit(x, [60.0, 60.0, 60.0, 60.0], ["ev"], limit=100.0, ascending=[False])
    assert got.tolist() == pytest.approx([0.0, 60.0, 40.0, 60.0])             # the best first, the crossing bet cut
    lay = S.day_limit(x.iloc[:2], [50.0, 50.0], ["ev"], limit=200.0, unit_risk=[3.0, 3.0], ascending=[False])
    assert lay.tolist() == pytest.approx([200.0 / 3.0 - 50.0, 50.0])          # a lay spends its liability


def test_place_bets_carry_their_own_market_and_voids_are_counted(built, frames):
    for sid in ("S048", "S049", "S050", "S051"):
        b = built[sid][0]
        place_ids = set(frames["runners"].loc[frames["runners"]["market_type"].eq("PLACE"), "market_id"])
        assert set(b["market"]) == {"place"} and set(b["race_key"]) <= place_ids, sid
    assert set(built["S048"][0]["entry_type"]) == {"sp"} and set(built["S048"][0]["exit"]) == {"result"}
    tbp = built["S052"][0]
    assert set(tbp["market"]) == {"3tbp"} and set(tbp["side"]) == {"back", "lay"} and set(tbp["exit"]) == {"result"}
    assert (tbp.loc[tbp["side"].eq("back"), "stake"] <= 10.0).all()
    lays = tbp[tbp["side"].eq("lay")]
    assert (lays["stake"] * (lays["entry_price"] - 1.0) <= 20.0 + 1e-9).all()


# ---- look-ahead: a later column changed must not change a selection -----------------------------------------------

LATER_TAGS = ["c11", "c12", "o120", "o60", "o30", "o15", "o10", "o5", "o3", "o1", "last"]
MORNING = ["S008", "S008-BASE", "S011", "S024", "S024-OTHER", "S028", "S029", "S030", "S031", "S033", "S034", "S039",
           "S040", "S042", "S043", "S044", "S045", "S047"]


def _scramble(runners: pd.DataFrame, tags, rng, close: bool = True, days=None) -> pd.DataFrame:
    """The runners frame with the blocks of ``tags`` and (with ``close``) the BSP and the final status changed."""
    r = runners.copy()
    rows = r["race_date"].isin(days) if days is not None else pd.Series(True, index=r.index)
    for tag in tags:
        for c in B.LEVELS + ["last_traded"]:
            col = f"{tag}_{c}"
            r.loc[rows, col] = r.loc[rows, col] * rng.uniform(0.6, 1.6, int(rows.sum()))
        r.loc[rows, f"{tag}_mtm"] = r.loc[rows, f"{tag}_mtm"] * 3.0
    if close:
        r.loc[rows, "bsp"] = r.loc[rows, "bsp"] * rng.uniform(0.5, 2.0, int(rows.sum()))
        flip = rows & r["final_status"].isin(["WINNER", "LOSER"])
        r.loc[flip, "final_status"] = np.where(r.loc[flip, "final_status"].eq("WINNER"), "LOSER", "WINNER")
    return r


def test_a_morning_selection_reads_nothing_after_its_mark(frames, built):
    """The BSP, the result and every mark from 11:00 on are changed on the last day: no morning rule moves (the
    walk-forward tables of S008 read earlier days' closes, so only the last day's are changed)."""
    alt = changed(frames, runners=_scramble(frames["runners"], LATER_TAGS, np.random.default_rng(1), days=[DAYS[-1]]))
    for sid in MORNING + ["S038"]:
        tags_ok = sid != "S038"                       # S038 reads the 11:00 and 12:00 marks: only the close is changed
        a = alt if tags_ok else changed(frames, runners=_scramble(frames["runners"], LATER_TAGS[2:], np.random.default_rng(1),
                                                                  days=[DAYS[-1]]))
        pd.testing.assert_frame_equal(sel(build(sid, a)[0]), sel(built[sid][0]), check_dtype=False, obj=sid)


def test_a_selection_before_the_off_reads_no_close_and_no_result(frames, built):
    """Only the BSP and the final status change (every day): the pre-off rules pick the same runners at the same
    stakes and prices; what may change is void, won and the exit price."""
    r = frames["runners"].copy()
    rng = np.random.default_rng(2)
    r["bsp"] = r["bsp"] * rng.uniform(0.5, 2.0, len(r))
    ok = r["final_status"].isin(["WINNER", "LOSER"])
    r.loc[ok, "final_status"] = np.where(r.loc[ok, "final_status"].eq("WINNER"), "LOSER", "WINNER")
    alt = changed(frames, runners=r)
    for sid in ("S011", "S012", "S030", "S033", "S034", "S035", "S038", "S039", "S040", "S042", "S043", "S044", "S045",
                "S047", "S049", "S050", "S051"):
        pd.testing.assert_frame_equal(sel(build(sid, alt)[0]), sel(built[sid][0]), check_dtype=False, obj=sid)
    # S048's order (its limit) is fixed before the off; the place SP decides only whether it is matched
    orders = lambda fr: S.place_chance(fr, "o5", fallback_last=True)[["market_id", "selection_id", "wmid", "p_place"]]  # noqa: E731
    pd.testing.assert_frame_equal(orders(alt), orders(frames))


def test_the_real_fill_rules_read_no_close_on_the_day_they_cut(frames, built):
    f = frames["fills"].copy()
    last = f["race_date"].eq(DAYS[-1])
    f.loc[last, "bsp"] = f.loc[last, "bsp"] * 1.7
    f.loc[last, "result"] = np.where(f.loc[last, "result"].eq("WINNER"), "LOSER", "WINNER")
    alt = changed(frames, fills=f, runners=_scramble(frames["runners"], [], np.random.default_rng(3), days=[DAYS[-1]]))
    for sid in ("S015", "S016", "S021", "S022", "S023"):
        pd.testing.assert_frame_equal(sel(build(sid, alt)[0]), sel(built[sid][0]), check_dtype=False, obj=sid)


def test_the_raw_book_rules_read_no_poll_after_their_entry(frames, world):
    """A day function's selection is the same when every raw poll after 11:00 UK is changed and the day's closes
    and results are swapped (their entries are all in the morning)."""
    rng = np.random.default_rng(4)
    for day in (NR_DAY, STALE_DAY):
        runners_day = frames["runners"][frames["runners"]["race_date"].eq(day)].reset_index(drop=True)
        fills_day = frames["fills"][frames["fills"]["race_date"].eq(day)].reset_index(drop=True)
        raw = B.type_raw(pd.DataFrame(world.raw[day]).assign(market_id=lambda d: d["market_id"].astype(str)))
        late = raw["t"] > B.uk_time(day, "11:00")
        raw2 = raw.copy()
        for c in B.LEVELS + ["last_traded"]:
            raw2.loc[late, c] = raw2.loc[late, c] * rng.uniform(0.5, 1.8, int(late.sum()))
        a = B.make_ctx(day, raw, runners_day, fills_day)
        b = B.make_ctx(day, raw2, _scramble(runners_day, LATER_TAGS, rng), fills_day)
        for key in ("S032", "S036", "S041", "S046"):
            pd.testing.assert_frame_equal(sel(S.DAY_FUNCS[key](b)), sel(S.DAY_FUNCS[key](a)), check_dtype=False, obj=f"{key} {day}")
        for key in ("S024", "S029"):
            pd.testing.assert_frame_equal(S.DAY_FUNCS[key](b), S.DAY_FUNCS[key](a), obj=f"{key} {day}")


def test_the_replay_places_the_same_orders_whatever_the_close(world):
    D = world.toy_day(REPLAY_DAYS[0])
    arms = {k: S.REPLAY_ARMS[k] for k in ("S001", "S019", "S037")}
    a = B.replay_day(D, arms, draws=300)
    D2 = world.toy_day(REPLAY_DAYS[0])
    D2.settled = {k: (b * 1.9, "LOSER" if st == "WINNER" else "WINNER") for k, (b, st) in D2.settled.items()}
    b = B.replay_day(D2, arms, draws=300)
    cols = ["id", "race_key", "selection", "side", "stake", "entry_price", "ts"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))
    assert not np.allclose(a["exit_price"], b["exit_price"])


# ---- the planted cases, read closely ------------------------------------------------------------------------------

def test_the_withdrawal_trade_waits_for_the_book_to_show_the_removal(built, world):
    b = built["S036"][0]
    stale = world.runners[(NR_DAY, 1, 5)]
    assert set(b["race_date"]) == {NR_DAY} and set(b["selection"]) == {str(stale["sid"])} and set(b["side"]) == {"back"}
    t = pd.to_datetime(b["ts"], utc=True).iloc[0]
    assert t >= B.uk_time(NR_DAY, "09:20") + pd.Timedelta(seconds=60) and b["cut_factor"].eq(1.0).all()


def test_no_new_money_after_a_removal_drops_only_the_later_fills(built, frames):
    base, cut = built["BASE-B"][0], built["S023"][0]
    r = frames["runners"]
    mid = r.loc[r["final_status"].eq("REMOVED") & r["market_type"].eq("WIN"), "market_id"].iloc[0]
    dropped = base.merge(cut, on=["race_date", "race_key", "selection", "ts"], how="left", indicator=True)
    dropped = dropped[dropped["_merge"].eq("left_only")]
    assert len(dropped) == 1 and set(dropped["race_key"]) == {mid} and len(cut) == len(base) - 1
    assert (pd.to_datetime(dropped["ts"], utc=True) > frames["removals"]["first_seen"].min()).all()


def test_the_free_removal_markets_and_the_stale_factor_trigger(built, frames, world):
    free = built["S024"][0]
    mid = world.runners[(FREE_DAY, 2, 6)]["win_id"]
    assert set(free["race_key"]) == {mid} and str(world.runners[(FREE_DAY, 2, 6)]["sid"]) not in set(free["selection"])
    assert mid not in set(built["S024-OTHER"][0]["race_key"])
    stale = built["S046"][0]
    j, fav = world.runners[(STALE_DAY, 2, 3)], world.runners[(STALE_DAY, 2, 0)]
    assert set(stale["race_key"]) == {j["win_id"]} and set(stale["selection"]) == {str(fav["sid"])}
    assert (stale["stake"] <= 50.0).all() and (stale["stake"] * (stale["entry_price"] - 1) <= 200.0 + 1e-9).all()


def test_the_reliability_book_moves_the_surplus_to_the_unreliable_quote():
    b = pd.DataFrame({"market_id": ["m"] * 3, "back1": [2.0, 3.0, 4.0], "lay1": [2.02, 3.03, 5.0],
                      "back1_size": [100.0, 100.0, 2.0], "lay1_size": [100.0, 100.0, 2.0], "n_rows": 3, "n_active": 3})
    x = S.reliability_book(b, d0=10.0)
    m = (1 / b["back1"] + 1 / b["lay1"]) / 2
    assert x.sum() == pytest.approx(1.0) and m.sum() > 1.0
    assert (m - x).iloc[2] > 10 * (m - x).iloc[:2].max()                 # the wide, small quote gives up the surplus
    assert S.reliability_book(b.assign(n_active=4), d0=10.0).isna().all()   # a runner missing: not a whole race


def test_the_coherent_book_lay_finds_the_short_offer_in_an_overround_book(frames, world):
    """S045 on a planted book: at 08:00 every price of one race 12% short (an overround of mids) and one runner's
    quote wide and small. The surplus goes to that quote, its coherent price is far over its lay, and it alone is
    laid."""
    day, k = "2026-10-21", world.runners[("2026-10-21", 2, 2)]
    r = frames["runners"].copy()
    race = r["market_id"].eq(k["win_id"])
    one = race & r["selection_id"].eq(k["sid"])
    for c in ("c08_back1", "c08_lay1"):
        r.loc[race, c] = r.loc[race, c] * 0.88
    r.loc[one, "c08_lay1"] = r.loc[one, "c08_back1"] * 1.30
    r.loc[one, "c08_back1_size"], r.loc[one, "c08_lay1_size"] = 3.0, 12.0
    b, days = build("S045", changed(frames, runners=r))
    assert days == [d for d in DAYS if d >= "2026-10-20"]
    assert set(b["selection"]) == {str(k["sid"])} and set(b["race_date"]) == {day} and set(b["side"]) == {"lay"}
    assert b["stake"].iloc[0] == pytest.approx(12.0) and b["entry_price"].iloc[0] <= 10.0     # never more than the size shown
    assert len(build("S045", frames)[0]) == 0                               # the toy books themselves hold no such offer


def test_the_runner_reads_the_whole_b_live_registry(frames, monkeypatch, tmp_path, capsys):
    """research/strat100/run.py end to end on the toy frames: every b_live strategy scored, no error, the pairs read
    against their benchmarks, and no strategy changed a shared frame."""
    import json
    monkeypatch.setattr(B, "load_b_live", lambda: frames)
    assert run.main(["--dataset", "b_live", "--out", str(tmp_path), "--draws", "200", "--strategy-minutes", "5"]) == 0
    res = json.loads((tmp_path / "strat100_results.json").read_text())
    got = {r["id"]: r for r in res["strategies"]}
    assert set(got) == set(BY_ID) and res["partial"] is True
    assert not [r["id"] for r in got.values() if r.get("error")]
    assert not [r["id"] for r in got.values() if "CHANGED THE SHARED FRAMES" in (r.get("note") or "")]
    assert got["S001"]["verdict"].startswith("benchmark") and got["BASE-B"]["verdict"].startswith("benchmark")
    assert got["S002"]["pair_verdict"] == "not read" and got["S015"]["pair_verdict"] == "not read"   # 17 toy days: under 30
    assert got["S011"]["confirm_ret_2pct"] is not None and got["S011"]["confirm_ret_2pct"] < got["S011"]["confirm_ret"]
    assert got["S015"]["confirm_ret_2pct"] == got["S015"]["confirm_ret"]          # real fills: no worse-price print
    daily = pd.read_csv(tmp_path / "strat100_daily.csv")
    assert set(daily["id"]) == set(BY_ID) and "DELAYED" in frames["meta"]["delayed_key"]


def test_the_files_keep_to_the_house_rules():
    for f in ("research/strat100/data_b.py", "research/strat100/strategies_b_live.py", "tests/test_strat100_b_live.py"):
        text = (ROOT / f).read_text()
        assert EM_DASH not in text, f
    src = (ROOT / "research/strat100/strategies_b_live.py").read_text()
    assert "import boto3" not in src and "sqlite3" not in src            # no source is read by the strategies module
