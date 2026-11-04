"""The live pricing engine: feed events in, calibrated Match Odds probabilities and the runs line out.

    engine = Engine(load_bundle(store), paths=1000)
    for ev in feed.events():
        engine.on_event(ev)
        if ev.kind == "delivery":
            quote = engine.price()

The state is cricket/state/match_state.py's, stepped one delivery at a time; the simulator's batch is built from it
with the production bundle's current ratings (cricket/model/production.py). Who faces the coming ball, until the
feed says: the batters swap ends on odd runs run and at the end of each over; after a wicket the next listed
player of the XI not yet in takes strike; a new over's bowler is the bowling XI's average until the first ball of
the over names them. Men's 20-over T20 only, as the model was trained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cricket.model import simulator as sim
from cricket.model import calibration as calib
from cricket.model.production import baseline_p
from cricket.state.match_state import State, overs_to_balls, step


@dataclass
class _Innings:
    number: int
    team: str
    state: State
    lags: deque = field(default_factory=lambda: deque([(0, 0)] * 6, maxlen=6))
    bsw: int = 0
    seen: set = field(default_factory=set)
    striker: str | None = None
    non: str | None = None
    bowler: str | None = None          # None at the start of an over: the XI average until the ball names them


class Engine:
    def __init__(self, bundle: dict, paths: int = 1000, seed: int = 0):
        self.b = bundle["bundle"]
        self.booster = bundle["booster"]
        self.prior = np.asarray(self.b["prior"])
        self.pv = sim.prior_vector(self.prior)
        self.bat = bundle["bat_ratings"].set_index("player_id")
        self.bowl = bundle["bowl_ratings"].set_index("player_id")
        self.venues = bundle["venues"].set_index("venue_id")
        self.comps = bundle["comps"].set_index("event_name")
        self.ex_v = np.asarray(self.b["extras_runs"]["values"])
        self.ex_p = np.asarray(self.b["extras_runs"]["p"])
        self.paths = paths
        self.seed = seed
        self.setup: dict | None = None
        self.inn: _Innings | None = None
        self.first_total: int | None = None
        self.n_priced = 0

    # ----------------------------------------------------------------------------------------------------------
    def _vec(self, table, pid) -> np.ndarray:
        if pid is None or pid not in table.index:
            return self.pv.copy()
        r = table.loc[pid]
        return np.r_[[r[f"p{k}"] for k in range(8)], r["n_eff"]].astype(float)

    def _vecs(self, table, pids) -> np.ndarray:
        return np.array([self._vec(table, p) for p in pids]).reshape(len(pids), 9)

    def _ctx(self) -> np.ndarray:
        pop = self.b["population"]
        vid, ev = self.setup.get("venue_id"), self.setup.get("event_name")
        v = self.venues.loc[vid] if vid in self.venues.index else None
        c = self.comps.loc[ev] if ev in self.comps.index else None
        return np.array([[v.venue_rpb if v is not None else pop["rpb"], v.venue_wpb if v is not None else pop["wpb"],
                          c.comp_rpb if c is not None else pop["rpb"]]])

    def _other(self, team: str) -> str:
        return [t for t in self.setup["teams"] if t != team][0]

    # ----------------------------------------------------------------------------------------------------------
    def on_event(self, ev) -> None:
        p = ev.payload
        if ev.kind == "setup":
            if p.get("overs") != 20 or (p.get("balls_per_over") or 6) != 6:
                raise ValueError("the model prices men's 20-over, six-ball T20 only")
            self.setup = p
        elif ev.kind == "innings":
            if self.inn is not None and self.inn.number == 1:
                self.first_total = self.inn.state.runs
            n = int(p["innings"])
            max_balls = 120
            target = None
            if n == 2:
                target = int(p["target_runs"]) if p.get("target_runs") else (self.first_total or 0) + 1
                revised = overs_to_balls(p.get("target_overs")) if p.get("target_overs") else None
                max_balls = revised or 120
            xi = self.setup["xi"][p["team"]]
            self.inn = _Innings(n, p["team"], State("T20", n, max_balls, target), striker=xi[0],
                                non=xi[1] if len(xi) > 1 else None)
        elif ev.kind == "delivery":
            self._delivery(p)

    def _delivery(self, d: dict) -> None:
        i = self.inn
        i.striker, i.non, i.bowler = d["batter_id"], d["non_striker_id"], d["bowler_id"]
        i.seen.update({d["batter_id"], d["non_striker_id"]})
        i.state = step(i.state, {"runs_total": d["runs_total"], "legal": d["legal"], "team_wickets": d["team_wickets"],
                                 "batter": d["batter_id"], "non_striker": d["non_striker_id"],
                                 "bowler": d["bowler_id"]})
        legal, out = bool(d["legal"]), int(d["team_wickets"]) > 0
        if legal:
            i.lags.append((int(d["runs_total"]), int(out)))
        i.bsw = 0 if out else i.bsw + int(legal)
        ran = int(d["runs_batter"]) + int(d.get("byes") or 0) + int(d.get("legbyes") or 0) + max(
            int(d.get("wides") or 0) - 1, 0)
        if out:
            gone = d.get("player_out_id")
            nxt = next((p for p in self.setup["xi"][i.team] if p not in i.seen), None)
            if gone == i.non:
                i.non = nxt
            else:
                i.striker = nxt
            if nxt:
                i.seen.add(nxt)
        elif ran % 2 == 1:
            i.striker, i.non = i.non, i.striker
        if legal and i.state.legal_balls % 6 == 0:
            i.striker, i.non = i.non, i.striker
            i.bowler = None

    # ----------------------------------------------------------------------------------------------------------
    def batch(self) -> sim.Batch:
        i, s = self.inn, self.inn.state
        xi = self.setup["xi"][i.team]
        bxi = self.setup["xi"][self._other(i.team)]
        team_bowl = sim.team_bowling(self._vecs(self.bowl, bxi), self.prior)
        todo = [p for p in xi if p not in i.seen and p not in (i.striker, i.non)]
        lag = np.array(i.lags, dtype=float)
        kw = {}
        if i.number == 1:
            chase_bat = self._vecs(self.bat, bxi)
            kw = {"open2": sim.fill_queue(chase_bat[:2], self.prior, 2)[None],
                  "queue2": sim.fill_queue(chase_bat[2:], self.prior)[None],
                  "team_bowl2": sim.team_bowling(self._vecs(self.bowl, xi), self.prior)[None],
                  "max_balls2": np.array([120])}
        return sim.Batch(
            innings=np.array([i.number]), runs=np.array([s.runs]), wickets=np.array([s.wickets]),
            legal=np.array([s.legal_balls]), max_balls=np.array([s.max_balls]),
            target=np.array([float(s.target) if s.target is not None else np.nan]), bsw=np.array([i.bsw]),
            lag_r=lag[None, :, 0], lag_w=lag[None, :, 1], striker=self._vec(self.bat, i.striker)[None],
            non=self._vec(self.bat, i.non)[None],
            bowler=(self._vec(self.bowl, i.bowler) if i.bowler else team_bowl)[None],
            queue=sim.fill_queue(self._vecs(self.bat, todo), self.prior)[None], team_bowl=team_bowl[None],
            ctx=self._ctx(), **kw)

    def price(self) -> dict:
        """The quote before the coming ball: P(each side wins), raw and calibrated, and in the first innings the
        distribution of the innings total (quantiles and the full paths' mean)."""
        i, s = self.inn, self.inn.state
        self.n_priced += 1
        seed = self.seed + self.n_priced
        out = {"innings": i.number, "batting": i.team, "runs": s.runs, "wickets": s.wickets, "legal": s.legal_balls,
               "target": s.target, "paths": self.paths, "model": self.b["version"]}
        B = self.batch()
        if i.number == 1:
            p, totals = sim.match_probability(self.booster, B, self.paths, self.ex_v, self.ex_p, seed)
            p_bat = float(p[0])
            p_cal = self._calibrate("first_innings", p, {"runs_before": s.runs, "wickets_before": s.wickets,
                                                          "legal_before": s.legal_balls})
            q = np.quantile(totals[0], [0.1, 0.25, 0.5, 0.75, 0.9])
            out["innings_total"] = {"q10": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q90": q[4],
                                    "mean": float(totals[0].mean())}
            out["totals"] = totals[0]
            bat_first = i.team
        else:
            p = sim.win_probability(self.booster, B, self.paths, self.ex_v, self.ex_p, seed)
            bl = s.max_balls - s.legal_balls
            p_chase_cal = self._calibrate("chase", p, {"rrr": min(max(6.0 * (s.target - s.runs) / max(bl, 1), 0), 36),
                                                       "wkts_in_hand": 10 - s.wickets, "balls_left": bl})
            p_bat = 1.0 - float(p[0])
            p_cal = None if p_chase_cal is None else 1.0 - p_chase_cal
            bat_first = self._other(i.team)
        other = self._other(bat_first)
        out["p_raw"] = {bat_first: p_bat, other: 1.0 - p_bat}
        out["p"] = None if p_cal is None else {bat_first: p_cal, other: 1.0 - p_cal}
        return out

    def _calibrate(self, name: str, p: np.ndarray, state: dict) -> float | None:
        """The bundle's calibration layer for ``name``; None when the bundle has none (the quote is then raw only)."""
        cal = (self.b.get("calibration") or {}).get(name)
        if not cal:
            return None
        base = (self.b.get("baselines") or {}).get(name)
        p_base = np.array([baseline_p(base, state)]) if base else None
        if cal["method"] == "stack" and p_base is None:
            return None
        return float(calib.apply(cal, p, p_base)[0])

    def p_over_line(self, quote: dict, line: float) -> float | None:
        """P(the first-innings total is over ``line``) from a first-innings quote's paths (the runs-line market)."""
        t = quote.get("totals")
        return None if t is None else float((np.asarray(t) > line).mean())
