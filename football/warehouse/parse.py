"""Reading Betfair Historic Data stream files into the silver rows.

A file is one market's life as JSON lines of 'mcm' messages: each has a publish time (``pt``, milliseconds UTC) and,
per market, an optional ``marketDefinition`` (the market's whole definition, sent again whenever it changes) and
optional ``rc`` runner changes (``id``, ``ltp`` and, on a handicap line, ``hc``). The BASIC plan carries the last
traded price only, about once a minute a runner, before the off and in play.

``Batch`` reads files one after another and holds the rows of the five silver tables (football.warehouse.schema);
``Batch.tables()`` gives them as Arrow tables. A price is stamped with the market's state when it was published: a
definition in the same message applies first (Betfair sends the suspension and the price that caused it together).
A "price" outside Betfair's range (1.01 to 1000) is not a trade: a void market is closed with 1.0 on every runner.
Such updates are counted on the market (``n_ltp_invalid``) and kept out of silver/tick.
"""

from __future__ import annotations

import bz2
import json
from datetime import datetime, timezone
from functools import lru_cache

import pyarrow as pa

from football.warehouse.schema import SILVER, SILVER_DEFINITION, SILVER_MARKET, SILVER_RUNNER, SILVER_SPELL, SILVER_TICK

PARSER_VERSION = 2          # 2: a price outside Betfair's range is not a price (kept out of silver/tick, counted)
DRAW = "The Draw"
LTP_MIN, LTP_MAX = 1.01, 1000.0


@lru_cache(maxsize=8192)
def iso_ms(text: str | None) -> int | None:
    """An ISO time ('2026-01-29T20:00:00.000Z') as milliseconds since the epoch, UTC; None if blank or unreadable."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1000))


def split_event_name(name: str | None) -> tuple[str | None, str | None]:
    """('Home', 'Away') from 'Home v Away'; (None, None) when the name is not one match of two sides."""
    if not name:
        return None, None
    parts = name.split(" v ")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None, None
    return parts[0].strip(), parts[1].strip()


def runner_roles(market_type: str | None, runners: list[dict], home: str | None, away: str | None) -> dict:
    """{(selection id, handicap): role} for a market's runners: home, away, draw, over or under.

    Match odds: the draw by its name, the sides by their names in the event's name, else by Betfair's order (home
    first). Over/under: by the runner's name. Asian handicap: the sides by name, else the first selection id in
    Betfair's order is the home side."""
    out: dict = {}
    mt = market_type or ""
    by_order = sorted(runners, key=lambda r: (r.get("sortPriority") or 0))
    if mt == "MATCH_ODDS":
        sides = [r for r in by_order if r.get("name") != DRAW]
        for r in by_order:
            key = (int(r["id"]), float(r.get("hc") or 0.0))
            name = r.get("name")
            if name == DRAW:
                out[key] = "draw"
            elif home and name == home:
                out[key] = "home"
            elif away and name == away:
                out[key] = "away"
            elif len(sides) == 2:
                out[key] = "home" if r is sides[0] else "away"
        return out
    if mt.startswith("OVER_UNDER"):
        for r in by_order:
            name = (r.get("name") or "").lower()
            role = "over" if name.startswith("over") else "under" if name.startswith("under") else None
            if role:
                out[(int(r["id"]), float(r.get("hc") or 0.0))] = role
        return out
    if mt == "ASIAN_HANDICAP":
        ids: list[int] = []
        for r in by_order:
            if int(r["id"]) not in ids:
                ids.append(int(r["id"]))
        for r in by_order:
            sid, name = int(r["id"]), r.get("name")
            role = None
            if home and name == home:
                role = "home"
            elif away and name == away:
                role = "away"
            elif len(ids) == 2:
                role = "home" if sid == ids[0] else "away"
            if role:
                out[(sid, float(r.get("hc") or 0.0))] = role
        return out
    return out


class _Market:
    """One market's reading of a file."""

    __slots__ = ("market_id", "defn", "defs", "sel", "hc", "pt", "seq", "ltp", "ip", "st", "in_play", "status",
                 "in_play_pt", "closed_pt", "first_pt", "last_pt", "n_messages", "mt_first", "mt_last", "mt_changes",
                 "n_invalid")

    def __init__(self, market_id: str):
        self.market_id = market_id
        self.defn: dict | None = None
        self.defs: list[tuple] = []
        self.sel, self.hc, self.pt, self.seq, self.ltp, self.ip, self.st = [], [], [], [], [], [], []
        self.in_play, self.status = False, None
        self.in_play_pt = self.closed_pt = self.first_pt = self.last_pt = None
        self.n_messages = 0
        self.mt_first = self.mt_last = None
        self.mt_changes = 0
        self.n_invalid = 0

    def read(self, seq: int, pt: int, mc: dict) -> None:
        self.n_messages += 1
        if self.first_pt is None:
            self.first_pt = pt
        self.last_pt = pt
        d = mc.get("marketDefinition")
        if d is not None:
            self.defn = d
            ip, st, mt = bool(d.get("inPlay")), d.get("status"), iso_ms(d.get("marketTime"))
            if self.mt_first is None:
                self.mt_first = mt
            elif mt != self.mt_last:
                self.mt_changes += 1
            self.mt_last = mt
            if ip and self.in_play_pt is None:
                self.in_play_pt = pt
            if st == "CLOSED" and self.closed_pt is None:
                self.closed_pt = pt
            self.in_play, self.status = ip, st
            self.defs.append((seq, pt, st, ip, d.get("betDelay"), d.get("numberOfActiveRunners"), mt,
                              d.get("version"), d.get("complete"), d.get("crossMatching"),
                              iso_ms(d.get("settledTime"))))
        for rc in mc.get("rc") or ():
            p = rc.get("ltp")
            if p is None:
                continue
            if not LTP_MIN <= p <= LTP_MAX:        # e.g. 1.0 on every runner as a void market is closed: not a trade
                self.n_invalid += 1
                continue
            self.sel.append(int(rc["id"]))
            self.hc.append(float(rc.get("hc") or 0.0))
            self.pt.append(pt)
            self.seq.append(seq)
            self.ltp.append(float(p))
            self.ip.append(self.in_play)
            self.st.append(self.status)


class Batch:
    """The silver rows of the files read so far."""

    def __init__(self):
        self.cols = {s.name: {n: [] for n in s.names} for s in SILVER}
        self.failed: list[tuple[str, str]] = []
        self.files = 0

    # -- reading -----------------------------------------------------------------------------------------------

    def add_file(self, raw: bytes, plan_path: str, sha256: str) -> int:
        """Read one stream file (bz2 or plain); the number of markets it held. A file that cannot be read is
        noted in ``failed`` and adds nothing."""
        try:
            text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
            markets: dict[str, _Market] = {}
            for seq, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                msg = json.loads(line)
                pt = msg.get("pt")
                if pt is None:
                    continue
                for mc in msg.get("mc") or ():
                    mid = mc.get("id")
                    if mid is None:
                        continue
                    m = markets.get(mid)
                    if m is None:
                        m = markets[mid] = _Market(mid)
                    m.read(seq, int(pt), mc)
            held = [m for m in markets.values() if m.defn]
            if not held:
                raise ValueError("no market definition in the file")
            staged = Batch()                       # a file is all or nothing: stage its rows, then append
            for m in held:
                staged._add_market(m, plan_path, sha256)
            for name, cols in staged.cols.items():
                mine = self.cols[name]
                for k, v in cols.items():
                    mine[k].extend(v)
            self.files += 1
            return len(held)
        except Exception as exc:                     # noqa: BLE001 - a bad file must not stop the month
            self.failed.append((plan_path, f"{type(exc).__name__}: {exc}"[:300]))
            return 0

    def _add_market(self, m: _Market, plan_path: str, sha256: str) -> None:
        d = m.defn or {}
        mid, mtype = m.market_id, d.get("marketType")
        event_id = int(d["eventId"])
        home, away = split_event_name(d.get("eventName"))
        runners = d.get("runners") or []
        roles = runner_roles(mtype, runners, home, away)

        # per-runner price statistics, one pass over the ticks
        stats: dict[tuple, list] = {}
        for sel, hc, pt, ltp, ip in zip(m.sel, m.hc, m.pt, m.ltp, m.ip):
            s = stats.get((sel, hc))
            if s is None:                       # [first, last_pre, last, n, n_pre, first_pt, last_pre_pt, last_pt]
                s = stats[(sel, hc)] = [ltp, None, ltp, 0, 0, pt, None, pt]
            s[2], s[3], s[7] = ltp, s[3] + 1, pt
            if not ip:
                s[1], s[4], s[6] = ltp, s[4] + 1, pt

        n_ticks, n_pre = len(m.ltp), sum(1 for x in m.ip if not x)
        pre_pts = [pt for pt, ip in zip(m.pt, m.ip) if not ip]

        # spells: runs of unchanged (in play, status)
        spells: list[tuple] = []
        cur = None
        for seq, pt, st, ip, *_ in m.defs:
            state = (ip, st)
            if cur is None:
                cur = (pt, state)
            elif state != cur[1]:
                spells.append((cur[0], pt, cur[1]))
                cur = (pt, state)
        if cur is not None:
            spells.append((cur[0], m.last_pt, cur[1]))

        c = self.cols[SILVER_MARKET.name]
        statuses = [r.get("status") for r in runners]
        row = {
            "market_id": mid, "event_id": event_id,
            "event_type_id": int(d["eventTypeId"]) if d.get("eventTypeId") is not None else None,
            "market_type": mtype or "UNKNOWN", "market_name": d.get("name"), "event_name": d.get("eventName"),
            "home_name": home, "away_name": away, "country_code": d.get("countryCode"),
            "timezone": d.get("timezone"), "betting_type": d.get("bettingType"),
            "number_of_winners": d.get("numberOfWinners"), "market_base_rate": d.get("marketBaseRate"),
            "turn_in_play_enabled": d.get("turnInPlayEnabled"), "persistence_enabled": d.get("persistenceEnabled"),
            "bsp_market": d.get("bspMarket"), "regulators": ",".join(d.get("regulators") or []) or None,
            "market_time": m.mt_last, "market_time_first": m.mt_first, "n_market_time_changes": m.mt_changes,
            "open_date": iso_ms(d.get("openDate")), "suspend_time": iso_ms(d.get("suspendTime")),
            "settled_time": iso_ms(d.get("settledTime")), "first_pt": m.first_pt, "last_pt": m.last_pt,
            "in_play_pt": m.in_play_pt, "closed_pt": m.closed_pt,
            "first_trade_pt": m.pt[0] if m.pt else None, "last_pre_trade_pt": pre_pts[-1] if pre_pts else None,
            "last_trade_pt": m.pt[-1] if m.pt else None, "status_final": d.get("status"),
            "in_play_final": bool(d.get("inPlay")), "n_messages": m.n_messages, "n_definitions": len(m.defs),
            "n_ticks": n_ticks, "n_ticks_pre": n_pre, "n_ticks_inplay": n_ticks - n_pre,
            "n_ltp_invalid": m.n_invalid,
            "n_runners": len(runners), "n_winners": sum(1 for s in statuses if s == "WINNER"),
            "n_removed": sum(1 for s in statuses if s == "REMOVED"),
            "n_susp_inplay": sum(1 for _, _, (ip, st) in spells if ip and st == "SUSPENDED"),
            "kickoff_delay_s": (int(round((m.in_play_pt - m.mt_last) / 1000.0))
                                if m.in_play_pt is not None and m.mt_last is not None else None),
            "plan_path": plan_path, "file_sha256": sha256, "parser_version": PARSER_VERSION,
        }
        for k, v in row.items():
            c[k].append(v)

        c = self.cols[SILVER_RUNNER.name]
        seen = set()
        for r in runners:
            key = (int(r["id"]), float(r.get("hc") or 0.0))
            if key in seen:
                continue
            seen.add(key)
            self._runner_row(c, mid, event_id, mtype, key, r, roles.get(key), stats.get(key), True)
        for key in sorted(k for k in stats if k not in seen):      # traded, but not in the last definition
            self._runner_row(c, mid, event_id, mtype, key, {}, None, stats[key], False)

        c = self.cols[SILVER_DEFINITION.name]
        for seq, pt, st, ip, delay, active, mt, version, complete, cross, settled in m.defs:
            for k, v in (("market_id", mid), ("seq", seq), ("pt", pt), ("status", st), ("in_play", ip),
                         ("bet_delay", delay), ("number_of_active_runners", active), ("market_time", mt),
                         ("version", version), ("complete", complete), ("cross_matching", cross),
                         ("settled_time", settled)):
                c[k].append(v)

        c = self.cols[SILVER_SPELL.name]
        for i, (start, end, (ip, st)) in enumerate(spells):
            for k, v in (("market_id", mid), ("event_id", event_id), ("market_type", mtype or "UNKNOWN"),
                         ("spell_no", i), ("start_pt", start), ("end_pt", end),
                         ("duration_s", (end - start) / 1000.0), ("in_play", ip), ("status", st)):
                c[k].append(v)

        c = self.cols[SILVER_TICK.name]
        n = n_ticks
        if n:
            mt, ko = m.mt_last, m.in_play_pt
            c["market_id"].extend([mid] * n)
            c["event_id"].extend([event_id] * n)
            c["market_type"].extend([mtype or "UNKNOWN"] * n)
            c["selection_id"].extend(m.sel)
            c["handicap"].extend(m.hc)
            c["pt"].extend(m.pt)
            c["seq"].extend(m.seq)
            c["ltp"].extend(m.ltp)
            c["in_play"].extend(m.ip)
            c["status"].extend(m.st)
            c["t_off_s"].extend([int(round((mt - pt) / 1000.0)) for pt in m.pt] if mt is not None else [None] * n)
            c["t_inplay_s"].extend([int(round((pt - ko) / 1000.0)) if ip and ko is not None else None
                                    for pt, ip in zip(m.pt, m.ip)])

    @staticmethod
    def _runner_row(c: dict, mid: str, event_id: int, mtype, key: tuple, r: dict, role, s, in_defn: bool) -> None:
        status = r.get("status")
        row = {
            "market_id": mid, "event_id": event_id, "market_type": mtype or "UNKNOWN", "selection_id": key[0],
            "handicap": key[1], "runner_name": r.get("name"), "sort_priority": r.get("sortPriority"), "role": role,
            "status_final": status, "won": {"WINNER": 1, "LOSER": 0}.get(status),
            "ltp_first": s[0] if s else None, "ltp_last_pre": s[1] if s else None, "ltp_last": s[2] if s else None,
            "n_ticks": s[3] if s else 0, "n_ticks_pre": s[4] if s else 0,
            "first_trade_pt": s[5] if s else None, "last_pre_trade_pt": s[6] if s else None,
            "last_trade_pt": s[7] if s else None, "in_definition": in_defn,
        }
        for k, v in row.items():
            c[k].append(v)

    # -- output ------------------------------------------------------------------------------------------------

    def tables(self) -> dict[str, pa.Table]:
        """The five silver tables of the files read, each with its contract's columns and types."""
        out = {}
        for spec in SILVER:
            cols = self.cols[spec.name]
            arrays = [pa.array(cols[c.name], type=c.type) for c in spec.columns]
            out[spec.name] = pa.Table.from_arrays(arrays, schema=spec.schema)
        return out


def parse_files(items: list[tuple[str, str, bytes]]) -> tuple[dict[str, pa.Table], list[tuple[str, str]], int]:
    """(silver tables, failed files, files read) for a list of (plan path, sha256, bytes): the unit of work a
    worker process takes."""
    b = Batch()
    for plan_path, sha, raw in items:
        b.add_file(raw, plan_path, sha)
    return b.tables(), b.failed, b.files
