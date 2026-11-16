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

Three things Betfair's files do that a reader must not take at face value (each measured on the 2026 files):

- **The kick-off time is rewritten at the kick-off.** ``marketTime`` is the schedule until the market turns in
  play; on that very message Betfair often replaces it with the actual kick-off (14,902 markets of 165,734). The
  schedule a trader knew is the one in force before the turn: ``market_time_pre``, with ``schedule_set_pt``, the
  moment it was last changed. Everything measured "before the off" is measured from it.
- **The in-play flag can be wrong.** A market is sometimes turned in play by mistake a day early and turned back,
  or drops out of play for a few minutes in mid-match (56 markets). The turn that is the kick-off is the one
  nearest the kick-off time in force when it happened (``in_play_pt``); a price is in play from that message on,
  whatever the flag said in between.
- **Some markets never turn in play** and still trade while the match is played (2,597 match-odds markets). Their
  pre-off life ends at the scheduled kick-off: ``off_pt`` is the turn in play where there is one, else
  ``market_time_pre``, and a price is ``pre_off`` only before it.
"""

from __future__ import annotations

import bz2
import json
from datetime import datetime, timezone
from functools import lru_cache

import pyarrow as pa

from football.warehouse.schema import SILVER, SILVER_DEFINITION, SILVER_MARKET, SILVER_RUNNER, SILVER_SPELL, SILVER_TICK

#: 2: a price outside Betfair's range is not a price (kept out of silver/tick, counted)
#: 3: the schedule as known before the off, the turn in play that is the kick-off, the off of a market that never
#:    turned in play; a file with a field of the wrong kind fails alone
PARSER_VERSION = 3
PT_MAX = 4_102_444_800_000       # 1 Jan 2100: a publish time beyond it is not a time
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


def _price(v) -> float | None:
    """A price as a float; None for a blank, a non-number ('NaN', 'Infinity') or a value outside Betfair's range."""
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    return p if LTP_MIN <= p <= LTP_MAX else None


def _whole(v, lo: int = -(2 ** 15), hi: int = 2 ** 15 - 1) -> int | None:
    """A whole number within [lo, hi]; None for anything else (a field of the wrong kind is a blank, not a crash)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return None
    return int(v) if float(v) == int(v) and lo <= v <= hi else None


def _text(v) -> str | None:
    return v if isinstance(v, str) else None if v is None else str(v)


def _number(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def _id(v, lo: int = -(2 ** 31), hi: int = 2 ** 31 - 1) -> int | None:
    """An id Betfair writes as a number or as its digits in text ('1')."""
    if isinstance(v, str) and v.lstrip("-").isdigit():
        v = int(v)
    return _whole(v, lo, hi)


def _flag(v) -> bool | None:
    return v if isinstance(v, bool) else None


def kick_off_turn(turns: list[tuple]) -> tuple | None:
    """Of a market's turns in play (seq, pt, the kick-off time in force in that message), the one that is the
    kick-off: the nearest in time to the kick-off time it was made under (the earliest of equals). A turn made a
    day early by mistake is far from its schedule; a second turn after a mid-match glitch is an hour from it."""
    if not turns:
        return None
    return min(turns, key=lambda t: (abs(t[1] - t[2]) if t[2] is not None else float("inf"), t[0]))


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

    __slots__ = ("market_id", "defn", "defs", "sel", "hc", "pt", "seq", "ltp", "st", "in_play", "status",
                 "turns", "closed_pt", "closed_seq", "first_pt", "last_pt", "n_messages", "mt_first", "mt_last",
                 "mt_changes", "n_invalid")

    def __init__(self, market_id: str):
        self.market_id = market_id
        self.defn: dict | None = None
        self.defs: list[tuple] = []
        self.sel, self.hc, self.pt, self.seq, self.ltp, self.st = [], [], [], [], [], []
        self.in_play, self.status = False, None
        self.turns: list[tuple] = []                 # (seq, pt, kick-off time in force): each turn in play
        self.closed_pt = self.closed_seq = self.first_pt = self.last_pt = None
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
            if ip and not self.in_play:
                self.turns.append((seq, pt, mt))
            if st == "CLOSED" and self.closed_pt is None:
                self.closed_pt, self.closed_seq = pt, seq
            self.in_play, self.status = ip, _text(st)
            self.defs.append((seq, pt, _text(st), ip, _whole(d.get("betDelay")),
                              _whole(d.get("numberOfActiveRunners")), mt,
                              _whole(d.get("version"), -2 ** 63, 2 ** 63 - 1), _flag(d.get("complete")),
                              _flag(d.get("crossMatching")), iso_ms(d.get("settledTime"))))
        for rc in mc.get("rc") or ():
            p = rc.get("ltp")
            if p is None:
                continue
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not LTP_MIN <= p <= LTP_MAX:
                self.n_invalid += 1                # e.g. 1.0 on every runner as a void market is closed: not a trade
                continue
            self.sel.append(int(rc["id"]))
            self.hc.append(float(rc.get("hc") or 0.0))
            self.pt.append(pt)
            self.seq.append(seq)
            self.ltp.append(float(p))
            self.st.append(self.status)

    def clock(self) -> dict:
        """The market's clock, read once the whole file is in: the turn in play that is the kick-off, the schedule
        as it stood before the off and when it was last changed, and the off."""
        turn = kick_off_turn(self.turns)
        if turn is not None:
            end = turn[0]                                           # definitions before the kick-off's message
        elif self.closed_seq is not None:
            end = self.closed_seq
        else:
            end = None
        before = [d for d in self.defs if end is None or d[0] < end] or self.defs[:1]
        mt_pre, set_pt = None, None
        for d in reversed(before):                                  # the last run of one kick-off time
            if mt_pre is None:
                mt_pre, set_pt = d[6], d[1]
            elif d[6] == mt_pre:
                set_pt = d[1]
            else:
                break
        in_play_pt = turn[1] if turn else None
        return {"turn_seq": turn[0] if turn else None, "in_play_pt": in_play_pt, "market_time_pre": mt_pre,
                "schedule_set_pt": set_pt if mt_pre is not None else None,
                "off_pt": in_play_pt if in_play_pt is not None else mt_pre}


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
                if _whole(pt, 1, PT_MAX) is None:
                    raise ValueError(f"a publish time that is not a time: {pt!r}")
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
            if len(held) < len(markets):             # prices that belong to nothing: say so, do not drop them
                lost = sorted(k for k, m in markets.items() if not m.defn)
                raise ValueError(f"markets with prices and no definition: {lost[:3]}")
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
        clock = m.clock()
        turn_seq, in_play_pt, off_pt = clock["turn_seq"], clock["in_play_pt"], clock["off_pt"]
        mt_pre = clock["market_time_pre"]
        # a price is in play from the kick-off's message on; before the off until then (a market that never turned
        # in play: until its scheduled kick-off)
        if turn_seq is not None:
            in_play = [seq >= turn_seq for seq in m.seq]
            pre_off = [not x for x in in_play]
        else:
            in_play = [False] * len(m.seq)
            pre_off = [off_pt is None or pt < off_pt for pt in m.pt]

        # per-runner price statistics, one pass over the ticks
        stats: dict[tuple, list] = {}
        for sel, hc, pt, ltp, pre in zip(m.sel, m.hc, m.pt, m.ltp, pre_off):
            s = stats.get((sel, hc))
            if s is None:                       # [first, last_pre, last, n, n_pre, first_pt, last_pre_pt, last_pt]
                s = stats[(sel, hc)] = [ltp, None, ltp, 0, 0, pt, None, pt]
            s[2], s[3], s[7] = ltp, s[3] + 1, pt
            if pre:
                s[1], s[4], s[6] = ltp, s[4] + 1, pt

        n_ticks, n_pre, n_in = len(m.ltp), sum(pre_off), sum(in_play)
        pre_pts = [pt for pt, pre in zip(m.pt, pre_off) if pre]

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
            "event_type_id": _id(d.get("eventTypeId")),
            "market_type": _text(mtype) or "UNKNOWN", "market_name": _text(d.get("name")),
            "event_name": _text(d.get("eventName")),
            "home_name": home, "away_name": away, "country_code": _text(d.get("countryCode")),
            "timezone": _text(d.get("timezone")), "betting_type": _text(d.get("bettingType")),
            "number_of_winners": _whole(d.get("numberOfWinners")),
            "market_base_rate": _number(d.get("marketBaseRate")),
            "turn_in_play_enabled": _flag(d.get("turnInPlayEnabled")),
            "persistence_enabled": _flag(d.get("persistenceEnabled")),
            "bsp_market": _flag(d.get("bspMarket")),
            "regulators": ",".join(str(x) for x in d.get("regulators") or []) or None,
            "market_time": m.mt_last, "market_time_first": m.mt_first, "market_time_pre": mt_pre,
            "schedule_set_pt": clock["schedule_set_pt"], "n_market_time_changes": min(m.mt_changes, 2 ** 15 - 1),
            "open_date": iso_ms(d.get("openDate")), "suspend_time": iso_ms(d.get("suspendTime")),
            "settled_time": iso_ms(d.get("settledTime")), "first_pt": m.first_pt, "last_pt": m.last_pt,
            "in_play_pt": in_play_pt, "n_inplay_turns": min(len(m.turns), 2 ** 15 - 1), "off_pt": off_pt,
            "closed_pt": m.closed_pt,
            "first_trade_pt": m.pt[0] if m.pt else None, "last_pre_trade_pt": pre_pts[-1] if pre_pts else None,
            "last_trade_pt": m.pt[-1] if m.pt else None, "status_final": _text(d.get("status")),
            "in_play_final": bool(d.get("inPlay")), "n_messages": m.n_messages, "n_definitions": len(m.defs),
            "n_ticks": n_ticks, "n_ticks_pre": n_pre, "n_ticks_inplay": n_in,
            "n_ltp_invalid": m.n_invalid,
            "n_runners": len(runners), "n_winners": sum(1 for s in statuses if s == "WINNER"),
            "n_removed": sum(1 for s in statuses if s == "REMOVED"),
            "n_susp_inplay": min(sum(1 for _, _, (ip, st) in spells if ip and st == "SUSPENDED"), 2 ** 15 - 1),
            "kickoff_delay_s": (int(round((in_play_pt - mt_pre) / 1000.0))
                                if in_play_pt is not None and mt_pre is not None
                                and abs(in_play_pt - mt_pre) < 2 ** 31 * 1000 else None),
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
            for k, v in (("market_id", mid), ("event_id", event_id), ("market_type", _text(mtype) or "UNKNOWN"),
                         ("spell_no", i), ("start_pt", start), ("end_pt", end),
                         ("duration_s", (end - start) / 1000.0), ("in_play", ip), ("status", st)):
                c[k].append(v)

        c = self.cols[SILVER_TICK.name]
        n = n_ticks
        if n:
            c["market_id"].extend([mid] * n)
            c["event_id"].extend([event_id] * n)
            c["market_type"].extend([_text(mtype) or "UNKNOWN"] * n)
            c["selection_id"].extend(m.sel)
            c["handicap"].extend(m.hc)
            c["pt"].extend(m.pt)
            c["seq"].extend(m.seq)
            c["ltp"].extend(m.ltp)
            c["in_play"].extend(in_play)
            c["pre_off"].extend(pre_off)
            c["status"].extend(m.st)
            c["t_off_s"].extend([_whole(round((mt_pre - pt) / 1000.0), -2 ** 31, 2 ** 31 - 1) for pt in m.pt]
                                if mt_pre is not None else [None] * n)
            c["t_inplay_s"].extend([_whole(round((pt - in_play_pt) / 1000.0), -2 ** 31, 2 ** 31 - 1) if ip else None
                                    for pt, ip in zip(m.pt, in_play)])

    @staticmethod
    def _runner_row(c: dict, mid: str, event_id: int, mtype, key: tuple, r: dict, role, s, in_defn: bool) -> None:
        status = r.get("status")
        row = {
            "market_id": mid, "event_id": event_id, "market_type": _text(mtype) or "UNKNOWN", "selection_id": key[0],
            "handicap": key[1], "runner_name": _text(r.get("name")), "sort_priority": _whole(r.get("sortPriority")),
            "role": role, "status_final": _text(status),
            "won": {"WINNER": 1, "LOSER": 0}.get(status) if isinstance(status, str) else None,
            "bsp": _price(r.get("bsp")),
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
