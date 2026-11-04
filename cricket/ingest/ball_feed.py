"""The ball feed (build spec 5.3): one interface for every source of live deliveries, and the record of each event.

Every event carries the UTC time it reached us (``receipt_utc``): the feed's lag is measured from it against
Betfair's first price move after the same delivery, so the time is taken at receipt and never filled in later.
A vendor's own timestamp, where it gives one, goes in ``source_utc``; None where it gives none.

Sources:
- ``CricsheetReplay``: a held Cricsheet match replayed delivery by delivery, for testing the live path end to end.
  Its events have no source time (Cricsheet has none: no data found) and its receipt time is when the replay
  emitted them.
- a priced vendor (Sportmonks or EntitySport, build spec 3): an adapter per vendor, once one is chosen; it maps the
  vendor's player IDs to Cricsheet's through the people register (cricket/tables/people_register).

Event kinds: ``setup`` (teams, XIs as Cricsheet player IDs in listed order, venue, competition, batting first),
``innings`` (an innings starts: number, batting team), ``delivery`` (Cricsheet-shaped: batter_id, non_striker_id,
bowler_id, runs_batter, runs_total, wides, noballs, byes, legbyes, legal, team_wickets, player_out_id),
``result`` (winner or result).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Protocol

from cricket.ingest.cricsheet import parse_match


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class FeedEvent:
    kind: str
    match_id: str
    payload: dict
    receipt_utc: str = field(default_factory=utc_now)
    source_utc: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class BallFeed(Protocol):
    def events(self) -> Iterator[FeedEvent]: ...


DELIVERY_KEYS = ("innings", "over", "ball", "batter", "bowler", "non_striker", "batter_id", "bowler_id",
                 "non_striker_id", "runs_batter", "runs_extras", "runs_total", "wides", "noballs", "byes", "legbyes",
                 "penalty", "legal", "team_wickets", "wicket_kind", "player_out_id")


class CricsheetReplay:
    """A held Cricsheet match, replayed: setup, each innings, each delivery in order, the result."""

    def __init__(self, match_id: str, raw: dict, skip_super_overs: bool = True):
        self.match_id = match_id
        self.part = parse_match(match_id, raw)
        self.skip_super_overs = skip_super_overs

    def events(self) -> Iterator[FeedEvent]:
        m = self.part["matches"][0]
        xis: dict[str, list[str]] = {}
        for p in sorted(self.part["players"], key=lambda r: (r["team"], r["list_order"])):
            xis.setdefault(p["team"], []).append(p["player_id"])
        inns = [i for i in self.part["innings"] if not (self.skip_super_overs and i["super_over"])]
        bat_first = inns[0]["team"] if inns else m["team1"]
        yield FeedEvent("setup", self.match_id, {
            "match_type": m["match_type"], "overs": m["overs"], "balls_per_over": m["balls_per_over"],
            "event_name": m["event_name"], "venue": m["venue"], "venue_id": m["venue_id"],
            "teams": [m["team1"], m["team2"]], "bat_first": bat_first, "xi": xis, "start_date": m["start_date"]})
        for inn in inns:
            yield FeedEvent("innings", self.match_id, {"innings": inn["innings"], "team": inn["team"],
                                                        "target_runs": inn["target_runs"],
                                                        "target_overs": inn["target_overs"]})
            for d in self.part["deliveries"]:
                if d["innings"] == inn["innings"]:
                    yield FeedEvent("delivery", self.match_id, {k: d[k] for k in DELIVERY_KEYS})
        yield FeedEvent("result", self.match_id, {"winner": m["winner"], "result": m["result"],
                                                  "method": m["method"]})


class FeedRecorder:
    """Every event a feed hands over, kept as one JSON line with its receipt time (to a file, and to the store)."""

    def __init__(self, path, store=None, key: str | None = None):
        self.path = path
        self.store, self.key = store, key
        self.lines: list[str] = []

    def wrap(self, feed: BallFeed) -> Iterator[FeedEvent]:
        with open(self.path, "a") as fh:
            for ev in feed.events():
                line = ev.to_json()
                fh.write(line + "\n")
                fh.flush()
                self.lines.append(line)
                yield ev
        if self.store is not None and self.key:
            self.store.put(self.key, ("\n".join(self.lines) + "\n").encode())
