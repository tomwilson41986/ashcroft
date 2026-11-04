"""A vendor's player names to Cricsheet's IDs, so a live feed's players get their ratings.

For each name the vendor gives for a side, among the people who have played for that side in Cricsheet (the
candidates), in order:

1. the name is one of Cricsheet's names for a candidate (people_register name or unique_name, or people_names:
   full names and other spellings), compared without case, accents or punctuation;
2. else first initial and surname match a candidate's register name ("Virat Kohli" and "V Kohli");
3. else no data found: None, and the engine rates the player at the prior.

A name matching two candidates the same way is left unmapped (None): a wrong rating is worse than the prior.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ").replace(".", " ")).strip()


def initial_surname(name: str) -> str | None:
    parts = norm(name).split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else None


class PlayerMap:
    def __init__(self, register: pd.DataFrame, names: pd.DataFrame | None, players: pd.DataFrame,
                 matches: pd.DataFrame, since: str | None = None):
        m = matches if since is None else matches[matches.start_date >= since]
        p = players[players.match_id.isin(m.match_id)]
        self.squads = p.groupby("team").player_id.apply(lambda s: set(s.dropna())).to_dict()
        alias: dict[str, set[str]] = {}
        for col in ("name", "unique_name"):
            for pid, n in zip(register.identifier, register[col]):
                alias.setdefault(norm(n), set()).add(pid)
        if names is not None:
            for pid, n in zip(names.identifier, names.name):
                alias.setdefault(norm(n), set()).add(pid)
        self.alias = alias
        short: dict[str, set[str]] = {}
        for pid, n in zip(register.identifier, register.name):
            k = initial_surname(n)
            if k:
                short.setdefault(k, set()).add(pid)
        self.short = short

    def find(self, name: str, team: str) -> str | None:
        squad = self.squads.get(team, set())
        hit = self.alias.get(norm(name), set()) & squad
        if len(hit) == 1:
            return next(iter(hit))
        if len(hit) > 1:
            return None
        k = initial_surname(name)
        hit = (self.short.get(k, set()) & squad) if k else set()
        return next(iter(hit)) if len(hit) == 1 else None

    def xi(self, names: list[str], team: str) -> tuple[list[str | None], list[str]]:
        """IDs for a listed XI, and the names left unmapped (to report)."""
        ids = [self.find(n, team) for n in names]
        return ids, [n for n, i in zip(names, ids) if i is None]
