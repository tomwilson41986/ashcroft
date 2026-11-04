"""A market book as the trading layer reads it: one point in time, each runner's ladder, from the recorder's rows
(cricket/market/recorder.py) or from a raw listMarketBook reply."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from cricket.market.recorder import DEPTH


@dataclass
class Runner:
    back: list[tuple[float, float]] = field(default_factory=list)   # (price, size), best (highest) first
    lay: list[tuple[float, float]] = field(default_factory=list)    # (price, size), best (lowest) first
    ltp: float | None = None
    status: str | None = None

    @property
    def mid(self) -> float | None:
        if self.back and self.lay:
            return 2.0 / (1.0 / self.back[0][0] + 1.0 / self.lay[0][0])   # the mid in probability, as a price
        return None


@dataclass
class Book:
    market_id: str
    pt: dt.datetime
    status: str
    inplay: bool
    bet_delay: int
    runners: dict[int, Runner]

    @property
    def open(self) -> bool:
        return self.status == "OPEN"

    def back_book(self) -> float | None:
        """Sum of 1 / best back price over the runners (the book percentage on the back side)."""
        if not self.runners or any(not r.back for r in self.runners.values()):
            return None
        return sum(1.0 / r.back[0][0] for r in self.runners.values())


def _t(text: str) -> dt.datetime:
    return dt.datetime.strptime(text.rstrip("Z")[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)


def books_from_rows(df: pd.DataFrame) -> list[Book]:
    """The recorder's rows to books, in poll-time order."""
    out = []
    for (mid, pt), g in df.groupby(["market_id", "pt"], sort=False):
        runners = {}
        for r in g.itertuples():
            back = [(getattr(r, f"back{k}_p"), getattr(r, f"back{k}_s")) for k in range(1, DEPTH + 1)]
            lay = [(getattr(r, f"lay{k}_p"), getattr(r, f"lay{k}_s")) for k in range(1, DEPTH + 1)]
            runners[int(r.selection_id)] = Runner(back=[x for x in back if pd.notna(x[0])],
                                                  lay=[x for x in lay if pd.notna(x[0])],
                                                  ltp=None if pd.isna(r.ltp) else r.ltp, status=r.runner_status)
        f = g.iloc[0]
        out.append(Book(str(mid), _t(pt), f.status, str(f.inplay) in ("True", "true", "1"),
                        int(f.bet_delay) if pd.notna(f.bet_delay) else 0, runners))
    out.sort(key=lambda b: b.pt)
    return out
