"""The handicap marks on GreyhoundAPI's cards against GBGB's results (the owner, 24 Nov 2026: "yes, look into getting
the actual handicap marks").

A GB handicap (class HP) starts the dogs from staggered marks: GBGB's result gives each runner's ``trapHandicap``
("R7": receives 7 metres; "Scr": scratch; greyhound/data.py ``handicap_start``), but only after the race. The model
reads proxies (trap order) because no card source held the marks: greyhounds.today, the recorder written for them,
has served no card since 7 Oct 2026 (its ``/api/content/<track>`` answers ``{"content": []}`` for every track).
GreyhoundAPI's cards, snapshotted every two hours since 12 Nov (sources/greyhoundapi.py ``snapshot_cards``, raw at
``greyhoundapi/raw/cards/<UTC day>/<HHMM>.json.gz``), carry ``runners[].handicap_metres``. This check says whether
that field is the mark, and whether it is known before the off:

1. Each card runner of a GB race, in every snapshot taken before its scheduled off: its mark, the snapshot's lead.
2. Matched to GBGB's result by UK day, dog name and trap (a dog runs once a day; no track name is needed).
3. For GBGB's handicaps: how many were on a card, how many runners had a mark before the off, how many of those
   marks equal GBGB's start, how early the mark was first there, and whether it ever changed.
4. For the other races: what the field holds (0, empty or a mark).

Descriptive, 0 trials (ledger greyhound-handicap-marks-1124). Reads S3 only; nothing reaches the API or Betfair.

    python -m research.greyhound.handicap_marks --out out/study
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-handicap-marks-1124"
PREFIX = "greyhoundapi"
UK = ZoneInfo("Europe/London")
LEADS = [(0, 1, "<1h"), (1, 3, "1-3h"), (3, 12, "3-12h"), (12, 1e9, "12h+")]


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def snapshot_time(key: str) -> datetime | None:
    m = re.search(r"cards/(\d{4}-\d{2}-\d{2})/(\d{2})(\d{2})\.json\.gz$", key)
    return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:00+00:00") if m else None


def card_rows(snap: dict, at: datetime, region: str = "GB") -> list[dict]:
    """One row a runner of each ``region`` race in one snapshot, taken before the race's scheduled off."""
    out = []
    seen = set()
    for part in (snap or {}).values():
        for r in part or []:
            if r.get("region") != region or r.get("race_id") in seen:
                continue
            start = (r.get("scheduled_start") or {}).get("utc")
            try:
                off = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            except ValueError:
                continue
            lead = (off - at).total_seconds() / 3600.0
            if lead <= 0:
                continue
            seen.add(r.get("race_id"))
            base = {"race_id": str(r.get("race_id")), "track": (r.get("track") or {}).get("name"),
                    "off_utc": off, "day": off.astimezone(UK).strftime("%Y-%m-%d"), "grade": r.get("grade"),
                    "api_is_handicap": r.get("is_handicap"), "snap": at, "lead_h": lead}
            for p in r.get("runners") or []:
                out.append({**base, "trap": _num(p.get("trap")), "dog_name": p.get("dog_name"),
                            "status": p.get("runner_status"), "mark": _num(p.get("handicap_metres")),
                            "has_field": "handicap_metres" in p})
    return out


def lead_band(h: float) -> str:
    return next(b for lo, hi, b in LEADS if lo <= h < hi)


def per_runner(cards: pd.DataFrame) -> pd.DataFrame:
    """Each card runner once: the first lead its mark was there, the last mark before the off, whether it changed."""
    c = cards.sort_values("snap")
    key = ["race_id", "trap", "dog_name"]
    has = c[c.mark.notna()]
    first = has.groupby(key).lead_h.max().rename("first_mark_lead_h")
    last = has.groupby(key).mark.last().rename("mark")
    changed = (has.groupby(key).mark.nunique() > 1).rename("mark_changed")
    base = c.groupby(key).agg(day=("day", "first"), track=("track", "first"), grade=("grade", "first"),
                              api_is_handicap=("api_is_handicap", "last"), status=("status", "last"),
                              first_seen_lead_h=("lead_h", "max"), snapshots=("snap", "size"))
    return base.join([first, last, changed]).reset_index()


def compare(runner: pd.DataFrame, gbgb: pd.DataFrame) -> dict:
    """The card's marks against GBGB's results over the days both hold."""
    days = sorted(runner.day.unique())
    g = gbgb[gbgb.race_date.isin(days)].copy()
    g["name"] = norm_name(g.dog_name)
    g["trap"] = pd.to_numeric(g.trap, errors="coerce").astype(float)
    r = runner.assign(name=norm_name(runner.dog_name.fillna("")))
    m = g.merge(r[["day", "name", "trap", "mark", "first_mark_lead_h", "mark_changed", "api_is_handicap", "status",
                   "race_id", "grade"]].rename(columns={"day": "race_date", "race_id": "api_race_id",
                                                        "grade": "api_grade"}),
                on=["race_date", "name", "trap"], how="left")
    out: dict = {"days": [days[0], days[-1]] if days else None, "card_days": len(days),
                 "gbgb_runners": int(len(g)), "gbgb_races": int(g.race_id.nunique())}
    hp = m[m.handicap]
    on_card = hp.api_race_id.notna()
    with_mark = hp.mark.notna()
    both = hp[with_mark & hp.hcp_start.notna()]
    eq = (both.mark - both.hcp_start).abs() < 1e-9
    races = hp.groupby("race_id").agg(card=("api_race_id", lambda s: s.notna().all()),
                                      marks=("mark", lambda s: s.notna().all()),
                                      track=("track", "first"))
    out["handicaps"] = {
        "races": int(hp.race_id.nunique()), "runners": int(len(hp)),
        "runners_on_a_card": int(on_card.sum()), "runners_with_a_card_mark": int(with_mark.sum()),
        "races_every_runner_on_a_card": int(races.card.sum()), "races_every_runner_with_a_mark": int(races.marks.sum()),
        "runners_compared": int(len(both)), "marks_equal": int(eq.sum()),
        "share_equal": float(eq.mean()) if len(both) else None,
        "mismatches": both.loc[~eq, ["race_date", "track", "race_time", "trap", "dog_name", "trap_handicap",
                                     "hcp_start", "mark"]].head(25).to_dict("records"),
        "mark_changed_runners": int(hp.mark_changed.fillna(False).astype(bool).sum()),
        "api_is_handicap": {str(k): int(v) for k, v in hp.api_is_handicap.astype(str).value_counts().items()},
        "first_mark_lead": {b: int(v) for b, v in hp.first_mark_lead_h.dropna().map(lead_band).value_counts().items()},
        "first_mark_lead_h_quantiles": ({str(q): float(hp.first_mark_lead_h.quantile(q))
                                         for q in (0.05, 0.25, 0.5, 0.75)} if with_mark.any() else None),
        "races_by_track": {str(k): int(v) for k, v in races.groupby("track").size().items()},
        "races_with_marks_by_track": {str(k): int(v) for k, v in races[races.marks].groupby("track").size().items()},
    }
    other = m[~m.handicap & m.api_race_id.notna()]
    out["other_races"] = {
        "runners_on_a_card": int(len(other)),
        "mark": {("empty" if pd.isna(k) else str(k)): int(v)
                 for k, v in other.mark.value_counts(dropna=False).head(12).items()},
        "api_is_handicap": {str(k): int(v) for k, v in other.api_is_handicap.astype(str).value_counts().items()},
    }
    return out


def markdown(res: dict) -> str:
    h = res["compare"]["handicaps"]
    o = res["compare"]["other_races"]
    lines = ["# GreyhoundAPI card handicap marks against GBGB's results", "",
             f"Snapshots read: {res['snapshots']} ({res['snapshot_days'][0]} to {res['snapshot_days'][1]}); GB card "
             f"runners before the off: {res['card_runners']}. GBGB days compared: {res['compare']['days']}.", "",
             "## GBGB's handicaps (class HP) on those days", "",
             f"- races {h['races']}, runners {h['runners']}",
             f"- runners on a card {h['runners_on_a_card']}, with a card mark before the off {h['runners_with_a_card_mark']}",
             f"- races with every runner on a card {h['races_every_runner_on_a_card']}, with every mark {h['races_every_runner_with_a_mark']}",
             f"- marks compared {h['runners_compared']}, equal to GBGB's start {h['marks_equal']}"
             + (f" ({h['share_equal']:.1%})" if h["share_equal"] is not None else ""),
             f"- runners whose mark changed between snapshots {h['mark_changed_runners']}",
             f"- the mark first there, hours before the off: {h['first_mark_lead']}; quantiles {h['first_mark_lead_h_quantiles']}",
             f"- the API's is_handicap on these runners: {h['api_is_handicap']}",
             f"- races by track {h['races_by_track']}; with every mark {h['races_with_marks_by_track']}", ""]
    if h["mismatches"]:
        lines += ["Mismatches (up to 25):", ""] + [f"- {x}" for x in h["mismatches"]] + [""]
    lines += ["## The other races", "", f"- runners on a card {o['runners_on_a_card']}; the field holds {o['mark']}",
              f"- the API's is_handicap {o['api_is_handicap']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    from greyhound.data import load_runs
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/handicap_marks")
    a = ap.parse_args(argv)
    store = Store()
    keys = sorted(k for k in store.listing(f"{PREFIX}/raw/cards/") if k.endswith(".json.gz"))
    rows = []
    for k in keys:
        at = snapshot_time(k)
        raw = store.get(k)
        if at is None or not raw:
            continue
        rows += card_rows(json.loads(gzip.decompress(raw)), at)
    if not rows:
        print("No data found: no GB card runner before the off in", len(keys), "snapshots")
        return 0
    cards = pd.DataFrame(rows)
    runner = per_runner(cards)
    years = sorted({int(d[:4]) for d in runner.day.unique()})
    gbgb = load_runs(store, years=years)
    res = {"tag": TAG, "snapshots": len(keys),
           "snapshot_days": [keys[0].split("/raw/cards/")[-1][:10], keys[-1].split("/raw/cards/")[-1][:10]],
           "card_runners": int(len(runner)), "compare": compare(runner, gbgb)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
