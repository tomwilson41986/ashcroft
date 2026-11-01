"""Japanese (JRA) and Hong Kong (HKJC) race results through Parse, for the model on international racing (the owner's
ask, 1 Nov 2026; reports/international_racing_parse.md).

- Japan, race.netkeiba.com: ``list_races`` a date (1 credit; JRA races on Saturdays, Sundays and some Mondays), then
  ``get_race_result`` each race (1 credit): every runner's finish position, gate, horse number, sex, age, weight
  carried, jockey, trainer, time, margin, favouritism, win odds, last 600 metres, corner positions, body weight.
- Hong Kong, racing.hkjc.com: ``get_race_results`` a date and race number (1 credit; Wednesdays, Saturdays and
  Sundays, September to mid-July), then ``get_sectional_times`` (1 credit): the race header (venue, class, distance,
  going, course, rating band) and every runner's position, margin and time at each section.

Raw responses go to ``sources/parse/raw/<source>/<endpoint>/...`` through ``sources.parse_odds.save`` (fetch time in
UTC, the call, the credits). A call already held is not repeated, so a rerun only fills gaps. ``--build`` makes the
tables ``sources/parse/jra_results.parquet``, ``hk_results.parquet`` and ``hk_sectionals.parquet`` from raw. The
key comes from ``PARSE_API_KEY``; the credit floor of ``sources.parse_odds`` applies.

    python -m sources.parse_racing --jra --from 2025-11-01 --to 2026-10-31 --root /tmp/parse
    python -m sources.parse_racing --hk --from 2025-11-01 --to 2026-10-31 --root /tmp/parse
    python -m sources.parse_racing --build --root /tmp/parse
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sources.parse_odds as po  # noqa: E402
from sources.common import Store  # noqa: E402

#: The owner's subscribed copies (1 Nov 2026).
po.SCRAPERS.setdefault("netkeiba", "15804c29-ea40-4bdc-b625-92724581c494")
po.SCRAPERS.setdefault("hkjc_racing", "6c1b4761-92f4-4cda-ac70-e5220fd9b9c7")

JRA_DAYS = {5, 6, 0}          # Saturday, Sunday, Monday (holiday Mondays race; others return no meetings)
HK_DAYS = {2, 5, 6}           # Wednesday, Saturday, Sunday
HK_MAX_RACES = 11


def days(d_from: date, d_to: date, weekdays: set[int]) -> list[date]:
    out, d = [], d_from
    while d <= d_to:
        if d.weekday() in weekdays:
            out.append(d)
        d += timedelta(days=1)
    return out


def held_tags(store: Store, source: str, endpoint: str) -> set[str]:
    out = set()
    for key in store.listing(f"{po.SOURCE}/raw/{source}/{endpoint}/"):
        m = re.search(r"/\d{6,12}_(.+)\.json\.gz$", key)
        if m:
            out.add(m.group(1))
    return out


def _pool(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------------------------------- Japan

def jra(store: Store, p: po.Parse, d_from: date, d_to: date, workers: int = 6) -> dict:
    held = held_tags(store, "netkeiba", "list_races")
    todo = [d for d in days(d_from, d_to, JRA_DAYS) if f"d{d:%Y%m%d}" not in held]

    def list_day(d):
        try:
            po.save(store, p, "netkeiba", "list_races", {"date": d.isoformat()}, f"d{d:%Y%m%d}")
            return 1
        except po.OutOfCredits:
            return 0

    listed = sum(_pool(list_day, todo, workers))
    races = [r["race_id"] for r in table_jra_races(store).to_dict("records") if r.get("has_result")]
    held_r = held_tags(store, "netkeiba", "get_race_result")
    todo_r = [r for r in races if f"r{r}" not in held_r]

    def result(rid):
        try:
            po.save(store, p, "netkeiba", "get_race_result", {"race_id": rid}, f"r{rid}")
            return 1
        except po.OutOfCredits:
            return 0

    got = sum(_pool(result, todo_r, workers))
    return {"days_listed": listed, "races_with_result": len(races), "results_fetched": got}


def table_jra_races(store: Store) -> pd.DataFrame:
    rows = []
    for env in po.raw_files(store, "netkeiba", "list_races"):
        d = po._data(env) or {}
        for m in d.get("meetings") or []:
            for r in m.get("races") or []:
                rows.append({"race_date": d.get("date"), "kaisai_id": m.get("kaisai_id"), "meeting": m.get("meeting"),
                             "venue": m.get("venue"), "weather_code": m.get("weather_code"),
                             "turf_condition": m.get("turf_condition"), "dirt_condition": m.get("dirt_condition"),
                             **{k: r.get(k) for k in ("race_id", "race_number", "race_name", "grade", "grade_code",
                                                      "start_time", "surface", "distance_m", "headcount",
                                                      "has_result")}})
    df = pd.DataFrame(rows)
    return df.drop_duplicates("race_id", keep="last").reset_index(drop=True) if not df.empty else df


def table_jra_results(store: Store) -> pd.DataFrame:
    races = table_jra_races(store)
    head = races.set_index("race_id") if not races.empty else pd.DataFrame()
    rows = []
    for env in po.raw_files(store, "netkeiba", "get_race_result"):
        d = po._data(env) or {}
        rid = d.get("race_id")
        h = head.loc[rid].to_dict() if rid in getattr(head, "index", []) else {}
        base = {"race_id": rid, "race_date": h.get("race_date"), "venue": d.get("venue") or h.get("venue"),
                "meeting": d.get("meeting"), "race_number": d.get("race_number"), "race_name": d.get("race_name"),
                "grade": d.get("grade"), "start_time_jst": d.get("start_time"), "surface": d.get("surface"),
                "distance_m": d.get("distance_m"), "race_class": "|".join(d.get("race_class") or []),
                "headcount": d.get("headcount"), "weather_code": h.get("weather_code"),
                "going": h.get("turf_condition") if d.get("surface") == "turf" else h.get("dirt_condition"),
                "prize_1st_man_yen": (d.get("prize_money_man_yen") or [None])[0]}
        for r in d.get("results") or []:
            rows.append({**base, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(["race_id", "horse_id"], keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- Hong Kong

def hk(store: Store, p: po.Parse, d_from: date, d_to: date, workers: int = 6) -> dict:
    """Race 1 of every Wednesday, Saturday and Sunday finds the meetings; then races 2..11 of each meeting until one
    comes back empty; then the sectionals of every race with finishers."""
    held = held_tags(store, "hkjc_racing", "get_race_results")

    def res(d, n):
        tag = f"d{d:%Y%m%d}_r{n}"
        if tag in held:
            return None
        try:
            return po.save(store, p, "hkjc_racing", "get_race_results", {"date": f"{d:%Y%m%d}", "race_no": n}, tag)
        except po.OutOfCredits:
            return None

    probe = days(d_from, d_to, HK_DAYS)
    _pool(lambda d: res(d, 1), probe, workers)
    meet_days = sorted({r["date"] for r in _hk_raw(store) if r["race_no"] == 1 and r["finishers"]})

    def rest(dstr):
        d = date(int(dstr[:4]), int(dstr[4:6]), int(dstr[6:]))
        for n in range(2, HK_MAX_RACES + 2):
            env = res(d, n)
            if env is None:
                if f"d{dstr}_r{n}" not in held:     # out of credits
                    return
                continue
            if not (po._data(env) or {}).get("finishers"):
                return

    _pool(rest, meet_days, workers)
    races = sorted({(r["date"], r["race_no"]) for r in _hk_raw(store) if r["finishers"]})
    held_s = held_tags(store, "hkjc_racing", "get_sectional_times")

    def sec(k):
        dstr, n = k
        tag = f"d{dstr}_r{n}"
        if tag in held_s:
            return 0
        try:
            po.save(store, p, "hkjc_racing", "get_sectional_times", {"date": dstr, "race_no": n}, tag)
            return 1
        except po.OutOfCredits:
            return 0

    got = sum(_pool(sec, races, workers))
    return {"days_probed": len(probe), "meetings": len(meet_days), "races": len(races), "sectionals_fetched": got}


def _hk_raw(store: Store) -> list[dict]:
    out = []
    for env in po.raw_files(store, "hkjc_racing", "get_race_results"):
        d = po._data(env) or {}
        par = env.get("params") or {}
        out.append({"date": str(par.get("date")), "race_no": int(par.get("race_no") or 0),
                    "finishers": d.get("finishers") or [], "data": d})
    return out


def _hk_sections(store: Store) -> dict[tuple[str, int], dict]:
    out = {}
    for env in po.raw_files(store, "hkjc_racing", "get_sectional_times"):
        d = po._data(env)
        if d:
            out[(str(d.get("date")), int(d.get("race_no") or 0))] = d
    return out


def table_hk_results(store: Store) -> pd.DataFrame:
    sec = _hk_sections(store)
    rows = []
    for r in _hk_raw(store):
        if not r["finishers"]:
            continue
        h = sec.get((r["date"], r["race_no"]), {})
        base = {"race_date": f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}", "race_no": r["race_no"],
                "venue": h.get("venue"), "race_name": h.get("race_name"), "race_class": h.get("class"),
                "distance_m": h.get("distance"), "rating_band": h.get("rating_band"), "surface": h.get("surface"),
                "course": h.get("course"), "going": h.get("going"),
                "race_times": json.dumps(h.get("race_times")) if h else None}
        for f in r["finishers"]:
            rows.append({**base, **f})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(["race_date", "race_no", "horse_id"], keep="last").reset_index(drop=True)


def table_hk_sectionals(store: Store) -> pd.DataFrame:
    rows = []
    for (dstr, n), d in _hk_sections(store).items():
        for ru in d.get("runners") or []:
            for s in ru.get("sections") or []:
                rows.append({"race_date": f"{dstr[:4]}-{dstr[4:6]}-{dstr[6:]}", "race_no": n,
                             "horse_id": ru.get("horse_id"), "horse_name": ru.get("horse_name"),
                             "finishing_order": ru.get("finishing_order"), "section": s.get("section"),
                             "position": s.get("position"), "margin": s.get("margin"), "time": s.get("time"),
                             "splits": json.dumps(s.get("splits") or [])})
    return pd.DataFrame(rows)


TABLES = {"jra_races": table_jra_races, "jra_results": table_jra_results, "hk_results": table_hk_results,
          "hk_sectionals": table_hk_sectionals}


def build(store: Store) -> dict:
    counts = {}
    for name, fn in TABLES.items():
        df = fn(store)
        counts[name] = len(df)
        if not df.empty:
            store.put_parquet(f"{po.SOURCE}/{name}.parquet", df)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--jra", action="store_true")
    ap.add_argument("--hk", action="store_true")
    ap.add_argument("--from", dest="d_from", default=None)
    ap.add_argument("--to", dest="d_to", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    store = Store(root=a.root) if a.root else Store()
    report: dict = {}
    if a.jra or a.hk:
        p = po.Parse()
        d_to = date.fromisoformat(a.d_to) if a.d_to else date.today() - timedelta(days=1)
        d_from = date.fromisoformat(a.d_from) if a.d_from else d_to - timedelta(days=365)
        if a.jra:
            report["jra"] = jra(store, p, d_from, d_to, a.workers)
        if a.hk:
            report["hk"] = hk(store, p, d_from, d_to, a.workers)
        report["credits_spent"] = p.spent
        report["credits_remaining"] = p.remaining
    if a.build:
        report["tables"] = build(store)
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
