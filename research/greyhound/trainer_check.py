"""Is GreyhoundAPI's per-run trainer the trainer ON THE DAY? (the feature review of 17 Nov 2026, D1: "unverified").

GBGB serves each dog's CURRENT trainer on every past result (greyhound/pit/core.py ``trainer_at_the_time``). If the
API's trainer is historic, a dog that changed yard shows its old trainer on runs before the move and the new one
after; if the API serves today's trainer too, a dog's API trainer never changes across its runs fetched together.

Read-only, from the sources store: the API's GB runner tables (``greyhoundapi/runs_GB_<year>.parquet``), GBGB's year
tables (``gbgb/runs_<year>.parquet``; no fetch), the raw GBGB day files' write times (core.raw_fetch_lag) and the
daily GBGB dog records (``gbgb_dogs/<day>/dogs.jsonl.gz``, from 14 Nov 2026). The API is linked to GBGB run by run
(greyhound/pit/trainers.py ``link_runs``, the slot and the name) and its names mapped to GBGB's (``map_names``, by
the name and by first initial and surname only).

Measured:

1. ``agreement``: the share of linked runs where the API's trainer is GBGB's as served, on all days and on the days
   GBGB was read within core.FRESH_DAYS of racing (GBGB's trainer then is the trainer at the time: the direct test).
2. ``api_movers``: dogs whose API trainer changes across their runs; their spells (clean when a trainer never comes
   back after a change); the share of their runs agreeing with GBGB's latest served trainer (their current one), in
   the last API spell and in the earlier ones; the change dates (a sample).
3. ``snapshot_movers``: dogs whose trainer differs between two GBGB dog records; the API's trainer on their runs up to
   the last record with the old trainer and from the first with the new one.

The verdict (thresholds fixed here, before the run): ``historic`` when at least ``MIN_MOVERS`` dogs change API
trainer, at least ``LAST_AGREE`` of their last-spell runs agree with GBGB's current trainer and at most
``EARLY_AGREE`` of their earlier-spell runs do, and (where measured) the fresh-day agreement is at least
``FRESH_AGREE``; ``current`` when under ``CURRENT_SHARE`` of the dogs with ``MIN_RUNS`` linked runs ever change API
trainer; ``unclear`` otherwise; ``no data found`` with nothing linked.

    python -m research.greyhound.trainer_check --out out/trainer_check.json
    python -m research.greyhound.trainer_check --root <local store> --first-year 2020 --out out/trainer_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_MOVERS = 50
LAST_AGREE = 0.8
EARLY_AGREE = 0.3
FRESH_AGREE = 0.9
CURRENT_SHARE = 0.002
MIN_RUNS = 10
SAMPLE = 50
GBGB_COLS = ["race_date", "race_time", "track", "trap", "dog_id", "dog_name", "trainer"]


def gbgb_runs(store, first_year: int) -> pd.DataFrame:
    keys = sorted(k for k in store.listing("gbgb/") if re.search(r"gbgb/runs_\d{4}\.parquet$", k)
                  and int(k[-12:-8]) >= first_year)
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is not None and len(t):
            parts.append(t[[c for c in GBGB_COLS if c in t.columns]])
    if not parts:
        return pd.DataFrame(columns=GBGB_COLS)
    g = pd.concat(parts, ignore_index=True).dropna(subset=["dog_id", "race_date"])
    g["race_date"] = g.race_date.astype(str).str[:10]
    g["race_time"] = g.race_time.astype(str).str[:5]
    g["dog_id"] = g.dog_id.astype("int64")
    return g.drop_duplicates(["race_date", "race_time", "track", "trap", "dog_id"]).reset_index(drop=True)


def share(x: pd.Series) -> float | None:
    return round(float(x.mean()), 4) if len(x) else None


def api_movers(r: pd.DataFrame) -> dict:
    """``r``: linked runs (dog_id, race_date, race_time, trainer (GBGB served), api (mapped))."""
    r = r.sort_values(["dog_id", "race_date", "race_time"], kind="stable")
    nd = r.groupby("dog_id").api.nunique()
    runs_n = r.groupby("dog_id").size()
    eligible = runs_n[runs_n >= MIN_RUNS].index
    movers = nd[nd > 1].index
    m = r[r.dog_id.isin(movers)].copy()
    out = {"dogs_linked": int(nd.size), "dogs_with_min_runs": int(len(eligible)),
           "dogs_api_trainer_changes": int(len(movers)),
           "share_of_dogs_with_min_runs": round(len(set(movers) & set(eligible)) / max(len(eligible), 1), 5)}
    if not len(m):
        return out
    prev = m.groupby("dog_id").api.shift(1)
    m["new_spell"] = prev.isna() | (m.api != prev)
    m["spell"] = m.groupby("dog_id").new_spell.cumsum()
    nsp = m.groupby("dog_id").spell.max()
    out["clean_spell_share"] = round(float((nsp == nd.loc[nsp.index]).mean()), 4)
    current = r.groupby("dog_id").trainer.last()
    m["current"] = m.dog_id.map(current)
    m["agree_current"] = m.api == m.current
    last = m.spell == m.dog_id.map(nsp)
    out["last_spell_agree_current"] = share(m.agree_current[last])
    out["earlier_spells_agree_current"] = share(m.agree_current[~last])
    out["runs_in_last_spell"], out["runs_in_earlier_spells"] = int(last.sum()), int((~last).sum())
    ch = m[m.new_spell & (m.spell > 1)]
    out["change_dates_by_month"] = ch.race_date.str[:7].value_counts().sort_index().to_dict()
    sample = ch.drop_duplicates("dog_id").head(SAMPLE)
    spell_tr = m.groupby(["dog_id", "spell"]).api.first()
    out["sample"] = [{"dog_id": int(x.dog_id), "date": x.race_date, "from": spell_tr.get((x.dog_id, x.spell - 1)),
                      "to": x.api, "gbgb_current": x.current} for x in sample.itertuples()]
    return out


def snapshot_movers(store, r: pd.DataFrame) -> dict:
    from greyhound.pit.core import norm_name
    from sources import gbgb_dogs
    snap = gbgb_dogs.table(store)
    if not len(snap):
        return {"note": "no data found: no GBGB dog records"}
    snap = snap.dropna(subset=["trainer"]).sort_values(["dog_id", "day"])
    nt = snap.groupby("dog_id").trainer.nunique()
    mv = nt[nt > 1].index
    out = {"records": int(len(snap)), "dogs": int(snap.dog_id.nunique()), "days": sorted(snap.day.unique())[:1] +
           sorted(snap.day.unique())[-1:], "dogs_trainer_changed": int(len(mv))}
    rows = []
    for dog, s in snap[snap.dog_id.isin(mv)].groupby("dog_id"):
        prev = s.trainer.shift(1)
        k = s[(s.trainer != prev) & prev.notna()].iloc[-1:]
        if not len(k):
            continue
        new_day, new_tr = k.day.iloc[0], k.trainer.iloc[0]
        old = s[s.day < new_day]
        old_day, old_tr = old.day.iloc[-1], old.trainer.iloc[-1]
        d = r[r.dog_id == dog]
        b, a = d[d.race_date <= old_day], d[d.race_date >= new_day]
        nb, na = b.api.astype(str).str.lstrip("?").map(norm_name), a.api.astype(str).str.lstrip("?").map(norm_name)
        rows.append({"dog_id": int(dog), "old": old_tr, "old_day": old_day, "new": new_tr, "new_day": new_day,
                     "api_runs_before": int(len(b)), "api_old_before": int((nb == norm_name(old_tr)).sum()),
                     "api_runs_after": int(len(a)), "api_new_after": int((na == norm_name(new_tr)).sum())})
    t = pd.DataFrame(rows)
    if len(t):
        out["api_old_share_before"] = round(float(t.api_old_before.sum() / max(t.api_runs_before.sum(), 1)), 4)
        out["api_new_share_after"] = round(float(t.api_new_after.sum() / max(t.api_runs_after.sum(), 1)), 4)
        out["dogs_with_api_runs_both_sides"] = int(((t.api_runs_before > 0) & (t.api_runs_after > 0)).sum())
        out["sample"] = t.head(SAMPLE).to_dict("records")
    return out


def verdict(res: dict) -> tuple[str, list[str]]:
    why = []
    if not res.get("agreement", {}).get("runs_linked"):
        return "no data found", ["no API run linked to a GBGB run"]
    mv = res["api_movers"]
    fresh = res["agreement"].get("fresh_days_agree")
    if mv.get("share_of_dogs_with_min_runs", 0) < CURRENT_SHARE:
        why.append(f"under {CURRENT_SHARE:.1%} of dogs with {MIN_RUNS}+ linked runs ever change API trainer")
        return "current", why
    ok = (mv["dogs_api_trainer_changes"] >= MIN_MOVERS and (mv.get("last_spell_agree_current") or 0) >= LAST_AGREE
          and (mv.get("earlier_spells_agree_current") if mv.get("earlier_spells_agree_current") is not None else 1)
          <= EARLY_AGREE and (fresh is None or fresh >= FRESH_AGREE))
    why += [f"dogs changing API trainer {mv['dogs_api_trainer_changes']} (need {MIN_MOVERS})",
            f"last spell agrees with GBGB's current {mv.get('last_spell_agree_current')} (need {LAST_AGREE})",
            f"earlier spells agree {mv.get('earlier_spells_agree_current')} (need at most {EARLY_AGREE})",
            f"fresh-day agreement {fresh} (need {FRESH_AGREE} where measured)"]
    return ("historic" if ok else "unclear"), why


def run(store, first_year: int = 2014) -> dict:
    from greyhound.pit import core, trainers
    res: dict = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "first_year": first_year,
                 "thresholds": {"min_movers": MIN_MOVERS, "last_agree": LAST_AGREE, "early_agree": EARLY_AGREE,
                                "fresh_agree": FRESH_AGREE, "current_share": CURRENT_SHARE, "min_runs": MIN_RUNS}}
    api = trainers.load_context(store)["api_gb_trainer"]
    g = gbgb_runs(store, first_year)
    res["sources"] = {"api_gb_rows": 0 if api is None else int(len(api)), "gbgb_runs": int(len(g)),
                      "gbgb_from": str(g.race_date.min()) if len(g) else None,
                      "gbgb_to": str(g.race_date.max()) if len(g) else None}
    if api is None or not len(g):
        res["verdict"], res["why"] = "no data found", ["no API GB table or no GBGB year table in the store"]
        return res
    res["sources"].update({"api_from": str(api.start_local.min())[:10], "api_to": str(api.start_local.max())[:10],
                           "api_trainer_filled": share(api.trainer.notna())})
    link, ln = trainers.link_runs(api, g)
    # names only (no co-occurrence): a map learnt from GBGB's served trainer would pull the API toward agreeing with it
    mapped, mn = trainers.map_names(link.api_trainer, g.trainer.where(link.api_trainer.notna()), cooccur=False)
    res["link"], res["names"] = ln, mn
    r = g.assign(api=mapped, api_id=link.api_trainer_id)[mapped.notna()]
    lag = core.raw_fetch_lag(store)
    fresh = r.race_date.map(lag).le(core.FRESH_DAYS)
    agree = r.api == r.trainer
    res["agreement"] = {"runs_linked": int(len(r)), "all_days_agree": share(agree),
                        "fresh_days_runs": int(fresh.sum()), "fresh_days_agree": share(agree[fresh]),
                        "stale_days_agree": share(agree[~fresh]),
                        "by_year": agree.groupby(r.race_date.str[:4]).mean().round(4).to_dict()}
    res["api_movers"] = api_movers(r)
    try:
        res["snapshot_movers"] = snapshot_movers(store, r)
    except Exception as exc:                                  # the dog records are optional to the verdict
        res["snapshot_movers"] = {"note": f"not read: {type(exc).__name__}: {str(exc)[:200]}"}
    res["verdict"], res["why"] = verdict(res)
    return res


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local sources store (default S3)")
    ap.add_argument("--first-year", type=int, default=2014)
    ap.add_argument("--out", default="out/trainer_check.json")
    a = ap.parse_args(argv)
    res = run(Store(root=a.root), a.first_year)
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res.get(k) for k in ("sources", "link", "agreement", "verdict", "why")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
