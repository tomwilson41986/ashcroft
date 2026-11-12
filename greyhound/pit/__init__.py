"""The greyhound metrics database: point-in-time tables, one per family of metrics, built from every GBGB run (2014
on) and the Betfair records, each row a dog in a race, each column known before that race's day.

The owner, 6 Nov 2026: "how could we build a database with the most advanced set of greyhound metrics in the world?
... work on all steps that are possible now". The tables (``TABLES``, built in this order, a later one may read an
earlier one):

| Table | Prefix | What |
|---|---|---|
| ids | id_ | each GBGB run's Betfair selection id (a key: never a feature) |
| ability | kf_ | Kalman levels of speed, early pace and finishing speed, with their spread |
| trouble | tr_ | the learnt price of each comment code; the dog's trouble history and clean-figure level |
| connections | cx_ | the trainer's A/E against the SP, in all and in today's situation (break, grade move, new yard) |
| market | mk_ | the dog's and the yard's record against the SP and the Betfair SP; exchange against bookmakers |
| breeding | br_ | the sire's and dam's progeny: speed, pace, finishing, A/E, stamina by distance band |
| life | lf_ | age, season, weight, trials, the grade path, track and distance experience |
| weather | wx_ | the weather at the off (NOAA ISD-Lite, the nearest station) and its learnt effect on the run-in |
| draw | dr_ | each trap's bias as a drifting state: winners against the SP, first-bend leads (step 2.3) |
| bend | bd_ | a fitted first bend: the chance of leading, the expected bend position, the win given the bend (step 2.4) |
| form | fm_ | fitted decay and shrinkage, a trend-and-age rating, the market's memory (the BFSP plan's steps 2.2, 2.5, 2.6) |
| figs | fg_ | corrected figures (own standards and shrunk variants), decayed levels shrunk by evidence, class in lengths, a market-implied rating (the BFSP plan of 10 Nov 2026) |
| xtime | xt_ | the expected time of each dog in each race: standard, meeting variant, the dog's level, its preferences, age, days off, trouble |
| dyad | dy_ | pair memory: what each rival has done to this dog beyond the ratings, and the curl of the margins |
| cycle | oc_ | a bitch's season cycle inferred from her absences (dogs as the control) |
| trials | tl_ | the trials data.clean drops: the trial figure, days since, solo or in company, a level from the trials |
| ext | nx_ | what the dogs drawn beside it do to a dog's time (a plus-minus on the neighbours' residuals), collision momentum |
| fitness | ff_ | fitness and fatigue: an impulse response over the dog's whole load history, trials included |
| fear | fr_ | the fear trace: interference tied to a place on the track, extinguished by clean passes, renewed at a new track |
| clock | ck_ | the wait in the racing kennels, the dog's habitual hour, the session, daylight, the fireworks calendar |
| optics | op_ | how much better the printed card looks than the dog ran: the look, the substance, the surplus |
| litter | lt_ | the litter as a live sensor: the littermates' current form, their level, the runt |

Each table is written to the sources store as ``<STORE_PREFIX>/<table>.parquet`` with ``<table>.meta.json``: the code
hash, the engine's code hash, when built, the rows, the date range, each column's coverage, and the table's own fit notes. The registry
(``registry()``, written to reports/greyhound_metrics_registry.md by ``--registry``) lists every column with its
definition and source. A fast-loop block ``pit_<table>`` (greyhound/blocks) joins a table onto the cached matrix by
race and dog, so each family is scored like any other block, on both targets.

    python -m greyhound.pit --build all --first-year 2014 --out out/pit       # the runner: S3
    python -m greyhound.pit --build ability,trouble --root <local store>        # a local store
    python -m greyhound.pit --registry
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TABLES = ("ids", "ability", "trouble", "connections", "market", "breeding", "life", "weather", "figs", "form", "draw",
          "bend", "xtime", "dyad", "cycle", "trials",
          # the discovery tables of 11 Nov 2026 (each reads the expected-time table)
          "ext", "fitness", "fear", "clock", "optics", "litter")
KEY = ["race_id", "dog_id"]
#: where the tables are stored. "pit2": the tables of the engine as fixed on 11 Nov 2026 (the audit's foundation
#: faults: greyhound/data.py, greyhound/metrics.py). The tables built from the engine before it stay at
#: "greyhound/pit", for the model that was trained on them: a table read by a model must be the table of the engine
#: the model was fitted on (the audit of 11 Nov 2026, finding 34: nothing tied a stored table to the engine code that
#: built it). Each table's meta now carries the engine's code hash (``engine``), and ``stale`` says when a stored
#: table was built by other engine code than the code reading it
STORE_PREFIX = "greyhound/pit2"


def module(name: str):
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}; known: {TABLES}")
    return importlib.import_module(f"greyhound.pit.{name}")


def code_hash(name: str) -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in ("core.py", f"{name}.py"):
        p = root / f
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:12]


def engine_hash() -> str:
    """The hash of the code that makes the engine's per-run figures (the fast loop's cache key)."""
    from greyhound.fastloop import code_hash as engine
    return engine()


def stale(name: str, store=None) -> bool | None:
    """True when the stored table was built by other engine code than this; None when it cannot be told (no meta, or
    a table from before the hash was recorded)."""
    from sources.common import Store
    store = store or Store()
    try:
        meta = json.loads(store.get(f"{STORE_PREFIX}/{name}.meta.json"))
    except Exception:
        return None
    return None if "engine" not in meta else meta["engine"] != engine_hash()


def coverage(tab: pd.DataFrame, runs: pd.DataFrame) -> dict:
    cols = [c for c in tab.columns if c not in KEY]
    d = tab.merge(runs[KEY + ["race_date"]], on=KEY, how="left")
    by_year = d.groupby(d.race_date.str[:4])
    return {"rows": int(len(tab)), "from": str(d.race_date.min()), "to": str(d.race_date.max()),
            "non_null": {c: round(float(tab[c].notna().mean()), 4) for c in cols},
            "non_null_by_year": {c: {y: round(float(v), 3) for y, v in by_year[c].apply(lambda s: s.notna().mean()).items()}
                                 for c in cols[:3]}}


def build_tables(names, runs: pd.DataFrame, store=None, ctx: dict | None = None, keep: bool = False) -> dict:
    """Build ``names`` in ``TABLES`` order; write each to ``store`` (when given); return each table's meta, or with
    ``keep`` the tables themselves (the live model's card build: greyhound/live.py)."""
    from greyhound.pit import core
    ctx = dict(ctx or {})
    figs = core.per_run_figures(runs)
    metas = {}
    for name in [t for t in TABLES if t in names]:
        mod = module(name)
        t0 = datetime.now(timezone.utc)
        tab, notes = mod.build(runs, figs, ctx)
        ctx[name] = tab
        meta = {"table": name, "hash": code_hash(name), "engine": engine_hash(),
                "built_utc": t0.isoformat(timespec="seconds"),
                "seconds": round((datetime.now(timezone.utc) - t0).total_seconds()), "source": mod.SOURCE,
                "feature": getattr(mod, "FEATURE", True), "card_unsafe": list(getattr(mod, "CARD_UNSAFE", ())),
                "notes": notes, **coverage(tab, runs)}
        if store is not None:
            store.put_parquet(f"{STORE_PREFIX}/{name}.parquet", tab)
            store.put(f"{STORE_PREFIX}/{name}.meta.json", json.dumps(meta, indent=1, default=str).encode())
        metas[name] = meta
    return {n: ctx[n] for n in metas} if keep else metas


def load(name: str, store=None) -> pd.DataFrame | None:
    from sources.common import Store
    store = store or Store()
    return store.get_parquet(f"{STORE_PREFIX}/{name}.parquet")


def attach(df: pd.DataFrame, name: str, store=None) -> list[str]:
    """Join table ``name`` onto ``df`` by race and dog (in place); its feature columns' names."""
    mod = module(name)
    if not getattr(mod, "FEATURE", True):
        raise ValueError(f"{name} is a key table, not features")
    tab = load(name, store)
    if tab is None:
        raise SystemExit(f"no table {name}: run python -m greyhound.pit --build {name}")
    cols = [c for c in tab.columns if c.startswith(mod.PREFIX)]
    j = df[KEY].merge(tab[KEY + cols].drop_duplicates(KEY), on=KEY, how="left")
    for c in cols:
        df[c] = j[c].to_numpy()
    return cols


def registry() -> list[dict]:
    rows = []
    for name in TABLES:
        mod = module(name)
        for col, what in mod.COLUMNS.items():
            rows.append({"table": name, "column": col, "definition": what, "source": mod.SOURCE,
                         "feature": getattr(mod, "FEATURE", True),
                         "card_unsafe": col in getattr(mod, "CARD_UNSAFE", ())})
    return rows


def registry_markdown() -> str:
    lines = ["# The greyhound metrics registry", "",
             "Every column of the point-in-time metrics tables (greyhound/pit/), generated by `python -m greyhound.pit "
             "--registry`. Each is known before the run's day; card-unsafe columns are known only shortly before the "
             "off (the weigh-in). Coverage and the scores on both targets are in each table's meta and the fast-loop "
             "reports.", "", "| Table | Column | Definition | Source | Card-safe |", "|---|---|---|---|---|"]
    for r in registry():
        lines.append(f"| {r['table']} | `{r['column']}` | {r['definition']} | {r['source']} | "
                     f"{'no' if r['card_unsafe'] else 'yes'}{'' if r['feature'] else ' (key)'} |")
    return "\n".join(lines) + "\n"
