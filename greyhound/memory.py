"""The residual memory, served (the owner, 17 Nov 2026: "yes serve the residual memory layer").

The layer of research/greyhound/residual_memory.py (its docstring has the idea): for a runner on day D, the dog's
misses - log(BSP) less log(our price), centred in the race - on the days before D, each weighted exp(-age in days /
TAU) (the time-decay form), give

    m = (decayed sum of misses) / (decayed count + K)          (0 with none)
    log price' = log price + B x (m less the race's mean m)     (the race's mean log price stays)
    chance'    proportional to chance x exp(-C x m), normalised in the race

The constants were fitted on the live recipe's (stack_mlsi) out-of-sample prices of 1 Jun - 31 Aug 2026 and scored on
1 Sep - 11 Nov (ledger greyhound-memory-live-1117: RMSE of log(BSP) 0.2756 to 0.2709, -0.0048, z -21.7; log-loss
-0.00047, z -4.7). They are fixed here; a refit is a new registration.

The misses are the model's OWN: each day's price before the layer (``pred_bsp_base``) against the BSP that followed.
They come from the recipe's published out-of-sample day files (``<oos prefix>/<day>.parquet``) where a day has one,
else from the served predictions (``greyhound/live/predictions/<day>.csv``). The BSP is the price files' (the
nightly load), joined by date, off time, track and trap (greyhound/parity.py ``attach_bsp``). A day with no BSP yet
adds nothing. Misses older than LOOKBACK days weigh under exp(-60/12) = 0.7% and are not read.

The forecast fits of the strategies read out-of-sample prices that must match the served ones: ``ensure_oos`` writes
the recipe's out-of-sample day files with the layer applied (walk-forward, each day from the days before it) to
``<oos prefix>_mem``, which the served predictions name. ``GREYHOUND_MEMORY=off`` serves the price without the layer.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAU = 12.0
K = 3.0
B = 1.1856
C = 1.0516
LOOKBACK = 60
TAG = "greyhound-memory-live-1117"
PRED_PREFIX = "greyhound/live/predictions"
KEYS = ["race_date", "race_time", "track", "trap"]


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_MEMORY", "on")).strip().lower() not in ("off", "0", "no", "false")


def misses(hist: pd.DataFrame) -> pd.DataFrame:
    """One row a dog-day: the day's sum and count of the dog's misses (log BSP less log base price, less the race's
    mean of it; a race counts only when every runner has both)."""
    h = hist.copy()
    ok = (pd.to_numeric(h.bsp, errors="coerce") > 1) & (pd.to_numeric(h.pred_bsp_base, errors="coerce") > 1)
    e = np.log(h.bsp.where(ok).astype(float)) - np.log(h.pred_bsp_base.where(ok).astype(float))
    whole = ok.groupby(h.race_id).transform("all")
    h["miss"] = (e - e.groupby(h.race_id).transform("mean")).where(whole)
    h = h[h.miss.notna() & h.dog_id.notna()]
    h["dog_id"] = pd.to_numeric(h.dog_id, errors="coerce")
    return h.groupby(["dog_id", "race_date"], as_index=False).agg(s=("miss", "sum"), n=("miss", "count"))


def memory_on(day_misses: pd.DataFrame, day: date) -> pd.DataFrame:
    """Each dog's memory on ``day`` from its misses on days strictly before it: dog_id, m, n (decayed count)."""
    d = day_misses[pd.to_datetime(day_misses.race_date) < pd.Timestamp(day)]
    if not len(d):
        return pd.DataFrame(columns=["dog_id", "m", "n"])
    age = (pd.Timestamp(day) - pd.to_datetime(d.race_date)).dt.days.to_numpy(float)
    w = np.exp(-age / TAU)
    g = pd.DataFrame({"dog_id": d.dog_id.to_numpy(), "s": d.s.to_numpy(float) * w, "n": d.n.to_numpy(float) * w}) \
        .groupby("dog_id", as_index=False).sum()
    g["m"] = g.s / (g.n + K)
    return g[["dog_id", "m", "n"]]


def apply(pred: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """``pred`` (race_id, dog_id, p_model, pred_bsp) with the layer: the base price and chance kept as
    ``pred_bsp_base`` and ``p_model_base``, the memory as ``memory_m``."""
    out = pred.copy()
    did = pd.to_numeric(out.dog_id, errors="coerce")
    m = did.map(mem.set_index("dog_id").m) if len(mem) else pd.Series(np.nan, index=out.index)
    m = m.fillna(0.0).to_numpy(float)
    race = out.race_id.to_numpy()
    out["pred_bsp_base"], out["p_model_base"], out["memory_m"] = out.pred_bsp, out.p_model, np.round(m, 5)
    z = np.log(np.clip(out.p_model.to_numpy(float), 1e-12, 1.0)) - C * m
    s = pd.Series(np.exp(z - pd.Series(z).groupby(race).transform("max").to_numpy()))
    out["p_model"] = (s / s.groupby(race).transform("sum")).to_numpy()
    mc = m - pd.Series(m).groupby(race).transform("mean").to_numpy()
    ok = pd.to_numeric(out.pred_bsp, errors="coerce") > 1
    out["pred_bsp"] = np.where(ok, np.exp(np.log(out.pred_bsp.where(ok).astype(float)) + B * mc), out.pred_bsp)
    return out


def _with_bsp(frame: pd.DataFrame, prices: pd.DataFrame | None) -> pd.DataFrame:
    from greyhound.parity import attach_bsp
    f = frame.copy()
    f["trap"] = pd.to_numeric(f.trap, errors="coerce").astype("Int64")
    f = f.dropna(subset=["trap"]).assign(trap=lambda x: x.trap.astype("int64"),
                                          race_date=lambda x: x.race_date.astype(str),
                                          race_time=lambda x: x.race_time.astype(str))
    return attach_bsp(f, prices)


def history(store, day: date, oos_prefix: str | None, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """The base prices of the LOOKBACK days before ``day`` with their BSP: the recipe's out-of-sample day file where
    there is one, else the served predictions (their pre-layer price)."""
    if prices is None:
        from greyhound.model import load_prices
        prices = load_prices(store, f"{(day - timedelta(days=LOOKBACK)).year}-{day.year}")
    parts = []
    for i in range(LOOKBACK, 0, -1):
        d = day - timedelta(days=i)
        ds = f"{d:%Y-%m-%d}"
        f = store.get_parquet(f"{oos_prefix.rstrip('/')}/{ds}.parquet") if oos_prefix else None
        if f is None or not len(f):
            raw = store.get(f"{PRED_PREFIX}/{ds}.csv")
            if not raw:
                continue
            f = pd.read_csv(io.BytesIO(raw))
        base = "pred_bsp_base" if "pred_bsp_base" in f else "pred_bsp"
        cols = [c for c in ("race_id", "dog_id", *KEYS) if c in f]
        if len(cols) < 6 or base not in f:
            continue
        parts.append(f[cols + [base]].rename(columns={base: "pred_bsp_base"}).assign(race_date=ds))
    if not parts:
        return pd.DataFrame(columns=["race_id", "dog_id", *KEYS, "pred_bsp_base", "bsp"])
    h = pd.concat(parts, ignore_index=True)
    return _with_bsp(h, prices)


def serve(store, day: date, pred: pd.DataFrame, oos_prefix: str | None) -> tuple[pd.DataFrame, dict]:
    """Today's priced card with the layer, and notes (dogs with a memory, the spread of m)."""
    if not switch_on():
        return pred, {"memory": "off (GREYHOUND_MEMORY)"}
    mem = memory_on(misses(history(store, day, oos_prefix)), day)
    out = apply(pred, mem)
    have = out.memory_m.abs() > 0
    return out, {"memory": TAG, "tau": TAU, "k": K, "b": B, "c": C, "runners_with_a_memory": int(have.sum()),
                 "runners": int(len(out)), "m_sd": round(float(out.memory_m[have].std()), 4) if have.sum() > 1 else None}


def ensure_oos(store, base_prefix: str, prices: pd.DataFrame | None = None) -> dict:
    """The recipe's out-of-sample day files with the layer applied, walk-forward, written to ``<base>_mem`` (each day
    from the base files of the days before it); only the days the target lacks are written."""
    base = base_prefix.rstrip("/")
    target = f"{base}_mem"
    have = {k.rsplit("/", 1)[-1] for k in store.listing(f"{target}/")}
    days = sorted(k.rsplit("/", 1)[-1] for k in store.listing(f"{base}/") if k.endswith(".parquet"))
    todo = [d for d in days if d not in have]
    if not todo:
        return {"prefix": target, "written": 0, "days": len(days)}
    frames = {d: store.get_parquet(f"{base}/{d}") for d in days}
    allp = pd.concat([f.assign(race_date=d[:10]) for d, f in frames.items() if f is not None and len(f)],
                     ignore_index=True)
    if prices is None:
        from greyhound.model import load_prices
        yrs = sorted(allp.race_date.str[:4].unique())
        prices = load_prices(store, f"{yrs[0]}-{yrs[-1]}")
    h = _with_bsp(allp.assign(pred_bsp_base=allp.pred_bsp), prices)
    dm = misses(h)
    n = 0
    for d in todo:
        f = frames.get(d)
        if f is None or not len(f):
            continue
        out = apply(f, memory_on(dm, date.fromisoformat(d[:10])))
        buf = io.BytesIO()
        out.to_parquet(buf, index=False)
        store.put(f"{target}/{d}", buf.getvalue())
        n += 1
    return {"prefix": target, "written": n, "days": len(days)}
