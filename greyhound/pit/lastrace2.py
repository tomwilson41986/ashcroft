"""The last race's speed with the meeting's variant taken out (reports/greyhound_feature_review.md, "lastrace": race
speed with no meeting variant; subtract the variant). The lastrace table's ``lr_race_speed`` is the last race's
winning time against the course standard: a fast night makes every race on it look fast.

- The race speed of a race: (its course standard - its winner's calculated time) / 0.08, lengths, above 0 fast; the
  standard is ``lastrace.standards`` (winners on EARLIER days, the class shrunk to the course), the same as
  ``lr_race_speed``.
- The meeting's variant for a race: the mean race speed of the OTHER races at that track on that day (at least
  ``MIN_RACES`` of them with a speed; NaN under), leave-one-out so a race is not measured against itself. Lengths are
  not scaled by distance (a simple first cut).
- ``lr2_variant``: the variant of the dog's last race; ``lr2_race_speed_v``: that race's speed less it (clipped at
  25 as ``lr_race_speed``).

The last race is the dog's run on its last EARLIER day (that day's last run), as in lastrace.py; a card row
(``is_card``) is never a last run. Every column reads only earlier days: card-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, lastrace

PREFIX = "lr2_"
SOURCE = "GBGB results (calculated times, positions) of the dog's last race and the other races at its meeting"
DEPENDS = ("lastrace",)
KEY = ["race_id", "dog_id"]
MIN_RACES = 3
CLIP = 25.0

COLUMNS = {
    f"{PREFIX}variant": "the last race's meeting variant: the mean race speed (winner against the course standard, "
                        f"lengths, above 0 fast) of the other races at that track that day (at least {MIN_RACES})",
    f"{PREFIX}race_speed_v": "the last race's speed (as lr_race_speed) less its meeting variant, lengths, above 0 "
                             "fast for its night",
}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    n = len(runs)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    cols = ["race_id", "dog_id", "t", "track", "distance_m", "race_class", "position", "adjusted_time"]
    df = pd.DataFrame({c: (runs[c].to_numpy() if c in runs else np.nan) for c in cols})
    for c in ("distance_m", "position", "adjusted_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["d"] = core.day(runs).to_numpy()
    df["_i"] = np.arange(n)

    h = df[~card].copy()
    h["atime"] = h.adjusted_time.where((h.adjusted_time / h.distance_m).between(*lastrace.PACE))
    std = lastrace.standards(h)
    win = h[(h.position == 1) & h.atime.notna()].drop_duplicates("race_id").set_index("race_id").atime
    races = h.drop_duplicates("race_id")[["race_id", "d", "track"]].copy()
    races["track"] = races.track.astype(object).where(races.track.notna(), "?").astype(str)
    races["rs"] = ((races.race_id.map(std) - races.race_id.map(win)) / core.LENGTH_S).to_numpy()
    has = races.rs.notna()
    g = races.assign(s=races.rs.fillna(0.0), k=has.astype(float)).groupby(["track", "d"])
    S, K = g.s.transform("sum"), g.k.transform("sum")
    others = K - has
    races["v"] = ((S - races.rs.fillna(0.0)) / others).where(others >= MIN_RACES)
    races["rsv"] = (races.rs - races.v).clip(-CLIP, CLIP)
    h = h.merge(races[["race_id", "v", "rsv"]], on="race_id", how="left")

    h = h.sort_values(["dog_id", "t"], kind="stable")
    P = h.drop_duplicates(["dog_id", "d"], keep="last")[["dog_id", "d", "v", "rsv"]].sort_values("d", kind="stable")
    P = P.assign(L_d=P.d)
    t = df[["_i", "dog_id", "d"]].sort_values("d", kind="stable")
    m = pd.merge_asof(t, P, on="d", by="dog_id", allow_exact_matches=False).sort_values("_i").reset_index(drop=True)
    assert len(m) == n and (m.L_d.dropna() < m.d[m.L_d.notna()]).all()

    out = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    out[f"{PREFIX}variant"] = m.v.to_numpy(float).astype(np.float32)
    out[f"{PREFIX}race_speed_v"] = m.rsv.to_numpy(float).astype(np.float32)
    assert list(out.columns) == KEY + list(COLUMNS)
    notes = {"rows": int(n), "card_rows": int(card.sum()), "races": int(len(races)),
             "races_with_a_speed": int(has.sum()), "races_with_a_variant": int(races.v.notna().sum()),
             "rows_with_a_variant": round(float(m.v.notna().mean()), 4) if n else None,
             "min_races": MIN_RACES, "card_safe": list(COLUMNS)}
    return out, notes
