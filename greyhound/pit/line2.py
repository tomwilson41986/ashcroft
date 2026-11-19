"""The neighbours' lines weighted by arrival, and the pace collision (the feature review of 17 Nov 2026,
reports/greyhound_feature_review.md, row "line": "Neighbour squeeze ignores whether the neighbour arrives at the same
time"). A research variant of greyhound/pit/line.py (``ln_``), whose output it leaves alone.

Each dog's lean is line.py's: the box its habitual line wants (``2.5 x ln_path_hab - 1.5``) less its trap (above 0
its line is outside its trap). A neighbour matters only if it reaches the first bend with the dog: each pair is
weighted by ext.py's contact chance ``c = exp(-|xt_sec_i - xt_sec_k| / ext.SCALE)``, the expected first sectionals
of the two in lengths (the expected-time table, made before the race).

- **squeeze**, traps 1 and 2 away: the dog drawn ``d`` traps inside reaches this dog's box when it wants out by ``d``
  boxes or more, so it squeezes by ``max(lean_k - (d - 1), 0)``; the one ``d`` outside by ``max(-lean_k - (d - 1), 0)``
  (``d`` = 1 is line.py's ``ln_nb_squeeze``), each times ``c``;
- **conflict**: that squeeze where the dog's own line crosses (it wants in against the inside squeeze, out against
  the outside), as line.py's ``ln_nb_conflict``, arrival-weighted and over traps 1 and 2 away;
- **pace collision**: for each pair 1 or 2 traps apart, the closeness of arrival ``c`` x the lane convergence (the
  inside dog's lean less the outside dog's: a wide runner drawn inside a railer; ``max(closing - (d - 1), 0)`` boxes
  of crossing) x the neighbour's harm (ext.py's rating of what it has cost the dogs on that side on earlier days:
  ``nx_own_out`` of a dog inside, ``nx_own_in`` of a dog outside, in lengths, shrunk; the immediate-neighbour rating
  stands for 2 traps away too). ``ln2_collide`` is the same without the harm (the geometry alone).

A vacant trap, a neighbour with no line or no expected sectional adds 0; a dog with no expected sectional (a track
and trip without timed sectionals) or a trap outside 1 to 6 has none of the columns. Point in time and card-safe as
the tables it reads (line, xtime, ext: earlier days and the card's draw); the field is the rows of the race, so a
non-runner after the card is read moves the columns of the dogs within 2 traps of it (line.py's "field as run").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, ext, line

DEPENDS = ("line", "ext", "xtime")
PREFIX = "ln2_"
SOURCE = ("GBGB results (the comment's running line, trap: sources/gbgb.py) through the line, expected-time "
          "(xt_sec) and neighbour-externality (nx_own_in, nx_own_out) tables, and the card's trap draw")
SLOTS = 6
PAD = 2
COLUMNS = {
    f"{PREFIX}sq1": "the squeeze of the dogs drawn immediately either side (line.py's ln_nb_squeeze), each weighted "
                    "by the chance it reaches the first bend with this dog (exp(-|expected sectional gap| / "
                    f"{ext.SCALE:g} length))",
    f"{PREFIX}sq2": "the same for the dogs drawn 2 traps away (an inside dog that wants out 2 boxes or more, an "
                    "outside dog that wants in 2 or more, less 1 box)",
    f"{PREFIX}sq": "ln2_sq1 + ln2_sq2",
    f"{PREFIX}conflict": "the arrival-weighted squeeze where the dog's own line crosses it (it wants in against the "
                         "inside squeeze, out against the outside), traps 1 and 2 away",
    f"{PREFIX}close": "the sum of the arrival closeness to the dogs drawn 1 and 2 traps away (0 to 4)",
    f"{PREFIX}collide": "pace collision geometry: the sum over the dogs 1 and 2 traps away of arrival closeness x "
                        "lane convergence (boxes by which the inside dog's lean exceeds the outside dog's, less the "
                        "gap less 1)",
    f"{PREFIX}collide_harm": "the pace collision score: each pair's closeness x convergence x the neighbour's harm "
                             "rating on that side (ext.py nx_own_out of a dog inside, nx_own_in of a dog outside, "
                             "lengths)",
}


def _inputs(runs: pd.DataFrame, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """The line, expected sectional and harm columns for ``df``'s rows (from ``ctx``, or built here)."""
    ln = ctx.get("line")
    if ln is None:
        ln, _ = line.build(runs, df)
    xt = ctx.get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    nx = ctx.get("ext")
    if nx is None:
        nx, _ = ext.build(runs, df, {"xtime": xt})
    key = ["race_id", "dog_id"]
    j = df[key].merge(ln[key + ["ln_trap_mis"]].drop_duplicates(key), on=key, how="left")
    j = j.merge(xt[key + ["xt_sec"]].drop_duplicates(key), on=key, how="left")
    return j.merge(nx[key + ["nx_own_out", "nx_own_in"]].drop_duplicates(key), on=key, how="left")


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    n = len(df)
    j = _inputs(runs, df, dict(ctx or {}))
    lean = -j.ln_trap_mis.to_numpy(float)                            # want - trap: above 0 its line is outside
    xs = j.xt_sec.to_numpy(float)
    h_out, h_in = j.nx_own_out.to_numpy(float), j.nx_own_in.to_numpy(float)
    trap = pd.to_numeric(df.trap, errors="coerce").to_numpy(float)
    okt = np.isfinite(trap) & (trap >= 1) & (trap <= SLOTS) & (trap == np.round(trap))
    rc = pd.factorize(df.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0
    s = np.where(okt, trap - 1 + PAD, 0).astype(np.int64)
    cell = np.where(okt, rc * (SLOTS + 2 * PAD) + s, -1 - np.arange(n))
    first = okt & ~pd.Series(cell).duplicated().to_numpy()          # the first row of a trap stands for it

    def grid(v: np.ndarray) -> np.ndarray:
        g = np.full((nr, SLOTS + 2 * PAD), np.nan)
        g[rc[first], s[first]] = v[first]
        return g

    G_lean, G_xs, G_hout, G_hin = grid(lean), grid(xs), grid(h_out), grid(h_in)
    own = np.nan_to_num(lean)
    acc = {k: np.zeros(n) for k in ("sq1", "sq2", "conflict", "close", "collide", "collide_harm")}
    for d in (1, 2):
        for side in (-1, 1):                                         # -1: the dog d traps inside; +1: outside
            at = np.clip(s + side * d, 0, SLOTS + 2 * PAD - 1)
            lk, xk = G_lean[rc, at], G_xs[rc, at]
            hk = (G_hout if side < 0 else G_hin)[rc, at]
            c = np.nan_to_num(np.exp(-np.abs(xs - xk) / ext.SCALE))  # no neighbour or no sectional: 0
            lk0 = np.nan_to_num(lk)
            sq = np.clip(lk0 - (d - 1), 0, None) if side < 0 else np.clip(-lk0 - (d - 1), 0, None)
            acc[f"sq{d}"] += c * sq
            wants = np.clip(-own, 0, None) if side < 0 else np.clip(own, 0, None)
            acc["conflict"] += c * wants * sq
            acc["close"] += c
            closing = (lk0 - own) if side < 0 else (own - lk0)       # the inside dog's lean less the outside dog's
            conv = np.where(np.isfinite(lk), np.clip(closing - (d - 1), 0, None), 0.0)
            acc["collide"] += c * conv
            acc["collide_harm"] += c * conv * np.nan_to_num(hk)
    acc["sq"] = acc["sq1"] + acc["sq2"]
    valid = okt & np.isfinite(xs)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    for name in COLUMNS:
        out[name] = np.where(valid, acc[name[len(PREFIX):]], np.nan).astype(np.float32)
    notes = {"rows": int(n), "rows_without_a_trap_1_to_6": int((~okt).sum()),
             "rows_with_an_expected_sectional": round(float(np.isfinite(xs).mean()), 4) if n else 0.0,
             "rows_with_a_line": round(float(np.isfinite(lean).mean()), 4) if n else 0.0,
             "rows_with_a_harm_rating": round(float(np.isfinite(h_out).mean()), 4) if n else 0.0,
             "contact_scale_lengths": ext.SCALE}
    return out, notes
