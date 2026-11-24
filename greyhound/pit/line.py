"""The running line and the role, and whether they have changed; the lines of the dogs drawn either side.

A greyhound is a railer, a middle runner or a wide runner, and a fast or a slow breaker, and the racing manager draws
it a trap to suit. This table measures, for each dog in each race:

- **its line and whether it is moving**: the habitual line read from the comments of its earlier runs (``Rls`` 1,
  ``Mid`` 2, ``W`` 3: greyhound/metrics.py ``parse_comment``), and the last two run-days against everything before
  them, for the line, the break, the early pace and whether it led (a dog whose role is changing);
- **today's trap against that line**: the line as a box (rails 1, middle 3.5, wide 6: ``2.5 x line - 1.5``) against
  the trap drawn, the trap against the traps of its firsts and seconds and its usual trap, whether it has ever had
  this trap, and its finishing position from this trap and the ones beside it against all its runs;
- **the neighbours' lines**: how far the dog drawn inside wants to come out and the dog drawn outside wants to come
  in (the squeeze), that squeeze where this dog's own line crosses theirs (the conflict), the empty traps beside it,
  and whether the neighbours break faster than it does.

Screened on 12 Nov 2026 against what the stack still misses (the residual of the Betfair SP on the stack's forecast,
1 Jun to 10 Nov 2026, 110,788 runners): the set took the residual's error from 0.2872 to 0.2859 (z -7.9) and from
0.2854 to 0.2844 beyond the optics table; the squeeze was the strongest single column (rank correlation 0.0375), then
the conflict (0.0235) and the vacant traps (-0.0186). The screen is a first read, not a registered trial.

**Point in time.** A dog's history is its race rows (no trial, no card row, a trap 1 to 6) folded to one row a day,
and a feature reads the days strictly before the race's day: never the day's own result, nor an earlier race the
same day. The card supplies the trap and which dogs are drawn in the race: the field is every row of ``runs`` with
the race's id, a card row included, and a neighbour's line is its own history before the day. A row whose trap is
not 1 to 6 keeps the columns that need no trap and has no value for the others.

**The field as run.** GBGB's results list the dogs that ran; a card lists the dogs declared, and a reserve may run
from a withdrawn dog's trap. Where a dog comes out after the card is read, the four neighbour columns of the dogs
beside it are made from the card's field and the stored table's from the field that ran (``CARD_UNSAFE`` is left
empty: the card row is priced on what the card shows, which is what a bettor has; the difference is the non-runner).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "ln_"
SOURCE = ("GBGB results (the comment's running line, break, early pace and lead; trap; finishing position: "
          "sources/gbgb.py) of the dog's and its trap neighbours' earlier race days, and the card's trap draw")
TRAPS = 6
LAST = 2                      # the run-days that are "now" in a drift; everything before them is the habit
BIG = 1 << 21                 # day numbers are below this: a dog's block of the (dog, day) key
COLUMNS = {
    "ln_path_hab": "The dog's habitual running line: the mean line of its earlier runs whose comment gives one "
                   "(1 rails, 2 middle, 3 wide)",
    "ln_path_drift": "The mean line of its last two run-days less the mean of all before them (above 0: running wider)",
    "ln_brk_drift": "The mean break of its last two run-days (very quick 2 to very slow -2, no code 0, runs with a "
                    "comment) less the mean of all before them",
    "ln_ep_drift": "The mean early-pace code (EP 1, lacked EP -1, none 0) of its last two run-days less all before them",
    "ln_led_drift": "The share of runs it led in its last two run-days less the share in all before them",
    "ln_trap_mis": "Today's trap less the box its line wants (2.5 x ln_path_hab - 1.5): above 0 drawn wider than its "
                   "line, below 0 drawn inside it",
    "ln_trap_mis_abs": "The size of ln_trap_mis: boxes between the trap and the line",
    "ln_trap_vs_best": "The distance of today's trap from the mean trap of its earlier firsts and seconds",
    "ln_trap_vs_usual": "Today's trap less the mean trap of its earlier runs",
    "ln_first_trap": "1 when it has run before but never from today's trap; 0 when it has; none with no earlier run",
    "ln_form_near_trap": "Its mean finishing position (0 first to 1 last) from today's trap and the traps beside it, "
                         "less the same over all its earlier runs (below 0: better near this box)",
    "ln_nb_squeeze": "How far the dog drawn inside wants to come out plus how far the dog drawn outside wants to "
                     "come in, in boxes (their line's box against their trap; 0 with no such neighbour or no line)",
    "ln_nb_conflict": "The squeeze where the lines cross: how far this dog wants in x the inside neighbour's move "
                      "out, plus how far it wants out x the outside neighbour's move in",
    "ln_nb_vacant": "The empty traps beside it in the race's field (0 to 2; traps 1 and 6 have one side)",
    "ln_nb_brk": "The mean habitual break of the dogs drawn either side less its own (above 0: the neighbours "
                 "break faster)",
}
_SUMS = ["n", "cm_n", "path_s", "path_n", "brk_s", "ep_s", "led_s", "trap_s", "pos_s", "pos_n", "bt_s", "bt_n"]


def _comment_codes(df: pd.DataFrame) -> pd.DataFrame:
    """The comment's path, break, early pace and lead a run (the engine's per-run figures carry them already)."""
    if {"path", "brk", "epace", "led"} <= set(df.columns):
        return df[["path", "brk", "epace", "led"]]
    from greyhound.metrics import parse_comment
    com = df.comment.fillna("") if "comment" in df else pd.Series("", index=df.index)
    uniq = {c: parse_comment(c) for c in pd.unique(com)}
    return pd.DataFrame([uniq[c] for c in com.to_numpy()], index=df.index)[["path", "brk", "epace", "led"]]


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else runs
    n = len(df)
    codes = _comment_codes(df)
    flag = lambda c: df[c].fillna(False).to_numpy(bool) if c in df else np.zeros(n, bool)  # noqa: E731
    trap_f = pd.to_numeric(df.trap, errors="coerce").to_numpy(float)
    okt = np.isfinite(trap_f) & (trap_f >= 1) & (trap_f <= TRAPS) & (trap_f == np.round(trap_f))
    trap = np.where(okt, trap_f, 1).astype(np.int64)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    dated = np.isfinite(dn)
    dn = np.where(dated, dn, 0).astype(np.int64)
    dc = pd.factorize(df.dog_id)[0].astype(np.int64)
    hist = okt & dated & ~flag("is_card") & ~flag("is_trial")       # a run that is history for later days

    # ---- each run's facts (read only from later days)
    com = df.comment.fillna("").astype(str).str.strip().to_numpy() if "comment" in df else np.full(n, "")
    has = (com != "").astype(float)
    path = codes.path.to_numpy(float)
    pos = pd.to_numeric(df.position, errors="coerce").to_numpy(float)
    field = (pd.to_numeric(df.field, errors="coerce").to_numpy(float) if "field" in df
             else df.groupby("race_id").dog_id.transform("size").to_numpy(float))
    with np.errstate(invalid="ignore"):
        placed = (pos >= 1) & (pos <= 8)
        posn = np.where(placed, np.minimum((pos - 1) / np.maximum(field - 1, 1), 1.0), np.nan)
        best = (pos >= 1) & (pos <= 2)                               # its best runs: first or second
    okp = np.isfinite(posn)
    h = {"n": np.ones(n), "cm_n": has,
         "path_s": np.nan_to_num(path), "path_n": np.isfinite(path).astype(float),
         "brk_s": np.nan_to_num(codes.brk.to_numpy(float)) * has,    # no break code is a moderate break (0)
         "ep_s": np.nan_to_num(codes.epace.to_numpy(float)) * has,
         "led_s": np.nan_to_num(codes.led.to_numpy(float)) * has,
         "trap_s": trap.astype(float), "pos_s": np.nan_to_num(posn), "pos_n": okp.astype(float),
         "bt_s": trap * best, "bt_n": best.astype(float)}
    for k in range(1, TRAPS + 1):
        at = trap == k
        h[f"tr{k}"] = at.astype(float)
        h[f"tp{k}_s"] = np.where(at & okp, posn, 0.0)
        h[f"tp{k}_n"] = (at & okp).astype(float)
    cols = list(h)
    H = pd.DataFrame(h)[hist]
    H["dc"], H["dn"] = dc[hist], dn[hist]

    # ---- one row a dog a day, in (dog, day) order; G: the dog's sums to the end of each of its days
    D = H.groupby(["dc", "dn"], sort=True)[cols].sum().reset_index()
    G = D.groupby("dc", sort=False)[cols].cumsum().to_numpy(float)
    key = D.dc.to_numpy(np.int64) * BIG + D.dn.to_numpy(np.int64)
    block = np.searchsorted(key, dc * BIG)                           # where the dog's days start
    upto = np.searchsorted(key, dc * BIG + dn)                       # its first day NOT before today
    seen = upto - block                                              # its run-days strictly before today

    def sums(back: int) -> np.ndarray:
        """The dog's sums over its earlier run-days, the last ``back`` of them left out."""
        res = np.zeros((n, len(cols)))
        some = seen > back
        if len(G):
            res[some] = G[upto[some] - 1 - back]
        return res

    ci = {c: j for j, c in enumerate(cols)}
    C, E = sums(0), sums(LAST)                                       # every earlier day; all but the last two
    Lw = C - E                                                       # the last two run-days
    c = lambda a, name: a[:, ci[name]]  # noqa: E731
    path_hab = _div(c(C, "path_s"), c(C, "path_n"))
    brk_hab = _div(c(C, "brk_s"), c(C, "cm_n"))
    drift = lambda s, m: _div(c(Lw, s), c(Lw, m)) - _div(c(E, s), c(E, m))  # noqa: E731
    f = {"path_hab": path_hab, "path_drift": drift("path_s", "path_n"), "brk_drift": drift("brk_s", "cm_n"),
         "ep_drift": drift("ep_s", "cm_n"), "led_drift": drift("led_s", "cm_n")}

    # ---- today's trap against the line
    tr = np.where(okt, trap_f, np.nan)
    want = 2.5 * path_hab - 1.5                                      # the line as a box: rails 1, middle 3.5, wide 6
    f["trap_mis"] = tr - want
    f["trap_mis_abs"] = np.abs(tr - want)
    f["trap_vs_best"] = np.abs(tr - _div(c(C, "bt_s"), c(C, "bt_n")))
    f["trap_vs_usual"] = tr - _div(c(C, "trap_s"), c(C, "n"))
    idx = np.arange(n)
    cnt = C[:, [ci[f"tr{k}"] for k in range(1, TRAPS + 1)]]
    f["first_trap"] = np.where(okt & (c(C, "n") > 0), (cnt[idx, trap - 1] == 0).astype(float), np.nan)
    ps = C[:, [ci[f"tp{k}_s"] for k in range(1, TRAPS + 1)]]
    pn = C[:, [ci[f"tp{k}_n"] for k in range(1, TRAPS + 1)]]
    ns, nn = np.zeros(n), np.zeros(n)
    for d in (-1, 0, 1):
        k = trap - 1 + d
        v = (k >= 0) & (k < TRAPS)
        kk = np.clip(k, 0, TRAPS - 1)
        ns += np.where(v, ps[idx, kk], 0.0)
        nn += np.where(v, pn[idx, kk], 0.0)
    f["form_near_trap"] = np.where(okt, _div(ns, nn) - _div(c(C, "pos_s"), c(C, "pos_n")), np.nan)

    # ---- the neighbours: the race's field by trap (every row of the race; the first row of a trap stands for it)
    rc = pd.factorize(df.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0
    lean = want - tr                                                 # above 0: its line is outside its trap
    there = np.zeros((nr, TRAPS + 2), bool)
    lean_g = np.full((nr, TRAPS + 2), np.nan)
    brk_g = np.full((nr, TRAPS + 2), np.nan)
    rows = np.flatnonzero(okt)[::-1]                                 # written last to first, so the first row stays
    there[rc[rows], trap[rows]] = True
    lean_g[rc[rows], trap[rows]] = lean[rows]
    brk_g[rc[rows], trap[rows]] = brk_hab[rows]
    in_there, out_there = there[rc, trap - 1], there[rc, trap + 1]
    sq_in = np.nan_to_num(np.clip(lean_g[rc, trap - 1], 0, None))    # the inside dog's move out, towards this box
    sq_out = np.nan_to_num(np.clip(-lean_g[rc, trap + 1], 0, None))  # the outside dog's move in
    own = np.nan_to_num(lean)
    nb = np.stack([brk_g[rc, trap - 1], brk_g[rc, trap + 1]], axis=1)
    nb_n = np.isfinite(nb).sum(axis=1)
    f["nb_squeeze"] = np.where(okt, sq_in + sq_out, np.nan)
    f["nb_conflict"] = np.where(okt, np.clip(-own, 0, None) * sq_in + np.clip(own, 0, None) * sq_out, np.nan)
    f["nb_vacant"] = np.where(okt, (~in_there & (trap > 1)).astype(float) + (~out_there & (trap < TRAPS)), np.nan)
    f["nb_brk"] = np.where(okt, _div(np.nansum(nb, axis=1), nb_n) - brk_hab, np.nan)

    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    for name in COLUMNS:
        out[name] = np.asarray(f[name[len(PREFIX):]], dtype=np.float32)
    seats = pd.DataFrame({"r": rc[okt], "t": trap[okt]})
    notes = {"rows": int(n), "history_rows": int(hist.sum()), "dog_days": int(len(D)),
             "rows_without_a_trap_1_to_6": int((~okt).sum()),
             "rows_sharing_a_trap_in_a_race": int(seats.duplicated(["r", "t"], keep=False).sum()),
             "rows_with_a_line": round(float(np.isfinite(path_hab).mean()), 4) if n else 0.0,
             "history_runs_with_a_line": round(float(np.isfinite(path[hist]).mean()), 4) if hist.any() else 0.0}
    return out, notes
