"""The sequence table: a small recurrent network over each dog's last eight runs, refitted walk-forward each month,
and the named forecasts it makes of the dog's next run (screened in discovery wave 6, 13 Nov 2026: scratchpad
disc6/seqenc; made a table on 14 Nov 2026, the owner: "Yes, work on these. Implement").

**The network.** A GRU (``HIDDEN`` units) reads the dog's last ``L`` (8) real runs dated strictly before the race
day, oldest first. A token a run: the facts of that run (finish centred, did not finish, beaten lengths, the time
against the track-and-trip standard of EARLIER days, the time against the field, the sectional rank, the log SP
share race-centred, the SP rank, favourite, won, field size, grade, trap, the comment codes: break, early pace,
path, led, trouble, finish) and where it sits against today (days ago, the gap to the run before it, same track,
the log trip change), with the grade scale of the run as a learnt embedding. The race card of today's run (trap,
field, grade, trip, days since the last run, history length) joins the GRU's last state into an ``EMB``-dim code;
a context layer adds the mean code of the field it meets. Five heads, all facts of THIS run (the next run of the
history): the SP share (a softmax over the race against the GBGB SP shares), the winner (a softmax against the
result), the finish centred, the time against standard, the sectional rank.

**Columns** are the named forecasts (their meaning is the same at every refit; the screen's embedding directions
are left out: their meaning moves with each refit, an independent review of 13 Nov 2026).

**Point in time.**

- A run's history: its dog's runs on EARLIER days (not trials, not cards); a run earlier the same day is never
  read. The time standard of a past run: the track-and-trip mean of calculated times on days before that run.
- The refit calendar: the rows of month M (from ``START``) are scored by a model fitted on races dated before the
  first of M: early stopping on the month before M, training on every race from the first fit's start to that
  month (an expanding window; the first fit reads the ``FIRST_MONTHS`` months before its month of early stopping,
  ``WINDOW_MONTHS`` makes the later ones rolling), warm-started from the model of the month before, which itself saw
  only earlier days. Each fit is asserted to end before M. Runs before the first fit's start are read only as
  history (the GBGB store from 2014: a dog's earlier runs, the time standards).
- What a row reads of its own race is card facts: its runners (the rows of ``runs`` with its ``race_id``), traps,
  field, class and trip. No result of its day enters its features; tests/test_greyhound_seq.py shuffles one day's
  results and requires every row of that day and before to be unchanged, and a card row to get what its run had.
- Nothing reads the trainer, the kennel or the owner (GBGB shows a dog's CURRENT trainer on its past rows).

**Determinism.** Every fit draws from generators seeded by the seed and the month; torch runs single-threaded with
deterministic algorithms; the batches are the same races in the same order on a rerun. Two builds give identical
tables (tests/test_greyhound_seq.py). ``SEEDS`` averages the forecasts of independent fits, one a seed.

**Runtime.** About 0.12 ms a race an epoch on one thread (GBGB 1 Jul 2025 to 5 Nov 2026, 14 monthly fits: 390 s).
From 2014 with the first fit on Dec 2021 to Nov 2023 and 35 expanding refits to Nov 2026, about 37 million
race-epochs: some 80 minutes here, up to about two hours on a GitHub runner (14 Nov 2026; ``STEP_MONTHS`` 2 or a
rolling ``WINDOW_MONTHS`` halves it if needed).

**Coverage.** Rows from ``START`` (2024-01-01: the stack trains on the 29 months before its test start) carry the
forecasts; earlier rows are NaN except ``se_nhist``. A race of more than ``SLOTS`` runners scores its first
``SLOTS`` (none in GBGB's 2025-2026 files).

**Not on the card.** This table is for research scoring first (the fast loop, ``pit_seq``). A card build (greyhound/
live.py ``card_tables``) would refit every month from 2024 for each card; it would need the latest month's fitted
weights stored by the nightly build (with the month it scores) and loaded to score the card, with no refit.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from greyhound.pit import KEY

PREFIX = "se_"
SOURCE = ("GBGB results (finishing position, beaten lengths, times, sectional, SP, comment, class, track, trip, trap: "
          "sources/gbgb.py), read by a GRU over each dog's last 8 runs on earlier days, refitted on the first of each "
          "month from races before it")
COLUMNS = {
    "se_lshare": "the network's forecast log SP share of the dog in today's race (a log-softmax over the field)",
    "se_vs_last": "the forecast log SP share, race-centred, less the race-centred log SP share of the dog's last run",
    "se_win_vs_sp": "the forecast log chance of winning less the forecast log SP share (the result head against the market head)",
    "se_time": "the forecast time against the track-and-trip standard (per cent of the standard / 2; below 0 faster)",
    "se_pos": "the forecast finish, centred ((position - (field + 1) / 2) / field; below 0 nearer the front)",
    "se_sec": "the forecast sectional rank in the race, centred (below 0 faster to the first bend)",
    "se_nhist": "the runs on earlier days the network read (0 to 8)",
}

L = 8                         # runs read
SLOTS = 6                     # runners a race
START = "2024-01-01"          # the first month scored
FIRST_MONTHS = 24             # the first fit's window, before its month of early stopping
WINDOW_MONTHS = None          # each later fit's window in months; None: expanding (every race from the first fit's start)
STEP_MONTHS = 1               # months between refits
EPOCHS0, PATIENCE0, LR0 = 30, 3, 2e-3   # the screen's settings (disc6/seqenc, revision 1)
EPOCHS, PATIENCE, LR = 6, 2, 1e-3
BATCH = 256                   # races a step
HIDDEN, EMB = 48, 12
SEEDS = (0,)
MIN_FIT = 200                 # races: fewer and the first fit waits
STD_MIN = 30                  # runs behind a track-and-trip standard
SCALES = ("A", "B", "D", "S", "P", "H")   # greyhound.data.GRADE_SCALE's ladders (0: none)
HEADS = ("sp", "win", "pos", "time", "sec")


def _day(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s).to_numpy("datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int64)


def _token_facts(h: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """The facts of each real run (``h``: no cards, no trials), read only as the history of LATER days."""
    from greyhound.metrics import parse_comment
    g = h.groupby("race_id")
    field = h.field.astype(float)
    pos = pd.to_numeric(h.position, errors="coerce")
    f = pd.DataFrame(index=h.index)
    pos_c = (pos - (field + 1) / 2) / field
    f["pos_c"] = pos_c.fillna(0.5)                                         # no finishing position: as last
    f["dnf"] = h.dnf.fillna(False).astype(float) if "dnf" in h else 0.0
    # the track-and-trip standard: the mean calculated time on days BEFORE the run's day (STD_MIN runs at least)
    a = h.adjusted_time.where(h.adjusted_time > 0)
    d = (h.assign(_a=a).groupby(["track", "distance_m", "_dn"], dropna=False)._a.agg(["sum", "count"]).reset_index()
         .sort_values(["track", "distance_m", "_dn"]))
    cs = d.groupby(["track", "distance_m"], dropna=False)[["sum", "count"]].cumsum().to_numpy() - d[["sum", "count"]].to_numpy()
    d["std"] = np.where(cs[:, 1] >= STD_MIN, cs[:, 0] / np.maximum(cs[:, 1], 1), np.nan)
    std = h[["track", "distance_m", "_dn"]].merge(d[["track", "distance_m", "_dn", "std"]], how="left",
                                                  on=["track", "distance_m", "_dn"])["std"].to_numpy()
    tv = ((a - std) / std * 100).clip(-6, 6)
    f["time_std"] = tv.fillna(0)
    f["time_na"] = tv.isna().astype(float)
    sr = g.sectional.rank(method="average")
    sn = g.sectional.transform("count")
    sv = ((sr - (sn + 1) / 2) / sn.clip(lower=1)).where(sn >= 3)
    f["sec_rank"] = sv.fillna(0)
    f["sec_na"] = sv.isna().astype(float)
    f["beaten"] = (h.beaten_lengths.clip(0, 15) / 5).fillna(1.0)
    ls = np.log(h.sp_p_norm.clip(lower=1e-4))
    ls_c = ls - ls.groupby(h.race_id).transform("mean")
    f["ls_c"] = ls_c.fillna(0)
    f["fav"] = (h.market_pos == 1).astype(float)
    f["won"] = h.won.fillna(0)
    f["sp_na"] = h.sp_p_norm.isna().astype(float)
    spr = g.sp_p_norm.rank(ascending=False, method="average")
    f["sp_rank"] = ((spr - (field + 1) / 2) / field).fillna(0)
    rt = h.run_time.where(h.run_time > 0)
    f["time_fld"] = (((g.run_time.transform("mean") - rt) / 0.08).clip(-12, 12) / 4).fillna(0)
    f["field"] = field - 5.5
    f["grade"] = (h.grade.fillna(5) - 5) / 3
    f["grade_na"] = h.grade.isna().astype(float)
    for k in range(1, 7):
        f[f"trap{k}"] = (h.trap == k).astype(float)
    memo = {c: parse_comment(c) for c in pd.unique(h.comment.fillna("").astype(str))}
    pc = pd.DataFrame([memo[c] for c in h.comment.fillna("").astype(str)], index=h.index)
    for c in ("brk", "epace", "path"):
        f["c_" + c] = pc[c].astype(float).fillna(0).clip(-3, 6)
        f["c_" + c + "_na"] = pc[c].isna().astype(float)
    f["c_led"] = pc.led.astype(float)
    f["c_trouble"] = pc.trouble.astype(float).clip(0, 4)
    f["c_fin"] = pc.fin.astype(float)
    labels = {"pos": pos_c.where(pos.notna()).to_numpy(float), "time": (tv / 2).to_numpy(float),
              "sec": sv.to_numpy(float), "ls_c": ls_c.to_numpy(float)}
    return f.to_numpy(np.float32), labels


def _net(d: int, c: int, ns: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            h, e = HIDDEN, EMB
            s.se = nn.Embedding(ns, 4)
            s.inp = nn.Sequential(nn.Linear(d + 4, h), nn.ReLU())
            s.gru = nn.GRU(h, h, batch_first=True)
            s.card = nn.Sequential(nn.Linear(c + 4, 24), nn.ReLU())
            s.mix = nn.Sequential(nn.Dropout(0.1), nn.Linear(h + 24, 48), nn.ReLU(), nn.Linear(48, e), nn.Tanh())
            s.ctx = nn.Sequential(nn.Linear(2 * e, 24), nn.ReLU(), nn.Linear(24, len(HEADS)))
            s.out = nn.Linear(e, len(HEADS))

        def forward(s, xs, xh, xc, xg, m):
            b, k = m.shape
            x = s.inp(torch.cat([xs, s.se(xh)], -1).view(b * k, L, -1))
            _, hn = s.gru(x)
            z = s.mix(torch.cat([hn[0], s.card(torch.cat([xc, s.se(xg)], -1).view(b * k, -1))], -1)).view(b, k, -1)
            mf = m.unsqueeze(-1).float()
            mean = (z * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)      # the field it meets
            return s.out(z) + s.ctx(torch.cat([z, mean.expand_as(z)], -1))

    return Net()


def _month(dn: np.ndarray) -> np.ndarray:
    """Months since January 1970 of each day number."""
    return dn.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)


def _first(mo: int) -> int:
    """The day number of the first of month ``mo`` (months since January 1970)."""
    return int(np.datetime64(int(mo), "M").astype("datetime64[D]").astype(np.int64))


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    import torch
    t0 = time.time()
    n = len(runs)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    trial = runs.is_trial.fillna(False).to_numpy(bool) if "is_trial" in runs else np.zeros(n, bool)
    dn = _day(runs.race_date)
    tt = runs.t.to_numpy("datetime64[s]").astype(np.int64) if "t" in runs else dn
    dog = runs.dog_id.to_numpy(np.int64)
    rid = runs.race_id.to_numpy(np.int64)
    out = {c: np.full(n, np.nan) for c in COLUMNS}

    # ---- the history pool: real runs, ordered by dog, day, time
    real = ~card & ~trial
    hi = np.flatnonzero(real)
    h = runs.iloc[hi].assign(_dn=dn[hi])
    F, lab = _token_facts(h)
    o = np.lexsort((tt[hi], dn[hi], dog[hi]))
    hdog, hdn = dog[hi][o], dn[hi][o]
    F, hls = F[o], lab["ls_c"][o]
    sp_held = ~np.isnan(hls)
    sid = {s: i + 1 for i, s in enumerate(SCALES)}
    hsc = runs.grade_scale.iloc[hi].map(sid).fillna(0).to_numpy(np.int64)[o] if "grade_scale" in runs else np.zeros(len(hi), np.int64)
    htrk = pd.Series(runs.track.to_numpy()[hi][o])
    hdist = runs.distance_m.to_numpy(float)[hi][o]

    # ---- every row's history (right-aligned, newest last) and its count
    big = np.int64(1) << 21
    keys = hdog * big + hdn
    end = np.searchsorted(keys, dog * big + dn, "left")
    start = np.searchsorted(keys, dog * big, "left")
    nh = np.minimum(end - start, L)
    out["se_nhist"] = nh.astype(float)
    last = np.where(nh > 0, end - 1, 0)
    last_ls = np.where((nh > 0) & sp_held[last], hls[last], np.nan)

    # ---- the races scored and fitted on: those from the first fit's window on
    s_mo = int(np.datetime64(START, "M").astype(np.int64))
    months = _month(dn)
    first_fit = _first(s_mo - FIRST_MONTHS - 1)
    use = dn >= first_fit
    notes = {"rows": int(n), "card_rows": int(card.sum()), "history_runs": int(len(hi)), "folds": {}}
    if not use.any() or months.max() < s_mo:
        tab = pd.DataFrame({"race_id": rid, "dog_id": runs.dog_id.to_numpy()})
        for c in COLUMNS:
            tab[c] = out[c].astype(np.float32)
        notes["seconds"] = round(time.time() - t0, 1)
        return tab, notes
    ui = np.flatnonzero(use)
    ou = ui[np.lexsort((rid[ui], tt[ui], dn[ui]))]                          # time order; a race's rows together
    r_of = np.cumsum(np.r_[True, rid[ou][1:] != rid[ou][:-1]]) - 1
    R = int(r_of[-1]) + 1
    slot = pd.Series(np.ones(len(ou))).groupby(r_of).cumcount().to_numpy()
    over = slot >= SLOTS
    notes["rows_beyond_slots"] = int(over.sum())
    keep = ~over
    rows = np.full((R, SLOTS), -1, np.int64)
    rows[r_of[keep], slot[keep]] = ou[keep]
    rdn = np.full(R, 0, np.int64)
    rdn[r_of] = dn[ou]
    rfit = np.ones(R, bool)                                                 # a race with a result (not a card, a trial)
    np.logical_and.at(rfit, r_of, real[ou])

    # per-row inputs (rows used only)
    nu = len(ou)
    pos_in = np.full(n, -1, np.int64)
    pos_in[ou] = np.arange(nu)
    idx = end[ou, None] - L + np.arange(L)[None, :]
    valid = idx >= start[ou, None]
    ridx = np.where(valid, idx, 0)
    hday = hdn[ridx]
    qd = dn[ou][:, None]
    assert (hday[valid] < np.broadcast_to(qd, hday.shape)[valid]).all()
    assert (hdog[ridx][valid] == np.broadcast_to(dog[ou][:, None], hday.shape)[valid]).all()
    ago = np.where(valid, qd - hday, 0)
    prev = np.concatenate([np.zeros((nu, 1), np.int64), hday[:, :-1]], 1)
    pv = np.concatenate([np.zeros((nu, 1), bool), valid[:, :-1]], 1)
    gap = np.where(valid & pv, hday - prev, 0)
    qtrk = runs.track.to_numpy()[ou]
    same = (htrk.to_numpy()[ridx] == qtrk[:, None])
    dq = runs.distance_m.to_numpy(float)[ou][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.nan_to_num(np.log(hdist[ridx] / dq), nan=0.0, posinf=0.0, neginf=0.0)
    rel = np.stack([np.log1p(ago) / 3, np.log1p(gap) / 3, same.astype(float), np.clip(dd, -1, 1),
                    np.repeat(np.isnan(dq), L, 1).astype(float)], -1).astype(np.float32) * valid[:, :, None]
    HSC = hsc[ridx] * valid
    dsl = np.where(nh[ou] > 0, ago[:, -1], 0)
    qr = runs.iloc[ou]
    cardf = np.column_stack([qr.field.to_numpy(float) - 5.5, (qr.grade.fillna(5).to_numpy(float) - 5) / 3,
                             qr.grade.isna().to_numpy(float), np.log(qr.distance_m.fillna(450).to_numpy(float) / 450),
                             qr.distance_m.isna().to_numpy(float), np.log1p(dsl) / 3, nh[ou] / L,
                             (nh[ou] == 0).astype(float)]
                            + [(qr.trap.to_numpy() == k).astype(float) for k in range(1, 7)]).astype(np.float32)
    gsc = qr.grade_scale.map(sid).fillna(0).to_numpy(np.int64) if "grade_scale" in qr else np.zeros(nu, np.int64)
    # labels (training races only)
    sp_lab = qr.sp_p_norm.fillna(0).to_numpy(np.float32)
    win_lab = qr.won.fillna(0).to_numpy(np.float32)
    aux = np.full((nu, 3), np.nan, np.float32)
    hpos = np.full(n, -1, np.int64)
    hpos[hi] = np.arange(len(hi))
    hr = hpos[ou]
    okr = hr >= 0
    for j, k in enumerate(("pos", "time", "sec")):
        aux[okr, j] = lab[k][hr[okr]]
    aux_m = np.isfinite(aux).astype(np.float32)
    aux = np.nan_to_num(aux)
    D = F.shape[1] + rel.shape[2] + 1

    def batch(races):
        rr = rows[races]
        m = rr >= 0
        p = pos_in[np.where(m, rr, ou[0])]                                  # padding reads row 0, masked
        v = valid[p] & m[..., None]
        xs = np.concatenate([F[ridx[p]], rel[p], np.ones(v.shape + (1,), np.float32)], -1) * v[..., None]
        T = torch.from_numpy
        return (T(np.ascontiguousarray(xs, dtype=np.float32)), T(HSC[p] * v), T(cardf[p] * m[..., None]),
                T(gsc[p] * m), T(m), p)

    def targets(p, m):
        T = torch.from_numpy
        y = sp_lab[p] * m
        y = y / np.clip(y.sum(1, keepdims=True), 1e-9, None)
        w = win_lab[p] * m
        w = w / np.clip(w.sum(1, keepdims=True), 1e-9, None)
        return T(y.astype(np.float32)), T(w.astype(np.float32)), T(aux[p] * m[..., None]), T(aux_m[p] * m[..., None])

    def ce(sc, y, m):
        return -(y * torch.log_softmax(sc.masked_fill(~m, -1e9), 1).clamp(min=-50)).sum(1).mean()

    def losses(net, races):
        xs, xh, xc, xg, m, p = batch(races)
        y, yw, ya, am = targets(p, m.numpy())
        o = net(xs, xh, xc, xg, m)
        l_sp = ce(o[..., 0], y, m)
        l_w = ce(o[..., 1], yw, m)
        l_a = (((o[..., 2:] - ya) ** 2) * am).sum() / am.sum().clamp(min=1)
        return l_sp + 0.3 * l_w + 0.5 * l_a, l_sp

    # ---- the walk-forward fits, one model a seed
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    last_mo = int(months.max())
    score_mos = list(range(s_mo, last_mo + 1, STEP_MONTHS))
    acc = np.zeros((len(SEEDS), nu, len(HEADS)), np.float64)
    have = np.zeros(nu, bool)
    try:
        for si, seed in enumerate(SEEDS):
            net = None
            for mo in score_mos:
                ts = time.time()
                m0, m1 = _first(mo), _first(mo + STEP_MONTHS)
                v0 = _first(mo - 1)
                f0 = first_fit if (net is not None and WINDOW_MONTHS is None) else \
                    _first(mo - 1 - (FIRST_MONTHS if net is None else WINDOW_MONTHS))
                fit = np.flatnonzero(rfit & (rdn >= f0) & (rdn < v0))
                val = np.flatnonzero(rfit & (rdn >= v0) & (rdn < m0))
                sc = np.flatnonzero((rdn >= m0) & (rdn < m1))
                log = {"fit_races": int(len(fit)), "val_races": int(len(val)), "scored_races": int(len(sc))}
                if net is None and len(fit) < MIN_FIT:
                    notes["folds"].setdefault(str(np.datetime64(mo, "M")), {})[f"seed{seed}"] = {**log, "skipped": True}
                    continue
                assert not len(fit) or rdn[fit].max() < m0
                assert not len(val) or rdn[val].max() < m0
                torch.manual_seed(int(seed) * 100003 + mo)
                rng = np.random.default_rng([int(seed), mo])
                first = net is None
                if first:
                    net = _net(F.shape[1] + rel.shape[2] + 1, cardf.shape[1], len(SCALES) + 1)
                assert D == net.inp[0].in_features - 4
                epochs, pat, lr = (EPOCHS0, PATIENCE0, LR0) if first else (EPOCHS, PATIENCE, LR)
                opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)

                def ev():
                    net.eval()
                    with torch.no_grad():
                        tot, k = 0.0, 0
                        for b in range(0, len(val), 4096):
                            vb = val[b:b + 4096]
                            tot += float(losses(net, vb)[1]) * len(vb)
                            k += len(vb)
                        return tot / max(k, 1)

                best = (ev() if (len(val) and not first) else np.inf, {a: x.clone() for a, x in net.state_dict().items()})
                bad, lg = 0, []
                for _ in range(epochs if len(fit) else 0):
                    net.train()
                    perm = rng.permutation(fit)
                    for b in range(0, len(perm), BATCH):
                        opt.zero_grad()
                        loss, _ = losses(net, perm[b:b + BATCH])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                    v = ev() if len(val) else 0.0
                    lg.append(round(v, 5))
                    if v < best[0] - 1e-5 or not len(val):
                        best, bad = (v, {a: x.clone() for a, x in net.state_dict().items()}), 0
                    else:
                        bad += 1
                        if bad >= pat:
                            break
                net.load_state_dict(best[1])
                net.eval()
                with torch.no_grad():
                    for b in range(0, len(sc), 4096):
                        races = sc[b:b + 4096]
                        xs, xh, xc, xg, m, p = batch(races)
                        o = net(xs, xh, xc, xg, m).clone()
                        o[..., 0] = torch.log_softmax(o[..., 0].masked_fill(~m, -1e9), 1)
                        o[..., 1] = torch.log_softmax(o[..., 1].masked_fill(~m, -1e9), 1)
                        mm = m.numpy()
                        acc[si, p[mm]] = o.numpy()[mm]
                        have[p[mm]] = True
                log.update({"fit_from": str(np.datetime64(f0, "D")) if len(fit) else None,
                            "val_ce": lg, "best": None if not np.isfinite(best[0]) else round(float(best[0]), 5),
                            "seconds": round(time.time() - ts, 1)})
                notes["folds"].setdefault(str(np.datetime64(mo, "M")), {})[f"seed{seed}"] = log
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det)

    res = acc.mean(0)
    rows_u = ou[have]
    r = res[have]
    lp = r[:, 0]
    rr = r_of[have]
    lp_c = lp - pd.Series(lp).groupby(rr).transform("mean").to_numpy()
    out["se_lshare"][rows_u] = lp
    out["se_vs_last"][rows_u] = lp_c - last_ls[rows_u]
    out["se_win_vs_sp"][rows_u] = r[:, 1] - lp
    out["se_pos"][rows_u] = r[:, 2]
    out["se_time"][rows_u] = r[:, 3]
    out["se_sec"][rows_u] = r[:, 4]
    tab = pd.DataFrame({"race_id": rid, "dog_id": runs.dog_id.to_numpy()})
    for c in COLUMNS:
        tab[c] = out[c].astype(np.float32)
    assert list(tab.columns) == KEY + list(COLUMNS)
    notes.update({"scored_rows": int(have.sum()), "seeds": list(SEEDS), "start": START,
                  "refit": f"the first of every {STEP_MONTHS} month(s) from {START}, on races before it: early stopping on "
                           f"the month before, training on "
                           f"{'every race from ' + str(np.datetime64(first_fit, 'D')) if WINDOW_MONTHS is None else f'the {WINDOW_MONTHS} months'}"
                           f" before that ({FIRST_MONTHS} months at the first fit), warm-started; {EPOCHS0} epochs at "
                           f"most at the first fit (patience {PATIENCE0}), {EPOCHS} after (patience {PATIENCE})",
                  "fit_seconds": round(float(sum(f[k].get("seconds", 0) for f in notes["folds"].values() for k in f)), 1),
                  "seconds": round(time.time() - t0, 1)})
    return tab, notes
