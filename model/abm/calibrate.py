"""
Pattern-oriented calibration of the ABM's behavioural parameters.

The interaction parameters (blocking gap, headway, lane-change rate,
interference hazard, drafting saving, lead-effort policy) have no
tractable likelihood, so they are fitted by simulation-based inference:
a population Monte-Carlo / ABC-SMC loop that keeps the parameter draws
whose simulated summary patterns sit closest to the observed ones.

Following Grimm & Railsback's pattern-oriented modelling, several
patterns are matched at once, all of them computable from the database's
race comments and results (no sectionals needed):

    trouble_rate        share of runs with 'hampered / denied a run / ...'
    front_win_share     share of winners that raced prominently early
    early_finish_corr   correlation between early position and finishing
                        position (how much the early order persists)
    median_win_margin,  distribution of winning margins in lengths
    p90_win_margin
    contested_lead_penalty  win rate of prominent runners when >= 2 horses
                        disputed the lead minus when the lead was uncontested
    low_draw_win_share  share of winners drawn in the inside third of the
                        stalls in fields of 10+ (draw / ground-loss check)

A model that reproduces all of them at once is far more credible than one
tuned to finishing order alone.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from model.abm.agents import build_field
from model.abm.simulate import RaceSimulator, SimConfig, SimResult
from model.abm.track import Track
from model.perf_figures import ensure_raceid, parse_beaten_lengths

log = logging.getLogger(__name__)

# parameter -> (lower, upper) uniform prior
DEFAULT_PRIORS = {
    "trouble_hazard": (0.0, 0.20),
    "draft_saving": (0.0, 0.30),
    "s0": (0.8, 3.0),
    "headway_T": (0.05, 0.40),
    "p_switch_out": (0.2, 1.5),
    "lead_effort_contested": (0.50, 0.90),
    "lead_effort_uncontested": (0.30, 0.60),
    "process_noise_sd": (0.05, 0.40),
    "follow_gain": (0.05, 0.30),
    "stall_spread": (0.15, 0.6),
    "trouble_speed_drop": (0.03, 0.15),
}


@dataclass
class PatternTargets:
    trouble_rate: float
    front_win_share: float
    early_finish_corr: float
    median_win_margin: float
    p90_win_margin: float
    contested_lead_penalty: float
    low_draw_win_share: float = np.nan

    def as_array(self) -> np.ndarray:
        return np.array([self.trouble_rate, self.front_win_share, self.early_finish_corr,
                         self.median_win_margin, self.p90_win_margin, self.contested_lead_penalty,
                         self.low_draw_win_share], float)


# rough scale of each pattern for the normalised loss
PATTERN_SCALE = np.array([0.03, 0.08, 0.10, 0.5, 1.5, 0.05, 0.05])


def observed_patterns(df: pd.DataFrame) -> PatternTargets:
    """Patterns from real results (needs comment-derived columns from
    model.pace_metrics: early_pos, had_trouble)."""
    d = ensure_raceid(df).copy()
    if "early_pos" not in d.columns or "had_trouble" not in d.columns:
        from model.pace_metrics import parse_run_style
        parsed = d["comment"].apply(parse_run_style).apply(pd.Series)
        for c in parsed.columns:
            d[c] = parsed[c]
    pos = pd.to_numeric(d["placing_numerical"], errors="coerce")
    winners = d[pos == 1]
    trouble_rate = float(d["had_trouble"].mean())
    front_win_share = float((winners["early_pos"] >= 4.5).mean())
    ok = pos.notna() & d["early_pos"].notna()
    early_finish_corr = float(np.corrcoef(-d.loc[ok, "early_pos"], pos[ok])[0, 1]) if ok.sum() > 10 else np.nan
    second = d[pos == 2]
    margins = second["total_dst_bt"].map(parse_beaten_lengths).dropna() if "total_dst_bt" in d.columns else pd.Series(dtype=float)
    med = float(margins.median()) if len(margins) else np.nan
    p90 = float(margins.quantile(0.9)) if len(margins) else np.nan
    n_front = d.assign(_f=(d["early_pos"] >= 5.0).astype(int)).groupby("raceid")["_f"].transform("sum")
    prom = d["early_pos"] >= 5.0
    won = (pos == 1).astype(float)
    contested = prom & (n_front >= 2); lone = prom & (n_front == 1)
    pen = float(won[contested].mean() - won[lone].mean()) if contested.any() and lone.any() else np.nan
    stall = pd.to_numeric(d["stall"], errors="coerce"); nr = pd.to_numeric(d["number_of_runners"], errors="coerce")
    big = (pos == 1) & (nr >= 10) & stall.notna()
    low = float(((stall[big] - 1) / nr[big] < 1 / 3).mean()) if big.any() else np.nan
    return PatternTargets(trouble_rate, front_win_share, early_finish_corr, med, p90, pen, low)


def simulated_patterns(results: list[SimResult]) -> PatternTargets:
    tr, fw, ef, margins, cont, lone, low = [], [], [], [], [], [], []
    for r in results:
        S, N = r.positions.shape
        if N >= 10:
            win_stall = r.field.stall[np.argmax(r.positions == 1, axis=1)]
            low.append(np.mean((win_stall - 1) / N < 1 / 3))
        tr.append(r.trouble.mean())
        win_early = r.early_rank[r.positions == 1]
        fw.append(np.nanmean(win_early <= max(2, round(0.3 * N))))
        er, ps = r.early_rank.ravel(), r.positions.ravel()
        ok = np.isfinite(er)
        if ok.sum() > 10:
            ef.append(np.corrcoef(er[ok], ps[ok])[0, 1])
        second = np.where(r.positions == 2, r.margins_lengths, np.nan)
        margins.append(np.nanmean(second, axis=1))
        front = r.early_rank <= 2
        won = r.positions == 1
        nc = r.n_contesting[:, None] if np.isfinite(r.n_contesting).any() else np.full((S, 1), np.nan)
        contested = front & (nc >= 1); alone = front & (nc == 0)
        if contested.any():
            cont.append(won[contested].mean())
        if alone.any():
            lone.append(won[alone].mean())
    margins = np.concatenate(margins) if margins else np.array([np.nan])
    pen = (np.mean(cont) - np.mean(lone)) if cont and lone else np.nan
    return PatternTargets(float(np.mean(tr)), float(np.mean(fw)), float(np.mean(ef)) if ef else np.nan,
                          float(np.nanmedian(margins)), float(np.nanquantile(margins, 0.9)), float(pen),
                          float(np.mean(low)) if low else np.nan)


def pattern_loss(sim: PatternTargets, obs: PatternTargets, weights=None) -> float:
    s, o = sim.as_array(), obs.as_array()
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float)
    z = (s - o) / PATTERN_SCALE
    ok = np.isfinite(z)
    return float(np.sum(w[ok] * z[ok] ** 2) / max(np.sum(w[ok]), 1e-9))


def abc_smc(loss_fn, priors: dict, n_particles: int = 48, n_rounds: int = 4, keep_frac: float = 0.3,
            seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Population Monte-Carlo ABC.

    Round 0 samples the prior; each later round perturbs the best
    ``keep_frac`` particles with a Gaussian kernel of twice their weighted
    covariance (truncated to the prior box). Returns all evaluated
    particles with their losses and round index.
    """
    rng = np.random.default_rng(seed)
    names = list(priors)
    lo = np.array([priors[k][0] for k in names]); hi = np.array([priors[k][1] for k in names])
    particles = rng.uniform(lo, hi, (n_particles, len(names)))
    rows = []
    for rd in range(n_rounds):
        losses = np.array([loss_fn(dict(zip(names, p))) for p in particles])
        for p, l in zip(particles, losses):
            rows.append({**dict(zip(names, p)), "loss": l, "round": rd})
        if verbose:
            log.info("ABC round %d: best loss %.3f, median %.3f", rd, np.nanmin(losses), np.nanmedian(losses))
        if rd == n_rounds - 1:
            break
        k = max(2, int(keep_frac * n_particles))
        keep = particles[np.argsort(losses)[:k]]
        cov = 2.0 * np.cov(keep.T) + 1e-9 * np.eye(len(names))
        parents = keep[rng.integers(0, k, n_particles)]
        particles = np.clip(parents + rng.multivariate_normal(np.zeros(len(names)), cov, n_particles), lo, hi)
    return pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)


def calibrate_interactions(df_races: pd.DataFrame, observed: PatternTargets | None = None,
                           base_cfg: SimConfig | None = None, priors: dict | None = None,
                           n_sims: int = 150, n_particles: int = 32, n_rounds: int = 3,
                           max_races: int = 200, abilities: str = "composite", seed: int = 0) -> dict:
    """Fit the interaction parameters on a sample of real races.

    Returns the best SimConfig, the particle table and the observed vs
    simulated patterns at the optimum.
    """
    d = ensure_raceid(df_races)
    observed = observed or observed_patterns(d)
    priors = priors or DEFAULT_PRIORS
    base_cfg = base_cfg or SimConfig()
    race_ids = d["raceid"].unique()[:max_races]
    fields = []
    for i, rid in enumerate(race_ids):
        g = d[d["raceid"] == rid]
        if len(g) < 4:
            continue
        fld = build_field(g, abilities=abilities, rng=np.random.default_rng(seed + i))
        fields.append((fld, Track.from_race(fld.dist_furlongs, fld.going, fld.track_name)))

    def loss_fn(theta: dict) -> float:
        cfg = replace(base_cfg, **theta)
        sim = RaceSimulator(cfg)
        results = [sim.run(f, t, n_sims=n_sims, seed=seed + j) for j, (f, t) in enumerate(fields)]
        return pattern_loss(simulated_patterns(results), observed)

    table = abc_smc(loss_fn, priors, n_particles=n_particles, n_rounds=n_rounds, seed=seed)
    best = table.iloc[0]
    best_cfg = replace(base_cfg, **{k: float(best[k]) for k in priors})
    sim = RaceSimulator(best_cfg)
    results = [sim.run(f, t, n_sims=n_sims, seed=seed + j) for j, (f, t) in enumerate(fields)]
    return {"config": best_cfg, "params": {k: float(best[k]) for k in priors}, "loss": float(best["loss"]),
            "particles": table, "observed": asdict(observed), "simulated": asdict(simulated_patterns(results)),
            "n_races": len(fields)}
