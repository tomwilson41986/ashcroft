"""
ABM outputs -> features for the LightGBM BFSP model.

The ABM is a feature generator, not a replacement model. Its win
probability is dominated by the same ability signal the statistical model
already has, so the useful columns are the *decomposition*:

    abm_win_prob        full simulation (ability + pace + traffic + draw)
    abm_win_prob_solo   interaction-free simulation (ability + own effort)
    abm_pace_delta      the difference: what tactics/traffic add or remove
    abm_place_prob      place probability under UK each-way terms
    abm_exp_beaten_l    expected lengths behind the winner (margin model)
    abm_p_led_early, abm_p_led_2f, abm_p_trouble, abm_p_blocked_final
    abm_pace_contest    expected number of horses disputing the lead
    abm_pace_collapse_p P(halfway leader finishes outside the first three)
    abm_race_entropy    entropy of the simulated win distribution
    abm_win_sd          Monte-Carlo standard error of abm_win_prob
    rAbmWin             within-race rank of abm_win_prob (1 = most likely)

``add_abm_features`` runs one simulation per race. It is CPU-bound
(~0.3-1 s per race at 500 sims), so compute it for the evaluation window,
persist the result, and merge it into training with train_bfsp.py
--abm-features.
"""

from __future__ import annotations

import logging
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from model.abm.agents import AbilityConfig, build_field
from model.abm.simulate import RaceSimulator, SimConfig, SimResult
from model.abm.track import Track
from model.perf_figures import ensure_raceid

log = logging.getLogger(__name__)

ABM_FEATURES = [
    "abm_win_prob", "abm_win_prob_solo", "abm_pace_delta", "abm_place_prob", "abm_exp_pos",
    "abm_exp_beaten_l", "abm_p_led_early", "abm_p_led_2f", "abm_p_trouble", "abm_p_blocked_final",
    "abm_exp_w_left", "abm_pace_contest", "abm_pace_collapse_p", "abm_race_entropy", "abm_win_sd",
    "rAbmWin",
]
ABM_KEY_COLS = ["race_date", "race_time", "track", "horse_name"]


def place_terms(n_runners: int, is_handicap: bool = False) -> int:
    """UK each-way places paid by field size."""
    if n_runners < 5:
        return 1
    if n_runners <= 7:
        return 2
    if n_runners <= 15:
        return 3
    return 4 if is_handicap else 3


def summarise(res: SimResult, res_solo: SimResult | None = None, is_handicap: bool = False) -> pd.DataFrame:
    """One row per runner with the ABM feature columns."""
    S, N = res.positions.shape
    places = place_terms(N, is_handicap)
    win = (res.positions == 1).mean(axis=0)
    p_place = (res.positions <= places).mean(axis=0)
    p_win_solo = (res_solo.positions == 1).mean(axis=0) if res_solo is not None else np.full(N, np.nan)
    finite_m = np.where(np.isfinite(res.margins_lengths), res.margins_lengths, np.nan)
    ent = -np.sum(np.where(win > 0, win * np.log(win), 0.0))
    collapse = np.nanmean(res.leader_half_pos > 3) if np.isfinite(res.leader_half_pos).any() else np.nan
    df = pd.DataFrame({
        "horse_name": res.names,
        "abm_win_prob": win,
        "abm_win_prob_solo": p_win_solo,
        "abm_pace_delta": win - p_win_solo,
        "abm_place_prob": p_place,
        "abm_exp_pos": res.positions.mean(axis=0),
        "abm_exp_beaten_l": np.nanmean(finite_m, axis=0),
        "abm_p_led_early": res.led_early.mean(axis=0),
        "abm_p_led_2f": res.led_2f_out.mean(axis=0),
        "abm_p_trouble": res.trouble.mean(axis=0),
        "abm_p_blocked_final": (res.blocked_final_s > 1.0).mean(axis=0),
        "abm_exp_w_left": res.w_left_frac.mean(axis=0),
        "abm_pace_contest": np.nanmean(res.n_contesting) if np.isfinite(res.n_contesting).any() else np.nan,
        "abm_pace_collapse_p": collapse,
        "abm_race_entropy": ent,
        "abm_win_sd": np.sqrt(win * (1 - win) / S),
    })
    df["rAbmWin"] = df["abm_win_prob"].rank(ascending=False, method="min")
    return df


def simulate_race(df_race: pd.DataFrame, n_sims: int = 500, seed: int = 42, abilities: str = "composite",
                  sim_cfg: SimConfig | None = None, ability_cfg: AbilityConfig | None = None,
                  with_solo: bool = True) -> tuple[pd.DataFrame, SimResult, SimResult | None]:
    """Simulate one race (DataFrame of runners) and return features + raw results."""
    fld = build_field(df_race, abilities=abilities, cfg=ability_cfg, rng=np.random.default_rng(seed))
    track = Track.from_race(fld.dist_furlongs, fld.going, fld.track_name)
    sim = RaceSimulator(sim_cfg or SimConfig())
    res = sim.run(fld, track, n_sims=n_sims, seed=seed, interactions=True)
    res_solo = sim.run(fld, track, n_sims=n_sims, seed=seed, interactions=False) if with_solo else None
    is_hcp = False
    if "race_type" in df_race.columns:
        is_hcp = df_race["race_type"].astype(str).str.lower().str.contains("h", regex=False).any()
    feats = summarise(res, res_solo, is_handicap=is_hcp)
    return feats, res, res_solo


def _simulate_one(args):
    """Worker: simulate one race, return (row index, feature frame) or None."""
    idx, g, n_sims, seed, abilities, sim_cfg, ability_cfg, with_solo = args
    try:
        feats, _, _ = simulate_race(g, n_sims=n_sims, seed=seed, abilities=abilities,
                                    sim_cfg=sim_cfg, ability_cfg=ability_cfg, with_solo=with_solo)
    except Exception as exc:  # keep going on odd rows; report the race
        log.warning("ABM failed for race %s: %s", g.iloc[0].get("raceid", "?"), exc)
        return None
    return idx, feats[ABM_FEATURES].values


def add_abm_features(df: pd.DataFrame, n_sims: int = 500, seed: int = 42, abilities: str = "composite",
                     race_col: str = "raceid", sim_cfg: SimConfig | None = None,
                     ability_cfg: AbilityConfig | None = None, with_solo: bool = True,
                     max_races: int | None = None, min_runners: int = 3, log_every: int = 200,
                     n_jobs: int = 1) -> pd.DataFrame:
    """Run the ABM for every race in ``df`` and attach the ABM feature columns.

    Rows are matched back by (race id, horse_name). Races with fewer than
    ``min_runners`` runners get NaN features. ``n_jobs`` > 1 fans races out
    across processes (the workload is embarrassingly parallel).
    """
    df = ensure_raceid(df) if race_col == "raceid" else df
    out = df.copy()
    for c in ABM_FEATURES:
        out[c] = np.nan
    race_ids = out[race_col].unique()
    if max_races:
        race_ids = race_ids[:max_races]
    jobs = []
    for i, rid in enumerate(race_ids):
        idx = out.index[out[race_col] == rid]
        if len(idx) < min_runners:
            continue
        jobs.append((idx, out.loc[idx], n_sims, seed + i, abilities, sim_cfg, ability_cfg, with_solo))
    t0 = time.time()
    results = Pool(n_jobs).imap_unordered(_simulate_one, jobs, chunksize=4) if n_jobs > 1 else map(_simulate_one, jobs)
    for i, r in enumerate(results):
        if r is None:
            continue
        idx, vals = r
        out.loc[idx, ABM_FEATURES] = vals
        if log_every and (i + 1) % log_every == 0:
            el = time.time() - t0
            log.info("ABM features: %d/%d races (%.1fs, %.2fs/race)", i + 1, len(jobs), el, el / (i + 1))
    return out


def load_abm_features(path: str) -> pd.DataFrame:
    """Load a persisted ABM feature file (parquet or csv) keyed by ABM_KEY_COLS."""
    feats = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in ABM_KEY_COLS if c not in feats.columns]
    if missing:
        raise ValueError(f"ABM feature file missing key columns: {missing}")
    feats["race_date"] = pd.to_datetime(feats["race_date"])
    return feats[ABM_KEY_COLS + [c for c in ABM_FEATURES if c in feats.columns]]


def merge_abm_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Left-merge persisted ABM features onto a training frame."""
    d = df.copy()
    d["race_date"] = pd.to_datetime(d["race_date"])
    d = d.drop(columns=[c for c in ABM_FEATURES if c in d.columns], errors="ignore")
    return d.merge(feats, on=ABM_KEY_COLS, how="left")
