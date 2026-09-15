"""
Vectorised Monte-Carlo race simulator.

State arrays are (S, N): S independent simulations of one N-runner race,
advanced together with a fixed time step. Per step:

    1. jockey policy  -> target speed (early effort, follow-the-leader with a
                         desired gap, contested/uncontested lead, final kick)
    2. physiology     -> cap by sustainable-speed ceiling + remaining surge
    3. environment    -> bend speed loss
    4. traffic        -> IDM-style blocking behind a slower horse in the same
                         lane, lane changes to pass, rail-seeking, stochastic
                         interference ("hampered") while boxed in
    5. drafting       -> reduced anaerobic cost when tucked in behind
    6. integrate      -> speed lag, W' balance, rail progress (wide lanes lose
                         ground on bends), finishing-time interpolation

``interactions=False`` turns off 4-5 and the follow policy: every horse runs
its own race. The difference between the two runs isolates the tactical /
traffic signal the statistical model cannot see (see features.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model.abm.agents import FieldParams
from model.abm.energy import EnergyConfig, speed_ceiling, step_energy, step_speed, surge_available
from model.abm.track import Track
from model.perf_figures import LENGTH_METRES, lbs_to_speed_factor


@dataclass
class SimConfig:
    dt: float = 0.25
    n_sims: int = 1000
    seed: int = 42
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    interactions: bool = True
    ability_noise: bool = True
    process_noise_sd: float = 0.15       # m/s jitter on the target speed per step
    early_phase_frac: float = 0.35
    # traffic
    s0: float = 1.5                      # m, minimum nose-to-tail gap
    headway_T: float = 0.10              # s, safe time headway
    lane_tol: float = 0.8                # lanes closer than this interact
    pass_window_m: float = 2.0           # a lane is free if no horse within this along-track window (bends)
    pass_window_straight_m: float = 1.5  # horses fan out and squeeze through on the straight
    p_switch_out: float = 1.5            # per second, when blocked and wanting to pass
    p_switch_in: float = 1.0             # per second, drift toward the rail when free
    straight_fan_m: float = 600.0        # in the last straight, blocked horses also switch out to find a run
    trouble_hazard: float = 0.01         # per second while boxed in with traffic (calibrate: abm-calibrate)
    trouble_speed_drop: float = 0.08
    trouble_w_cost: float = 3.0
    crowd_radius_m: float = 5.0
    # drafting
    draft_min: float = 0.5
    draft_max: float = 3.0
    draft_saving: float = 0.15           # fraction of anaerobic cost saved behind one horse
    draft_saving_double: float = 0.25    # behind two or more
    # pace policy
    lead_effort_uncontested: float = 0.38
    lead_effort_contested: float = 0.65
    front_style_min: float = 4.5         # runners at/above this early-position figure try to lead
    establish_m: float = 150.0           # metres over which front-runners push to establish the lead
    establish_effort: float = 0.5        # effort (x dv) used while establishing the lead
    stall_spread: float = 0.35           # initial lanes per stall (field compresses into ranks quickly)
    max_lane: float = 6.0
    early_rail_boost: float = 2.0        # multiplier on inward drift during the run to the first bend
    contest_gap_m: float = 2.5
    follow_gain: float = 0.12            # 1/s
    follow_min_frac: float = 0.96        # a follower will not drop below this x v_cp mid-race
    solo_effort: float = 0.35            # even-pace effort used by the interaction-free reference run
    # environment
    bend_loss: float = 0.015             # speed loss at the reference radius
    bend_ref_radius: float = 150.0
    bend_energy_mult: float = 1.5        # extra anaerobic cost on bends (x kappa x ref radius)
    max_time_factor: float = 1.6         # abort at max_time_factor x par time


@dataclass
class SimResult:
    names: list
    times: np.ndarray            # (S, N) finishing time s
    positions: np.ndarray        # (S, N) finishing rank 1..N
    margins_lengths: np.ndarray  # (S, N) lengths behind the winner
    early_rank: np.ndarray       # (S, N) rank at 25% of the trip
    led_early: np.ndarray        # (S, N) bool, leader at 25%
    led_2f_out: np.ndarray       # (S, N) bool, leader 2f out
    trouble: np.ndarray          # (S, N) bool, suffered interference
    blocked_final_s: np.ndarray  # (S, N) seconds boxed in *while wanting to pass* over the last 600 m
    w_left_frac: np.ndarray      # (S, N) W' fraction at the line
    lane_mean: np.ndarray        # (S, N) mean lane over the race
    n_contesting: np.ndarray     # (S,) horses within contest_gap of the leader at 15%
    leader_half_pos: np.ndarray  # (S,) finishing position of the horse leading at halfway
    ability_lbs_drawn: np.ndarray  # (S, N) ability + form noise actually used
    field: FieldParams
    track: Track
    config: SimConfig


class RaceSimulator:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()

    # ------------------------------------------------------------------
    def run(self, fld: FieldParams, track: Track | None = None, n_sims: int | None = None,
            seed: int | None = None, interactions: bool | None = None) -> SimResult:
        cfg = self.cfg
        ecfg = cfg.energy
        S = int(n_sims or cfg.n_sims)
        N = fld.n
        inter = cfg.interactions if interactions is None else interactions
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        track = track or Track.from_race(fld.dist_furlongs, fld.going, fld.track_name)
        L = track.length_m
        dt = cfg.dt

        # --- per-simulation parameter draws -----------------------------
        lbs_noise = rng.normal(0.0, fld.form_sd_lbs[None, :], (S, N)) if cfg.ability_noise else np.zeros((S, N))
        v_cp = fld.v_cp[None, :] * lbs_to_speed_factor(lbs_noise, fld.dist_furlongs)
        dv = fld.dv_surge[None, :] * (1.0 + rng.normal(0.0, 0.05, (S, N)))
        w0 = fld.w_prime[None, :] * np.clip(1.0 + rng.normal(0.0, 0.10, (S, N)), 0.5, 1.5)
        keen = rng.random((S, N)) < fld.keenness[None, :]
        kick = fld.kick_dist_m[None, :] * np.clip(1.0 + rng.normal(0.0, 0.10, (S, N)), 0.5, 1.5)
        early_effort = np.clip(fld.early_effort[None, :] + rng.normal(0.0, 0.05, (S, N)), 0.05, 0.95)
        desired_gap = fld.desired_gap_m[None, :] * np.clip(1.0 + rng.normal(0.0, 0.15, (S, N)), 0.3, 2.0)
        # the field forms ranks within the first 100 m: initial spread is a
        # compressed image of the stalls (10 stalls -> ~3.5 horse-widths)
        lane = cfg.stall_spread * (np.broadcast_to(fld.stall[None, :] - 1.0, (S, N)).astype(float)) + rng.normal(0, 0.1, (S, N))
        lane = np.clip(lane, 0.0, cfg.max_lane)
        rail_seek = fld.rail_seek[None, :]
        trouble_prone = fld.trouble_prone[None, :]
        front_style = np.broadcast_to(fld.style[None, :] >= cfg.front_style_min, (S, N))
        if not inter:
            # neutralise trait spreads and lanes so the reference run is ability + form noise only
            dv = np.full((S, N), dv.mean()); w0 = np.full((S, N), w0.mean()); kick = np.full((S, N), kick.mean())
            lane = np.zeros((S, N))

        # --- state -------------------------------------------------------
        x = np.zeros((S, N)); v = np.full((S, N), ecfg.v_start); w = w0.copy()
        finished = np.zeros((S, N), dtype=bool); t_fin = np.full((S, N), np.nan)
        trouble = np.zeros((S, N), dtype=bool); blocked_final = np.zeros((S, N)); lane_sum = np.zeros((S, N))
        early_rank = np.full((S, N), np.nan); led_early = np.zeros((S, N), dtype=bool)
        led_2f = np.zeros((S, N), dtype=bool); n_contest = np.full(S, np.nan); leader_half = np.full(S, -1)
        snap_early = np.zeros(S, dtype=bool); snap_2f = np.zeros(S, dtype=bool)
        snap_contest = np.zeros(S, dtype=bool); snap_half = np.zeros(S, dtype=bool)
        v_at_finish = np.full((S, N), np.nan)

        v_par = float(fld.meta.get("v_par", v_cp.mean()))
        t_max = cfg.max_time_factor * L / v_par
        n_steps = int(np.ceil(t_max / dt))
        rows = np.arange(S)[:, None]
        eye = np.eye(N, dtype=bool)[None, :, :]

        for step in range(n_steps):
            t = step * dt
            active = ~finished
            if not active.any():
                break
            kappa = track.curvature_at(x)
            frac = x / L
            w_frac = w / np.where(w0 > 0, w0, 1.0)

            # leader per sim (among active; finished horses are at L and excluded)
            x_act = np.where(active, x, -np.inf)
            lead_idx = np.argmax(x_act, axis=1)
            x_lead = x_act[np.arange(S), lead_idx][:, None]
            v_lead = v[np.arange(S), lead_idx][:, None]
            gap_lead = x_lead - x
            is_leader = active & (gap_lead <= 0.05)

            # --- jockey policy: target speed ------------------------------
            in_kick = active & (x >= L - kick)
            early = frac < cfg.early_phase_frac
            if inter:
                close = active & (gap_lead < cfg.contest_gap_m) & ~is_leader
                n_close = close.sum(axis=1, keepdims=True)
                contested = (n_close >= 1) & early
                lead_effort = np.where(contested, cfg.lead_effort_contested, cfg.lead_effort_uncontested)
                # followers track the leader's speed and settle into their preferred gap
                follow_target = v_lead + cfg.follow_gain * (gap_lead - desired_gap)
                follow_target = np.clip(follow_target, cfg.follow_min_frac * v_cp, v_cp + dv)
                # front-runners push early to establish the lead, then set their own fractions
                establishing = front_style & (x < cfg.establish_m)
                target = np.where(establishing, v_cp + cfg.establish_effort * dv,
                                  np.where(is_leader, v_cp + lead_effort * dv, follow_target))
                target = np.where(keen & early & ~is_leader, target + 0.3, target)
            else:
                # solo reference: everyone runs the same even-pace race, so the
                # result is ability + form noise only (a Harville-like baseline)
                target = v_cp + cfg.solo_effort * dv
            target = np.where(in_kick, v_cp + dv, target)

            # --- physiology caps --------------------------------------------
            ceiling = speed_ceiling(v_cp, w_frac, ecfg)
            target = np.minimum(target, ceiling + surge_available(dv, w_frac, ecfg))
            # --- environment --------------------------------------------------
            target = target * (1.0 - cfg.bend_loss * kappa * cfg.bend_ref_radius)

            cost_mult = np.full((S, N), track.going_energy) * (1.0 + cfg.bend_energy_mult * kappa * cfg.bend_ref_radius * 0.01)
            cost_mult = np.where(keen & early, cost_mult * ecfg.keen_cost_multiplier, cost_mult)

            if inter:
                # --- pairwise geometry (S, N, N): entry [s, i, j] = horse j relative to i
                dx = x[:, None, :] - x[:, :, None]
                dl = np.abs(lane[:, None, :] - lane[:, :, None])
                both = active[:, :, None] & active[:, None, :] & ~eye
                same_lane = both & (dl < cfg.lane_tol)
                ahead = same_lane & (dx > 0)
                gap_ahead = np.where(ahead, dx, np.inf).min(axis=2)
                j_ahead = np.where(ahead, dx, np.inf).argmin(axis=2)
                v_ahead = v[rows, j_ahead]
                blocked = active & np.isfinite(gap_ahead) & (gap_ahead < cfg.s0 + v * cfg.headway_T)
                want_pass = blocked & (target > v_ahead + 0.15)
                target = np.where(blocked, np.minimum(target, v_ahead), target)

                # lane changes: out to pass, in toward the rail when free
                window = np.where(kappa > 0, cfg.pass_window_m, cfg.pass_window_straight_m)[:, :, None]
                lane_out = lane[:, :, None] + 1.0
                occ_out = (both & (np.abs(lane[:, None, :] - lane_out) < cfg.lane_tol) & (np.abs(dx) < window)).any(axis=2)
                seek_run = want_pass | (blocked & (x > L - cfg.straight_fan_m))
                switch_out = seek_run & ~occ_out & (rng.random((S, N)) < cfg.p_switch_out * dt)
                lane_in = lane[:, :, None] - 1.0
                occ_in = (both & (np.abs(lane[:, None, :] - lane_in) < cfg.lane_tol) & (np.abs(dx) < window)).any(axis=2)
                early_run = x < min(track.run_to_bend_m, 0.2 * L)
                p_in = cfg.p_switch_in * rail_seek * np.where(early_run, cfg.early_rail_boost, 1.0) * dt
                switch_in = active & (lane >= 1.0) & ~occ_in & ~blocked & (rng.random((S, N)) < p_in)
                lane = lane + switch_out - switch_in
                lane = np.clip(lane, 0.0, cfg.max_lane)

                # interference while boxed in with traffic around
                crowd = (both & (np.abs(dx) < cfg.crowd_radius_m) & (dl < 2.0)).sum(axis=2)
                hit = blocked & want_pass & (crowd >= 2) & (rng.random((S, N)) < cfg.trouble_hazard * trouble_prone * dt)
                v = np.where(hit, v * (1.0 - cfg.trouble_speed_drop), v)
                w = np.where(hit, np.clip(w - cfg.trouble_w_cost, 0, None), w)
                trouble |= hit
                blocked_final += np.where(want_pass & (x > L - 600.0), dt, 0.0)

                # drafting
                drafting = active & np.isfinite(gap_ahead) & (gap_ahead > cfg.draft_min) & (gap_ahead < cfg.draft_max)
                n_shield = (both & (dx > cfg.draft_min) & (dx < cfg.draft_max) & (dl < 1.6)).sum(axis=2)
                saving = np.where(n_shield >= 2, cfg.draft_saving_double, np.where(drafting, cfg.draft_saving, 0.0))
                cost_mult = cost_mult * (1.0 - saving)

            target = target + rng.normal(0.0, cfg.process_noise_sd, (S, N))
            target = np.where(active, target, v)

            # --- integrate ----------------------------------------------------
            v_new = step_speed(v, target, dt, ecfg)
            w = step_energy(w, v_new, v_cp, w0, dt, ecfg, cost_mult)
            progress = v_new * dt / (1.0 + lane * track.lane_width_m * kappa)
            x_new = np.where(active, x + progress, x)

            crossed = active & (x_new >= L)
            if crossed.any():
                t_cross = t + (L - x) / np.where(progress > 0, progress, 1e-9) * dt
                t_fin = np.where(crossed, t_cross, t_fin)
                v_at_finish = np.where(crossed, v_new, v_at_finish)
                finished |= crossed
                x_new = np.where(crossed, L, x_new)

            # --- snapshots -------------------------------------------------------
            new_lead_x = np.where(finished, L, x_new).max(axis=1)
            take = ~snap_early & (new_lead_x >= 0.25 * L)
            if take.any():
                order = np.argsort(-x_new[take], axis=1)
                ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(1, N + 1)[None, :].repeat(take.sum(), 0), axis=1)
                early_rank[take] = ranks
                led_early[take] = ranks == 1
                snap_early[take] = True
            take = ~snap_contest & (new_lead_x >= 0.15 * L)
            if take.any():
                xl = x_new[take].max(axis=1, keepdims=True)
                n_contest[take] = ((xl - x_new[take]) < cfg.contest_gap_m).sum(axis=1) - 1
                snap_contest[take] = True
            take = ~snap_half & (new_lead_x >= 0.5 * L)
            if take.any():
                leader_half[take] = np.argmax(x_new[take], axis=1)
                snap_half[take] = True
            take = ~snap_2f & (new_lead_x >= L - 2 * 201.168)
            if take.any():
                led_2f[take] = x_new[take] >= x_new[take].max(axis=1, keepdims=True) - 1e-6
                snap_2f[take] = True

            lane_sum += lane * dt
            x, v = x_new, v_new

        # --- wrap up ---------------------------------------------------------
        unfinished = np.isnan(t_fin)
        if unfinished.any():
            t_fin = np.where(unfinished, t_max + (L - x) / np.maximum(v_cp, 1.0), t_fin)
            v_at_finish = np.where(unfinished, v_cp, v_at_finish)
        order = np.argsort(t_fin, axis=1)
        positions = np.empty_like(order); np.put_along_axis(positions, order, np.arange(1, N + 1)[None, :].repeat(S, 0), axis=1)
        t_win = t_fin.min(axis=1, keepdims=True)
        margins = (t_fin - t_win) * np.nanmean(v_at_finish, axis=1, keepdims=True) / LENGTH_METRES
        lhp = np.where(leader_half >= 0, positions[np.arange(S), np.clip(leader_half, 0, N - 1)], np.nan)
        return SimResult(names=list(fld.names), times=t_fin, positions=positions, margins_lengths=margins,
                         early_rank=early_rank, led_early=led_early, led_2f_out=led_2f, trouble=trouble,
                         blocked_final_s=blocked_final, w_left_frac=w / np.where(w0 > 0, w0, 1.0),
                         lane_mean=lane_sum / max(t_fin.max(), dt), n_contesting=n_contest,
                         leader_half_pos=lhp, ability_lbs_drawn=fld.ability_lbs[None, :] + lbs_noise,
                         field=fld, track=track, config=cfg)
