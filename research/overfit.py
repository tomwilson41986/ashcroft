"""The overfitting check every promotion gate reads (the owner, 5 Nov 2026: "wire in").

A rule picked as the best of N tried has a Sharpe that is the maximum of N draws, not one draw. The deflated Sharpe
ratio (Bailey and Lopez de Prado 2014, ``model.uncertainty.deflated_sharpe_ratio``) is the probability that the
rule's true Sharpe beats the best Sharpe N rules with no edge would have reached by luck. A gate passes a rule only
when that probability is at least ``DSR_THRESHOLD``.

    dsr_check(returns, n_trials, trial_sharpes)   the check on one rule's per-period returns
    deflate_table(rows, series, ...)               the check on every rule of a table at once, each deflated by all
    TrialLedger(path)                              the honest count: every rule a programme has ever scored, kept in
                                                   a JSON file (local or s3://bucket/key) so the count outlives a run

The returns are a rule's per-period returns: per race, per day or per bet, the same period for every rule compared.
This module only reads ``model.uncertainty``; the live horse model and its jobs are not touched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from model.uncertainty import _return_moments, deflated_sharpe_ratio

log = logging.getLogger(__name__)

DSR_THRESHOLD = 0.95     # Bailey and Lopez de Prado's bar: 95% that the Sharpe is not the luck of the best of N
MIN_OBS = 30             # fewer periods than this: no Sharpe worth deflating, the check fails


def null_sr_variance(n_obs: int) -> float:
    """The variance of a Sharpe estimate over ``n_obs`` periods when the true Sharpe is zero: 1 / (n - 1). It stands
    in for the spread of the trials' Sharpes when they are not known (a registered count with no per-trial results)."""
    return 1.0 / (n_obs - 1) if n_obs and n_obs > 1 else float("nan")


def dsr_check(returns, n_trials: int, trial_sharpes=None, threshold: float = DSR_THRESHOLD,
              min_obs: int = MIN_OBS) -> dict:
    """The deflated Sharpe of one rule's per-period ``returns``, deflated by ``n_trials`` rules tried.

    ``trial_sharpes`` (the Sharpe of every rule tried, on the same periods) gives the spread of the trials' Sharpes;
    without at least two, the null spread ``null_sr_variance`` is used. The spread and the count must describe the
    same trials: when fewer Sharpes are given than ``n_trials`` counts (a few rules' Sharpes against a ledger's
    count), their spread is floored at the null spread, so a handful of near-identical rules cannot collapse the
    expected maximum to zero (audit ME-02). ``passed`` is True only with ``min_obs`` periods and a DSR at or above
    ``threshold``; a check that cannot be computed never passes.

    The check deflates whatever series it is given. A low-variance series such as daily CLV (a mean of price
    ratios, not a settled result) reaches a high daily Sharpe whatever the count, so for a rule's promotion pass
    its settled profit and loss a period (or CLV x stake net of commission) and keep CLV as the effect size
    (audit B7)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n_trials = max(int(n_trials or 1), 1)
    out = {"n_obs": len(r), "n_trials": n_trials, "threshold": threshold}
    if len(r) < min_obs:
        return {**out, "sharpe": None, "sr_star": None, "dsr": None, "passed": False,
                "reason": f"{len(r)} periods, fewer than {min_obs}"}
    t = np.asarray(trial_sharpes if trial_sharpes is not None else [], float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        var, source = null_sr_variance(len(r)), "null"
    elif len(t) >= n_trials:
        var, source = float(t.var(ddof=1)), "trials"
    else:                    # fewer Sharpes than trials counted: never a spread below the null's
        var = max(float(t.var(ddof=1)), null_sr_variance(len(r)))
        source = "trials" if var > null_sr_variance(len(r)) else "null floor"
    res = deflated_sharpe_ratio(r, n_trials=n_trials, sr_variance=var)
    dsr = res["dsr"]
    ok = bool(np.isfinite(dsr) and dsr >= threshold)
    return {**out, "sharpe": _round(res["sharpe"]), "sr_star": _round(res["sr_star"]), "dsr": _round(dsr),
            "sr_variance": _round(var), "sr_variance_source": source, "passed": ok,
            "reason": "" if ok else (f"DSR {dsr:.3f} below {threshold}" if np.isfinite(dsr) else "DSR not computable")}


def deflate_table(series: dict, n_trials: int | None = None, ledger: TrialLedger | None = None,
                  threshold: float = DSR_THRESHOLD, min_obs: int = MIN_OBS) -> dict:
    """The check on every rule of a table at once: ``series`` maps each rule's name to its per-period returns.

    Every rule is deflated by the same count: the larger of ``n_trials``, the rules in the table, and (with a
    ``ledger``) every rule the programme has ever scored, the table's rules recorded first. The spread of the
    trials' Sharpes is taken from the table's rules and the ledger's, all on the same kind of period."""
    sharpes = {k: _return_moments(v)[0] for k, v in series.items()}
    if ledger is not None:
        for k, v in series.items():
            ledger.record(k, returns=v)
        ledger.save()
        pool = ledger.sharpes()
        count = max(ledger.n_trials, len(series), int(n_trials or 0))
    else:
        pool = np.array(list(sharpes.values()), float)
        count = max(len(series), int(n_trials or 0))
    return {k: dsr_check(v, count, pool, threshold, min_obs) for k, v in series.items()}


class TrialLedger:
    """Every rule a research programme has ever scored, by name, with its latest Sharpe and moments.

    Keyed by name, so re-scoring the same rule each night updates it and does not inflate the count; a new rule adds
    one. Kept in a JSON file: a local path, or ``s3://bucket/key`` (boto3, the job's AWS credentials). A ledger that
    cannot be read starts empty and says so; it is never silently smaller than the rules in hand (``deflate_table``
    takes the larger count)."""

    def __init__(self, path: str | Path | None = None, programme: str = ""):
        self.path, self.programme = (str(path) if path is not None else None), programme
        self.trials: dict[str, dict] = {}
        raw = self._read()
        if raw:
            self.trials = json.loads(raw).get("trials", {})

    # -- storage -------------------------------------------------------------------------------------------------
    def _s3(self):
        import boto3
        bucket, _, key = self.path[5:].partition("/")
        return boto3.client("s3"), bucket, key

    def _read(self) -> bytes | None:
        if not self.path:
            return None
        try:
            if self.path.startswith("s3://"):
                s3, b, k = self._s3()
                return s3.get_object(Bucket=b, Key=k)["Body"].read()
            p = Path(self.path)
            return p.read_bytes() if p.exists() else None
        except Exception as e:  # noqa: BLE001 -- a missing object is the first run; anything else is logged
            if "NoSuchKey" not in repr(e):
                log.warning("trial ledger %s not read (%s): starting empty", self.path, e)
            return None

    def save(self) -> None:
        if not self.path:
            return
        body = json.dumps({"programme": self.programme, "n_trials": self.n_trials, "trials": self.trials},
                          indent=1, default=float).encode()
        if self.path.startswith("s3://"):
            try:
                s3, b, k = self._s3()
                s3.put_object(Bucket=b, Key=k, Body=body, ContentType="application/json")
            except Exception as e:  # noqa: BLE001 -- the run keeps the count in hand; it is not carried over
                log.warning("trial ledger %s NOT saved (%s): the count will not carry over", self.path, e)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_bytes(body)

    # -- the count -----------------------------------------------------------------------------------------------
    def record(self, name: str, returns=None, sharpe: float | None = None, **meta) -> dict:
        e = dict(self.trials.get(name, {}), **meta)
        if returns is not None:
            sr, n, sk, ku = _return_moments(returns)
            e.update({"sharpe": _round(sr), "n_obs": int(n), "skew": _round(sk), "kurtosis": _round(ku)})
        if sharpe is not None:
            e["sharpe"] = float(sharpe)
        self.trials[str(name)] = e
        return e

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def sharpes(self) -> np.ndarray:
        return np.array([t.get("sharpe") if t.get("sharpe") is not None else np.nan for t in self.trials.values()],
                        float)


def _round(x, nd: int = 6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def pbo_cscv(returns, n_splits: int = 16) -> dict:
    """The probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein,
    Lopez de Prado and Zhu 2017), the owner's model brief of 18 Nov 2026 (Phase 1).

    ``returns`` is a periods x trials matrix: each column one rule or model tried, each row one period (a race day),
    the same periods for every trial. The rows are cut into ``n_splits`` contiguous blocks; for every way of taking
    half the blocks as in-sample, the trial with the best in-sample Sharpe is found and its rank among all trials on
    the other half read. The PBO is the share of those splits where the in-sample best lands at or below the
    out-of-sample median (logit of its relative rank <= 0). Near 0.5: picking the best tells nothing; near 0: the
    best in sample stays good out of sample.

    Returns {"pbo", "splits", "logits"}; with fewer than 2 trials or too few rows a block, {"pbo": None, ...}.
    """
    from itertools import combinations

    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or n_splits < 2 or n_splits % 2 or x.shape[0] < 2 * n_splits:
        return {"pbo": None, "splits": 0, "logits": []}
    blocks = np.array_split(np.arange(x.shape[0]), n_splits)

    def sharpe(m):
        sd = m.std(axis=0, ddof=1)
        return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0)

    n = x.shape[1]
    logits = []
    for ins in combinations(range(n_splits), n_splits // 2):
        ins_rows = np.concatenate([blocks[i] for i in ins])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in ins])
        best = int(np.argmax(sharpe(x[ins_rows])))
        oos = sharpe(x[oos_rows])
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1     # 1..n, ties shared
        w = rank / (n + 1.0)
        logits.append(float(np.log(w / (1.0 - w))))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "splits": len(logits), "logits": logits.tolist()}
