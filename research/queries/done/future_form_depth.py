"""Future form deeper: does the signal keep growing past five races back and five runs forward?

The per-cell read (future_form_residuals, run 36303528074) found the best
model's miss explained most by the rivals' later finishing positions, win
rates and figures from the last five races through their next five runs, the
deepest cells of the owner's grid, and growing with the depth both ways. This
builds the same block with the windows extended (the last 10 and 20 races back,
the next 10 runs forward) and reads every cell the same way: a quadratic,
demeaned within race, fitted to the best model's log error on five months and
applied to the sixth; the change in mean |log error| (x 10^4) with a
race-bootstrap interval; then groups jointly, the owner's grid against the grid
with the deeper windows. Read-only.
"""
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

#: iteration 87's research artifact (run 36284959031): r/oos_cw_dm_lr02_mcs500.csv
ARTIFACT_ID = 10923550944
OOS_NAME = "oos_cw_dm_lr02_mcs500.csv"
KEY = ["race_date", "race_time", "track", "horse_name"]
OUT = Path("out/future_form_depth")
#: the owner's grid and the deeper one read here
GRID = ((1, 3, 5), (1, 2, 3, 5))
DEEP = ((1, 3, 5, 10, 20), (1, 2, 3, 5, 10))
N_BOOT = 1000
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fetch_oos():
    """The best model's out-of-sample forecasts, from iteration 87's research artifact (actions: read)."""
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        say("no GITHUB_TOKEN: no forecasts to read")
        return None
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    z = requests.get(f"https://api.github.com/repos/{repo}/actions/artifacts/{ARTIFACT_ID}/zip", headers=auth,
                     timeout=600)
    if not z.ok:
        say(f"artifact {ARTIFACT_ID}: download failed ({z.status_code})")
        return None
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    name = next((n for n in zf.namelist() if n.endswith("/" + OOS_NAME) or n == OOS_NAME), None)
    if name is None:
        say(f"artifact {ARTIFACT_ID}: no {OOS_NAME} in {zf.namelist()[:10]}")
        return None
    return pd.read_csv(zf.open(name))


def matrix_with_blocks():
    """The cached matrix's rows, future form and form lines built on them as training builds them."""
    from model import blocks, feature_cache
    from model.blocks import form_lines, future_form
    key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
    cols = list(dict.fromkeys(KEY + list(future_form.READS) + list(form_lines.READS)))
    hit = feature_cache.load(".feature_cache", key, columns=cols)
    if hit is None:
        say(f"no cached matrix for key {key}: nothing to read")
        return None
    mat, _ = hit
    say(f"matrix: {len(mat):,} rows")
    deepen(future_form)
    mat, _ = blocks.attach_as_trained(mat, ["form_lines", "future_form"])
    say(f"future form ({len(future_form.FEATURES)} features, windows {future_form.BACK} back, "
        f"{future_form.FORWARD} forward) and form lines built")
    return mat


def names(back, forward, stats):
    return ([f"ff_l{b}_n{f}_{s}" for b in back for f in forward for s in stats]
            + [f"ff_l1_n{f}_adj" for f in forward])


def deepen(ff):
    """The block's windows extended in place: build() reads them from the module."""
    ff.BACK, ff.FORWARD = DEEP
    ff.FEATURES = names(DEEP[0], DEEP[1], ff.STATS)


def _norm_keys(df):
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"]).dt.normalize()
    for c in ("race_time", "track", "horse_name"):
        out[c] = out[c].astype(str).str.strip()
    return out


def demean(a, race):
    """Each column less its race mean (a: 2-D)."""
    means = pd.DataFrame(a).groupby(race).transform("mean").to_numpy()
    return a - means


def design(x, train):
    """z, z^2 and a missing flag; z standardised on the training rows."""
    miss = ~np.isfinite(x)
    mu = np.nanmean(x[train & ~miss]) if (train & ~miss).any() else 0.0
    sd = np.nanstd(x[train & ~miss]) if (train & ~miss).any() else 1.0
    z = np.where(miss, 0.0, (x - mu) / (sd if sd > 0 else 1.0))
    z = np.clip(z, -5, 5)
    cols = [z, z * z]
    if miss.any() and not miss.all():
        cols.append(miss.astype(float))
    return np.column_stack(cols)


def oof_adjustment(X, e_dm, race, month, ridge=0.0):
    """The fitted within-race adjustment for each row, each month from a fit on the other months."""
    Xd = demean(X, race)
    adj = np.zeros(len(e_dm))
    for m in np.unique(month):
        te = month == m
        tr = ~te
        A = Xd[tr]
        G = A.T @ A + ridge * np.eye(A.shape[1])
        try:
            beta = np.linalg.solve(G, A.T @ e_dm[tr])
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, e_dm[tr], rcond=None)[0]
        adj[te] = Xd[te] @ beta
    return adj


def delta_with_ci(e0, adj, race_codes, rng):
    """Change in mean |log error| from subtracting the adjustment, and a race-bootstrap 90% interval."""
    d = np.abs(e0 - adj) - np.abs(e0)
    sums = np.bincount(race_codes, weights=d)
    counts = np.bincount(race_codes).astype(float)
    n_r = len(sums)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, n_r, n_r)
        boots[i] = sums[pick].sum() / counts[pick].sum()
    return d.mean(), np.quantile(boots, 0.05), np.quantile(boots, 0.95)


def main():
    from model.blocks import form_lines, future_form as ff
    OUT.mkdir(parents=True, exist_ok=True)
    oos = fetch_oos()
    if oos is None:
        return 0
    say(f"forecasts: {len(oos):,} runners")
    mat = matrix_with_blocks()
    if mat is None:
        return 0
    feats = list(ff.FEATURES) + list(form_lines.FEATURES)
    left = _norm_keys(oos[KEY + ["predicted_bfsp", "bfsp"]]).drop_duplicates(KEY, keep=False)
    right = _norm_keys(mat[KEY + feats]).drop_duplicates(KEY, keep=False)
    del mat
    m = left.merge(right, on=KEY, how="inner")
    ok = (m["predicted_bfsp"] > 1) & (m["bfsp"] > 1)
    m = m[ok].reset_index(drop=True)
    say(f"joined: {len(m):,} runners of {len(oos):,}")
    if len(m) < 1000:
        return 0
    e0 = np.log(m["predicted_bfsp"].to_numpy(float)) - np.log(m["bfsp"].to_numpy(float))
    race = pd.factorize(m["race_date"].dt.strftime("%Y-%m-%d") + "|" + m["track"] + "|" + m["race_time"])[0]
    month = m["race_date"].dt.to_period("M").astype(str).replace({"2025-09": "2025-10"}).to_numpy()
    e_dm = demean(e0[:, None], race)[:, 0]
    print(f"\nThe model's mean |log error| on these runners: {np.abs(e0).mean():.4f}")
    rng = np.random.default_rng(0)
    m["placebo"] = np.random.default_rng(1).normal(size=len(m))
    ones = np.ones(len(m), bool)
    rows = []
    for c in feats + ["placebo"]:
        x = m[c].to_numpy(float)
        adj = oof_adjustment(design(x, ones), e_dm, race, month)
        d, lo, hi = delta_with_ci(e0, adj, race, rng)
        rows.append({"feature": c, "read": float(np.isfinite(x).mean()), "delta": d, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_feature.csv", index=False)
    d = res.set_index("feature")["delta"]
    lo = res.set_index("feature")["lo"]
    hi = res.set_index("feature")["hi"]
    print(f"  placebo {1e4 * d['placebo']:+.2f} ({1e4 * lo['placebo']:+.2f} to {1e4 * hi['placebo']:+.2f})")

    print("\nThe grid (x 10^4, * = interval below zero), back (rows) by forward (columns), each statistic")
    for s in list(ff.STATS) + ["adj"]:
        print(f"  {s}")
        print("        " + "".join(f"{'n' + str(f):>10s}" for f in ff.FORWARD))
        for b in ff.BACK:
            cells = [f"ff_l{b}_n{f}_{s}" for f in ff.FORWARD]
            if not any(c in d.index for c in cells):
                continue
            print(f"    l{b:<3d}" + "".join(
                f"{1e4 * d[c]:+9.2f}{'*' if hi[c] < 0 else ' '}" if c in d.index else f"{'':10s}" for c in cells))

    grid = names(GRID[0], GRID[1], ff.STATS)
    deep_only = [c for c in ff.FEATURES if c not in grid]
    groups = [("the owner's grid (l1-l5, n1-n5)", grid),
              ("the grid with the deeper windows", list(ff.FEATURES)),
              ("the deeper cells alone", deep_only),
              ("finishing positions, the grid", [c for c in grid if c.endswith("_nfp")]),
              ("finishing positions, every depth", [c for c in ff.FEATURES if c.endswith("_nfp")]),
              ("win rate, every depth", [c for c in ff.FEATURES if c.endswith("_wr")]),
              ("figures, every depth", [c for c in ff.FEATURES if c.endswith(("_perf", "_adj"))]),
              ("nfp, wr, perf at l5 and deeper", [c for c in ff.FEATURES if c.split("_")[1] in ("l5", "l10", "l20")
                                                  and c.endswith(("_nfp", "_wr", "_perf"))]),
              ("form lines (in the model)", list(form_lines.FEATURES))]
    print("\nGroups jointly (ridge 100, every feature's z, z^2 and missing flag):")
    for label, cols in groups:
        X = np.column_stack([design(m[c].to_numpy(float), ones) for c in cols])
        adj = oof_adjustment(X, e_dm, race, month, ridge=100.0)
        dd, l_, h_ = delta_with_ci(e0, adj, race, rng)
        print(f"  {label:40s} {len(cols):3d} features  {1e4 * dd:+7.2f} ({1e4 * l_:+7.2f} to {1e4 * h_:+7.2f})")
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
