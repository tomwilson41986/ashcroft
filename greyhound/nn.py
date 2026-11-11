"""A neural challenger for the BFSP forecast (the BFSP plan, reports/greyhound_bfsp_plan.md, step 4): a set model over
each race's field, trained on the Betfair SP book's shares as the tree model's ``share`` objective is.

Each runner's features (rank-gauss transformed on the training rows, a missing flag where a feature is often missing)
pass through an MLP; the field's mean and maximum of those embeddings are joined back to each runner (a DeepSets
model: what a dog is worth depends on who it meets), then two layers give its score. The race's softmax of the scores
is fitted to the SP book's shares by cross-entropy (greyhound/fastloop.py ``race_soft_objective``'s loss), early-stopped
on the last tenth of the fit window, within a time budget. CPU PyTorch, deterministic on a fixed thread count.

    from greyhound import nn
    model = nn.fit(X_tr, race_tr, target_tr, X_es, race_es, target_es, seconds=2400)
    scores = nn.predict(model, X_test, race_test)
"""

from __future__ import annotations

import time

import numpy as np

QUANTILES = 101
MISSING_FLAG = 0.01          # a missing flag for each feature missing on more than this share of the training rows
HIDDEN = (128, 64)
BATCH_RACES = 2048
MAX_EPOCHS = 40
PATIENCE = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
THREADS = 4


class Transform:
    """Rank-gauss: each feature mapped through its training quantiles onto the normal scale; NaN to 0 with a flag."""

    def __init__(self, X: np.ndarray):
        from scipy.stats import norm
        qs = np.linspace(0, 1, QUANTILES)
        self.knots = np.nanquantile(X, qs, axis=0)                          # (QUANTILES, d)
        self.z = norm.ppf(np.clip(qs, 0.001, 0.999))
        self.flag = np.flatnonzero(np.isnan(X).mean(0) > MISSING_FLAG)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape, dtype=np.float32)
        for j in range(X.shape[1]):
            k = self.knots[:, j]
            if not np.isfinite(k).any():
                continue
            kk, idx = np.unique(k, return_index=True)
            v = X[:, j]
            ok = np.isfinite(v)
            out[ok, j] = np.interp(v[ok], kk, self.z[idx]) if len(kk) > 1 else 0.0
        flags = np.isnan(X[:, self.flag]).astype(np.float32)
        return np.concatenate([out, flags], axis=1)


def _net(d: int):
    import torch
    from torch import nn

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = HIDDEN
            self.phi = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(3 * h2, h2), nn.ReLU(), nn.Linear(h2, 1))

        def forward(self, x, race, n_races):
            h = self.phi(x)
            k = h.shape[1]
            idx = race.unsqueeze(1).expand(-1, k)
            s = torch.zeros(n_races, k).index_add_(0, race, h)
            cnt = torch.zeros(n_races).index_add_(0, race, torch.ones(len(race))).clamp(min=1)
            mean = (s / cnt.unsqueeze(1))[race]
            mx = torch.full((n_races, k), -1e9).scatter_reduce(0, idx, h, reduce="amax", include_self=True)[race]
            return self.rho(torch.cat([h, mean, mx], dim=1)).squeeze(1)

    return SetNet()


def _race_xent(score, race, target, n_races):
    """The races' cross-entropy of the softmax of ``score`` against the shares ``target``, a mean a race."""
    import torch
    mx = torch.full((n_races,), -1e9).scatter_reduce(0, race, score, reduce="amax", include_self=True)
    e = torch.exp(score - mx[race])
    lse = torch.log(torch.zeros(n_races).index_add_(0, race, e)) + mx
    return -(target * (score - lse[race])).sum() / n_races


def _batches(race: np.ndarray, rng, batch_races: int):
    """Row blocks of whole races (rows grouped by race), races shuffled into batches."""
    starts = np.r_[0, np.flatnonzero(race[1:] != race[:-1]) + 1, len(race)]
    order = rng.permutation(len(starts) - 1)
    for b in range(0, len(order), batch_races):
        pick = np.sort(order[b:b + batch_races])
        rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick])
        yield rows


def _tensors(Xt, race, target=None):
    import torch
    codes = np.r_[0, np.cumsum(race[1:] != race[:-1])]
    out = [torch.from_numpy(Xt), torch.from_numpy(codes.astype(np.int64)), int(codes[-1]) + 1 if len(codes) else 0]
    if target is not None:
        out.append(torch.from_numpy(np.nan_to_num(target).astype(np.float32)))
    return out


def fit(X_tr, race_tr, t_tr, X_es, race_es, t_es, seconds: float = 2400.0, seed: int = 0) -> dict:
    """The fitted model and its transform; early-stopped on the ``es`` rows' cross-entropy within ``seconds``. Rows
    grouped by race; ``t`` each runner's share of its race (rows of races whose shares do not sum to 1 are dropped)."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)

    def whole(race, t):
        s = np.bincount(np.r_[0, np.cumsum(race[1:] != race[:-1])], weights=np.nan_to_num(t))
        codes = np.r_[0, np.cumsum(race[1:] != race[:-1])]
        return (np.abs(s - 1) < 1e-6)[codes] & np.isfinite(t)

    k_tr, k_es = whole(race_tr, t_tr), whole(race_es, t_es)
    tf = Transform(X_tr[k_tr])
    Xtr, Xes = tf(X_tr[k_tr]), tf(X_es[k_es])
    rtr, tes_r, ttr, tes = race_tr[k_tr], race_es[k_es], t_tr[k_tr], t_es[k_es]
    net = _net(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    es_x, es_r, es_n, es_t = _tensors(Xes, tes_r, tes)
    best, best_state, bad, t0, hist = np.inf, None, 0, time.monotonic(), []
    for epoch in range(MAX_EPOCHS):
        net.train()
        for rows in _batches(rtr, rng, BATCH_RACES):
            x, r, n, t = _tensors(Xtr[rows], rtr[rows], ttr[rows])
            opt.zero_grad()
            loss = _race_xent(net(x, r, n), r, t, n)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(_race_xent(net(es_x, es_r, es_n), es_r, es_t, es_n))
        hist.append(round(v, 5))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE or time.monotonic() - t0 > seconds:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"net": net, "tf": tf, "epochs": len(hist), "es_xent": hist, "best": round(best, 5),
            "seconds": round(time.monotonic() - t0)}


def predict(model: dict, X: np.ndarray, race: np.ndarray, chunk_races: int = 20_000) -> np.ndarray:
    """The scores (log-odds within the race) of rows grouped by race."""
    import torch
    net, tf = model["net"], model["tf"]
    net.eval()
    starts = np.r_[0, np.flatnonzero(race[1:] != race[:-1]) + 1, len(race)]
    out = np.empty(len(race))
    with torch.no_grad():
        for b in range(0, len(starts) - 1, chunk_races):
            a, z = starts[b], starts[min(b + chunk_races, len(starts) - 1)]
            x, r, n = _tensors(tf(X[a:z]), race[a:z])
            out[a:z] = net(x, r, n).numpy()
    return out
