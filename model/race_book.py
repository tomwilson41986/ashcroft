"""The race book: positions across a whole race, judged by the closing price (the owner's model of 30 Sep 2026).

The owner's aim: trade each market before the off; take positions against the horses the model rates as poor
value; make a new book from a subset of the runners (three to seven of them in some races), taking horses at our
own price or even small individual underlays so long as the race as a whole has positive price expectation; and
above all the sharpest closing-line value (CLV) possible. Three layers, each a function here.

1. **The closing price** (`ClosingModel`). What each runner will return at the off, forecast from the market's
   price now and our own. A conditional logit fitted to the Betfair SP book itself: for race r the closing
   probabilities ``q_i = (1/BSP_i) / sum_j 1/BSP_j`` are soft labels and ``q_hat = softmax(X beta)`` within the
   race, fitted by cross-entropy. X holds the log market probability now and the log model probability, each on
   its own and times the runner's traded volume and the race's, so the forecast leans on the market where money
   has traded and on the model where it has not. It is a whole book summing to 1. Its miss,
   ``log q_close - log q_hat``, has a spread sigma by volume band: the price risk of a position.

2. **The edge of each position** (`position_scenarios`). Backing a runner at price m and closing out at BSP
   (laying the same money back at the off) returns ``m / BSP - 1`` per unit whatever the result; laying at m
   returns ``1 - m / BSP``. With ``1/BSP_i = O q_i`` (O the BSP book, about 1.0) and q drawn around q_hat, each
   position's expected CLV is ``m O E[q] - 1`` backing and ``1 - m O E[q] `` laying.

3. **The book** (`build_book`). Stakes on every runner of the race at once, backs and lays, chosen to maximise
   expected log growth of the race's value: at the close (mode "close", price risk only, the trade the owner
   wants) or at the result (mode "hold", Kelly on runners of which exactly one wins). Constraints:

   - the race's expected value is at least ``theta`` of the money at risk (positive price expectation overall);
   - no runner is backed or laid at an expected edge below ``-delta`` (small underlays allowed, poor value not);
   - the race's exposure (stakes plus lay liabilities) and each runner's stake are capped (bank, liquidity).

   A horse level with our price, or a little under it, comes into the book when it pays in the states where
   the rest of the book loses: one runner wins, so the runners' returns are negatively correlated, and a
   near-fair one can lower the book's risk by more than it costs in expectation. The owner's intuition falls
   out of the optimisation instead of being a rule of its own.

`dutch_book` is the owner's rule in its plainest form, for comparison: runners in order of expected edge, each
added while its own edge is at least ``-delta`` and the subset's book keeps at least ``theta`` of expected
value; stakes dutched so the subset returns the same whichever of it wins.

Returns are per unit of the stake vector the caller passes; commission is on the race's net winnings, as the
exchange charges it (a book pays commission once, on its net).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMMISSION = 0.05
_EPS = 1e-12
FEATURES = ("lq_market", "lq_model", "vol", "lq_market_x_vol", "lq_model_x_vol", "lq_market_x_depth",
            "lq_model_x_depth")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def race_codes(race) -> np.ndarray:
    """Integer codes 0..R-1 for a race key, in order of first appearance."""
    return pd.factorize(pd.Series(race).to_numpy())[0]


def normalise_within(x, g: np.ndarray) -> np.ndarray:
    """x divided by its race total."""
    x = np.asarray(x, float)
    return x / np.bincount(g, weights=x)[g]


def group_log_softmax(u: np.ndarray, g: np.ndarray) -> np.ndarray:
    """log softmax of u within each group, stable."""
    mx = np.full(g.max() + 1, -np.inf)
    np.maximum.at(mx, g, u)
    z = u - mx[g]
    return z - np.log(np.bincount(g, weights=np.exp(z)))[g]


# ---------------------------------------------------------------------------
# 1. the closing price
# ---------------------------------------------------------------------------

def closing_inputs(d: pd.DataFrame, race_col: str = "race", market_col: str = "morningwap",
                   model_col: str = "predicted_bfsp", vol_col: str = "morning_vol") -> pd.DataFrame:
    """Race-normalised log probabilities of the market now and of the model, and log traded volumes."""
    g = race_codes(d[race_col])
    out = pd.DataFrame(index=d.index)
    out["lq_market"] = np.log(normalise_within(1.0 / d[market_col].to_numpy(float), g))
    out["lq_model"] = np.log(normalise_within(1.0 / d[model_col].to_numpy(float), g))
    vol = np.log1p(pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float))
    out["log_vol"] = vol
    out["log_depth"] = np.log1p(np.bincount(g, weights=np.expm1(vol))[g])
    return out


@dataclass
class ClosingModel:
    """The conditional-logit forecast of the Betfair SP book, and the spread of its miss by volume band."""

    beta: np.ndarray
    vol_mean: float
    vol_sd: float
    depth_mean: float
    depth_sd: float
    band_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    band_sigma: np.ndarray = field(default_factory=lambda: np.array([0.3]))
    book: float = 1.0                       # median race sum of 1/BSP in training
    n_train: int = 0

    def design(self, inputs: pd.DataFrame) -> np.ndarray:
        v = (inputs["log_vol"].to_numpy(float) - self.vol_mean) / self.vol_sd
        dep = (inputs["log_depth"].to_numpy(float) - self.depth_mean) / self.depth_sd
        lm, lf = inputs["lq_market"].to_numpy(float), inputs["lq_model"].to_numpy(float)
        return np.column_stack([lm, lf, v, lm * v, lf * v, lm * dep, lf * dep])

    def predict(self, inputs: pd.DataFrame, g: np.ndarray) -> np.ndarray:
        """q_hat: the forecast closing probability of every runner, a book of 1 per race."""
        return np.exp(group_log_softmax(self.design(inputs) @ self.beta, g))

    def sigma(self, inputs: pd.DataFrame) -> np.ndarray:
        """The spread of log(q_close / q_hat) for each runner, by its volume band."""
        idx = np.searchsorted(self.band_edges, inputs["log_vol"].to_numpy(float), side="right")
        return self.band_sigma[np.clip(idx, 0, len(self.band_sigma) - 1)]


def _spread(x: np.ndarray) -> float:
    """Standard deviation for standardising, 1 for a column that does not vary (floating dust included)."""
    sd = float(np.std(x))
    return sd if sd > 1e-6 else 1.0


def fit_closing_model(inputs: pd.DataFrame, bsp, g: np.ndarray, l2: float = 1e-4, n_bands: int = 5) -> ClosingModel:
    """Fit the conditional logit to the closing book by cross-entropy (soft labels 1/BSP, normalised)."""
    q = normalise_within(1.0 / np.asarray(bsp, float), g)
    lv, ld = inputs["log_vol"].to_numpy(float), inputs["log_depth"].to_numpy(float)
    m = ClosingModel(beta=np.zeros(len(FEATURES)), vol_mean=float(lv.mean()), vol_sd=_spread(lv),
                     depth_mean=float(ld.mean()), depth_sd=_spread(ld), n_train=len(q))
    X = m.design(inputs)
    n_races = g.max() + 1

    def loss(beta):
        lqh = group_log_softmax(X @ beta, g)
        val = -float(np.sum(q * lqh)) / n_races + 0.5 * l2 * float(beta @ beta)
        grad = -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / n_races + l2 * beta
        return val, grad

    start = np.zeros(len(FEATURES))
    start[0], start[1] = 0.5, 0.5                                   # half market, half model
    res = minimize(loss, start, jac=True, method="L-BFGS-B")
    m.beta = res.x
    resid = np.log(q) - np.log(m.predict(inputs, g))
    m.band_edges = np.quantile(lv, np.linspace(0, 1, n_bands + 1)[1:-1])
    idx = np.searchsorted(m.band_edges, lv, side="right")
    m.band_sigma = np.array([1.4826 * np.median(np.abs(resid[idx == b] - np.median(resid[idx == b])))
                             if (idx == b).sum() > 20 else float(np.std(resid)) for b in range(n_bands)])
    m.book = float(np.median(np.bincount(g, weights=1.0 / np.asarray(bsp, float))))
    return m


# ---------------------------------------------------------------------------
# 2. scenarios and the edge of each position
# ---------------------------------------------------------------------------

def closing_draws(q_hat, sigma, book: float = 1.0, n_draws: int = 400, rng=None) -> np.ndarray:
    """Draws of the closing implied probabilities 1/BSP for one race (n_draws x runners): each runner's log
    probability moves around q_hat by its sigma, then the race is renormalised to the BSP book, so a horse that
    shortens takes its share from the rest."""
    rng = np.random.default_rng(0) if rng is None else rng
    q_hat, sigma = np.asarray(q_hat, float), np.asarray(sigma, float)
    lq = np.log(np.clip(q_hat, _EPS, 1.0)) + sigma * rng.standard_normal((n_draws, len(q_hat))) - 0.5 * sigma ** 2
    lq -= lq.max(axis=1, keepdims=True)
    q = np.exp(lq)
    return book * q / q.sum(axis=1, keepdims=True)


def close_payoffs(inv_bsp_draws: np.ndarray, back, lay=None) -> np.ndarray:
    """Per-unit value at the close of each position in each draw: backs ``m/BSP - 1``, then lays ``1 - m/BSP``."""
    back = np.asarray(back, float)
    lay = back if lay is None else np.asarray(lay, float)
    return np.hstack([back * inv_bsp_draws - 1.0, 1.0 - lay * inv_bsp_draws])


def result_payoffs(back, lay=None) -> np.ndarray:
    """Per-unit payoff of each position for each winner (runners x positions): backs pay ``m - 1`` on their own
    horse and lose 1 otherwise; lays lose ``m - 1`` on their horse and win 1 otherwise."""
    back = np.asarray(back, float)
    lay = back if lay is None else np.asarray(lay, float)
    n = len(back)
    eye = np.eye(n)
    return np.hstack([eye * back[None, :] - 1.0, 1.0 - eye * lay[None, :]])


# ---------------------------------------------------------------------------
# 3. the book
# ---------------------------------------------------------------------------

@dataclass
class Book:
    back: np.ndarray                # stake on each runner
    lay: np.ndarray                 # backer's stake laid on each runner (liability = stake x (price - 1))
    expected: float                 # expected value of the book, same units as the stakes
    exposure: float                 # stakes plus lay liabilities
    edges: np.ndarray               # expected value per unit of each position (backs, then lays)
    growth: float = 0.0             # expected log growth the optimiser reached

    @property
    def n_positions(self) -> int:
        return int((self.back > 1e-9).sum() + (self.lay > 1e-9).sum())


def _net(v: np.ndarray, commission: float) -> np.ndarray:
    return v - commission * np.maximum(v, 0.0)


def build_book(payoffs: np.ndarray, weights, lay_price, delta: float = 0.03, theta: float = 0.03,
               exposure_cap: float = 0.5, stake_cap=None, allow_back=None, allow_lay=None,
               commission: float = COMMISSION) -> Book:
    """Kelly on the race's value under the owner's constraints.

    ``payoffs`` (outcomes x 2n) is the per-unit value of each position (n backs then n lays) in each outcome
    (closing draws, or winners); ``weights`` the outcomes' probabilities. Maximises ``sum_k w_k log(1 + V_k -
    c max(V_k, 0))`` over non-negative stakes with the exposure at most ``exposure_cap``, every position used
    having an expected edge of at least ``-delta``, and the book's expected value at least ``theta`` of its
    exposure. ``allow_back`` / ``allow_lay`` (booleans per runner) switch positions off, e.g. for thin markets.
    """
    A = np.asarray(payoffs, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    lay_price = np.asarray(lay_price, float)
    n = len(lay_price)
    edges = w @ A
    risk = np.concatenate([np.ones(n), lay_price - 1.0])            # money at risk per unit of each position
    hi = np.full(2 * n, np.inf) if stake_cap is None else np.concatenate([np.broadcast_to(stake_cap, n)] * 2)
    hi = np.minimum(hi, exposure_cap / risk)
    ok = edges >= -delta
    if allow_back is not None:
        ok[:n] &= np.asarray(allow_back, bool)
    if allow_lay is not None:
        ok[n:] &= np.asarray(allow_lay, bool)
    hi = np.where(ok, hi, 0.0)
    zero = Book(np.zeros(n), np.zeros(n), 0.0, 0.0, edges)
    # the book's value floor is linear, E[V] >= theta x exposure, so it can only hold if some position clears
    # theta on its own money at risk; those positions make the (feasible) starting book
    good = (edges > theta * risk) & (hi > 0)
    if not np.any(good):
        return zero

    def neg(x):
        v = A @ x
        wv = np.maximum(1.0 + _net(v, commission), _EPS)
        slope = np.where(v > 0, 1.0 - commission, 1.0)
        return -float(w @ np.log(wv)), -(A * (w * slope / wv)[:, None]).sum(axis=0)

    x0 = np.where(good, np.minimum(hi, 0.1 * exposure_cap / risk), 0.0)
    x0 *= min(1.0, 0.5 * exposure_cap / max(float(risk @ x0), _EPS))
    cons = [{"type": "ineq", "fun": lambda x: exposure_cap - risk @ x, "jac": lambda x: -risk},
            {"type": "ineq", "fun": lambda x: (edges - theta * risk) @ x, "jac": lambda x: edges - theta * risk}]
    res = minimize(neg, x0, jac=True, method="SLSQP", bounds=list(zip(np.zeros(2 * n), hi)), constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-12})
    x = np.clip(res.x, 0.0, hi)
    x[x < 1e-9 * max(exposure_cap, 1.0)] = 0.0
    if neg(x)[0] > neg(x0)[0] or (edges - theta * risk) @ x < -1e-9:
        x = x0                                                      # never worse than the start, never infeasible
    if risk @ x <= 0:
        return zero
    return Book(x[:n], x[n:], float(edges @ x), float(risk @ x), edges, -neg(x)[0])


def dutch_book(back, edges_back, q_close, book: float = 1.0, delta: float = 0.03, theta: float = 0.03,
               max_runners: int | None = None, allow=None) -> np.ndarray:
    """The owner's rule as it stands: back the runners in order of expected edge, each while its own edge is
    at least ``-delta`` and the subset's book keeps an expected value of at least ``theta``; stakes dutched
    (``1/m`` each, summing to 1) so the subset returns the same whichever of it wins. The subset's expected
    value is ``book * sum q_close / sum 1/m - 1``: its expected share of the closing book over its book now."""
    back, e, q = (np.asarray(a, float) for a in (back, edges_back, q_close))
    order = np.argsort(-e)
    chosen: list[int] = []
    for i in order:
        if e[i] < -delta:
            break
        if allow is not None and not allow[i]:
            continue
        trial = chosen + [int(i)]
        if book * q[trial].sum() / (1.0 / back[trial]).sum() - 1.0 < theta:
            break
        chosen = trial
        if max_runners and len(chosen) >= max_runners:
            break
    s = np.zeros(len(back))
    if chosen:
        s[chosen] = (1.0 / back[chosen]) / (1.0 / back[chosen]).sum()
    return s


# ---------------------------------------------------------------------------
# settlement
# ---------------------------------------------------------------------------

def settle_close(back_stakes, lay_stakes, back, lay, bsp, commission: float = COMMISSION) -> float:
    """The race's value closed out at the BSP, net of commission on the race's net winnings."""
    inv = 1.0 / np.asarray(bsp, float)
    v = float(np.asarray(back_stakes) @ (np.asarray(back) * inv - 1.0)
              + np.asarray(lay_stakes) @ (1.0 - np.asarray(lay) * inv))
    return v - commission * max(v, 0.0)


def settle_result(back_stakes, lay_stakes, back, lay, winner: int | None, commission: float = COMMISSION) -> float:
    """The race's profit held to the result (``winner`` an index into the runners, None for none of them)."""
    back_stakes, lay_stakes = np.asarray(back_stakes, float), np.asarray(lay_stakes, float)
    p = -back_stakes.sum() + lay_stakes.sum()
    if winner is not None:
        p += back_stakes[winner] * np.asarray(back, float)[winner] - lay_stakes[winner] * np.asarray(lay, float)[winner]
    return p - commission * max(p, 0.0)


# ---------------------------------------------------------------------------
# the owner's staking, walk-forward: every horse backed to win the same amount
# ---------------------------------------------------------------------------

TO_WIN = 250.0
MIN_VOL = 100.0


def whole_races(d: pd.DataFrame, field: pd.Series | None = None, race_col: str = "race") -> pd.DataFrame:
    """The races where every runner has a morning price, a BSP and a forecast (and three runners or more).
    ``field`` is each race's number of runners forecast (default: the rows of ``d``)."""
    ok = (d["morningwap"] > 1) & (d["bsp"] > 1) & (d["predicted_bfsp"] > 1)
    got = d[ok].groupby(race_col).size()
    n = d.groupby(race_col).size() if field is None else field
    keep = got.index[(got.reindex(got.index) == n.reindex(got.index)) & (got >= 3)]
    return d[d[race_col].isin(keep) & ok].copy()


def expected_clv_walk_forward(d: pd.DataFrame, n_draws: int = 300, race_col: str = "race",
                              min_train_races: int = 500) -> pd.DataFrame:
    """Each runner's expected CLV for a back at its morning price, E[morning / BSP] - 1, from a closing model fitted
    on the months before the runner's own. A month is scored only once the months before it hold
    ``min_train_races`` races; until then it is only fitted on (a stray day of prices before the first full month
    must not make a model). ``d``: whole races, with race_date, morningwap, morning_vol, predicted_bfsp and bsp.
    The draws are seeded by month, the races taken in the order of ``d``, so a run repeats exactly. Returns the
    scored months' rows with ``ev`` and ``month``."""
    d = d.copy()
    d["month"] = d["race_date"].astype(str).str[:7]
    out = []
    months = sorted(d["month"].unique())
    for test_month in months[1:]:
        train, test = d[d["month"] < test_month], d[d["month"] == test_month]
        if train[race_col].nunique() < min_train_races:
            continue
        model = fit_closing_model(closing_inputs(train, race_col=race_col), train["bsp"], race_codes(train[race_col]))
        rng = np.random.default_rng(int(test_month.replace("-", "")))
        for _, r in test.groupby(race_col, sort=False):
            inputs = closing_inputs(r, race_col=race_col)
            g = np.zeros(len(r), int)
            draws = closing_draws(model.predict(inputs, g), model.sigma(inputs), book=model.book, n_draws=n_draws,
                                  rng=rng)
            out.append(r.assign(ev=(r["morningwap"].to_numpy(float) * draws).mean(axis=0) - 1.0))
    return pd.concat(out) if out else d.iloc[:0].assign(ev=np.nan)


def to_win_selection(scored: pd.DataFrame, bar: float = 0.03, to_win: float = TO_WIN, min_vol: float = MIN_VOL,
                     race_col: str = "race", commission: float = COMMISSION) -> pd.DataFrame:
    """The owner's staking on the scored rows: every horse whose expected CLV is at least ``bar`` (and at least
    ``min_vol`` matched in the morning) backed to win ``to_win`` at its morning price, stake to_win / (price - 1).
    One row per race with a bet: its bets, stakes, expected CLV, CLV closed out at the BSP and profit held to the
    result, each net of commission on the race's net winnings."""
    s = scored[(scored["ev"] >= bar) & (scored["morning_vol"].fillna(0) >= min_vol)].copy()
    if s.empty:
        return pd.DataFrame(columns=[race_col, "race_date", "bets", "staked", "expected", "clv", "result"])
    m, bsp = s["morningwap"].to_numpy(float), s["bsp"].to_numpy(float)
    stake = to_win / (m - 1.0)
    won = s["won"].astype(bool).to_numpy()
    s["stake"], s["exp_v"] = stake, stake * s["ev"].to_numpy(float)
    s["clv_v"] = stake * (m / bsp - 1.0)
    s["res_v"] = np.where(won, stake * (m - 1.0), -stake)
    g = s.groupby(race_col, sort=False).agg(race_date=("race_date", "first"), bets=("stake", "size"),
                                            staked=("stake", "sum"), expected=("exp_v", "sum"),
                                            clv=("clv_v", "sum"), result=("res_v", "sum")).reset_index()
    g["clv"] -= commission * g["clv"].clip(lower=0)
    g["result"] -= commission * g["result"].clip(lower=0)
    return g


def ratio_interval(num: np.ndarray, den: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """A 90% race-bootstrap interval for sum(num) / sum(den)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2 or den.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


# ---------------------------------------------------------------------------
# 5. live: the closing model as a file, and one race's expected CLV at the prices on offer
# ---------------------------------------------------------------------------

TIGHT = 1.25          # lay / back at most this: the mid is the market's price now, else the back price


def save_closing_model(m: ClosingModel, path, **about) -> None:
    """The fitted closing model as JSON (``about``: where it was fitted, and anything else worth keeping)."""
    import json
    from pathlib import Path
    out = {**about, "features": FEATURES, "beta": [float(b) for b in m.beta], "vol_mean": m.vol_mean,
           "vol_sd": m.vol_sd, "depth_mean": m.depth_mean, "depth_sd": m.depth_sd,
           "band_edges": [float(e) for e in m.band_edges], "band_sigma": [float(s) for s in m.band_sigma],
           "book": m.book, "n_train": m.n_train}
    Path(path).write_text(json.dumps(out, indent=1) + "\n")


def load_closing_model(path) -> ClosingModel:
    import json
    from pathlib import Path
    j = json.loads(Path(path).read_text())
    return ClosingModel(beta=np.asarray(j["beta"], float), vol_mean=j["vol_mean"], vol_sd=j["vol_sd"],
                        depth_mean=j["depth_mean"], depth_sd=j["depth_sd"],
                        band_edges=np.asarray(j["band_edges"], float), band_sigma=np.asarray(j["band_sigma"], float),
                        book=j["book"], n_train=j["n_train"])


def market_now(back, lay) -> np.ndarray:
    """The market's price now, as the closing model reads it: the geometric mid of the best back and lay when the
    spread is tight (lay / back at most TIGHT), otherwise the back price."""
    back = np.asarray(back, float)
    lay = np.asarray(lay, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tight = np.isfinite(lay) & (lay > 1) & (lay / back <= TIGHT)
        return np.where(tight, np.sqrt(back * np.where(tight, lay, back)), back)


def race_expected_clv(back, lay, model_price, matched, model: ClosingModel, n_draws: int = 50000,
                      rng=None, market=None) -> tuple[np.ndarray, np.ndarray]:
    """One race's expected CLV at every runner's best back price: E[back / BSP] - 1 over the closing model's draws
    of the BSP book, from the market's price now (market_now, or ``market`` when given), our price and each
    runner's matched money (zeros when the feed carries none: use a model fitted without volume). Returns
    (expected CLV, 1 / E[1 / BSP]); a runner with no back price has no expected CLV (NaN)."""
    back = np.asarray(back, float)
    now = market_now(back, lay) if market is None else np.asarray(market, float)
    d = pd.DataFrame({"race": 0, "morningwap": now, "predicted_bfsp": np.asarray(model_price, float),
                      "morning_vol": np.asarray(matched, float)})
    inputs = closing_inputs(d)
    g = np.zeros(len(d), int)
    draws = closing_draws(model.predict(inputs, g), model.sigma(inputs), book=model.book, n_draws=n_draws,
                          rng=np.random.default_rng(0) if rng is None else rng)
    inv = draws.mean(axis=0)
    return back * inv - 1.0, 1.0 / inv
