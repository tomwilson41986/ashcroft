"""Speed-figure definitions scored by how well each predicts the next race (local audit, GBGB Jul 2025 - Nov 2026).

Standards, offsets, pars and constants are fitted on TRAIN (to 2026-02-28) only; the predictor's one coefficient on
FIT races (2025-10-01 .. 2026-02-28); the score on EVAL races (2026-03-01 on). A dog's predictor is the mean of its
last 3 figures from earlier days within 180 days."""
import sys, json
import numpy as np, pandas as pd
from research.greyhound.times_audit.load import load, L

TRAIN_END, FIT_FROM, EVAL_FROM = pd.Timestamp("2026-02-28"), pd.Timestamp("2025-10-01"), pd.Timestamp("2026-03-01")


def base():
    d = load()
    g = d[~d.trial & ~d.hcap & ~d.hurdle & d.adjusted_time.notna() & d.going_ok & d.position.notna()
          & d.dog_id.notna()].copy()
    tr = g[(g.day <= TRAIN_END) & (g.position == 1)]
    cnt = tr.groupby("key").size()
    g = g[g.key.isin(cnt[cnt >= 100].index)].copy()
    g = g.sort_values(["race_id", "position"])
    g["cum"] = (g.groupby("race_id").beaten_lengths.cumsum() - g.beaten_lengths.where(g.position == 1, 0)).fillna(0)
    g.loc[g.position == 1, "cum"] = 0.0
    w = g[g.position == 1].groupby("race_id").adjusted_time.min()
    g["wcalc"] = g.race_id.map(w)
    g["wraw"] = g.race_id.map(g[g.position == 1].groupby("race_id").run_time.min())
    g = g[g.wcalc.notna()]
    return g


def figure(g, how):
    """A figure in percent of the time, higher faster, for every run, by the definition ``how`` (dict)."""
    tr = (g.day <= TRAIN_END)
    k = how.get("spl", 0.08)
    cap = how.get("cap", None)
    lengths = g.cum.clip(upper=cap) if cap else g.cum
    wt = g.wraw if how.get("raw") else g.wcalc
    t = wt + k * lengths
    if how.get("going_alpha") is not None:                          # rescale GBGB's allowance
        t = g.wraw + how["going_alpha"] * g.going / 100 + k * lengths
    par = g[tr & (g.position == 1)].assign(t=wt).groupby("key").t.median()
    P = 100 * np.log(g.key.map(par) / t)
    if how.get("unit") == "lengths":                                # the live convention: seconds over 0.08
        P = (g.key.map(par) - t) / L
    if how.get("link"):
        P = P + g.key.map(link_offsets(g.assign(P=P), tr)).fillna(0)
    if how.get("variant"):
        P = P - meeting_variant(g.assign(P=P), tr, how["variant"])
    return P


def link_offsets(g, tr, lam=5.0, days=28):
    """Key offsets (percent) making consecutive runs of the same dog at different keys agree on average (train only)."""
    x = g[tr].sort_values(["dog_id", "day"])[["dog_id", "day", "key", "P"]]
    nx = x.groupby("dog_id").shift(-1)
    m = nx.day.notna() & ((nx.day - x.day).dt.days <= days) & (nx.key != x.key)
    a, b, y = x.key[m].to_numpy(), nx.key[m].to_numpy(), (nx.P[m] - x.P[m]).to_numpy()
    y = y - np.median(g[tr].sort_values(["dog_id","day"]).groupby("dog_id").P.diff().dropna())   # typical change
    keys = sorted(set(a) | set(b)); ix = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(y), len(keys)))
    A[np.arange(len(y)), [ix[k] for k in b]] += 1.0
    A[np.arange(len(y)), [ix[k] for k in a]] -= 1.0
    # offsets o so that (P_b + o_b) - (P_a + o_a) ~ 0  =>  A o ~ -y ; ridge to 0
    o = np.linalg.solve(A.T @ A + lam * np.eye(len(keys)), A.T @ (-y))
    return pd.Series(o - o.mean(), index=keys)


def meeting_variant(g, tr, kind):
    """The night's residual pace beyond GBGB's going: winners' figure against their grade's par at the key (train),
    averaged over the meeting and shrunk by n / (n + k), k the within over between variance (train)."""
    w = g[g.position == 1]
    gp = w[w.day <= TRAIN_END].groupby(["key", "cls"]).P.median()
    r = w.P - pd.MultiIndex.from_frame(w[["key", "cls"]]).map(lambda t: gp.get(t, np.nan)).to_numpy()
    r = r.fillna(r.median())
    grp = r.groupby(w.meeting_id)
    mean, n = grp.mean(), grp.size()
    trm = w[w.day <= TRAIN_END].meeting_id.unique()
    within = grp.var()[n > 1].reindex(trm).mean()
    between = max(mean.reindex(trm)[n.reindex(trm) >= 3].var() - within * (1 / n.reindex(trm)[n.reindex(trm) >= 3]).mean(), 1e-6)
    kk = within / between
    v = mean * n / (n + kk)
    return g.meeting_id.map(v).fillna(0.0)


def predictor(g, F, last=3, days=180):
    """The mean of the dog's last ``last`` figures from earlier days within ``days`` (NaN when none)."""
    x = g[["dog_id", "day"]].assign(F=F.to_numpy()).sort_values(["dog_id", "day"])
    out = pd.Series(np.nan, index=x.index)
    for _, s in x.groupby("dog_id", sort=False):
        days_ = s.day.to_numpy(); f = s.F.to_numpy(); res = np.full(len(s), np.nan)
        for i in range(len(s)):
            j = np.searchsorted(days_, days_[i])          # earlier days only
            lo = np.searchsorted(days_, days_[i] - np.timedelta64(days, "D"))
            prev = f[max(lo, j - last):j]
            prev = prev[np.isfinite(prev)]
            if len(prev):
                res[i] = prev.mean()
        out.loc[s.index] = res
    return out.reindex(g.index)


def score(g, X):
    """Conditional logit on one predictor (the race-centred X), beta fitted on FIT races, log-loss on EVAL races."""
    df = g[["race_id", "day", "position"]].assign(X=X)
    ok = df.groupby("race_id").X.transform(lambda s: s.notna().all()) & (df.groupby("race_id").X.transform("size") >= 4)
    df = df[ok].copy()
    df["xc"] = df.X - df.groupby("race_id").X.transform("mean")
    df["won"] = (df.position == 1).astype(float)
    df = df[df.groupby("race_id").won.transform("sum") == 1]
    fit = df[(df.day >= FIT_FROM) & (df.day <= TRAIN_END)]
    ev = df[df.day >= EVAL_FROM]

    def prep(x):
        x = x.sort_values("race_id")
        codes = x.race_id.to_numpy(); starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
        return x.xc.to_numpy(), x.won.to_numpy(), starts

    def ll(b, p):
        xc, won, starts = p
        z = b * xc
        m = np.maximum.reduceat(z, starts)
        lse = np.log(np.add.reduceat(np.exp(z - np.repeat(m, np.diff(np.r_[starts, len(z)]))), starts)) + m
        return -(np.add.reduceat(z * won, starts) - lse).mean()
    pf, pe = prep(fit), prep(ev)
    bs = np.linspace(0.0, 3.0, 301)
    b = bs[int(np.argmin([ll(v, pf) for v in bs]))]
    e = ev.assign(r=ev.groupby("race_id").X.rank(ascending=False))
    rho = e[["r", "position"]].corr().iloc[0, 1]
    pe_ll = ll(b, pe)
    return {"races_eval": int(ev.race_id.nunique()), "beta": round(float(b), 3), "logloss_eval": round(float(pe_ll), 5),
            "uniform": round(float(np.log(ev.groupby("race_id").size()).mean()), 5), "rank_corr": round(float(rho), 4)}


def class_figure(g, F, alpha=0.5):
    """A rating anchored on the field: the race's strength (the mean of its runners' prior predictors, earlier days
    only) plus the dog's figure against the race's mean figure, blended with the time figure by ``alpha``."""
    prior = predictor(g, F)
    S = prior.groupby(g.race_id).transform("mean")
    rel = F - F.groupby(g.race_id).transform("mean")
    CF = (S + rel).fillna(F)
    return alpha * F + (1 - alpha) * CF


def grade_par(g, F):
    """The figure against its grade's par at the key (train winners' median figure there): over or under par."""
    tr = g.day <= TRAIN_END
    par = F[tr & (g.position == 1)].groupby([g.key, g.cls]).median()
    p = pd.MultiIndex.from_frame(g[["key", "cls"]]).map(lambda t: par.get(t, np.nan))
    return F - np.asarray(p, float)


def race_ll(g, X):
    """Per EVAL race: its log-loss under the one-coefficient conditional logit fitted on FIT races (as ``score``)."""
    df = g[["race_id", "day", "position"]].assign(X=X)
    ok = df.groupby("race_id").X.transform(lambda s: s.notna().all()) & (df.groupby("race_id").X.transform("size") >= 4)
    df = df[ok].copy()
    df["xc"] = df.X - df.groupby("race_id").X.transform("mean")
    df["won"] = (df.position == 1).astype(float)
    df = df[df.groupby("race_id").won.transform("sum") == 1].sort_values("race_id")

    def per_race(b, x):
        z = b * x.xc.to_numpy(); codes = x.race_id.to_numpy()
        st = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]; reps = np.diff(np.r_[st, len(z)])
        m = np.maximum.reduceat(z, st)
        lse = np.log(np.add.reduceat(np.exp(z - np.repeat(m, reps)), st)) + m
        return -(np.add.reduceat(z * x.won.to_numpy(), st) - lse), codes[st]
    fit = df[(df.day >= FIT_FROM) & (df.day <= TRAIN_END)]; ev = df[df.day >= EVAL_FROM]
    bs = np.linspace(0, 3, 301); b = bs[int(np.argmin([per_race(v, fit)[0].mean() for v in bs]))]
    ll, rid = per_race(b, ev)
    day = ev.groupby("race_id").day.first().reindex(rid).to_numpy()
    return pd.DataFrame({"race_id": rid, "day": day, "ll": ll}), b


def paired(a, b, draws=2000, seed=0):
    """b minus a on the races both score, the mean and its standard error by day (bootstrap)."""
    m = a.merge(b, on=["race_id", "day"], suffixes=("_a", "_b"))
    dd = (m.ll_b - m.ll_a).groupby(m.day).agg(["sum", "size"])
    rng = np.random.default_rng(seed); n = len(dd)
    boots = [dd.iloc[rng.integers(0, n, n)].pipe(lambda x: x["sum"].sum() / x["size"].sum()) for _ in range(draws)]
    diff = float(dd["sum"].sum() / dd["size"].sum()); se = float(np.std(boots))
    return {"races": int(len(m)), "diff": round(diff, 5), "se": round(se, 5), "z": round(diff / se, 1) if se > 0 else None}
