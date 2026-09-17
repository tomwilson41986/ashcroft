"""
Bet-level analysis of walk-forward predictions: overlays and per-race ranks.

Answers two questions on real out-of-sample output, model and market side
by side:

    overlay_table / cumulative_overlays / overlay_by_price
        How do bets perform by edge tier (model probability vs market
        probability)? Actual win rate vs what the model and the market each
        implied, ROI at BSP (net of commission) for backing, ROI for laying
        the underlays, cluster-bootstrap CIs by race.
    rank_table / rank_disagreement / rank_by_field
        How does the model's n-th choice in a race perform (win rate, place
        rate, ROI) versus the market's n-th choice, and what happens when the
        model's top pick is not the favourite?
    blended probabilities (log-linear, Benter) via ``blend_lambda``.

Conventions: back returns are per unit stake with commission on net
winnings; lay returns are per unit *stake* (liability BSP-1) with
commission on the winnings when the horse loses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.abm.features import place_terms
from model.diagnostics import market_implied_probs, race_normalise
from model.perf_figures import ensure_raceid

EDGE_TIERS = [-100, -50, -25, -10, 0, 10, 25, 50, 100, np.inf]
PRICE_BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, np.inf]
FIELD_BANDS = [0, 5, 8, 11, 15, 100]


def _won(series: pd.Series) -> np.ndarray:
    s = series
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    return pd.to_numeric(s, errors="coerce").fillna(0).values > 0


def prepare_bets(df: pd.DataFrame, p_col: str = "predicted_win_prob_norm", price_col: str = "bfsp",
                 race_col: str = "raceid", y_col: str = "won", position_col: str = "placing_numerical",
                 commission: float = 0.05, blend_lambda: float | None = None) -> pd.DataFrame:
    """One row per runner with model/market probabilities, edge, ranks and returns."""
    d = ensure_raceid(df).copy()
    d["bsp"] = pd.to_numeric(d[price_col], errors="coerce")
    d = d[(d["bsp"] > 1.0) & d[p_col].notna()].copy()
    d["p_model"] = race_normalise(d, p_col, race_col)
    d["p_market"] = market_implied_probs(d, "bsp", race_col)
    d["won"] = _won(d[y_col])
    d["edge_pct"] = (d["p_model"] / d["p_market"] - 1.0) * 100.0
    d["field"] = d.groupby(race_col)[race_col].transform("size")
    d["model_rank"] = d.groupby(race_col)["p_model"].rank(ascending=False, method="first")
    d["market_rank"] = d.groupby(race_col)["bsp"].rank(ascending=True, method="first")
    pos = pd.to_numeric(d[position_col], errors="coerce") if position_col in d.columns else pd.Series(np.nan, index=d.index)
    terms = d["field"].map(lambda n: place_terms(int(n)))
    d["placed"] = (pos <= terms).fillna(False).values
    d["ret_back"] = np.where(d["won"], (d["bsp"] - 1.0) * (1.0 - commission), -1.0)
    d["ret_lay"] = np.where(d["won"], -(d["bsp"] - 1.0), 1.0 * (1.0 - commission))
    if blend_lambda is not None:
        lam = float(blend_lambda)
        logit = lam * np.log(d["p_model"].clip(1e-9)) + (1.0 - lam) * np.log(d["p_market"].clip(1e-9))
        d["_e"] = np.exp(logit)
        d["p_blend"] = d["_e"] / d.groupby(race_col)["_e"].transform("sum")
        d["edge_blend_pct"] = (d["p_blend"] / d["p_market"] - 1.0) * 100.0
        d = d.drop(columns=["_e"])
    return d


# ---------------------------------------------------------------------------
# Cluster bootstrap (fast: resample races via multinomial weights)
# ---------------------------------------------------------------------------

def cluster_bootstrap_roi(bets: pd.DataFrame, ret_col: str = "ret_back", race_col: str = "raceid",
                          n_boot: int = 500, level: float = 0.9, random_state: int = 0) -> tuple[float, float]:
    """CI for ROI (mean return per unit stake) resampling whole races."""
    g = bets.groupby(race_col)[ret_col].agg(["sum", "size"])
    if len(g) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    R = len(g)
    w = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(float)
    roi = (w @ g["sum"].values) / (w @ g["size"].values)
    a = (1 - level) / 2
    return float(np.quantile(roi, a)), float(np.quantile(roi, 1 - a))


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

def overlay_table(d: pd.DataFrame, edge_col: str = "edge_pct", tiers=EDGE_TIERS) -> pd.DataFrame:
    """Performance by edge tier. Compare win_rate with model_p (what the
    model promised) and market_p (what the market implied)."""
    t = d.copy()
    t["tier"] = pd.cut(t[edge_col], tiers, right=False)
    out = (t.groupby("tier", observed=True)
           .agg(n=("won", "size"), win_rate=("won", "mean"), model_p=("p_model", "mean"), market_p=("p_market", "mean"),
                avg_bsp=("bsp", "mean"), roi_back=("ret_back", "mean"), roi_lay=("ret_lay", "mean"))
           .reset_index())
    out["tier"] = out["tier"].astype(str)
    out["share_pct"] = 100 * out["n"] / len(t)
    return out


def cumulative_overlays(d: pd.DataFrame, edge_col: str = "edge_pct", thresholds=(0, 5, 10, 15, 20, 30, 50, 100),
                        race_col: str = "raceid", n_boot: int = 300) -> pd.DataFrame:
    """Bet everything with edge >= threshold (the current strategy shape)."""
    rows = []
    for thr in thresholds:
        b = d[d[edge_col] >= thr]
        if len(b) == 0:
            continue
        lo, hi = cluster_bootstrap_roi(b, "ret_back", race_col, n_boot=n_boot)
        rows.append({"min_edge_pct": thr, "n_bets": len(b), "bets_per_race": len(b) / d[race_col].nunique(),
                     "win_rate": b["won"].mean(), "model_p": b["p_model"].mean(), "market_p": b["p_market"].mean(),
                     "avg_bsp": b["bsp"].mean(), "roi_back": b["ret_back"].mean(), "roi_ci_lo": lo, "roi_ci_hi": hi,
                     "roi_lay_underlays": d.loc[d[edge_col] <= -thr, "ret_lay"].mean() if thr > 0 else np.nan})
    return pd.DataFrame(rows)


def overlay_by_price(d: pd.DataFrame, min_edge: float = 10.0, edge_col: str = "edge_pct", bands=PRICE_BANDS) -> pd.DataFrame:
    """Overlay bets by BSP band, with the all-runners ROI in the same band as
    the favourite-longshot-bias reference."""
    t = d.copy()
    t["band"] = pd.cut(t["bsp"], bands, right=False)
    base = t.groupby("band", observed=True).agg(all_n=("won", "size"), all_roi=("ret_back", "mean"), all_win_rate=("won", "mean"))
    b = t[t[edge_col] >= min_edge].groupby("band", observed=True).agg(
        n=("won", "size"), win_rate=("won", "mean"), model_p=("p_model", "mean"), market_p=("p_market", "mean"),
        roi_back=("ret_back", "mean"))
    out = b.join(base).reset_index()
    out["band"] = out["band"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------

def rank_table(d: pd.DataFrame, rank_col: str = "model_rank", max_rank: int = 8) -> pd.DataFrame:
    """Per-race rank performance (ranks beyond max_rank pooled)."""
    t = d.copy()
    t["rank"] = np.where(t[rank_col] >= max_rank, max_rank, t[rank_col]).astype(int)
    out = (t.groupby("rank")
           .agg(n=("won", "size"), win_rate=("won", "mean"), place_rate=("placed", "mean"), model_p=("p_model", "mean"),
                market_p=("p_market", "mean"), avg_bsp=("bsp", "mean"), roi_back=("ret_back", "mean"), roi_lay=("ret_lay", "mean"))
           .reset_index())
    out["rank"] = out["rank"].astype(str).where(out["rank"] < max_rank, f"{max_rank}+")
    return out


def rank_disagreement(d: pd.DataFrame, max_rank: int = 4) -> pd.DataFrame:
    """The model's top pick, split by where the market ranks it (and vice versa).

    The rows where the two disagree are the only ones where the model can
    add information; agreement rows just restate the market."""
    rows = []
    top = d[d["model_rank"] == 1].copy()
    top["mr"] = np.where(top["market_rank"] >= max_rank, max_rank, top["market_rank"]).astype(int)
    for mr, g in top.groupby("mr"):
        rows.append({"pick": "model #1", "other_rank": f"market #{mr}" + ("+" if mr == max_rank else ""), "n": len(g),
                     "share_pct": 100 * len(g) / len(top), "win_rate": g["won"].mean(), "model_p": g["p_model"].mean(),
                     "market_p": g["p_market"].mean(), "avg_bsp": g["bsp"].mean(), "roi_back": g["ret_back"].mean()})
    fav = d[d["market_rank"] == 1].copy()
    fav["mr"] = np.where(fav["model_rank"] >= max_rank, max_rank, fav["model_rank"]).astype(int)
    for mr, g in fav.groupby("mr"):
        rows.append({"pick": "market #1", "other_rank": f"model #{mr}" + ("+" if mr == max_rank else ""), "n": len(g),
                     "share_pct": 100 * len(g) / len(fav), "win_rate": g["won"].mean(), "model_p": g["p_model"].mean(),
                     "market_p": g["p_market"].mean(), "avg_bsp": g["bsp"].mean(), "roi_back": g["ret_back"].mean()})
    return pd.DataFrame(rows)


def rank_by_field(d: pd.DataFrame, bands=FIELD_BANDS) -> pd.DataFrame:
    """Model #1 vs market #1 by field size."""
    t = d.copy()
    t["field_band"] = pd.cut(t["field"], bands, right=True)
    rows = []
    for fb, g in t.groupby("field_band", observed=True):
        m1 = g[g["model_rank"] == 1]; k1 = g[g["market_rank"] == 1]
        rows.append({"field": str(fb), "races": g["raceid"].nunique(), "model1_win": m1["won"].mean(),
                     "model1_roi": m1["ret_back"].mean(), "market1_win": k1["won"].mean(), "market1_roi": k1["ret_back"].mean(),
                     "agree_pct": 100 * (m1["market_rank"] == 1).mean()})
    return pd.DataFrame(rows)


def concordance_by_field(d: pd.DataFrame, bands=FIELD_BANDS) -> pd.DataFrame:
    from model.diagnostics import concordance_index
    t = d.copy()
    t["field_band"] = pd.cut(t["field"], bands, right=True)
    rows = []
    for fb, g in t.groupby("field_band", observed=True):
        pos = pd.to_numeric(g.get("placing_numerical"), errors="coerce")
        rows.append({"field": str(fb), "races": g["raceid"].nunique(),
                     "model_concordance": concordance_index(g["p_model"].values, pos.values, g["raceid"].values),
                     "market_concordance": concordance_index(g["p_market"].values, pos.values, g["raceid"].values)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segments
#
# Field size and price band were the only two cuts this module made, and they
# are the two the model is least likely to be wrong about in an interesting
# way. A model can be honest overall and badly wrong on a corner of the sport
# -- all-weather sprints, a class of race, a month when the ground changed --
# and a pooled number cannot show it. These are the cuts the framework asks
# for: race code, race type, class, month.
#
# Every table pools segments below ``min_n`` into "(other)" rather than
# dropping them, so the rows always add back to the whole. A table that
# quietly loses a tenth of the sample is how a segment finding becomes a
# selection effect.
# ---------------------------------------------------------------------------

GOING_GROUPS = {
    "Heavy": "Soft/Heavy", "Soft To Heavy": "Soft/Heavy", "Soft": "Soft/Heavy",
    "Yielding To Soft": "Soft/Heavy", "Yielding": "Good/Yielding",
    "Good To Yielding": "Good/Yielding", "Good To Soft": "Good/Yielding",
    "Good": "Good", "Good To Firm": "Good/Fast", "Firm": "Good/Fast",
    "Standard": "Standard (AW)", "Standard To Slow": "Standard (AW)",
    "Standard To Fast": "Standard (AW)", "Slow": "Standard (AW)",
}


def broad_race_type(rt) -> str:
    """Collapse the free-text race type into the categories people bet in."""
    if pd.isna(rt):
        return "Unknown"
    rt = str(rt).lower()
    if "handicap" in rt and "chase" in rt:
        return "Handicap Chase"
    if "handicap" in rt and "hurdle" in rt:
        return "Handicap Hurdle"
    if "chase" in rt:
        return "Non-Hcp Chase"
    if "hurdle" in rt:
        return "Non-Hcp Hurdle"
    if "nh flat" in rt or "bumper" in rt:
        return "NH Flat"
    if "handicap" in rt or "nursery" in rt:
        return "Handicap Flat"
    if "maiden" in rt:
        return "Maiden"
    if "novice" in rt:
        return "Novices"
    return "Other Flat"


def segment_labels(d: pd.DataFrame) -> pd.DataFrame:
    """Add the label columns the segment tables group by.

    Each one is skipped rather than invented when its source column is
    absent, so an older predictions file still reports the cuts it can."""
    t = d.copy()
    if "race_type" in t.columns:
        t["race_group"] = t["race_type"].map(broad_race_type)
    if "going_description" in t.columns:
        t["going_group"] = t["going_description"].map(GOING_GROUPS).fillna("Other")
    if "race_date" in t.columns:
        t["month"] = pd.to_datetime(t["race_date"], errors="coerce").dt.strftime("%Y-%m")
    if "race_class" in t.columns:
        cls = t["race_class"].astype(str).str.strip()
        t["class_band"] = cls.where(cls.str.len() > 0, "Unknown").fillna("Unknown")
    if "race_code" in t.columns:
        t["code"] = t["race_code"].astype(str).str.strip().replace("", "Unknown").fillna("Unknown")
    return t


def _pooled(t: pd.DataFrame, col: str, min_n: int, count_on: pd.Series | None = None) -> pd.Series:
    """Segment labels with small levels folded into "(other)"."""
    lab = t[col].astype(str).fillna("Unknown")
    counts = (lab[count_on] if count_on is not None else lab).value_counts()
    keep = set(counts[counts >= min_n].index)
    return lab.where(lab.isin(keep), "(other)")


def rank_by_segment(d: pd.DataFrame, col: str, min_n: int = 200) -> pd.DataFrame:
    """Model #1 against market #1 within each level of ``col``.

    ``min_n`` counts the model's top picks, i.e. the bets, not the runners:
    a segment with 200 runners and 20 races cannot say anything about ROI."""
    if col not in d.columns:
        return pd.DataFrame()
    t = d.copy()
    t["_seg"] = _pooled(t, col, min_n, count_on=(t["model_rank"] == 1))
    rows = []
    for seg, g in t.groupby("_seg", observed=True):
        m1 = g[g["model_rank"] == 1]
        k1 = g[g["market_rank"] == 1]
        if m1.empty:
            continue
        lo, hi = cluster_bootstrap_roi(m1)
        rows.append({"segment": seg, "races": int(g["raceid"].nunique()), "runners": len(g),
                     "bets": len(m1), "win_rate": m1["won"].mean(), "avg_bsp": m1["bsp"].mean(),
                     "roi_back": m1["ret_back"].mean(), "ci_lo": lo, "ci_hi": hi,
                     "market1_roi": k1["ret_back"].mean() if len(k1) else np.nan,
                     "agree_pct": 100 * (m1["market_rank"] == 1).mean()})
    out = pd.DataFrame(rows)
    return out.sort_values("bets", ascending=False).reset_index(drop=True) if len(out) else out


def concordance_by_segment(d: pd.DataFrame, col: str, min_n: int = 200) -> pd.DataFrame:
    """Model and market c-index within each level of ``col``.

    Ordering is the thing the model is measurably behind on, so the segment
    question worth asking is where the gap is widest, not only where the ROI
    happens to look best on this sample."""
    from model.diagnostics import concordance_index
    if col not in d.columns:
        return pd.DataFrame()
    t = d.copy()
    t["_seg"] = _pooled(t, col, min_n)
    rows = []
    for seg, g in t.groupby("_seg", observed=True):
        pos = pd.to_numeric(g.get("placing_numerical"), errors="coerce")
        mc = concordance_index(g["p_model"].values, pos.values, g["raceid"].values)
        kc = concordance_index(g["p_market"].values, pos.values, g["raceid"].values)
        rows.append({"segment": seg, "races": int(g["raceid"].nunique()), "runners": len(g),
                     "model_concordance": mc, "market_concordance": kc, "gap": mc - kc})
    out = pd.DataFrame(rows)
    return out.sort_values("runners", ascending=False).reset_index(drop=True) if len(out) else out


def bet_report(df: pd.DataFrame, commission: float = 0.05, blend_lambda: float | None = 0.5, **kw) -> dict:
    d = prepare_bets(df, commission=commission, blend_lambda=blend_lambda, **kw)
    out = {
        "n_runners": len(d), "n_races": int(d["raceid"].nunique()),
        "overlay_tiers": overlay_table(d),
        "cumulative_overlays": cumulative_overlays(d),
        "overlay_by_price_10": overlay_by_price(d, 10.0),
        "model_ranks": rank_table(d, "model_rank"),
        "market_ranks": rank_table(d, "market_rank"),
        "disagreement": rank_disagreement(d),
        "rank_by_field": rank_by_field(d),
        "concordance_by_field": concordance_by_field(d),
    }
    seg = segment_labels(d)
    for name, col in (("by_race_code", "code"), ("by_race_type", "race_group"),
                      ("by_race_class", "class_band"), ("by_month", "month"),
                      ("by_going", "going_group")):
        tbl = rank_by_segment(seg, col)
        if len(tbl):
            out[name] = tbl
    ctbl = concordance_by_segment(seg, "race_group")
    if len(ctbl):
        out["concordance_by_race_type"] = ctbl
    if blend_lambda is not None:
        out["blend_lambda"] = blend_lambda
        out["blend_cumulative_overlays"] = cumulative_overlays(d, edge_col="edge_blend_pct")
    return out
