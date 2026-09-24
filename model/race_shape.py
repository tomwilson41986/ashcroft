"""Run style, race shape and what a position is worth in that shape.

The database has no sectionals and no in-running positions, but every run
carries a race comment, and a comment narrates the race in order: its first
positional phrase says where the horse raced early. This module turns those
comments into four things, all lag-safe:

1. **A run style per past run** (`early_style_from_comment`): the style score of
   the comment's first positional phrase, on the 1-6 scale `model.pace_metrics`
   uses (6 made all ... 1 held up in rear), its class (lead / prominent / mid /
   rear), and the field-size-normalised early position `epf_norm` (0 = led,
   1 = last). A comment with no positional phrase is unknown, never midfield.

2. **A projected run style for today** (`add_style_projection`): per horse, a
   recency-weighted distribution over the four classes and a projected
   `epf_norm`, from its earlier runs only, shrunk toward the population for a
   horse with few runs and filled from it for a debutant.

3. **The race shape** (`add_race_shape`): from the projected styles of the
   field -- how many runners want the lead, the chance the lead is contested
   or that nobody takes it, how much of the field races on the pace, and how
   clear the most likely leader is.

4. **The value of a position within the shape** (`add_position_value`): learned
   from earlier days only, at course x distance x field size, pooled toward
   coarser levels. Two readings:
     - by EXPECTED position: how runners projected to lead (or race prominently,
       mid-division, in rear) fared in races of this shape here. This is
       directly what a forecast made before the off can use.
     - by ACTUAL position, weighted by today's chance of getting it: what
       leading (etc.) was worth in this shape here, times the probability this
       runner leads. It separates the chance of the position from its value.
   Both are measured on two outcomes, the normalised finishing position and
   lengths beaten converted to pounds at the trip (`add_run_outcomes`).

Every statistic of a race uses only races run on EARLIER DAYS (as
`model.lagsafe` does by default): a 06:00 forecast knows nothing of the card
it is pricing, and a backtest must not either. The per-run columns describing
the race itself (`RACE_SHAPE_POST_RACE`) exist only to feed those statistics
and are never features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.pace_metrics import epf_norm_from_style

# ---------------------------------------------------------------------------
# 1. The comment's first positional phrase
# ---------------------------------------------------------------------------

#: Style scores, on `model.pace_metrics`' scale: 6 made all ... 1 held up in rear.
#: Every pattern is anchored on word boundaries: unanchored, "led," matched
#: inside "pulled," and "travelled," and "led to" inside "failed to".
_STYLE_VOCAB: list[tuple[float, str]] = [
    (6.0, r"made (?:virtually |almost |nearly )?all|made most|made (?:the )?running"
          r"|set (?:a |the )?(?:\w+ )?pace|soon led|led|went clear early|bolted clear early|overall leader"),
    (5.5, r"disputed (?:the )?lead|disputed|with (?:the )?leaders?|joint[- ]lead|upsides (?:the )?leaders?"
          r"|dueled|duelled|vied for (?:the )?lead|shared (?:the )?lead|in (?:the )?leading (?:pair|duo)"
          r"|alongside (?:the )?leader|good early speed"),
    (5.0, r"(?:chased|tracked|pressed|chasing|tracking|pressing|chase|track) (?:the )?(?:clear |runaway |long[- ]time )?"
          r"(?:leader|winner)|(?:raced |went )?(?:in )?(?:2nd|second)"),
    (4.5, r"(?:chased|tracked|chasing|tracking|pressed|pressing|chase|track) (?:the )?"
          r"(?:leaders|leading (?:pair|trio|group|quartet|bunch|two|three)|front (?:pair|two|three|rank|group))"
          r"|(?:just )?behind (?:the )?leaders?|in (?:the )?leading (?:trio|group|three|quartet|bunch)"),
    (4.0, r"prominent(?:ly)?|close[- ]up|handy|front rank|close to (?:the )?pace|near (?:the )?(?:lead|pace|front)"
          r"|in touch|in-touch|tracked|chased|on (?:the )?pace"),
    (3.5, r"front of mid[- ]?(?:division|divison|field)|held up in touch|towards (?:the )?front"),
    (3.0, r"(?:in )?[mn]id[- ]?(?:division|divison|field)|midfield|mid field"),
    (2.5, r"held up in mid[- ]?(?:division|divison|field)|rear of mid[- ]?(?:division|divison|field)|off the pace"),
    (1.5, r"towards (?:the )?rear|behind|^outpaced"),
    (1.0, r"held up|in rear|at (?:the )?rear|rear|in last|last pair|last trio|detached|dropped out"
          r"|always (?:behind|rear|in rear|last)|tailed off|waited with"),
]

#: A place in the field given as a place ("raced in 4th", "off the pace in 6th",
#: "close 3rd"). Only in these frames: a bare "4th" in a jumps comment is the
#: fourth obstacle ("mistake 4th"), not a position.
_ORD_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
              "eighth": 8, "ninth": 9, "tenth": 10}
_ORD = r"(?P<o>\d{1,2})(?:st|nd|rd|th)|(?P<w>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
_ORDINAL_RX = re.compile(
    rf"\b(?:(?:raced|ran|settled|travelled|sat|was|soon|keenly|freely) (?:\w+ )?in (?:a )?(?:close |distant )?"
    rf"|off the pace in |^(?:a )?close |^in (?:a )?(?:close )?)(?:{_ORD})\b")

#: Phrases about the start. A slow start is an early position only when the
#: comment gives no other: "dwelt, soon tracked leaders" raced prominently.
_START_VOCAB = (r"slowly away|slow away|dwelt|missed (?:the )?break|started (?:very )?slowly|badly away"
                r"|reared (?:as|when) (?:the )?stalls opened|awkward (?:start|leaving stalls)"
                r"|very slowly away|lost (?:several|many|a few)? ?lengths (?:at|leaving) (?:the )?start"
                r"|(?:very )?slow(?:ly)? into stride|a little slow(?:ly away)?"
                r"|went (?:right|left|badly right|badly left) (?:at |leaving |from )?(?:the )?start"
                r"|jumped (?:right|left|awkwardly) (?:out of|from|leaving) (?:the )?stalls")
_START_SCORE = 1.0
#: "broke well": a good start, and prominent unless the comment says otherwise.
_BROKE_WELL_RX = re.compile(r"\bbroke (?:well|smartly|fast|quickly)\b")
_BROKE_WELL_SCORE = 4.0

#: A lead taken in the first furlong or two is a front-runner's race, whatever
#: the comment's first word: "prominent early, led after 1f" set the pace.
_EARLY_LEAD_RX = re.compile(
    r"\b(?:led after (?:1|2|one|two|a|half a)\s?(?:f|furlongs?)\b|led after (?:the )?(?:1st|2nd|first|second)\b"
    r"|led early|soon led|led from (?:the )?start|quickly away and led"
    r"|broke (?:well|smartly|fast|quickly) (?:and |to )(?:lead|led)"
    r"|(?:went|pushed along|pushed|headway|sent|driven|ridden|broke well|niggled) to lead"
    r" (?:after (?:1|2|one|two|a|half a)\s?(?:f|furlongs?)|early|soon)\b)")
_EARLY_LEAD_SCORE = 5.8

_STYLE_RX = [(v, re.compile(rf"\b(?:{p})\b")) for v, p in _STYLE_VOCAB]
_START_RX = re.compile(rf"\b(?:{_START_VOCAB})\b")
#: How far after a slow start the comment may still place the horse, in characters.
_START_LOOKAHEAD = 45

#: Parsed values at or above this are a place in the field: value - ORDINAL_BASE.
ORDINAL_BASE = 100.0

STYLE_CLASSES = ("lead", "prominent", "mid", "rear")


def style_class(score) -> np.ndarray:
    """0 lead (>= 5.5), 1 prominent (>= 4.0), 2 mid (>= 2.5), 3 rear; NaN kept."""
    s = np.asarray(score, dtype=float)
    out = np.select([s >= 5.5, s >= 4.0, s >= 2.5, s >= 0], [0.0, 1.0, 2.0, 3.0], default=np.nan)
    return np.where(np.isnan(s), np.nan, out)


def _first_style(c: str, start: int = 0) -> tuple[int, float] | None:
    """(position, value) of the earliest positional phrase at or after `start`;
    on a tie the longer phrase wins ("held up in touch", not "held up"). A place
    given as an ordinal comes back as ORDINAL_BASE + place."""
    best = None
    for value, rx in _STYLE_RX:
        m = rx.search(c, start)
        if m is None:
            continue
        key = (m.start(), -(m.end() - m.start()))
        if best is None or key < best[0]:
            best = (key, value)
    m = _ORDINAL_RX.search(c, start)
    if m is not None:
        place = int(m.group("o")) if m.group("o") else _ORD_WORDS[m.group("w")]
        key = (m.start(), -(m.end() - m.start()))
        if place >= 1 and (best is None or key < best[0]):
            best = (key, ORDINAL_BASE + place)
    return None if best is None else (best[0][0], best[1])


def early_style_from_comment(comment) -> float:
    """The early position a comment gives: a style score (6 made all .. 1 held up
    in rear), or ORDINAL_BASE + place when it names the place ("raced in 4th").
    NaN when it gives none."""
    if not isinstance(comment, str) or not comment.strip():
        return np.nan
    c = comment.lower().strip()
    first = _first_style(c)
    out = np.nan if first is None else first[1]
    sm = _START_RX.search(c)
    bw = _BROKE_WELL_RX.search(c)
    for m, default in ((sm, _START_SCORE), (bw, _BROKE_WELL_SCORE)):
        if m is not None and (first is None or m.start() < first[0]):
            # the start comes first: take the next position if the comment
            # places the horse soon after it, otherwise the start stands
            nxt = _first_style(c, m.end())
            out = nxt[1] if nxt is not None and nxt[0] - m.end() <= _START_LOOKAHEAD else default
            break
    early = ",".join(re.split(r"[,;]", c)[:2])
    if (np.isnan(out) or (out < 5.5)) and _EARLY_LEAD_RX.search(early):
        return _EARLY_LEAD_SCORE
    return out


def parse_styles(comments: pd.Series) -> pd.Series:
    """`early_style_from_comment` over a column, parsing each distinct text once."""
    codes, uniques = pd.factorize(comments.astype("string").str.lower().str.strip(), sort=False)
    vals = np.array([early_style_from_comment(u) for u in uniques], dtype=float)
    out = np.full(len(comments), np.nan)
    ok = codes >= 0
    out[ok] = vals[codes[ok]]
    return pd.Series(out, index=comments.index)


# ---------------------------------------------------------------------------
# Shared pieces: race code, day index, outcomes of a past run
# ---------------------------------------------------------------------------

#: Lengths beaten are converted to pounds at the trip and capped here: a horse
#: beaten 30 lengths was not trying for most of them (the same floor
#: `model.perf_figures` puts under a bad run).
LBS_CAP = 20.0

#: Field-size bands used by every cell in this module and in `model.draw_curve`.
FIELD_BANDS = (0, 7, 11, 15, 99)          # 2-7 | 8-11 | 12-15 | 16+

#: Distance bands (furlongs) for the coarse levels of a pooling hierarchy.
DIST_BANDS = (0, 6.5, 8.5, 12.5, 16.5, 20.5, 24.5, 99)

#: Columns describing the race itself. They feed lagged statistics and must
#: never be features.
RACE_SHAPE_POST_RACE = frozenset({
    "rs_style", "rs_class", "rs_epf", "rs_nfp", "rs_nfp_c", "rs_lbs", "rs_lbs_c",
})


def race_code(df: pd.DataFrame) -> pd.Series:
    """flat / aw / hurdle / chase / nhflat, from race_type and surface_type."""
    rt = df.get("race_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    sf = df.get("surface_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    code = np.select(
        [rt.str.contains("chase"), rt.str.contains("hurdle"),
         rt.str.contains(r"nh flat|bumper|national hunt flat|n\.h\. flat")],
        ["chase", "hurdle", "nhflat"], default="flat")
    aw = sf.str.contains(r"\baw\b|all[- ]weather|polytrack|tapeta|fibresand|standard|sand")
    return pd.Series(np.where((code == "flat") & aw, "aw", code), index=df.index)


def day_index(df: pd.DataFrame) -> np.ndarray:
    """Race date as an integer day number (NaT becomes -1, which matches nothing)."""
    d = pd.to_datetime(df["race_date"], errors="coerce")
    out = d.values.astype("datetime64[D]").astype(np.int64)
    return np.where(d.isna().to_numpy(), -1, out)


def race_key(df: pd.DataFrame) -> pd.Series:
    if "raceid" in df.columns:
        return df["raceid"].astype(str)
    return (df["race_date"].astype(str) + "|" + df["track"].astype(str) + "|" + df["race_time"].astype(str))


def field_size(df: pd.DataFrame) -> pd.Series:
    n = pd.to_numeric(df.get("number_of_runners"), errors="coerce") if "number_of_runners" in df.columns else None
    rows = df.groupby(race_key(df), sort=False)["horse_name"].transform("size").astype(float)
    return rows if n is None else n.fillna(rows)


def add_run_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Two measures of how a past run went, each centred within its race.

    rs_nfp     normalised finishing position, 1 winner .. 0 last
    rs_nfp_c   rs_nfp less the race's mean (+ = beat the average runner)
    rs_lbs     lengths behind the winner in pounds at the trip, capped at LBS_CAP
    rs_lbs_c   the race's mean rs_lbs less this runner's (+ = beat the average)

    Pounds, not lengths, because a length is worth three pounds over five
    furlongs and under one over two miles: an average over trips has to be on
    the weight scale. Both are post-race and feed lagged statistics only.
    """
    from model.perf_figures import lbs_per_length, parse_beaten_lengths
    rk = race_key(df)
    n = field_size(df)
    pos = pd.to_numeric(df.get("placing_numerical"), errors="coerce")
    nfp = (n - pos) / (n - 1).where(n > 1)
    df["rs_nfp"] = nfp.clip(0, 1)
    df["rs_nfp_c"] = df["rs_nfp"] - df["rs_nfp"].groupby(rk).transform("mean")
    lb = df.get("total_dst_bt", pd.Series(np.nan, index=df.index)).map(parse_beaten_lengths)
    lb = lb.where(pos != 1, 0.0).where(pos.notna())
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")
    df["rs_lbs"] = (lb * lbs_per_length(dist.fillna(8.0).to_numpy())).clip(upper=LBS_CAP)
    df["rs_lbs_c"] = df["rs_lbs"].groupby(rk).transform("mean") - df["rs_lbs"]
    return df


# ---------------------------------------------------------------------------
# Lag-safe lookups: earlier days only, bit-exact whatever today's results are
# ---------------------------------------------------------------------------

_DAY_BITS = 20


def asof_decayed_mean(key, day, value, qkey, qday, halflife_days: float = np.inf):
    """Time-decayed mean and effective sample size of `value`, per query, over the
    history rows with the query's key on days STRICTLY BEFORE the query's day.

    A past day counts ``2 ** (-(query day - day) / halflife_days)``. The sums are
    cumulative within each key and read at the key's last day before the query,
    so nothing from the query's own day -- or any other key's -- enters them, not
    even as rounding: a leak test that flips today's results sees identical bits.
    Rows with a missing value or a negative key are not history; queries with a
    negative key get NaN.
    """
    key = np.asarray(key, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    v = np.asarray(value, dtype=float)
    qkey = np.asarray(qkey, dtype=np.int64)
    qday = np.asarray(qday, dtype=np.int64)
    out_m = np.full(len(qkey), np.nan)
    out_n = np.zeros(len(qkey))
    ok = (key >= 0) & (day >= 0) & np.isfinite(v)
    if not ok.any() or len(qkey) == 0:
        return out_m, out_n
    k, t, v = key[ok], day[ok], v[ok]
    base = int(min(t.min(), qday[qday >= 0].min() if (qday >= 0).any() else t.min()))
    t = t - base
    qd = np.clip(qday - base, 0, (1 << _DAY_BITS) - 1)
    code = (k << _DAY_BITS) | t
    u, inv = np.unique(code, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    uk = u >> _DAY_BITS
    ut = (u & ((1 << _DAY_BITS) - 1)).astype(float)
    new = np.r_[True, uk[1:] != uk[:-1]]
    blk = np.cumsum(new) - 1
    t0 = ut[np.flatnonzero(new)][blk]
    finite = np.isfinite(halflife_days)
    grow = np.exp2(np.minimum((ut - t0) / halflife_days, 1000.0)) if finite else np.ones(len(u))
    cs = pd.Series(grow * s).groupby(blk).cumsum().to_numpy()
    cc = pd.Series(grow * c).groupby(blk).cumsum().to_numpy()
    q = (qkey << _DAY_BITS) | qd
    p = np.searchsorted(u, q, side="left") - 1
    p0 = np.clip(p, 0, None)
    valid = (p >= 0) & (qkey >= 0) & (qday >= 0) & (uk[p0] == qkey)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        m = cs[p0] / cc[p0]
        decay = np.exp2(-np.minimum((qd - t0[p0]) / halflife_days, 1000.0)) if finite else 1.0
        n = cc[p0] * decay
    out_m[valid] = m[valid]
    out_n[valid] = np.broadcast_to(n, valid.shape)[valid]
    return out_m, out_n


def shrink(mean, n_eff, prior, k: float) -> np.ndarray:
    """Empirical-Bayes pull of a cell mean toward the level above it."""
    mean = np.asarray(mean, dtype=float)
    n = np.nan_to_num(np.asarray(n_eff, dtype=float), nan=0.0)
    prior = np.broadcast_to(np.asarray(prior, dtype=float), mean.shape)
    return np.where(n > 0, (np.nan_to_num(mean) * n + prior * k) / (n + k), prior)


def pooled_cell_value(levels, day, value, halflife_days: float, ks, q_levels=None, qday=None,
                      top_prior: float = 0.0):
    """A cell mean pooled down a hierarchy of keys, coarsest first.

    `levels` is a list of integer key arrays for the history (one per level);
    `q_levels` the same keys for the queries (default: the history rows
    themselves). Each level's decayed, earlier-days mean is shrunk toward the
    pooled value of the level above with strength `ks[i]` (in effective
    observations). Returns (value, effective sample size at the finest level).
    """
    q_levels = levels if q_levels is None else q_levels
    qday = day if qday is None else qday
    est = np.full(len(q_levels[0]), float(top_prior))
    n_last = np.zeros(len(q_levels[0]))
    for lvl, qlvl, k in zip(levels, q_levels, ks):
        m, n = asof_decayed_mean(lvl, day, value, qlvl, qday, halflife_days)
        est = shrink(m, n, est, k)
        n_last = n
    return est, n_last


def _codes(*parts) -> np.ndarray:
    """One integer code per distinct combination of the given columns."""
    if len(parts) == 1:
        c = pd.factorize(pd.Series(parts[0]).astype(str))[0]
    else:
        c = pd.factorize(pd.MultiIndex.from_arrays([pd.Series(p).astype(str).to_numpy() for p in parts]))[0]
    return c.astype(np.int64)


def bands(values, edges) -> np.ndarray:
    return pd.cut(pd.Series(values), list(edges), labels=False, right=True).to_numpy()


# ---------------------------------------------------------------------------
# 2. Per-run styles, and the projection for today
# ---------------------------------------------------------------------------

#: Half-life, in the horse's runs, of the weight on a past run's style.
STYLE_HALFLIFE_RUNS = 4.0

#: Strength, in runs, of the population prior a horse's style is shrunk toward.
#: Tuned on 2023-26Q1 comments (research/queries/style_projection_tuning.py, run
#: 18): half-life 4 runs, prior 4 runs, no per-code level -- four-class Brier
#: skill 0.042 over the base rates, p_lead calibrated decile by decile.
STYLE_PRIOR_RUNS = 4.0

#: Half-life in days of the population prior (styles drift with the riding).
STYLE_POP_HALFLIFE_DAYS = 1095.0


def add_run_styles(df: pd.DataFrame) -> pd.DataFrame:
    """rs_style (1-6), rs_class (0 lead .. 3 rear) and rs_epf (0 led .. 1 last)
    of each run, from its comment. Post-race: the inputs to the projection.

    A comment that names the place ("raced in 4th") gives rs_epf exactly,
    (place - 1) / (N - 1), and a style score standing for its class."""
    raw = parse_styles(df["comment"]).to_numpy() if "comment" in df.columns else np.full(len(df), np.nan)
    n = field_size(df).to_numpy(dtype=float)
    is_ord = raw >= ORDINAL_BASE
    place = np.where(is_ord, raw - ORDINAL_BASE, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ord_epf = np.clip((place - 1) / np.where(n > 1, n - 1, np.nan), 0, 1)
    ord_score = np.select([place == 1, place == 2, ord_epf <= 0.35, ord_epf <= 0.65], [6.0, 5.0, 4.5, 3.0],
                          default=1.5)
    style = np.where(is_ord, np.where(np.isnan(ord_epf) & (place > 1), np.nan, ord_score), raw)
    df["rs_style"] = style
    df["rs_class"] = style_class(style)
    df["rs_epf"] = np.where(is_ord, ord_epf, epf_norm_from_style(style, n))
    return df


def horse_decayed_prior(horse, day, values: dict, halflife_runs: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Recency-weighted sums of each value over the horse's runs on EARLIER DAYS.

    A run k runs back (counted in the horse's racing days) weighs
    ``2 ** (-k / halflife_runs)``. Returns ({name: decayed sum}, {name: decayed
    count of runs where it was known}, number of earlier racing days), each per
    row. Exact prefix sums: today's values never enter a row's prior, even as
    rounding.
    """
    h = _codes(horse)
    t = np.asarray(day, dtype=np.int64)
    t = t - t[t >= 0].min() if (t >= 0).any() else t
    code = (h << _DAY_BITS) | np.clip(t, 0, None)
    u, inv = np.unique(code, return_inverse=True)
    uh = u >> _DAY_BITS
    new = np.r_[True, uh[1:] != uh[:-1]]
    blk = np.cumsum(new) - 1
    k = np.arange(len(u)) - np.flatnonzero(new)[blk]            # racing-day index within horse
    grow = np.exp2(k / halflife_runs)
    shrinkf = np.exp2(-k / halflife_runs)
    g = pd.Series(blk)

    def prior(cell_vals):
        cs = pd.Series(grow * cell_vals).groupby(g).cumsum()
        return (cs.groupby(g).shift(1).fillna(0.0).to_numpy() * shrinkf)[inv]

    out, cnt = {}, {}
    for name, vals in values.items():
        vals = np.asarray(vals, dtype=float)
        okv = np.isfinite(vals)
        out[name] = prior(np.bincount(inv, weights=np.where(okv, vals, 0.0), minlength=len(u)))
        cnt[name] = prior(np.bincount(inv, weights=okv.astype(float), minlength=len(u)))
    runs = k[inv].astype(float)
    return out, cnt, runs


#: Strength, in runs, of the horse's all-code style that its style in today's
#: code (flat / all-weather / hurdle / chase / bumper) is shrunk toward. None
#: pools every code, which the tuning preferred: a separate level cost skill at
#: every setting tried (the code-specific history is too short to be worth its
#: noise).
STYLE_CODE_PRIOR_RUNS: float | None = None


def add_style_projection(df: pd.DataFrame, halflife_runs: float = STYLE_HALFLIFE_RUNS,
                         prior_runs: float = STYLE_PRIOR_RUNS,
                         code_prior_runs: float | None = STYLE_CODE_PRIOR_RUNS) -> pd.DataFrame:
    """Today's projected run style from the horse's earlier runs.

    p_lead, p_prom, p_mid, p_rear   chance of each early position class
    pred_epf                        projected epf_norm (0 = leads, 1 = last)
    style_n_eff                     recency-weighted count of parsed earlier runs

    Three levels, each shrunk toward the one above: the population (same code,
    distance band, debutant or not, from earlier days), the horse over all its
    runs (weight `prior_runs`), and the horse in today's code (weight
    `code_prior_runs`). A horse seen once is not a certain front-runner, and a
    debutant takes the population's figures.
    """
    if "rs_class" not in df.columns:
        df = add_run_styles(df)
    day = day_index(df)
    cls = df["rs_class"].to_numpy(dtype=float)
    ind = {f"c{c}": np.where(np.isnan(cls), np.nan, (cls == c).astype(float)) for c in range(4)}
    ind["epf"] = df["rs_epf"].to_numpy(dtype=float)
    horse = df["horse_name"].astype(str)
    sums, w, runs = horse_decayed_prior(horse, day, ind, halflife_runs)
    code = race_code(df)
    if code_prior_runs is not None:
        sums_c, w_c, _ = horse_decayed_prior(horse + "|" + code, day, ind, halflife_runs)

    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")
    pkey = _codes(code, bands(dist, DIST_BANDS), (runs == 0).astype(int))

    def level(name, pop_default):
        pop, _ = asof_decayed_mean(pkey, day, ind[name], pkey, day, STYLE_POP_HALFLIFE_DAYS)
        est = (sums[name] + prior_runs * np.where(np.isnan(pop), pop_default, pop)) / (w[name] + prior_runs)
        if code_prior_runs is not None:
            est = (sums_c[name] + code_prior_runs * est) / (w_c[name] + code_prior_runs)
        return est

    P = np.vstack([level(f"c{c}", 0.25) for c in range(4)]).T
    P = P / P.sum(axis=1, keepdims=True)
    for c, nm in enumerate(["p_lead", "p_prom", "p_mid", "p_rear"]):
        df[nm] = P[:, c]
    df["pred_epf"] = level("epf", 0.5)
    df["style_n_eff"] = w["c0"]
    return df


# ---------------------------------------------------------------------------
# 3. Race shape, from the projected styles of the field
# ---------------------------------------------------------------------------

#: A race has "no natural leader" when nobody's chance of leading reaches
#: SHAPE_LEAD_MIN, and a "contested" lead when a second runner's does.
SHAPE_LEAD_MIN = 0.30
SHAPE_SECOND_MIN = 0.30
SHAPE_NAMES = ("no natural leader", "one natural leader", "contested lead")


def add_race_shape(df: pd.DataFrame) -> pd.DataFrame:
    """How the race is likely to be run, from the projected styles of its field.

    shape_exp_leaders   expected number of runners that lead or dispute (sum of p_lead)
    shape_p_no_leader   chance no runner takes it up (product of 1 - p_lead)
    shape_p_contested   chance two or more do
    shape_front_share   share of the field expected on or close to the pace
    shape_lead1/lead2   the two largest chances of leading
    shape_lead_clarity  lead1 - lead2: how clear the likely leader is
    shape_mean_epf      the field's mean projected early position
    shape_bin           0 no natural leader, 1 one, 2 contested (SHAPE_NAMES)

    and for the runner within it:
    lead_share          its share of the field's claims on the lead
    lead_rank           1 = the most likely leader
    rel_epf             its projected early position less the field's mean
    """
    if "p_lead" not in df.columns:
        df = add_style_projection(df)
    rk = race_key(df)
    g = df.groupby(rk, sort=False)
    p = df["p_lead"].clip(0, 1 - 1e-9)
    df["shape_exp_leaders"] = g["p_lead"].transform("sum")
    log_q = np.log1p(-p)
    p0 = np.exp(log_q.groupby(rk).transform("sum"))
    p1 = p0 * (p / (1 - p)).groupby(rk).transform("sum")
    df["shape_p_no_leader"] = p0
    df["shape_p_contested"] = (1 - p0 - p1).clip(0, 1)
    n = g["p_lead"].transform("size").astype(float)
    df["shape_front_share"] = (df["p_lead"] + df["p_prom"]).groupby(rk).transform("sum") / n
    df["shape_lead1"] = g["p_lead"].transform("max")
    order = g["p_lead"].rank(ascending=False, method="first")
    df["shape_lead2"] = df["p_lead"].where(order == 2).groupby(rk).transform("max").fillna(0.0)
    df["shape_lead_clarity"] = df["shape_lead1"] - df["shape_lead2"]
    df["shape_mean_epf"] = g["pred_epf"].transform("mean")
    df["shape_bin"] = np.select([df["shape_lead1"] < SHAPE_LEAD_MIN, df["shape_lead2"] >= SHAPE_SECOND_MIN],
                                [0, 2], default=1).astype(float)
    df["lead_share"] = df["p_lead"] / df["shape_exp_leaders"].where(df["shape_exp_leaders"] > 0)
    df["lead_rank"] = g["p_lead"].rank(ascending=False, method="min")
    df["rel_epf"] = df["pred_epf"] - df["shape_mean_epf"]
    return df


# ---------------------------------------------------------------------------
# 4. What a position is worth within the shape, at this course and trip
# ---------------------------------------------------------------------------

#: Half-life in days of a past race in the position-value cells.
PV_HALFLIFE_DAYS = 1095.0

#: Shrinkage strengths (effective runners) for the four levels, coarsest first:
#: code x shape -> + distance band x field band -> course x code x trip x shape
#: -> + field band. The coarsest shrinks toward zero (the outcomes are centred).
PV_K = (400.0, 200.0, 100.0, 50.0)

POSITION_OUTCOMES = {"nfp": "rs_nfp_c", "lbs": "rs_lbs_c"}


def _position_levels(df: pd.DataFrame) -> list[np.ndarray]:
    code = race_code(df)
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")
    fb = bands(field_size(df), FIELD_BANDS)
    db = bands(dist, DIST_BANDS)
    shape = df["shape_bin"].to_numpy()
    track = df["track"].astype(str).str.lower().str.strip()
    return [_codes(code, shape), _codes(code, db, fb, shape),
            _codes(track, code, dist.round(0), shape), _codes(track, code, dist.round(0), fb, shape)]


def add_position_value(df: pd.DataFrame, halflife_days: float = PV_HALFLIFE_DAYS, ks=PV_K) -> pd.DataFrame:
    """The value of today's expected position within today's expected shape.

    pv_exp_nfp / pv_exp_lbs   how runners PROJECTED to race where this one is
                              projected to (lead / prominent / mid / rear) fared
                              in races of this projected shape at this course,
                              trip and field size, on earlier days: centred
                              finishing position, and pounds beaten less than the
                              race average. A forecast before the off can use
                              exactly this.
    pv_act_nfp / pv_act_lbs   sum over positions of today's chance of racing there
                              times what runners that ACTUALLY raced there were
                              worth in this shape here: the chance of a position
                              and its value, kept apart.
    pv_front_bias_lbs         lead minus rear, in pounds: how much this shape
                              here has favoured racing up with the pace.
    pv_n_eff                  effective runners behind pv_exp at the finest cell.
    """
    for col in ("rs_nfp_c", "rs_lbs_c"):
        if col not in df.columns:
            df = add_run_outcomes(df)
            break
    if "shape_bin" not in df.columns:
        df = add_race_shape(df)
    day = day_index(df)
    levels = _position_levels(df)
    P = df[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy(dtype=float)
    g_exp = P.argmax(axis=1).astype(np.int64)
    cls = df["rs_class"].to_numpy(dtype=float)
    act = np.where(np.isnan(cls), 0, cls).astype(np.int64)
    for name, col in POSITION_OUTCOMES.items():
        y = df[col].to_numpy(dtype=float)
        lv = [np.where(b >= 0, b * 4 + g_exp, -1) for b in levels]
        val, n = pooled_cell_value(lv, day, y, halflife_days, ks)
        df[f"pv_exp_{name}"] = val
        if name == "nfp":
            df["pv_n_eff"] = n
        hist = [np.where((b >= 0) & ~np.isnan(cls), b * 4 + act, -1) for b in levels]
        V = np.empty((len(df), 4))
        for s in range(4):
            q = [np.where(b >= 0, b * 4 + s, -1) for b in levels]
            V[:, s], _ = pooled_cell_value(hist, day, y, halflife_days, ks, q_levels=q, qday=day)
        df[f"pv_act_{name}"] = (P * V).sum(axis=1)
        if name == "lbs":
            df["pv_front_bias_lbs"] = V[:, 0] - V[:, 3]
    return df


#: The block's features, in the order a model sees them.
RACE_SHAPE_FEATURES = [
    "p_lead", "p_prom", "p_mid", "p_rear", "pred_epf", "style_n_eff",
    "shape_exp_leaders", "shape_p_no_leader", "shape_p_contested", "shape_front_share",
    "shape_lead1", "shape_lead2", "shape_lead_clarity", "shape_mean_epf", "shape_bin",
    "lead_share", "lead_rank", "rel_epf",
    "pv_exp_nfp", "pv_exp_lbs", "pv_act_nfp", "pv_act_lbs", "pv_front_bias_lbs", "pv_n_eff",
]


def add_race_shape_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The whole position block on a frame of history plus (optionally) today's card.

    Needs race_date, race_time, track, horse_name, number_of_runners,
    placing_numerical, total_dst_bt, dist_furlongs, race_type, surface_type and
    comment. Card rows carry no comment and no result; they get features from
    the history before their day and contribute nothing to it.
    """
    df = add_run_outcomes(df)
    df = add_run_styles(df)
    df = add_style_projection(df)
    df = add_race_shape(df)
    df = add_position_value(df)
    assert not set(RACE_SHAPE_FEATURES) & RACE_SHAPE_POST_RACE
    return df, list(RACE_SHAPE_FEATURES)
