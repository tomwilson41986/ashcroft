"""
Pace Prediction & Running Position Feature Engineering.

Extends the existing EPF-based pace metrics with a comprehensive framework
for predicting in-race pace scenarios and running positions. Features are
designed to capture:

1. **Running Style Profiles** — Granular multi-phase position extraction from
   comments (early, mid, late) with progression/regression tracking.
2. **Pace Scenario Prediction** — Race-level pace forecasts based on the
   assembled field's historical running styles.
3. **Track/Distance Pace Bias** — Course-specific front-runner advantage with
   expanding-mean priors (lag-safe).
4. **Pace Fit** — How well a horse's preferred running style matches the
   predicted pace scenario and track bias.
5. **Tactical Metrics** — Keenness, trouble-in-running, position sustainability.
6. **Field-size-normalised early position** (`epf_norm`, framework §6.1) and the
   run-style profile built on it — mean, SD (versatility) and modal style.
7. **Lead probability** (`predicted_lead_prob`, §6.2) — a softmax over each
   runner's *predicted* early position, so "wants to lead" is a probability that
   sums to one over the field rather than a count of front-runners.
8. **Position Change Statistic** (§6.4) — how much positional churn a race had,
   with a lag-safe course/distance baseline to flag abnormal races.

Why `epf_norm` and not the 1–6 style score: the style score is an absolute
description ("chased leaders"), and the same words mean a very different place
in the field in a 6-runner race and a 20-runner one. §6.1 asks for
``(early_position - 1) / (N - 1)``, which is 0 for the leader and 1 for the
last horse whatever the field size, and that is the scale every race-shape
aggregate below is built on.

There are no sectionals and no in-running positions in this database, so the
early position is *estimated* from the horse's in-running comment: each style
phrase carries an absolute rank anchor (a horse that "led" was first, in any
field) and a relative depth (a horse "in midfield" was halfway back, wherever
that is), and the two are combined against today's field size.

All features that feed a model are strictly lag-safe (shift-1 per horse, or
`race_lagged_expanding_mean` for keys that repeat inside a race). The per-run
columns parsed from *today's* comment — `epf_norm`, `pos_gain`, `run_style`,
`pcs_race` — describe the race being predicted and are listed in
`PACE_POST_RACE_ONLY`; they exist only so the career aggregates above them can
be built, and none of them appears in the exported feature lists.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.lagsafe import ensure_race_key, race_lagged_expanding_mean


# ---------------------------------------------------------------------------
# Comment parsing — multi-phase position extraction
# ---------------------------------------------------------------------------

def parse_run_style(comment: str) -> dict:
    """Extract detailed running narrative from race comment.

    Returns dict with:
        early_pos       : 1.0–6.0 early race position
        mid_move        : -3 to +3 mid-race movement
        late_move       : -3 to +3 finishing movement
        finishing_effort: -3 to +3 effort level at finish
        was_keen        : 0/1 raced keenly / pulled hard
        had_trouble     : 0/1 hampered / denied clear run
        led_at_furlong  : int or None — when horse took lead
    """
    result = {
        "early_pos": 3.0,
        "mid_move": 0.0,
        "late_move": 0.0,
        "finishing_effort": 0.0,
        "was_keen": 0,
        "had_trouble": 0,
        "led_at_furlong": np.nan,
    }

    if not comment or not isinstance(comment, str):
        return result

    c = comment.lower().strip()

    # --- Early position (first phrase) ---
    if re.search(r"made virtually all|made all|made most", c):
        result["early_pos"] = 6.0
    elif re.search(r"soon led|led,|led early|led after|led before", c):
        result["early_pos"] = 5.8
    elif re.search(r"disputed|with leader", c):
        result["early_pos"] = 5.5
    elif re.search(r"chased leader|tracked leader|chased winner", c):
        result["early_pos"] = 5.0
    elif re.search(r"chased leaders|tracked leaders|tracked front pair|tracked leading pair", c):
        result["early_pos"] = 4.5
    elif re.search(
        r"pressed leader|prominent|close up|in touch(?! midfield)|in-touch"
        r"|pressing leaders|chasing leaders|tracking leaders",
        c,
    ):
        result["early_pos"] = 4.0
    elif re.search(r"front of mid", c):
        result["early_pos"] = 3.5
    elif re.search(r"held up in touch", c):
        result["early_pos"] = 3.5
    elif re.search(r"held up in midfield|held up in mid-division|held up in mid division", c):
        result["early_pos"] = 2.5
    elif re.search(r"mid-division|midfield|mid division", c):
        result["early_pos"] = 3.0
    elif re.search(r"held up towards rear|towards rear", c):
        result["early_pos"] = 1.5
    elif re.search(
        r"held up behind|held up|behind|last pair|in rear|always rear|always behind",
        c,
    ):
        result["early_pos"] = 1.0

    # --- Mid-race movement (with furlong markers) ---
    m = re.search(r"(rapid |good |smooth )?headway.*?(\d+)f out", c)
    if m:
        qualifier = m.group(1) or ""
        f_out = int(m.group(2))
        base = 2.5 if "rapid" in qualifier or "good" in qualifier else 1.5
        if "smooth" in qualifier:
            base = 2.0
        if f_out >= 3:
            result["mid_move"] = base
        else:
            result["late_move"] = base
    elif "headway" in c:
        result["mid_move"] = 1.5

    # --- Late movement ---
    if re.search(r"ran on|stayed on strongly|stayed on well", c):
        result["late_move"] = max(result["late_move"], 3.0)
    elif re.search(r"stayed on|kept on well", c):
        result["late_move"] = max(result["late_move"], 2.0)
    elif re.search(r"kept on", c):
        result["late_move"] = max(result["late_move"], 1.5)

    if "weakened" in c:
        result["late_move"] = min(result["late_move"], -3.0)
    elif re.search(r"no extra|one pace", c):
        result["late_move"] = min(result["late_move"], -1.5)
    elif "faded" in c:
        result["late_move"] = min(result["late_move"], -2.5)

    # --- Finishing effort ---
    if re.search(r"ridden out|driven out", c):
        result["finishing_effort"] = 3.0
    elif re.search(r"pushed out|pushed clear", c):
        result["finishing_effort"] = 2.0
    elif re.search(r"unchallenged|easily|comfortably", c):
        result["finishing_effort"] = 1.0  # positive = won easily
    elif re.search(r"never dangerous|always behind|tailed off", c):
        result["finishing_effort"] = -3.0

    # --- Keenness ---
    if re.search(r"pulled hard|raced freely|raced keenly|keen\b", c):
        result["was_keen"] = 1

    # --- Trouble in running ---
    if re.search(
        r"hampered|squeezed|short of room|bumped|checked"
        r"|denied clear run|not clear run|badly hampered|fell|brought down"
        r"|unseated|slipped",
        c,
    ):
        result["had_trouble"] = 1

    # --- Led at furlong marker ---
    m = re.search(r"led (?:over |approaching )?(\d+)f out", c)
    if m:
        result["led_at_furlong"] = float(m.group(1))
    elif re.search(r"led inside final|led close home", c):
        result["led_at_furlong"] = 0.5

    return result


# ---------------------------------------------------------------------------
# Field-size-normalised early position (framework §6.1)
# ---------------------------------------------------------------------------

#: Style scores emitted by :func:`parse_run_style`, ascending (1 = held up in
#: rear, 6 = made all).
_STYLE_KNOTS = np.array([1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.8, 6.0])

#: Absolute rank anchor per style: "led" is first and "chased the leader" is
#: second in a 6-runner race and in a 20-runner race alike.
_STYLE_ANCHOR = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.50, 1.00, 0.50, 0.00, 0.00])

#: Relative depth per style, as a fraction of the field behind the leader:
#: "midfield" is halfway back wherever halfway is.
_STYLE_DEPTH = np.array([0.92, 0.80, 0.62, 0.50, 0.40, 0.12, 0.02, 0.00, 0.00, 0.00, 0.00])

#: An absolute anchor may never push a horse past this fraction of the field —
#: in a five-runner race "tracked the leaders" is second or third, not a fixed
#: rank that would land it in mid-division. Also what keeps the style -> position
#: map monotone at small field sizes.
ANCHOR_MAX_DEPTH = 0.25

#: epf_norm cut points for the §6.1 run-style labels, most forward first.
RUN_STYLE_LABELS = ("detached", "held-up", "mid-division", "prominent", "led")
RUN_STYLE_EDGES = (0.05, 0.25, 0.55, 0.85)   # led | prominent | mid | held-up | detached

#: Softmax temperature for :func:`predicted lead probability`. 0.12 means a
#: runner a tenth of the field more forward than a rival is ~2.3x as likely to
#: lead; smaller = winner-takes-all, larger = flatter.
LEAD_SOFTMAX_TEMPERATURE = 0.12

#: §6.2: a runner whose predicted epf_norm is below this is "going forward".
PACE_PRESSURE_THRESHOLD = 0.25

#: Share of the field going forward that separates a strong / even / slow pace.
PACE_CLASS_EDGES = (0.12, 0.30)

#: Half-life, in runs, of the recency weighting on the run-style profile (§6.1).
STYLE_RECENCY_HALFLIFE_RUNS = 4.0

#: How much of the predicted early position comes from the jockey's own
#: tendency rather than the horse's history.
JOCKEY_STYLE_WEIGHT = 0.15

#: Per-run columns parsed from the race being predicted. They may only ever
#: feed a lagged feature — see the module docstring and POST_RACE_ONLY in
#: model/custom_metrics.py.
PACE_POST_RACE_ONLY = frozenset({
    "epf_norm", "finish_pos_norm", "pos_gain", "run_style", "early_pos_est",
    "pcs_race", "pcs_race_norm", "pcs_vs_baseline",
})


def epf_norm_from_style(style, n_runners) -> np.ndarray:
    """``(early_position - 1) / (N - 1)`` implied by a comment style score.

    0 = led, 1 = last of the field. Each style carries an absolute rank anchor
    and a relative depth; the anchor is capped at ``ANCHOR_MAX_DEPTH`` of the
    field so the mapping stays monotone in small fields.
    """
    s = np.asarray(style, dtype=float)
    n = np.broadcast_to(np.asarray(n_runners, dtype=float), s.shape)
    denom = np.where(n > 1, n - 1, np.nan)

    # Evaluate the knots once per distinct field size, then force the result to
    # be non-increasing in the style score: the raw anchors are not monotone
    # ("prominent" carries one, "front of midfield" does not), and in a tiny
    # field an uncapped anchor would rank a prominent horse behind a midfield
    # one. Interpolating the *monotone* knot values keeps that true in between.
    key = np.where(np.isnan(denom), -1.0, denom)
    uniq, inv = np.unique(key, return_inverse=True)
    dd = np.where(uniq > 0, uniq, np.nan)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        tab = np.minimum(_STYLE_ANCHOR[None, :], ANCHOR_MAX_DEPTH * dd) / dd + _STYLE_DEPTH[None, :]
    tab = np.minimum.accumulate(np.clip(tab, 0.0, 1.0), axis=1)

    sc = np.clip(s, _STYLE_KNOTS[0], _STYLE_KNOTS[-1])
    j = np.clip(np.searchsorted(_STYLE_KNOTS, sc, side="right") - 1, 0, len(_STYLE_KNOTS) - 2)
    x0, x1 = _STYLE_KNOTS[j], _STYLE_KNOTS[j + 1]
    y0, y1 = tab[inv, j], tab[inv, j + 1]
    out = y0 + (sc - x0) / (x1 - x0) * (y1 - y0)
    return np.where(np.isnan(s) | np.isnan(denom), np.nan, out)


def run_style_from_epf_norm(epf_norm) -> np.ndarray:
    """§6.1 run style as an ordinal code: 4 led ... 0 detached (NaN preserved)."""
    e = np.asarray(epf_norm, dtype=float)
    code = np.full(e.shape, np.nan)
    ok = ~np.isnan(e)
    # searchsorted over the edges gives 0 for the most forward band
    code[ok] = 4 - np.searchsorted(np.asarray(RUN_STYLE_EDGES, float), e[ok], side="right")
    return np.clip(code, 0, 4)


def position_change_statistic(df: pd.DataFrame, from_col: str = "early_pos_est",
                              to_col: str = "placing_numerical",
                              race_col: str = "raceid") -> pd.DataFrame:
    """§6.4 Position Change Statistic, one row per race.

    ``PCS(A->B) = sqrt(sum_i (pos_A,i - pos_B,i)^2) / N``. Low = little changed
    between the two junctures, so the race was run to a shape that suited early
    position. The only two junctures this database supports are the comment's
    early position and the finish.

    ``pcs`` is in positions and grows with field size, so ``pcs_norm`` repeats
    the calculation on positions normalised to [0, 1] and is what the
    course/distance baseline is built from.
    """
    d = pd.DataFrame({
        "_race": ensure_race_key(df, race_col),
        "_a": pd.to_numeric(df[from_col], errors="coerce") if from_col in df.columns else np.nan,
        "_b": pd.to_numeric(df[to_col], errors="coerce") if to_col in df.columns else np.nan,
        "_n": pd.to_numeric(df["number_of_runners"], errors="coerce"),
    }).dropna(subset=["_a", "_b"])
    if d.empty:
        return pd.DataFrame(columns=["raceid", "n", "pcs", "pcs_norm"])
    denom = (d["_n"] - 1).where(d["_n"] > 1)
    d["_sq"] = (d["_a"] - d["_b"]) ** 2
    d["_sq_norm"] = ((d["_a"] - d["_b"]) / denom) ** 2
    g = d.groupby("_race", sort=False)
    out = g.agg(n=("_sq", "size"), _s=("_sq", "sum"), _sn=("_sq_norm", "sum")).reset_index()
    out["pcs"] = np.sqrt(out["_s"]) / out["n"]
    out["pcs_norm"] = np.sqrt(out["_sn"]) / out["n"]
    return out.rename(columns={"_race": "raceid"})[["raceid", "n", "pcs", "pcs_norm"]]


# ---------------------------------------------------------------------------
# PaceMetricsEngine — integrates into CustomMetricsEngine pipeline
# ---------------------------------------------------------------------------

def prior_mean_sd(df: pd.DataFrame, key: str, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(count, mean, sd) of `col` over each key's *earlier rows*.

    The same numbers as ``grp[col].apply(lambda x: x.shift(1).expanding().mean())``
    and its ``.std()`` twin, from three grouped cumulative sums instead of one
    Python call per group. The key here is always the horse — a horse runs once
    in a race, so stepping back a row is stepping back a race — and the database
    has hundreds of thousands of horses, which is a great many Python calls.
    """
    v = pd.to_numeric(df[col], errors="coerce")
    g = df[key]
    ok = v.notna().astype(float)
    # centre each group on its first value before squaring: the sum-of-squares
    # form loses precision otherwise, and a horse that has run the same race
    # three times should have a versatility of zero, not of 1e-8.
    centre = v.groupby(g, sort=False).transform("first").fillna(0.0)
    dv = (v - centre).fillna(0.0)
    n = ok.groupby(g, sort=False).cumsum() - ok
    s = dv.groupby(g, sort=False).cumsum() - dv
    q = (dv ** 2).groupby(g, sort=False).cumsum() - dv ** 2
    n_pos = n.where(n > 0)
    mean_d = s / n_pos
    var = (q - n * mean_d ** 2) / (n - 1).where(n > 1)
    return n, mean_d + centre.where(n_pos.notna()), np.sqrt(var.clip(lower=0))


class PaceMetricsEngine:
    """Advanced pace feature engineering.

    Call ``calculate(df)`` after EPF and NFP have been computed.
    All features are lag-safe.
    """

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all pace feature calculations in order."""
        df = df.copy()

        # Step 1: Parse detailed run style from comments
        df = self._parse_run_styles(df)

        # Step 2: Horse running style profiles (historical)
        df = self._calc_horse_style_profiles(df)

        # Step 2b: §6.1 run-style profile on the field-size-normalised scale
        df = self._calc_run_style_profile(df)

        # Step 3: Jockey & trainer style profiles
        df = self._calc_entity_style_profiles(df)

        # Step 4: Track/distance pace bias
        df = self._calc_track_distance_pace_bias(df)

        # Step 5: Race-level pace scenario prediction
        df = self._calc_predicted_pace_scenario(df)

        # Step 6: Pace fit — style vs scenario vs track
        df = self._calc_pace_fit(df)

        # Step 7: Tactical / behavioural features
        df = self._calc_tactical_features(df)

        # Step 8: Position sustainability
        df = self._calc_position_sustainability(df)

        # Step 8b: §6.4 position churn (PCS) and its course/distance baseline
        df = self._calc_position_churn(df)

        # Step 9: Within-race pace ranks
        df = self._calc_pace_ranks(df)

        return df

    # ------------------------------------------------------------------
    # Step 1: Parse multi-phase running style from comments
    # ------------------------------------------------------------------
    def _parse_run_styles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse comments into early_pos, mid_move, late_move, etc."""
        comment_col = "comment" if "comment" in df.columns else None
        if comment_col is None:
            # A live card has no in-running comments. The parsed columns still
            # have to exist, because every career aggregate below reads them —
            # they are simply empty for today and filled for the history.
            for col in ["early_pos", "mid_move", "late_move", "finishing_effort",
                        "led_at_furlong"]:
                df[col] = np.nan
            for col in ["was_keen", "had_trouble"]:
                df[col] = 0
        else:
            parsed = df[comment_col].apply(parse_run_style).apply(pd.Series)
            for col in parsed.columns:
                df[col] = parsed[col]

        # Composite: total position movement
        df["total_move"] = df["mid_move"] + df["late_move"]

        # Normalised early position (0–1 scale, field-adjusted)
        nr = pd.to_numeric(df["number_of_runners"], errors="coerce").replace(0, np.nan)
        df["early_pos_pct"] = (df["early_pos"] - 1) / 5.0  # 0=back, 1=front

        # §6.1 epf_norm: (early_position − 1) / (N − 1), 0 = led, 1 = last.
        # early_pos_pct above is the style score rescaled and ignores the field;
        # this is the field-size-normalised quantity the framework asks for.
        df["epf_norm"] = epf_norm_from_style(df["early_pos"], nr)
        df["early_pos_est"] = 1 + df["epf_norm"] * (nr - 1)
        df["run_style"] = run_style_from_epf_norm(df["epf_norm"])

        # pos_gain = ground made up between the early position and the finish,
        # both on the 0–1 scale. Positive = passed horses.
        place = pd.to_numeric(df["placing_numerical"], errors="coerce") \
            if "placing_numerical" in df.columns else pd.Series(np.nan, index=df.index)
        df["finish_pos_norm"] = (place - 1) / (nr - 1)
        df["pos_gain"] = df["epf_norm"] - df["finish_pos_norm"]

        return df

    # ------------------------------------------------------------------
    # Step 2: Horse running style profiles
    # ------------------------------------------------------------------
    def _calc_horse_style_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical running style per horse — lag-safe expanding means."""
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Career average early position (lag-safe)
        df["horse_career_early_pos"] = grp["early_pos"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career average late movement
        df["horse_career_late_move"] = grp["late_move"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career average total movement
        df["horse_career_total_move"] = grp["total_move"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career keenness rate
        df["horse_keen_rate"] = grp["was_keen"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        # Career trouble rate
        df["horse_trouble_rate"] = grp["had_trouble"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Running style consistency (std of early_pos — is the horse a fixed style?)
        df["horse_style_consistency"] = grp["early_pos"].apply(
            lambda x: x.shift(1).expanding().std()
        )

        # Last-run detailed metrics
        for i in range(1, 4):
            sfx = "" if i == 1 else str(i)
            df[f"LR{sfx}_early_pos"] = grp["early_pos"].shift(i)
            df[f"LR{sfx}_late_move"] = grp["late_move"].shift(i)
            df[f"LR{sfx}_total_move"] = grp["total_move"].shift(i)
            df[f"LR{sfx}_was_keen"] = grp["was_keen"].shift(i)

        # 3-run and 5-run rolling averages
        df["horse_early_pos_3r"] = grp["early_pos"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["horse_late_move_3r"] = grp["late_move"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["horse_early_pos_5r"] = grp["early_pos"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df["horse_late_move_5r"] = grp["late_move"].apply(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )

        # Style trend: is horse being ridden differently recently vs career?
        df["style_shift"] = df["horse_early_pos_3r"] - df["horse_career_early_pos"]

        return df

    # ------------------------------------------------------------------
    # Step 2b: §6.1 run-style profile — mean epf_norm, SD, modal style
    # ------------------------------------------------------------------
    def _calc_run_style_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """The horse's run-style profile on the field-size-normalised scale.

        §6.1: "Aggregate across the career with recency weighting to get the
        horse's run style profile: mean `epf_norm`, its SD (versatility), and
        the modal run style. A horse with low SD is tactically inflexible."

        Grouping on ``horse_name`` is the one key a row-wise ``shift(1)`` is
        safe on, because a horse runs once in a race.
        """
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)
        horse = df["horse_name"]

        _, df["horse_epf_norm_mean"], df["horse_epf_norm_sd"] = prior_mean_sd(
            df, "horse_name", "epf_norm")
        _, df["horse_pos_gain_mean"], _ = prior_mean_sd(df, "horse_name", "pos_gain")
        df["horse_epf_norm_recent"] = (
            grp["epf_norm"].shift(1).groupby(horse, sort=False)
            .ewm(halflife=STYLE_RECENCY_HALFLIFE_RUNS).mean()
            .droplevel(0).sort_index()
        )
        df["LR_epf_norm"] = grp["epf_norm"].shift(1)

        # Modal style: the career rate of each of the five §6.1 labels, then the
        # argmax. Five prior means beat an expanding mode and give the share of
        # the career spent in that style for free.
        labelled = df["run_style"].notna().astype(float)
        n_prior = labelled.groupby(horse, sort=False).cumsum() - labelled
        rates = {}
        for code in range(len(RUN_STYLE_LABELS)):
            flag = (df["run_style"] == code).astype(float) * labelled
            prior = flag.groupby(horse, sort=False).cumsum() - flag
            rates[f"_style_rate_{code}"] = prior / n_prior.where(n_prior > 0)
        rates = pd.DataFrame(rates, index=df.index)
        any_rate = rates.notna().any(axis=1)
        # fill before argmax: np.argmax picks a NaN over a real rate, and a horse
        # with no prior runs has no modal style at all.
        filled = rates.fillna(-1.0).values
        df["horse_modal_style"] = np.where(any_rate, filled.argmax(axis=1), np.nan)
        df["horse_style_mode_share"] = rates.max(axis=1).where(any_rate)
        # Tactical inflexibility: one style, always, and a low SD around it.
        df["horse_style_inflexible"] = df["horse_style_mode_share"] * (
            1.0 - df["horse_epf_norm_sd"].clip(0, 0.5) / 0.5
        )

        return df

    # ------------------------------------------------------------------
    # Step 3: Jockey & trainer running style profiles
    # ------------------------------------------------------------------
    def _calc_entity_style_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Jockey and trainer pace tendencies — lag-safe."""
        for entity_col, prefix in [
            ("jockey_name", "jockey"),
            ("trainer", "trainer"),
        ]:
            df = df.sort_values(
                [entity_col, "race_date", "race_time"]
            ).reset_index(drop=True)
            egrp = df.groupby(entity_col, group_keys=False)

            # Career average early position
            # Lagged by race: a trainer often saddles two runners in one race, so
            # a row-wise shift would show one of them the other's running style.
            df["_front"] = (df["early_pos"] > 4).astype(float)
            df["_holdup"] = (df["early_pos"] < 2).astype(float)
            df[f"{prefix}_career_early_pos"] = race_lagged_expanding_mean(df, entity_col, "early_pos")
            df[f"{prefix}_career_late_move"] = race_lagged_expanding_mean(df, entity_col, "late_move")
            df[f"{prefix}_front_rate"] = race_lagged_expanding_mean(df, entity_col, "_front")
            df[f"{prefix}_holdup_rate"] = race_lagged_expanding_mean(df, entity_col, "_holdup")
            df[f"{prefix}_keen_rate"] = race_lagged_expanding_mean(df, entity_col, "was_keen")
            df = df.drop(columns=["_front", "_holdup"])

        return df

    # ------------------------------------------------------------------
    # Step 4: Track/distance pace bias
    # ------------------------------------------------------------------
    def _calc_track_distance_pace_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Course-specific front-runner advantage — lag-safe expanding means.

        For each track+distance bucket, compute:
        - front_win_share: % of winners that were front-runners
        - holdup_win_share: % of winners that were hold-up
        - avg_winning_early_pos: average early position of winners
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Create track-distance key (round distance to nearest furlong)
        df["_td_key"] = df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)

        # Binary: was this a front-runner win?
        df["_front_won"] = ((df["early_pos"] >= 5.0) & (df["placing_numerical"] == 1)).astype(float)
        df["_back_won"] = ((df["early_pos"] <= 2.0) & (df["placing_numerical"] == 1)).astype(float)
        df["_winner_early_pos"] = np.where(
            df["placing_numerical"] == 1, df["early_pos"], np.nan
        )

        # Expanding mean per track-distance, lagged by RACE not by row: the
        # runners of one race share the key, so a row-wise shift(1) would let a
        # runner see its own race's result. See model/lagsafe.py.
        df["td_front_win_share"] = race_lagged_expanding_mean(df, "_td_key", "_front_won")
        df["td_holdup_win_share"] = race_lagged_expanding_mean(df, "_td_key", "_back_won")

        # Average winning early position at this track+distance
        # Only compute over winners
        df["td_avg_winner_pos"] = race_lagged_expanding_mean(df, "_td_key", "_winner_early_pos")

        # Track-level (all distances) front bias
        df["track_front_win_share"] = race_lagged_expanding_mean(df, "track", "_front_won")

        # Cleanup
        df.drop(columns=["_td_key", "_front_won", "_back_won", "_winner_early_pos"],
                inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 5: Predicted pace scenario (race-level)
    # ------------------------------------------------------------------
    def _calc_predicted_pace_scenario(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the pace scenario of a race from the assembled field.

        Uses each horse's historical running style to predict how fast
        the early pace will be. Key insight: pace is predictable because
        horses (and their connections) tend to repeat their preferred style.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Predicted early position for this horse (career avg as best predictor)
        df["pred_early_pos"] = df["horse_career_early_pos"].fillna(3.0)

        # Race-level pace prediction: average predicted early position of field
        df["pred_race_pace"] = df.groupby("raceid")["pred_early_pos"].transform("mean")

        # Predicted number of front-runners in race (EPF > 4)
        df["pred_n_front"] = df.groupby("raceid")["pred_early_pos"].transform(
            lambda x: (x > 4).sum()
        )
        # Predicted % of field that are front-runners
        nr = df["number_of_runners"].replace(0, np.nan)
        df["pred_front_pct"] = df["pred_n_front"] / nr * 100

        # Predicted number of hold-up runners
        df["pred_n_back"] = df.groupby("raceid")["pred_early_pos"].transform(
            lambda x: (x < 2.5).sum()
        )
        df["pred_back_pct"] = df["pred_n_back"] / nr * 100

        # Pace scenario classification (numeric for model)
        # Higher = faster predicted pace
        df["pred_pace_scenario"] = pd.cut(
            df["pred_front_pct"],
            bins=[-1, 15, 30, 50, 101],
            labels=[1, 2, 3, 4],
        ).astype(float)

        # Pace variance: std of predicted early positions in race
        # High variance = mixed pace, low = field all same style
        df["pred_pace_spread"] = df.groupby("raceid")["pred_early_pos"].transform("std")

        # Max predicted front-runner in race (the 'pace setter')
        df["pred_max_front"] = df.groupby("raceid")["pred_early_pos"].transform("max")

        # --- §6.2 step 1, on the field-size-normalised scale -----------------
        # Predicted epf_norm: the horse's recency-weighted career figure, nudged
        # toward the jockey's own tendency. Both are lag-safe; a horse with no
        # history falls back on the jockey, then on the middle of the field.
        rk = ensure_race_key(df)
        nr = pd.to_numeric(df["number_of_runners"], errors="coerce").replace(0, np.nan)
        horse = df["horse_epf_norm_recent"].fillna(df["horse_epf_norm_mean"])
        jockey = pd.Series(
            epf_norm_from_style(df.get("jockey_career_early_pos", pd.Series(np.nan, index=df.index)), nr),
            index=df.index,
        )
        blend = (1 - JOCKEY_STYLE_WEIGHT) * horse + JOCKEY_STYLE_WEIGHT * jockey.fillna(horse)
        pred = blend.where(horse.notna(), jockey)
        pred = pred.fillna(pred.groupby(rk).transform("mean")).fillna(0.5)
        df["pred_epf_norm"] = pred.clip(0, 1)

        # predicted_lead_prob: softmax over predicted early position within the
        # race, so it sums to 1 over the field — a probability of leading, not a
        # count of horses that might.
        z = -df["pred_epf_norm"] / LEAD_SOFTMAX_TEMPERATURE
        z = z - z.groupby(rk).transform("max")          # overflow guard only
        e = np.exp(z)
        df["predicted_lead_prob"] = e / e.groupby(rk).transform("sum")
        df["lead_prob_rank"] = df["predicted_lead_prob"].groupby(rk).rank(
            ascending=False, method="min"
        )
        df["lead_prob_top"] = df["predicted_lead_prob"].groupby(rk).transform("max")
        # Entropy of the lead-probability vector, 0 = one certain leader,
        # 1 = nobody has a claim: the cleanest read on a contested lead.
        p = df["predicted_lead_prob"].clip(lower=1e-12)
        ent = (-p * np.log(p)).groupby(rk).transform("sum")
        df["lead_prob_entropy"] = ent / np.log(nr.where(nr > 1))

        # pace_pressure / pace_contested / race_pace_class (§6.2), all from the
        # *predicted* epf_norm and therefore legal before the off.
        forward = (df["pred_epf_norm"] < PACE_PRESSURE_THRESHOLD).astype(float)
        df["pred_pace_pressure"] = forward.groupby(rk).transform("sum")
        df["pred_pace_share"] = df["pred_pace_pressure"] / nr
        df["pred_pace_contested"] = df["pred_epf_norm"].groupby(rk).transform("std")
        share = df["pred_pace_share"]
        df["pred_race_pace_class"] = np.select(
            [share.isna(), share >= PACE_CLASS_EDGES[1], share <= PACE_CLASS_EDGES[0]],
            [np.nan, 2.0, 0.0],
            default=1.0,
        )

        return df

    # ------------------------------------------------------------------
    # Step 6: Pace fit — how well does style match conditions?
    # ------------------------------------------------------------------
    def _calc_pace_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction features: horse style x race pace x track bias.

        Key features:
        - pace_advantage: Does this horse's style suit the predicted pace?
        - track_style_fit: Does this horse's style suit this track?
        - pace_mismatch: Is the horse out of its comfort zone?
        """
        # Pace advantage: front-runners benefit more in slow-pace races
        # Hold-up horses benefit more in fast-pace races
        # Quantify: style_pos * (1 / predicted_pace_intensity)
        pred_ep = df["pred_early_pos"].fillna(3.0)
        pred_pace = df["pred_race_pace"].fillna(3.0)

        # Relative position: how far ahead/behind the average pace
        df["pace_position_delta"] = pred_ep - pred_pace

        # Front-runner in slow pace = big advantage
        # Hold-up in fast pace = big advantage
        # Capture as interaction:
        df["pace_advantage"] = np.where(
            pred_ep >= 4.0,
            # Front-runner: advantage when pace is slow (few other fronts)
            (5 - pred_pace),
            np.where(
                pred_ep <= 2.5,
                # Hold-up: advantage when pace is fast (many fronts to tire)
                (pred_pace - 2.5),
                0  # Mid-field: neutral
            )
        )

        # Track style fit: horse's style matches track's winning profile
        td_front = df["td_front_win_share"].fillna(0.5)
        df["track_style_fit"] = np.where(
            pred_ep >= 4.0,
            td_front,  # Front-runner at front-runner-friendly track
            np.where(
                pred_ep <= 2.5,
                df["td_holdup_win_share"].fillna(0.2),  # Hold-up at hold-up track
                0.3  # Neutral
            )
        )

        # Pace mismatch: difference between horse's typical style and
        # what's optimal at this track+distance
        td_winner_pos = df["td_avg_winner_pos"].fillna(3.5)
        df["pace_mismatch"] = np.abs(pred_ep - td_winner_pos)

        # Competition for lead: if horse wants to lead but many others do too
        df["lead_competition"] = np.where(
            pred_ep >= 4.5,
            df["pred_n_front"].fillna(1) - 1,  # -1 to exclude self
            0
        )

        # Lone front-runner indicator (strong advantage historically)
        df["lone_front_runner"] = (
            (pred_ep >= 4.5) & (df["pred_n_front"] <= 1)
        ).astype(float)

        # --- §6.2 step 2: pace_suit = f(style, predicted shape, course bias) --
        # "This interaction is where pace data earns its keep": being a
        # front-runner is priced by the market, being the only front-runner in a
        # race at a course where the front-runner wins is not.
        nr = pd.to_numeric(df["number_of_runners"], errors="coerce").replace(0, np.nan)
        forwardness = 1 - 2 * df["pred_epf_norm"]            # +1 lone leader .. −1 backmarker
        lead_scarcity = 1 - 2 * df["pred_pace_share"]        # +1 nobody else goes forward
        # Course bias on the same scale: where do winners here race early?
        course_winner_pos = pd.Series(
            epf_norm_from_style(df.get("td_avg_winner_pos", pd.Series(np.nan, index=df.index)), nr),
            index=df.index,
        )
        df["course_front_edge"] = 0.5 - course_winner_pos     # >0: winners race forward here
        df["pace_suit"] = forwardness * (
            0.5 * lead_scarcity.fillna(0) + df["course_front_edge"].fillna(0)
        )
        # The same idea against the predicted *leader* rather than the count.
        df["lead_prob_edge"] = df["predicted_lead_prob"] - (1.0 / nr)

        return df

    # ------------------------------------------------------------------
    # Step 7: Tactical / behavioural features
    # ------------------------------------------------------------------
    def _calc_tactical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keenness, trouble patterns, and their predictive value."""
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Keenness rate over last 3 runs
        df["keen_rate_3r"] = grp["was_keen"].apply(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        # Was keen last run AND front-runner style -> at risk of burning out
        lr_keen = grp["was_keen"].shift(1).fillna(0)
        lr_ep = grp["early_pos"].shift(1).fillna(3)
        df["keen_front_risk"] = (lr_keen * (lr_ep >= 4)).astype(float)

        # Trouble rate (career)
        # Already computed in horse profiles, just reference
        # But add: trouble last run -> may have been unlucky
        df["LR_had_trouble"] = grp["had_trouble"].shift(1).fillna(0)

        # Jockey style match: does jockey typically ride this horse's style?
        # Difference between jockey's preferred style and horse's style
        df["jockey_horse_style_gap"] = np.abs(
            df["jockey_career_early_pos"].fillna(3) - df["horse_career_early_pos"].fillna(3)
        )

        # Trainer style match with jockey
        df["trainer_jockey_style_gap"] = np.abs(
            df["trainer_career_early_pos"].fillna(3) - df["jockey_career_early_pos"].fillna(3)
        )

        return df

    # ------------------------------------------------------------------
    # Step 8: Position sustainability
    # ------------------------------------------------------------------
    def _calc_position_sustainability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Can this horse sustain its early position? Or does it fade?

        Key concept: horses that lead early but fade (negative late_move)
        are unsustainable front-runners. Track this pattern.
        """
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        # Front-runner sustainability: for horses that run prominently,
        # what's their average late movement?
        # Computed as: career avg(late_move) only when early_pos >= 4
        shifted_ep = grp["early_pos"].shift(1)
        shifted_lm = grp["late_move"].shift(1)

        front_mask = shifted_ep >= 4
        front_lm = shifted_lm.where(front_mask)
        df["front_sustainability"] = front_lm.groupby(
            df["horse_name"]
        ).expanding().mean().droplevel(0).sort_index()

        # Hold-up finishing ability: for horses that run from behind,
        # how strong is their finish?
        back_mask = shifted_ep <= 2.5
        back_lm = shifted_lm.where(back_mask)
        df["holdup_finish_ability"] = back_lm.groupby(
            df["horse_name"]
        ).expanding().mean().droplevel(0).sort_index()

        # Position drop: how much does this horse typically lose from early to finish?
        # Approximated by: (early_pos_rank - NFP) trend
        # Higher = loses more positions, lower = sustains/improves
        if "NFP" in df.columns:
            df["_pos_drop_raw"] = df["early_pos_pct"] - df["NFP"]
            df["horse_pos_drop"] = grp["_pos_drop_raw"].apply(
                lambda x: x.shift(1).expanding().mean()
            )
            df.drop(columns=["_pos_drop_raw"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 8b: §6.4 Position Change Statistic
    # ------------------------------------------------------------------
    def _calc_position_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Positional churn per race, plus the course/distance baseline.

        ``pcs_race`` describes the race being predicted (it needs that race's
        finishing positions) and is a screen, not a feature. The feature is
        ``td_pcs_mean``: how much a course and distance churns, measured over
        *earlier* races only. A low value says early position tends to hold,
        which is exactly when a front-runner's draw and style matter most.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
        rk = ensure_race_key(df)

        pcs = position_change_statistic(df)
        if pcs.empty:
            df["pcs_race"] = np.nan
            df["pcs_race_norm"] = np.nan
        else:
            m = pcs.set_index("raceid")
            df["pcs_race"] = rk.map(m["pcs"])
            df["pcs_race_norm"] = rk.map(m["pcs_norm"])

        df["_td_key"] = df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)
        # Race-lagged: every runner shares the track+distance key, so a row-wise
        # shift would hand a runner its own race's churn. See model/lagsafe.py.
        df["td_pcs_mean"] = race_lagged_expanding_mean(df, "_td_key", "pcs_race_norm")
        df["track_pcs_mean"] = race_lagged_expanding_mean(df, "track", "pcs_race_norm")
        df["pcs_vs_baseline"] = df["pcs_race_norm"] - df["td_pcs_mean"]
        df.drop(columns=["_td_key"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 9: Within-race pace ranks
    # ------------------------------------------------------------------
    def _calc_pace_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank horses within race on pace-related features."""
        df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

        rank_cols = {
            "rPredEarlyPos": "pred_early_pos",
            "rHorseCareerEarlyPos": "horse_career_early_pos",
            "rHorseLateMoveCareer": "horse_career_late_move",
            "rPaceAdvantage": "pace_advantage",
            "rTrackStyleFit": "track_style_fit",
            "rFrontSustain": "front_sustainability",
            "rHoldupFinish": "holdup_finish_ability",
            "rStyleConsistency": "horse_style_consistency",
            "rLeadProb": "predicted_lead_prob",
            "rPaceSuit": "pace_suit",
            "rPosGain": "horse_pos_gain_mean",
        }

        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df.groupby("raceid")[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )

        return df


# ---------------------------------------------------------------------------
# Feature list for train_bfsp.py integration
# ---------------------------------------------------------------------------

# Horse running style profile features
HORSE_STYLE_FEATURES = [
    "horse_career_early_pos",
    "horse_career_late_move",
    "horse_career_total_move",
    "horse_keen_rate",
    "horse_trouble_rate",
    "horse_style_consistency",
    "LR_early_pos",
    "LR_late_move",
    "LR_total_move",
    "LR_was_keen",
    "LR2_early_pos",
    "LR2_late_move",
    "LR2_total_move",
    "LR3_early_pos",
    "LR3_late_move",
    "LR3_total_move",
    "horse_early_pos_3r",
    "horse_late_move_3r",
    "horse_early_pos_5r",
    "horse_late_move_5r",
    "style_shift",
]

# §6.1 run-style profile on the field-size-normalised scale
RUN_STYLE_PROFILE_FEATURES = [
    "horse_epf_norm_mean",
    "horse_epf_norm_recent",
    "horse_epf_norm_sd",
    "horse_pos_gain_mean",
    "horse_modal_style",
    "horse_style_mode_share",
    "horse_style_inflexible",
    "LR_epf_norm",
]

# Entity style features
ENTITY_STYLE_FEATURES = [
    "jockey_career_early_pos",
    "jockey_career_late_move",
    "jockey_front_rate",
    "jockey_holdup_rate",
    "jockey_keen_rate",
    "trainer_career_early_pos",
    "trainer_career_late_move",
    "trainer_front_rate",
    "trainer_holdup_rate",
    "trainer_keen_rate",
]

# Track/distance pace bias features
TRACK_PACE_BIAS_FEATURES = [
    "td_front_win_share",
    "td_holdup_win_share",
    "td_avg_winner_pos",
    "track_front_win_share",
]

# Predicted pace scenario features
PACE_SCENARIO_FEATURES = [
    "pred_early_pos",
    "pred_race_pace",
    "pred_n_front",
    "pred_front_pct",
    "pred_n_back",
    "pred_back_pct",
    "pred_pace_scenario",
    "pred_pace_spread",
    "pred_max_front",
    # §6.2 on the normalised scale
    "pred_epf_norm",
    "predicted_lead_prob",
    "lead_prob_rank",
    "lead_prob_top",
    "lead_prob_entropy",
    "pred_pace_pressure",
    "pred_pace_share",
    "pred_pace_contested",
    "pred_race_pace_class",
]

# Pace fit / interaction features
PACE_FIT_FEATURES = [
    "pace_position_delta",
    "pace_advantage",
    "track_style_fit",
    "pace_mismatch",
    "lead_competition",
    "lone_front_runner",
    "pace_suit",
    "course_front_edge",
    "lead_prob_edge",
]

# Tactical features
TACTICAL_FEATURES = [
    "keen_rate_3r",
    "keen_front_risk",
    "LR_had_trouble",
    "jockey_horse_style_gap",
    "trainer_jockey_style_gap",
]

# Position sustainability features
SUSTAINABILITY_FEATURES = [
    "front_sustainability",
    "holdup_finish_ability",
    "horse_pos_drop",
]

# §6.4 position churn — course/distance baselines (the per-race PCS itself is a
# screen over completed races, not a feature)
POSITION_CHURN_FEATURES = [
    "td_pcs_mean",
    "track_pcs_mean",
]

# Within-race pace rankings
PACE_RANK_FEATURES = [
    "rPredEarlyPos",
    "rHorseCareerEarlyPos",
    "rHorseLateMoveCareer",
    "rPaceAdvantage",
    "rTrackStyleFit",
    "rFrontSustain",
    "rHoldupFinish",
    "rStyleConsistency",
    "rLeadProb",
    "rPaceSuit",
    "rPosGain",
]

# Combined list for easy import
ALL_PACE_FEATURES = (
    HORSE_STYLE_FEATURES
    + RUN_STYLE_PROFILE_FEATURES
    + ENTITY_STYLE_FEATURES
    + TRACK_PACE_BIAS_FEATURES
    + PACE_SCENARIO_FEATURES
    + PACE_FIT_FEATURES
    + TACTICAL_FEATURES
    + SUSTAINABILITY_FEATURES
    + POSITION_CHURN_FEATURES
    + PACE_RANK_FEATURES
)

assert not (set(ALL_PACE_FEATURES) & PACE_POST_RACE_ONLY), \
    "a column parsed from the race being predicted reached the model feature list"
