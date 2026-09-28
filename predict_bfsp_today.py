#!/usr/bin/env python3
"""
Generate BFSP Predictions for Today's Races.

Fetches today's race cards from horseracebase.com, calculates all custom
metrics on the historical database, builds feature vectors for declared
runners, and predicts BFSP using the trained regression model.

Usage:
    python predict_bfsp_today.py                         # Predict today
    python predict_bfsp_today.py --date 2026-03-10       # Specific date
    python predict_bfsp_today.py --dry-run               # Skip email
    python predict_bfsp_today.py --from-db               # Use DB data (no HRB login)

Environment variables (in .env):
    HRB_USERNAME        - horseracebase.com login
    HRB_PASSWORD        - horseracebase.com password
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from model.bfsp_model import (
    assert_meta_is_servable,
    attach_serving_rule,
    predict_prices,
)
from model.bfsp_features import blocks_needed
from model.custom_metrics import CustomMetricsEngine
from train_bfsp import (
    ALL_FEATURE_COLS,
    BFSPTrainer,
    build_context_features,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_historical(db_path: str, start_date: str | None = None) -> pd.DataFrame:
    """Load historical race results, optionally filtered by start date."""
    conn = sqlite3.connect(db_path)
    if start_date:
        df = pd.read_sql_query(
            "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
            conn,
            params=[start_date],
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM race_results ORDER BY race_date, race_time", conn
        )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


#: A manifest in the model directory serves several boosters as one: the geometric
#: mean of their prices, renormalised per race, as the research loop scores an
#: average (scripts/research_loop.average_arms). Without one, one booster serves.
ENSEMBLE_MANIFEST = "bfsp_ensemble.json"


def _check_gate(gate: dict, names: list[str]) -> dict:
    """A manifest's gate, refused unless it names a member, leaves at least one other to
    rank the runners, and gives every rank a weight between 0 and 1."""
    member, weights = gate.get("member"), gate.get("weights")
    if member not in names:
        raise ValueError(f"the gate names {member!r}, which is not a member ({', '.join(names)})")
    if len(names) < 2:
        raise ValueError("a gate needs at least one other member to rank the runners")
    if not isinstance(weights, list) or not weights:
        raise ValueError("a gate needs its weights by rank: a list, the last for every rank past it")
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)) or w.min() < 0 or w.max() > 1:
        raise ValueError(f"a gate's weights lie between 0 and 1: {weights}")
    return {"member": member, "weights": [float(x) for x in w]}


def _race_normalised_log(log_price: np.ndarray, race: np.ndarray) -> np.ndarray:
    """Log prices shifted so that each race's implied probabilities sum to one."""
    book = pd.Series(np.exp(-log_price)).groupby(race).transform("sum").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return log_price + np.log(book)


class AveragedBooster:
    """Several boosters served as one.

    Each member's output is turned into its raw price by its own fit's rule (its
    target and offset, as `attach_serving_rule` put them on it), and `predict`
    returns the mean of the log prices, under the target "log_bfsp". After
    `predict_prices` normalises each race's book to one, that is exactly the research
    loop's average of the members' normalised prices (the geometric mean,
    renormalised: scripts/research_loop.average_arms), because a member's own
    normalisation divides every runner in a race by the same number. A member reads
    its own features, by name, from the frame it is given.

    A gate (the manifest's "gate": {"member": ..., "weights": [w1, ..., wK]}) gives one
    member the runners the others rank near the top of their race instead of a share of
    every runner. Each member's price is normalised to a book of one in its race; the
    other members are averaged as above (the base); a runner the base ranks r-th in its
    race takes the gated member's log price with weight w_r (wK for every rank past K)
    and the base's with 1 - w_r. That is the blend the research queries scored
    (research/queries/done/gate_clv.py: race_xent for the pair's leaders). It needs each
    runner's race, which `predict_prices` passes to a booster that `reads_races`."""

    serving_target = "log_bfsp"
    serving_offset = 0.0                        # each member's own offset is applied in predict

    def __init__(self, members: list[tuple[str, "lgb.Booster", list[str]]], gate: dict | None = None):
        if len(members) < 2:
            raise ValueError("an averaged model needs at least two members")
        self.members = members
        self._features = list(dict.fromkeys(c for _, _, cols in members for c in cols))
        self.gate = _check_gate(gate, [n for n, _, _ in members]) if gate else None

    @property
    def reads_races(self) -> bool:
        return self.gate is not None

    def _member_logs(self, X: pd.DataFrame, num_iteration=None) -> list[np.ndarray]:
        from model.bfsp_model import invert_target
        logs = []
        for _, b, cols in self.members:
            out = (np.asarray(b.predict(X[cols], num_iteration=num_iteration), dtype=float)
                   + float(getattr(b, "serving_offset", 0.0) or 0.0))
            price = invert_target(out, X, getattr(b, "serving_target", None) or "log_bfsp")
            with np.errstate(divide="ignore", invalid="ignore"):
                logs.append(np.log(price))
        return logs

    def predict(self, X: pd.DataFrame, num_iteration=None, races=None) -> np.ndarray:
        logs = self._member_logs(X, num_iteration)
        if self.gate is None:
            return np.mean(logs, axis=0)
        if races is None:
            raise ValueError("a gated average ranks the runners of each race: it needs their races "
                             "(model.bfsp_model.predict_prices passes them)")
        race = pd.factorize(np.asarray(races))[0]
        names = [n for n, _, _ in self.members]
        g = names.index(self.gate["member"])
        norm = [_race_normalised_log(lp, race) for lp in logs]
        base = _race_normalised_log(np.mean([lp for i, lp in enumerate(norm) if i != g], axis=0), race)
        weights = np.asarray(self.gate["weights"], dtype=float)
        rank = pd.Series(base).groupby(race).rank(method="first").fillna(len(weights)).to_numpy()
        w = weights[np.minimum(rank, len(weights)).astype(int) - 1]
        return w * norm[g] + (1 - w) * base

    def feature_name(self) -> list[str]:
        return list(self._features)

    def num_feature(self) -> int:
        return len(self._features)

    def num_trees(self) -> int:
        return sum(b.num_trees() for _, b, _ in self.members)


def booster_path(model_dir: str) -> str:
    """The booster file a model directory holds: `bfsp_model.lgb`, or `bfsp_model.lgb.xz` where
    only the compressed file is there. A booster past GitHub's 100 MB file limit (10,000 rounds
    of 127 leaves is 123 MB) can only be committed compressed. With both present the plain file
    is read and the other is stale: `tests/test_averaged_serving.py` refuses that in the repo."""
    plain = os.path.join(model_dir, "bfsp_model.lgb")
    if os.path.exists(plain):
        if os.path.exists(plain + ".xz"):
            log.warning("%s holds both bfsp_model.lgb and bfsp_model.lgb.xz; reading the plain file", model_dir)
        return plain
    return plain + ".xz" if os.path.exists(plain + ".xz") else plain


def _read_booster(path: str) -> lgb.Booster:
    """A booster from its file, plain or xz-compressed (`bfsp_model.lgb.xz`)."""
    if path.endswith(".xz"):
        import lzma
        with lzma.open(path, "rt") as f:
            return lgb.Booster(model_str=f.read())
    return lgb.Booster(model_file=path)


def _objective(booster: lgb.Booster, meta: dict) -> str:
    """The loss the booster was fitted with, from its own parameters. The metadata's
    "objective" is the training config's label ("l2" for any loss set through --param
    objective=...), so a Huber model would be logged as l2."""
    params = getattr(booster, "params", None) or {}
    return str(params.get("objective") or meta.get("objective", "?"))


def _load_member(member_dir: str) -> tuple[lgb.Booster, list[str], dict, dict]:
    """One member of an averaged model: its booster, features, vocabulary and metadata,
    held to the same checks as a single served model."""
    path = booster_path(member_dir)
    meta_path = os.path.join(member_dir, "bfsp_model_meta.json")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"{member_dir}: an averaged model's member needs bfsp_model.lgb (or .lgb.xz) "
                                f"and bfsp_model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    booster = _read_booster(path)
    cols = list(meta.get("feature_cols") or [])
    assert_meta_is_servable({**meta, "feature_cols": cols})
    if booster.num_feature() != len(cols):
        raise ValueError(f"{member_dir}: the booster reads {booster.num_feature()} features, "
                         f"its metadata lists {len(cols)}")
    attach_serving_rule(booster, meta)
    return booster, cols, meta.get("categorical_vocab", {}) or {}, meta


def load_averaged_model(model_dir: str, manifest: str) -> tuple[AveragedBooster, list[str], dict]:
    """The members a manifest names, served as one (`AveragedBooster`).

    The manifest is {"members": [{"name": ..., "dir": ...}, ...]}, each dir relative
    to the model directory ("." for the model directory itself, whose metadata then
    also gives the history's start), and optionally a "gate" (`AveragedBooster`). Every
    member must be servable on its own and read the same categorical vocabulary (the
    same training matrix numbers the tracks the same way). Each member's output is
    priced by its own target's rule."""
    with open(manifest) as f:
        spec = json.load(f)
    members, vocab = [], None
    for m in spec.get("members", []):
        name, sub = m.get("name") or m.get("dir"), m.get("dir", ".")
        booster, cols, mvocab, meta = _load_member(os.path.normpath(os.path.join(model_dir, sub)))
        if vocab is None:
            vocab = mvocab
        elif mvocab != vocab:
            raise ValueError(f"member {name} numbers its categories differently from the first member: "
                             f"its vocabulary would price a different track")
        members.append((name, booster, cols))
        log.info("Averaged model member %s: %d features, objective=%s, trained through %s", name, len(cols),
                 _objective(booster, meta), meta.get("trained_through", "?"))
    model = AveragedBooster(members, gate=spec.get("gate"))
    log.info("Loaded an averaged BFSP model: %d members, %d features in all", len(members), model.num_feature())
    if model.gate:
        log.info("Gated: member %s takes weights %s by the other members' rank in the race",
                 model.gate["member"], model.gate["weights"])
    return model, model.feature_name(), vocab or {}


def load_bfsp_model(model_dir: str) -> tuple[lgb.Booster, list[str], dict]:
    """Load the trained BFSP model, its feature columns and its categorical vocabulary.

    The vocabulary has to travel with the model: `cat.codes` numbers whatever
    categories are present in the frame it is given, so a track that is one
    integer across the training history is a different integer on a six-race
    card unless the levels are pinned. It is the third return value, and
    `build_context_features` takes it.

    A manifest (`ENSEMBLE_MANIFEST`) in the directory serves its members as one
    (`load_averaged_model`); without one, the directory's single booster serves."""
    manifest = os.path.join(model_dir, ENSEMBLE_MANIFEST)
    if os.path.exists(manifest):
        return load_averaged_model(model_dir, manifest)
    model_path = booster_path(model_dir)
    meta_path = os.path.join(model_dir, "bfsp_model_meta.json")

    if not os.path.exists(model_path):
        log.error(f"BFSP model not found at {model_path}")
        log.error("Run 'python train_bfsp.py' first to train the model.")
        sys.exit(1)

    model = _read_booster(model_path)

    feature_cols, vocab, meta = [], {}, {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols", [])
        vocab = meta.get("categorical_vocab", {}) or {}

    if not feature_cols:
        feature_cols = [c for c in ALL_FEATURE_COLS]

    # Refuse an artefact that should not be served. The model this guard was
    # written for had NFP_residual, FSALB and win_surprise -- the race's own
    # result -- as its three most important features by gain, and this loader
    # served it every morning without a word.
    assert_meta_is_servable({**meta, "feature_cols": feature_cols})

    # The booster does not remember what it was fitted on -- a save/load round
    # trip drops it -- so the rule for turning its output back into a price is
    # pinned here, at the one point every caller of this loader goes through.
    attach_serving_rule(model, meta)

    log.info("Loaded BFSP model (%d features, objective=%s, target=%s, trained through %s)",
             len(feature_cols), _objective(model, meta),
             meta.get("target", "?"), meta.get("trained_through", "?"))
    if not vocab:
        log.warning(
            "No categorical vocabulary in the model metadata: track_cat and "
            "race_type_cat will be numbered from today's card, which is not how "
            "they were numbered in training. Retrain to store one."
        )
    return model, feature_cols, vocab


# ---------------------------------------------------------------------------
# Race Card Fetching
# ---------------------------------------------------------------------------

def fetch_racecard_from_hrb(target_date: date) -> pd.DataFrame:
    """Fetch today's race card from horseracebase.com."""
    try:
        from daily_predictions import (
            create_session,
            get_todays_runners,
            login,
        )
    except ImportError:
        log.error("Could not import daily_predictions module")
        return pd.DataFrame()

    session = create_session()
    user_id = login(session)
    if not user_id:
        log.error("Failed to login to horseracebase.com")
        log.error("Set HRB_USERNAME and HRB_PASSWORD in .env file")
        return pd.DataFrame()

    runners = get_todays_runners(session, user_id, target_date)
    return runners


def read_card(path: str, target_date: date) -> pd.DataFrame:
    """A card saved to a file: a racecard CSV (csv/racecards/) or the HTML card page, read as the
    06:00 fetch reads it (daily_predictions.scrape_racecard_html)."""
    if not path.lower().endswith((".html", ".htm")):
        return pd.read_csv(path, dtype={"race_time": str})
    from daily_predictions import scrape_racecard_html

    with open(path, encoding="utf-8") as f:
        html = f.read()

    class _Page:
        text = html

        def raise_for_status(self):
            pass

    class _Saved:
        def get(self, url):
            return _Page()

    card = scrape_racecard_html(_Saved(), target_date)
    return card if card is not None else pd.DataFrame(columns=["race_time", "track", "horse_name"])


def get_runners_from_db(db_path: str, target_date: str) -> pd.DataFrame:
    """Get runners for a specific date from the local database.

    Used when HRB credentials aren't available — simulates predictions
    on historical race cards stored in the database.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results WHERE race_date = ? ORDER BY race_time",
        conn,
        params=[target_date],
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------

def prepare_and_predict(
    historical_df: pd.DataFrame,
    target_runners: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    target_date: date,
    vocab: dict | None = None,
) -> pd.DataFrame:
    """Calculate metrics on history, build features for runners, predict BFSP.

    This is the core prediction function that:
    1. Concatenates history with target runners (if not already in history)
    2. Runs CustomMetricsEngine on the full dataset
    3. Extracts feature vectors for target date runners
    4. Predicts log(BFSP) and converts to BFSP
    """
    # The shape, intent, freshness and window blocks are minutes of work over the
    # whole history. Each is built only for a model that reads it -- the same
    # features either way for the one that does. A drop-in block the model reads
    # (model/blocks) switches on the engine blocks it is computed from.
    from model import blocks as drop_in
    served_blocks = drop_in.used_by(feature_cols)
    engine = CustomMetricsEngine(**{**blocks_needed(feature_cols), **drop_in.engine_flags(served_blocks)})

    # Check if target runners are already in the historical data
    target_date_str = str(target_date)
    runners_in_db = (
        historical_df["race_date"].dt.strftime("%Y-%m-%d") == target_date_str
    ).any()

    if runners_in_db:
        # Runners are in the database — use the full dataset
        log.info("Target date found in database, using full historical data")
        full_df = historical_df.copy()
    else:
        # Append target runners to history for metric calculation
        log.info("Appending target runners to historical data")
        target_runners = target_runners.copy()
        target_runners["race_date"] = pd.to_datetime(target_date)

        # The morning card lacks fields the model is trained with -- distance, race type,
        # surface, sex, sire, career runs, claims, the race's OR spread -- and a missing
        # category is scored as the first in its vocabulary (QA review C1). Fill them from
        # the card's own text and the horses', jockeys' and tracks' earlier rows.
        from model.card_enrich import enrich_card
        target_runners = enrich_card(target_runners, historical_df)

        # Ensure compatible schemas
        for col in historical_df.columns:
            if col not in target_runners.columns:
                target_runners[col] = np.nan
        for col in target_runners.columns:
            if col not in historical_df.columns:
                historical_df[col] = np.nan

        full_df = pd.concat(
            [historical_df, target_runners], ignore_index=True
        )

    # Calculate all custom metrics on the full dataset
    log.info("Calculating all custom metrics...")
    full_df = engine.calculate_all(full_df)

    # Build context features
    log.info("Building context features...")
    full_df = build_context_features(full_df, vocab=vocab)

    # Extract rows for the target date
    full_df["race_date"] = pd.to_datetime(full_df["race_date"])
    target_mask = full_df["race_date"].dt.strftime("%Y-%m-%d") == target_date_str

    # Drop-in blocks, built as training built them: on the runs the feature
    # matrix holds (those with a usable price) plus the card's.
    if served_blocks:
        log.info("Building drop-in blocks %s...", ", ".join(served_blocks))
        full_df, _ = drop_in.attach_as_trained(full_df, served_blocks, card=target_mask.to_numpy())

    target_df = full_df[target_mask].copy()

    if len(target_df) == 0:
        log.error(f"No runners found for {target_date}")
        return pd.DataFrame()

    log.info(f"Predicting BFSP for {len(target_df)} runners on {target_date}")

    # Ensure all features exist
    for col in feature_cols:
        if col not in target_df.columns:
            target_df[col] = np.nan

    # One price, one probability, race book = 1. See model/bfsp_model.py.
    #
    # This path used to look for a price calibrator in --model-dir while the
    # file lived in models/, so it served the normalised price and warned about
    # a bias the normalisation had already removed. The calibrator is out of
    # the serving path entirely now, in every script, so live, training and
    # evaluation quote the same number.
    target_df = predict_prices(model, target_df, feature_cols, race_col="raceid")
    target_df["predicted_bfsp_norm"] = target_df["predicted_bfsp"]

    # If actual BFSP is available, compute edge
    if "bfsp" in target_df.columns:
        actual_bfsp = pd.to_numeric(target_df["bfsp"], errors="coerce")
        target_df["actual_bfsp"] = actual_bfsp
        target_df["bfsp_diff"] = target_df["predicted_bfsp"] - actual_bfsp
        target_df["bfsp_diff_pct"] = (
            (target_df["predicted_bfsp"] - actual_bfsp) / actual_bfsp * 100
        )
        # Value detection: model thinks horse is shorter (more likely to win)
        # than the market suggests
        target_df["value_edge"] = (
            actual_bfsp / target_df["predicted_bfsp"] - 1
        ) * 100

    return target_df


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

#: The CSV --output-csv writes: the race and the runner as the card has them,
#: then the model's price, probability and rank in the race, then the result
#: when the day is in the database.
OUTPUT_COLS = [
    "race_date", "race_time", "track", "race_name", "race_type", "race_class", "dist_furlongs",
    "going_description", "number_of_runners", "horse_name", "jockey_name", "trainer", "stall",
    "official_rating", "pounds", "horse_age", "days_since_lr", "headgear", "odds",
    "predicted_bfsp", "predicted_win_prob_norm", "model_rank",
    "actual_bfsp", "bfsp_diff_pct", "value_edge", "placing_numerical",
]


def with_model_rank(predictions: pd.DataFrame) -> pd.DataFrame:
    """The model's rank of each runner in its race: 1 = its shortest price."""
    out = predictions.copy()
    race = out["raceid"] if "raceid" in out.columns else (
        out["race_date"].astype(str) + "|" + out["track"].astype(str) + "|" + out["race_time"].astype(str))
    out["model_rank"] = out.groupby(race)["predicted_bfsp"].rank(method="min").astype("Int64")
    return out


def _race_key(df: pd.DataFrame) -> pd.Series:
    """Course and off time as the card writes them (12-hour, 1:24 or 1.24), for matching one fetch to another."""
    def hhmm(v) -> str:
        s = str(v).strip().replace(".", ":")
        h, _, m = s.partition(":")
        return f"{int(h)}:{int(m):02d}" if h.isdigit() and m.isdigit() else s
    return df["track"].astype(str).str.strip().str.casefold() + "|" + df["race_time"].map(hhmm)


def _runner_key(df: pd.DataFrame) -> pd.Series:
    return _race_key(df) + "|" + df["horse_name"].astype(str).str.split().str.join(" ").str.casefold()


def without_non_runners(predictions: pd.DataFrame, card: pd.DataFrame):
    """The predictions with the runners since withdrawn taken out, each race's book back to 100%.

    A non-runner declared after the card was fetched stays in the predictions, and the race's
    other runners are priced too long by its share of the book. HRB's card drops a runner once
    it is withdrawn, so a later fetch names them: every predicted runner of a race the card still
    lists that the card no longer does. The race's remaining probabilities are rescaled to sum to
    one, the model's own normalisation over the smaller field, and its prices, ranks and field
    size follow; races that lost no one are left exactly as they were. This is the quick update:
    a full re-run also rebuilds what depends on the field (the within-race readings, the draw
    adjusted for non-runners, the gate's ranks).

    A race the card no longer lists at all is left as it was, not voided: an abandonment and a
    failed fetch look the same from here. Runners on the card the predictions never priced
    cannot be priced without a re-run and are only reported.

    Returns (predictions, non_runners, unpriced): the updated predictions; the runners taken
    out, with the share of the book each carried; the card's runners with no prediction.
    """
    if "race_date" in predictions.columns and predictions["race_date"].nunique() > 1:
        raise ValueError("predictions for one day at a time: the card is one day's")
    out = predictions.copy()
    if "predicted_win_prob_norm" not in out.columns:
        out["predicted_win_prob_norm"] = 1.0 / out["predicted_bfsp"]
    race, runner = _race_key(out), _runner_key(out)
    card_race, card_runner = _race_key(card), _runner_key(card)
    gone = (race.isin(set(card_race)) & ~runner.isin(set(card_runner))).to_numpy()
    ids = [c for c in ("race_date", "race_time", "track", "horse_name") if c in out.columns]
    non_runners = out.loc[gone, ids + ["predicted_bfsp", "predicted_win_prob_norm"]].rename(
        columns={"predicted_win_prob_norm": "share_of_book"}).reset_index(drop=True)
    new = (card_race.isin(set(race)) & ~card_runner.isin(set(runner))).to_numpy()
    unpriced = card.loc[new, [c for c in ("race_time", "track", "horse_name") if c in card.columns]]

    lost = set(race[gone])
    out, race = out[~gone], race[~gone]
    hit = race.isin(lost).to_numpy()
    if hit.any():
        r = race[hit]
        p = out.loc[hit, "predicted_win_prob_norm"]
        p = p / p.groupby(r).transform("sum")
        out.loc[hit, "predicted_win_prob_norm"] = p
        out.loc[hit, "predicted_bfsp"] = 1.0 / p
        if "number_of_runners" in out.columns:
            out.loc[hit, "number_of_runners"] = r.map(r.value_counts())
        if "model_rank" in out.columns:
            out.loc[hit, "model_rank"] = (1.0 / p).groupby(r).rank(method="min").astype(int)
    return out.reset_index(drop=True), non_runners, unpriced.reset_index(drop=True)


def update_for_non_runners(base_csv: str, output_csv: str, card: pd.DataFrame) -> pd.DataFrame:
    """Write base_csv's predictions without the runners the card no longer lists (without_non_runners),
    and beside them <output>_non_runners.csv, the runners taken out. Returns the non-runners."""
    base = pd.read_csv(base_csv, dtype={"race_time": str})
    card = card.copy()
    card["race_time"] = card["race_time"].astype(str)
    out, nr, unpriced = without_non_runners(base, card)
    out.to_csv(output_csv, index=False)
    nr_csv = os.path.splitext(output_csv)[0] + "_non_runners.csv"
    nr.to_csv(nr_csv, index=False)
    races = nr.groupby(["track", "race_time"]).ngroups if len(nr) else 0
    log.info(f"{len(nr)} non-runners since {base_csv} in {races} races, their races re-normalised: "
             f"{len(out)} runners written to {output_csv}, the non-runners to {nr_csv}")
    for r in nr.itertuples(index=False):
        log.info(f"  NR {r.race_time} {r.track}: {r.horse_name} ({r.share_of_book:.1%} of the book)")
    if len(unpriced):
        log.warning(f"{len(unpriced)} runners on the card have no prediction (a re-run prices them): "
                    + "; ".join(f"{r.race_time} {r.track} {r.horse_name}" for r in unpriced.itertuples(index=False)))
    return nr


def format_predictions(predictions: pd.DataFrame, target_date: date) -> str:
    """Format predictions into a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"  BFSP PREDICTIONS — {target_date.strftime('%A %d %B %Y')}"
    )
    lines.append("=" * 78)

    if len(predictions) == 0:
        lines.append("  No runners found.")
        return "\n".join(lines)

    n_races = predictions["raceid"].nunique() if "raceid" in predictions.columns else 0
    lines.append(
        f"  {len(predictions)} runners across {n_races} races"
    )
    lines.append("")

    group_col = "raceid" if "raceid" in predictions.columns else None
    has_actual = "actual_bfsp" in predictions.columns and predictions["actual_bfsp"].notna().any()

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            race_df = race_df.sort_values("predicted_bfsp")

            track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
            rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"
            n_run = len(race_df)

            lines.append(f"  {track} — {rtime} ({n_run} runners)")
            lines.append("  " + "-" * 74)

            if has_actual:
                lines.append(
                    f"  {'#':<3} {'Horse':<22} {'Pred BFSP':>10} "
                    f"{'Actual':>8} {'Diff%':>7} {'P(Win)':>7} "
                    f"{'Value':>7} {'Result':>7}"
                )
            else:
                lines.append(
                    f"  {'#':<3} {'Horse':<22} {'Pred BFSP':>10} "
                    f"{'P(Win)':>7} {'Rank':>5}"
                )
            lines.append("  " + "-" * 74)

            for rank, (_, row) in enumerate(race_df.iterrows(), 1):
                horse = str(row.get("horse_name", "?"))[:21]
                pred_bfsp = row.get("predicted_bfsp", 0)
                p_win = row.get("predicted_win_prob_norm", 0)

                if has_actual:
                    actual = row.get("actual_bfsp", np.nan)
                    diff_pct = row.get("bfsp_diff_pct", np.nan)
                    value = row.get("value_edge", np.nan)
                    placing = row.get("placing_numerical", np.nan)

                    actual_str = f"{actual:.1f}" if pd.notna(actual) else "-"
                    diff_str = f"{diff_pct:+.1f}%" if pd.notna(diff_pct) else "-"
                    value_str = f"{value:+.1f}%" if pd.notna(value) else "-"

                    if pd.notna(placing) and placing == 1:
                        result_str = "  WON"
                    elif pd.notna(placing) and placing <= 3:
                        result_str = f"  {int(placing)}nd" if placing == 2 else f"  {int(placing)}rd"
                    elif pd.notna(placing):
                        result_str = f"  {int(placing)}th"
                    else:
                        result_str = "  -"

                    lines.append(
                        f"  {rank:<3} {horse:<22} {pred_bfsp:>10.2f} "
                        f"{actual_str:>8} {diff_str:>7} {p_win:>6.1%} "
                        f"{value_str:>7} {result_str:>7}"
                    )
                else:
                    lines.append(
                        f"  {rank:<3} {horse:<22} {pred_bfsp:>10.2f} "
                        f"{p_win:>6.1%} {rank:>5}"
                    )

            lines.append("")

    # Top picks
    lines.append("  TOP SELECTIONS (lowest predicted BFSP per race)")
    lines.append("  " + "=" * 74)

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            top = race_df.sort_values("predicted_bfsp").iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            pred_bfsp = top.get("predicted_bfsp", 0)
            p_win = top.get("predicted_win_prob_norm", 0)

            extra = ""
            if has_actual:
                actual = top.get("actual_bfsp", np.nan)
                placing = top.get("placing_numerical", np.nan)
                if pd.notna(actual):
                    extra += f" (Actual: {actual:.1f}"
                    if pd.notna(placing):
                        result = "WON" if placing == 1 else f"Finished {int(placing)}"
                        extra += f", {result}"
                    extra += ")"

            lines.append(
                f"  {track} {rtime}: {horse} "
                f"— Pred BFSP {pred_bfsp:.2f}, P(Win) {p_win:.1%}{extra}"
            )

    # Value bets (horses whose predicted BFSP is shorter than actual)
    if has_actual:
        value_df = predictions[
            predictions["value_edge"].notna() & (predictions["value_edge"] > 5)
        ].sort_values("value_edge", ascending=False)

        if len(value_df) > 0:
            lines.append("")
            lines.append("  VALUE BETS (Model shorter than market by >5%)")
            lines.append("  " + "=" * 74)
            for _, row in value_df.head(20).iterrows():
                horse = row.get("horse_name", "?")
                track = row.get("track", "?")
                rtime = row.get("race_time", "?")
                pred = row.get("predicted_bfsp", 0)
                actual = row.get("actual_bfsp", 0)
                edge = row.get("value_edge", 0)
                placing = row.get("placing_numerical", np.nan)

                result = ""
                if pd.notna(placing):
                    result = " -> WON" if placing == 1 else f" -> {int(placing)}th"

                lines.append(
                    f"  {track} {rtime}: {horse} "
                    f"(Pred {pred:.2f} vs Actual {actual:.2f}, "
                    f"Edge {edge:+.1f}%){result}"
                )

    # Accuracy summary if actual BFSP available
    if has_actual:
        actual_vals = predictions["actual_bfsp"].dropna()
        pred_vals = predictions.loc[actual_vals.index, "predicted_bfsp"]

        if len(actual_vals) > 0:
            mae = np.mean(np.abs(pred_vals - actual_vals))
            median_ape = np.median(np.abs(pred_vals - actual_vals) / actual_vals) * 100
            corr = np.corrcoef(np.log(pred_vals), np.log(actual_vals))[0, 1]

            # How often does our top pick (lowest predicted BFSP) win?
            top_pick_wins = 0
            total_races = 0
            for _, race_df in predictions.groupby(group_col):
                if race_df["placing_numerical"].notna().any():
                    total_races += 1
                    top = race_df.sort_values("predicted_bfsp").iloc[0]
                    if top.get("placing_numerical") == 1:
                        top_pick_wins += 1

            lines.append("")
            lines.append("  ACCURACY SUMMARY")
            lines.append("  " + "=" * 74)
            lines.append(f"  BFSP MAE:           {mae:.2f}")
            lines.append(f"  Median APE:         {median_ape:.1f}%")
            lines.append(f"  Log correlation:    {corr:.4f}")
            if total_races > 0:
                lines.append(
                    f"  Top pick wins:      {top_pick_wins}/{total_races} "
                    f"({top_pick_wins/total_races:.1%})"
                )

    lines.append("")
    lines.append("  " + "-" * 74)
    lines.append("  Generated by BFSP Prediction Model (all custom metrics)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate BFSP predictions for today's races"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help="Path to model artifacts",
    )
    parser.add_argument(
        "--from-db", action="store_true",
        help="Use runners from the database (no HRB login needed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print predictions to stdout (no email)",
    )
    parser.add_argument(
        "--last-n-days", type=int, default=None,
        help="Run predictions on the last N days in the database",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Save predictions to CSV file",
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01",
        help="Earliest historical date to load (default: 2020-01-01). Reduces memory.",
    )
    parser.add_argument(
        "--non-runners", type=str, default=None, metavar="PREDICTIONS_CSV",
        help="Update an earlier run's predictions (its --output-csv) for the runners withdrawn since: "
             "drop every runner the card as it stands no longer lists and re-normalise its race "
             "(without_non_runners). Needs no database or model; writes --output-csv and, beside it, "
             "the non-runners",
    )
    parser.add_argument(
        "--card", type=str, default=None,
        help="With --non-runners: the card as it stands from a file, a racecard CSV or the HTML card page "
             "(predict-now's probe_card saves it), in place of a fetch from HRB",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    if args.non_runners:
        if not args.output_csv:
            parser.error("--non-runners writes the updated predictions to --output-csv")
        if args.card:
            card = read_card(args.card, target_date)
        elif os.getenv("HRB_USERNAME"):
            card = fetch_racecard_from_hrb(target_date)
        else:
            parser.error("--non-runners needs the card as it stands: HRB_USERNAME and HRB_PASSWORD, or --card")
        if len(card) == 0:
            log.error("No card to check the predictions against")
            sys.exit(1)
        update_for_non_runners(args.non_runners, args.output_csv, card)
        return

    if not os.path.exists(args.db):
        log.error(f"Database not found: {args.db}")
        log.error("Run 'python train_bfsp.py --generate-data' first.")
        sys.exit(1)

    # Load model
    model, feature_cols, vocab = load_bfsp_model(args.model_dir)

    # Load historical data
    log.info(f"Loading historical data (from {args.start_date})...")
    historical = load_historical(args.db, start_date=args.start_date)
    log.info(f"  {len(historical):,} historical rows loaded")

    if args.last_n_days:
        # Run backtesting over last N days
        log.info(f"Running predictions for last {args.last_n_days} days...")
        all_dates = sorted(historical["race_date"].dt.date.unique())
        test_dates = all_dates[-args.last_n_days:]

        all_predictions = []
        for td in test_dates:
            # Use all data before this date as history
            history_before = historical[
                historical["race_date"].dt.date < td
            ].copy()
            runners_on_date = historical[
                historical["race_date"].dt.date == td
            ].copy()

            if len(history_before) < 100 or len(runners_on_date) == 0:
                continue

            preds = prepare_and_predict(
                history_before, runners_on_date, model, feature_cols, td, vocab
            )
            if len(preds) > 0:
                all_predictions.append(preds)

        if all_predictions:
            combined = pd.concat(all_predictions, ignore_index=True)
            report = format_predictions(combined, target_date)
            print(report)

            if args.output_csv:
                out = with_model_rank(combined)
                out[[c for c in OUTPUT_COLS if c in out.columns]].to_csv(args.output_csv, index=False)
                log.info(f"Saved predictions to {args.output_csv}")
        else:
            log.warning("No predictions generated")
        return

    # Single date prediction
    if args.from_db:
        log.info(f"Getting runners from database for {target_date}...")
        target_runners = get_runners_from_db(args.db, str(target_date))
    else:
        # Try HRB login
        username = os.getenv("HRB_USERNAME")
        if username:
            log.info(f"Fetching race card from horseracebase.com for {target_date}...")
            target_runners = fetch_racecard_from_hrb(target_date)
        else:
            log.warning(
                "No HRB credentials found. Using database data. "
                "Set HRB_USERNAME and HRB_PASSWORD in .env for live race cards."
            )
            target_runners = get_runners_from_db(args.db, str(target_date))

    if len(target_runners) == 0:
        # Fall back to latest date in DB
        latest_date = historical["race_date"].max().date()
        log.warning(
            f"No runners found for {target_date}. "
            f"Falling back to latest DB date: {latest_date}"
        )
        target_date = latest_date
        target_runners = get_runners_from_db(args.db, str(target_date))

    if len(target_runners) == 0:
        log.error("No runners found in database.")
        sys.exit(1)

    log.info(f"Found {len(target_runners)} runners for {target_date}")

    # Separate history (before target date) from target runners
    history_before = historical[
        historical["race_date"].dt.date < target_date
    ].copy()

    if len(history_before) < 100:
        log.warning("Limited history — using full dataset for metrics")
        history_before = historical.copy()

    # Generate predictions
    predictions = prepare_and_predict(
        history_before, target_runners, model, feature_cols, target_date, vocab
    )

    if len(predictions) == 0:
        log.error("No predictions generated")
        sys.exit(1)

    # Format and display
    report = format_predictions(predictions, target_date)
    print(report)

    # Save to CSV if requested
    if args.output_csv:
        out = with_model_rank(predictions)
        out[[c for c in OUTPUT_COLS if c in out.columns]].to_csv(args.output_csv, index=False)
        log.info(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
