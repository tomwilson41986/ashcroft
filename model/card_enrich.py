"""Give the live card the fields the model was trained with (QA review C1).

The model is trained on result rows, which carry every field horseracebase
exports. The morning card that reaches it live -- the HTML path, which is what
serves whenever the CSV download comes back empty -- carries the race header
and, per runner, stall, horse, age, weight, jockey, trainer, OR, headgear and
days since last run. Eleven fields the features read are simply absent, and a
missing category is scored as the first entry of its vocabulary, so every live
race was read as a Beginners Chase on a "Beach" surface, every horse as a colt.

Each field is filled from what is known at 06:00 -- the card's own text, or the
horse's, jockey's and track's earlier rows -- in the form the results table
uses (checked against it: research/queries/done/card_field_semantics.py):

  dist_furlongs     the header's distance text ('2m½f', '2m3.5f', '7f', '1m2f110y')
  race_type         the race name's key words, through a table learned from
                    history (e.g. novices + handicap + chase -> 'Handicap Novices
                    Chase'), with rules for a signature history has not seen
  surface_type      the track's surface, by going class where a course has both
                    turf and all-weather tracks (Kempton, Lingfield, Newcastle,
                    Southwell)
  track_direction   the track's direction at this trip (straight or round), by
                    going class, falling back to the track's usual one
  horse_sex         the horse's latest recorded sex; a filly of 5+ is a mare, a
                    colt of 5+ a horse (a gelding since its last run cannot be known)
  stallion, dam_stallion   the horse's (constant per horse)
  career_runs       the table's count at the horse's last run, plus that run (the table
                    counts the runs before each one; history may start after the first)
  jockeys_claim     the jockey's latest claim (0 for a jockey never seen)
  max_or_in_race, median_or   over the card's ORs, zeros for unrated as the table has them;
                    a median ending in .5 is missing, as the table's INTEGER column stores it
  official_rating   0 where blank: the table records unrated as 0, not missing
  prize_money       '84,405' -> 84405

A value the card already has is never overwritten. A debutant has no history to
fill from, so its pedigree and sex come from the card's own horse tooltip
(daily_predictions.parse_horse_title: 'Bay, Male, Stallion - X, Dam - Y'): the
sire as history spells it, the damsire from the dam's other offspring, a filly or
mare from 'Female', and for 'Male' the sex history's debutants of that age and
race code most often are. Without it a debutant's sire, damsire and sex were
missing at 06:00 though training always had them: on the parity days (28 and 25
March) that moved the 944's price for debutants by a mean 0.26 in log terms, over
half of all the difference between the live path and training.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

CARD_FILLED = ("dist_furlongs", "race_type", "surface_type", "track_direction", "horse_sex", "stallion",
               "dam_stallion", "career_runs", "jockeys_claim", "max_or_in_race", "median_or",
               "official_rating", "prize_money")

log = logging.getLogger(__name__)

#: What a debutant's 'Male' on the card can be; history's commonest at its age and race code is taken.
_MALE = ("Colt", "Gelding", "Horse", "Rig")

_AW_GOING = re.compile(r"standard|\bslow\b", re.I)
_DIST = re.compile(r"(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*f)?\s*(?:(\d+)\s*y)?", re.I)


def parse_distance_furlongs(text) -> float:
    """'2m½f' -> 16.5, '2m3.5f' -> 19.5, '7f' -> 7, '1m' -> 8, '1m2f110y' -> 10.5, junk -> NaN."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    t = text.strip().replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
    t = re.sub(r"(\d)\.5f", r"\g<1>.5f", t)
    t = re.sub(r"m\.(\d)", r"m0.\1", t)            # '2m.5f' (a bare ½ furlong) -> '2m0.5f'
    m = _DIST.fullmatch(t.replace(" ", ""))
    if not m or not any(m.groups()):
        return np.nan
    miles, furlongs, yards = (float(g) if g else 0.0 for g in m.groups())
    return miles * 8 + furlongs + yards / 220.0


def is_aw_going(going) -> bool:
    return isinstance(going, str) and bool(_AW_GOING.search(going))


def _norm(s) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = re.sub(r"\s*\((?:[A-Z]{2,3})\)\s*$", "", s)          # country suffix: 'Kodiac (IRE)'
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _full(s) -> str:
    """The name as a key with its country suffix kept: 'Lady Luck (IRE)' -> 'ladyluckire'."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _per_value(s: pd.Series, fn) -> pd.Series:
    """fn over a column, called once per distinct value: history runs to millions of rows."""
    codes, uniques = pd.factorize(s)
    out = np.array([fn(v) for v in uniques] + [fn(None)], dtype=object)    # code -1 (missing) -> fn(None)
    return pd.Series(out[codes], index=s.index)


# ---------------------------------------------------------------------------
# Race type from the race name
# ---------------------------------------------------------------------------

_WORDS = {
    "handicap": r"handicap|h'cap|\bhcap\b",
    "nursery": r"nursery",
    "hurdle": r"hurdle",
    "chase": r"chase|steeple",
    "novice": r"novice",
    "maiden": r"maiden",
    "beginners": r"beginner",
    "hunters": r"hunter",
    "nhflat": r"nh flat|national hunt flat|bumper|flat race",
    "claiming": r"claiming|claimer",
    "selling": r"selling|seller",
}
_WORDS_RX = {k: re.compile(v, re.I) for k, v in _WORDS.items()}


def race_signature(name) -> tuple:
    name = name if isinstance(name, str) else ""
    return tuple(k for k, rx in _WORDS_RX.items() if rx.search(name))


def race_type_by_rule(sig: tuple) -> str:
    s = set(sig)
    if "nursery" in s:
        return "Handicap Nursery"
    if "handicap" in s:
        if "novice" in s and "hurdle" in s:
            return "Handicap Novices Hurdle"
        if "novice" in s and "chase" in s:
            return "Handicap Novices Chase"
        for k, v in (("hurdle", "Handicap Hurdle"), ("chase", "Handicap Chase"), ("selling", "Handicap Seller"),
                     ("maiden", "Handicap Maiden")):
            if k in s:
                return v
        return "Handicap"
    if "maiden" in s:
        if "hurdle" in s:
            return "Maiden Hurdle"
        if "hunters" in s and "chase" in s:
            return "Maiden Hunters Chase"
        if "nhflat" in s:
            return "Maiden NH Flat"
        return "Maiden"
    if "novice" in s:
        if "hurdle" in s:
            return "Novices Hurdle"
        if "hunters" in s and "chase" in s:
            return "Novices Hunters Chase"
        if "chase" in s:
            return "Novices Chase"
        return "Novices"
    if "beginners" in s and "chase" in s:
        return "Beginners Chase"
    if "hunters" in s and "chase" in s:
        return "Hunters Chase"
    if "nhflat" in s:
        return "NH Flat"
    if "claiming" in s:
        return "Claimer Hurdle" if "hurdle" in s else "Claimer"
    if "selling" in s:
        return "Seller"
    if "hurdle" in s:
        return "Hurdle"
    if "chase" in s:
        return "Chase"
    return "Non-Handicap"


def learn_race_types(history: pd.DataFrame, min_count: int = 10) -> dict:
    """signature -> the race type history most often gives races with that signature."""
    if "race_name" not in history.columns or "race_type" not in history.columns:
        return {}
    h = history[["race_name", "race_type"]].dropna().drop_duplicates()
    if h.empty:
        return {}
    h = h.assign(sig=h["race_name"].map(race_signature))
    counts = h.groupby(["sig", "race_type"]).size().reset_index(name="n")
    counts = counts.sort_values(["sig", "n"], ascending=[True, False]).drop_duplicates("sig")
    return {r.sig: r.race_type for r in counts.itertuples() if r.n >= min_count}


# ---------------------------------------------------------------------------
# The fill
# ---------------------------------------------------------------------------

def _missing(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])


def _fill(card: pd.DataFrame, col: str, values) -> None:
    values = pd.Series(values, index=card.index)
    if col not in card.columns:
        card[col] = values
        return
    miss = _missing(card[col])
    card.loc[miss, col] = values[miss]


def _last_by(history: pd.DataFrame, keys: pd.Series, col: str) -> pd.Series:
    """Latest non-missing value of `col` per key, history in date order."""
    if col not in history.columns:
        return pd.Series(dtype=object)
    v = history[col].where(~_missing(history[col]))
    return v.groupby(keys).last()


def _as_history_names(names: pd.Series, known: pd.Series) -> pd.Series:
    """Each name as history spells it: the same name, suffix and all, else the one name history has
    with the same base ('Kodiac' for a card's 'Kodiac (GB)'), else the card's own."""
    kn = pd.Series(pd.unique(known[~_missing(known)].astype(str)), dtype=object)
    full, base = _per_value(kn, _full), _per_value(kn, _norm)
    single = base.map(base.value_counts()) == 1
    by_full = dict(zip(full, kn))
    by_base = dict(zip(base[single], kn[single]))
    out = _per_value(names, _full).map(by_full)
    out = out.where(out.notna(), _per_value(names, _norm).map(by_base))
    return out.where(out.notna(), names)


def _male_sex_guess(card: pd.DataFrame, hist: pd.DataFrame) -> pd.Series:
    """For a male debutant: the male sex history's debutants of its age (five and over as one) and
    race code most often were, else of its race code, else Gelding."""
    from model.race_shape import race_code
    guess = pd.Series("Gelding", index=card.index, dtype=object)
    if not {"horse_sex", "career_runs", "horse_age"} <= set(hist.columns):
        return guess
    h = hist[pd.to_numeric(hist["career_runs"], errors="coerce").eq(0) & hist["horse_sex"].isin(_MALE)]
    if h.empty:
        return guess
    hk = pd.DataFrame({"age": pd.to_numeric(h["horse_age"], errors="coerce").clip(upper=5).to_numpy(),
                       "code": race_code(h).to_numpy(), "sex": h["horse_sex"].to_numpy()})
    commonest = lambda s: s.value_counts().index[0]          # noqa: E731
    by_age_code = hk.groupby(["age", "code"])["sex"].agg(commonest)
    by_code = hk.groupby("code")["sex"].agg(commonest)
    age = pd.to_numeric(card.get("horse_age", pd.Series(np.nan, index=card.index)), errors="coerce").clip(upper=5)
    code = race_code(card)
    g = pd.Series([by_age_code.get((a, c)) for a, c in zip(age, code)], index=card.index, dtype=object)
    g = g.where(g.notna(), code.map(by_code))
    return g.where(g.notna(), guess)


def _fill_from_card_pedigree(card: pd.DataFrame, hist: pd.DataFrame) -> dict:
    """Where the horse's own rows left its sire, damsire or sex missing (a debutant), the card's
    tooltip (card_stallion, card_dam, card_sex): the sire as history spells it, the damsire from the
    dam's other offspring, the sex from Female or Male. The counts filled, by field."""
    n = {}
    if "card_stallion" in card.columns and "stallion" in card.columns:
        # the tooltip against history, where both name the sire: a check that the two spell sires alike
        both = ~_missing(card["stallion"]) & ~_missing(card["card_stallion"])
        if both.any():
            agree = _per_value(card.loc[both, "stallion"], _norm) == _per_value(card.loc[both, "card_stallion"], _norm)
            n["sire as history has it"] = f"{int(agree.sum())} of {int(both.sum())}"
    if "card_stallion" in card.columns and "stallion" in hist.columns:
        miss = _missing(card["stallion"]) if "stallion" in card.columns else pd.Series(True, index=card.index)
        _fill(card, "stallion", _as_history_names(card["card_stallion"], hist["stallion"]))
        n["sire"] = int((miss & ~_missing(card["stallion"])).sum())
    if "card_dam" in card.columns and {"dam", "dam_stallion"} <= set(hist.columns):
        miss = _missing(card["dam_stallion"]) if "dam_stallion" in card.columns else pd.Series(True, index=card.index)
        dam = _as_history_names(card["card_dam"], hist["dam"])
        by_dam = _last_by(hist, _per_value(hist["dam"], _full), "dam_stallion").drop("", errors="ignore")
        _fill(card, "dam_stallion", _per_value(dam, _full).map(by_dam))
        n["damsire"] = int((miss & ~_missing(card["dam_stallion"])).sum())
    if "card_dam" in card.columns and {"horse_name", "stallion", "horse_sex"} <= set(hist.columns):
        # a first foal has no siblings to read it from, but a dam that raced is in history herself, and
        # her sire is the damsire; a mare or filly of that name only, so a namesake gelding cannot answer
        miss = _missing(card["dam_stallion"]) if "dam_stallion" in card.columns else pd.Series(True, index=card.index)
        mares = hist[hist["horse_sex"].isin(["Filly", "Mare"])]
        own_sire = _last_by(mares, _per_value(mares["horse_name"], _full), "stallion").drop("", errors="ignore")
        dam_key = _per_value(_as_history_names(card["card_dam"], mares["horse_name"]), _full)
        _fill(card, "dam_stallion", dam_key.map(own_sire))
        n["damsire from the dam's own races"] = int((miss & ~_missing(card["dam_stallion"])).sum())
    if "card_sex" in card.columns:
        miss = _missing(card["horse_sex"]) if "horse_sex" in card.columns else pd.Series(True, index=card.index)
        said = card["card_sex"].astype(str).str.strip().str.lower()
        age = pd.to_numeric(card.get("horse_age", pd.Series(np.nan, index=card.index)), errors="coerce")
        sex = np.where(said.eq("female"), np.where(age >= 5, "Mare", "Filly"),
                       np.where(said.eq("male"), _male_sex_guess(card, hist).to_numpy(), None))
        _fill(card, "horse_sex", pd.Series(sex, index=card.index, dtype=object))
        n["sex"] = int((miss & ~_missing(card["horse_sex"])).sum())
    return n


def enrich_card(card: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """The card with every field in CARD_FILLED present, filled only where missing.

    `history` is the result rows before the card's date (the frame the live path
    already loads); nothing from the card's own races is read from it."""
    card = card.copy()
    hist = history.copy()
    order = [c for c in ("race_date", "race_time") if c in hist.columns]
    if order:
        hist = hist.sort_values(order, kind="stable")

    # header fields
    if "race_distance" in card.columns:
        _fill(card, "dist_furlongs", card["race_distance"].map(parse_distance_furlongs))
    if "prize_money" in card.columns:
        pm = card["prize_money"]
        if not pd.api.types.is_numeric_dtype(pm):             # text ('4,187'), object or string dtype
            card["prize_money"] = pd.to_numeric(pm.astype(str).str.replace(r"[£,\s]", "", regex=True), errors="coerce")
    if "race_name" in card.columns:
        table = learn_race_types(hist)
        sig = card["race_name"].map(race_signature)
        _fill(card, "race_type", sig.map(lambda s: table.get(s, race_type_by_rule(s))))

    # track fields: by going class, and for direction by trip
    if "track" in card.columns:
        aw_card = _per_value(card.get("going_description", pd.Series(index=card.index, dtype=object)), is_aw_going)
        tk_card = _per_value(card["track"], _norm)
        tk_hist = (_per_value(hist["track"], _norm) if "track" in hist.columns
                   else pd.Series(index=hist.index, dtype=object))
        aw_hist = _per_value(hist.get("going_description", pd.Series(index=hist.index, dtype=object)), is_aw_going)
        for col in ("surface_type", "track_direction"):
            if col not in hist.columns:
                continue
            by_class = _last_by(hist, tk_hist + "|" + aw_hist.astype(str), col)
            by_track = _last_by(hist, tk_hist, col)
            v = (tk_card + "|" + aw_card.astype(str)).map(by_class).fillna(tk_card.map(by_track))
            if col == "track_direction" and "dist_furlongs" in hist.columns and "dist_furlongs" in card.columns:
                dist_h = pd.to_numeric(hist["dist_furlongs"], errors="coerce").round().astype("Int64").astype(str)
                dist_c = pd.to_numeric(card["dist_furlongs"], errors="coerce").round().astype("Int64").astype(str)
                d = hist.assign(_k=tk_hist + "|" + aw_hist.astype(str) + "|" + dist_h)
                mode = d.dropna(subset=[col]).groupby("_k")[col].agg(lambda s: s.value_counts().index[0])
                v = (tk_card + "|" + aw_card.astype(str) + "|" + dist_c).map(mode).fillna(v)
            _fill(card, col, v)

    # horse fields: the exact name first, so namesakes bred in different countries stay apart
    # ('Lady Luck (IRE)', 'Lady Luck (GB)'); the name without its suffix only for a name history
    # has never seen exactly (a card printing 'Lady Luck' for history's 'Lady Luck (IRE)')
    if "horse_name" in card.columns and "horse_name" in hist.columns:
        fk_card, fk_hist = _per_value(card["horse_name"], _full), _per_value(hist["horse_name"], _full)
        bk_card, bk_hist = _per_value(card["horse_name"], _norm), _per_value(hist["horse_name"], _norm)
        exact = fk_card.isin(set(fk_hist))

        def by_horse(per_full: pd.Series, per_base: pd.Series) -> pd.Series:
            return fk_card.map(per_full).where(exact, bk_card.map(per_base))

        for col in ("stallion", "dam_stallion", "horse_sex"):
            _fill(card, col, by_horse(_last_by(hist, fk_hist, col), _last_by(hist, bk_hist, col)))
        filled = _fill_from_card_pedigree(card, hist)
        if filled:
            log.info("Card pedigree for horses with no history: " + ", ".join(f"{k} {v}" for k, v in filled.items()))
        age = pd.to_numeric(card.get("horse_age"), errors="coerce")
        sex = card["horse_sex"]
        card["horse_sex"] = np.where((sex == "Filly") & (age >= 5), "Mare",
                                     np.where((sex == "Colt") & (age >= 5), "Horse", sex))
        card.loc[_missing(pd.Series(sex, index=card.index)), "horse_sex"] = np.nan
        # the table's own count continued (its last value counts the runs before that one, so +1):
        # the live job loads history from 2020, and counting rows would miss every run before it
        last = pd.to_numeric(by_horse(_last_by(hist, fk_hist, "career_runs"), _last_by(hist, bk_hist, "career_runs")),
                             errors="coerce")
        counted = by_horse(fk_hist.value_counts(), bk_hist.value_counts())
        _fill(card, "career_runs", (last + 1).fillna(counted).fillna(0).astype(float))

    # jockey claim: printed after the name if the card shows it ('J Smith (3)', which is then
    # stripped so the name matches history), else the jockey's latest recorded claim, else 0
    if "jockey_name" in card.columns:
        text = card["jockey_name"].astype(str)
        printed = pd.to_numeric(text.str.extract(r"\((\d{1,2})\)\s*$", expand=False), errors="coerce")
        card["jockey_name"] = card["jockey_name"].where(printed.isna(),
                                                        text.str.replace(r"\s*\(\d{1,2}\)\s*$", "", regex=True))
        jk_hist = _per_value(hist["jockey_name"], _norm) if "jockey_name" in hist.columns else None
        last = (_last_by(hist, jk_hist, "jockeys_claim") if jk_hist is not None else pd.Series(dtype=object))
        from_history = pd.to_numeric(_per_value(card["jockey_name"], _norm).map(last), errors="coerce").fillna(0.0)
        _fill(card, "jockeys_claim", printed.fillna(from_history))

    # race OR spread over the card, unrated as 0 the way the table records them
    if "official_rating" in card.columns:
        orr = pd.to_numeric(card["official_rating"], errors="coerce")
        card["official_rating"] = orr.fillna(0.0)
        race = card["race_date"].astype(str) + "|" + card["track"].astype(str) + "|" + card["race_time"].astype(str)
        _fill(card, "max_or_in_race", card["official_rating"].groupby(race).transform("max"))
        # the median over every runner, unrated as 0 (an all-unrated race has 0, not a gap);
        # the column is an INTEGER and a median ending in .5 is stored as missing, in 99.96%
        # of 385,657 rows (research/queries/done/card_gaps_check.py), so the model was trained
        # on a gap there and the card gives it one
        med = card["official_rating"].groupby(race).transform("median")
        _fill(card, "median_or", med.where(med % 1 == 0))
    return card
