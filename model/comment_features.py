"""What the in-running comments of a horse's PREVIOUS runs say about it.

Form figures record where a horse finished, not why. A horse that was hampered,
denied a clear run or slowly away and still finished close ran better than its
figure; one that was eased, or ran green on debut, was not asked for its best;
one that raced keen or wide spent energy the figure does not show. Those are
the classic "eyecatchers" the form book is read for.

The market reads comments too, so the test for each feature here is not
whether it predicts the result but whether it predicts it beyond the price:
scripts/residual_screen.py asks exactly that.

Every feature is lagged by the horse's own runs (a horse runs at most once a
day), so nothing about the race being predicted is read. The current run's
comment is post-race and never a feature; `COMMENT_POST_RACE` lists the
same-run flags so the leakage guard can refuse them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_minutes

# (class, pattern) -- lower-case, word-bounded, matched anywhere in the comment.
COMMENT_CLASSES: dict[str, str] = {
    "trouble": (r"\b(?:badly |slightly |sn )?hampered\b|\bhmpd\b|\bbumped\b|\bchecked\b|\bsqueezed (?:out|up)\b"
                r"|\bshort of room\b|\bno clear run\b|\bnot clear run\b|\bdenied (?:a )?clear run\b|\bno room\b"
                r"|\bnot much room\b|\bblocked\b|\bsnatched up\b|\bclipped heels\b|\bstumbled\b|\bcarried (?:left|right|wide)\b"
                r"|\bmet trouble\b|\binterference\b|\btightened up\b|\bbadly placed\b|\bhad to be switched\b"),
    "switched": r"\bswitched\b|\bswitch(?:ed)? (?:left|right|out|wide)\b",
    "slow_start": (r"\bslowly away\b|\bdwelt\b|\bmissed the break\b|\bstarted slowly\b|\bbadly away\b|\bslow start\b"
                   r"|\breared\b|\bwhipped round\b|\bfly-?jumped\b|\bveered (?:left|right) (?:start|leaving)\b|\bsluggish start\b"),
    "keen": (r"\bkeen\b|\bpulled hard\b|\btook (?:a )?(?:strong|keen|fierce|good) hold\b|\braced freely\b|\bover-?raced\b"
             r"|\brefused to settle\b|\bfailed to settle\b|\bwould not settle\b|\bhead high\b"),
    "wide": r"\bwide\b|\b(?:three|four|five|3|4|5) wide\b|\bwidest\b",
    "finished_well": (r"\bran on\b|\bstayed on\b|\bkept on\b|\bfinished (?:well|strongly|fast|best)\b|\bfast finish\b"
                      r"|\bnever nearer\b|\bnearest finish\b|\bgood late\b|\bstrong late\b|\blate headway\b|\bflew home\b"),
    "tender": (r"\bnot knocked about\b|\bnot given a hard time\b|\bnot pushed\b|\beased\b|\bhands and heels\b"
               r"|\bkindly ridden\b|\bnot persevered\b|\bshaped with promise\b|\bshowed promise\b|\bshaped well\b"
               r"|\bgreen\b|\bnovicey\b|\binexperience\b|\bcaught the eye\b|\bwill improve\b|\bnot unduly punished\b"),
    "weakened": (r"\bweakened\b|\bfaded\b|\bno extra\b|\btired\b|\bdropped away\b|\bone pace\b|\bone-paced\b"
                 r"|\btailed off\b|\bbeaten when\b|\bdropped out\b|\bstopped quickly\b|\blost place\b|\bstruggling\b"),
    "no_finish": r"\bpulled up\b|\bfell\b|\bunseated\b|\brefused\b|\bbrought down\b|\bran out\b|\bslipped up\b|\bcarried out\b",
    "problem": (r"\blost action\b|\blame\b|\bbled\b|\bbroke blood vessel\b|\bnosebleed\b|\bgurgled\b|\bwind\b"
                r"|\bheart\b|\bsore\b|\bunsound\b|\bstiff\b|\bhung (?:left|right|badly)\b|\bhanging\b|\blost (?:a )?shoe\b"
                r"|\bspread a plate\b|\bsaddle slipped\b|\bbit slipped\b|\btack\b"),
    "easy_win": (r"\bwon (?:easily|readily|cosily|comfortably|impressively|going away|with plenty in hand)\b"
                 r"|\bdrew clear\b|\bquickened clear\b|\bpulled clear\b|\bsauntered\b|\bcantered\b|\bheld on well\b"),
}
EXCUSES = ("trouble", "slow_start", "keen", "wide", "problem")

#: Same-run flags: descriptions of the race they come from, never features.
COMMENT_POST_RACE = tuple(f"_c_{k}" for k in COMMENT_CLASSES) + ("_c_excuse", "_c_nfp", "_c_lb")


def comment_flags(comments: pd.Series) -> pd.DataFrame:
    """One 0/1 column per class for each comment (NaN where there is none)."""
    c = comments.astype("string").str.lower()
    missing = c.isna() | (c.str.strip() == "")
    out = pd.DataFrame(index=comments.index)
    for k, rx in COMMENT_CLASSES.items():
        out[f"_c_{k}"] = c.str.contains(rx, regex=True).astype("float").where(~missing)
    out["_c_excuse"] = out[[f"_c_{k}" for k in EXCUSES]].max(axis=1)
    return out


def add_comment_features(df: pd.DataFrame, horse_col: str = "horse_name", comment_col: str = "comment",
                         pos_col: str = "placing_numerical", n_col: str = "number_of_runners",
                         beaten_col: str = "total_dst_bt", windows=(3, 6)) -> tuple[pd.DataFrame, list[str]]:
    """Attach lagged comment features; returns (frame, feature names).

    Per class k:
        LR_c_k          the flag in the horse's last run
        L{w}_c_k        share of its last w runs with the flag (w in `windows`)
    Plus interactions with how the last run finished:
        LR_excuse_close     an excuse last time and beaten 5 lengths or less
        LR_excuse_nfp       an excuse last time x its normalised finishing position
        LR_noexcuse_poor    no excuse last time and finished in the bottom half
        LR_tender_close     tenderly handled last time and beaten 5 lengths or less
        LR_finished_well_nfp  ran on last time x its normalised finishing position
        LR_easy_win         won easily last time
        runs_since_trouble  runs since the horse last had trouble (NaN: never)
    """
    from model.perf_figures import parse_beaten_lengths

    d = pd.DataFrame(index=df.index)
    d["_h"] = df[horse_col].astype(str)
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    d["_t"] = race_minutes(df["race_time"]) if "race_time" in df.columns else 0.0
    flags = comment_flags(df[comment_col]) if comment_col in df.columns else pd.DataFrame(index=df.index)
    d = d.join(flags)
    pos = pd.to_numeric(df[pos_col], errors="coerce")
    n = pd.to_numeric(df[n_col], errors="coerce")
    d["_c_nfp"] = ((n - pos) / (n - 1).replace(0, np.nan)).clip(0, 1)
    lb = df[beaten_col].map(parse_beaten_lengths).astype(float) if beaten_col in df.columns else np.nan
    d["_c_lb"] = pd.Series(lb, index=df.index).where(pos != 1, 0.0)

    d = d.sort_values(["_h", "_date", "_t"], kind="stable")
    g = d.groupby("_h", sort=False)
    out = pd.DataFrame(index=d.index)
    names: list[str] = []
    classes = [f"_c_{k}" for k in COMMENT_CLASSES] + ["_c_excuse"]
    for col in classes:
        k = col[3:]
        s1 = g[col].shift(1)
        out[f"LR_c_{k}"] = s1
        names.append(f"LR_c_{k}")
        for w in windows:
            shifted = g[col].shift(1)
            out[f"L{w}_c_{k}"] = (shifted.groupby(d["_h"], sort=False)
                                  .rolling(w, min_periods=1).mean().droplevel(0).reindex(d.index))
            names.append(f"L{w}_c_{k}")
    lr_nfp, lr_lb = g["_c_nfp"].shift(1), g["_c_lb"].shift(1)
    lr_exc = out["LR_c_excuse"]
    out["LR_excuse_close"] = (lr_exc * (lr_lb <= 5).astype(float)).where(lr_exc.notna() & lr_lb.notna())
    out["LR_excuse_nfp"] = lr_exc * lr_nfp
    out["LR_noexcuse_poor"] = ((1 - lr_exc) * (lr_nfp < 0.5).astype(float)).where(lr_exc.notna() & lr_nfp.notna())
    lr_tender = out["LR_c_tender"]
    out["LR_tender_close"] = (lr_tender * (lr_lb <= 5).astype(float)).where(lr_tender.notna() & lr_lb.notna())
    out["LR_finished_well_nfp"] = out["LR_c_finished_well"] * lr_nfp
    out["LR_easy_win"] = out["LR_c_easy_win"] * (g["_c_nfp"].shift(1) == 1).astype(float)
    # runs since the horse last met trouble: count of runs since the last flagged one
    tr = d["_c_trouble"].fillna(0.0)
    run_no = g.cumcount()
    last_tr = run_no.where(tr > 0).groupby(d["_h"], sort=False).ffill()
    prev_last = last_tr.groupby(d["_h"], sort=False).shift(1)
    out["runs_since_trouble"] = (run_no - prev_last).where(prev_last.notna())
    names += ["LR_excuse_close", "LR_excuse_nfp", "LR_noexcuse_poor", "LR_tender_close",
              "LR_finished_well_nfp", "LR_easy_win", "runs_since_trouble"]
    res = df.copy()
    for c in names:
        res[c] = out[c].reindex(df.index)
    return res, names
