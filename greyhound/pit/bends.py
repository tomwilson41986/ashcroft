"""The bends feed: each run's race told bend by bend from GBGB's comment, its sectional and its finish (the owner,
16 Nov 2026: "parse the bends feed from comments + in running position, run style"; ledger
greyhound-bends-batch44-1116).

GBGB publishes no running positions at the bends (its results carry the first sectional, the finish and a comment:
sources/gbgb.py). The comment names events at places in the race: ``Led2To4``, ``LdFr4``, ``Chl3``, ``Crd1&2``,
``Bmp3/4``, ``CrdRunUp&1``, ``LedRunIn``. A place is a bend number (1-8), the run-up, the run-in or the near line, or
a fraction of the race (``1/4``, ``1/2``, ``3/4``). Butler, Butler and Gaine (2025, https://hdl.handle.net/10468/18155)
found the position at the fourth bend the one that matters most for the result, on 78,595 GB runs.

**One run, parsed** (``run_feed``), each place put on one scale, the share of the race run (``frac``):

- a bend ``k`` of a race with ``nb`` bends (from the trip: 2 under 350 m, 4 under 600 m, 6 under 800 m, else 8) at
  ``k / (nb + 1)``; the run-up 0.1; a fraction of the race at itself; the run-in and the near line 0.95. These are
  approximations of where the places fall, not measured distances;
- **leading**: ``LedK`` leads at K; ``LedToK`` from the start to K; ``LedFrK`` / ``LdFrK`` from K to the line;
  ``LedKToM`` from K to M; ``AlwaysLed``, ``SnLd``, ``LedRunUp`` lead early; ``LedRunIn`` / ``LdNrLn`` at the line;
- **challenging** (``Chl``), **trouble** (core.INTERFERENCE's words: crowded, bumped, baulked, checked, ...),
  **eased** (``Esd``), **moved off** (``MvdOff``), each at its places; a bare place after ``&`` takes the event before
  it (``Crd1&2``);
- **the position in running**: the dog's rank by first sectional among the race's runners with one (at 0.15), every
  place it led (1), and its finish (at 1.0); a place where another runner is recorded leading puts it no better than
  2. Between the anchors the position is interpolated: an estimate of where it ran, not a published position.

**The columns** (a dog's EARLIER days only, a day's runs averaged, the mean of its last ``N`` days; the race-relative
ones from today's runners' pre-race columns): ``bn_pos_first`` (the estimated position at the first bend),
``bn_pos_last`` (at the last bend), ``bn_gain_late`` (places made from the last bend to the line), ``bn_gain_all``
(from the first timing point to the line), ``bn_sec_rank`` (rank at the first timing point), ``bn_led`` (the share of
runs it led anywhere), ``bn_led_late`` (led at or after the last bend), ``bn_lost_lead`` (led, then beaten),
``bn_chl`` (challenged), ``bn_tr_early`` / ``bn_tr_mid`` / ``bn_tr_late`` (trouble before the second bend, in the
middle, from the last bend on), ``bn_eased``, ``bn_n`` (earlier days parsed), ``bn_pos_last_vs`` (``bn_pos_last``
less the field's mean), ``bn_pos_last_rank`` (1 the field's lowest: the expected leader into the home straight),
``bn_pos_first_rank``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit.speed import earlier

PREFIX = "bn_"
SOURCE = "GBGB results (comment, first sectional, finishing position, trip: sources/gbgb.py)"
N = 5
SEC_FRAC = 0.15
COLUMNS = {
    "bn_pos_first": "the estimated position at the first bend (comment, sectional rank and finish), mean of its last 5 days",
    "bn_pos_last": "the estimated position at the last bend, mean of its last 5 days",
    "bn_gain_late": "places made from the last bend to the line, mean of its last 5 days",
    "bn_gain_all": "places made from the first timing point to the line, mean of its last 5 days",
    "bn_sec_rank": "its rank by first sectional in the race (the position at the first timing point), mean of its last 5 days",
    "bn_led": "the share of its last 5 days it led anywhere (comment)",
    "bn_led_late": "the share of its last 5 days it led at or after the last bend",
    "bn_lost_lead": "the share of its last 5 days it led and was beaten",
    "bn_chl": "the share of its last 5 days it challenged",
    "bn_tr_early": "trouble before the second bend, mean of its last 5 days (a slight 0.5, badly 2)",
    "bn_tr_mid": "trouble from the second bend to before the last, mean of its last 5 days",
    "bn_tr_late": "trouble from the last bend on, mean of its last 5 days",
    "bn_eased": "the share of its last 5 days it was eased",
    "bn_n": "the number of earlier days with a parsed run",
    "bn_pos_last_vs": "bn_pos_last less the mean of the other runners' in today's race",
    "bn_pos_last_rank": "the rank of bn_pos_last in today's race (1 the lowest: the expected leader into the home straight)",
    "bn_pos_first_rank": "the rank of bn_pos_first in today's race (1 the lowest)",
}

_UNIT = re.compile(r"\d/\d|\d|[A-Z]{2,}(?![a-z])|[A-Z][a-z]*|[a-z]+")
_TROUBLE_W = {"crd", "crowded", "bmp", "bumped", "blk", "baulked", "balked", "hmp", "hampered", "imp", "impeded",
              "stb", "stumbled", "fcd", "forced", "ck", "check", "checked", "ko", "knocked", "brtdn", "brought", "fell",
              "fll", "struck", "hit"}
_LED_W = {"led", "ld"}
_BAD_W = {"b", "badly", "vb", "bb"}
_SLT_W = {"slt", "slightly"}
_JOIN = {"to", "t", "fr", "from", "over", "off", "down", "into", "rls", "w", "wide", "mid", "vw", "middle", "rails",
         "and"}


def bends_for(distance_m: float) -> int:
    d = float(distance_m) if np.isfinite(distance_m) else 480.0
    return 2 if d < 350 else 4 if d < 600 else 6 if d < 800 else 8


def place_frac(p: str, nb: int) -> float:
    if p in ("runup", "rnup"):
        return 0.1
    if p in ("runin", "rnin", "nrln", "nearline"):
        return 0.95
    if "/" in p:
        a, b = p.split("/")
        return float(a) / float(b) if float(b) else np.nan
    k = int(p)
    return min(k, nb) / (nb + 1.0)


def _units(part: str) -> list[str]:
    """A comment part's pieces in order, CamelCase split: 'SnLdToBBlkW1' -> sn ld to b blk w 1; places kept whole."""
    return [u.lower() for u in _UNIT.findall(part)]


def parse(comment, nb: int) -> dict:
    """One comment -> led spans (list of (from, to) fracs), challenge, trouble (early/mid/late), eased, moved off."""
    first_b, last_b = 1.0 / (nb + 1.0), nb / (nb + 1.0)
    second_b = 2.0 / (nb + 1.0)
    out = {"led": [], "chl": 0.0, "tr_early": 0.0, "tr_mid": 0.0, "tr_late": 0.0, "eased": 0.0, "moved": 0.0}
    events: list[dict] = []
    for tok in [t.strip() for t in str(comment or "").split(",") if t.strip()]:
        tok_events: list[dict] = []
        for part in tok.replace(" ", "").replace("-", "").split("&"):
            u = _units(part)
            ev = None
            prev = None
            mod = 1.0
            i = 0
            while i < len(u):
                w = u[i]
                nxt = u[i + 1] if i + 1 < len(u) else None
                place = None
                if re.fullmatch(r"\d/\d|\d", w):
                    place = place_frac(w, nb)
                elif w in ("run", "rn") and nxt in ("up", "in"):
                    place = 0.1 if nxt == "up" else 0.95
                    i += 1
                elif w in ("runup", "rnup"):
                    place = 0.1
                elif w in ("runin", "rnin"):
                    place = 0.95
                elif w in ("nr", "near") and nxt in ("ln", "line"):
                    place = 0.95
                    i += 1
                elif w in ("nrln", "nearline"):
                    place = 0.95
                if place is not None:
                    target = ev or (tok_events[-1] if tok_events else None)
                    if target is not None:
                        target["places"].append(place)
                    i += 1
                    prev = "place"
                    continue
                if w in _LED_W or w in ("aled", "ald", "alwaysled"):
                    mode = "at"
                    if prev in ("to",) and ev is not None and ev["type"] != "led":
                        mode = "from"                         # 'ChlFr4ToLed5': took the lead at 5
                    elif prev in ("always", "a") or w in ("aled", "ald", "alwaysled"):
                        mode = "always"
                    elif prev in ("sn", "soon"):
                        mode = "soon"
                    ev = {"type": "led", "mode": mode, "places": [], "w": 1.0}
                    tok_events.append(ev)
                elif w in ("to", "t") and ev is not None and ev["type"] == "led" and ev["mode"] != "soon":
                    ev["mode"] = "span" if ev["places"] else "to"
                elif w in ("fr", "from") and ev is not None and ev["type"] == "led":
                    ev["mode"] = "from"
                elif w in ("chl", "chal", "challenged"):
                    ev = {"type": "chl", "places": [], "w": 1.0}
                    tok_events.append(ev)
                elif w in ("esd", "eased"):
                    ev = {"type": "eased", "places": [], "w": 1.0}
                    tok_events.append(ev)
                elif w in ("mvd", "moved") and nxt == "off":
                    ev = {"type": "moved", "places": [], "w": 1.0}
                    tok_events.append(ev)
                    i += 1
                elif w in _BAD_W:
                    mod = 2.0
                elif w in _SLT_W:
                    mod = 0.5
                elif w in _TROUBLE_W:
                    if ev is not None and ev["type"] == "trouble" and prev in _TROUBLE_W:
                        pass                                  # 'FcdCk': one event
                    else:
                        ev = {"type": "trouble", "places": [], "w": mod}
                        tok_events.append(ev)
                    mod = 1.0
                prev = w
                i += 1
            if not u:
                continue
        events += tok_events
    for ev in events:
        pl = ev["places"]
        if ev["type"] == "led":
            m = ev["mode"]
            if m == "always":
                out["led"].append((0.0, 1.0))
            elif m == "soon":
                out["led"].append((first_b, first_b))
            elif m == "from" and pl:
                out["led"].append((pl[0], 1.0))
            elif m == "to" and pl:
                out["led"].append((0.0, pl[-1]))
            elif m == "span" and len(pl) >= 2:
                out["led"].append((pl[0], pl[-1]))
            elif pl:
                for x in pl:
                    out["led"].append((x, x))
        elif ev["type"] == "chl":
            out["chl"] = 1.0
        elif ev["type"] == "eased":
            out["eased"] = 1.0
        elif ev["type"] == "moved":
            out["moved"] = 1.0
        elif ev["type"] == "trouble":
            for f in pl or [first_b]:
                key = "tr_early" if f < second_b else "tr_late" if f >= last_b - 1e-9 else "tr_mid"
                out[key] += ev["w"]
    return out


def _interp(anchors: list[tuple[float, float]], x: float) -> float:
    a = sorted(anchors)
    xs, ys = [p[0] for p in a], [p[1] for p in a]
    return float(np.interp(x, xs, ys))


def run_feed(runs: pd.DataFrame) -> pd.DataFrame:
    """One row a run: the parsed comment and the estimated positions (first bend, last bend), from that run's own
    result (for later days only)."""
    r = runs
    n = len(r)
    card = r.is_card.to_numpy(bool) if "is_card" in r else np.zeros(n, bool)
    dist = pd.to_numeric(r.distance_m, errors="coerce").to_numpy(float) if "distance_m" in r else np.full(n, 480.0)
    pos = pd.to_numeric(r.position, errors="coerce").to_numpy(float)
    sec = pd.to_numeric(r.sectional, errors="coerce") if "sectional" in r else pd.Series(np.nan, index=r.index)
    sec = sec.where((sec > 1) & (sec < 15))
    srank = sec.groupby(r.race_id.to_numpy()).rank(method="average").to_numpy(float)
    comments = r.comment.tolist() if "comment" in r else [None] * n
    nbs = np.array([bends_for(d) for d in dist])
    parsed = [parse(c, int(b)) if not cd else None for c, b, cd in zip(comments, nbs, card)]
    # every place a runner of the race is recorded leading: the others are no better than 2 there
    race = r.race_id.to_numpy()
    led_pts: dict = {}
    for i, p in enumerate(parsed):
        if p:
            for a, b in p["led"]:
                led_pts.setdefault(race[i], []).append((a, b, i))
    rows = {k: np.full(n, np.nan) for k in ("pos_first", "pos_last", "gain_late", "gain_all", "sec_rank", "led",
                                            "led_late", "lost_lead", "chl", "tr_early", "tr_mid", "tr_late", "eased")}
    for i, p in enumerate(parsed):
        if p is None or not np.isfinite(pos[i]):
            continue
        nb = int(nbs[i])
        fb, lb = 1.0 / (nb + 1.0), nb / (nb + 1.0)
        anchors = [(1.0, pos[i])]
        if np.isfinite(srank[i]):
            anchors.append((SEC_FRAC, srank[i]))
        for a, b in p["led"]:
            for x in sorted({a, b, (a + b) / 2.0}):
                if 0.0 <= x < 1.0:
                    anchors.append((x, 1.0))
        if len(anchors) == 1:
            anchors.append((SEC_FRAC, pos[i]))
        pf, pl = _interp(anchors, fb), _interp(anchors, lb)
        mine = p["led"]
        for a, b, j in led_pts.get(race[i], []):
            if j == i:
                continue
            for x, cur in ((fb, "f"), (lb, "l")):
                if a - 1e-9 <= x <= b + 1e-9 and not any(a2 - 1e-9 <= x <= b2 + 1e-9 for a2, b2 in mine):
                    if cur == "f":
                        pf = max(pf, 2.0)
                    else:
                        pl = max(pl, 2.0)
        rows["pos_first"][i], rows["pos_last"][i] = pf, pl
        rows["gain_late"][i] = pl - pos[i]
        rows["sec_rank"][i] = srank[i]
        rows["gain_all"][i] = (srank[i] - pos[i]) if np.isfinite(srank[i]) else np.nan
        led = bool(mine)
        rows["led"][i] = float(led)
        rows["led_late"][i] = float(any(b >= lb - 1e-9 for _, b in mine))
        rows["lost_lead"][i] = float(led and pos[i] > 1)
        for k in ("chl", "tr_early", "tr_mid", "tr_late", "eased"):
            rows[k][i] = p[k]
    return pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(), **rows}, index=r.index)


def _rank_low(race, x) -> np.ndarray:
    return pd.Series(x).groupby(race).rank(method="average").to_numpy()


def _vs_field(race, x) -> np.ndarray:
    s = pd.Series(x)
    tot = s.groupby(race).transform("sum").to_numpy()
    cnt = s.notna().groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(x) & (cnt > 1), x - (tot - np.nan_to_num(x)) / np.maximum(cnt - 1, 1), np.nan)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    r = runs
    feed = run_feed(r)
    day = core.day(r)
    dn = ((day - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float)
    dog = pd.to_numeric(r.dog_id, errors="coerce").to_numpy(float)
    race = r.race_id.to_numpy()
    m = lambda c: earlier(dog, dn, feed[c].to_numpy(float), "mean", n=N)  # noqa: E731
    out = {"bn_pos_first": m("pos_first"), "bn_pos_last": m("pos_last"), "bn_gain_late": m("gain_late"),
           "bn_gain_all": m("gain_all"), "bn_sec_rank": m("sec_rank"), "bn_led": m("led"), "bn_led_late": m("led_late"),
           "bn_lost_lead": m("lost_lead"), "bn_chl": m("chl"), "bn_tr_early": m("tr_early"), "bn_tr_mid": m("tr_mid"),
           "bn_tr_late": m("tr_late"), "bn_eased": m("eased"),
           "bn_n": earlier(dog, dn, feed["pos_last"].to_numpy(float), "count")}
    out["bn_pos_last_vs"] = _vs_field(race, out["bn_pos_last"])
    out["bn_pos_last_rank"] = _rank_low(race, out["bn_pos_last"])
    out["bn_pos_first_rank"] = _rank_low(race, out["bn_pos_first"])
    tab = pd.DataFrame({"race_id": r.race_id.to_numpy(), "dog_id": r.dog_id.to_numpy(),
                        **{c: np.asarray(out[c], float).astype(np.float32) for c in COLUMNS}}, index=r.index)
    parsed = feed.pos_last.notna()
    notes = {"runs_parsed": int(parsed.sum()), "runs_led": int((feed.led == 1).sum()),
             "runs_led_late": int((feed.led_late == 1).sum()), "runs_with_trouble":
             int(((feed.tr_early + feed.tr_mid + feed.tr_late) > 0).sum()),
             "mean_abs_gain_late": round(float(feed.gain_late.abs().mean()), 3) if parsed.any() else None,
             "winners_led_late_share": round(float(feed.loc[pd.to_numeric(r.position, errors="coerce").eq(1)
                                                            & parsed, "led_late"].mean()), 3) if parsed.any() else None}
    return tab, notes
