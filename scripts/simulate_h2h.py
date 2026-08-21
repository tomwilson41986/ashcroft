"""
Head-to-head Monte Carlo match simulation between two horses.

Parses the logged-in horseracebase form pages saved by
scripts/fetch_horse_form.py, converts each career run into a
performance figure (lbs, on the official-rating scale), then samples
race-day performances for both horses and counts who finishes in
front over N simulated runs.

Performance figure per run:
    base            = horse's OR at the time (fallback: race median OR)
    non-winner      = base - beaten_lengths * lbs_per_length(distance)
    winner          = base + min(winning_margin_lbs, 5)   (+2 if margin unknown)
Runs with no usable rating (early maidens, OR and median OR both 0)
are dropped, and beaten-run figures are floored at base - 20 lbs so
eased/tailed-off runs count as bad days, not absurd ones.

Race-day sampling: recency-weighted bootstrap of the horse's actual
performance figures (half-life 270 days) plus N(0, 3) day-to-day
noise. The bootstrap keeps the real shape of each horse's form —
including genuine flop runs — rather than assuming a clean bell curve.

Usage:
    python scripts/simulate_h2h.py \
        --horse-a data/h2h/horse_397142.html \
        --horse-b data/h2h/horse_399032.html \
        --runs 10000 --seed 42
"""

import argparse
import html as html_mod
import json
import math
import re
from datetime import date

import numpy as np

REFERENCE_DATE = date(2026, 8, 18)   # "today" for recency weighting
HALF_LIFE_DAYS = 270.0
DAY_NOISE_LBS = 3.0                  # race-day variance on top of form spread

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

# Short-distance margins in lengths
MARGIN_WORDS = {"NSE": 0.05, "SH": 0.1, "HD": 0.2, "SNK": 0.25, "NK": 0.3,
                "DH": 0.0, "DIST": 30.0}


def lbs_per_length(dist_furlongs: float) -> float:
    """Standard weight-for-distance scale: ~3 lbs/length at 5f down to
    ~0.75 at 20f, linearly interpolated."""
    pts = [(5, 3.0), (8, 2.0), (12, 1.5), (16, 1.0), (20, 0.75)]
    if dist_furlongs <= pts[0][0]:
        return pts[0][1]
    if dist_furlongs >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= dist_furlongs <= x1:
            frac = (dist_furlongs - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return 1.5


def parse_distance(txt: str) -> float:
    """'1m6f' -> 14, '1m2½f' -> 10.5, '7f' -> 7, '1m½f' -> 8.5"""
    txt = txt.strip().replace("½", ".5")
    miles = furlongs = 0.0
    m = re.match(r"(?:(\d+)m)?(?:([\d.]+)f)?", txt)
    if m:
        miles = float(m.group(1) or 0)
        furlongs = float(m.group(2) or 0)
    return miles * 8 + furlongs


def parse_margin(txt: str) -> float | None:
    """'8L' -> 8.0, 'SH' -> 0.1, '1.8L' -> 1.8"""
    txt = txt.strip().rstrip(".")
    if not txt:
        return None
    if txt.upper() in MARGIN_WORDS:
        return MARGIN_WORDS[txt.upper()]
    m = re.match(r"([\d.]+)L$", txt)
    return float(m.group(1)) if m else None


def strip_tags(fragment: str) -> str:
    return html_mod.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_form_page(path: str) -> dict:
    """Extract horse name, profile line and run history from a saved page."""
    raw = open(path, encoding="utf-8").read()
    raw = re.sub(r"<script.*?</script>", "", raw, flags=re.S)

    name_m = re.search(r"<title>(.*?) - Horse Racing Form</title>", raw)
    name = html_mod.unescape(name_m.group(1)) if name_m else path

    tables = re.findall(r"<table.*?</table>", raw, flags=re.S)
    form_tables = [t for t in tables if "BFP" in t]
    if not form_tables:
        raise ValueError(f"No form table found in {path} — page may be behind login wall")
    form = form_tables[-1]

    rows = re.findall(r"<tr.*?</tr>", form, flags=re.S)
    runs = []
    current = None
    for r in rows:
        cells = [strip_tags(c) for c in re.findall(r"<t[dh].*?</t[dh]>", r, flags=re.S)]
        if not cells:
            continue
        if cells[0].startswith("Date"):
            continue
        if len(cells) >= 12:
            # main run row
            date_m = re.match(r"(\d+)\.(\w+)\.(\d+)", cells[0])
            if not date_m:
                continue
            d, mon, yy = int(date_m.group(1)), date_m.group(2), int(date_m.group(3))
            run_date = date(2000 + yy, MONTHS[mon], d)

            posran = cells[2]
            pos_m = re.match(r"(\d+)/(\d+)\s*(.*)", posran)
            if not pos_m:
                continue
            pos, ran = int(pos_m.group(1)), int(pos_m.group(2))
            margin = parse_margin(pos_m.group(3))

            current = {
                "date": run_date.isoformat(),
                "days_ago": (REFERENCE_DATE - run_date).days,
                "track": cells[1],
                "pos": pos,
                "ran": ran,
                "margin_lengths": margin,
                "race_type": cells[3],
                "dist_furlongs": parse_distance(cells[4]),
                "going": cells[5],
                "race_class": cells[6],
                "or_rating": int(cells[10]) if cells[10].isdigit() else 0,
                "odds": cells[11],
                "median_or": None,
            }
            runs.append(current)
        elif current is not None and len(cells) <= 2:
            # comment row carries MedianOR
            med = re.search(r"MedianOR\s+([\d.]+)", cells[0])
            if med:
                current["median_or"] = float(med.group(1))
    return {"name": name, "runs": runs}


def performance_figure(run: dict) -> float | None:
    base = run["or_rating"]
    if base <= 0:
        base = run["median_or"] or 0
    if base <= 0:
        return None   # unrated early-career run, no usable baseline

    lpl = lbs_per_length(run["dist_furlongs"])
    margin = run["margin_lengths"]
    if run["pos"] == 1:
        credit = min(margin * lpl, 5.0) if margin is not None else 2.0
        return base + credit
    if margin is None:
        return base   # placed with no recorded margin
    # Floor at base - 20 lbs: horses eased/tailed off when out of
    # contention produce meaningless beaten distances (standard
    # handicapping treats such runs as unratable beyond this point).
    return max(base - margin * lpl, base - 20.0)


def build_features(horse: dict) -> dict:
    figs, weights, detail = [], [], []
    for run in horse["runs"]:
        perf = performance_figure(run)
        if perf is None:
            detail.append({**run, "perf": None, "weight": 0.0, "used": False})
            continue
        w = 0.5 ** (run["days_ago"] / HALF_LIFE_DAYS)
        figs.append(perf)
        weights.append(w)
        detail.append({**run, "perf": round(perf, 1), "weight": round(w, 3), "used": True})

    figs = np.array(figs, dtype=float)
    weights = np.array(weights, dtype=float)
    wmean = float(np.average(figs, weights=weights))
    wvar = float(np.average((figs - wmean) ** 2, weights=weights))
    return {
        "name": horse["name"],
        "figures": figs,
        "weights": weights / weights.sum(),
        "weighted_mean": wmean,
        "weighted_std": math.sqrt(wvar),
        "runs_used": len(figs),
        "runs_total": len(horse["runs"]),
        "detail": detail,
    }


def parse_weight(txt: str) -> float:
    """Carried weight in lbs. Accepts '9-12', '9st12', '9st 12lb' or plain lbs."""
    txt = txt.strip().lower().replace("st", "-").replace("lb", "")
    m = re.match(r"(\d+)\s*-\s*(\d+)", txt)
    if m:
        return int(m.group(1)) * 14 + int(m.group(2))
    return float(txt)


def simulate(fa: dict, fb: dict, n_runs: int, seed: int,
             weight_a: float = 0.0, weight_b: float = 0.0) -> dict:
    """weight_a/weight_b: carried weight in lbs (0 = level weights).
    In a handicap, each extra lb carried costs 1 lb of performance."""
    rng = np.random.default_rng(seed)
    draw_a = rng.choice(fa["figures"], size=n_runs, p=fa["weights"]) \
        + rng.normal(0, DAY_NOISE_LBS, n_runs)
    draw_b = rng.choice(fb["figures"], size=n_runs, p=fb["weights"]) \
        + rng.normal(0, DAY_NOISE_LBS, n_runs)
    diff = (draw_a - weight_a) - (draw_b - weight_b)
    a_wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())    # measure-zero with continuous noise
    b_wins = n_runs - a_wins - ties
    return {
        "runs": n_runs,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_pct": 100 * a_wins / n_runs,
        "b_win_pct": 100 * b_wins / n_runs,
        "mean_margin_lbs": float(diff.mean()),
        "weight_a_lbs": weight_a,
        "weight_b_lbs": weight_b,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horse-a", required=True)
    ap.add_argument("--horse-b", required=True)
    ap.add_argument("--runs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-a", default="0",
                    help="Weight carried by horse A, e.g. '9-7' (default: level)")
    ap.add_argument("--weight-b", default="0",
                    help="Weight carried by horse B, e.g. '9-12' (default: level)")
    ap.add_argument("--out", default="data/h2h/simulation_result.json")
    args = ap.parse_args()

    a = build_features(parse_form_page(args.horse_a))
    b = build_features(parse_form_page(args.horse_b))

    for f in (a, b):
        print(f"\n{f['name']}  ({f['runs_used']}/{f['runs_total']} runs used)")
        print(f"  recency-weighted ability: {f['weighted_mean']:.1f} lbs "
              f"(spread {f['weighted_std']:.1f})")
        for r in f["detail"]:
            flag = "" if r["used"] else "  [skipped — unrated]"
            perf = f"{r['perf']:>6}" if r["perf"] is not None else "     -"
            print(f"  {r['date']}  {r['track'][:22]:<22} {r['pos']:>2}/{r['ran']:<2} "
                  f"{r['race_type'][:9]:<9} OR {r['or_rating']:>3}  perf {perf} "
                  f" w={r['weight']:.3f}{flag}")

    wa, wb = parse_weight(args.weight_a), parse_weight(args.weight_b)
    res = simulate(a, b, args.runs, args.seed, wa, wb)
    print("\n" + "=" * 64)
    print(f"MATCH RESULT over {res['runs']:,} simulated races (seed {args.seed})")
    if wa or wb:
        print(f"  Weights: A carries {int(wa)//14}-{int(wa)%14} ({wa:.0f} lbs), "
              f"B carries {int(wb)//14}-{int(wb)%14} ({wb:.0f} lbs)")
    print(f"  {a['name']:<30} {res['a_wins']:>6,}  ({res['a_win_pct']:.1f}%)")
    print(f"  {b['name']:<30} {res['b_wins']:>6,}  ({res['b_win_pct']:.1f}%)")
    print(f"  Mean ability gap: {res['mean_margin_lbs']:+.1f} lbs "
          f"({'A' if res['mean_margin_lbs'] > 0 else 'B'} ahead)")
    print("=" * 64)

    out = {
        "horse_a": {k: a[k] for k in ("name", "weighted_mean", "weighted_std",
                                      "runs_used", "runs_total", "detail")},
        "horse_b": {k: b[k] for k in ("name", "weighted_mean", "weighted_std",
                                      "runs_used", "runs_total", "detail")},
        "simulation": res,
        "config": {"seed": args.seed, "half_life_days": HALF_LIFE_DAYS,
                   "day_noise_lbs": DAY_NOISE_LBS,
                   "reference_date": REFERENCE_DATE.isoformat()},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
