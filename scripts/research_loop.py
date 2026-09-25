#!/usr/bin/env python3
"""The research loop's price-forecast comparison, one fold per job.

An iteration used to fit its six models one after another in a single job --
the base on three folds, then the variant on the same three -- and took two
and a half hours, most of it waiting on fits that did not depend on each
other. Here each (arm, fold) is its own job, so the wall clock is one fold's
fit; the base, which is the served feature set and changes only when the
served model does, is fitted once and reused from the cache while its inputs
are unchanged; and `"recipe": "quick"` screens on a recipe about seven times
faster a fit (model/bfsp_model.py QUICK_PARAMS) before a block is confirmed on
the served one.

    plan     read research/loop.json and the cached matrix's dates: the folds,
             the key the base run is cached under
    arms     which arms need fitting, given whether the base is cached
    fit      one arm on one fold (evaluate_oos.py --fold, the serial run's fold)
    gather   each arm's folds into one prediction file and one summary
    compare  the paired comparison, the replicate check and the early-price
             trade, into reports/bfsp_compare/compare.md

The arms: `base` (production features, plus `bfsp_base_blocks`, less
`bfsp_base_withhold`); one arm per variant; and, with `"replicate_base": true`,
`base_rep`, the base fitted again in another job, whose distance from `base` is
the noise floor any difference between arms has to clear.

A variant is the base plus its own blocks, less its own withheld features, with
its own extra arguments. Several can run at once, each against the same base:

    "variants": [{"name": "fv", "blocks": "form_variants"},
                 {"name": "fv_no_lb", "blocks": "form_variants", "drop": "fv_lb"},
                 {"name": "r6000", "args": "--num-boost-round 6000"}]

Without "variants" the single variant is the old top-level keys (arm `blocks`:
`blocks`, `bfsp_variant_drop`, `bfsp_variant_withhold`, `bfsp_variant_args`).

Every step reads the same config, so the YAML only wires jobs together.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: The locked holdout starts here and the loop never scores it.
EVAL_UNTIL = "2026-04-01"
START_DATE = os.environ.get("START_DATE", "2021-01-01")
FEATURE_CACHE = ".feature_cache"
FIT_DIR = "reports/fit"
OUT_DIR = "reports/bfsp_compare"
BASE_CACHE_DIR = "reports/base_cache"

#: What decides the base run's predictions besides the matrix and the arguments.
FIT_CODE = ("evaluate_oos.py", "model/bfsp_model.py")


def load_config(path: str = "research/loop.json") -> dict:
    return json.loads((REPO / path).read_text())


def recipe(cfg: dict) -> str:
    r = cfg.get("recipe", "served")
    if r not in ("served", "quick"):
        raise SystemExit(f"loop.json recipe {r!r}: served or quick")
    return r


def common_args(cfg: dict) -> list[str]:
    """Arguments every arm shares: the matrix, the folds, the recipe."""
    return ["--start-date", START_DATE, "--feature-cache", FEATURE_CACHE,
            "--eval-from", cfg.get("bfsp_from", "2025-07-01"), "--eval-until", EVAL_UNTIL,
            "--step-days", "91", "--val-window", "91", "--recipe", recipe(cfg)]


BASE_ARMS = ("base", "base_rep")


def variants(cfg: dict) -> list[dict]:
    """The variant arms, each {name, blocks, drop, withhold, args}."""
    if "variants" not in cfg:
        return [{"name": "blocks", "blocks": cfg.get("blocks", ""), "drop": cfg.get("bfsp_variant_drop", ""),
                 "withhold": cfg.get("bfsp_variant_withhold", ""), "args": cfg.get("bfsp_variant_args", "")}]
    out, seen = [], set()
    for v in cfg["variants"]:
        name = v.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name) or name in BASE_ARMS or name in seen:
            raise SystemExit(f"variant name {name!r}: lower-case, unique, not {BASE_ARMS}")
        seen.add(name)
        out.append({"name": name, "blocks": v.get("blocks", ""), "drop": v.get("drop", ""),
                    "withhold": v.get("withhold", ""), "args": v.get("args", "")})
    if not out:
        raise SystemExit("variants is empty")
    return out


def _join_blocks(*parts: str) -> str:
    seen = []
    for p in parts:
        for b in (p or "").split(","):
            if b.strip() and b.strip() not in seen:
                seen.append(b.strip())
    return ",".join(seen)


def arm_args(cfg: dict, arm: str) -> list[str]:
    base_blocks = cfg.get("bfsp_base_blocks", "")
    if arm in BASE_ARMS:
        a = ["--blocks", base_blocks] if base_blocks else []
        wh = cfg.get("bfsp_base_withhold", "")
        return a + (["--withhold", wh] if wh else [])
    for v in variants(cfg):
        if v["name"] == arm:
            a = ["--blocks", _join_blocks(base_blocks, v["blocks"])]
            if v["drop"]:
                a += ["--drop-features", v["drop"]]
            wh = _join_blocks(cfg.get("bfsp_base_withhold", ""), v["withhold"])
            if wh:
                a += ["--withhold", wh]
            return a + shlex.split(v["args"])
    raise SystemExit(f"unknown arm {arm!r}")


def fit_code_hash() -> str:
    h = hashlib.sha256()
    for rel in FIT_CODE:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return h.hexdigest()[:16]


def base_key(cfg: dict, feature_key: str, folds: list) -> str:
    """Everything the base run's predictions are a function of.

    The matrix (its key covers the feature code and the data), the arguments,
    the fold plan, the fitting code and the library. Fits are deterministic
    (DEFAULT_PARAMS), so an equal key means the same predictions: reusing them
    is the base run, not an approximation of it."""
    import lightgbm
    parts = [feature_key, json.dumps(common_args(cfg)), json.dumps(arm_args(cfg, "base")),
             json.dumps(folds, sort_keys=True), fit_code_hash(), lightgbm.__version__,
             sys.version.split()[0]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _outputs(pairs: dict) -> None:
    """Write step outputs (GITHUB_OUTPUT when set, stdout otherwise)."""
    target = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in pairs.items()]
    if target:
        with open(target, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_plan(args) -> None:
    cfg = load_config(args.config)
    from model import feature_cache
    fkey = feature_cache.cache_key(args.db, START_DATE)
    Path(FIT_DIR).mkdir(parents=True, exist_ok=True)
    plan_file = Path(FIT_DIR) / "folds.json"
    subprocess.run([sys.executable, "evaluate_oos.py", "--db", args.db, *common_args(cfg),
                    "--list-folds", str(plan_file)], check=True, cwd=REPO)
    folds = json.loads(plan_file.read_text())
    if not folds:
        raise SystemExit("the fold plan is empty")
    _outputs({
        "folds": json.dumps([f["fold"] for f in folds]),
        "base_key": base_key(cfg, fkey, folds),
        "recipe": recipe(cfg),
        "replicate": "true" if cfg.get("replicate_base") else "false",
    })


def all_arms(cfg: dict) -> list[str]:
    return ["base"] + [v["name"] for v in variants(cfg)] + (["base_rep"] if cfg.get("replicate_base") else [])


def cmd_arms(args) -> None:
    cfg = load_config(args.config)
    arms = [a for a in all_arms(cfg) if not (a == "base" and args.base_hit == "true")]
    _outputs({"arms": json.dumps(arms)})


def cmd_fit(args) -> None:
    cfg = load_config(args.config)
    Path(FIT_DIR).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "evaluate_oos.py", "--db", args.db, *common_args(cfg),
           "--fold", str(args.fold), "--tag", f"{args.arm}_f{args.fold}", *arm_args(cfg, args.arm),
           "--output-csv", f"{FIT_DIR}/oos.csv", "--output-json", f"{FIT_DIR}/summary.json"]
    print("+", shlex.join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=REPO).returncode)


def _fold_files(src: str, arm: str, kind: str, ext: str) -> list[tuple[int, str]]:
    pat = re.compile(rf"{kind}_{re.escape(arm)}_f(\d+)\.{ext}$")
    found = []
    for p in glob.glob(os.path.join(src, "**", f"{kind}_{arm}_f*.{ext}"), recursive=True):
        m = pat.search(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def gather(src: str, out: str, arm: str, expect: list[int] | None = None) -> bool:
    """One arm's fold files into oos_<arm>.csv and summary_<arm>.json. False if
    the arm has no files here (a base served from the cache)."""
    import pandas as pd
    csvs = _fold_files(src, arm, "oos", "csv")
    if not csvs:
        return False
    got = [k for k, _ in csvs]
    if expect is not None and sorted(got) != sorted(expect):
        raise SystemExit(f"{arm}: folds {got} gathered, {sorted(expect)} planned")
    Path(out).mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p, dtype={"race_time": str}) for _, p in csvs], ignore_index=True)
    df.to_csv(os.path.join(out, f"oos_{arm}.csv"), index=False)
    per_fold = [json.loads(Path(p).read_text()) for _, p in _fold_files(src, arm, "summary", "json")]
    merged = {
        "arm": arm, "folds": got, "n_predictions": int(len(df)),
        "fold_fits": [f for s in per_fold for f in s.get("fold_fits", [])],
        "train_config": per_fold[0].get("train_config") if per_fold else None,
        "recipe": per_fold[0].get("recipe") if per_fold else None,
        "n_features_used": per_fold[0].get("n_features_used") if per_fold else None,
        "missing_features": per_fold[0].get("missing_features") if per_fold else None,
        "elapsed_seconds": [s.get("elapsed_seconds") for s in per_fold],
        "cpu": sorted({s.get("cpu") for s in per_fold if s.get("cpu")}),
        "per_fold": per_fold,
    }
    Path(out, f"summary_{arm}.json").write_text(json.dumps(merged, indent=2, default=str))
    print(f"{arm}: {len(df):,} predictions from folds {got}")
    return True


def cmd_gather(args) -> None:
    expect = json.loads(args.folds) if args.folds else None
    for arm in all_arms(load_config(args.config)):
        gather(args.src, OUT_DIR, arm, expect)


def replicate_report(a_path: str, b_path: str) -> str:
    """How far two fits of the same arm are apart: identical, or by how much."""
    import numpy as np
    import pandas as pd
    key = ["race_date", "race_time", "track", "horse_name"]
    a = pd.read_csv(a_path, dtype={"race_time": str})
    b = pd.read_csv(b_path, dtype={"race_time": str})
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    d = np.abs(np.log(m["predicted_bfsp_a"]) - np.log(m["predicted_bfsp_b"]))
    same = float((d == 0).mean()) if len(m) else float("nan")
    return (f"{len(m):,} runners paired; identical predictions on {same:.1%}; "
            f"median |log ratio| {float(np.median(d)) if len(m) else float('nan'):.6f}, "
            f"max {float(d.max()) if len(m) else float('nan'):.6f}")


def _note(cfg: dict, v: dict, base_cached: bool) -> str:
    bb, bwh = cfg.get("bfsp_base_blocks", ""), cfg.get("bfsp_base_withhold", "")
    base = f"production features{' + ' + bb if bb else ''}{' without ' + bwh if bwh else ''}"
    return (f"{cfg.get('tag', '')} / {v['name']}: {base} vs the same"
            f"{' + ' + v['blocks'] if v['blocks'] else ''}{' without ' + v['drop'] if v['drop'] else ''}"
            f"{' without ' + v['withhold'] if v['withhold'] else ''}{' with ' + v['args'] if v['args'] else ''}; "
            f"recipe {recipe(cfg)}{'; base from the cache' if base_cached else ''}")


def summary_table(rows: list[tuple[str, dict]]) -> str:
    """One line per variant: what the decision rule turns on."""
    L = ["| variant | paired error (90% CI) | rank 1 | Brier skill vs base (90% CI) | concordance | decision |",
         "|---|---|---|---|---|---|"]
    for name, j in rows:
        r1 = j.get("rank1_delta")
        L.append(f"| {name} | {j['delta']:+.4f} ({j['delta_ci'][0]:+.4f} to {j['delta_ci'][1]:+.4f}) | "
                 f"{'' if r1 is None else f'{r1:+.4f}'} | {j['brier_skill']:+.5f} ({j['brier_skill_ci'][0]:+.5f} to "
                 f"{j['brier_skill_ci'][1]:+.5f}) | {j['concordance']:+.5f} | "
                 f"{'**replaces the base**' if j['replaces'] else 'base stands'} |")
    return "\n".join(L)


def cmd_compare(args) -> None:
    cfg = load_config(args.config)
    out = Path(OUT_DIR)
    base = out / "oos_base.csv"
    if not base.exists():
        raise SystemExit("no base predictions: neither fitted here nor restored from the cache")
    cached = args.base_cached == "true"
    rows, sections = [], []
    for v in variants(cfg):
        name = v["name"]
        md, js = out / f"compare_{name}.md", out / f"compare_{name}.json"
        subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", str(base),
                        "--variant", str(out / f"oos_{name}.csv"), "--out", str(md), "--json-out", str(js),
                        "--note", _note(cfg, v, cached)], check=True, cwd=REPO)
        rows.append((name, json.loads(js.read_text())))
        sections.append(md.read_text())
    text = (f"# {cfg.get('tag', '')}: {len(rows)} variant(s) against the base, recipe {recipe(cfg)}\n\n"
            + summary_table(rows) + "\n\n" + "\n\n".join(sections))
    rep = out / "oos_base_rep.csv"
    if rep.exists():
        subprocess.run([sys.executable, "scripts/compare_oos_runs.py", "--base", str(base),
                        "--variant", str(rep), "--out", str(out / "replicate.md"),
                        "--note", "replicate: the base fitted twice, in separate jobs"],
                       check=True, cwd=REPO)
        text += ("\n\n## Replicate: the base fitted twice\n\n" + replicate_report(str(base), str(rep))
                 + "\n\n" + (out / "replicate.md").read_text())
    for arm in ["base"] + [v["name"] for v in variants(cfg)]:
        res = subprocess.run([sys.executable, "scripts/clv_betfair.py", "--predictions",
                              str(out / f"oos_{arm}.csv"), "--db", args.db, "--until", EVAL_UNTIL],
                             cwd=REPO, capture_output=True, text=True)
        body = res.stdout + (res.stderr[-2000:] if res.returncode else "")
        (out / f"clv_{arm}.txt").write_text(body)
        text += f"\n\n### Early-price trade on Betfair's morning prices: {arm}\n```\n{body}\n```\n"
    (out / "compare.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="research/loop.json")
    ap.add_argument("--db", default="horse_racing.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    a = sub.add_parser("arms")
    a.add_argument("--base-hit", default="false")
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True)
    f.add_argument("--fold", required=True, type=int)
    g = sub.add_parser("gather")
    g.add_argument("--src", default="fits")
    g.add_argument("--folds", default=None, help="JSON list of the planned fold indices")
    c = sub.add_parser("compare")
    c.add_argument("--base-cached", default="false")
    args = ap.parse_args()
    {"plan": cmd_plan, "arms": cmd_arms, "fit": cmd_fit, "gather": cmd_gather,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
