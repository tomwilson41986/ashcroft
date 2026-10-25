"""The Professor's record: each tip settled from the trader's own ledger (every tip is a horse the syndicate holds, so
the ledger's `settle` row carries its result, WINNER, LOSER or REMOVED, and the Betfair SP the lay matched at), scored
at 1 point level stakes two ways, as the proofing norm expects (reports/professor_design.md):

- at the advised price, held to the result (what a follower who took the tip got); a non-runner is void;
- at the Betfair SP after 2% commission on winnings, and the CLV (advised / BSP - 1): the claim the model supports.

The withheld tips (the random fifth never published) are settled and reported apart, so the published impact can be
measured against them. A tip whose race has no settle row yet stays `pending`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from professor import store
from professor.candidates import read_ledger

COMMISSION = 0.02


def ledger_results(day: str) -> pd.DataFrame:
    """The ledger's settle rows of the day, the last a horse: result and BSP by market and selection."""
    led = read_ledger(day)
    if led is None:
        return pd.DataFrame(columns=["market_id", "selection_id", "result", "bsp"])
    st = led[led["event"] == "settle"].sort_values("t")
    st = st.drop_duplicates(["market_id", "selection_id"], keep="last")
    out = st[["market_id", "selection_id", "result", "bsp"]].copy()
    out["market_id"] = out["market_id"].astype(str)
    out["selection_id"] = pd.to_numeric(out["selection_id"], errors="coerce").astype("Int64")
    return out


def settle_tips(tips: pd.DataFrame, results: pd.DataFrame, commission: float = COMMISSION) -> pd.DataFrame:
    """Every tip with its result and its points at the advised price and at the BSP."""
    t = tips.copy()
    t["market_id"] = t["market_id"].astype(str)
    t["selection_id"] = pd.to_numeric(t["selection_id"], errors="coerce").astype("Int64")
    t = t.merge(results, on=["market_id", "selection_id"], how="left", suffixes=("", "_ledger"))
    res = t["result"].fillna("PENDING").astype(str).str.upper()
    t["result"] = res.where(res.isin(["WINNER", "LOSER", "REMOVED"]), "PENDING")
    adv = pd.to_numeric(t["advised"], errors="coerce")
    bsp = pd.to_numeric(t["bsp"], errors="coerce")
    won, lost, void = t["result"].eq("WINNER"), t["result"].eq("LOSER"), t["result"].isin(["REMOVED", "PENDING"])
    t["pts_advised"] = np.where(won, adv - 1.0, np.where(lost, -1.0, 0.0))
    t["pts_bsp"] = np.where(won & (bsp > 1), (bsp - 1.0) * (1 - commission), np.where(lost, -1.0, 0.0))
    t["clv"] = np.where(~void & (bsp > 1), adv / bsp - 1.0, np.nan)
    t["staked"] = np.where(void, 0.0, 1.0)
    return t


def summary(settled: pd.DataFrame) -> dict:
    """The record of a set of settled tips: published and withheld apart."""
    out = {}
    for name, part in (("published", settled[~settled["withheld"].astype(bool)]),
                       ("withheld", settled[settled["withheld"].astype(bool)])):
        run = part[part["staked"] > 0]
        out[name] = {
            "tips": int(len(part)), "settled": int(len(run)), "winners": int(run["result"].eq("WINNER").sum()),
            "non_runners": int(part["result"].eq("REMOVED").sum()), "pending": int(part["result"].eq("PENDING").sum()),
            "pts_advised": round(float(run["pts_advised"].sum()), 2),
            "roi_advised": round(float(run["pts_advised"].sum() / len(run)), 4) if len(run) else None,
            "pts_bsp": round(float(run["pts_bsp"].sum()), 2),
            "roi_bsp": round(float(run["pts_bsp"].sum() / len(run)), 4) if len(run) else None,
            "mean_clv": round(float(run["clv"].mean()), 4) if run["clv"].notna().any() else None,
        }
    return out


def settle_file(path: str | Path, day: str | None = None) -> tuple[pd.DataFrame, dict]:
    """A tips.json (a dry run's, kept in reports/, or the day's in S3) settled against its day's ledger."""
    tips = pd.DataFrame(json.loads(Path(path).read_text()))
    if tips.empty:
        return tips, {}
    day = day or str(tips["day"].iloc[0])
    settled = settle_tips(tips, ledger_results(day))
    return settled, summary(settled)


COLS = ["day", "rank", "tier", "withheld", "off_uk", "venue", "runner", "advised", "price_source", "advised_exchange",
        "exp_bsp", "p_win", "overlay", "result", "bsp", "pts_advised", "pts_bsp", "clv"]


def report(settled: pd.DataFrame, summ: dict) -> str:
    lines = []
    for r in settled.sort_values(["day", "withheld", "rank"]).to_dict("records"):
        bsp = f"{r['bsp']:.2f}" if r.get("bsp") == r.get("bsp") and r.get("bsp") else "-"
        clv = f"{r['clv']:+.0%}" if r.get("clv") == r.get("clv") and r.get("clv") is not None else "-"
        lines.append(f"  {r['day']} {r['off_uk']} {r['venue']} {r['runner']} [{r['tier']}]: advised {r['advised']:.2f}, "
                     f"{r['result']}, BSP {bsp}, CLV {clv}, {r['pts_advised']:+.2f} pts at the advised price, "
                     f"{r['pts_bsp']:+.2f} at the BSP")
    for name, s in summ.items():
        if s.get("tips"):
            lines.append(f"  {name}: {s['settled']} settled ({s['winners']} won, {s['non_runners']} non-runners, "
                         f"{s['pending']} pending): {s['pts_advised']:+.2f} pts at the advised price, "
                         f"{s['pts_bsp']:+.2f} at the BSP, mean CLV {s['mean_clv'] if s['mean_clv'] is None else format(s['mean_clv'], '+.1%')}")
    return "\n".join(lines)


def write(day: str, settled: pd.DataFrame, summ: dict) -> None:
    out = store.out_dir(day)
    store.write_csv(store.key(day, "settled.csv"), settled[[c for c in COLS if c in settled.columns]],
                    out / "settled.csv")
    store.write_json(store.key(day, "settled_summary.json"), summ, out / "settled_summary.json")
