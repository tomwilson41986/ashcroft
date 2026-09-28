"""Staking for automated trading: level, stake-to-win-a-set-amount, and Kelly, backtested on real prices.

The owner asked which staking to automate: Kelly, or variable staking (stake to win a set amount on
each horse), backing overlays and, where it helps the race as a whole, runners at fair odds or a
little under. This scores each on the out-of-sample forecasts of two models (the candidate pair
served for dry runs, and the complete-careers pair gated with race_xent, the best model found)
against Betfair's real morning prices (the morning volume-weighted price, MORNINGWAP), morning
volumes, BSPs and results, 31 Dec 2025 to 31 Mar 2026 (the locked holdout excluded).

Fills are assumed at the morning price, each stake capped at a tenth of the morning's matched volume
on that runner and at £50, and only where at least £100 was matched in the morning (the rule's
floor). Commission 5% on winnings. Every strategy is settled two ways: held to the result at the
morning price, and traded out at BSP (lay at BSP to level the book: the closing-line value).

The fair probability for the Kelly and value strategies pools the model and the morning market
log-linearly with weight 0.5 on the model (trading/pricing.py; fixed in advance, not fitted here):
the model alone is not a better forecast of the result than the market, and Kelly on its raw
probabilities is known to ruin the bank (STAKING_REPORT.md). One Kelly arm on the raw model's
probabilities is kept to show it again.

Caveat that decides everything: these forecasts are built from result rows, so they may know
race-day facts a 06:00 bettor does not (final field, going, the jockey who rode). The edge has not
been confirmed at bet time (reports/clv_betfair_2026q1.md); the trading bot's paper mode is the
forward test. This query compares strategies with each other; it does not prove the edge.
Read-only.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from research_loop import average_arms  # noqa: E402
from trading import pricing, staking  # noqa: E402

IT96 = (36346975763, "research-iter96-served-968s5xh-kalman-handicap-travel-116")
IT97 = (36358454411, "research-iter97-served-968s5xh-travel-partners-117")
IT102 = (36381991232, "research-iter102-served-pair-complete-careers-123")
IT103 = (36384069890, "research-iter103-served-race-xent-complete-careers-124")
KEY = ["race_date", "race_time", "track", "horse_name"]
SOURCES = {
    "xthub_all3": (*IT96, "oos_cw_dm_xthub_kr_hca_tv.csv"),
    "dml_all3": (*IT97, "oos_cw_dm_xt_dml_kr_hca_tv.csv"),
    "main_h18": (*IT102, "oos_base.csv"),
    "dml_h18": (*IT102, "oos_dml_h18_t21.csv"),
    "rx16_h18": (*IT103, "oos_rx_h18_t21_r16k.csv"),
}
STEP = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0]
OUT = Path("out/staking_backtest")
COMMISSION = 0.05
BANK0 = 1000.0
UNIT = 10.0            # level stake, £
TARGET = 20.0          # stake-to-win target, £ after commission
MAX_STAKE = 50.0       # per bet, £
VOL_SHARE = 0.10       # of the morning's matched volume on the runner
MIN_VOL = 100.0        # the rule's floor
MIN_EDGE = 0.02        # value strategies: at least 2% expected profit per unit on the pooled price
W_MODEL = 0.5          # the model's weight in the pooled price, fixed in advance
UNDERLAY = 0.10        # the owner's rule: the favourite backed down to 10% under fair


def _zip(run: int, artifact: str, cache: dict) -> zipfile.ZipFile:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    if (run, artifact) not in cache:
        r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{run}/artifacts",
                         headers=auth, params={"name": artifact}, timeout=60)
        arts = r.json().get("artifacts", []) if r.ok else []
        if not arts:
            raise SystemExit(f"artifact {artifact} of run {run} not found ({r.status_code})")
        z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
        z.raise_for_status()
        cache[(run, artifact)] = zipfile.ZipFile(io.BytesIO(z.content))
    return cache[(run, artifact)]


def fetch(name: str, cache: dict) -> Path:
    run, artifact, member = SOURCES[name]
    zf = _zip(run, artifact, cache)
    inner = next(n for n in zf.namelist() if n.split("/")[-1] == member)
    path = OUT / f"oos_{name}.csv"
    path.write_bytes(zf.read(inner))
    return path


def gate(pair: Path, rx: Path, weights: list, out: Path) -> Path:
    """race_xent's normalised log price for the pair's leaders, the pair's for the rest (gate_clv.py)."""
    a = pd.read_csv(pair, dtype={"race_time": str})
    b = pd.read_csv(rx, dtype={"race_time": str}).set_index(KEY).reindex(a.set_index(KEY).index)
    race = (a["race_date"].astype(str) + "|" + a["track"].astype(str) + "|" + a["race_time"].astype(str)).to_numpy()
    la, lb = np.log(a["predicted_bfsp"].to_numpy(float)), np.log(b["predicted_bfsp"].to_numpy(float))
    rank = pd.Series(la).groupby(race).rank(method="first").to_numpy()
    wr = np.asarray(weights)[np.minimum(rank, len(weights)).astype(int) - 1]
    p = np.exp(-(wr * lb + (1 - wr) * la))
    pn = p / pd.Series(p).groupby(race).transform("sum").to_numpy()
    o = a.copy()
    o["predicted_win_prob_norm"] = pn
    o["predicted_bfsp"] = 1.0 / pn
    o.to_csv(out, index=False)
    return out


def frame(path: Path) -> pd.DataFrame:
    """The forecasts joined to the morning prices, with the model's, the market's and the pooled chances.

    The chances are taken over every runner with a morning price, so a race's long shots keep their
    share; the £100 morning-volume floor only decides which runners may be backed."""
    from clv_betfair import load
    d = load(str(path), "horse_racing.db", None, "2026-04-01")
    race = d["race"].to_numpy()
    d["p_model"] = pricing.normalise(1.0 / d["predicted_bfsp"].to_numpy(float), race)
    d["p_mkt"] = pricing.market_probabilities(d["morningwap"].to_numpy(float), race)
    d["p_pool"] = pricing.pooled_probabilities(d["p_model"], d["p_mkt"], race, W_MODEL)
    d["bettable"] = d["morning_vol"] >= MIN_VOL
    d["cap"] = np.where(d["bettable"], np.minimum(MAX_STAKE, VOL_SHARE * d["morning_vol"].to_numpy(float)), 0.0)
    d["mins"] = d["mins"].fillna(0)
    return d.sort_values(["race_date", "mins", "race"]).reset_index(drop=True)


def strategies(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, bool]]:
    """Each strategy's stakes on one race before the caps: (£, False), or (fractions of the bank, True)."""
    price = g["morningwap"].to_numpy(float)
    ok = g["bettable"].to_numpy(bool)
    kprice = np.where(ok, price, np.nan)             # race Kelly counts every runner, backs only these
    rule = ok & (g["pred_move"].to_numpy(float) >= 0.2)
    p_pool = g["p_pool"].to_numpy(float)
    ev = staking.edge(p_pool, price, COMMISSION)
    value = ok & (ev >= MIN_EDGE)
    fav = np.zeros(len(g), bool)
    fav[int(np.argmax(p_pool))] = True
    hedge = value.any() & fav & ok & ~value & (ev >= -UNDERLAY)
    fw = staking.fixed_win_stakes(price, TARGET, COMMISSION)
    owner = np.where(value | hedge, fw, 0.0)
    # an owner's race must expect a profit as a whole, or its hedge is dropped
    if hedge.any() and staking.race_expectation(p_pool, price, owner, COMMISSION) <= 0:
        owner = np.where(value, fw, 0.0)
    return {
        "rule_level": (np.where(rule, UNIT, 0.0), False),
        "rule_to_win": (np.where(rule, fw, 0.0), False),
        "value_level": (np.where(value, UNIT, 0.0), False),
        "value_to_win": (np.where(value, fw, 0.0), False),
        "owner_to_win": (owner, False),
        "kelly_single_q": (np.where(ok, staking.single_kelly(p_pool, price, COMMISSION, 0.25), 0.0), True),
        "kelly_race_q": (staking.race_kelly(p_pool, kprice, COMMISSION, 0.25), True),
        "kelly_race_h": (staking.race_kelly(p_pool, kprice, COMMISSION, 0.5), True),
        "kelly_race_q_raw": (staking.race_kelly(g["p_model"].to_numpy(float), kprice, COMMISSION, 0.25), True),
    }


def simulate(d: pd.DataFrame) -> pd.DataFrame:
    """Every strategy's bets, race by race in time order; each Kelly bank compounds on its own."""
    banks: dict = {}
    rows = []
    for race, g in d.groupby("race", sort=False):
        plan = strategies(g)
        cap = g["cap"].to_numpy(float)
        for settle in ("hold", "trade"):
            pay = (g["hold_morning"] if settle == "hold" else g["net"]).to_numpy(float)
            for name, (stakes, is_fraction) in plan.items():
                if is_fraction:
                    stakes = stakes * banks.setdefault((name, settle), BANK0)
                stakes = np.minimum(np.nan_to_num(stakes), cap)
                stakes = np.where(stakes >= 2.0, stakes, 0.0)              # Betfair's minimum back stake
                if stakes.sum() <= 0:
                    continue
                profit = float(np.sum(stakes * pay))
                if is_fraction:
                    banks[(name, settle)] = max(banks[(name, settle)] + profit, 0.0)
                rows.append({"strategy": name, "settle": settle, "race": race,
                             "race_date": g["race_date"].iloc[0], "bets": int((stakes > 0).sum()),
                             "turnover": float(stakes.sum()), "profit": profit})
    return pd.DataFrame(rows)


def summary(r: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for (name, settle), g in r.groupby(["strategy", "settle"], sort=False):
        t, p = g["turnover"].to_numpy(), g["profit"].to_numpy()
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        roi = p[idx].sum(1) / t[idx].sum(1)
        bank = BANK0 + np.cumsum(p)
        peak = np.maximum.accumulate(np.r_[BANK0, bank])[1:]
        dd = float(np.max(np.minimum((peak - bank) / peak, 1.0))) if len(bank) else 0.0     # 100% = ruined
        losing = (p < 0).astype(int)
        run = best = 0
        for x in losing:
            run = run + 1 if x else 0
            best = max(best, run)
        daily = g.groupby("race_date")["profit"].sum()
        out.append({"strategy": name, "settle": settle, "races": len(g), "bets": int(g["bets"].sum()),
                    "turnover": round(t.sum()), "profit": round(p.sum()), "roi_%": round(100 * p.sum() / t.sum(), 2),
                    "roi_lo": round(100 * np.percentile(roi, 5), 2), "roi_hi": round(100 * np.percentile(roi, 95), 2),
                    "final_bank": round(bank[-1]), "max_drawdown_%": round(100 * dd, 1),
                    "longest_losing_races": best, "losing_days_%": round(100 * (daily < 0).mean(), 1),
                    "worst_day": round(daily.min())})
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    files = {n: fetch(n, cache) for n in SOURCES}
    average_arms([str(files["xthub_all3"]), str(files["dml_all3"])], str(OUT / "oos_pair97.csv"))
    average_arms([str(files["main_h18"]), str(files["dml_h18"])], str(OUT / "oos_pair_h18.csv"))
    gate(OUT / "oos_pair_h18.csv", files["rx16_h18"], STEP, OUT / "oos_gate_step_h18.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for model in ("gate_step_h18", "pair97"):
        d = frame(OUT / f"oos_{model}.csv")
        print(f"\n===== {model}: {d['race'].nunique():,} races, {len(d):,} runners with >= £{MIN_VOL:.0f} "
              f"matched in the morning, {d['race_date'].min()} to {d['race_date'].max()} =====")
        r = simulate(d)
        r.to_csv(OUT / f"races_{model}.csv", index=False)
        s = summary(r)
        print(s.to_string(index=False))
        binding = (d["cap"] < MAX_STAKE).mean()
        print(f"\nstake cap: a tenth of the morning volume is below £{MAX_STAKE:.0f} on {100 * binding:.0f}% of "
              f"runners (median morning volume £{d['morning_vol'].median():,.0f})")


if __name__ == "__main__":
    main()
