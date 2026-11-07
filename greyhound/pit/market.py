"""The market as data: how each dog and each yard has been priced against what happened, and how the exchange has
read the dog against the bookmakers.

From the dog's earlier days: winners over the SP's expected winners (A/E, all runs and the last 10), the same against
the Betfair SP where the price files hold it, the mean log of BSP over SP (the exchange longer or shorter than the
bookmakers), and finishing against the market's order (market position less finishing position, last 5). For the
trainer: A/E against the Betfair SP over the last 365 days (shrunk to 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "mk_"
K = 5.0
SOURCE = "GBGB results (SP, market position); Betfair's greyhound price files (sources/greyhound_prices.py: the BSP)"


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs.copy()
    prices = (ctx or {}).get("prices")
    if "bsp" not in df:
        from greyhound.parity import attach_bsp
        df = attach_bsp(df, prices)
    racing = ~df.is_trial.fillna(False)
    df["_w"] = df.won.where(racing & df.sp_p_norm.notna())
    df["_e"] = df.sp_p_norm.where(racing)
    df["_wb"] = df.won.where(racing & df.bsp_p_norm.notna())
    df["_eb"] = df.bsp_p_norm.where(racing)
    df["_drift"] = np.log(df.bsp.where(df.bsp > 1) / df.sp_decimal.where(df.sp_decimal > 1))
    df["_mvf"] = (df.market_pos - df.position).where(racing & df.market_pos.notna())
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    for n, lab in ((None, "all"), (10, "10")):
        w = core.dog_lagged(df, "_w", n, "sum")
        e = core.dog_lagged(df, "_e", n, "sum")
        out[f"{PREFIX}sp_ae_{lab}"] = (w + 1.0) / (e + 1.0)
        wb = core.dog_lagged(df, "_wb", n, "sum")
        eb = core.dog_lagged(df, "_eb", n, "sum")
        out[f"{PREFIX}bsp_ae_{lab}"] = ((wb + 1.0) / (eb + 1.0)).where(eb.notna())
    out[f"{PREFIX}drift5"] = core.dog_lagged(df, "_drift", 5)
    out[f"{PREFIX}vs_market5"] = core.dog_lagged(df, "_mvf", 5)
    out[f"{PREFIX}last_sp_p"] = core.dog_lagged(df, "_e", 1)
    # the trainer at the time only (core.trainer_at_the_time: GBGB serves the current trainer on past results)
    tr = core.trainer_at_the_time(df, (ctx or {}).get("fetch_lag"))
    t = df.assign(trainer=tr, _wb=df._wb.fillna(0.0).where(tr.notna(), 0.0),
                  _eb=df._eb.fillna(0.0).where(tr.notna(), 0.0),
                  _r=(df.bsp_p_norm.notna() & tr.notna()).astype(float))
    w = core.prior_window(t, ["trainer"], ["_wb", "_eb", "_r"], 365)
    out[f"{PREFIX}tr_bsp_ae365"] = ((w._wb + K) / (w._eb + K)).where((w._r > 0) & tr.notna())
    return out.reset_index(drop=True), {"bsp_rows": int(df.bsp.notna().sum()), "rows": int(len(df))}


COLUMNS = {
    f"{PREFIX}sp_ae_all": "the dog's winners over the SP's expected winners, every earlier race (+1 each side)",
    f"{PREFIX}sp_ae_10": "the same, the last 10 race days",
    f"{PREFIX}bsp_ae_all": "the dog's winners over the Betfair SP's expected winners, every earlier race",
    f"{PREFIX}bsp_ae_10": "the same, the last 10",
    f"{PREFIX}drift5": "mean log(BSP / SP) over the last 5: the exchange longer (+) or shorter than the bookmakers",
    f"{PREFIX}vs_market5": "mean market position less finishing position, last 5 (+: beat the market's order)",
    f"{PREFIX}last_sp_p": "the SP's chance for the dog last time",
    f"{PREFIX}tr_bsp_ae365": "the trainer's winners over the Betfair SP's expected winners, last 365 days (shrunk)",
}
