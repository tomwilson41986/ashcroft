"""
Financial-style feature engineering for horse racing.

Inspired by quantitative finance indicators (RSI, Bollinger Bands, MACD,
Sharpe ratio), adapted for racing form analysis. These capture momentum,
mean-reversion, and risk-adjusted performance signals that complement
the existing custom metrics.
"""

import numpy as np
import pandas as pd


# Feature column names for integration with train_bfsp.py
FINANCIAL_FEATURES = [
    "form_rsi_5",
    "form_rsi_10",
    "form_z_score",
    "nfp_macd",
    "nfp_macd_signal",
    "mean_reversion_score",
    "form_sharpe_3",
    "form_sharpe_5",
    "nfp_acceleration",
    "prize_momentum",
    "class_momentum",
    "win_density_30d",
    "win_density_60d",
    "layoff_adjusted_form",
    "or_momentum",
    "nfp_skew_5",
    "drawdown_from_peak_nfp",
]

FINANCIAL_RANK_FEATURES = [
    "rFormRSI5",
    "rFormZScore",
    "rNFPMACD",
    "rMeanReversion",
    "rFormSharpe3",
    "rNFPAccel",
    "rPrizeMomentum",
    "rWinDensity30d",
    "rLayoffAdjForm",
    "rORMomentum",
    "rDrawdownPeakNFP",
]


def calculate_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all financial-style features.

    Requires prior calculation of NFP, career metrics, and race metadata.
    Must be called after core custom metrics are computed.

    Args:
        df: DataFrame with race data including NFP, career stats, etc.
            Must be sorted by race_date, race_time.

    Returns:
        DataFrame with financial feature columns added.
    """
    df = df.copy()

    # Group by horse for time-series features
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    # Pre-compute horse groups with NFP history
    horse_groups = df.groupby("horse_name")

    # --- RSI (Relative Strength Index) on NFP ---
    df = _calc_form_rsi(df, horse_groups)

    # --- Z-Score (Bollinger Band-style deviation) ---
    df = _calc_form_z_score(df, horse_groups)

    # --- MACD on NFP ---
    df = _calc_nfp_macd(df, horse_groups)

    # --- Mean Reversion Score ---
    df = _calc_mean_reversion(df, horse_groups)

    # --- Sharpe-like Form Ratio ---
    df = _calc_form_sharpe(df, horse_groups)

    # --- NFP Acceleration (2nd derivative) ---
    df = _calc_nfp_acceleration(df, horse_groups)

    # --- Prize Money Momentum ---
    df = _calc_prize_momentum(df, horse_groups)

    # --- Class Momentum ---
    df = _calc_class_momentum(df, horse_groups)

    # --- Win Density (rolling win rate by time) ---
    df = _calc_win_density(df)

    # --- Layoff-Adjusted Form ---
    df = _calc_layoff_adjusted_form(df)

    # --- OR Momentum ---
    df = _calc_or_momentum(df, horse_groups)

    # --- NFP Skew (asymmetry of recent form) ---
    df = _calc_nfp_skew(df, horse_groups)

    # --- Drawdown from Peak NFP ---
    df = _calc_drawdown_from_peak(df, horse_groups)

    # --- Within-race ranks for financial features ---
    df = _calc_financial_ranks(df)

    return df


def _calc_form_rsi(df: pd.DataFrame, groups) -> pd.DataFrame:
    """RSI-like indicator on NFP changes. 0-100 scale, >50 = improving."""
    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["form_rsi_5"] = np.nan
        df["form_rsi_10"] = np.nan
        return df

    for window in [5, 10]:
        col = f"form_rsi_{window}"
        rsi_vals = np.full(len(df), np.nan)

        for name, idx in groups.groups.items():
            if len(idx) < 3:
                continue
            nfp = df.loc[idx, nfp_col].values
            changes = np.diff(nfp)

            for i in range(1, len(idx)):
                lookback = changes[max(0, i - window):i]
                if len(lookback) < 2:
                    continue
                gains = lookback[lookback > 0]
                losses = -lookback[lookback < 0]
                avg_gain = gains.mean() if len(gains) > 0 else 0.0
                avg_loss = losses.mean() if len(losses) > 0 else 0.0
                denom = avg_gain + avg_loss
                if denom > 0:
                    rsi_vals[idx[i]] = 100.0 * avg_gain / denom
                else:
                    rsi_vals[idx[i]] = 50.0

        df[col] = rsi_vals

    return df


def _calc_form_z_score(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Z-score of recent form vs career mean (Bollinger Band-style)."""
    z_vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["form_z_score"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 5:
            continue
        nfp = df.loc[idx, nfp_col].values
        for i in range(4, len(idx)):
            career = nfp[:i]
            career_mean = career.mean()
            career_std = career.std()
            if career_std > 0.01:
                recent_mean = nfp[max(0, i - 3):i].mean()
                z_vals[idx[i]] = (recent_mean - career_mean) / career_std

    df["form_z_score"] = z_vals
    return df


def _calc_nfp_macd(df: pd.DataFrame, groups) -> pd.DataFrame:
    """MACD-style signal: fast EWM - slow EWM of NFP."""
    macd_vals = np.full(len(df), np.nan)
    signal_vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["nfp_macd"] = np.nan
        df["nfp_macd_signal"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 5:
            continue
        nfp_series = pd.Series(df.loc[idx, nfp_col].values)
        fast = nfp_series.ewm(span=3, min_periods=2).mean()
        slow = nfp_series.ewm(span=10, min_periods=3).mean()
        macd = fast - slow
        signal = macd.ewm(span=3, min_periods=2).mean()

        for i in range(len(idx)):
            macd_vals[idx[i]] = macd.iloc[i]
            signal_vals[idx[i]] = signal.iloc[i]

    df["nfp_macd"] = macd_vals
    df["nfp_macd_signal"] = signal_vals
    return df


def _calc_mean_reversion(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Mean reversion score: negative deviation from career mean."""
    rev_vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["mean_reversion_score"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 3:
            continue
        nfp = df.loc[idx, nfp_col].values
        for i in range(2, len(idx)):
            career = nfp[:i]
            career_mean = career.mean()
            career_std = career.std()
            if career_std > 0.01:
                # Negative: after good run, expect reversion down
                # Positive: after bad run, expect reversion up
                rev_vals[idx[i]] = -(nfp[i - 1] - career_mean) / career_std

    df["mean_reversion_score"] = rev_vals
    return df


def _calc_form_sharpe(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Sharpe-like ratio: mean NFP / std NFP for recent runs."""
    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["form_sharpe_3"] = np.nan
        df["form_sharpe_5"] = np.nan
        return df

    for window in [3, 5]:
        col = f"form_sharpe_{window}"
        vals = np.full(len(df), np.nan)

        for name, idx in groups.groups.items():
            if len(idx) < window + 1:
                continue
            nfp = df.loc[idx, nfp_col].values
            for i in range(window, len(idx)):
                recent = nfp[i - window:i]
                std = recent.std()
                if std > 0.01:
                    vals[idx[i]] = recent.mean() / std

        df[col] = vals

    return df


def _calc_nfp_acceleration(df: pd.DataFrame, groups) -> pd.DataFrame:
    """2nd derivative of form: rate of change of form change."""
    vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["nfp_acceleration"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 4:
            continue
        nfp = df.loc[idx, nfp_col].values
        for i in range(3, len(idx)):
            # change_recent - change_prior (2nd derivative)
            change_1 = nfp[i - 1] - nfp[i - 2]
            change_2 = nfp[i - 2] - nfp[i - 3]
            vals[idx[i]] = change_1 - change_2

    df["nfp_acceleration"] = vals
    return df


def _calc_prize_momentum(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Prize money trend: ratio of short-term to long-term avg."""
    vals = np.full(len(df), np.nan)

    pmw_col = None
    for c in ["PMW", "preracehorsecareerPMW", "prize_money"]:
        if c in df.columns:
            pmw_col = c
            break

    if pmw_col is None:
        df["prize_momentum"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 5:
            continue
        pm = pd.Series(df.loc[idx, pmw_col].values.astype(float))
        fast = pm.ewm(span=3, min_periods=2).mean()
        slow = pm.ewm(span=10, min_periods=3).mean()
        for i in range(len(idx)):
            if slow.iloc[i] > 0:
                vals[idx[i]] = fast.iloc[i] / slow.iloc[i]

    df["prize_momentum"] = vals
    return df


def _calc_class_momentum(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Class trajectory: recent avg class - longer avg class."""
    vals = np.full(len(df), np.nan)

    class_col = None
    for c in ["race_class", "race_class_num"]:
        if c in df.columns:
            class_col = c
            break

    if class_col is None:
        df["class_momentum"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 5:
            continue
        cls = df.loc[idx, class_col].values.astype(float)
        for i in range(4, len(idx)):
            recent_3 = cls[max(0, i - 3):i]
            recent_5 = cls[max(0, i - 5):i]
            avg_3 = np.nanmean(recent_3)
            avg_5 = np.nanmean(recent_5)
            vals[idx[i]] = avg_3 - avg_5  # negative = dropping in class

    df["class_momentum"] = vals
    return df


def _calc_win_density(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate by calendar time (30d, 60d windows)."""
    df["race_date_dt"] = pd.to_datetime(df["race_date"])

    for days, col in [(30, "win_density_30d"), (60, "win_density_60d")]:
        vals = np.full(len(df), np.nan)

        for name, grp in df.groupby("horse_name"):
            idx = grp.index.values
            if len(idx) < 2:
                continue
            dates = grp["race_date_dt"].values
            won = grp["won"].values if "won" in grp.columns else (
                grp["placing_numerical"].values == 1
            ).astype(float)

            for i in range(1, len(idx)):
                cutoff = dates[i] - np.timedelta64(days, "D")
                mask = (dates[:i] >= cutoff)
                if mask.sum() > 0:
                    vals[idx[i]] = won[:i][mask].mean()

        df[col] = vals

    if "race_date_dt" in df.columns:
        df.drop(columns=["race_date_dt"], inplace=True, errors="ignore")

    return df


def _calc_layoff_adjusted_form(df: pd.DataFrame) -> pd.DataFrame:
    """Form decayed by time since last run."""
    if "EXP_NFP5" in df.columns and "days_since_lr" in df.columns:
        days = pd.to_numeric(df["days_since_lr"], errors="coerce").fillna(30)
        df["layoff_adjusted_form"] = df["EXP_NFP5"] * np.exp(-0.01 * days)
    elif "LR5NFPtotal" in df.columns and "days_since_lr" in df.columns:
        days = pd.to_numeric(df["days_since_lr"], errors="coerce").fillna(30)
        nfp5 = pd.to_numeric(df["LR5NFPtotal"], errors="coerce").fillna(0)
        df["layoff_adjusted_form"] = (nfp5 / 5.0) * np.exp(-0.01 * days)
    else:
        df["layoff_adjusted_form"] = np.nan

    return df


def _calc_or_momentum(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Official rating trajectory: EWM rate of change."""
    vals = np.full(len(df), np.nan)

    or_col = None
    for c in ["official_rating", "or_num"]:
        if c in df.columns:
            or_col = c
            break

    if or_col is None:
        df["or_momentum"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 3:
            continue
        ors = df.loc[idx, or_col].values.astype(float)
        for i in range(2, len(idx)):
            recent = ors[max(0, i - 3):i]
            if len(recent) >= 2 and not np.all(np.isnan(recent)):
                valid = recent[~np.isnan(recent)]
                if len(valid) >= 2:
                    vals[idx[i]] = valid[-1] - valid[0]

    df["or_momentum"] = vals
    return df


def _calc_nfp_skew(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Skewness of recent NFP distribution (asymmetry of form)."""
    vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["nfp_skew_5"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 6:
            continue
        nfp = df.loc[idx, nfp_col].values
        for i in range(5, len(idx)):
            recent = nfp[i - 5:i]
            std = recent.std()
            if std > 0.01:
                mean = recent.mean()
                skew = np.mean(((recent - mean) / std) ** 3)
                vals[idx[i]] = skew

    df["nfp_skew_5"] = vals
    return df


def _calc_drawdown_from_peak(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Maximum drawdown from peak NFP (how far below best form)."""
    vals = np.full(len(df), np.nan)

    nfp_col = "NFP" if "NFP" in df.columns else None
    if nfp_col is None:
        df["drawdown_from_peak_nfp"] = np.nan
        return df

    for name, idx in groups.groups.items():
        if len(idx) < 2:
            continue
        nfp = df.loc[idx, nfp_col].values
        running_peak = nfp[0]
        for i in range(1, len(idx)):
            if nfp[i - 1] > running_peak:
                running_peak = nfp[i - 1]
            vals[idx[i]] = nfp[i - 1] - running_peak  # negative = below peak

    df["drawdown_from_peak_nfp"] = vals
    return df


def _calc_financial_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Within-race rankings for financial features."""
    rank_map = {
        "form_rsi_5": "rFormRSI5",
        "form_z_score": "rFormZScore",
        "nfp_macd": "rNFPMACD",
        "mean_reversion_score": "rMeanReversion",
        "form_sharpe_3": "rFormSharpe3",
        "nfp_acceleration": "rNFPAccel",
        "prize_momentum": "rPrizeMomentum",
        "win_density_30d": "rWinDensity30d",
        "layoff_adjusted_form": "rLayoffAdjForm",
        "or_momentum": "rORMomentum",
        "drawdown_from_peak_nfp": "rDrawdownPeakNFP",
    }

    raceid_col = "raceid" if "raceid" in df.columns else None
    if raceid_col is None:
        for col, rank_col in rank_map.items():
            df[rank_col] = np.nan
        return df

    for col, rank_col in rank_map.items():
        if col in df.columns:
            df[rank_col] = df.groupby(raceid_col)[col].rank(
                method="min", ascending=False, na_option="bottom"
            )
        else:
            df[rank_col] = np.nan

    return df
