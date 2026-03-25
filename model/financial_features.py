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
    df = df.sort_values(
        ["horse_name", "race_date", "race_time"]
    ).reset_index(drop=True)

    # --- RSI (Relative Strength Index) on NFP ---
    df = _calc_form_rsi(df)

    # --- Z-Score (Bollinger Band-style deviation) ---
    df = _calc_form_z_score(df)

    # --- MACD on NFP ---
    df = _calc_nfp_macd(df)

    # --- Mean Reversion Score ---
    df = _calc_mean_reversion(df)

    # --- Sharpe-like Form Ratio ---
    df = _calc_form_sharpe(df)

    # --- NFP Acceleration (2nd derivative) ---
    df = _calc_nfp_acceleration(df)

    # --- Prize Money Momentum ---
    df = _calc_prize_momentum(df)

    # --- Class Momentum ---
    df = _calc_class_momentum(df)

    # --- Win Density (rolling win rate by time) ---
    df = _calc_win_density(df)

    # --- Layoff-Adjusted Form ---
    df = _calc_layoff_adjusted_form(df)

    # --- OR Momentum ---
    df = _calc_or_momentum(df)

    # --- NFP Skew (asymmetry of recent form) ---
    df = _calc_nfp_skew(df)

    # --- Drawdown from Peak NFP ---
    df = _calc_drawdown_from_peak(df)

    # --- Within-race ranks for financial features ---
    df = _calc_financial_ranks(df)

    return df


def _calc_form_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """RSI-like indicator on NFP changes. 0-100 scale, >50 = improving."""
    if "NFP" not in df.columns:
        df["form_rsi_5"] = np.nan
        df["form_rsi_10"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    # NFP change between consecutive runs
    nfp_change = grp["NFP"].diff()

    gains = nfp_change.clip(lower=0)
    losses = (-nfp_change).clip(lower=0)

    for window in [5, 10]:
        avg_gain = gains.groupby(
            df["horse_name"], sort=False
        ).rolling(window, min_periods=2).mean().droplevel(0).sort_index()
        avg_loss = losses.groupby(
            df["horse_name"], sort=False
        ).rolling(window, min_periods=2).mean().droplevel(0).sort_index()
        denom = avg_gain + avg_loss
        rsi = np.where(denom > 0, 100.0 * avg_gain / denom, 50.0)
        rsi = np.where(avg_gain.isna() | avg_loss.isna(), np.nan, rsi)
        # Shift by 1 to ensure lag safety (use prior data only)
        df[f"form_rsi_{window}"] = pd.Series(rsi, index=df.index)

    return df


def _calc_form_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of recent form vs career mean (Bollinger Band-style)."""
    if "NFP" not in df.columns:
        df["form_z_score"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    # Career mean and std of prior runs
    career_mean = shifted.groupby(
        df["horse_name"], sort=False
    ).expanding(min_periods=4).mean().droplevel(0).sort_index()
    career_std = shifted.groupby(
        df["horse_name"], sort=False
    ).expanding(min_periods=4).std().droplevel(0).sort_index()

    # Recent 3-run mean (lagged)
    recent_mean = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(3, min_periods=3).mean().droplevel(0).sort_index()

    df["form_z_score"] = np.where(
        career_std > 0.01,
        (recent_mean - career_mean) / career_std,
        np.nan,
    )

    return df


def _calc_nfp_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD-style signal: fast EWM - slow EWM of NFP."""
    if "NFP" not in df.columns:
        df["nfp_macd"] = np.nan
        df["nfp_macd_signal"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    fast = shifted.groupby(
        df["horse_name"], sort=False
    ).apply(lambda x: x.ewm(span=3, min_periods=2).mean())
    slow = shifted.groupby(
        df["horse_name"], sort=False
    ).apply(lambda x: x.ewm(span=10, min_periods=3).mean())

    # Flatten multi-index if present
    if isinstance(fast.index, pd.MultiIndex):
        fast = fast.droplevel(0).sort_index()
        slow = slow.droplevel(0).sort_index()

    macd = fast - slow
    signal = macd.groupby(
        df["horse_name"], sort=False
    ).apply(lambda x: x.ewm(span=3, min_periods=2).mean())
    if isinstance(signal.index, pd.MultiIndex):
        signal = signal.droplevel(0).sort_index()

    df["nfp_macd"] = macd
    df["nfp_macd_signal"] = signal

    return df


def _calc_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reversion score: negative deviation from career mean."""
    if "NFP" not in df.columns:
        df["mean_reversion_score"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    career_mean = shifted.groupby(
        df["horse_name"], sort=False
    ).expanding(min_periods=2).mean().droplevel(0).sort_index()
    career_std = shifted.groupby(
        df["horse_name"], sort=False
    ).expanding(min_periods=2).std().droplevel(0).sort_index()

    last_nfp = grp["NFP"].shift(1)

    df["mean_reversion_score"] = np.where(
        career_std > 0.01,
        -(last_nfp - career_mean) / career_std,
        np.nan,
    )

    return df


def _calc_form_sharpe(df: pd.DataFrame) -> pd.DataFrame:
    """Sharpe-like ratio: mean NFP / std NFP for recent runs."""
    if "NFP" not in df.columns:
        df["form_sharpe_3"] = np.nan
        df["form_sharpe_5"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    for window in [3, 5]:
        roll_mean = shifted.groupby(
            df["horse_name"], sort=False
        ).rolling(window, min_periods=window).mean().droplevel(0).sort_index()
        roll_std = shifted.groupby(
            df["horse_name"], sort=False
        ).rolling(window, min_periods=window).std().droplevel(0).sort_index()

        df[f"form_sharpe_{window}"] = np.where(
            roll_std > 0.01,
            roll_mean / roll_std,
            np.nan,
        )

    return df


def _calc_nfp_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """2nd derivative of form: rate of change of form change."""
    if "NFP" not in df.columns:
        df["nfp_acceleration"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    lag1 = grp["NFP"].shift(1)
    lag2 = grp["NFP"].shift(2)
    lag3 = grp["NFP"].shift(3)

    change_1 = lag1 - lag2
    change_2 = lag2 - lag3
    df["nfp_acceleration"] = change_1 - change_2

    return df


def _calc_prize_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Prize money trend: ratio of short-term to long-term avg."""
    pmw_col = None
    for c in ["PMW", "preracehorsecareerPMW", "prize_money"]:
        if c in df.columns:
            pmw_col = c
            break

    if pmw_col is None:
        df["prize_momentum"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp[pmw_col].shift(1)

    fast = shifted.groupby(
        df["horse_name"], sort=False
    ).apply(lambda x: x.ewm(span=3, min_periods=2).mean())
    slow = shifted.groupby(
        df["horse_name"], sort=False
    ).apply(lambda x: x.ewm(span=10, min_periods=3).mean())

    if isinstance(fast.index, pd.MultiIndex):
        fast = fast.droplevel(0).sort_index()
        slow = slow.droplevel(0).sort_index()

    df["prize_momentum"] = np.where(slow > 0, fast / slow, np.nan)

    return df


def _calc_class_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Class trajectory: recent avg class - longer avg class."""
    class_col = None
    for c in ["race_class_num", "race_class"]:
        if c in df.columns:
            class_col = c
            break

    if class_col is None:
        df["class_momentum"] = np.nan
        return df

    # Convert to numeric (handles "Class 4" -> 4 etc.)
    class_numeric = pd.to_numeric(
        df[class_col].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    grp = class_numeric.groupby(df["horse_name"], sort=False)
    shifted = grp.shift(1)

    avg_3 = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(3, min_periods=3).mean().droplevel(0).sort_index()
    avg_5 = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(5, min_periods=4).mean().droplevel(0).sort_index()

    df["class_momentum"] = avg_3 - avg_5

    return df


def _calc_win_density(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate by calendar time (30d, 60d windows)."""
    if "won" not in df.columns:
        df["win_density_30d"] = np.nan
        df["win_density_60d"] = np.nan
        return df

    df["_race_date_dt"] = pd.to_datetime(df["race_date"])

    grp = df.groupby("horse_name", sort=False)
    shifted_won = grp["won"].shift(1)
    shifted_counter = grp["won"].shift(1).notna().astype(float)

    for days, col in [(30, "win_density_30d"), (60, "win_density_60d")]:
        window_str = f"{days}D"
        df_tmp = pd.DataFrame({
            "won_s": shifted_won,
            "cnt_s": shifted_counter,
            "entity": df["horse_name"],
        }, index=df["_race_date_dt"])

        grp_tmp = df_tmp.groupby("entity", sort=False)
        rolling_wins = grp_tmp["won_s"].rolling(
            window_str, min_periods=1
        ).sum().droplevel(0).sort_index()
        rolling_runs = grp_tmp["cnt_s"].rolling(
            window_str, min_periods=1
        ).sum().droplevel(0).sort_index()

        result = rolling_wins.values / np.where(
            rolling_runs.values > 0, rolling_runs.values, np.nan
        )
        no_prior = shifted_counter.isna()
        result[no_prior.values] = np.nan
        df[col] = result

    df.drop(columns=["_race_date_dt"], errors="ignore", inplace=True)

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


def _calc_or_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Official rating trajectory: recent change."""
    or_col = None
    for c in ["official_rating", "or_num"]:
        if c in df.columns:
            or_col = c
            break

    if or_col is None:
        df["or_momentum"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    lag1 = grp[or_col].shift(1)
    lag3 = grp[or_col].shift(3)

    df["or_momentum"] = lag1 - lag3

    return df


def _calc_nfp_skew(df: pd.DataFrame) -> pd.DataFrame:
    """Skewness of recent NFP distribution (asymmetry of form)."""
    if "NFP" not in df.columns:
        df["nfp_skew_5"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    # Use rolling stats to compute skewness vectorized
    roll_mean = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(5, min_periods=5).mean().droplevel(0).sort_index()
    roll_std = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(5, min_periods=5).std().droplevel(0).sort_index()

    # For skewness, we need the third moment — use pandas skew()
    roll_skew = shifted.groupby(
        df["horse_name"], sort=False
    ).rolling(5, min_periods=5).skew().droplevel(0).sort_index()

    df["nfp_skew_5"] = np.where(roll_std > 0.01, roll_skew, np.nan)

    return df


def _calc_drawdown_from_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Maximum drawdown from peak NFP (how far below best form)."""
    if "NFP" not in df.columns:
        df["drawdown_from_peak_nfp"] = np.nan
        return df

    grp = df.groupby("horse_name", sort=False)
    shifted = grp["NFP"].shift(1)

    # Running peak of lagged NFP
    running_peak = shifted.groupby(
        df["horse_name"], sort=False
    ).cummax()

    df["drawdown_from_peak_nfp"] = shifted - running_peak

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
