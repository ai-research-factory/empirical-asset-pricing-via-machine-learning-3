"""
Feature engineering for empirical asset pricing.

Resamples daily data to monthly frequency and computes:
- Target: next-month excess return
- Momentum features: past 1, 3, 6, 12 month returns
- Volatility: past 30-day realized volatility
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resample_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV data to monthly frequency per ticker.

    Uses last trading day of each month for close price,
    and aggregates volume.
    """
    frames = []
    for ticker, group in daily_df.groupby("ticker"):
        monthly = group.resample("ME").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        monthly = monthly.dropna(subset=["close"])
        monthly["ticker"] = ticker
        frames.append(monthly)

    result = pd.concat(frames)
    result = result.sort_index()
    logger.info(
        f"Resampled to monthly: {result['ticker'].nunique()} tickers, "
        f"{len(result)} total rows"
    )
    return result


def compute_monthly_return(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly return from close prices."""
    df = monthly_df.copy()
    df["monthly_return"] = df.groupby("ticker")["close"].pct_change()
    return df


def compute_target_return(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target: next-month return (shifted forward by 1 month)."""
    df = df.copy()
    df["target_return"] = df.groupby("ticker")["monthly_return"].shift(-1)
    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum features: cumulative returns over past N months.

    Features:
    - mom_1m: past 1-month return
    - mom_3m: past 3-month cumulative return
    - mom_6m: past 6-month cumulative return
    - mom_12m: past 12-month cumulative return
    """
    df = df.copy()
    for ticker, group in df.groupby("ticker"):
        ret = group["monthly_return"]
        df.loc[group.index, "mom_1m"] = ret
        df.loc[group.index, "mom_3m"] = (1 + ret).rolling(3).apply(np.prod, raw=True) - 1
        df.loc[group.index, "mom_6m"] = (1 + ret).rolling(6).apply(np.prod, raw=True) - 1
        df.loc[group.index, "mom_12m"] = (1 + ret).rolling(12).apply(np.prod, raw=True) - 1
    return df


def compute_volatility(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trailing 30-day realized volatility from daily returns,
    mapped to the monthly DataFrame.
    """
    df = monthly_df.copy()

    for ticker, daily_group in daily_df.groupby("ticker"):
        daily_ret = daily_group["close"].pct_change()
        # Rolling 30-day standard deviation of daily returns, annualized
        vol_30d = daily_ret.rolling(30, min_periods=20).std() * np.sqrt(252)
        # Resample to monthly: take last value of each month
        monthly_vol = vol_30d.resample("ME").last()
        mask = df["ticker"] == ticker
        ticker_months = df.loc[mask].index
        common_idx = ticker_months.intersection(monthly_vol.index)
        df.loc[df.index.isin(common_idx) & mask, "volatility_30d"] = monthly_vol.loc[common_idx].values

    return df


def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        daily_df: Daily OHLCV DataFrame with 'ticker' column

    Returns:
        Monthly DataFrame with features and target_return
    """
    logger.info("Building features...")

    # Step 1: Resample to monthly
    monthly = resample_to_monthly(daily_df)

    # Step 2: Compute monthly returns
    monthly = compute_monthly_return(monthly)

    # Step 3: Compute target (next-month return)
    monthly = compute_target_return(monthly)

    # Step 4: Momentum features
    monthly = compute_momentum_features(monthly)

    # Step 5: Volatility from daily data
    monthly = compute_volatility(daily_df, monthly)

    # Drop rows where target or all features are NaN
    feature_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "volatility_30d"]
    before = len(monthly)
    monthly = monthly.dropna(subset=["target_return"] + feature_cols)
    logger.info(f"Dropped {before - len(monthly)} rows with NaN features/target")

    logger.info(
        f"Final dataset: {monthly['ticker'].nunique()} tickers, "
        f"{len(monthly)} rows, "
        f"date range: {monthly.index.min()} to {monthly.index.max()}"
    )
    return monthly
