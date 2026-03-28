"""
LightGBM model for return prediction with walk-forward evaluation.
"""
import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
)

logger = logging.getLogger(__name__)

# Paper-aligned default hyperparameters
DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_child_samples": 20,
    "n_estimators": 200,
    "verbose": -1,
    "seed": 42,
}

FEATURE_COLS = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "volatility_30d"]


def train_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Optional[dict] = None,
) -> tuple[np.ndarray, lgb.LGBMRegressor]:
    """Train LightGBM and return predictions on test set."""
    p = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators", 200)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **p)
    model.fit(
        train_df[FEATURE_COLS],
        train_df["target_return"],
        eval_set=[(test_df[FEATURE_COLS], test_df["target_return"])],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    preds = model.predict(test_df[FEATURE_COLS])
    return preds, model


def walk_forward_lgbm(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    config: Optional[BacktestConfig] = None,
) -> tuple[list[BacktestResult], dict]:
    """Run walk-forward evaluation with LightGBM."""
    config = config or BacktestConfig(n_splits=5, min_train_size=60)
    validator = WalkForwardValidator(config)

    # Sort by date, group by month for cross-sectional prediction
    dates = df.index.unique().sort_values()
    results = []

    for window_idx, (train_idx, test_idx) in enumerate(validator.split(dates.to_frame())):
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]

        train_data = df[df.index.isin(train_dates)].copy()
        test_data = df[df.index.isin(test_dates)].copy()

        if len(train_data) < 100 or len(test_data) < 10:
            continue

        # Standardize features using train-only statistics
        means = train_data[FEATURE_COLS].mean()
        stds = train_data[FEATURE_COLS].std().replace(0, 1)
        train_data[FEATURE_COLS] = (train_data[FEATURE_COLS] - means) / stds
        test_data[FEATURE_COLS] = (test_data[FEATURE_COLS] - means) / stds

        preds, _ = train_predict_lgbm(train_data, test_data, params)
        test_data = test_data.copy()
        test_data["prediction"] = preds

        # Form long-short portfolio per month
        monthly_returns = []
        for date, group in test_data.groupby(level=0):
            if len(group) < 20:
                continue
            n_decile = max(1, len(group) // 10)
            sorted_g = group.sort_values("prediction")
            long_ret = sorted_g.tail(n_decile)["target_return"].mean()
            short_ret = sorted_g.head(n_decile)["target_return"].mean()
            monthly_returns.append(long_ret - short_ret)

        if not monthly_returns:
            continue

        ret_series = pd.Series(monthly_returns)
        positions = pd.Series([1.0] * len(ret_series))
        net_returns = calculate_costs(ret_series, positions, config)

        gross_metrics = compute_metrics(ret_series, periods_per_year=12)
        net_metrics = compute_metrics(net_returns, periods_per_year=12)

        result = BacktestResult(
            window=window_idx,
            train_start=str(train_dates.min().date()),
            train_end=str(train_dates.max().date()),
            test_start=str(test_dates.min().date()),
            test_end=str(test_dates.max().date()),
            gross_sharpe=gross_metrics["sharpeRatio"],
            net_sharpe=net_metrics["sharpeRatio"],
            annual_return=net_metrics["annualReturn"],
            max_drawdown=net_metrics["maxDrawdown"],
            total_trades=len(monthly_returns) * 2,
            hit_rate=net_metrics["hitRate"],
            pnl_series=net_returns,
        )
        results.append(result)
        logger.info(
            f"Window {window_idx}: train {result.train_start}->{result.train_end}, "
            f"test {result.test_start}->{result.test_end}, "
            f"net_sharpe={result.net_sharpe:.4f}"
        )

    # Aggregate metrics
    if results:
        avg_metrics = {
            "avg_net_sharpe": np.mean([r.net_sharpe for r in results]),
            "avg_gross_sharpe": np.mean([r.gross_sharpe for r in results]),
            "avg_hit_rate": np.mean([r.hit_rate for r in results]),
            "positive_windows": sum(1 for r in results if r.net_sharpe > 0),
            "total_windows": len(results),
        }
    else:
        avg_metrics = {"avg_net_sharpe": 0.0, "avg_gross_sharpe": 0.0,
                       "avg_hit_rate": 0.0, "positive_windows": 0, "total_windows": 0}

    return results, avg_metrics
