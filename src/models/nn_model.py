"""
Feed-forward Neural Network model for return prediction with walk-forward evaluation.

Architecture: MLP with 3-5 hidden layers, ReLU activations, Dropout.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "volatility_30d"]

# Paper-aligned default hyperparameters
DEFAULT_NN_PARAMS = {
    "hidden_dims": [64, 32, 16],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 50,
    "patience": 10,
}


class ReturnPredictor(nn.Module):
    """Multi-layer perceptron for return prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_predict_nn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Train neural network and return predictions on test set."""
    p = {**DEFAULT_NN_PARAMS, **(params or {})}

    device = torch.device("cpu")

    X_train = torch.FloatTensor(train_df[FEATURE_COLS].values.copy()).to(device)
    y_train = torch.FloatTensor(train_df["target_return"].values.copy()).to(device)
    X_test = torch.FloatTensor(test_df[FEATURE_COLS].values.copy()).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True)

    model = ReturnPredictor(
        input_dim=len(FEATURE_COLS),
        hidden_dims=p["hidden_dims"],
        dropout=p["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(p["epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= len(X_train)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()

    return preds


def walk_forward_nn(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    config: Optional[BacktestConfig] = None,
) -> tuple[list[BacktestResult], dict]:
    """Run walk-forward evaluation with Neural Network."""
    config = config or BacktestConfig(n_splits=5, min_train_size=60)
    validator = WalkForwardValidator(config)

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

        preds = train_predict_nn(train_data, test_data, params)
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
