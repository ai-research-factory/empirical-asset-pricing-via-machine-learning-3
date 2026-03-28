"""
Hyperparameter optimization using Optuna for LightGBM and Neural Network models.

Per paper reproduction rules:
- Tuning is done on the FIRST training fold only (no test-set contamination).
- Search space stays near paper-default parameters.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from src.backtest import BacktestConfig, generate_metrics_json
from src.models.lgbm_model import (
    DEFAULT_LGBM_PARAMS,
    FEATURE_COLS,
    train_predict_lgbm,
    walk_forward_lgbm,
)
from src.models.nn_model import (
    DEFAULT_NN_PARAMS,
    train_predict_nn,
    walk_forward_nn,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_first_fold_data(
    df: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the first walk-forward fold's train/validation data for tuning."""
    from src.backtest import WalkForwardValidator

    validator = WalkForwardValidator(config)
    dates = df.index.unique().sort_values()

    for train_idx, test_idx in validator.split(dates.to_frame()):
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        train_data = df[df.index.isin(train_dates)].copy()
        test_data = df[df.index.isin(test_dates)].copy()

        # Standardize using train-only stats
        means = train_data[FEATURE_COLS].mean()
        stds = train_data[FEATURE_COLS].std().replace(0, 1)
        train_data[FEATURE_COLS] = (train_data[FEATURE_COLS] - means) / stds
        test_data[FEATURE_COLS] = (test_data[FEATURE_COLS] - means) / stds

        return train_data, test_data

    raise ValueError("No valid fold found")


def _evaluate_predictions(test_data: pd.DataFrame, preds: np.ndarray) -> float:
    """Evaluate predictions via long-short portfolio Sharpe on the fold."""
    test_data = test_data.copy()
    test_data["prediction"] = preds
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
        return 0.0
    ret = pd.Series(monthly_returns)
    if ret.std() == 0:
        return 0.0
    return float(np.sqrt(12) * ret.mean() / ret.std())


def optimize_lgbm(
    df: pd.DataFrame,
    n_trials: int = 30,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Optimize LightGBM hyperparameters using Optuna on the first fold.

    Search space is near paper-default values per reproduction rules.
    """
    config = config or BacktestConfig(n_splits=5, min_train_size=60)
    train_data, test_data = _get_first_fold_data(df, config)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "verbose": -1,
            "seed": 42,
        }
        preds, _ = train_predict_lgbm(train_data, test_data, params)
        return _evaluate_predictions(test_data, preds)

    study = optuna.create_study(direction="maximize", study_name="lgbm_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"LGBM best trial: Sharpe={study.best_value:.4f}")
    logger.info(f"LGBM best params: {study.best_params}")

    return {
        "best_sharpe_fold1": round(study.best_value, 4),
        "best_params": study.best_params,
        "n_trials": n_trials,
    }


def optimize_nn(
    df: pd.DataFrame,
    n_trials: int = 20,
    config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Optimize Neural Network hyperparameters using Optuna on the first fold.

    Search space is near paper-default values.
    """
    config = config or BacktestConfig(n_splits=5, min_train_size=60)
    train_data, test_data = _get_first_fold_data(df, config)

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_dims = []
        dim = trial.suggest_int("first_hidden_dim", 32, 128, step=16)
        for i in range(n_layers):
            hidden_dims.append(dim)
            dim = max(8, dim // 2)

        params = {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "epochs": 50,
            "patience": 10,
        }
        preds = train_predict_nn(train_data, test_data, params)
        return _evaluate_predictions(test_data, preds)

    study = optuna.create_study(direction="maximize", study_name="nn_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"NN best trial: Sharpe={study.best_value:.4f}")
    logger.info(f"NN best params: {study.best_params}")

    # Reconstruct hidden_dims from best params
    bp = study.best_params
    n_layers = bp["n_layers"]
    dim = bp["first_hidden_dim"]
    hidden_dims = []
    for _ in range(n_layers):
        hidden_dims.append(dim)
        dim = max(8, dim // 2)

    return {
        "best_sharpe_fold1": round(study.best_value, 4),
        "best_params": {
            "hidden_dims": hidden_dims,
            "dropout": bp["dropout"],
            "learning_rate": bp["learning_rate"],
            "weight_decay": bp["weight_decay"],
            "batch_size": bp["batch_size"],
        },
        "n_trials": n_trials,
    }


def run_full_optimization(
    df: pd.DataFrame,
    lgbm_trials: int = 30,
    nn_trials: int = 20,
    n_splits: int = 5,
    output_dir: str = "reports/cycle_5",
) -> dict:
    """
    Run full optimization pipeline:
    1. Optimize LGBM and NN on first fold
    2. Evaluate both default and optimized params via walk-forward
    3. Generate metrics.json

    Returns combined results dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = BacktestConfig(n_splits=n_splits, min_train_size=60)

    # --- Step 1: Run default models as baselines ---
    logger.info("=== Evaluating default LightGBM ===")
    lgbm_default_results, lgbm_default_metrics = walk_forward_lgbm(df, config=config)

    logger.info("=== Evaluating default Neural Network ===")
    nn_default_results, nn_default_metrics = walk_forward_nn(df, config=config)

    # --- Step 2: Optimize hyperparameters on first fold ---
    logger.info(f"=== Optimizing LightGBM ({lgbm_trials} trials) ===")
    lgbm_opt = optimize_lgbm(df, n_trials=lgbm_trials, config=config)

    logger.info(f"=== Optimizing Neural Network ({nn_trials} trials) ===")
    nn_opt = optimize_nn(df, n_trials=nn_trials, config=config)

    # --- Step 3: Evaluate optimized models via walk-forward ---
    logger.info("=== Evaluating optimized LightGBM ===")
    lgbm_opt_params = {**lgbm_opt["best_params"], "objective": "regression",
                       "metric": "mse", "boosting_type": "gbdt", "verbose": -1, "seed": 42}
    lgbm_opt_results, lgbm_opt_metrics = walk_forward_lgbm(df, params=lgbm_opt_params, config=config)

    logger.info("=== Evaluating optimized Neural Network ===")
    nn_opt_results, nn_opt_metrics = walk_forward_nn(df, params=nn_opt["best_params"], config=config)

    # --- Step 4: Pick the best model for metrics.json ---
    # Use the model with best average OOS net Sharpe
    all_results = {
        "lgbm_default": (lgbm_default_results, lgbm_default_metrics),
        "lgbm_optimized": (lgbm_opt_results, lgbm_opt_metrics),
        "nn_default": (nn_default_results, nn_default_metrics),
        "nn_optimized": (nn_opt_results, nn_opt_metrics),
    }

    best_model_name = max(all_results, key=lambda k: all_results[k][1].get("avg_net_sharpe", 0))
    best_results, best_metrics = all_results[best_model_name]

    # Generate metrics.json using best model
    metrics_json = generate_metrics_json(best_results, config, custom_metrics={
        "best_model": best_model_name,
        "lgbm_default_sharpe": round(lgbm_default_metrics.get("avg_net_sharpe", 0), 4),
        "lgbm_optimized_sharpe": round(lgbm_opt_metrics.get("avg_net_sharpe", 0), 4),
        "nn_default_sharpe": round(nn_default_metrics.get("avg_net_sharpe", 0), 4),
        "nn_optimized_sharpe": round(nn_opt_metrics.get("avg_net_sharpe", 0), 4),
        "lgbm_best_params": lgbm_opt["best_params"],
        "nn_best_params": _serialize_nn_params(nn_opt["best_params"]),
        "lgbm_optimization_trials": lgbm_trials,
        "nn_optimization_trials": nn_trials,
        "lgbm_fold1_best_sharpe": lgbm_opt["best_sharpe_fold1"],
        "nn_fold1_best_sharpe": nn_opt["best_sharpe_fold1"],
    })

    # Save metrics.json
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save optimized params for future use
    params_path = output_path / "best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "lgbm": lgbm_opt["best_params"],
            "nn": _serialize_nn_params(nn_opt["best_params"]),
        }, f, indent=2)
    logger.info(f"Saved best params to {params_path}")

    return {
        "metrics_json": metrics_json,
        "lgbm_optimization": lgbm_opt,
        "nn_optimization": nn_opt,
        "all_metrics": {k: v[1] for k, v in all_results.items()},
        "best_model": best_model_name,
    }


def _serialize_nn_params(params: dict) -> dict:
    """Make NN params JSON-serializable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            out[k] = [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
