"""
Main entry point for the Empirical Asset Pricing project.

Commands:
    build-dataset: Download S&P 500 data and build monthly features
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Download S&P 500 data and build feature dataset."""
    from src.data.sp500_loader import download_sp500_data
    from src.features.build_features import build_features

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download daily data
    daily_df = download_sp500_data(
        period=args.period,
        cache=not args.no_cache,
        delay=args.delay,
    )

    # Build features
    features_df = build_features(daily_df)

    # Save
    features_df.to_parquet(output_path)
    logger.info(f"Saved dataset to {output_path}")

    # Print summary
    n_tickers = features_df["ticker"].nunique()
    date_range = features_df.index.max() - features_df.index.min()
    years = date_range.days / 365.25
    feature_cols = [c for c in features_df.columns if c not in (
        "open", "high", "low", "close", "volume", "ticker",
        "monthly_return", "target_return",
    )]
    print(f"\nDataset Summary:")
    print(f"  Tickers: {n_tickers}")
    print(f"  Rows: {len(features_df)}")
    print(f"  Date range: {features_df.index.min().date()} to {features_df.index.max().date()}")
    print(f"  Years of data: {years:.1f}")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Target column: target_return")
    print(f"  Output: {output_path}")


def cmd_optimize_hyperparams(args: argparse.Namespace) -> None:
    """Run hyperparameter optimization for LightGBM and NN models."""
    import pandas as pd
    from src.models.optimize import run_full_optimization

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}. Run 'build-dataset' first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

    results = run_full_optimization(
        df,
        lgbm_trials=args.lgbm_trials,
        nn_trials=args.nn_trials,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("Hyperparameter Optimization Results")
    print(f"{'='*60}")
    print(f"Best model: {results['best_model']}")
    for model_name, metrics in results["all_metrics"].items():
        sharpe = metrics.get("avg_net_sharpe", 0)
        print(f"  {model_name}: avg_net_sharpe={sharpe:.4f}")
    print(f"\nLGBM best params (fold 1 Sharpe={results['lgbm_optimization']['best_sharpe_fold1']:.4f}):")
    for k, v in results["lgbm_optimization"]["best_params"].items():
        print(f"  {k}: {v}")
    print(f"\nNN best params (fold 1 Sharpe={results['nn_optimization']['best_sharpe_fold1']:.4f}):")
    for k, v in results["nn_optimization"]["best_params"].items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {args.output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical Asset Pricing via Machine Learning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build-dataset command
    build_parser = subparsers.add_parser(
        "build-dataset",
        help="Download S&P 500 data and build monthly feature dataset",
    )
    build_parser.add_argument(
        "--output",
        default="data/processed/sp500_monthly_features.parquet",
        help="Output parquet file path",
    )
    build_parser.add_argument(
        "--period",
        default="15y",
        help="Data lookback period (default: 15y)",
    )
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download (ignore cache)",
    )
    build_parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds (default: 0.2)",
    )
    build_parser.set_defaults(func=cmd_build_dataset)

    # optimize-hyperparams command
    opt_parser = subparsers.add_parser(
        "optimize-hyperparams",
        help="Run Optuna hyperparameter optimization for LightGBM and NN models",
    )
    opt_parser.add_argument(
        "--data",
        default="data/processed/sp500_monthly_features.parquet",
        help="Input parquet file path",
    )
    opt_parser.add_argument(
        "--lgbm-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for LightGBM (default: 30)",
    )
    opt_parser.add_argument(
        "--nn-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for NN (default: 20)",
    )
    opt_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of walk-forward windows (default: 5)",
    )
    opt_parser.add_argument(
        "--output-dir",
        default="reports/cycle_5",
        help="Output directory for results",
    )
    opt_parser.set_defaults(func=cmd_optimize_hyperparams)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
