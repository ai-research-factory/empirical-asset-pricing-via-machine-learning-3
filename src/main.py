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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
