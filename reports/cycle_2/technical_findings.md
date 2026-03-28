# Phase 2: Data Pipeline Construction (S&P 500) - Technical Findings

## Implementation Summary

Built a complete data pipeline that fetches S&P 500 stock data from the ARF Data API, resamples to monthly frequency, and computes features for asset pricing ML models.

### Components

1. **`src/data/sp500_loader.py`** - Data acquisition module
   - Fetches S&P 500 ticker list from Wikipedia (503 tickers)
   - Cross-references with ARF Data API available tickers (74 overlap)
   - Downloads 15 years of daily OHLCV data per ticker
   - Implements local caching in `data/raw/sp500_daily.parquet`

2. **`src/features/build_features.py`** - Feature engineering module
   - Monthly resampling using month-end frequency
   - Target: next-month return (`target_return`) via forward shift
   - Momentum features: 1, 3, 6, 12-month cumulative returns
   - Volatility: 30-day realized volatility (annualized)
   - NaN rows dropped after feature computation

3. **`src/main.py`** - CLI entry point
   - `build-dataset` command with configurable output path, period, cache, and delay

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Tickers | 74 |
| Total rows | 12,019 |
| Date range | 2012-03-31 to 2026-02-28 |
| Years of data | ~13.9 |
| Features | mom_1m, mom_3m, mom_6m, mom_12m, volatility_30d |

### Bug Fix: Momentum Feature Assignment

Fixed a critical bug in `compute_momentum_features()` where using `df.loc[group.index, col]` with a non-unique DatetimeIndex caused momentum values to be overwritten across tickers sharing the same date. The last ticker processed would overwrite all others, resulting in identical momentum values for all tickers on a given date. The fix builds per-ticker DataFrames independently and concatenates them.

### Known Limitations

- **Ticker coverage**: Only 74 of 503 S&P 500 tickers are available in the ARF Data API (14.7%). The acceptance criterion of 400+ tickers cannot be met with the current API. See `docs/open_questions.md`.
- **Survivorship bias**: Using current S&P 500 constituents introduces survivorship bias, as delisted or removed stocks are excluded.
- **Data start dates vary**: Some tickers have shorter histories due to IPO dates or corporate events.

### Feature Definitions

- **mom_1m**: Monthly return (close-to-close)
- **mom_3m**: Cumulative return over past 3 months
- **mom_6m**: Cumulative return over past 6 months
- **mom_12m**: Cumulative return over past 12 months
- **volatility_30d**: Annualized standard deviation of daily returns over trailing 30 trading days
- **target_return**: Next month's return (prediction target)
