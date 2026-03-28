# Phase 2: Data Pipeline Construction (S&P 500) - Technical Findings

## Implementation Summary

Built a complete data pipeline that fetches S&P 500 stock data from the ARF Data API, resamples to monthly frequency, and computes features for asset pricing ML models.

### Components

1. **`src/data/sp500_loader.py`** - Data acquisition module
   - Fetches S&P 500 ticker list from Wikipedia (503 tickers)
   - Downloads all tickers directly from ARF Data API (supports unlisted tickers)
   - Parallel downloads via ThreadPoolExecutor for performance (~9 min for 503 tickers)
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
| Tickers | 502 |
| Total rows | 79,848 |
| Date range | 2012-03-31 to 2026-02-28 |
| Years of data | ~13.9 |
| Features | mom_1m, mom_3m, mom_6m, mom_12m, volatility_30d |

### Key Changes in This Cycle

- **Full S&P 500 coverage**: Changed from downloading only tickers listed in the ARF API catalog (74) to attempting all 503 S&P 500 tickers directly. The API supports unlisted tickers, yielding 502 tickers in the final dataset.
- **Parallel downloads**: Replaced sequential downloads with `ThreadPoolExecutor` (10 workers), reducing download time from ~58 minutes to ~9 minutes.
- **Bug fix (prior commit)**: Fixed momentum feature cross-ticker overwrite caused by non-unique DatetimeIndex.

### Known Limitations

- **Survivorship bias**: Using current S&P 500 constituents introduces survivorship bias, as delisted or removed stocks are excluded.
- **Data start dates vary**: Some tickers have shorter histories due to IPO dates or corporate events (1 ticker lost after NaN filtering).
- **Risk-free rate**: The current `target_return` is raw next-month return, not excess return. Risk-free rate subtraction is deferred to model training.

### Feature Definitions

- **mom_1m**: Monthly return (close-to-close)
- **mom_3m**: Cumulative return over past 3 months
- **mom_6m**: Cumulative return over past 6 months
- **mom_12m**: Cumulative return over past 12 months
- **volatility_30d**: Annualized standard deviation of daily returns over trailing 30 trading days
- **target_return**: Next month's return (prediction target)
