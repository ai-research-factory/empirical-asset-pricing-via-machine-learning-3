# Open Questions and Assumptions

## Survivorship Bias

- Using the current S&P 500 constituent list introduces survivorship bias. Stocks that were removed from the index (due to delisting, acquisition, etc.) are excluded. This may inflate backtested returns relative to the paper's methodology.

## Feature Engineering Assumptions

- Monthly resampling uses pandas month-end (`ME`) convention. Close price is taken as the last available close of each month.
- Volatility is computed as trailing 30-day daily return standard deviation, annualized by sqrt(252). The paper may use a different volatility estimator.
- Momentum features use simple cumulative returns (product of (1+r) over the window minus 1), consistent with the standard momentum literature.

## Missing from Paper Reproduction

- The paper likely includes additional firm characteristics beyond momentum and volatility (e.g., size, book-to-market, profitability). These will be added in future phases if data permits.
- Risk-free rate subtraction is deferred to the model training phase. The current `target_return` is the raw next-month return, not excess return.

## ARF Data API Notes

- The API supports downloading tickers beyond its listed catalog (182 listed, but all 503 S&P 500 tickers returned data successfully).
- Data period: ~15 years of daily data available (from ~2011). Sufficient for walk-forward validation.

## Phase 5: Hyperparameter Optimization Notes

- Walk-forward validation yields only 3 valid windows given 14 years of data and a 60-month minimum training size. This limits the statistical significance of Sharpe ratio comparisons.
- Optuna tuning is performed on the first fold only, per paper methodology. This means the tuned parameters are optimized for a specific market regime (2014-2020) and may not generalize.
- Neural network optimization on fold 1 did not transfer well to later folds. The NN optimized configuration actually degraded OOS performance, suggesting overfitting to the tuning fold's market regime.
- The feature set (5 features) may be too limited for the NN to show its advantage. The paper likely uses many more firm characteristics.
- LightGBM consistently outperforms NN across all configurations, which aligns with the paper's finding that tree-based models are competitive with neural networks for moderate feature sets.
