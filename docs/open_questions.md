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
