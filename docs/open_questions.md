# Open Questions and Assumptions

## ARF Data API Limitations

- **Ticker coverage gap**: The ARF Data API provides only 74 S&P 500 tickers out of 503. The acceptance criterion of 400+ tickers cannot be met with the current API. The paper uses the full S&P 500 universe; our reduced universe may affect cross-sectional signal strength and portfolio construction.
- **Data period**: The API returns ~14 years of data (from ~2011). The paper uses a longer history. This is sufficient for walk-forward validation but limits the number of training windows.

## Survivorship Bias

- Using the current S&P 500 constituent list introduces survivorship bias. Stocks that were removed from the index (due to delisting, acquisition, etc.) are excluded. This may inflate backtested returns relative to the paper's methodology.

## Feature Engineering Assumptions

- Monthly resampling uses pandas month-end (`ME`) convention. Close price is taken as the last available close of each month.
- Volatility is computed as trailing 30-day daily return standard deviation, annualized by √252. The paper may use a different volatility estimator.
- Momentum features use simple cumulative returns (product of (1+r) over the window minus 1), consistent with the standard momentum literature.

## Missing from Paper Reproduction

- The paper likely includes additional firm characteristics beyond momentum and volatility (e.g., size, book-to-market, profitability). These will be added in future phases if data permits.
- Risk-free rate subtraction is deferred to the model training phase. The current `target_return` is the raw next-month return, not excess return.
