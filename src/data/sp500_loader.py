"""
S&P 500 data loader using ARF Data API.

Fetches S&P 500 ticker list from Wikipedia, then downloads
daily OHLCV data from the ARF Data API for available tickers.
"""
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARF_API_BASE = "https://ai.1s.xyz/api/data"
ARF_TICKERS_URL = f"{ARF_API_BASE}/tickers"
ARF_OHLCV_URL = f"{ARF_API_BASE}/ohlcv"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia."""
    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ARF-DataPipeline/1.0)"}
    resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    sp500_table = tables[0]
    tickers = sp500_table["Symbol"].tolist()
    # Normalize: BRK.B -> BRK-B for API compatibility
    tickers = [t.replace(".", "-") for t in tickers]
    logger.info(f"Found {len(tickers)} S&P 500 tickers from Wikipedia")
    return tickers


def get_available_tickers() -> list[str]:
    """Fetch the list of tickers available on the ARF Data API."""
    logger.info("Fetching available tickers from ARF Data API...")
    resp = requests.get(ARF_TICKERS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Response is {"presets": {"Category": [{"ticker": "...", "name": "..."}]}}
    all_tickers = []
    presets = data.get("presets", data)
    if isinstance(presets, dict):
        for category_items in presets.values():
            if isinstance(category_items, list):
                for item in category_items:
                    if isinstance(item, dict) and "ticker" in item:
                        all_tickers.append(item["ticker"])
                    elif isinstance(item, str):
                        all_tickers.append(item)
    return list(set(all_tickers))


def get_sp500_available_tickers() -> list[str]:
    """Return S&P 500 tickers that are available in the ARF Data API."""
    sp500 = set(get_sp500_tickers())
    available = set(get_available_tickers())
    overlap = sorted(sp500 & available)
    logger.info(
        f"S&P 500 tickers available in ARF API: {len(overlap)} / {len(sp500)}"
    )
    return overlap


def download_ticker_data(
    ticker: str, interval: str = "1d", period: str = "15y"
) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker from ARF Data API."""
    params = {"ticker": ticker, "interval": interval, "period": period}
    try:
        resp = requests.get(ARF_OHLCV_URL, params=params, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            logger.warning(f"Empty data for {ticker}")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df["ticker"] = ticker
        return df
    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def download_sp500_data(
    interval: str = "1d",
    period: str = "15y",
    cache: bool = True,
    delay: float = 0.2,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for all S&P 500 tickers.

    The ARF Data API supports tickers beyond its listed catalog,
    so we attempt all S&P 500 tickers directly.

    Args:
        interval: Data interval (default '1d')
        period: Lookback period (default '15y')
        cache: If True, use cached data when available
        delay: Delay between API calls in seconds

    Returns:
        DataFrame with columns: open, high, low, close, volume, ticker
    """
    cache_path = RAW_DIR / "sp500_daily.parquet"

    if cache and cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    tickers = get_sp500_tickers()
    if not tickers:
        raise RuntimeError("No S&P 500 tickers found from Wikipedia")

    max_workers = max(1, int(1 / delay)) if delay > 0 else 10
    logger.info(
        f"Downloading data for {len(tickers)} tickers "
        f"(max_workers={max_workers})..."
    )
    frames = []

    def _fetch(ticker: str) -> tuple[str, pd.DataFrame | None]:
        df = download_ticker_data(ticker, interval=interval, period=period)
        return ticker, df

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            ticker, df = future.result()
            if df is not None and len(df) > 0:
                frames.append(df)
            if i % 50 == 0 or i == len(tickers):
                logger.info(f"[{i}/{len(tickers)}] Downloaded so far...")

    if not frames:
        raise RuntimeError("No data downloaded for any ticker")

    combined = pd.concat(frames)
    combined = combined.sort_index()

    # Cache the raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    logger.info(
        f"Downloaded data for {combined['ticker'].nunique()} tickers, "
        f"{len(combined)} total rows. Cached to {cache_path}"
    )
    return combined
