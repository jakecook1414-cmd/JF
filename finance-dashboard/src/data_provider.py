from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf


CALENDAR = xcals.get_calendar("XNYS")


def _normalize_tickers(tickers: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(tickers, str):
        return [tickers.upper()]
    return [str(t).upper() for t in tickers]


def _extract_close(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance with multiple tickers and group_by="ticker"
        frames = []
        for t in tickers:
            if t in df.columns.get_level_values(0):
                slice_df = df[t]
                if "Close" in slice_df.columns:
                    frames.append(slice_df["Close"].rename(t))
                elif "Adj Close" in slice_df.columns:
                    frames.append(slice_df["Adj Close"].rename(t))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    # Single ticker case
    if "Close" in df.columns:
        return df["Close"].to_frame(name=tickers[0])
    if "Adj Close" in df.columns:
        return df["Adj Close"].to_frame(name=tickers[0])
    return pd.DataFrame()


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.ffill(limit=2)
    df = df.dropna(how="any")
    return df


@dataclass
class YahooProvider:
    def get_prices(
        self,
        tickers: Union[str, Iterable[str]],
        start,
        end,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        tickers_list = _normalize_tickers(tickers)
        df = yf.download(
            tickers=tickers_list,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
        )
        close = _extract_close(df, tickers_list)
        return _clean_prices(close)

    def get_returns(self, prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        if prices.empty:
            return prices
        if method == "log":
            rets = np.log(prices).diff()
        else:
            rets = prices.pct_change()
        return rets.dropna(how="any")


def trading_days_between(start, end) -> int:
    try:
        sessions = CALENDAR.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return max(len(sessions) - 1, 0)
    except Exception:
        return 0


def ensure_trading_horizon(start, end) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    horizon = trading_days_between(start_ts, end_ts)
    return start_ts, end_ts, horizon
