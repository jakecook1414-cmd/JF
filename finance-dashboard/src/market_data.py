"""
Market data provider interface and Yahoo implementation.
"""
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class OptionChain:
    calls: pd.DataFrame
    puts: pd.DataFrame


class MarketDataProvider:
    def get_prices(self, tickers: List[str], start, end) -> pd.DataFrame:
        raise NotImplementedError

    def get_clean_prices(self, ticker: str, start, end) -> pd.Series:
        raise NotImplementedError

    def get_option_chain(self, ticker: str, expiry: str) -> OptionChain:
        raise NotImplementedError

    def get_option_expiries(self, ticker: str) -> List[str]:
        raise NotImplementedError

    def get_spot(self, ticker: str) -> float:
        raise NotImplementedError


class YahooProvider(MarketDataProvider):
    def get_prices(self, tickers: List[str], start, end) -> pd.DataFrame:
        return yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

    def get_clean_prices(self, ticker: str, start, end) -> pd.Series:
        data = self.get_prices([ticker], start, end)
        if data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                prices = data[ticker]["Close"].copy()
            else:
                prices = data["Close"].copy()
        else:
            prices = data["Close"].copy()

        prices = prices[~prices.index.duplicated(keep="first")]
        prices = prices.sort_index()
        prices = prices.ffill(limit=2)
        prices = prices.dropna()
        return prices

    def get_option_chain(self, ticker: str, expiry: str) -> OptionChain:
        try:
            chain = yf.Ticker(ticker).option_chain(expiry)
            return OptionChain(chain.calls.copy(), chain.puts.copy())
        except Exception:
            return OptionChain(pd.DataFrame(), pd.DataFrame())

    def get_option_expiries(self, ticker: str) -> List[str]:
        try:
            return list(yf.Ticker(ticker).options)
        except Exception:
            return []

    def get_spot(self, ticker: str) -> float:
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            return float(hist["Close"].iloc[-1])
        except Exception:
            return float("nan")


def get_provider() -> MarketDataProvider:
    return YahooProvider()
