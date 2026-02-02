"""
Options engine utilities for chain fetch and pricing.
"""
from math import isfinite
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from src.market_data import get_provider


@st.cache_data(ttl=300)
def fetch_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    provider = get_provider()
    chain = provider.get_option_chain(ticker, expiry)
    return chain.calls.copy(), chain.puts.copy()


def _to_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan


def contract_mid(row: pd.Series) -> float:
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    last = _to_float(row.get("lastPrice"))

    if isfinite(bid) and isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if isfinite(last) and last > 0:
        return last
    return np.nan


def contract_liquidity_metrics(row: pd.Series) -> Dict[str, float]:
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    mid = contract_mid(row)
    spread_pct = np.nan
    if isfinite(mid) and mid > 0 and isfinite(bid) and isfinite(ask) and ask > 0 and bid > 0:
        spread_pct = (ask - bid) / mid

    volume = _to_float(row.get("volume"))
    oi = _to_float(row.get("openInterest"))
    iv = _to_float(row.get("impliedVolatility"))

    return {
        "spread_pct": spread_pct,
        "volume": volume if isfinite(volume) else np.nan,
        "open_interest": oi if isfinite(oi) else np.nan,
        "iv": iv if isfinite(iv) else np.nan,
    }


def _find_strike_row(df: pd.DataFrame, strike: float, tol: float = 0.5) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    strikes = df.get("strike")
    if strikes is None:
        return pd.Series(dtype=float)
    idx = (strikes - strike).abs().idxmin()
    if pd.isna(idx):
        return pd.Series(dtype=float)
    if abs(float(strikes.loc[idx]) - strike) > tol:
        return pd.Series(dtype=float)
    return df.loc[idx]


def price_call_debit_spread(spread_row: pd.Series, calls_df: pd.DataFrame) -> Dict[str, float]:
    long_strike = _to_float(spread_row.get("long_strike"))
    short_strike = _to_float(spread_row.get("short_strike"))
    contracts = _to_float(spread_row.get("contracts"))
    entry_debit = _to_float(spread_row.get("entry_debit"))

    long_row = _find_strike_row(calls_df, long_strike)
    short_row = _find_strike_row(calls_df, short_strike)

    if long_row.empty or short_row.empty:
        return {"error": "missing_strike"}

    long_mid = contract_mid(long_row)
    short_mid = contract_mid(short_row)
    if not isfinite(long_mid) or not isfinite(short_mid):
        return {"error": "missing_mid"}

    spread_mark = long_mid - short_mid
    width = short_strike - long_strike
    max_profit = max(width - spread_mark, 0.0)
    max_loss = max(spread_mark, 0.0)
    breakeven = long_strike + spread_mark

    pnl_per = spread_mark - entry_debit
    pnl_dollars = pnl_per * contracts * 100.0 if isfinite(contracts) else pnl_per * 100.0
    pnl_pct = (pnl_per / entry_debit) * 100.0 if entry_debit and entry_debit > 0 else np.nan

    metrics = {
        "long_mid": long_mid,
        "short_mid": short_mid,
        "spread_mark": spread_mark,
        "pnl_$": pnl_dollars,
        "pnl_%": pnl_pct,
        "width": width,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven": breakeven,
    }

    metrics.update({"long_" + k: v for k, v in contract_liquidity_metrics(long_row).items()})
    metrics.update({"short_" + k: v for k, v in contract_liquidity_metrics(short_row).items()})
    return metrics


def price_long_option(long_row: pd.Series, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> Dict[str, float]:
    opt_type = str(long_row.get("type", "call")).lower()
    strike = _to_float(long_row.get("strike"))
    contracts = _to_float(long_row.get("contracts"))
    entry_price = _to_float(long_row.get("entry_price"))

    df = calls_df if opt_type == "call" else puts_df
    row = _find_strike_row(df, strike)
    if row.empty:
        return {"error": "missing_strike"}

    mid = contract_mid(row)
    if not isfinite(mid):
        return {"error": "missing_mid"}

    pnl_per = mid - entry_price
    pnl_dollars = pnl_per * contracts * 100.0 if isfinite(contracts) else pnl_per * 100.0
    pnl_pct = (pnl_per / entry_price) * 100.0 if entry_price and entry_price > 0 else np.nan

    metrics = {
        "mid": mid,
        "pnl_$": pnl_dollars,
        "pnl_%": pnl_pct,
    }
    metrics.update(contract_liquidity_metrics(row))
    return metrics
