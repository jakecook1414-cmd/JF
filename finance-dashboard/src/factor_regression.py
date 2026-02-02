"""
Factor regression utilities (OLS) using yfinance for data and statsmodels for fitting.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf


@dataclass
class FactorRegressionResult:
    data: pd.DataFrame
    coefficients: pd.DataFrame
    metrics: Dict[str, float]


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    cleaned = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in cleaned:
            cleaned.append(t)
    return cleaned


def download_adjusted_prices(tickers: Iterable[str], start, end) -> pd.DataFrame:
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        close = data["Close"].to_frame(name=tickers[0])

    close = close.dropna(how="all")
    return close


def run_factor_regression(
    asset: str,
    factors: Iterable[str],
    start,
    end,
    min_obs: int = 60,
) -> Optional[FactorRegressionResult]:
    asset = str(asset).strip().upper()
    factors = _normalize_tickers(factors)
    factors = [f for f in factors if f != asset]
    if not asset or not factors:
        return None

    prices = download_adjusted_prices([asset] + factors, start, end)
    if prices.empty:
        return None

    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna()
    if returns.empty or len(returns) < min_obs:
        return None

    y = returns[asset]
    X = returns[factors]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues
    coeffs = pd.DataFrame({
        "coef": params,
        "t_stat": tvals,
        "p_value": pvals,
    })

    alpha_daily = float(params.get("const", 0.0))
    alpha_annual = alpha_daily * 252.0
    tracking_error = float(model.resid.std() * np.sqrt(252))
    info_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0.0

    metrics = {
        "n_obs": float(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "tracking_error": tracking_error,
        "info_ratio": float(info_ratio),
    }

    data = pd.concat([y.rename("asset"), X[factors]], axis=1)
    data["fitted"] = model.fittedvalues
    data["residual"] = model.resid

    return FactorRegressionResult(data=data, coefficients=coeffs, metrics=metrics)
