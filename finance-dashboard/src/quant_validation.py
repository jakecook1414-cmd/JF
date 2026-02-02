from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.quant_forecast import simulate_bootstrap, simulate_regime_bootstrap


def calibration_report(
    prices: pd.Series,
    horizon_days: int = 20,
    lookback_days: int = 756,
    n_paths: int = 2000,
    method: str = "bootstrap",
) -> pd.DataFrame:
    prices = prices.dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    if len(prices) < lookback_days + horizon_days + 1:
        return pd.DataFrame()

    rows = []
    for i in range(lookback_days, len(prices) - horizon_days):
        window = prices.iloc[i - lookback_days:i]
        start_price = prices.iloc[i]
        future_price = prices.iloc[i + horizon_days]

        if method == "regime":
            _, terminal = simulate_regime_bootstrap(window, horizon_days, n_paths)
        else:
            _, terminal = simulate_bootstrap(window, horizon_days, n_paths)

        if terminal.size == 0:
            continue

        p10, p90 = np.percentile(terminal, [10, 90])
        p5, p95 = np.percentile(terminal, [5, 95])

        rows.append({
            "date": prices.index[i],
            "start": start_price,
            "future": future_price,
            "p10": p10,
            "p90": p90,
            "p5": p5,
            "p95": p95,
            "in_10_90": p10 <= future_price <= p90,
            "in_5_95": p5 <= future_price <= p95,
        })

    return pd.DataFrame(rows)


def coverage_summary(calib: pd.DataFrame) -> Dict[str, float]:
    if calib.empty:
        return {"cov_10_90": 0.0, "cov_5_95": 0.0}
    return {
        "cov_10_90": float(calib["in_10_90"].mean()),
        "cov_5_95": float(calib["in_5_95"].mean()),
    }


def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
    prices = prices.dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    log_rets = np.log(prices).diff()
    return log_rets.rolling(window).std() * np.sqrt(252)


def drawdown_curve(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    equity = prices / prices.iloc[0]
    peak = equity.cummax()
    return (equity / peak) - 1.0


def returns_distribution(prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    prices = prices.dropna()
    if prices.empty:
        return np.array([]), np.array([])
    rets = prices.pct_change().dropna()
    return rets.values, rets.index.values
