from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _clean_series(prices: pd.Series) -> pd.Series:
    if prices is None:
        return pd.Series(dtype=float)
    prices = prices.dropna()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.sort_index()


def _log_returns(prices: pd.Series) -> pd.Series:
    prices = _clean_series(prices)
    if prices.empty:
        return pd.Series(dtype=float)
    return np.log(prices).diff().dropna()


def _scale_to_target_vol(samples: np.ndarray, target_vol: float) -> np.ndarray:
    sample_vol = np.std(samples)
    if sample_vol > 0:
        return samples * (target_vol / sample_vol)
    return samples


def _bootstrap_from_returns(
    log_rets: pd.Series,
    horizon_days: int,
    n_paths: int,
    seed: int,
    vol_scale: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.choice(log_rets.values, size=(n_paths, horizon_days), replace=True)

    if vol_scale:
        recent_vol = log_rets.tail(60).std()
        if np.isnan(recent_vol) or recent_vol == 0:
            recent_vol = log_rets.std()
        samples = _scale_to_target_vol(samples, float(recent_vol))
    return samples


def simulate_bootstrap(
    prices: pd.Series,
    horizon_days: int,
    n_paths: int = 10000,
    lookback_days: int = 756,
    seed: int = 42,
    vol_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    prices = _clean_series(prices)
    if len(prices) < 2:
        return np.array([]), np.array([])

    log_rets = _log_returns(prices)
    if log_rets.empty:
        return np.array([]), np.array([])

    if len(log_rets) > lookback_days:
        log_rets = log_rets.iloc[-lookback_days:]

    samples = _bootstrap_from_returns(log_rets, horizon_days, n_paths, seed, vol_scale)

    start = float(prices.iloc[-1])
    paths = np.empty((n_paths, horizon_days + 1), dtype=float)
    paths[:, 0] = start
    for i in range(horizon_days):
        paths[:, i + 1] = paths[:, i] * np.exp(samples[:, i])
    terminal = paths[:, -1]
    return paths, terminal


def _regime_mask(prices: pd.Series) -> pd.Series:
    prices = _clean_series(prices)
    sma_200 = prices.rolling(200).mean()
    trend_up = prices > sma_200

    log_rets = _log_returns(prices)
    vol_20 = log_rets.rolling(20).std()
    vol_med = vol_20.rolling(252, min_periods=20).median()
    vol_high = vol_20 > vol_med

    reg = (trend_up & vol_high).astype(int) * 2 + (trend_up & ~vol_high).astype(int) * 1 + (~trend_up & vol_high).astype(int) * 3
    reg = reg.reindex(prices.index).ffill()
    return reg


def simulate_regime_bootstrap(
    prices: pd.Series,
    horizon_days: int,
    n_paths: int = 10000,
    lookback_days: int = 756,
    seed: int = 42,
    vol_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    prices = _clean_series(prices)
    if len(prices) < 260:
        return simulate_bootstrap(prices, horizon_days, n_paths, lookback_days, seed, vol_scale)

    log_rets = _log_returns(prices)
    if log_rets.empty:
        return np.array([]), np.array([])

    if len(log_rets) > lookback_days:
        log_rets = log_rets.iloc[-lookback_days:]

    regime = _regime_mask(prices)
    current_regime = regime.iloc[-1]
    aligned = regime.reindex(log_rets.index).ffill()
    subset = log_rets[aligned == current_regime]
    if subset.empty or len(subset) < 30:
        subset = log_rets

    samples = _bootstrap_from_returns(subset, horizon_days, n_paths, seed, vol_scale)

    start = float(prices.iloc[-1])
    paths = np.empty((n_paths, horizon_days + 1), dtype=float)
    paths[:, 0] = start
    for i in range(horizon_days):
        paths[:, i + 1] = paths[:, i] * np.exp(samples[:, i])
    terminal = paths[:, -1]
    return paths, terminal


def terminal_percentiles(terminal: np.ndarray) -> Dict[int, float]:
    if terminal.size == 0:
        return {}
    return {p: float(np.percentile(terminal, p)) for p in [5, 25, 50, 75, 95]}


def prob_above(terminal: np.ndarray, threshold: float) -> float:
    if terminal.size == 0:
        return 0.0
    return float(np.mean(terminal > threshold))
