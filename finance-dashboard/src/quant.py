"""
Quant utilities: clean prices, bootstrap simulation, regime-conditioned simulation,
and walk-forward backtesting.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from src.market_data import get_provider
from arch import arch_model
import vectorbt as vbt


CALENDAR = xcals.get_calendar("XNYS")
PROVIDER = get_provider()


@dataclass
class WalkForwardResult:
    segment_metrics: pd.DataFrame
    overall_metrics: Dict[str, float]
    equity_curve: pd.Series


def get_clean_prices(ticker: str, start, end) -> pd.Series:
    return PROVIDER.get_clean_prices(ticker, start, end)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    return np.log(prices).diff().dropna()


def _robust_vol(log_returns: pd.Series) -> float:
    if log_returns.empty:
        return 0.0
    med = log_returns.median()
    mad = (log_returns - med).abs().median()
    return float(1.4826 * mad)


def _garch_vol_forecast(log_returns: pd.Series) -> float:
    if len(log_returns) < 50:
        return _robust_vol(log_returns)
    returns_pct = log_returns * 100.0
    am = arch_model(returns_pct, p=1, q=1, mean="Zero", vol="Garch", dist="normal")
    res = am.fit(disp="off")
    f = res.forecast(horizon=1)
    var = f.variance.values[-1][0]
    vol = np.sqrt(var) / 100.0
    return float(vol)


def _scale_to_target_vol(samples: np.ndarray, target_vol: float) -> np.ndarray:
    sample_vol = _robust_vol(pd.Series(samples.flatten()))
    if sample_vol > 0:
        return samples * (target_vol / sample_vol)
    return samples


def simulate_paths_bootstrap(
    prices: pd.Series,
    horizon_days: int,
    n_paths: int,
    lookback_days: int = 756,
    vol_scale: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    prices = prices.dropna()
    if len(prices) < 2:
        return np.array([]), np.array([])

    log_rets = compute_log_returns(prices)
    if log_rets.empty:
        return np.array([]), np.array([])

    if len(log_rets) > lookback_days:
        log_rets = log_rets.iloc[-lookback_days:]

    rng = np.random.default_rng(seed)
    samples = rng.choice(log_rets.values, size=(n_paths, horizon_days), replace=True)

    if vol_scale:
        target_vol = _garch_vol_forecast(log_rets)
        samples = _scale_to_target_vol(samples, target_vol)

    start = float(prices.iloc[-1])
    paths = np.empty((n_paths, horizon_days + 1), dtype=float)
    paths[:, 0] = start
    for i in range(horizon_days):
        paths[:, i + 1] = paths[:, i] * np.exp(samples[:, i])

    terminal = paths[:, -1]
    return paths, terminal


def _regime_mask(prices: pd.Series) -> pd.Series:
    sma_200 = prices.rolling(200).mean()
    trend = prices > sma_200

    log_rets = compute_log_returns(prices)
    vol_20 = log_rets.rolling(20).std()
    vol_252_median = vol_20.rolling(252).median()
    vol = vol_20 > vol_252_median

    regime = (trend & vol).astype(int) * 2 + (trend & ~vol).astype(int) * 1 + (~trend & vol).astype(int) * 3
    regime = regime.reindex(prices.index).ffill()
    return regime


def simulate_paths_regime(
    prices: pd.Series,
    horizon_days: int,
    n_paths: int,
    lookback_days: int = 756,
    vol_scale: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    prices = prices.dropna()
    if len(prices) < 260:
        return simulate_paths_bootstrap(prices, horizon_days, n_paths, lookback_days, vol_scale, seed)

    regime = _regime_mask(prices)
    current_regime = regime.iloc[-1]

    log_rets = compute_log_returns(prices)
    if len(log_rets) > lookback_days:
        log_rets = log_rets.iloc[-lookback_days:]

    aligned_regime = regime.reindex(log_rets.index).ffill()
    subset = log_rets[aligned_regime == current_regime]

    if subset.empty or len(subset) < 30:
        subset = log_rets

    rng = np.random.default_rng(seed)
    samples = rng.choice(subset.values, size=(n_paths, horizon_days), replace=True)

    if vol_scale:
        target_vol = _garch_vol_forecast(log_rets)
        samples = _scale_to_target_vol(samples, target_vol)

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


def backtest_sma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    prices = prices.dropna()
    if len(prices) < slow + 2:
        return pd.Series(dtype=float)

    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = (sma_fast > sma_slow).astype(int)
    signal = signal.shift(1).fillna(0)

    daily_ret = prices.pct_change().fillna(0)
    strat_ret = daily_ret * signal
    strat_ret.name = "strategy_return"
    return strat_ret


def _metrics_from_returns(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
        }

    equity = (1 + returns).cumprod()
    total_days = max(len(returns), 1)
    years = total_days / 252
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_drawdown = float(drawdown.min())

    vol = float(returns.std() * np.sqrt(252))
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

    nonzero = returns[returns != 0]
    win_rate = float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.0

    turnover = float(returns.diff().abs().sum() / max(len(returns), 1))

    return {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "vol": vol,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "turnover": turnover,
    }


def walk_forward_eval(
    prices: pd.Series,
    strategy_fn: Callable[[pd.Series], pd.Series],
    train_days: int = 756,
    test_days: int = 126,
    step_days: int = 126,
) -> WalkForwardResult:
    prices = prices.dropna()
    if len(prices) < train_days + test_days + 5:
        empty = pd.DataFrame(columns=["start", "end", "cagr", "max_drawdown", "vol", "sharpe", "win_rate", "turnover"])
        return WalkForwardResult(empty, _metrics_from_returns(pd.Series(dtype=float)), pd.Series(dtype=float))

    segment_rows = []
    all_returns = []

    start_idx = train_days
    while start_idx + test_days <= len(prices):
        train_slice = prices.iloc[start_idx - train_days:start_idx]
        test_slice = prices.iloc[start_idx:start_idx + test_days]

        _ = train_slice  # placeholder for future calibration
        returns = strategy_fn(test_slice)
        returns = returns.reindex(test_slice.index).fillna(0)

        metrics = _metrics_from_returns(returns)
        segment_rows.append({
            "start": test_slice.index[0],
            "end": test_slice.index[-1],
            **metrics,
        })
        all_returns.append(returns)

        start_idx += step_days

    if all_returns:
        all_returns = pd.concat(all_returns).sort_index()
    else:
        all_returns = pd.Series(dtype=float)

    overall = _metrics_from_returns(all_returns)
    equity = (1 + all_returns).cumprod() if not all_returns.empty else pd.Series(dtype=float)

    segment_df = pd.DataFrame(segment_rows)
    return WalkForwardResult(segment_df, overall, equity)


def get_terminal_distribution(
    ticker: str,
    horizon_days: int,
    method: str = "bootstrap",
    n_paths: int = 10000,
) -> np.ndarray:
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=365 * 5)
    prices = get_clean_prices(ticker, start, end)
    if prices.empty:
        return np.array([])
    if method == "regime":
        _, terminal = simulate_paths_regime(prices, horizon_days, n_paths)
    else:
        _, terminal = simulate_paths_bootstrap(prices, horizon_days, n_paths)
    return terminal


def trading_days_between(start, end) -> int:
    try:
        sessions = CALENDAR.sessions_in_range(start, end)
        return max(len(sessions) - 1, 0)
    except Exception:
        return 0


def dte_from_expiry(expiry_str: str) -> int:
    try:
        exp = pd.Timestamp(expiry_str).date()
        today = pd.Timestamp.utcnow().date()
        return trading_days_between(pd.Timestamp(today), pd.Timestamp(exp))
    except Exception:
        return 0


def calibration_report(
    prices: pd.Series,
    horizon_days: int = 20,
    lookback_days: int = 756,
    n_paths: int = 2000,
    method: str = "bootstrap",
) -> pd.DataFrame:
    prices = prices.dropna()
    if len(prices) < lookback_days + horizon_days + 1:
        return pd.DataFrame()

    rows = []
    for i in range(lookback_days, len(prices) - horizon_days):
        window = prices.iloc[i - lookback_days:i]
        start_price = prices.iloc[i]
        future_price = prices.iloc[i + horizon_days]

        if method == "regime":
            _, terminal = simulate_paths_regime(window, horizon_days, n_paths)
        else:
            _, terminal = simulate_paths_bootstrap(window, horizon_days, n_paths)

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


def _pf_stats(pf) -> Dict[str, float]:
    stats = pf.stats()
    return {
        "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
        "max_drawdown": float(stats.get("Max Drawdown [%]", 0.0)) / 100.0,
        "total_return": float(stats.get("Total Return [%]", 0.0)) / 100.0,
    }


def sma_research(prices: pd.Series, fast_range, slow_range) -> pd.DataFrame:
    rows = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            fast_ma = vbt.MA.run(prices, window=fast)
            slow_ma = vbt.MA.run(prices, window=slow)
            entries = fast_ma.ma_crossed_above(slow_ma.ma).shift(1).fillna(False)
            exits = fast_ma.ma_crossed_below(slow_ma.ma).shift(1).fillna(False)
            pf = vbt.Portfolio.from_signals(prices, entries, exits)
            stats = _pf_stats(pf)
            rows.append({"fast": fast, "slow": slow, **stats})
    return pd.DataFrame(rows)


def momentum_research(prices: pd.Series, lookbacks) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        mom = prices / prices.shift(lb) - 1.0
        entries = (mom > 0).shift(1).fillna(False)
        exits = (mom <= 0).shift(1).fillna(False)
        pf = vbt.Portfolio.from_signals(prices, entries, exits)
        stats = _pf_stats(pf)
        rows.append({"lookback": lb, **stats})
    return pd.DataFrame(rows)


def walk_forward_research(
    prices: pd.Series,
    strategy: str,
    param_grid: Dict[str, list],
    train_days: int,
    test_days: int,
    step_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.dropna()
    if len(prices) < train_days + test_days + 5:
        return pd.DataFrame(), pd.DataFrame()

    segments = []
    best_rows = []
    start_idx = train_days
    while start_idx + test_days <= len(prices):
        train = prices.iloc[start_idx - train_days:start_idx]
        test = prices.iloc[start_idx:start_idx + test_days]

        if strategy == "sma":
            results = sma_research(train, param_grid["fast"], param_grid["slow"])
            best = results.sort_values("sharpe", ascending=False).head(1)
            best_params = best.iloc[0]
            fast = int(best_params["fast"])
            slow = int(best_params["slow"])
            fast_ma = vbt.MA.run(test, window=fast)
            slow_ma = vbt.MA.run(test, window=slow)
            entries = fast_ma.ma_crossed_above(slow_ma.ma).shift(1).fillna(False)
            exits = fast_ma.ma_crossed_below(slow_ma.ma).shift(1).fillna(False)
            pf = vbt.Portfolio.from_signals(test, entries, exits)
        else:
            results = momentum_research(train, param_grid["lookback"])
            best = results.sort_values("sharpe", ascending=False).head(1)
            best_params = best.iloc[0]
            lb = int(best_params["lookback"])
            mom = test / test.shift(lb) - 1.0
            entries = (mom > 0).shift(1).fillna(False)
            exits = (mom <= 0).shift(1).fillna(False)
            pf = vbt.Portfolio.from_signals(test, entries, exits)

        stats = _pf_stats(pf)
        segments.append({
            "start": test.index[0],
            "end": test.index[-1],
            **stats,
        })
        best_rows.append(best_params.to_dict())
        start_idx += step_days

    seg_df = pd.DataFrame(segments)
    best_df = pd.DataFrame(best_rows)
    return seg_df, best_df
