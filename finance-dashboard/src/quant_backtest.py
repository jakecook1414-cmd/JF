from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    metrics: Dict[str, float]
    trades: int


@dataclass
class WalkForwardResult:
    segment_metrics: pd.DataFrame
    overall_metrics: Dict[str, float]
    equity_curve: pd.Series


def _clean_series(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.sort_index()


def sma_crossover_signal(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    prices = _clean_series(prices)
    if len(prices) < slow + 2:
        return pd.Series(dtype=float)
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = (sma_fast > sma_slow).astype(int)
    return signal.shift(1).fillna(0)


def momentum_signal(prices: pd.Series, lookback: int = 252) -> pd.Series:
    prices = _clean_series(prices)
    if len(prices) < lookback + 2:
        return pd.Series(dtype=float)
    mom = prices / prices.shift(lookback) - 1.0
    signal = (mom > 0).astype(int)
    return signal.shift(1).fillna(0)


def _apply_costs(positions: pd.Series, cost_bps: float, slippage_bps: float) -> pd.Series:
    if positions.empty:
        return positions
    delta = positions.diff().abs().fillna(0)
    total_bps = float(cost_bps + slippage_bps) / 10000.0
    return delta * total_bps


def _metrics_from_returns(returns: pd.Series, positions: pd.Series) -> Tuple[Dict[str, float], int]:
    if returns.empty:
        return {
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
        }, 0

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

    turnover = float(positions.diff().abs().sum() / max(len(positions), 1))
    trades = int((positions.diff().abs() > 0).sum())

    metrics = {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "vol": vol,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "turnover": turnover,
    }
    return metrics, trades


def run_backtest(
    prices: pd.Series,
    signal_fn: Callable[..., pd.Series],
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    **kwargs,
) -> BacktestResult:
    prices = _clean_series(prices)
    if prices.empty:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float), {}, 0)

    positions = signal_fn(prices, **kwargs)
    positions = positions.reindex(prices.index).fillna(0)
    daily_ret = prices.pct_change().fillna(0)

    costs = _apply_costs(positions, cost_bps, slippage_bps)
    strat_ret = (daily_ret * positions) - costs
    strat_ret.name = "strategy_return"

    equity = (1 + strat_ret).cumprod()
    metrics, trades = _metrics_from_returns(strat_ret, positions)
    metrics["trades"] = trades
    return BacktestResult(strat_ret, equity, metrics, trades)


def walk_forward_eval(
    prices: pd.Series,
    signal_fn: Callable[..., pd.Series],
    train_days: int = 756,
    test_days: int = 126,
    step_days: int = 126,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    **kwargs,
) -> WalkForwardResult:
    prices = _clean_series(prices)
    if len(prices) < train_days + test_days + 5:
        empty = pd.DataFrame(columns=[
            "start", "end", "cagr", "max_drawdown", "vol", "sharpe", "win_rate", "turnover", "trades"
        ])
        return WalkForwardResult(empty, _metrics_from_returns(pd.Series(dtype=float), pd.Series(dtype=float))[0], pd.Series(dtype=float))

    segment_rows = []
    all_returns = []
    all_positions = []

    start_idx = train_days
    while start_idx + test_days <= len(prices):
        train_slice = prices.iloc[start_idx - train_days:start_idx]
        test_slice = prices.iloc[start_idx:start_idx + test_days]

        history = pd.concat([train_slice, test_slice])
        positions = signal_fn(history, **kwargs)
        positions = positions.reindex(test_slice.index).fillna(0)

        daily_ret = test_slice.pct_change().fillna(0)
        costs = _apply_costs(positions, cost_bps, slippage_bps)
        strat_ret = (daily_ret * positions) - costs
        strat_ret.name = "strategy_return"

        metrics, trades = _metrics_from_returns(strat_ret, positions)
        metrics["trades"] = trades
        segment_rows.append({
            "start": test_slice.index[0],
            "end": test_slice.index[-1],
            **metrics,
        })
        all_returns.append(strat_ret)
        all_positions.append(positions)

        start_idx += step_days

    if all_returns:
        all_returns = pd.concat(all_returns).sort_index()
        all_positions = pd.concat(all_positions).sort_index()
    else:
        all_returns = pd.Series(dtype=float)
        all_positions = pd.Series(dtype=float)

    overall_metrics, trades = _metrics_from_returns(all_returns, all_positions)
    overall_metrics["trades"] = trades
    equity = (1 + all_returns).cumprod() if not all_returns.empty else pd.Series(dtype=float)

    segment_df = pd.DataFrame(segment_rows)
    return WalkForwardResult(segment_df, overall_metrics, equity)
