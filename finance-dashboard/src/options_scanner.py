"""
Options scanner utilities for call flow and debit spreads.
"""
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from src.market_data import get_provider

from src.options_engine import contract_mid, contract_liquidity_metrics, fetch_chain
from src.quant import get_terminal_distribution


def _parse_expiries(expiries: List[str]) -> List[Dict]:
    out = []
    for e in expiries:
        try:
            dt = datetime.strptime(e, "%Y-%m-%d")
            out.append({"expiry": e, "date": dt})
        except Exception:
            continue
    return out


def scan_ticker_for_calls(ticker: str, min_dte: int = 20, max_dte: int = 60, max_expiries: int = 6):
    provider = get_provider()
    expiries = provider.get_option_expiries(ticker)
    if not expiries:
        return pd.DataFrame(), pd.DataFrame()

    exp_list = _parse_expiries(expiries)
    today = datetime.utcnow().date()
    exp_list = [e for e in exp_list if min_dte <= (e["date"].date() - today).days <= max_dte]
    exp_list = exp_list[:max_expiries]

    frames = []
    for e in exp_list:
        calls, _ = fetch_chain(ticker, e["expiry"])
        if calls.empty:
            continue
        calls = calls.copy()
        calls["expiry"] = e["expiry"]
        calls["mid"] = calls.apply(contract_mid, axis=1)
        calls["spread_pct"] = calls.apply(lambda r: contract_liquidity_metrics(r)["spread_pct"], axis=1)
        calls["unusual_volume"] = calls["volume"] / calls["openInterest"].replace(0, 1)
        frames.append(calls)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_calls = pd.concat(frames, ignore_index=True)
    top_vol = all_calls.sort_values("volume", ascending=False).head(15)
    top_unusual = all_calls.sort_values("unusual_volume", ascending=False).head(15)
    return top_vol, top_unusual


def _liquidity_score(row) -> float:
    spread_pct = row.get("spread_pct", np.nan)
    oi = row.get("openInterest", 0) or 0
    vol = row.get("volume", 0) or 0

    spread_score = max(0.0, 1.0 - min(float(spread_pct), 0.2) / 0.1) if pd.notna(spread_pct) else 0.0
    oi_score = min(1.0, oi / 1000.0)
    vol_score = min(1.0, vol / 500.0)
    return 100.0 * (0.5 * spread_score + 0.3 * oi_score + 0.2 * vol_score)


def _flow_score(row) -> float:
    uv = row.get("unusual_volume", 0.0) or 0.0
    return 100.0 * min(1.0, uv / 2.0)


def build_call_debit_candidates(
    ticker: str,
    expiry: str,
    min_oi: int = 300,
    min_vol: int = 100,
    max_spread_pct: float = 0.10,
    n_paths: int = 10000,
    horizon_days: int = 30,
):
    calls, _ = fetch_chain(ticker, expiry)
    if calls.empty:
        return []

    calls = calls.copy()
    calls["mid"] = calls.apply(contract_mid, axis=1)
    calls["spread_pct"] = calls.apply(lambda r: contract_liquidity_metrics(r)["spread_pct"], axis=1)
    calls["unusual_volume"] = calls["volume"] / calls["openInterest"].replace(0, 1)

    # Liquidity filters
    filt = (
        (calls["bid"] > 0) &
        (calls["ask"] > 0) &
        (calls["spread_pct"] <= max_spread_pct) &
        (calls["openInterest"] >= min_oi) &
        (calls["volume"] >= min_vol)
    )
    calls = calls[filt]
    if calls.empty:
        return []

    provider = get_provider()
    spot = provider.get_spot(ticker)
    if not np.isfinite(spot):
        spot = float(calls["lastPrice"].dropna().iloc[0])
    strikes = sorted(calls["strike"].unique())
    strikes = [s for s in strikes if s >= spot * 0.9 and s <= spot * 1.2]
    if not strikes:
        return []

    diffs = pd.Series(strikes).diff().dropna()
    step = float(diffs.median()) if not diffs.empty else 5.0
    steps = [step, 2 * step, 3 * step]

    terminal = get_terminal_distribution(ticker, horizon_days, n_paths=n_paths)
    if terminal.size == 0:
        return []

    candidates = []
    for long_strike in strikes:
        for s in steps:
            short_strike = long_strike + s
            long_row = calls.loc[(calls["strike"] - long_strike).abs().idxmin()]
            short_row = calls.loc[(calls["strike"] - short_strike).abs().idxmin()]
            if long_row.empty or short_row.empty:
                continue

            long_mid = contract_mid(long_row)
            short_mid = contract_mid(short_row)
            if not np.isfinite(long_mid) or not np.isfinite(short_mid):
                continue

            debit = long_mid - short_mid
            if debit <= 0:
                continue

            width = float(short_row["strike"] - long_row["strike"])
            max_profit = max(width - debit, 0.0)
            max_loss = max(debit, 0.0)
            breakeven = float(long_row["strike"] + debit)

            payoff = np.clip(terminal - long_row["strike"], 0, width) - debit
            ev = float(np.mean(payoff))
            p_be = float(np.mean(payoff > 0))
            p50 = float(np.percentile(payoff, 50))

            liq = _liquidity_score(long_row) * 0.5 + _liquidity_score(short_row) * 0.5
            flow = _flow_score(long_row) * 0.5 + _flow_score(short_row) * 0.5
            edge = 100.0 * (0.6 * p_be + 0.4 * max(min(ev / max_loss, 1.0), -1.0))
            score = 0.35 * liq + 0.25 * flow + 0.40 * edge

            candidates.append({
                "ticker": ticker,
                "expiry": expiry,
                "long_strike": float(long_row["strike"]),
                "short_strike": float(short_row["strike"]),
                "debit": debit,
                "width": width,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
                "ev": ev,
                "p_be": p_be,
                "p50": p50,
                "liq_score": liq,
                "flow_score": flow,
                "edge_score": edge,
                "score": score,
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:10]
