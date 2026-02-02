"""
Black-Scholes greeks and implied vol with safe fallback.
"""
import math
from typing import Optional


try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _pv_implied_vol
    from py_vollib.black_scholes.greeks.analytical import delta as _pv_delta
    from py_vollib.black_scholes.greeks.analytical import gamma as _pv_gamma
    from py_vollib.black_scholes.greeks.analytical import theta as _pv_theta
    from py_vollib.black_scholes.greeks.analytical import vega as _pv_vega
    _HAS_PV = True
except Exception:
    _HAS_PV = False


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(spot: float, strike: float, t: float, r: float, iv: float) -> Optional[float]:
    if spot <= 0 or strike <= 0 or iv <= 0 or t <= 0:
        return None
    return (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))


def _d2(d1: float, iv: float, t: float) -> float:
    return d1 - iv * math.sqrt(t)


def _bs_price(flag: str, spot: float, strike: float, t: float, r: float, iv: float) -> float:
    d1 = _d1(spot, strike, t, r, iv)
    if d1 is None:
        return float("nan")
    d2 = _d2(d1, iv, t)
    if flag == "c":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _implied_vol_bisect(price: float, spot: float, strike: float, t: float, r: float, flag: str) -> Optional[float]:
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None
    low, high = 1e-6, 5.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        mid_price = _bs_price(flag, spot, strike, t, r, mid)
        if not math.isfinite(mid_price):
            return None
        if mid_price > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _safe_iv(option_price: float, spot: float, strike: float, t: float, r: float, flag: str) -> Optional[float]:
    try:
        if option_price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
            return None
        if _HAS_PV:
            iv = _pv_implied_vol(option_price, spot, strike, t, r, flag)
        else:
            iv = _implied_vol_bisect(option_price, spot, strike, t, r, flag)
        if iv is not None and math.isfinite(iv) and iv > 0:
            return float(iv)
    except Exception:
        return None
    return None


def greeks_from_price(
    option_price: float,
    spot: float,
    strike: float,
    dte: int,
    risk_free_rate: float,
    flag: str,
):
    t = dte / 252.0
    iv = _safe_iv(option_price, spot, strike, t, risk_free_rate, flag)
    if iv is None:
        return {"iv": None, "delta": None, "gamma": None, "theta": None, "vega": None}

    if _HAS_PV:
        return {
            "iv": iv,
            "delta": float(_pv_delta(flag, spot, strike, t, risk_free_rate, iv)),
            "gamma": float(_pv_gamma(flag, spot, strike, t, risk_free_rate, iv)),
            "theta": float(_pv_theta(flag, spot, strike, t, risk_free_rate, iv)),
            "vega": float(_pv_vega(flag, spot, strike, t, risk_free_rate, iv)),
        }

    d1 = _d1(spot, strike, t, risk_free_rate, iv)
    if d1 is None:
        return {"iv": None, "delta": None, "gamma": None, "theta": None, "vega": None}
    d2 = _d2(d1, iv, t)

    if flag == "c":
        delta = _norm_cdf(d1)
        theta = (-spot * _norm_pdf(d1) * iv / (2 * math.sqrt(t)) - risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-spot * _norm_pdf(d1) * iv / (2 * math.sqrt(t)) + risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2))

    gamma = _norm_pdf(d1) / (spot * iv * math.sqrt(t))
    vega = spot * _norm_pdf(d1) * math.sqrt(t)

    return {
        "iv": iv,
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }
