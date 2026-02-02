import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.market_data import get_provider

from src.alerts import append_alert_log, evaluate_alerts, load_alerts, save_alerts, scanner_can_trigger
from src.data_provider import YahooProvider
from src.options_engine import fetch_chain, price_call_debit_spread, price_long_option
from src.options_greeks import greeks_from_price
from src.options_scanner import build_call_debit_candidates, scan_ticker_for_calls
from src.quant import dte_from_expiry as trading_dte, get_terminal_distribution
from src.quant_backtest import momentum_signal, run_backtest, sma_crossover_signal, walk_forward_eval
from src.quant_forecast import prob_above, simulate_bootstrap, simulate_regime_bootstrap, terminal_percentiles
from src.quant_validation import calibration_report, coverage_summary, drawdown_curve, realized_volatility, returns_distribution
from src.factor_regression import run_factor_regression
from src.ai_assistant import build_snapshot_payload, run_ai_assistant

st.set_page_config(page_title="Jake's Investment Dashboard", layout="wide")
load_dotenv()

st.title("Investment Dashboard")
st.caption("Local-only MVP: watchlist, charts, portfolio PnL. Not financial advice.")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
provider = get_provider()

if st.sidebar.button("Restart App"):
    if not st.session_state.get("restarted"):
        st.session_state["restarted"] = True
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
        st.stop()


default_watchlist = ["SPY", "QQQ", "NVDA", "AMD", "AVGO", "PLTR", "TSLA", "GLD", "FCX"]
watchlist = st.sidebar.text_area(
    "Watchlist tickers (comma-separated)",
    value=",".join(default_watchlist)
).upper().replace(" ", "")

tickers = [t for t in watchlist.split(",") if t]

range_map = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "6M": 180,
    "1Y": 365,
    "2Y": 730,
    "5Y": 1825
}
range_label = st.sidebar.selectbox("Time range", list(range_map.keys()), index=4)
days = range_map[range_label]

end = datetime.today()
start = end - timedelta(days=days)

st.sidebar.divider()
st.sidebar.subheader("Portfolio (manual)")
st.sidebar.caption("Enter positions below. Shares + cost basis are used for PnL.")

positions_csv = st.sidebar.text_area(
    "Positions CSV (ticker,shares,cost_basis)",
    value="NVDA,2,450\nAMD,5,170\nGLD,3,185\nFCX,6,41",
    height=140
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=600)
def get_prices(tickers_list, start_date, end_date):
    return provider.get_prices(tickers_list, start_date, end_date)


quant_provider = YahooProvider()


@st.cache_data(ttl=600)
def get_quant_series(ticker, start_date, end_date):
    df = quant_provider.get_prices(ticker, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    col = ticker if ticker in df.columns else df.columns[0]
    return df[col].dropna()


@st.cache_data(ttl=600)
def run_forecast(series, horizon_days, n_paths, model):
    if model == "regime":
        return simulate_regime_bootstrap(series, horizon_days, n_paths)
    return simulate_bootstrap(series, horizon_days, n_paths)


@st.cache_data(ttl=600)
def run_calibration(series, horizon_days, lookback_days, n_paths, model):
    return calibration_report(series, horizon_days, lookback_days, n_paths, model)

@st.cache_data(ttl=600)
def run_factors(asset, factors, start_date, end_date, min_obs):
    return run_factor_regression(asset, factors, start_date, end_date, min_obs=min_obs)

METRIC_HELP = {
    "cagr": "CAGR: annualized growth rate of the strategy. Higher is generally better.",
    "max_drawdown": "Max drawdown: largest peak-to-trough loss. Smaller (less negative) is better.",
    "vol": "Volatility: annualized return variability. Lower means smoother, but can imply lower returns.",
    "sharpe": "Sharpe: return per unit of volatility. Higher is better; >1 is often good.",
    "win_rate": "Win rate: % of positive return days/trades. Higher is better, but not the only metric.",
    "turnover": "Turnover: how often positions change. Lower means fewer trades and costs.",
    "trades": "Trades: number of position changes in the period.",
    "coverage_80": "Coverage (80% band): how often realized prices fell within the 80% forecast band.",
    "coverage_90": "Coverage (90% band): how often realized prices fell within the 90% forecast band.",
    "alpha": "Alpha: return not explained by factors. Positive is good, but not guaranteed.",
    "r2": "R²: fraction of returns explained by the factors. Higher means more explained.",
}


def metric_with_help(label, value, key, delta=None):
    st.metric(label, value, delta=delta, help=METRIC_HELP.get(key, ""))


def metrics_help_expander(keys):
    with st.expander("What these metrics mean", expanded=False):
        for key in keys:
            text = METRIC_HELP.get(key)
            if text:
                st.caption(text)


def backtest_metrics_from_returns(returns: pd.Series, positions: pd.Series) -> dict:
    if returns.empty:
        return {
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "trades": 0,
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
    turnover = float(positions.diff().abs().sum() / max(len(positions), 1))
    trades = int((positions.diff().abs() > 0).sum())
    return {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "vol": vol,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "turnover": turnover,
        "trades": trades,
    }

def build_ai_snapshot(watch_df, positions_df, range_label, factor_summary):
    watch = watch_df.reset_index().rename(columns={"index": "ticker"})
    watch = watch[["ticker", "Price", "Day Change ($)", "Day Change (%)", f"Return ({range_label}) %"]]
    watch = watch.sort_values(by=f"Return ({range_label}) %", ascending=False)
    top_watch = watch.head(8).to_dict(orient="records")
    laggards = watch.tail(5).to_dict(orient="records")

    positions = positions_df[[
        "ticker",
        "shares",
        "cost_basis",
        "last_price",
        "market_value",
        "pnl_$",
        "pnl_%",
        "weight_%",
    ]].sort_values("weight_%", ascending=False)
    positions = positions.head(12).to_dict(orient="records")

    return {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "range_label": range_label,
        "watchlist_top": top_watch,
        "watchlist_laggards": laggards,
        "portfolio": positions,
        "factor_regression": factor_summary,
    }

def extract_close(downloaded, tickers_list):
    # yf.download returns different shapes depending on count of tickers
    if len(tickers_list) == 1:
        close = downloaded["Close"].to_frame(name=tickers_list[0])
        return close

    frames = []
    for t in tickers_list:
        if (t in downloaded.columns.get_level_values(0)):
            frames.append(downloaded[t]["Close"].rename(t))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")


def parse_positions(csv_text):
    rows = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().upper() for p in line.split(",")]
        if len(parts) != 3:
            continue
        t, shares, cb = parts
        try:
            rows.append({"ticker": t, "shares": float(shares), "cost_basis": float(cb)})
        except ValueError:
            continue
    return pd.DataFrame(rows)


def pct_change(series):
    return (series.iloc[-1] / series.iloc[0] - 1.0) * 100.0


def build_positions_table(positions_csv_text, close_prices):
    positions_df = parse_positions(positions_csv_text)
    if positions_df.empty:
        for col in ["last_price", "market_value", "cost_value", "pnl_$", "pnl_%", "weight_%"]:
            positions_df[col] = pd.Series(dtype=float)
        return positions_df, close_prices

    pos_tickers = positions_df["ticker"].unique().tolist()
    missing = [t for t in pos_tickers if t not in close_prices.columns]

    if missing:
        raw2 = get_prices(missing, start, end)
        close2 = extract_close(raw2, missing)
        close_prices = close_prices.join(close2, how="outer")

    last_prices = close_prices.ffill().iloc[-1]
    positions_df["last_price"] = positions_df["ticker"].map(last_prices.to_dict())
    positions_df["market_value"] = positions_df["shares"] * positions_df["last_price"]
    positions_df["cost_value"] = positions_df["shares"] * positions_df["cost_basis"]
    positions_df["pnl_$"] = positions_df["market_value"] - positions_df["cost_value"]
    positions_df["pnl_%"] = (positions_df["pnl_$"] / positions_df["cost_value"]) * 100

    total_mv = positions_df["market_value"].sum()
    if total_mv > 0:
        positions_df["weight_%"] = (positions_df["market_value"] / total_mv) * 100
    else:
        positions_df["weight_%"] = 0.0

    return positions_df, close_prices


def load_csv_with_columns(path, columns):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def dte_from_expiry(expiry_str):
    return trading_dte(expiry_str)


# -----------------------------
# Pull data
# -----------------------------
if not tickers:
    st.warning("Add at least one ticker to the watchlist.")
    st.stop()

alerts_path = os.path.join("data", "alerts.json")
alerts = load_alerts(alerts_path)
alert_tickers = sorted({str(a.get("ticker", "")).upper() for a in alerts if a.get("ticker")})
data_tickers = sorted(set(tickers) | set(alert_tickers))

raw = get_prices(data_tickers, start, end)
close = extract_close(raw, data_tickers)

if close.empty:
    st.error("No price data returned. Check tickers or try a different range.")
    st.stop()

# -----------------------------
# Watchlist Table
# -----------------------------
latest = close.iloc[-1]
prev = close.iloc[-2] if len(close) > 1 else close.iloc[-1]

close_ffill = close.ffill()
latest_prices = close_ffill.iloc[-1].to_dict()
prev_prices = close_ffill.iloc[-2] if len(close_ffill) > 1 else close_ffill.iloc[-1]
day_pct_moves = ((close_ffill.iloc[-1] / prev_prices) - 1.0) * 100.0
day_pct_moves = day_pct_moves.to_dict()

now_utc = datetime.now(timezone.utc)
alerts, triggered_events = evaluate_alerts(alerts, latest_prices, day_pct_moves, now_utc)
save_alerts(alerts, alerts_path)
append_alert_log(triggered_events, os.path.join("data", "alerts_log.csv"))

watch_df = pd.DataFrame({
    "Price": latest,
    "Day Change ($)": (latest - prev),
    "Day Change (%)": ((latest / prev - 1) * 100)
})

# Range return
range_returns = close.apply(pct_change, axis=0).rename(f"Return ({range_label}) %")
watch_df = watch_df.join(range_returns)

watch_df = watch_df.sort_values(by=f"Return ({range_label}) %", ascending=False)

c1, c2 = st.columns([1.1, 0.9], gap="large")

with c1:
    st.subheader("Watchlist")
    st.dataframe(
        watch_df.style.format({
            "Price": "{:.2f}",
            "Day Change ($)": "{:.2f}",
            "Day Change (%)": "{:.2f}",
            f"Return ({range_label}) %": "{:.2f}"
        }),
        use_container_width=True,
        height=420
    )

with c2:
    st.subheader("Price Chart")
    selected = st.selectbox("Chart ticker", tickers, index=0)
    fig = plt.figure()
    plt.plot(close.index, close[selected])
    plt.title(f"{selected} (Adj Close) - {range_label}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig, clear_figure=True)

st.divider()

# Build positions snapshot for AI/portfolio sections
positions_table, close = build_positions_table(positions_csv, close)

# -----------------------------
# Tabs: Alerts / Quant / Options
# -----------------------------
alerts_tab, quant_tab, options_tab = st.tabs(["Alerts", "Quant", "Options"])

with alerts_tab:
    if triggered_events:
        lines = [f"{e['ticker']} {e['type']} @ {e['value']} (price {e.get('price')})" for e in triggered_events]
        st.warning("Alerts triggered: " + " | ".join(lines))

    st.subheader("Alerts")
    with st.expander("Manage Alerts", expanded=False):
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            display_cols = ["id", "ticker", "type", "value", "enabled", "cooldown_minutes", "last_triggered"]
            for col in display_cols:
                if col not in alerts_df.columns:
                    alerts_df[col] = ""
            st.dataframe(alerts_df[display_cols], use_container_width=True, height=240)

            st.caption("Toggle or delete alerts below.")
            for alert in alerts:
                cols = st.columns([1.6, 0.8, 0.8, 0.8])
                cols[0].write(
                    f"{alert.get('id','')} | {alert.get('ticker','')} | {alert.get('type','')} | {alert.get('value','')}"
                )
                toggle_label = "Disable" if alert.get("enabled", True) else "Enable"
                if cols[1].button(toggle_label, key=f"toggle-{alert.get('id')}"):
                    alert["enabled"] = not alert.get("enabled", True)
                    save_alerts(alerts, alerts_path)
                    st.rerun()
                if cols[2].button("Delete", key=f"delete-{alert.get('id')}"):
                    alerts = [a for a in alerts if a.get("id") != alert.get("id")]
                    save_alerts(alerts, alerts_path)
                    st.rerun()
        else:
            st.info("No active alerts yet.")

        st.markdown("**Add alert**")
        with st.form("add_alert_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            ticker = c1.text_input("Ticker").upper().strip()
            alert_type = c2.selectbox("Type", ["price_above", "price_below", "pct_move_day"])
            value = c3.number_input("Value", min_value=0.0, step=0.1)
            c4, c5 = st.columns(2)
            cooldown = c4.number_input("Cooldown (minutes)", min_value=0, step=1, value=60)
            enabled = c5.checkbox("Enabled", value=True)
            submitted = st.form_submit_button("Add Alert")
            if submitted:
                new_alert = {
                    "id": uuid.uuid4().hex[:8],
                    "ticker": ticker,
                    "type": alert_type,
                    "value": float(value),
                    "enabled": bool(enabled),
                    "cooldown_minutes": int(cooldown),
                    "last_triggered": "",
                }
                alerts.append(new_alert)
                save_alerts(alerts, alerts_path)
                st.success("Alert added.")
                st.rerun()

        st.markdown("**Triggered alerts feed**")
        log_path = os.path.join("data", "alerts_log.csv")
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            if not log_df.empty:
                log_df = log_df.sort_values("timestamp", ascending=False)
                st.dataframe(log_df, use_container_width=True, height=240)
            else:
                st.caption("No triggered alerts yet.")
        else:
            st.caption("No triggered alerts yet.")

with quant_tab:
    st.subheader("Quant Toolkit")
    intent = st.radio(
        "What do you want to answer today?",
        [
            "What could the price be by X date? (Forecast)",
            "Would this strategy have worked historically? (Backtest)",
            "Is my forecast model trustworthy? (Validation/Calibration)",
            "What drives this ticker? (Factors/Exposures)",
        ],
        horizontal=False,
    )
    beginner_mode = st.toggle("Beginner mode", value=True)

    st.info(
        "Assumptions & limitations\n"
        "- Uses historical distributions; does not know future news/earnings\n"
        "- Assumes execution at close (or next bar) as modeled\n"
        "- Slippage is ignored unless explicitly set\n"
        "- Outputs are scenarios, not guarantees\n"
        "- Model fit does not imply causal drivers\n"
    )

    qc1, qc2, qc3 = st.columns(3)
    qticker = qc1.selectbox("Ticker", data_tickers, index=0, key="quant_ticker")
    default_start = (datetime.today() - timedelta(days=365 * 5)).date()
    qstart = qc2.date_input("Start date", value=default_start, key="quant_start")
    qend = qc3.date_input("End date", value=datetime.today().date(), key="quant_end")

    series = pd.Series(dtype=float)
    if qstart >= qend:
        st.warning("Start date must be earlier than end date.")
    else:
        series = get_quant_series(qticker, qstart, qend)
        if series.empty:
            st.warning("Not enough data for the selected range.")

    st.caption("Use the tabs below or follow the guided flow based on your selected question.")
    intent_map = {
        "What could the price be by X date? (Forecast)": "Forecast",
        "Would this strategy have worked historically? (Backtest)": "Backtest",
        "Is my forecast model trustworthy? (Validation/Calibration)": "Validation",
        "What drives this ticker? (Factors/Exposures)": "Factors",
    }
    st.info(f"Suggested tab: **{intent_map.get(intent, 'Forecast')}**")
    forecast_tab, backtest_tab, validation_tab, factors_tab, ai_tab = st.tabs(
        ["Forecast", "Backtest", "Validation", "Factors", "AI"]
    )

    with forecast_tab:
        st.subheader("Forecast: What could the price be by X date?")
        preset = st.selectbox(
            "Preset",
            ["Swing (20d)", "Position (60d)", "Longer-term (120d)", "Custom"],
            index=0,
            key="forecast_preset",
        )
        preset_map = {
            "Swing (20d)": {"horizon": 20, "paths": 10000, "model": "bootstrap"},
            "Position (60d)": {"horizon": 60, "paths": 10000, "model": "bootstrap"},
            "Longer-term (120d)": {"horizon": 120, "paths": 20000, "model": "regime"},
        }
        if preset in preset_map:
            st.session_state["forecast_horizon"] = preset_map[preset]["horizon"]
            st.session_state["forecast_paths"] = preset_map[preset]["paths"]
            st.session_state["forecast_model"] = preset_map[preset]["model"]
        horizon = st.selectbox(
            "Horizon (trading days)",
            [5, 20, 60, 120],
            index=1,
            key="forecast_horizon",
        )
        last_price = float(series.iloc[-1]) if not series.empty else 0.0
        target = st.number_input("Target price", min_value=0.0, step=1.0, value=last_price)

        if beginner_mode:
            model = st.session_state.get("forecast_model", "bootstrap")
            n_paths = st.session_state.get("forecast_paths", 10000)
            with st.expander("Advanced settings", expanded=False):
                model = st.selectbox("Model", ["bootstrap", "regime"], index=0, key="forecast_model")
                n_paths = st.slider(
                    "Paths", min_value=2000, max_value=20000, step=1000, value=10000, key="forecast_paths"
                )
        else:
            model = st.selectbox("Model", ["bootstrap", "regime"], index=0, key="forecast_model")
            n_paths = st.slider(
                "Paths", min_value=2000, max_value=20000, step=1000, value=10000, key="forecast_paths"
            )

        if series.empty:
            st.caption("Load a ticker and date range to run forecasts.")
        else:
            paths, terminal = run_forecast(series, int(horizon), int(n_paths), model)
            if terminal.size == 0:
                st.warning("Not enough data to run simulation.")
            else:
                percentiles = terminal_percentiles(terminal)
                p25 = percentiles.get(25, 0.0)
                p75 = percentiles.get(75, 0.0)
                p50 = percentiles.get(50, float(np.percentile(terminal, 50)))
                p10 = float(np.percentile(terminal, 10))
                p90 = float(np.percentile(terminal, 90))
                p5 = percentiles.get(5, 0.0)
                p95 = percentiles.get(95, 0.0)

                c1, c2, c3 = st.columns(3)
                c1.metric("Expected range (P25–P75)", f"{p25:.2f} – {p75:.2f}")
                c2.metric("Conservative range (P10–P90)", f"{p10:.2f} – {p90:.2f}")
                if not beginner_mode:
                    c3.metric("Tail range (P5–P95)", f"{p5:.2f} – {p95:.2f}")
                else:
                    with c3:
                        with st.expander("Tail range (P5–P95)"):
                            st.write(f"{p5:.2f} – {p95:.2f}")

                prob_current = prob_above(terminal, float(series.iloc[-1]))
                prob_target = prob_above(terminal, float(target))
                st.metric("P(terminal > current)", f"{prob_current*100:.1f}%")
                st.metric("P(terminal > target)", f"{prob_target*100:.1f}%")

                dates = pd.bdate_range(start=series.index[-1], periods=len(paths[0]))
                p10_series = np.percentile(paths, 10, axis=0)
                p50_series = np.percentile(paths, 50, axis=0)
                p90_series = np.percentile(paths, 90, axis=0)

                st.caption("Forecast fan chart (calendar dates)")
                fig3 = plt.figure()
                plt.fill_between(dates, p10_series, p90_series, color="#6baed6", alpha=0.25, label="P10–P90")
                plt.plot(dates, p50_series, color="#08519c", label="P50")
                plt.title(f"{qticker} Forecast ({horizon} trading days)")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                st.pyplot(fig3, clear_figure=True)

                st.caption("Terminal distribution at horizon")
                figh = plt.figure()
                plt.hist(terminal, bins=50, color="steelblue", alpha=0.7)
                plt.axvline(p50, color="black", linewidth=1, label="P50")
                plt.axvline(p10, color="gray", linestyle="--", linewidth=1, label="P10/P90")
                plt.axvline(p90, color="gray", linestyle="--", linewidth=1)
                plt.legend()
                plt.xlabel("Terminal price")
                plt.ylabel("Frequency")
                st.pyplot(figh, clear_figure=True)

    with backtest_tab:
        st.subheader("Backtest: Would this strategy have worked historically?")
        preset = st.selectbox(
            "Preset",
            ["SMA 50/200", "Momentum 252d", "Custom"],
            index=0,
            key="backtest_preset",
        )
        if preset == "SMA 50/200":
            st.session_state["bt_strategy"] = "SMA Crossover"
            st.session_state["bt_fast"] = 50
            st.session_state["bt_slow"] = 200
        elif preset == "Momentum 252d":
            st.session_state["bt_strategy"] = "Time-series Momentum"
            st.session_state["bt_lookback"] = 252

        strategy = st.selectbox("Strategy", ["SMA Crossover", "Time-series Momentum"], index=0, key="bt_strategy")
        params = {}
        signal_fn = sma_crossover_signal
        if strategy == "SMA Crossover":
            fast = st.number_input("Fast SMA", min_value=5, step=5, value=50, key="bt_fast")
            slow = st.number_input("Slow SMA", min_value=20, step=10, value=200, key="bt_slow")
            params = {"fast": int(fast), "slow": int(slow)}
            signal_fn = sma_crossover_signal
        else:
            lookback = st.number_input("Lookback (trading days)", min_value=20, step=10, value=252, key="bt_lookback")
            params = {"lookback": int(lookback)}
            signal_fn = momentum_signal

        if beginner_mode:
            cost_bps = 5.0
            slippage_bps = 2.0
            train_days = 756
            test_days = 126
            step_days = 126
            with st.expander("Advanced settings", expanded=False):
                cost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, step=1.0, value=5.0)
                slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, step=1.0, value=2.0)
                train_days = st.number_input("Train days", min_value=252, step=126, value=756)
                test_days = st.number_input("Test days", min_value=21, step=21, value=126)
                step_days = st.number_input("Step days", min_value=21, step=21, value=126)
        else:
            cost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, step=1.0, value=5.0)
            slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, step=1.0, value=2.0)
            train_days = st.number_input("Train days", min_value=252, step=126, value=756)
            test_days = st.number_input("Test days", min_value=21, step=21, value=126)
            step_days = st.number_input("Step days", min_value=21, step=21, value=126)

        if series.empty:
            st.caption("Load a ticker and date range to run backtests.")
        else:
            wf = walk_forward_eval(
                series,
                signal_fn,
                int(train_days),
                int(test_days),
                int(step_days),
                cost_bps=float(cost_bps),
                slippage_bps=float(slippage_bps),
                **params,
            )

            st.subheader("Overall Metrics (Walk-forward)")
            mcols = st.columns(4)
            with mcols[0]:
                metric_with_help("CAGR", f"{wf.overall_metrics.get('cagr', 0.0)*100:.2f}%", "cagr")
            with mcols[1]:
                metric_with_help("Max Drawdown", f"{wf.overall_metrics.get('max_drawdown', 0.0)*100:.2f}%", "max_drawdown")
            with mcols[2]:
                metric_with_help("Vol", f"{wf.overall_metrics.get('vol', 0.0)*100:.2f}%", "vol")
            with mcols[3]:
                metric_with_help("Sharpe", f"{wf.overall_metrics.get('sharpe', 0.0):.2f}", "sharpe")
            mcols2 = st.columns(3)
            with mcols2[0]:
                metric_with_help("Win Rate", f"{wf.overall_metrics.get('win_rate', 0.0)*100:.1f}%", "win_rate")
            with mcols2[1]:
                metric_with_help("Turnover", f"{wf.overall_metrics.get('turnover', 0.0):.2f}", "turnover")
            with mcols2[2]:
                metric_with_help("Trades", f"{int(wf.overall_metrics.get('trades', 0))}", "trades")

            metrics_help_expander(["cagr", "max_drawdown", "vol", "sharpe", "win_rate", "turnover", "trades"])

            full = run_backtest(series, signal_fn, cost_bps=float(cost_bps), slippage_bps=float(slippage_bps), **params)
            if not full.equity_curve.empty:
                daily_ret = series.pct_change().fillna(0)
                bench_equity = (1 + daily_ret).cumprod()
                st.subheader("Equity Curve vs Buy-and-Hold")
                fig4 = plt.figure()
                plt.plot(full.equity_curve.index, full.equity_curve.values, label="Strategy")
                plt.plot(bench_equity.index, bench_equity.values, label="Buy & Hold", linestyle="--")
                plt.xlabel("Date")
                plt.ylabel("Equity")
                plt.legend()
                st.pyplot(fig4, clear_figure=True)

                bench_metrics = backtest_metrics_from_returns(daily_ret, pd.Series(1, index=daily_ret.index))
                st.caption("Benchmark: Buy-and-Hold (same ticker)")
                bcols = st.columns(4)
                bcols[0].metric("CAGR", f"{bench_metrics.get('cagr', 0.0)*100:.2f}%")
                bcols[1].metric("Max Drawdown", f"{bench_metrics.get('max_drawdown', 0.0)*100:.2f}%")
                bcols[2].metric("Vol", f"{bench_metrics.get('vol', 0.0)*100:.2f}%")
                bcols[3].metric("Sharpe", f"{bench_metrics.get('sharpe', 0.0):.2f}")

            if beginner_mode:
                with st.expander("Detailed walk-forward segments", expanded=False):
                    if not wf.segment_metrics.empty:
                        st.dataframe(wf.segment_metrics, use_container_width=True, height=240)
                    else:
                        st.caption("Not enough history for walk-forward segments.")
            else:
                st.subheader("Segment Metrics")
                if not wf.segment_metrics.empty:
                    st.dataframe(wf.segment_metrics, use_container_width=True, height=240)
                else:
                    st.caption("Not enough history for walk-forward segments.")

    with validation_tab:
        st.subheader("Calibration & Diagnostics")
        st.caption(
            "Coverage measures how often realized prices fall inside forecast bands. "
            "If an 80% band only captures ~60% of outcomes, the model is overconfident."
        )
        calib_model = st.selectbox("Model", ["bootstrap", "regime"], index=0, key="calib_model")
        horizon_days = st.selectbox("Horizon (trading days)", [5, 20, 60, 120], index=1, key="calib_horizon")
        calib_paths = st.number_input("Paths per window", min_value=500, step=500, value=2000)
        lookback_days = st.number_input("Lookback days", min_value=252, step=126, value=756)

        if series.empty:
            st.caption("Load a ticker and date range to run calibration.")
        else:
            if st.button("Run calibration"):
                calib = run_calibration(series, int(horizon_days), int(lookback_days), int(calib_paths), calib_model)
                if calib.empty:
                    st.warning("Not enough history for calibration.")
                else:
                    cov = coverage_summary(calib)
                    metric_with_help(
                        "Coverage 80% band (P10–P90)",
                        f"{cov['cov_10_90']*100:.1f}%",
                        "coverage_80",
                    )
                    metric_with_help(
                        "Coverage 90% band (P5–P95)",
                        f"{cov['cov_5_95']*100:.1f}%",
                        "coverage_90",
                    )
                    metrics_help_expander(["coverage_80", "coverage_90"])

                    cov_ok = abs(cov["cov_10_90"] - 0.8) <= 0.1 and abs(cov["cov_5_95"] - 0.9) <= 0.1
                    cue = "OK" if cov_ok else "Needs tuning"
                    st.metric("Model confidence cue", cue)

                    def render_validation_charts():
                        figc = plt.figure()
                        plt.plot(calib["date"], calib["future"], label="Realized")
                        plt.plot(calib["date"], calib["p10"], label="P10", linestyle="--")
                        plt.plot(calib["date"], calib["p90"], label="P90", linestyle="--")
                        plt.plot(calib["date"], calib["p5"], label="P5", linestyle=":")
                        plt.plot(calib["date"], calib["p95"], label="P95", linestyle=":")
                        plt.legend()
                        plt.xlabel("Date")
                        plt.ylabel("Price")
                        st.pyplot(figc, clear_figure=True)

                        figcov = plt.figure()
                        cov_80 = calib["in_10_90"].rolling(20).mean()
                        cov_90 = calib["in_5_95"].rolling(20).mean()
                        plt.plot(calib["date"], cov_80, label="Rolling 20D coverage 80% band")
                        plt.plot(calib["date"], cov_90, label="Rolling 20D coverage 90% band")
                        plt.axhline(0.8, color="gray", linestyle="--", linewidth=1)
                        plt.axhline(0.9, color="gray", linestyle=":", linewidth=1)
                        plt.legend()
                        plt.xlabel("Date")
                        plt.ylabel("Coverage")
                        st.pyplot(figcov, clear_figure=True)

                    if beginner_mode:
                        with st.expander("Detailed calibration charts", expanded=False):
                            render_validation_charts()
                    else:
                        render_validation_charts()

            st.subheader("Diagnostics")
            vol20 = realized_volatility(series, 20)
            vol60 = realized_volatility(series, 60)
            def render_diagnostics():
                if not vol20.empty and not vol60.empty:
                    figv = plt.figure()
                    plt.plot(vol20.index, vol20.values, label="20D Vol")
                    plt.plot(vol60.index, vol60.values, label="60D Vol")
                    plt.legend()
                    plt.xlabel("Date")
                    plt.ylabel("Vol (annualized)")
                    st.pyplot(figv, clear_figure=True)

                rets, _ = returns_distribution(series)
                if rets.size > 0:
                    figh = plt.figure()
                    plt.hist(rets, bins=50, color="steelblue", alpha=0.7)
                    plt.xlabel("Daily Return")
                    plt.ylabel("Frequency")
                    st.pyplot(figh, clear_figure=True)

                dd = drawdown_curve(series)
                if not dd.empty:
                    figd = plt.figure()
                    plt.plot(dd.index, dd.values, color="firebrick")
                    plt.xlabel("Date")
                    plt.ylabel("Drawdown")
                    st.pyplot(figd, clear_figure=True)

            if beginner_mode:
                with st.expander("Diagnostics (advanced)", expanded=False):
                    render_diagnostics()
            else:
                render_diagnostics()

    with factors_tab:
        st.subheader("Factor Regression (OLS)")
        st.caption("Uses ETF factor proxies from Yahoo Finance. Betas are daily. Exposure ≠ causation.")
        fc1, fc2 = st.columns([1.2, 0.8])
        factor_text = fc1.text_input(
            "Factor tickers (comma-separated)",
            value="SPY,QQQ,IWM,TLT,HYG,GLD",
            key="factor_tickers",
        )
        min_obs = fc2.number_input("Min observations", min_value=30, step=10, value=60)

        factor_list = [f.strip().upper() for f in factor_text.split(",") if f.strip()]
        factor_list = [f for f in factor_list if f != qticker]

        if not factor_list:
            st.caption("Add at least one factor ticker (different from the asset ticker).")
        elif qstart >= qend:
            st.caption("Pick a valid date range to run the regression.")
        else:
            if st.button("Run factor regression"):
                with st.spinner("Running factor regression..."):
                    result = run_factors(qticker, tuple(factor_list), qstart, qend, int(min_obs))
                if result is None:
                    st.warning("Not enough data to run the regression. Try a longer date range.")
                else:
                    metrics = result.metrics
                    st.session_state["factor_summary"] = {
                        "asset": qticker,
                        "factors": factor_list,
                        "metrics": metrics,
                        "coefficients": result.coefficients.reset_index().rename(
                            columns={"index": "factor"}
                        ).to_dict(orient="records"),
                    }
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        metric_with_help("R²", f"{metrics.get('r2', 0.0):.3f}", "r2")
                    with mc2:
                        st.metric("Adj R²", f"{metrics.get('adj_r2', 0.0):.3f}")
                    with mc3:
                        metric_with_help("Alpha (annual)", f"{metrics.get('alpha_annual', 0.0)*100:.2f}%", "alpha")
                    with mc4:
                        st.metric("Info Ratio", f"{metrics.get('info_ratio', 0.0):.2f}")

                    coeffs = result.coefficients.copy()
                    if "const" in coeffs.index:
                        coeffs = coeffs.rename(index={"const": "alpha"})
                    coeff_rows = coeffs.reset_index().rename(columns={"index": "factor"})
                    exposure_lines = []
                    for _, row in coeff_rows.iterrows():
                        factor = row["factor"]
                        if factor == "alpha":
                            continue
                        exposure_lines.append(f"{factor}: ~{row['coef']*100:.0f}% sensitivity")
                    if exposure_lines:
                        st.subheader("Plain-language exposures")
                        st.caption("Approximate sensitivities based on historical co-movement.")
                        st.write(" • " + "\n • ".join(exposure_lines))

                    if beginner_mode:
                        with st.expander("Full coefficients table", expanded=False):
                            st.dataframe(
                                coeffs.style.format({"coef": "{:.4f}", "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
                                use_container_width=True,
                                height=240,
                            )
                    else:
                        st.subheader("Coefficients")
                        st.dataframe(
                            coeffs.style.format({"coef": "{:.4f}", "t_stat": "{:.2f}", "p_value": "{:.4f}"}),
                            use_container_width=True,
                            height=240,
                        )

                    st.subheader("Actual vs Fitted (Cumulative)")
                    data = result.data.copy()
                    if not data.empty:
                        actual = (1 + data["asset"]).cumprod()
                        fitted = (1 + data["fitted"]).cumprod()
                        fig = plt.figure()
                        plt.plot(actual.index, actual.values, label="Actual")
                        plt.plot(fitted.index, fitted.values, label="Fitted", linestyle="--")
                        plt.legend()
                        plt.xlabel("Date")
                        plt.ylabel("Growth of $1")
                        st.pyplot(fig, clear_figure=True)

    with ai_tab:
        st.subheader("AI Decision Support")
        st.caption("Summarizes dashboard data and highlights risks. Not financial advice.")

        if "factor_summary" not in st.session_state:
            st.session_state["factor_summary"] = {}

        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Set OPENAI_API_KEY in your environment to enable the AI assistant.")

        ac1, ac2, ac3 = st.columns([1.2, 1.1, 1.2])
        mode = ac1.selectbox("Mode", ["Digest + Recommendations", "Q&A"], index=0)
        quality = ac2.selectbox("Quality", ["Fast", "Deep"], index=0)
        include_web = ac3.checkbox("Include live market/news (web search)", value=False)

        model_map = {
            "Fast": "gpt-4.1-mini",
            "Deep": "gpt-4.1",
            "Fast_web": "gpt-4o-mini",
            "Deep_web": "gpt-4o",
        }
        if include_web:
            model = model_map.get(f"{quality}_web", "gpt-4o-mini")
        else:
            model = model_map.get(quality, "gpt-4.1-mini")

        if mode == "Q&A":
            question = st.text_area(
                "Ask a question about your data",
                value="What are the biggest risks in my portfolio right now?",
                height=100,
            )
            run_label = "Ask AI"
        else:
            question = st.text_area(
                "Optional focus (leave blank for full digest)",
                value="Summarize the watchlist and portfolio, then suggest risk controls.",
                height=100,
            )
            run_label = "Generate digest"

        if st.button(run_label):
            snapshot = build_ai_snapshot(watch_df, positions_table, range_label, st.session_state["factor_summary"])
            payload = build_snapshot_payload(snapshot, question)
            with st.spinner("Thinking..."):
                text, err = run_ai_assistant(payload, model=model, include_web=include_web)
            if err:
                st.error(err)
            else:
                st.markdown(text)

with options_tab:
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005)
    spreads_tab, longs_tab, scanner_tab, gains_tab = st.tabs(["My Spreads", "My Long Options", "Scanner + Flow", "Potential Gains"])

    with spreads_tab:
        st.subheader("Call Debit Spreads")
        spreads_path = os.path.join("data", "options_spreads.csv")
        spread_cols = [
            "position_id", "ticker", "expiry", "contracts", "entry_debit", "open_date",
            "notes", "long_strike", "short_strike"
        ]
        spreads_df = load_csv_with_columns(spreads_path, spread_cols)

        with st.expander("Add / Delete Spreads", expanded=False):
            with st.form("add_spread_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                ticker = c1.text_input("Ticker").upper().strip()
                expiry = c2.text_input("Expiry (YYYY-MM-DD)")
                contracts = c3.number_input("Contracts", min_value=1, step=1, value=1)
                c4, c5, c6 = st.columns(3)
                entry_debit = c4.number_input("Entry debit", min_value=0.0, step=0.05, value=1.0)
                long_strike = c5.number_input("Long strike", min_value=0.0, step=1.0, value=0.0)
                short_strike = c6.number_input("Short strike", min_value=0.0, step=1.0, value=0.0)
                notes = st.text_input("Notes")
                submitted = st.form_submit_button("Add spread")
                if submitted and ticker and expiry and long_strike and short_strike:
                    new_row = {
                        "position_id": uuid.uuid4().hex[:8],
                        "ticker": ticker,
                        "expiry": expiry,
                        "contracts": int(contracts),
                        "entry_debit": float(entry_debit),
                        "open_date": datetime.utcnow().strftime("%Y-%m-%d"),
                        "notes": notes,
                        "long_strike": float(long_strike),
                        "short_strike": float(short_strike),
                    }
                    spreads_df = pd.concat([spreads_df, pd.DataFrame([new_row])], ignore_index=True)
                    spreads_df.to_csv(spreads_path, index=False)
                    st.success("Spread added.")
                    st.rerun()

            if not spreads_df.empty:
                del_id = st.selectbox("Delete spread", spreads_df["position_id"].tolist(), key="del_spread")
                if st.button("Delete spread"):
                    spreads_df = spreads_df[spreads_df["position_id"] != del_id]
                    spreads_df.to_csv(spreads_path, index=False)
                    st.success("Spread deleted.")
                    st.rerun()

        if spreads_df.empty:
            st.info("No spreads yet. Add rows to data/options_spreads.csv")
        else:
            priced_rows = []
            for _, row in spreads_df.iterrows():
                ticker = str(row.get("ticker", "")).upper()
                expiry = str(row.get("expiry", ""))
                calls, _ = fetch_chain(ticker, expiry)
                metrics = price_call_debit_spread(row, calls)
                spot = latest_prices.get(ticker)
                dte = dte_from_expiry(expiry)

                long_greeks = greeks_from_price(metrics.get("long_mid"), spot, float(row.get("long_strike")), dte, risk_free_rate, "c") if spot else {}
                short_greeks = greeks_from_price(metrics.get("short_mid"), spot, float(row.get("short_strike")), dte, risk_free_rate, "c") if spot else {}
                net_delta = None
                net_vega = None
                if long_greeks.get("delta") is not None and short_greeks.get("delta") is not None:
                    net_delta = long_greeks["delta"] - short_greeks["delta"]
                if long_greeks.get("vega") is not None and short_greeks.get("vega") is not None:
                    net_vega = long_greeks["vega"] - short_greeks["vega"]

                priced = row.to_dict()
                if isinstance(metrics, dict):
                    priced.update(metrics)
                priced["long_iv"] = long_greeks.get("iv")
                priced["short_iv"] = short_greeks.get("iv")
                priced["long_delta"] = long_greeks.get("delta")
                priced["short_delta"] = short_greeks.get("delta")
                priced["long_gamma"] = long_greeks.get("gamma")
                priced["short_gamma"] = short_greeks.get("gamma")
                priced["long_theta"] = long_greeks.get("theta")
                priced["short_theta"] = short_greeks.get("theta")
                priced["long_vega"] = long_greeks.get("vega")
                priced["short_vega"] = short_greeks.get("vega")
                priced["net_delta"] = net_delta
                priced["net_vega"] = net_vega
                priced_rows.append(priced)

            priced_df = pd.DataFrame(priced_rows)
            st.dataframe(priced_df, use_container_width=True, height=260)

            sel_id = st.selectbox("Select spread", priced_df["position_id"].tolist())
            sel = priced_df[priced_df["position_id"] == sel_id].iloc[0]

            dte = dte_from_expiry(sel["expiry"])
            terminal = get_terminal_distribution(sel["ticker"], max(dte, 5), n_paths=10000)
            if terminal.size > 0:
                width = sel["short_strike"] - sel["long_strike"]
                debit = sel["entry_debit"]
                payoff = np.clip(terminal - sel["long_strike"], 0, width) - debit
                pnl = payoff * sel["contracts"] * 100.0

                pct = np.percentile(pnl, [5, 25, 50, 75, 95])
                st.write("P/L percentiles ($):", dict(zip([5, 25, 50, 75, 95], np.round(pct, 2))))

                st.caption("Payoff curve at expiration")
                fig = plt.figure()
                prices = np.linspace(sel["long_strike"] * 0.8, sel["short_strike"] * 1.2, 100)
                payoff_curve = np.clip(prices - sel["long_strike"], 0, width) - debit
                plt.plot(prices, payoff_curve * sel["contracts"] * 100.0)
                plt.axhline(0, color="gray", linewidth=1)
                plt.xlabel("Underlying price")
                plt.ylabel("P/L ($)")
                st.pyplot(fig, clear_figure=True)

    with longs_tab:
        st.subheader("Long Calls / Puts")
        longs_path = os.path.join("data", "options_longs.csv")
        long_cols = [
            "position_id", "ticker", "type", "expiry", "strike", "contracts",
            "entry_price", "open_date", "notes"
        ]
        longs_df = load_csv_with_columns(longs_path, long_cols)

        with st.expander("Add / Delete Long Options", expanded=False):
            with st.form("add_long_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                ticker = c1.text_input("Ticker").upper().strip()
                opt_type = c2.selectbox("Type", ["call", "put"])
                expiry = c3.text_input("Expiry (YYYY-MM-DD)")
                c4, c5, c6 = st.columns(3)
                strike = c4.number_input("Strike", min_value=0.0, step=1.0, value=0.0)
                contracts = c5.number_input("Contracts", min_value=1, step=1, value=1)
                entry_price = c6.number_input("Entry price", min_value=0.0, step=0.05, value=1.0)
                notes = st.text_input("Notes", key="long_notes")
                submitted = st.form_submit_button("Add option")
                if submitted and ticker and expiry and strike:
                    new_row = {
                        "position_id": uuid.uuid4().hex[:8],
                        "ticker": ticker,
                        "type": opt_type,
                        "expiry": expiry,
                        "strike": float(strike),
                        "contracts": int(contracts),
                        "entry_price": float(entry_price),
                        "open_date": datetime.utcnow().strftime("%Y-%m-%d"),
                        "notes": notes,
                    }
                    longs_df = pd.concat([longs_df, pd.DataFrame([new_row])], ignore_index=True)
                    longs_df.to_csv(longs_path, index=False)
                    st.success("Option added.")
                    st.rerun()

            if not longs_df.empty:
                del_id = st.selectbox("Delete option", longs_df["position_id"].tolist(), key="del_long")
                if st.button("Delete option"):
                    longs_df = longs_df[longs_df["position_id"] != del_id]
                    longs_df.to_csv(longs_path, index=False)
                    st.success("Option deleted.")
                    st.rerun()

        if longs_df.empty:
            st.info("No long options yet. Add rows to data/options_longs.csv")
        else:
            priced_rows = []
            for _, row in longs_df.iterrows():
                ticker = str(row.get("ticker", "")).upper()
                expiry = str(row.get("expiry", ""))
                calls, puts = fetch_chain(ticker, expiry)
                metrics = price_long_option(row, calls, puts)

                spot = latest_prices.get(ticker)
                dte = dte_from_expiry(expiry)
                flag = "c" if str(row.get("type", "call")).lower() == "call" else "p"
                greeks = greeks_from_price(metrics.get("mid"), spot, float(row.get("strike")), dte, risk_free_rate, flag) if spot else {}

                priced = row.to_dict()
                if isinstance(metrics, dict):
                    priced.update(metrics)
                priced["iv"] = greeks.get("iv")
                priced["delta"] = greeks.get("delta")
                priced["gamma"] = greeks.get("gamma")
                priced["theta"] = greeks.get("theta")
                priced["vega"] = greeks.get("vega")
                priced_rows.append(priced)

            priced_df = pd.DataFrame(priced_rows)
            st.dataframe(priced_df, use_container_width=True, height=260)

            sel_id = st.selectbox("Select option", priced_df["position_id"].tolist())
            sel = priced_df[priced_df["position_id"] == sel_id].iloc[0]

            dte = dte_from_expiry(sel["expiry"])
            terminal = get_terminal_distribution(sel["ticker"], max(dte, 5), n_paths=10000)
            if terminal.size > 0:
                strike = sel["strike"]
                entry = sel["entry_price"]
                if str(sel["type"]).lower() == "call":
                    payoff = np.maximum(terminal - strike, 0) - entry
                else:
                    payoff = np.maximum(strike - terminal, 0) - entry
                pnl = payoff * sel["contracts"] * 100.0
                pct = np.percentile(pnl, [5, 25, 50, 75, 95])
                st.write("P/L percentiles ($):", dict(zip([5, 25, 50, 75, 95], np.round(pct, 2))))

                st.caption("Payoff curve at expiration")
                fig = plt.figure()
                prices = np.linspace(strike * 0.6, strike * 1.4, 120)
                if str(sel["type"]).lower() == "call":
                    payoff_curve = np.maximum(prices - strike, 0) - entry
                else:
                    payoff_curve = np.maximum(strike - prices, 0) - entry
                plt.plot(prices, payoff_curve * sel["contracts"] * 100.0)
                plt.axhline(0, color="gray", linewidth=1)
                plt.xlabel("Underlying price")
                plt.ylabel("P/L ($)")
                st.pyplot(fig, clear_figure=True)

    with scanner_tab:
        st.subheader("Scanner + Flow")
        min_score = st.slider("Score threshold", min_value=50, max_value=100, value=80)
        min_oi = st.number_input("Min OI", min_value=0, value=300, step=50)
        min_vol = st.number_input("Min Volume", min_value=0, value=100, step=50)
        max_spread_pct = st.number_input("Max spread pct", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
        min_dte = st.number_input("Min DTE", min_value=5, value=20, step=5)
        max_dte = st.number_input("Max DTE", min_value=10, value=60, step=5)
        max_tickers = st.number_input("Max tickers", min_value=1, value=5, step=1)

        if st.button("Run scan"):
            scan_tickers = data_tickers[: int(max_tickers)]
            st.write(f"Scanning: {', '.join(scan_tickers)}")

            top_volume_rows = []
            top_unusual_rows = []
            candidate_rows = []
            scanner_alerts = []

            for t in scan_tickers:
                top_vol, top_unusual = scan_ticker_for_calls(t, int(min_dte), int(max_dte))
                if not top_vol.empty:
                    top_volume_rows.append(top_vol)
                if not top_unusual.empty:
                    top_unusual_rows.append(top_unusual)

                expiries = provider.get_option_expiries(t)

                for expiry in expiries[:4]:
                    dte = dte_from_expiry(expiry)
                    if dte < min_dte or dte > max_dte:
                        continue

                    candidates = build_call_debit_candidates(
                        t,
                        expiry,
                        min_oi=int(min_oi),
                        min_vol=int(min_vol),
                        max_spread_pct=float(max_spread_pct),
                        n_paths=10000,
                        horizon_days=max(dte, 5),
                    )
                    for c in candidates:
                        candidate_rows.append(c)
                        if c["score"] >= min_score and c["ev"] > 0:
                            if scanner_can_trigger(t, expiry, now_utc, cooldown_hours=6):
                                scanner_alerts.append(c)

            if top_volume_rows:
                top_vol_df = pd.concat(top_volume_rows).head(20)
                st.subheader("Top volume calls")
                st.dataframe(top_vol_df, use_container_width=True, height=240)

            if top_unusual_rows:
                top_unusual_df = pd.concat(top_unusual_rows).head(20)
                st.subheader("Top unusual volume calls")
                st.dataframe(top_unusual_df, use_container_width=True, height=240)

            if candidate_rows:
                cand_df = pd.DataFrame(candidate_rows).sort_values("score", ascending=False)
                st.subheader("Top call debit spread candidates")
                st.dataframe(cand_df, use_container_width=True, height=240)
                cand_path = os.path.join("data", "options_scanner_candidates.csv")
                cand_df.to_csv(cand_path, index=False)
                st.download_button(
                    "Download candidates CSV",
                    cand_df.to_csv(index=False).encode("utf-8"),
                    file_name="options_scanner_candidates.csv",
                    mime="text/csv",
                )

            if scanner_alerts:
                msg = "; ".join([f"{c['ticker']} {c['expiry']} score {c['score']:.1f}" for c in scanner_alerts[:5]])
                st.warning("Scanner alerts: " + msg)

                log_events = []
                for c in scanner_alerts:
                    log_events.append({
                        "timestamp": now_utc.isoformat(),
                        "id": "",
                        "ticker": c["ticker"],
                        "type": "scanner_candidate",
                        "alert_type": "scanner_candidate",
                        "value": c["score"],
                        "price": latest_prices.get(c["ticker"]),
                        "day_pct_move": day_pct_moves.get(c["ticker"]),
                        "message": f"{c['ticker']} {c['expiry']} score {c['score']:.1f}",
                        "details_json": json.dumps(c),
                    })
                append_alert_log(log_events, os.path.join("data", "alerts_log.csv"))

    with gains_tab:
        st.subheader("Potential Gains (Scenario + Expiry Payoff)")
        st.caption("Uses quant terminal distribution for scenario P/L percentiles.")

        spreads_path = os.path.join("data", "options_spreads.csv")
        longs_path = os.path.join("data", "options_longs.csv")
        spreads_df = load_csv_with_columns(spreads_path, spread_cols)
        longs_df = load_csv_with_columns(longs_path, long_cols)

        if spreads_df.empty and longs_df.empty:
            st.info("Add positions to options_spreads.csv or options_longs.csv.")
        else:
            total_delta = 0.0
            total_vega = 0.0
            pnl_rows = []

            for _, row in spreads_df.iterrows():
                ticker = str(row.get("ticker", "")).upper()
                expiry = str(row.get("expiry", ""))
                dte = dte_from_expiry(expiry)
                terminal = get_terminal_distribution(ticker, max(dte, 5), n_paths=10000)
                if terminal.size == 0:
                    continue
                width = float(row.get("short_strike")) - float(row.get("long_strike"))
                debit = float(row.get("entry_debit"))
                payoff = np.clip(terminal - float(row.get("long_strike")), 0, width) - debit
                pnl = payoff * float(row.get("contracts", 1)) * 100.0
                pct = np.percentile(pnl, [5, 25, 50, 75, 95])

                pnl_rows.append({
                    "position_id": row.get("position_id"),
                    "type": "call_debit_spread",
                    "ticker": ticker,
                    "p5": pct[0],
                    "p25": pct[1],
                    "p50": pct[2],
                    "p75": pct[3],
                    "p95": pct[4],
                })

                calls, _ = fetch_chain(ticker, expiry)
                metrics = price_call_debit_spread(row, calls)
                spot = latest_prices.get(ticker)
                if spot:
                    long_g = greeks_from_price(metrics.get("long_mid"), spot, float(row.get("long_strike")), dte, risk_free_rate, "c")
                    short_g = greeks_from_price(metrics.get("short_mid"), spot, float(row.get("short_strike")), dte, risk_free_rate, "c")
                    if long_g.get("delta") is not None and short_g.get("delta") is not None:
                        total_delta += (long_g["delta"] - short_g["delta"]) * float(row.get("contracts", 1))
                    if long_g.get("vega") is not None and short_g.get("vega") is not None:
                        total_vega += (long_g["vega"] - short_g["vega"]) * float(row.get("contracts", 1))

            for _, row in longs_df.iterrows():
                ticker = str(row.get("ticker", "")).upper()
                expiry = str(row.get("expiry", ""))
                dte = dte_from_expiry(expiry)
                terminal = get_terminal_distribution(ticker, max(dte, 5), n_paths=10000)
                if terminal.size == 0:
                    continue
                strike = float(row.get("strike"))
                entry = float(row.get("entry_price"))
                opt_type = str(row.get("type", "call")).lower()
                if opt_type == "call":
                    payoff = np.maximum(terminal - strike, 0) - entry
                    flag = "c"
                else:
                    payoff = np.maximum(strike - terminal, 0) - entry
                    flag = "p"
                pnl = payoff * float(row.get("contracts", 1)) * 100.0
                pct = np.percentile(pnl, [5, 25, 50, 75, 95])

                pnl_rows.append({
                    "position_id": row.get("position_id"),
                    "type": f"long_{opt_type}",
                    "ticker": ticker,
                    "p5": pct[0],
                    "p25": pct[1],
                    "p50": pct[2],
                    "p75": pct[3],
                    "p95": pct[4],
                })

                calls, puts = fetch_chain(ticker, expiry)
                metrics = price_long_option(row, calls, puts)
                spot = latest_prices.get(ticker)
                if spot:
                    g = greeks_from_price(metrics.get("mid"), spot, strike, dte, risk_free_rate, flag)
                    if g.get("delta") is not None:
                        total_delta += g["delta"] * float(row.get("contracts", 1))
                    if g.get("vega") is not None:
                        total_vega += g["vega"] * float(row.get("contracts", 1))

            if pnl_rows:
                pnl_df = pd.DataFrame(pnl_rows)
                st.dataframe(pnl_df, use_container_width=True, height=260)

            st.subheader("Options Portfolio Greeks (Net)")
            st.metric("Net Delta", f"{total_delta:.2f}")
            st.metric("Net Vega", f"{total_vega:.2f}")

# -----------------------------
# Portfolio Section
# -----------------------------
positions = positions_table.copy()

st.subheader("Portfolio (manual positions)")
if positions.empty:
    st.info("Add positions in the sidebar to see portfolio PnL and allocation.")
    st.stop()

total_mv = positions["market_value"].sum()

p1, p2 = st.columns([1.2, 0.8], gap="large")

with p1:
    st.dataframe(
        positions.sort_values("weight_%", ascending=False).style.format({
            "shares": "{:.2f}",
            "cost_basis": "{:.2f}",
            "last_price": "{:.2f}",
            "market_value": "{:,.2f}",
            "cost_value": "{:,.2f}",
            "pnl_$": "{:,.2f}",
            "pnl_%": "{:.2f}",
            "weight_%": "{:.2f}"
        }),
        use_container_width=True,
        height=380
    )

with p2:
    st.metric("Total Market Value", f"${total_mv:,.2f}")
    st.metric("Total PnL ($)", f"${positions['pnl_$'].sum():,.2f}")
    st.metric("Total PnL (%)", f"{(positions['pnl_$'].sum()/positions['cost_value'].sum())*100:.2f}%")

    st.caption("Allocation chart (by market value)")
    fig2 = plt.figure()
    plt.bar(positions["ticker"], positions["weight_%"])
    plt.xlabel("Ticker")
    plt.ylabel("Weight %")
    st.pyplot(fig2, clear_figure=True)

st.caption("Next upgrades: alerts, news feed, factor exposure, backtesting, export to CSV.")
