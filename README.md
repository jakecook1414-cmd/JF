# JF
Financial Analysis Platform

## Run (PowerShell)
```powershell
cd C:\Users\yubco\JF\finance-dashboard; python -m venv .venv
```

```powershell
Set-ExecutionPolicy -Scope Process Bypass; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

```powershell
python -m streamlit run app.py
```

## Quant Toolkit
Open the **Quant** tab in the Streamlit app for the professional-grade quant suite.

**New UX (Question-led)**
- Intent selector to route you to Forecast, Backtest, Validation, or Factors
- Beginner / Advanced mode toggle
- Presets for Forecast and Backtest to reduce setup time
- Always-visible assumptions & limitations block
- Daily summary card + CSV downloads for key outputs

**Forecast**
- Calendar-date fan chart (P10/P50/P90)
- Terminal distribution histogram at horizon
- Expected / Conservative / Tail ranges
- P(terminal > current) and P(terminal > target)

**Backtest**
- SMA Crossover and Time-series Momentum
- Walk-forward evaluation (train/test/step windows)
- Benchmark comparison vs buy-and-hold
- Transaction cost + slippage bps (advanced)

**Validation**
- Calibration report for forecast cones (coverage for 80%/90% bands)
- Model confidence cue + coverage explanation
- Diagnostics: realized volatility (20D/60D), return histogram, drawdown curve

**Factors**
- Factor regression outputs with plain-language exposures

## Dependencies
Key packages (see `requirements.txt` for full list):
- streamlit, pandas, numpy, matplotlib
- yfinance (data), exchange_calendars (trading days)
- empyrical, quantstats, arch, vectorbt (optional analytics/backtesting)
