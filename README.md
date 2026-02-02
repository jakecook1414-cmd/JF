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

**Forecast**
- Bootstrap or Regime bootstrap forecast cones
- Percentile bands, terminal distribution stats
- P(terminal > current) and P(terminal > target)

**Backtest**
- SMA Crossover and Time-series Momentum
- Walk-forward evaluation (train/test/step windows)
- Transaction cost + slippage bps

**Validation**
- Calibration report for forecast cones (coverage for 80%/90% bands)
- Diagnostics: realized volatility (20D/60D), return histogram, drawdown curve

## Dependencies
Key packages (see `requirements.txt` for full list):
- streamlit, pandas, numpy, matplotlib
- yfinance (data), exchange_calendars (trading days)
- empyrical, quantstats, arch, vectorbt (optional analytics/backtesting)
