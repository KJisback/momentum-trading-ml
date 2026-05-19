# Momentum Trading Strategy Using Machine Learning

Hackathon project for the IIT Mandi Xpecto '26 problem statement.

## Objective

Build and backtest a long-only machine-learning momentum strategy using daily stock data from Yahoo Finance. The strategy predicts whether each stock will have a positive next-week return, ranks the universe weekly, selects the top 2 stocks, and holds an equal-weight portfolio for one week.

## Universe

`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `TSLA`, `JPM`, `V`, `JNJ`, `BRK.B`

Yahoo Finance uses `BRK-B`; the code maps it back to `BRK.B` in outputs.

## Project Structure

```text
momentum-trading-ml/
  notebooks/
    momentum_ml_strategy.ipynb
  src/
    momentum_strategy.py
    saas_app.py
  web/
    index.html
    styles.css
    app.js
  outputs/
    CSVs and charts
  reports/
    short_report.md
    omega_x_checklist.md
  data/
    optional local data cache
  tests/
    synthetic-data validation tests
  run_backtest.py
  requirements.txt
  Dockerfile
```

## Setup

```powershell
cd F:\Kuku\Workspace\projects\momentum-trading-ml
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Open and run:

```text
notebooks\momentum_ml_strategy.ipynb
```

Or run the CLI:

```powershell
python run_backtest.py
```

Useful CLI options:

```powershell
python run_backtest.py --top-n 2 --entry-cost 0.001 --exit-cost 0.001
python run_backtest.py --n-jobs -1
python run_backtest.py --model-type hist_gradient_boosting
python run_backtest.py --tickers AAPL,MSFT,GOOGL,AMZN,META,TSLA,JPM,V,JNJ,BRK.B
```

Or run the reusable pipeline from Python:

```python
from src.momentum_strategy import run_pipeline

results = run_pipeline(output_dir="outputs")
print(results["performance"])
```

## Test

```powershell
pytest
```

The tests use synthetic data, so they validate feature engineering, training, selection, and backtest mechanics without requiring Yahoo Finance access.

## Docker

```powershell
docker compose up --build
```

Generated CSVs are written to `outputs\`.

## SaaS Dashboard

Run the backtest once, then start the dashboard:

```powershell
py -3.11 run_backtest.py
py -3.11 -m uvicorn src.saas_app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

Useful API endpoints:

- `GET /api/health`
- `GET /api/summary`
- `GET /api/equity`
- `GET /api/predictions`
- `GET /api/downloads/weekly_stock_predictions.csv`
- `POST /api/custom-run`

Custom watchlists:

- Local/backend mode supports live yfinance runs from the dashboard.
- GitHub Pages is static, so live runs call the deployed FastAPI backend configured in `web/config.js`.
- If the backend is asleep or unavailable, the static page still shows the default precomputed dashboard.
- The default backend target is `https://momentum-trading-ml-api.onrender.com`.

## Public Deployment

The public dashboard is exported to `docs/` and deployed through GitHub Pages. The live yfinance API is deployed as a Render web service.

Refresh the static build before publishing:

```powershell
py -3.11 export_static_site.py
```

Render backend settings:

```text
Service name: momentum-trading-ml-api
Build command: pip install --upgrade pip && pip install -r requirements.txt
Start command: uvicorn src.saas_app:app --host 0.0.0.0 --port $PORT
Health check: /api/health
```

The same settings are captured in `render.yaml`.

See `DEPLOYMENT.md` for the backend and frontend smoke-test checklist.

Expected public URL:

```text
https://kjisback.github.io/momentum-trading-ml/
```

Dashboard QOL features:

- Interactive chart modes for equity, weekly returns, drawdown, and rolling risk
- Full, 1-year, and 6-month chart ranges
- Hover tooltip with week-level values
- Plain-English performance readout
- Latest picks, cost impact, current drawdown, and 4-week average return

## Validation Setup

- Train: 2017-01-01 to 2022-12-31
- Test: 2023-01-01 to 2025-12-31
- Weekly rebalance: Friday weekly bars
- Portfolio: top 2 predicted probabilities, 50% each
- Transaction costs: 0.1% entry + 0.1% exit per weekly portfolio rebalance

## Outputs

The notebook/pipeline writes:

- `outputs\daily_ohlcv.csv`
- `outputs\weekly_features.csv`
- `outputs\weekly_stock_predictions.csv`
- `outputs\weekly_portfolio_returns.csv`
- `outputs\performance_metrics.csv`

## Current Backtest Snapshot

The latest run produced:

| Basis | Cumulative return | Annualized return | Annualized volatility | Sharpe | Max drawdown | Hit rate | Avg weekly return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before costs | 228.72% | 48.69% | 21.00% | 2.32 | -14.98% | 59.62% | 0.81% |
| After costs | 141.08% | 34.09% | 21.00% | 1.62 | -15.85% | 54.49% | 0.61% |

## Notes on Backtest Correctness

- Features are built from current and historical weekly data only.
- The target is next-week close-to-close return.
- Predictions are produced only on the 2023-2025 test period.
- Weekly selected stocks are held for the following week.
- Reported metrics include before-cost and after-cost results.
