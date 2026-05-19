# Production Checklist

This checklist captures the production-readiness pass for the momentum hackathon project.

## 1. Product Summary

A reproducible machine-learning research and backtesting project for a weekly long-only momentum strategy across 10 large-cap stocks.

## 2. Final Inferred Requirements

- Download adjusted daily OHLCV data from Yahoo Finance.
- Build weekly, history-safe technical features.
- Train a classifier on 2017-2022.
- Predict weekly positive-return probability on 2023-2025.
- Rank all stocks weekly, select top 2, and equal-weight the portfolio.
- Report performance before and after 0.1% entry plus 0.1% exit costs.
- Export notebook, CSVs, charts, and report material.

## 3. Architecture Decisions

- Project type: Python ML/backtesting package plus Jupyter notebook.
- Core engine: `src/momentum_strategy.py`.
- Interface: notebook for presentation, CLI for reproducible execution.
- Data source: Yahoo Finance via `yfinance`.
- Model: scikit-learn Random Forest classifier.
- Parallelism: default `n_jobs=1` for Windows/workspace safety; CLI can set `--n-jobs -1` when supported.
- Validation: fixed train/test split, with synthetic unit tests for mechanics.
- Deployment: Docker-ready batch job.

## 4. Tech Stack

- Python 3.11
- pandas, numpy
- yfinance
- scikit-learn
- matplotlib, seaborn
- pytest
- Jupyter
- Docker

## 5. Folder Structure

```text
momentum-trading-ml/
  data/
  notebooks/
  outputs/
  reports/
  src/
  tests/
  Dockerfile
  docker-compose.yml
  requirements.txt
  run_backtest.py
```

## 6. Database Schema

No database is required. Outputs are stored as CSV artifacts:

- `daily_ohlcv.csv`
- `weekly_features.csv`
- `weekly_stock_predictions.csv`
- `weekly_portfolio_returns.csv`
- `performance_metrics.csv`

## 7. Backend Implementation

The backend equivalent is the reusable Python strategy engine. It handles data loading, feature engineering, model training, ranking, backtesting, metrics, and CSV writing.

## 8. Authentication System

Not applicable. This is an offline research/backtest batch project with no user accounts or server.

## 9. API Routes

Not applicable. CLI and notebook execution replace HTTP routes.

## 10. Frontend Pages

The Jupyter notebook is the presentation interface.

## 11. UI Components

Notebook tables and plots cover model results, portfolio returns, drawdown, distribution, selection frequency, and feature importance.

## 12. State Management

Runtime state is explicit through `StrategyConfig`, CLI flags, and CSV outputs.

## 13. Validation Layer

- Input column checks for daily data.
- Strategy config validation.
- Prediction probability checks.
- Weekly return output checks.
- Unit tests using synthetic data.

## 14. Error Handling

The pipeline raises clear errors for missing columns, invalid config, empty Yahoo downloads, empty train/test splits, and invalid output artifacts.

## 15. Security Protections

- No secrets are stored.
- No credentials are required.
- Runtime paths are explicit and reproducible.
- Dependencies are declared in `requirements.txt`.

## 16. Integration Layer

Yahoo Finance is the only external integration. `BRK.B` is mapped to Yahoo's `BRK-B` symbol and mapped back in outputs.

## 17. Environment Variables

Optional examples live in `.env.example`. CLI flags are preferred for reproducibility.

## 18. Installation Commands

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 19. Run Instructions

```powershell
python run_backtest.py
```

Custom run:

```powershell
python run_backtest.py --top-n 2 --entry-cost 0.001 --exit-cost 0.001
```

## 20. Docker/Deployment Configs

```powershell
docker compose up --build
```

The batch job writes CSV outputs to the mounted `outputs/` directory.

## 21. Testing Strategy

```powershell
pytest
```

Tests cover config validation, weekly feature construction, model prediction, weekly ranking, top-2 selection, backtest metrics, and empty-return handling.

## 22. Optimization Notes

- Weekly features reduce model input size.
- Random Forest uses parallel jobs.
- Output files are CSV for easy judging and reporting.
- The notebook delegates logic to reusable code to avoid duplicated calculations.

## 23. Final Verification Checklist

- Syntax check passes.
- Unit tests pass.
- Full backtest runs with approved network access.
- CSV deliverables are produced.
- Notebook JSON is valid.
- README and report docs explain assumptions and usage.
