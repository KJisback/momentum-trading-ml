---
title: Momentum Trading ML API
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Momentum Trading ML

Momentum Trading ML is now a portfolio value and forecasting web app. It keeps the original top-N momentum strategy, then adds:

- Dated investment scenarios, for example "$100 in MSFT on 2025-06-11".
- Stocks, ETFs, indexes, crypto-style Yahoo symbols, and global listings using Yahoo Finance notation.
- Historical value, current value, gain/loss, CAGR, and forward confidence bands.
- Saved portfolios and scenario history behind signed Google-ready sessions.
- Admin overview for users, portfolios, scenarios, and model defaults.
- Dense dashboard UI with a toggleable terminal mode.

## Architecture

- Backend: FastAPI, Pydantic validation, yfinance, pandas, scikit-learn.
- Frontend: static HTML/CSS/JS served by FastAPI for simple Docker/cloud deployment.
- Storage: SQLite by default at `data/momentum_app.sqlite3`; automatically falls back to shared in-memory SQLite if the local filesystem rejects SQLite journaling.
- API style: REST JSON.
- Auth: Google-ready signed session endpoint. In production, place Google Identity Services in front of `/api/auth/google` and pass the verified email/name/google id.
- Security: CORS allowlist, auth bearer tokens, input validation, rate limiting, security headers, DB parameterization.

## Run Locally

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m uvicorn src.saas_app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

## Key Endpoints

```text
GET  /api/health
GET  /api/summary
GET  /api/equity
GET  /api/predictions
POST /api/custom-run
POST /api/scenarios
POST /api/auth/google
GET  /api/portfolios
POST /api/portfolios
GET  /api/scenarios
GET  /api/admin/overview
```

## Docker

```powershell
docker compose up --build
```

The app runs on:

```text
http://127.0.0.1:8010
```

## Tests

```powershell
py -3.11 -m pytest -q --basetemp=pytest-cache-files-local
node tests\static_smoke.js
```
