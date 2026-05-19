# Deployment Checklist

## Backend on Render

1. Open Render and create a new Blueprint from this repository.
2. Select `render.yaml`.
3. Confirm the service name is `momentum-trading-ml-api`.
4. Deploy the service.
5. Open:

```text
https://momentum-trading-ml-api.onrender.com/api/health
```

Expected response:

```json
{"status":"ok"}
```

The first request can be slow on a free Render service.

## Frontend on GitHub Pages

The static frontend reads its backend URL from:

```text
web/config.js
docs/config.js
```

Current backend URL:

```text
https://momentum-trading-ml-api.onrender.com
```

After changing `web/config.js`, run:

```powershell
py -3.11 export_static_site.py
```

Then commit and push.

## Smoke Tests

Local backend:

```powershell
py -3.11 -m uvicorn src.saas_app:app --host 127.0.0.1 --port 8020
```

Health:

```text
http://127.0.0.1:8020/api/health
```

Live custom run:

```text
POST /api/custom-run
```

Payload:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "topN": 2,
  "portfolioDescription": "Large-cap live scan"
}
```
