# Deployment Checklist

## Backend on Koyeb

1. Create a Koyeb web service from this GitHub repository.
2. Use the Dockerfile builder.
3. Set the exposed port to `8000`.
4. Set the health check path to `/api/health`.
5. Add the environment variables listed in `KOYEB.md`.
6. Deploy.

Expected health URL:

```text
https://momentum-trading-ml.koyeb.app/api/health
```

If Koyeb gives a different app URL, update `web/config.js`, run `py -3.11 export_static_site.py`, then commit and push.

## Frontend on GitHub Pages

The static frontend reads its backend URL from:

```text
web/config.js
docs/config.js
```

Current backend URL:

```text
https://momentum-trading-ml.koyeb.app
```

## Weekly Email

Weekly email is triggered by GitHub Actions, not platform cron.

Set GitHub repository secrets:

```text
MOMENTUM_API_BASE=https://momentum-trading-ml.koyeb.app
MOMENTUM_CRON_SECRET=<same value as Koyeb CRON_SECRET>
```

The workflow is:

```text
.github/workflows/weekly-email.yml
```

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
  "modelType": "hist_gradient_boosting",
  "portfolioDescription": "Large-cap live scan"
}
```
