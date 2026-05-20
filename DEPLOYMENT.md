# Deployment Checklist

## Backend on Hugging Face Spaces

1. Create a Hugging Face Space.
2. Choose Docker as the SDK.
3. Use port `7860`.
4. Add the environment variables listed in `HF_SPACE.md`.
5. Deploy the Docker Space.

Expected health URL:

```text
https://kjisback-momentum-trading-ml.hf.space/api/health
```

If Hugging Face gives a different Space URL, update `web/config.js`, run `py -3.11 export_static_site.py`, then commit and push.

## Frontend on GitHub Pages

The static frontend reads its backend URL from:

```text
web/config.js
docs/config.js
```

Current backend URL:

```text
https://kjisback-momentum-trading-ml.hf.space
```

## Weekly Email

Weekly email is triggered by GitHub Actions, not platform cron.

Set GitHub repository secrets:

```text
MOMENTUM_API_BASE=https://kjisback-momentum-trading-ml.hf.space
MOMENTUM_CRON_SECRET=<same value as Space CRON_SECRET>
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
