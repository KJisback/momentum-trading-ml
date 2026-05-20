# Hugging Face Space Deployment

Deploy the FastAPI backend as a Docker Space.

## Create The Space

1. Create a new Space on Hugging Face.
2. Suggested name:

```text
momentum-trading-ml
```

3. Choose:

```text
SDK: Docker
Visibility: Public
```

4. Push this repository's files to the Space repository, or connect/import from GitHub if available.

The Dockerfile runs FastAPI on port `7860`, which is the standard Docker Spaces port.
It installs `requirements-api.txt`, not the full notebook/dev requirements, to keep the free Space build smaller.

## Space README Metadata

The Space repository README should start with:

```yaml
---
title: Momentum Trading ML API
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---
```

## Environment Variables

Set these in the Space settings.

Required:

```text
ALLOWED_ORIGINS=https://kjisback.github.io,http://127.0.0.1:8010,http://127.0.0.1:8011,http://localhost:8010,http://localhost:8011
DEFAULT_ALERT_TICKERS=AAPL,MSFT,GOOGL,AMZN,META
DEFAULT_ALERT_MODEL=hist_gradient_boosting
```

XGBoost is intentionally not installed in the Space image by default. It remains a local/paid-deployment option because it can make free Space builds too large or slow.

Email alerts:

```text
SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM=
ALERT_RECIPIENTS=
CRON_SECRET=
```

## Frontend URL

The expected API URL is:

```text
https://kjisback-momentum-trading-ml.hf.space
```

If your Space URL differs, update `web/config.js`, export the static site, and push:

```powershell
py -3.11 export_static_site.py
git add web/config.js docs/config.js docs
git commit -m "Update Hugging Face backend URL"
git push origin main
```

## Weekly Email Cron

GitHub Actions triggers the weekly email endpoint. Set repository secrets:

```text
MOMENTUM_API_BASE=https://kjisback-momentum-trading-ml.hf.space
MOMENTUM_CRON_SECRET=<same value as Space CRON_SECRET>
```
