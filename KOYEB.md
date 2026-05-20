# Koyeb Deployment

Deploy the backend as a Koyeb web service from this GitHub repository.

## Service Settings

```text
Repository: KJisback/momentum-trading-ml
Branch: main
Builder: Dockerfile
Exposed port: 8000
Health check path: /api/health
```

The Dockerfile reads Koyeb's `PORT` environment variable.

## Environment Variables

Required:

```text
ALLOWED_ORIGINS=https://kjisback.github.io,http://127.0.0.1:8010,http://127.0.0.1:8011,http://localhost:8010,http://localhost:8011
DEFAULT_ALERT_TICKERS=AAPL,MSFT,GOOGL,AMZN,META
DEFAULT_ALERT_MODEL=hist_gradient_boosting
```

Required for email alerts:

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

After the Koyeb app is live, update:

```text
web/config.js
```

Then run:

```powershell
py -3.11 export_static_site.py
```

Commit and push the refreshed `docs/` files.

## Weekly Email Cron

This repo uses GitHub Actions for the weekly trigger instead of a paid platform cron.

Set these GitHub repository secrets:

```text
MOMENTUM_API_BASE=https://your-koyeb-app.koyeb.app
MOMENTUM_CRON_SECRET=the-same-secret-set-on-koyeb
```

The workflow calls:

```text
/api/cron/weekly-email
```
