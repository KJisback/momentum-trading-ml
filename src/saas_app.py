"""Dashboard API for the momentum strategy outputs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import smtplib
from email.message import EmailMessage

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.momentum_strategy import StrategyConfig, run_live_pipeline
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
DEFAULT_ALLOWED_ORIGINS = [
    "https://kjisback.github.io",
    "http://127.0.0.1:8010",
    "http://127.0.0.1:8011",
    "http://localhost:8010",
    "http://localhost:8011",
]


def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]

app = FastAPI(
    title="Momentum Strategy Dashboard",
    version="1.1.0",
    description="Human-readable dashboard API for the momentum strategy.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/data", StaticFiles(directory=PROJECT_ROOT / "docs" / "data", check_dir=False), name="data")


class CustomRunRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=3, max_length=25)
    topN: int = Field(default=2, ge=1, le=10)
    modelType: str = "random_forest"
    portfolioDescription: str = Field(default="Custom yfinance watchlist", max_length=140)
    entryCost: float = Field(default=0.001, ge=0, le=0.05)
    exitCost: float = Field(default=0.001, ge=0, le=0.05)
    start: str = "2017-01-01"
    end: str | None = None
    trainEnd: str = "2022-12-31"
    testStart: str = "2023-01-01"


class EmailAlertRequest(CustomRunRequest):
    recipients: list[str] | None = None


def read_output_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing output file: {name}. Run the backtest first.")
    return pd.read_csv(path)


def pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value * 100:.2f}%"


def rounded(value: float, digits: int = 2) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [value.strip() for value in raw.split(",") if value.strip()]


def industry_map(tickers: list[str]) -> dict[str, str]:
    industries: dict[str, str] = {}
    for ticker in tickers[:10]:
        try:
            info = yf.Ticker(ticker.replace(".", "-")).get_info()
            industries[ticker] = info.get("industry") or info.get("sector") or "Unknown"
        except Exception:
            industries[ticker] = "Unknown"
    return industries


def clean_tickers(raw_tickers: list[str]) -> list[str]:
    tickers = []
    for ticker in raw_tickers:
        cleaned = ticker.strip().upper()
        if not cleaned:
            continue
        if not TICKER_PATTERN.match(cleaned):
            raise HTTPException(status_code=422, detail=f"Invalid ticker: {ticker}")
        if cleaned not in tickers:
            tickers.append(cleaned)
    if len(tickers) < 3:
        raise HTTPException(status_code=422, detail="Choose at least 3 unique tickers.")
    return tickers


def ensure_market_columns(weekly: pd.DataFrame, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    weekly = weekly.copy()
    if {"market_return", "market_volatility"}.issubset(weekly.columns):
        if "previous_market_volatility" not in weekly.columns:
            weekly["previous_market_volatility"] = weekly["market_volatility"].shift(1)
        return weekly

    if predictions is not None and "next_week_return" in predictions.columns:
        market = (
            predictions.groupby("week", as_index=False)
            .agg(
                market_return=("next_week_return", "mean"),
                market_volatility=("next_week_return", "std"),
            )
            .sort_values("week")
        )
        market["previous_market_volatility"] = market["market_volatility"].shift(1)
        return weekly.merge(market, on="week", how="left")

    for column in ["market_return", "market_volatility", "previous_market_volatility"]:
        if column not in weekly.columns:
            weekly[column] = np.nan
    return weekly


def ensure_jensens_alpha(performance: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    performance = performance.copy()
    for column in ["jensens_alpha", "sortino_ratio", "beta"]:
        if column not in performance.columns:
            performance[column] = np.nan
    if performance["jensens_alpha"].notna().all() and performance["sortino_ratio"].notna().all() and performance["beta"].notna().all():
        return performance

    return_columns = {"before_costs": "gross_return", "after_costs": "net_return"}
    for basis, return_column in return_columns.items():
        mask = performance["basis"] == basis
        if mask.any() and return_column in weekly.columns and "market_return" in weekly.columns:
            returns = weekly[return_column]
            benchmark = weekly.get("benchmark_blend_return", weekly["market_return"])
            performance.loc[mask, "jensens_alpha"] = jensens_alpha(returns, benchmark)
            performance.loc[mask, "sortino_ratio"] = sortino_ratio(returns)
            performance.loc[mask, "beta"] = beta(returns, benchmark)
    return performance


def sortino_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    equity = (1 + returns).cumprod()
    annualized_return = equity.iloc[-1] ** (periods_per_year / len(returns)) - 1
    downside = returns.loc[returns < 0].std(ddof=1) * np.sqrt(periods_per_year)
    if downside and not np.isclose(downside, 0):
        return float(annualized_return / downside)
    return float("nan")


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 2 or np.isclose(aligned["benchmark"].var(ddof=1), 0):
        return float("nan")
    return float(np.cov(aligned["strategy"], aligned["benchmark"], ddof=1)[0, 1] / aligned["benchmark"].var(ddof=1))


def jensens_alpha(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 52) -> float:
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2 or np.isclose(aligned["benchmark"].var(ddof=1), 0):
        return float("nan")

    _, alpha = np.polyfit(aligned["benchmark"], aligned["strategy"], 1)
    return float(alpha * periods_per_year)


def build_summary_payload(
    performance: pd.DataFrame,
    weekly: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    portfolio_description: str = "Default momentum universe",
    data_as_of: str | None = None,
    industries: dict[str, str] | None = None,
) -> dict:
    weekly = ensure_market_columns(weekly, predictions)
    performance = ensure_jensens_alpha(performance, weekly)
    before = performance.loc[performance["basis"] == "before_costs"].iloc[0]
    after = performance.loc[performance["basis"] == "after_costs"].iloc[0]
    sorted_weekly = weekly.sort_values("week")
    latest_realized_week = sorted_weekly.iloc[-1]
    latest_prediction_week = predictions.sort_values("week").iloc[-1]["week"]
    latest_predictions = predictions[predictions["week"] == latest_prediction_week].sort_values("rank")
    selected = latest_predictions[latest_predictions["selected"] == True].copy()
    industry_lookup = industries or {}

    cost_drag = before["cumulative_return"] - after["cumulative_return"]
    best_week = weekly.loc[weekly["net_return"].idxmax()]
    worst_week = weekly.loc[weekly["net_return"].idxmin()]
    net_drawdown = weekly["net_equity"] / weekly["net_equity"].cummax() - 1
    rolling_avg = weekly["net_return"].rolling(4, min_periods=1).mean()
    rolling_vol = weekly["net_return"].rolling(4, min_periods=2).std().fillna(0)
    latest_rolling_avg = rolling_avg.iloc[-1]
    latest_rolling_vol = rolling_vol.iloc[-1]
    latest_net_return = latest_realized_week["net_return"]
    latest_market_volatility = latest_predictions["weekly_return"].std()
    previous_market_volatility = features[features["week"] < latest_prediction_week].groupby("week")["weekly_return"].std()
    previous_market_volatility = previous_market_volatility.iloc[-1] if not previous_market_volatility.empty else np.nan
    risk_mood = "Calm"
    if latest_rolling_vol > 0.04 or net_drawdown.iloc[-1] < -0.1:
        risk_mood = "Elevated"
    if latest_rolling_vol > 0.07 or net_drawdown.iloc[-1] < -0.18:
        risk_mood = "Stressed"

    return {
        "asOfWeek": latest_prediction_week,
        "dataAsOf": data_as_of,
        "portfolioDescription": portfolio_description,
        "universeSize": int(features["ticker"].nunique()),
        "testWeeks": int(len(weekly)),
        "selectedStocks": [
            {
                "ticker": row["ticker"],
                "probability": rounded(row["predicted_probability"] * 100, 1),
                "weight": rounded(row["weight"] * 100, 0),
                "realizedNextWeekReturn": pct(row["next_week_return"]),
                "isLiveForecast": bool(pd.isna(row["next_week_return"])),
                "industry": industry_lookup.get(row["ticker"], "Unknown"),
            }
            for _, row in selected.iterrows()
        ],
        "headline": {
            "netCumulativeReturn": pct(after["cumulative_return"]),
            "netAnnualizedReturn": pct(after["annualized_return"]),
            "netSharpe": rounded(after["sharpe_ratio"]),
            "netSortino": rounded(after["sortino_ratio"]),
            "netBeta": rounded(after["beta"]),
            "maxDrawdown": pct(after["max_drawdown"]),
            "jensensAlpha": pct(after["jensens_alpha"]),
            "avgWeeklyReturn": pct(after["average_weekly_return"]),
            "costDrag": pct(cost_drag),
            "latestWeeklyReturn": pct(latest_net_return),
            "currentDrawdown": pct(net_drawdown.iloc[-1]),
            "rollingAvgReturn4w": pct(latest_rolling_avg),
            "rollingVolatility4w": pct(latest_rolling_vol),
            "marketVolatility": pct(latest_market_volatility),
            "previousMarketVolatility": pct(previous_market_volatility),
            "riskMood": risk_mood,
        },
        "comparison": {
            "beforeCosts": {
                "cumulativeReturn": pct(before["cumulative_return"]),
                "annualizedReturn": pct(before["annualized_return"]),
                "sharpe": rounded(before["sharpe_ratio"]),
                "sortino": rounded(before["sortino_ratio"]),
                "beta": rounded(before["beta"]),
                "maxDrawdown": pct(before["max_drawdown"]),
                "jensensAlpha": pct(before["jensens_alpha"]),
            },
            "afterCosts": {
                "cumulativeReturn": pct(after["cumulative_return"]),
                "annualizedReturn": pct(after["annualized_return"]),
                "sharpe": rounded(after["sharpe_ratio"]),
                "sortino": rounded(after["sortino_ratio"]),
                "beta": rounded(after["beta"]),
                "maxDrawdown": pct(after["max_drawdown"]),
                "jensensAlpha": pct(after["jensens_alpha"]),
            },
        },
        "plainEnglish": [
            f"The strategy grew $1 to ${1 + after['cumulative_return']:.2f} after trading costs.",
            f"Live forecast picks are based on yfinance data through {data_as_of or latest_prediction_week}.",
            f"Jensen's alpha versus the equal-weight universe benchmark was {pct(after['jensens_alpha'])}.",
            f"The worst peak-to-trough decline was {pct(after['max_drawdown'])}.",
            f"Trading costs reduced cumulative return by {pct(cost_drag)}.",
            f"Latest market volatility was {pct(latest_market_volatility)}, versus {pct(previous_market_volatility)} the previous week.",
            f"The latest week returned {pct(latest_net_return)} after costs, with a {risk_mood.lower()} risk mood.",
        ],
        "bestWeek": {"week": best_week["week"], "return": pct(best_week["net_return"])},
        "worstWeek": {"week": worst_week["week"], "return": pct(worst_week["net_return"])},
        "benchmark": {
            "name": "US/India index blend",
            "indexes": ["S&P 500", "Nasdaq Composite", "Dow Jones", "Nifty 50", "Sensex"],
        },
    }


def build_equity_payload(weekly: pd.DataFrame) -> dict:
    weekly = ensure_market_columns(weekly)
    weekly = weekly.sort_values("week").copy()
    weekly["gross_drawdown"] = weekly["gross_equity"] / weekly["gross_equity"].cummax() - 1
    weekly["net_drawdown"] = weekly["net_equity"] / weekly["net_equity"].cummax() - 1
    weekly["rolling_net_return_4w"] = weekly["net_return"].rolling(4, min_periods=1).mean()
    weekly["rolling_gross_return_4w"] = weekly["gross_return"].rolling(4, min_periods=1).mean()
    weekly["rolling_net_volatility_4w"] = weekly["net_return"].rolling(4, min_periods=2).std().fillna(0)
    weekly["rolling_gross_volatility_4w"] = weekly["gross_return"].rolling(4, min_periods=2).std().fillna(0)
    return {
        "series": [
            {
                "week": row["week"],
                "grossEquity": rounded(row["gross_equity"], 4),
                "netEquity": rounded(row["net_equity"], 4),
                "grossReturn": rounded(row["gross_return"], 5),
                "netReturn": rounded(row["net_return"], 5),
                "grossDrawdown": rounded(row["gross_drawdown"], 5),
                "netDrawdown": rounded(row["net_drawdown"], 5),
                "rollingNetReturn4w": rounded(row["rolling_net_return_4w"], 5),
                "rollingGrossReturn4w": rounded(row["rolling_gross_return_4w"], 5),
                "rollingNetVolatility4w": rounded(row["rolling_net_volatility_4w"], 5),
                "rollingGrossVolatility4w": rounded(row["rolling_gross_volatility_4w"], 5),
                "marketReturn": rounded(row.get("market_return"), 5),
                "marketVolatility": rounded(row.get("market_volatility"), 5),
                "previousMarketVolatility": rounded(row.get("previous_market_volatility"), 5),
                "benchmarkReturn": rounded(row.get("benchmark_blend_return"), 5),
                "benchmarkEquity": rounded(row.get("benchmark_blend_equity"), 4),
                "sp500Equity": rounded(row.get("benchmark_gspc_equity"), 4),
                "nasdaqEquity": rounded(row.get("benchmark_ixic_equity"), 4),
                "dowEquity": rounded(row.get("benchmark_dji_equity"), 4),
                "niftyEquity": rounded(row.get("benchmark_nsei_equity"), 4),
                "sensexEquity": rounded(row.get("benchmark_bsesn_equity"), 4),
            }
            for _, row in weekly.iterrows()
        ]
    }


def build_predictions_payload(rows: pd.DataFrame, limit: int = 60, industries: dict[str, str] | None = None) -> dict:
    rows = rows.copy()
    market = (
        rows.groupby("week")["weekly_return"]
        .std()
        .rename("market_volatility")
        .reset_index()
        .sort_values("week")
    )
    market["previous_market_volatility"] = market["market_volatility"].shift(1)
    rows = rows.merge(market, on="week", how="left", suffixes=("", "_from_predictions"))
    rows = rows.sort_values(["week", "rank"], ascending=[False, True])
    rows = rows.head(max(1, min(limit, 500)))
    industry_lookup = industries or {}
    return {
        "rows": [
            {
                "week": row["week"],
                "ticker": row["ticker"],
                "probability": rounded(row["predicted_probability"] * 100, 1),
                "rank": int(row["rank"]),
                "selected": bool(row["selected"]),
                "weight": rounded(row["weight"] * 100, 0),
                "nextWeekReturn": pct(row["next_week_return"]),
                "isLiveForecast": bool(pd.isna(row["next_week_return"])),
                "marketVolatility": rounded(row.get("market_volatility_from_predictions", row.get("market_volatility")), 5),
                "previousMarketVolatility": rounded(row.get("previous_market_volatility"), 5),
                "industry": industry_lookup.get(row["ticker"], "Unknown"),
            }
            for _, row in rows.iterrows()
        ]
    }


@app.get("/")
def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="styles.css"', 'href="/static/styles.css"')
    html = html.replace('src="config.js"', 'src="/static/config.js"')
    html = html.replace('src="app.js"', 'src="/static/app.js"')
    return HTMLResponse(html)


@app.get("/api/health")
def health() -> dict:
    required = [
        "performance_metrics.csv",
        "weekly_stock_predictions.csv",
        "weekly_portfolio_returns.csv",
        "weekly_features.csv",
    ]
    files = {name: (OUTPUT_DIR / name).exists() for name in required}
    return {"status": "ok" if all(files.values()) else "missing_outputs", "files": files}


@app.get("/api/summary")
def summary() -> dict:
    return build_summary_payload(
        read_output_csv("performance_metrics.csv"),
        read_output_csv("weekly_portfolio_returns.csv"),
        read_output_csv("weekly_stock_predictions.csv"),
        read_output_csv("weekly_features.csv"),
    )


@app.get("/api/equity")
def equity() -> dict:
    return build_equity_payload(read_output_csv("weekly_portfolio_returns.csv"))


@app.get("/api/predictions")
def predictions(limit: int = 60) -> dict:
    return build_predictions_payload(read_output_csv("weekly_stock_predictions.csv"), limit)


def run_custom_payload(request: CustomRunRequest, include_industries: bool = True) -> dict:
    tickers = clean_tickers(request.tickers)
    if request.topN > len(tickers):
        raise HTTPException(status_code=422, detail="topN cannot exceed the number of tickers.")

    config = StrategyConfig(
        start=request.start,
        end=request.end or "2099-01-01",
        train_end=request.trainEnd,
        test_start=request.testStart,
        top_n=request.topN,
        entry_cost=request.entryCost,
        exit_cost=request.exitCost,
        model_type=request.modelType,
    )
    try:
        result = run_live_pipeline(output_dir=OUTPUT_DIR / "custom_preview", config=config, tickers=tickers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latest_rows = result["predictions"].sort_values(["week", "rank"], ascending=[False, True]).head(10)
    industries = industry_map(latest_rows["ticker"].tolist()) if include_industries else {}
    return {
        "tickers": tickers,
        "summary": build_summary_payload(
            result["performance"],
            result["weekly_returns"],
            result["predictions"],
            result["weekly_dataset"],
            request.portfolioDescription,
            result.get("as_of"),
            industries,
        ),
        "equity": build_equity_payload(result["weekly_returns"]),
        "predictions": build_predictions_payload(result["predictions"], 120, industries),
        "modelMetrics": result["model_metrics"],
    }


@app.post("/api/custom-run")
def custom_run(request: CustomRunRequest) -> dict:
    return run_custom_payload(request)


def send_email(subject: str, body: str, recipients: list[str]) -> None:
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username
    if not host or not sender or not recipients:
        raise HTTPException(status_code=503, detail="Email is not configured.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)


def alert_body(payload: dict) -> str:
    lines = [
        payload["summary"]["portfolioDescription"],
        f"Forecast week: {payload['summary']['asOfWeek']}",
        f"Data as of: {payload['summary'].get('dataAsOf') or 'runtime'}",
        "",
        "Companies to buy/watch this week:",
    ]
    for stock in payload["summary"]["selectedStocks"]:
        lines.append(
            f"- {stock['ticker']} | {stock['probability']}% confidence | "
            f"{stock['weight']}% weight | {stock.get('industry', 'Unknown')}"
        )
    return "\n".join(lines)


@app.post("/api/email-weekly-picks")
def email_weekly_picks(request: EmailAlertRequest) -> dict:
    payload = run_custom_payload(request)
    recipients = request.recipients or env_list("ALERT_RECIPIENTS")
    send_email("Momentum weekly picks", alert_body(payload), recipients)
    return {"status": "sent", "recipients": recipients}


@app.get("/api/cron/weekly-email")
def cron_weekly_email(token: str | None = None) -> dict:
    secret = os.getenv("CRON_SECRET")
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Invalid cron token.")
    tickers = env_list("DEFAULT_ALERT_TICKERS") or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    request = EmailAlertRequest(
        tickers=tickers,
        topN=int(os.getenv("DEFAULT_ALERT_TOP_N", "2")),
        modelType=os.getenv("DEFAULT_ALERT_MODEL", "xgboost"),
        portfolioDescription=os.getenv("DEFAULT_ALERT_DESCRIPTION", "Weekly momentum alert"),
    )
    return email_weekly_picks(request)


@app.get("/api/downloads/{file_name}")
def download(file_name: str) -> FileResponse:
    allowed = {
        "daily_ohlcv.csv",
        "weekly_features.csv",
        "weekly_stock_predictions.csv",
        "weekly_portfolio_returns.csv",
        "performance_metrics.csv",
    }
    if file_name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown download.")
    path = OUTPUT_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run the backtest before downloading this file.")
    return FileResponse(path, media_type="text/csv", filename=file_name)
