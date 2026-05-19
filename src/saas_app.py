"""Dashboard API for the momentum strategy outputs."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.momentum_strategy import StrategyConfig, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

app = FastAPI(
    title="Momentum Strategy Dashboard",
    version="1.1.0",
    description="Human-readable dashboard API for the momentum strategy.",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/data", StaticFiles(directory=PROJECT_ROOT / "docs" / "data", check_dir=False), name="data")


class CustomRunRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=3, max_length=25)
    topN: int = Field(default=2, ge=1, le=10)
    modelType: str = "random_forest"
    entryCost: float = Field(default=0.001, ge=0, le=0.05)
    exitCost: float = Field(default=0.001, ge=0, le=0.05)
    start: str = "2017-01-01"
    end: str = "2026-01-01"
    trainEnd: str = "2022-12-31"
    testStart: str = "2023-01-01"


def read_output_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing output file: {name}. Run the backtest first.")
    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def rounded(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


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


def build_summary_payload(
    performance: pd.DataFrame,
    weekly: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    before = performance.loc[performance["basis"] == "before_costs"].iloc[0]
    after = performance.loc[performance["basis"] == "after_costs"].iloc[0]
    latest_week = weekly.sort_values("week").iloc[-1]
    latest_predictions = predictions[predictions["week"] == latest_week["week"]].sort_values("rank")
    selected = latest_predictions[latest_predictions["selected"] == True].copy()

    cost_drag = before["cumulative_return"] - after["cumulative_return"]
    best_week = weekly.loc[weekly["net_return"].idxmax()]
    worst_week = weekly.loc[weekly["net_return"].idxmin()]
    net_drawdown = weekly["net_equity"] / weekly["net_equity"].cummax() - 1
    rolling_avg = weekly["net_return"].rolling(4, min_periods=1).mean()
    rolling_vol = weekly["net_return"].rolling(4, min_periods=2).std().fillna(0)
    latest_rolling_avg = rolling_avg.iloc[-1]
    latest_rolling_vol = rolling_vol.iloc[-1]
    latest_net_return = latest_week["net_return"]
    risk_mood = "Calm"
    if latest_rolling_vol > 0.04 or net_drawdown.iloc[-1] < -0.1:
        risk_mood = "Elevated"
    if latest_rolling_vol > 0.07 or net_drawdown.iloc[-1] < -0.18:
        risk_mood = "Stressed"

    return {
        "asOfWeek": latest_week["week"],
        "universeSize": int(features["ticker"].nunique()),
        "testWeeks": int(len(weekly)),
        "selectedStocks": [
            {
                "ticker": row["ticker"],
                "probability": rounded(row["predicted_probability"] * 100, 1),
                "weight": rounded(row["weight"] * 100, 0),
                "realizedNextWeekReturn": pct(row["next_week_return"]),
            }
            for _, row in selected.iterrows()
        ],
        "headline": {
            "netCumulativeReturn": pct(after["cumulative_return"]),
            "netAnnualizedReturn": pct(after["annualized_return"]),
            "netSharpe": rounded(after["sharpe_ratio"]),
            "maxDrawdown": pct(after["max_drawdown"]),
            "hitRate": pct(after["hit_rate"]),
            "avgWeeklyReturn": pct(after["average_weekly_return"]),
            "costDrag": pct(cost_drag),
            "latestWeeklyReturn": pct(latest_net_return),
            "currentDrawdown": pct(net_drawdown.iloc[-1]),
            "rollingAvgReturn4w": pct(latest_rolling_avg),
            "rollingVolatility4w": pct(latest_rolling_vol),
            "riskMood": risk_mood,
        },
        "comparison": {
            "beforeCosts": {
                "cumulativeReturn": pct(before["cumulative_return"]),
                "annualizedReturn": pct(before["annualized_return"]),
                "sharpe": rounded(before["sharpe_ratio"]),
                "maxDrawdown": pct(before["max_drawdown"]),
                "hitRate": pct(before["hit_rate"]),
            },
            "afterCosts": {
                "cumulativeReturn": pct(after["cumulative_return"]),
                "annualizedReturn": pct(after["annualized_return"]),
                "sharpe": rounded(after["sharpe_ratio"]),
                "maxDrawdown": pct(after["max_drawdown"]),
                "hitRate": pct(after["hit_rate"]),
            },
        },
        "plainEnglish": [
            f"The strategy grew $1 to ${1 + after['cumulative_return']:.2f} after trading costs.",
            f"It won in {pct(after['hit_rate'])} of test weeks from 2023 through 2025.",
            f"The worst peak-to-trough decline was {pct(after['max_drawdown'])}.",
            f"Trading costs reduced cumulative return by {pct(cost_drag)}.",
            f"The latest week returned {pct(latest_net_return)} after costs, with a {risk_mood.lower()} risk mood.",
        ],
        "bestWeek": {"week": best_week["week"], "return": pct(best_week["net_return"])},
        "worstWeek": {"week": worst_week["week"], "return": pct(worst_week["net_return"])},
    }


def build_equity_payload(weekly: pd.DataFrame) -> dict:
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
            }
            for _, row in weekly.iterrows()
        ]
    }


def build_predictions_payload(rows: pd.DataFrame, limit: int = 60) -> dict:
    rows = rows.sort_values(["week", "rank"], ascending=[False, True])
    rows = rows.head(max(1, min(limit, 500)))
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
            }
            for _, row in rows.iterrows()
        ]
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


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


@app.post("/api/custom-run")
def custom_run(request: CustomRunRequest) -> dict:
    tickers = clean_tickers(request.tickers)
    if request.topN > len(tickers):
        raise HTTPException(status_code=422, detail="topN cannot exceed the number of tickers.")

    config = StrategyConfig(
        start=request.start,
        end=request.end,
        train_end=request.trainEnd,
        test_start=request.testStart,
        top_n=request.topN,
        entry_cost=request.entryCost,
        exit_cost=request.exitCost,
        model_type=request.modelType,
    )
    try:
        result = run_pipeline(output_dir=OUTPUT_DIR / "custom_preview", config=config, tickers=tickers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "tickers": tickers,
        "summary": build_summary_payload(
            result["performance"],
            result["weekly_returns"],
            result["predictions"],
            result["weekly_dataset"],
        ),
        "equity": build_equity_payload(result["weekly_returns"]),
        "predictions": build_predictions_payload(result["predictions"], 120),
        "modelMetrics": result["model_metrics"],
    }


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
