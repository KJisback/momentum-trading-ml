"""Dashboard API for the momentum strategy outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(
    title="Momentum Strategy Dashboard",
    version="1.0.0",
    description="Human-readable dashboard API for the ML momentum strategy.",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def read_output_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing output file: {name}. Run the backtest first.")
    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def rounded(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


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
    performance = read_output_csv("performance_metrics.csv")
    weekly = read_output_csv("weekly_portfolio_returns.csv")
    predictions = read_output_csv("weekly_stock_predictions.csv")
    features = read_output_csv("weekly_features.csv")

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


@app.get("/api/equity")
def equity() -> dict:
    weekly = read_output_csv("weekly_portfolio_returns.csv").sort_values("week")
    weekly = weekly.copy()
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


@app.get("/api/predictions")
def predictions(limit: int = 60) -> dict:
    rows = read_output_csv("weekly_stock_predictions.csv").sort_values(["week", "rank"], ascending=[False, True])
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
