"""Dashboard API for the momentum strategy outputs."""

from __future__ import annotations

import os
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
import smtplib
import time
import uuid
from datetime import date, datetime, timedelta
from email.message import EmailMessage

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.momentum_strategy import StrategyConfig, run_live_pipeline, to_yfinance_ticker
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.getenv("MOMENTUM_DB_PATH", DATA_DIR / "momentum_app.sqlite3"))
MEMORY_DB_URI = "file:momentum_app_memory?mode=memory&cache=shared"
USE_MEMORY_DB = False
MEMORY_KEEPER: sqlite3.Connection | None = None
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
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
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/data", StaticFiles(directory=PROJECT_ROOT / "docs" / "data", check_dir=False), name="data")


RATE_LIMIT: dict[str, list[float]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    now = time.time()
    client = request.client.host if request.client else "unknown"
    bucket = RATE_LIMIT.setdefault(client, [])
    RATE_LIMIT[client] = [stamp for stamp in bucket if now - stamp < 60]
    if len(RATE_LIMIT[client]) >= int(os.getenv("RATE_LIMIT_PER_MINUTE", "90")):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    RATE_LIMIT[client].append(now)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


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


class AuthRequest(BaseModel):
    email: str = Field(..., max_length=160)
    name: str = Field(default="Portfolio analyst", max_length=120)
    googleId: str | None = Field(default=None, max_length=240)
    adminAccessCode: str | None = Field(default=None, max_length=120)


class HoldingInput(BaseModel):
    symbol: str = Field(..., max_length=12)
    allocation: float = Field(..., gt=0, le=100_000_000)
    purchaseDate: str = Field(..., max_length=10)
    assetType: str = Field(default="stock", max_length=24)


class ScenarioRequest(BaseModel):
    name: str = Field(default="Investment scenario", max_length=120)
    baseCurrency: str = Field(default="USD", max_length=8)
    holdings: list[HoldingInput] = Field(..., min_length=1, max_length=25)
    forecastHorizonWeeks: int = Field(default=26, ge=1, le=260)
    save: bool = False


class PortfolioRequest(BaseModel):
    name: str = Field(..., max_length=120)
    baseCurrency: str = Field(default="USD", max_length=8)
    holdings: list[HoldingInput] = Field(..., min_length=1, max_length=50)


def db_connection() -> sqlite3.Connection:
    if USE_MEMORY_DB:
        conn = sqlite3.connect(MEMORY_DB_URI, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    global USE_MEMORY_DB, MEMORY_KEEPER
    schema = """
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      name TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user',
      google_id TEXT,
      created_at TEXT NOT NULL,
      last_login_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS portfolios (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      name TEXT NOT NULL,
      base_currency TEXT NOT NULL,
      holdings_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS scenarios (
      id TEXT PRIMARY KEY,
      user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
      portfolio_id TEXT REFERENCES portfolios(id) ON DELETE SET NULL,
      name TEXT NOT NULL,
      request_json TEXT NOT NULL,
      result_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id);
    CREATE INDEX IF NOT EXISTS idx_scenarios_user ON scenarios(user_id);
    """
    try:
        with db_connection() as conn:
            conn.executescript(schema)
    except sqlite3.OperationalError:
        USE_MEMORY_DB = True
        MEMORY_KEEPER = sqlite3.connect(MEMORY_DB_URI, uri=True)
        MEMORY_KEEPER.row_factory = sqlite3.Row
        MEMORY_KEEPER.execute("PRAGMA foreign_keys=ON")
        MEMORY_KEEPER.executescript(schema)


@app.on_event("startup")
def startup() -> None:
    init_db()


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def auth_secret() -> bytes:
    return os.getenv("JWT_SECRET", "dev-only-change-me").encode("utf-8")


def sign_payload(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    signature = hmac.new(auth_secret(), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def verify_token(token: str) -> dict:
    try:
        body, signature = token.rsplit(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid session token.") from exc
    expected = hmac.new(auth_secret(), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid session signature.")
    payload = json.loads(body)
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Session expired.")
    return payload


def current_user(request: Request) -> dict:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    payload = verify_token(header.removeprefix("Bearer ").strip())
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (payload["sub"],)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return dict(row)


def optional_user(request: Request) -> dict | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return current_user(request)


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


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


def clean_symbol(raw_symbol: str) -> str:
    cleaned = raw_symbol.strip().upper()
    if not TICKER_PATTERN.match(cleaned):
        raise HTTPException(status_code=422, detail=f"Invalid symbol: {raw_symbol}")
    return cleaned


def parse_iso_date(value: str, field_name: str = "date") -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be YYYY-MM-DD.") from exc
    if parsed > date.today():
        raise HTTPException(status_code=422, detail=f"{field_name} cannot be in the future.")
    return parsed


def download_adjusted_closes(symbols: list[str], start: date, end: date | None = None) -> pd.DataFrame:
    yf_symbols = [to_yfinance_ticker(symbol) for symbol in symbols]
    data = yf.download(
        yf_symbols,
        start=start.isoformat(),
        end=((end or date.today()) + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames: list[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for symbol, yf_symbol in zip(symbols, yf_symbols):
            if yf_symbol not in available or "Close" not in data[yf_symbol].columns:
                continue
            frame = data[yf_symbol][["Close"]].dropna().rename(columns={"Close": symbol})
            frames.append(frame)
    elif len(symbols) == 1 and "Close" in data.columns:
        frames.append(data[["Close"]].dropna().rename(columns={"Close": symbols[0]}))

    if not frames:
        raise HTTPException(status_code=502, detail="Yahoo Finance returned no usable close prices for this request.")
    closes = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    missing = [symbol for symbol in symbols if symbol not in closes.columns or closes[symbol].dropna().empty]
    if missing:
        raise HTTPException(status_code=422, detail=f"No usable price data for: {', '.join(missing)}")
    return closes


def closest_price(closes: pd.Series, target: date) -> tuple[pd.Timestamp, float]:
    available = closes[closes.index.date >= target].dropna()
    if available.empty:
        raise HTTPException(status_code=422, detail=f"No market price available on or after {target.isoformat()}.")
    stamp = available.index[0]
    return stamp, float(available.iloc[0])


def project_value(current_value: float, weekly_returns: pd.Series, horizon: int) -> list[dict]:
    weekly = weekly_returns.dropna()
    mean = float(weekly.mean()) if not weekly.empty else 0.0
    vol = float(weekly.std(ddof=1)) if len(weekly) > 1 else 0.0
    start_week = pd.Timestamp(date.today())
    rows = []
    for step in range(1, horizon + 1):
        expected = current_value * ((1 + mean) ** step)
        band = 1.65 * vol * np.sqrt(step)
        rows.append(
            {
                "week": (start_week + pd.Timedelta(weeks=step)).date().isoformat(),
                "expected": rounded(expected, 2),
                "lower": rounded(current_value * ((1 + mean - band) ** step), 2),
                "upper": rounded(current_value * ((1 + mean + band) ** step), 2),
            }
        )
    return rows


def build_scenario_payload(request: ScenarioRequest) -> dict:
    holdings = [
        {
            "symbol": clean_symbol(item.symbol),
            "allocation": float(item.allocation),
            "purchaseDate": parse_iso_date(item.purchaseDate, "purchaseDate"),
            "assetType": item.assetType.strip().lower() or "stock",
        }
        for item in request.holdings
    ]
    symbols = list(dict.fromkeys(item["symbol"] for item in holdings))
    start = min(item["purchaseDate"] for item in holdings) - timedelta(days=7)
    closes = download_adjusted_closes(symbols, start)

    holding_results = []
    historical_lines: list[pd.Series] = []
    total_invested = sum(item["allocation"] for item in holdings)
    total_current = 0.0
    weighted_weekly_returns = []

    for item in holdings:
        series = closes[item["symbol"]].dropna()
        entry_date, entry_price = closest_price(series, item["purchaseDate"])
        current_date = series.index[-1]
        current_price = float(series.iloc[-1])
        shares = item["allocation"] / entry_price
        current_value = shares * current_price
        total_current += current_value
        gain = current_value - item["allocation"]
        total_return = gain / item["allocation"]
        days_held = max((current_date.date() - entry_date.date()).days, 1)
        cagr = (current_value / item["allocation"]) ** (365 / days_held) - 1
        weekly_returns = series.resample("W-FRI").last().pct_change().dropna()
        weighted_weekly_returns.append(weekly_returns * (item["allocation"] / total_invested))
        values = (series[series.index >= entry_date] * shares).rename(item["symbol"])
        historical_lines.append(values)
        holding_results.append(
            {
                "symbol": item["symbol"],
                "assetType": item["assetType"],
                "allocation": rounded(item["allocation"], 2),
                "entryDate": entry_date.date().isoformat(),
                "entryPrice": rounded(entry_price, 4),
                "shares": rounded(shares, 6),
                "currentDate": current_date.date().isoformat(),
                "currentPrice": rounded(current_price, 4),
                "currentValue": rounded(current_value, 2),
                "gain": rounded(gain, 2),
                "totalReturn": pct(total_return),
                "cagr": pct(cagr),
                "forecast": project_value(current_value, weekly_returns, request.forecastHorizonWeeks),
            }
        )

    history_frame = pd.concat(historical_lines, axis=1).ffill().fillna(0)
    history = [
        {"date": index.date().isoformat(), "value": rounded(float(row.sum()), 2)}
        for index, row in history_frame.iterrows()
    ]
    portfolio_weekly = pd.concat(weighted_weekly_returns, axis=1).sum(axis=1) if weighted_weekly_returns else pd.Series(dtype=float)
    projected = project_value(total_current, portfolio_weekly, request.forecastHorizonWeeks)
    total_gain = total_current - total_invested
    total_return = total_gain / total_invested
    return {
        "name": request.name,
        "baseCurrency": request.baseCurrency.upper(),
        "asOf": date.today().isoformat(),
        "summary": {
            "invested": rounded(total_invested, 2),
            "currentValue": rounded(total_current, 2),
            "gain": rounded(total_gain, 2),
            "totalReturn": pct(total_return),
            "bestHolding": max(holding_results, key=lambda row: row["gain"])["symbol"],
            "worstHolding": min(holding_results, key=lambda row: row["gain"])["symbol"],
            "forecastExpected": projected[-1]["expected"] if projected else rounded(total_current, 2),
            "forecastLower": projected[-1]["lower"] if projected else rounded(total_current, 2),
            "forecastUpper": projected[-1]["upper"] if projected else rounded(total_current, 2),
        },
        "holdings": holding_results,
        "history": history,
        "forecast": projected,
        "notes": [
            "Historical values use adjusted Yahoo Finance close prices.",
            "Forecast bands are statistical projections from realized weekly returns, not financial advice.",
            "Global symbols should use Yahoo Finance notation, for example RELIANCE.NS or ^NSEI.",
        ],
    }


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


@app.post("/api/auth/google")
def google_auth(request: AuthRequest) -> dict:
    email = request.email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    now = utc_now()
    admin_code = os.getenv("ADMIN_ACCESS_CODE")
    role = "admin" if admin_code and request.adminAccessCode == admin_code else "user"
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            role = row["role"] if row["role"] == "admin" else role
            conn.execute(
                "UPDATE users SET name = ?, google_id = COALESCE(?, google_id), role = ?, last_login_at = ? WHERE email = ?",
                (request.name.strip() or email, request.googleId, role, now, email),
            )
            user_id = row["id"]
        else:
            user_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO users (id, email, name, role, google_id, created_at, last_login_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, email, request.name.strip() or email, role, request.googleId, now, now),
            )
    token = sign_payload({"sub": user_id, "email": email, "role": role, "exp": int(time.time()) + SESSION_TTL_SECONDS})
    return {"token": token, "user": {"id": user_id, "email": email, "name": request.name, "role": role}}


@app.get("/api/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"user": {key: user[key] for key in ["id", "email", "name", "role", "created_at", "last_login_at"]}}


@app.get("/api/portfolios")
def list_portfolios(user: dict = Depends(current_user)) -> dict:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM portfolios WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],),
        ).fetchall()
    return {
        "rows": [
            {
                "id": row["id"],
                "name": row["name"],
                "baseCurrency": row["base_currency"],
                "holdings": json.loads(row["holdings_json"]),
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
            for row in rows
        ]
    }


@app.post("/api/portfolios")
def save_portfolio(request: PortfolioRequest, user: dict = Depends(current_user)) -> dict:
    holdings = [
        {
            "symbol": clean_symbol(item.symbol),
            "allocation": float(item.allocation),
            "purchaseDate": parse_iso_date(item.purchaseDate, "purchaseDate").isoformat(),
            "assetType": item.assetType.strip().lower() or "stock",
        }
        for item in request.holdings
    ]
    now = utc_now()
    portfolio_id = str(uuid.uuid4())
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO portfolios (id, user_id, name, base_currency, holdings_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (portfolio_id, user["id"], request.name, request.baseCurrency.upper(), json.dumps(holdings), now, now),
        )
    return {"id": portfolio_id, "name": request.name, "baseCurrency": request.baseCurrency.upper(), "holdings": holdings}


@app.post("/api/scenarios")
def run_scenario(request: ScenarioRequest, http_request: Request) -> dict:
    result = build_scenario_payload(request)
    user = optional_user(http_request)
    if request.save and user:
        now = utc_now()
        scenario_id = str(uuid.uuid4())
        with db_connection() as conn:
            conn.execute(
                """
                INSERT INTO scenarios (id, user_id, name, request_json, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario_id,
                    user["id"],
                    request.name,
                    request.json(),
                    json.dumps(result),
                    now,
                ),
            )
        result["scenarioId"] = scenario_id
    return result


@app.get("/api/scenarios")
def list_scenarios(user: dict = Depends(current_user)) -> dict:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, result_json, created_at FROM scenarios WHERE user_id = ? ORDER BY created_at DESC LIMIT 40",
            (user["id"],),
        ).fetchall()
    return {
        "rows": [
            {
                "id": row["id"],
                "name": row["name"],
                "createdAt": row["created_at"],
                "result": json.loads(row["result_json"]),
            }
            for row in rows
        ]
    }


@app.get("/api/admin/overview")
def admin_overview(user: dict = Depends(require_admin)) -> dict:
    with db_connection() as conn:
        users = conn.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"]
        portfolios = conn.execute("SELECT COUNT(*) AS total FROM portfolios").fetchone()["total"]
        scenarios = conn.execute("SELECT COUNT(*) AS total FROM scenarios").fetchone()["total"]
    return {
        "users": users,
        "portfolios": portfolios,
        "scenarios": scenarios,
        "modelDefaults": {
            "modelType": os.getenv("DEFAULT_ALERT_MODEL", "hist_gradient_boosting"),
            "topN": int(os.getenv("DEFAULT_ALERT_TOP_N", "2")),
            "dataSource": "Yahoo Finance via yfinance",
        },
    }


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
        modelType=os.getenv("DEFAULT_ALERT_MODEL", "hist_gradient_boosting"),
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
