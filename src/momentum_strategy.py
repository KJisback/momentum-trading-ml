"""Machine-learning momentum strategy utilities.

The module is designed for the IIT Mandi Xpecto '26 hackathon problem:
download daily stock data, build weekly classification labels, rank stocks by
predicted next-week positive-return probability, and backtest a top-2 long-only
portfolio with transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    XGBClassifier = None


RAW_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "JNJ", "BRK.B"]
YF_TICKER_MAP = {"BRK.B": "BRK-B"}
BENCHMARK_INDEXES = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones",
    "^NSEI": "Nifty 50",
    "^BSESN": "Sensex",
}
FEATURE_COLUMNS = [
    "ret_1w",
    "ret_2w",
    "ret_4w",
    "ret_8w",
    "ret_12w",
    "ret_26w",
    "vol_4w",
    "vol_12w",
    "ma_ratio_4w",
    "ma_ratio_12w",
    "price_position_26w",
    "volume_change_4w",
]


@dataclass(frozen=True)
class StrategyConfig:
    start: str = "2017-01-01"
    end: str = "2026-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    top_n: int = 2
    entry_cost: float = 0.001
    exit_cost: float = 0.001
    periods_per_year: int = 52
    random_state: int = 42
    n_jobs: int = 1
    model_type: str = "random_forest"
    benchmark_tickers: tuple[str, ...] = tuple(BENCHMARK_INDEXES.keys())


def validate_config(config: StrategyConfig, tickers: Iterable[str] = RAW_TICKERS) -> None:
    """Validate strategy settings before running an expensive download/backtest."""
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("At least one ticker is required.")
    if config.top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if config.top_n > len(ticker_list):
        raise ValueError("top_n cannot exceed the number of tickers.")
    if config.entry_cost < 0 or config.exit_cost < 0:
        raise ValueError("Transaction costs must be non-negative.")
    if pd.Timestamp(config.start) >= pd.Timestamp(config.end):
        raise ValueError("start must be earlier than end.")
    if pd.Timestamp(config.train_end) >= pd.Timestamp(config.test_start):
        raise ValueError("train_end must be earlier than test_start.")
    if config.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if config.n_jobs == 0:
        raise ValueError("n_jobs cannot be 0.")
    if config.model_type not in {"random_forest", "hist_gradient_boosting", "xgboost"}:
        raise ValueError("model_type must be 'random_forest', 'hist_gradient_boosting', or 'xgboost'.")


def to_yfinance_ticker(ticker: str) -> str:
    """Map problem-statement tickers to Yahoo Finance symbols."""
    return YF_TICKER_MAP.get(ticker, ticker)


def from_yfinance_ticker(ticker: str) -> str:
    """Map Yahoo Finance symbols back to problem-statement display symbols."""
    inverse = {value: key for key, value in YF_TICKER_MAP.items()}
    return inverse.get(ticker, ticker)


def download_daily_data(
    tickers: Iterable[str] = RAW_TICKERS,
    start: str = "2017-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Download daily OHLCV data and return a tidy dataframe.

    Yahoo uses BRK-B instead of BRK.B, so symbols are normalized after download.
    """
    ticker_list = list(tickers)
    yf_tickers = [to_yfinance_ticker(ticker) for ticker in ticker_list]
    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    missing_tickers: list[str] = []
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for requested_ticker, yf_ticker in zip(ticker_list, yf_tickers):
            if yf_ticker not in available:
                missing_tickers.append(requested_ticker)
                continue
            stock = data[yf_ticker].copy()
            if "Close" not in stock.columns:
                missing_tickers.append(requested_ticker)
                continue
            stock = stock.dropna(subset=["Close"])
            if stock.empty:
                missing_tickers.append(requested_ticker)
                continue
            stock["ticker"] = requested_ticker
            frames.append(stock.reset_index())
    else:
        stock = data.copy()
        if "Close" not in stock.columns:
            missing_tickers.append(ticker_list[0])
        else:
            stock = stock.dropna(subset=["Close"])
            if stock.empty:
                missing_tickers.append(ticker_list[0])
            else:
                stock["ticker"] = ticker_list[0]
                frames.append(stock.reset_index())

    if missing_tickers:
        joined = ", ".join(missing_tickers)
        raise ValueError(f"Yahoo Finance returned no usable price data for: {joined}.")

    daily = pd.concat(frames, ignore_index=True)
    daily.columns = [str(col).lower().replace(" ", "_") for col in daily.columns]
    daily = daily.rename(columns={"date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.dropna(subset=["close"]).sort_values(["ticker", "date"])
    if daily.empty:
        raise RuntimeError(
            "Yahoo Finance returned no usable price rows. Check network access, ticker availability, "
            "or rerun with an approved internet connection."
        )
    return daily


def build_weekly_dataset(daily: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    """Convert daily OHLCV data into weekly features and labels."""
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weekly_frames: list[pd.DataFrame] = []
    for ticker, stock in daily.groupby("ticker", sort=True):
        stock = stock.sort_values("date").set_index("date")
        weekly = pd.DataFrame(
            {
                "open": stock["open"].resample("W-FRI").first(),
                "high": stock["high"].resample("W-FRI").max(),
                "low": stock["low"].resample("W-FRI").min(),
                "close": stock["close"].resample("W-FRI").last(),
                "volume": stock["volume"].resample("W-FRI").sum(),
            }
        ).dropna(subset=["close"])
        weekly["ticker"] = ticker
        weekly["week"] = weekly.index
        weekly_frames.append(weekly.reset_index(drop=True))

    if not weekly_frames:
        raise RuntimeError("No weekly bars could be created from the downloaded daily data.")

    weekly = pd.concat(weekly_frames, ignore_index=True).sort_values(["ticker", "week"])
    grouped = weekly.groupby("ticker", group_keys=False)

    weekly["next_week_return"] = grouped["close"].transform(lambda close: close.shift(-1) / close - 1)
    weekly["target"] = (weekly["next_week_return"] > 0).astype(int)

    for window in [1, 2, 4, 8, 12, 26]:
        weekly[f"ret_{window}w"] = grouped["close"].pct_change(window)

    weekly["weekly_return"] = grouped["close"].pct_change()
    weekly["vol_4w"] = grouped["weekly_return"].rolling(4).std().reset_index(level=0, drop=True)
    weekly["vol_12w"] = grouped["weekly_return"].rolling(12).std().reset_index(level=0, drop=True)
    weekly["ma_4w"] = grouped["close"].rolling(4).mean().reset_index(level=0, drop=True)
    weekly["ma_12w"] = grouped["close"].rolling(12).mean().reset_index(level=0, drop=True)
    weekly["ma_ratio_4w"] = weekly["close"] / weekly["ma_4w"] - 1
    weekly["ma_ratio_12w"] = weekly["close"] / weekly["ma_12w"] - 1

    high_26w = grouped["close"].rolling(26).max().reset_index(level=0, drop=True)
    low_26w = grouped["close"].rolling(26).min().reset_index(level=0, drop=True)
    weekly["price_position_26w"] = (weekly["close"] - low_26w) / (high_26w - low_26w)

    volume_ma_4w = grouped["volume"].rolling(4).mean().reset_index(level=0, drop=True)
    weekly["volume_change_4w"] = weekly["volume"] / volume_ma_4w - 1

    required_columns = FEATURE_COLUMNS + (["next_week_return", "target"] if require_label else [])
    model_data = weekly.dropna(subset=required_columns).copy()
    model_data["week"] = pd.to_datetime(model_data["week"])
    return model_data


def build_benchmark_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Build weekly index benchmark returns and growth series."""
    if daily.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for ticker, rows in daily.groupby("ticker", sort=True):
        weekly = rows.sort_values("date").set_index("date")["close"].resample("W-FRI").last().dropna()
        frame = pd.DataFrame(
            {
                "week": weekly.index,
                f"benchmark_{_safe_column_name(ticker)}_return": weekly.pct_change(),
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    benchmark = frames[0]
    for frame in frames[1:]:
        benchmark = benchmark.merge(frame, on="week", how="outer")
    benchmark = benchmark.sort_values("week")

    return_columns = [column for column in benchmark.columns if column.endswith("_return")]
    benchmark["benchmark_blend_return"] = benchmark[return_columns].mean(axis=1)
    for column in return_columns + ["benchmark_blend_return"]:
        equity_column = column.replace("_return", "_equity")
        benchmark[equity_column] = (1 + benchmark[column].fillna(0)).cumprod()
    return benchmark.dropna(subset=["benchmark_blend_return"])


def _safe_column_name(value: str) -> str:
    return value.replace("^", "").replace(".", "_").replace("-", "_").lower()


def make_model(model_type: str = "random_forest", random_state: int = 42, n_jobs: int = 1) -> Pipeline:
    """Create the classifier pipeline.

    `hist_gradient_boosting` is the preferred scalable option for larger tabular
    datasets because it trains efficiently on bigger row counts.
    """
    if model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ValueError("xgboost is not installed. Install requirements.txt or choose another model.")
        estimator = XGBClassifier(
            n_estimators=450,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=1.5,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError("model_type must be 'random_forest', 'hist_gradient_boosting', or 'xgboost'.")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def train_predict(model_data: pd.DataFrame, config: StrategyConfig) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Train on 2017-2022 and return weekly out-of-sample predictions."""
    labeled = model_data.dropna(subset=["next_week_return", "target"]).copy()
    train_mask = labeled["week"] <= pd.Timestamp(config.train_end)
    test_mask = model_data["week"] >= pd.Timestamp(config.test_start)

    train = labeled.loc[train_mask].copy()
    test = model_data.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Training or test dataset is empty. Check date filters and input data.")

    model = make_model(config.model_type, config.random_state, config.n_jobs)
    model.fit(train[FEATURE_COLUMNS], train["target"])

    predictions = test.copy()
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])
    if probabilities.shape[1] == 1:
        positive_probability = np.ones(len(test)) if model.classes_[0] == 1 else np.zeros(len(test))
    else:
        positive_index = list(model.classes_).index(1)
        positive_probability = probabilities[:, positive_index]
    predictions["predicted_probability"] = positive_probability
    predictions["predicted_label"] = (predictions["predicted_probability"] >= 0.5).astype(int)

    scored = predictions.dropna(subset=["target"]).copy()
    metrics = {
        "test_accuracy": accuracy_score(scored["target"], scored["predicted_label"]) if not scored.empty else float("nan"),
        "test_roc_auc": _safe_roc_auc(scored["target"], scored["predicted_probability"]) if not scored.empty else float("nan"),
        "train_rows": int(len(train)),
        "test_rows": int(len(scored)),
        "prediction_rows": int(len(test)),
    }
    return model, predictions, metrics


def _safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def construct_portfolio(predictions: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Rank stocks weekly, select top N, and compute weighted contributions."""
    ranked = predictions.sort_values(["week", "predicted_probability"], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("week")["predicted_probability"].rank(method="first", ascending=False)
    ranked["selected"] = ranked["rank"] <= config.top_n
    ranked["weight"] = np.where(ranked["selected"], 1.0 / config.top_n, 0.0)
    ranked["return_contribution"] = ranked["weight"] * ranked["next_week_return"]
    return ranked


def backtest_portfolio(
    portfolio_rows: pd.DataFrame,
    config: StrategyConfig,
    benchmark_weekly: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weekly strategy returns before and after transaction costs."""
    realized_rows = portfolio_rows.dropna(subset=["next_week_return"]).copy()
    selected = realized_rows.loc[realized_rows["selected"]].copy()
    market = (
        realized_rows.groupby("week", as_index=False)
        .agg(
            market_return=("next_week_return", "mean"),
            market_volatility=("weekly_return", "std"),
        )
        .sort_values("week")
    )
    market["previous_market_volatility"] = market["market_volatility"].shift(1)
    weekly = (
        selected.groupby("week", as_index=False)
        .agg(
            selected_stocks=("ticker", lambda values: ", ".join(values)),
            gross_return=("return_contribution", "sum"),
            selected_count=("ticker", "count"),
        )
        .sort_values("week")
    )
    weekly = weekly.merge(market, on="week", how="left")
    if benchmark_weekly is not None and not benchmark_weekly.empty:
        weekly = weekly.merge(benchmark_weekly, on="week", how="left")
    if "benchmark_blend_return" not in weekly.columns:
        weekly["benchmark_blend_return"] = weekly["market_return"]

    weekly["entry_cost"] = config.entry_cost
    weekly["exit_cost"] = config.exit_cost
    weekly["total_cost"] = weekly["entry_cost"] + weekly["exit_cost"]
    weekly["net_return"] = weekly["gross_return"] - weekly["total_cost"]
    weekly["gross_equity"] = (1 + weekly["gross_return"]).cumprod()
    weekly["net_equity"] = (1 + weekly["net_return"]).cumprod()

    metrics = pd.DataFrame(
        [
            {
                "basis": "before_costs",
                **performance_metrics(weekly["gross_return"], config.periods_per_year, weekly["benchmark_blend_return"]),
            },
            {
                "basis": "after_costs",
                **performance_metrics(weekly["net_return"], config.periods_per_year, weekly["benchmark_blend_return"]),
            },
        ]
    )
    return weekly, metrics


def performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 52,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate common backtest metrics from periodic returns."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "average_weekly_return": np.nan,
            "jensens_alpha": np.nan,
            "sortino_ratio": np.nan,
            "beta": np.nan,
        }

    equity = (1 + returns).cumprod()
    cumulative_return = equity.iloc[-1] - 1
    annualized_return = equity.iloc[-1] ** (periods_per_year / len(returns)) - 1
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe_ratio = np.nan
    if annualized_volatility and not np.isclose(annualized_volatility, 0):
        sharpe_ratio = annualized_return / annualized_volatility
    downside = returns.loc[returns < 0].std(ddof=1) * np.sqrt(periods_per_year)
    sortino_ratio = np.nan
    if downside and not np.isclose(downside, 0):
        sortino_ratio = annualized_return / downside
    drawdown = equity / equity.cummax() - 1

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "average_weekly_return": float(returns.mean()),
        "jensens_alpha": _jensens_alpha(returns, benchmark_returns, periods_per_year),
        "sortino_ratio": float(sortino_ratio),
        "beta": _beta(returns, benchmark_returns),
    }


def _jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    periods_per_year: int,
) -> float:
    if benchmark_returns is None:
        return float("nan")

    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2 or np.isclose(aligned["benchmark"].var(ddof=1), 0):
        return float("nan")

    beta, alpha = np.polyfit(aligned["benchmark"], aligned["strategy"], 1)
    return float(alpha * periods_per_year)


def _beta(returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    if benchmark_returns is None:
        return float("nan")
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2 or np.isclose(aligned["benchmark"].var(ddof=1), 0):
        return float("nan")
    return float(np.cov(aligned["strategy"], aligned["benchmark"], ddof=1)[0, 1] / aligned["benchmark"].var(ddof=1))


def validate_outputs(portfolio_rows: pd.DataFrame, weekly_returns: pd.DataFrame, performance: pd.DataFrame) -> None:
    """Run final consistency checks on strategy outputs."""
    if portfolio_rows.empty:
        raise RuntimeError("No prediction rows were produced.")
    if weekly_returns.empty:
        raise RuntimeError("No weekly portfolio returns were produced.")
    if performance.empty:
        raise RuntimeError("No performance metrics were produced.")
    selected_counts = portfolio_rows.loc[portfolio_rows["selected"]].groupby("week")["ticker"].nunique()
    if selected_counts.empty or selected_counts.min() < 1:
        raise RuntimeError("At least one rebalance week has no selected stock.")
    probabilities = portfolio_rows["predicted_probability"]
    if not probabilities.between(0, 1).all():
        raise RuntimeError("Predicted probabilities must be between 0 and 1.")
    required_return_columns = ["gross_return", "net_return", "market_return", "market_volatility", "benchmark_blend_return"]
    if weekly_returns[required_return_columns].isna().any().any():
        raise RuntimeError("Weekly return output contains missing values.")


def run_pipeline(
    output_dir: str | Path = "outputs",
    config: StrategyConfig | None = None,
    tickers: Iterable[str] = RAW_TICKERS,
) -> dict:
    """Run the complete strategy and write hackathon deliverables."""
    config = config or StrategyConfig()
    ticker_list = list(tickers)
    validate_config(config, ticker_list)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    daily = download_daily_data(ticker_list, config.start, config.end)
    benchmark_daily = download_daily_data(config.benchmark_tickers, config.start, config.end)
    benchmark_weekly = build_benchmark_weekly(benchmark_daily)
    weekly_dataset = build_weekly_dataset(daily)
    model, predictions, model_metrics = train_predict(weekly_dataset, config)
    portfolio_rows = construct_portfolio(predictions, config)
    weekly_returns, performance = backtest_portfolio(portfolio_rows, config, benchmark_weekly)
    validate_outputs(portfolio_rows, weekly_returns, performance)

    daily.to_csv(output_path / "daily_ohlcv.csv", index=False)
    weekly_dataset.to_csv(output_path / "weekly_features.csv", index=False)
    portfolio_rows.to_csv(output_path / "weekly_stock_predictions.csv", index=False)
    weekly_returns.to_csv(output_path / "weekly_portfolio_returns.csv", index=False)
    performance.to_csv(output_path / "performance_metrics.csv", index=False)

    return {
        "model": model,
        "daily": daily,
        "weekly_dataset": weekly_dataset,
        "predictions": portfolio_rows,
        "weekly_returns": weekly_returns,
        "performance": performance,
        "model_metrics": model_metrics,
        "benchmark_weekly": benchmark_weekly,
    }


def run_live_pipeline(
    output_dir: str | Path = "outputs/custom_preview",
    config: StrategyConfig | None = None,
    tickers: Iterable[str] = RAW_TICKERS,
    as_of: date | None = None,
) -> dict:
    """Run the strategy with yfinance data through runtime and include the latest forecast week."""
    run_date = as_of or date.today()
    download_end = (run_date + timedelta(days=1)).isoformat()
    base_config = config or StrategyConfig()
    live_config = StrategyConfig(
        start=base_config.start,
        end=download_end,
        train_end=base_config.train_end,
        test_start=base_config.test_start,
        top_n=base_config.top_n,
        entry_cost=base_config.entry_cost,
        exit_cost=base_config.exit_cost,
        periods_per_year=base_config.periods_per_year,
        random_state=base_config.random_state,
        n_jobs=base_config.n_jobs,
        model_type=base_config.model_type,
        benchmark_tickers=base_config.benchmark_tickers,
    )
    ticker_list = list(tickers)
    validate_config(live_config, ticker_list)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    daily = download_daily_data(ticker_list, live_config.start, live_config.end)
    benchmark_daily = download_daily_data(live_config.benchmark_tickers, live_config.start, live_config.end)
    benchmark_weekly = build_benchmark_weekly(benchmark_daily)
    weekly_dataset = build_weekly_dataset(daily, require_label=False)
    model, predictions, model_metrics = train_predict(weekly_dataset, live_config)
    portfolio_rows = construct_portfolio(predictions, live_config)
    weekly_returns, performance = backtest_portfolio(portfolio_rows, live_config, benchmark_weekly)
    validate_outputs(portfolio_rows.dropna(subset=["next_week_return"]), weekly_returns, performance)

    daily.to_csv(output_path / "daily_ohlcv.csv", index=False)
    weekly_dataset.to_csv(output_path / "weekly_features.csv", index=False)
    portfolio_rows.to_csv(output_path / "weekly_stock_predictions.csv", index=False)
    weekly_returns.to_csv(output_path / "weekly_portfolio_returns.csv", index=False)
    performance.to_csv(output_path / "performance_metrics.csv", index=False)

    return {
        "model": model,
        "daily": daily,
        "weekly_dataset": weekly_dataset,
        "predictions": portfolio_rows,
        "weekly_returns": weekly_returns,
        "performance": performance,
        "model_metrics": model_metrics,
        "benchmark_weekly": benchmark_weekly,
        "as_of": run_date.isoformat(),
    }
