import numpy as np
import pandas as pd
import pytest

from src.momentum_strategy import (
    FEATURE_COLUMNS,
    StrategyConfig,
    backtest_portfolio,
    build_weekly_dataset,
    construct_portfolio,
    download_daily_data,
    performance_metrics,
    train_predict,
    validate_config,
)


def synthetic_daily_data() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2024-03-31")
    frames = []
    for idx, ticker in enumerate(["AAA", "BBB", "CCC"]):
        trend = np.linspace(0, 35 + idx * 8, len(dates))
        seasonal = np.sin(np.arange(len(dates)) / 18 + idx) * 8
        correction = np.where((np.arange(len(dates)) // 70) % 2 == 0, 0, -6)
        close = 100 + trend + seasonal + correction
        open_ = close * 0.999
        high = close * 1.01
        low = close * 0.99
        volume = 1_000_000 + idx * 100_000 + np.arange(len(dates)) * 10
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "ticker": ticker,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_validate_config_rejects_invalid_top_n():
    with pytest.raises(ValueError, match="top_n"):
        validate_config(StrategyConfig(top_n=4), ["A", "B"])


def test_build_weekly_dataset_creates_features_without_cross_ticker_target_leakage():
    weekly = build_weekly_dataset(synthetic_daily_data())

    assert set(FEATURE_COLUMNS).issubset(weekly.columns)
    assert weekly["ticker"].nunique() == 3
    assert weekly[FEATURE_COLUMNS].notna().all().all()

    last_rows = weekly.sort_values("week").groupby("ticker").tail(1)
    assert last_rows["next_week_return"].notna().all()


def test_train_rank_and_backtest_pipeline_on_synthetic_data():
    config = StrategyConfig(
        start="2020-01-01",
        end="2024-04-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        top_n=2,
        random_state=7,
    )
    weekly = build_weekly_dataset(synthetic_daily_data())
    _, predictions, model_metrics = train_predict(weekly, config)
    portfolio_rows = construct_portfolio(predictions, config)
    weekly_returns, performance = backtest_portfolio(portfolio_rows, config)

    selected = portfolio_rows.loc[portfolio_rows["selected"]]
    assert selected.groupby("week")["ticker"].nunique().eq(2).all()
    assert portfolio_rows["predicted_probability"].between(0, 1).all()
    assert not weekly_returns.empty
    assert {"market_return", "market_volatility", "previous_market_volatility"}.issubset(weekly_returns.columns)
    assert set(performance["basis"]) == {"before_costs", "after_costs"}
    assert "jensens_alpha" in performance.columns
    assert "sortino_ratio" in performance.columns
    assert "beta" in performance.columns
    assert model_metrics["train_rows"] > 0
    assert model_metrics["test_rows"] > 0


def test_performance_metrics_handles_empty_returns():
    metrics = performance_metrics(pd.Series(dtype=float))

    assert np.isnan(metrics["cumulative_return"])
    assert np.isnan(metrics["sharpe_ratio"])


def test_download_daily_data_reports_missing_yahoo_symbols(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Volume"]],
    )
    data = pd.DataFrame(
        [[100, 101, 99, 100.5, 1_000_000]] * len(dates),
        index=dates,
        columns=columns,
    )

    monkeypatch.setattr("src.momentum_strategy.yf.download", lambda *args, **kwargs: data)

    with pytest.raises(ValueError, match="BAD"):
        download_daily_data(["AAPL", "BAD"], "2024-01-01", "2024-02-01")
