from fastapi.testclient import TestClient

import pytest

import src.saas_app as saas_app
from src.saas_app import app


client = TestClient(app)


def test_health_endpoint_reports_output_files():
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "files" in payload


def test_summary_endpoint_is_human_readable_when_outputs_exist():
    response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["headline"]["netCumulativeReturn"].endswith("%")
    assert payload["headline"]["latestWeeklyReturn"].endswith("%")
    assert payload["headline"]["currentDrawdown"].endswith("%")
    assert payload["headline"]["rollingAvgReturn4w"].endswith("%")
    assert payload["headline"]["jensensAlpha"].endswith("%")
    assert payload["headline"]["riskMood"] in {"Calm", "Elevated", "Stressed"}
    assert payload["selectedStocks"]
    assert payload["plainEnglish"]


def test_equity_endpoint_returns_chart_series():
    response = client.get("/api/equity")

    assert response.status_code == 200
    series = response.json()["series"]
    assert series
    assert {
        "week",
        "grossEquity",
        "netEquity",
        "grossDrawdown",
        "netDrawdown",
        "rollingNetReturn4w",
        "rollingNetVolatility4w",
        "marketVolatility",
        "previousMarketVolatility",
    }.issubset(series[0])


def test_predictions_endpoint_returns_ranked_rows():
    response = client.get("/api/predictions?limit=5")

    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 5
    assert {"week", "ticker", "probability", "rank", "selected"}.issubset(rows[0])


def test_custom_run_rejects_invalid_ticker():
    response = client.post(
        "/api/custom-run",
        json={"tickers": ["AAPL", "MSFT", "BAD TICKER"], "topN": 2},
    )

    assert response.status_code == 422


def test_custom_run_returns_dashboard_payload(monkeypatch):
    weekly_features = saas_app.read_output_csv("weekly_features.csv")
    weekly_returns = saas_app.read_output_csv("weekly_portfolio_returns.csv")
    predictions = saas_app.read_output_csv("weekly_stock_predictions.csv")
    performance = saas_app.read_output_csv("performance_metrics.csv")

    def fake_pipeline(output_dir, config, tickers):
        return {
            "weekly_dataset": weekly_features[weekly_features["ticker"].isin(["AAPL", "MSFT", "GOOGL"])],
            "weekly_returns": weekly_returns,
            "predictions": predictions[predictions["ticker"].isin(["AAPL", "MSFT", "GOOGL"])],
            "performance": performance,
            "model_metrics": {"test_accuracy": 0.5, "test_roc_auc": 0.5},
        }

    monkeypatch.setattr(saas_app, "run_pipeline", fake_pipeline)
    response = client.post(
        "/api/custom-run",
        json={"tickers": ["AAPL", "MSFT", "GOOGL"], "topN": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tickers"] == ["AAPL", "MSFT", "GOOGL"]
    assert "summary" in payload
    assert "equity" in payload
    assert "predictions" in payload
