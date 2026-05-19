from fastapi.testclient import TestClient

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
    }.issubset(series[0])


def test_predictions_endpoint_returns_ranked_rows():
    response = client.get("/api/predictions?limit=5")

    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 5
    assert {"week", "ticker", "probability", "rank", "selected"}.issubset(rows[0])
