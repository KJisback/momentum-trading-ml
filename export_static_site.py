import json
import shutil
from pathlib import Path

from src.saas_app import equity, predictions, summary


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
WEB_DIR = PROJECT_ROOT / "web"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_asset(name: str) -> None:
    shutil.copy2(WEB_DIR / name, DOCS_DIR / name)


def copy_download(name: str) -> None:
    target_dir = DOCS_DIR / "downloads"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_DIR / name, target_dir / name)


if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    copy_asset("index.html")
    copy_asset("config.js")
    copy_asset("styles.css")
    copy_asset("app.js")

    write_json(DATA_DIR / "summary.json", summary())
    write_json(DATA_DIR / "equity.json", equity())
    write_json(DATA_DIR / "predictions.json", predictions(limit=80))
    write_json(
        DATA_DIR / "health.json",
        {
            "status": "ok",
            "files": {
                "performance_metrics.csv": True,
                "weekly_stock_predictions.csv": True,
                "weekly_portfolio_returns.csv": True,
                "weekly_features.csv": True,
            },
        },
    )

    for file_name in [
        "daily_ohlcv.csv",
        "weekly_features.csv",
        "weekly_stock_predictions.csv",
        "weekly_portfolio_returns.csv",
        "performance_metrics.csv",
    ]:
        copy_download(file_name)

    print(f"Static dashboard exported to {DOCS_DIR}")
