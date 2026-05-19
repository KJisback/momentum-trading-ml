import argparse
from pathlib import Path

from src.momentum_strategy import RAW_TICKERS, StrategyConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ML momentum strategy backtest.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated CSV outputs.")
    parser.add_argument("--start", default="2017-01-01", help="Download start date, inclusive.")
    parser.add_argument("--end", default="2026-01-01", help="Download end date, exclusive.")
    parser.add_argument("--train-end", default="2022-12-31", help="Last training date.")
    parser.add_argument("--test-start", default="2023-01-01", help="First out-of-sample test date.")
    parser.add_argument("--top-n", type=int, default=2, help="Number of stocks to select each week.")
    parser.add_argument("--entry-cost", type=float, default=0.001, help="Entry transaction cost as decimal.")
    parser.add_argument("--exit-cost", type=float, default=0.001, help="Exit transaction cost as decimal.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the classifier.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Random Forest. Use -1 if supported.")
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "hist_gradient_boosting"],
        default="random_forest",
        help="Classifier to train. Hist gradient boosting is better for larger datasets.",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(RAW_TICKERS),
        help="Comma-separated ticker list. Use BRK.B in user-facing input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    config = StrategyConfig(
        start=args.start,
        end=args.end,
        train_end=args.train_end,
        test_start=args.test_start,
        top_n=args.top_n,
        entry_cost=args.entry_cost,
        exit_cost=args.exit_cost,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        model_type=args.model_type,
    )
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    results = run_pipeline(output_dir=output_dir, config=config, tickers=tickers)

    print("Model metrics")
    for key, value in results["model_metrics"].items():
        print(f"{key}: {value}")

    print("\nPerformance metrics")
    print(results["performance"].to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")
