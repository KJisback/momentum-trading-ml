# Short Report: ML Momentum Strategy

## Model Used

The default model is a Random Forest classifier wrapped in a scikit-learn pipeline with median imputation and feature scaling. The model predicts the probability that a stock's next-week return will be positive.

## Key Features

The feature set focuses on price momentum, trend, volatility, and volume:

- 1, 2, 4, 8, 12, and 26-week returns
- 4 and 12-week rolling volatility
- Price versus 4 and 12-week moving averages
- Position within the 26-week high-low range
- Current volume versus 4-week average volume

## Validation Setup

- Training period: 2017-2022
- Test period: 2023-2025
- Universe: AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK.B
- Weekly rebalance using Friday weekly bars
- Strategy selects the top 2 stocks by predicted probability each week
- Equal weights: 50% and 50%
- Costs: 0.1% at entry and 0.1% at exit

## Backtesting Method

Each test week, all stocks are ranked by predicted probability of positive next-week return. The top 2 are selected and held for one week. Gross weekly portfolio return is the average of the selected stocks' realized next-week returns. Net return subtracts 0.2% total weekly transaction cost.

## Major Findings

Run the notebook to fill in the final values from `outputs/performance_metrics.csv`.

Recommended discussion points:

- Whether the model beats a simple long-only benchmark after costs
- Whether performance is concentrated in a few mega-cap technology names
- How much transaction costs reduce the edge
- Whether hit rate and Sharpe ratio remain attractive after costs
- Whether the train/test split is stable enough or walk-forward retraining is needed

## Limitations

- The model uses a fixed train/test split and does not retrain walk-forward.
- Yahoo Finance data may revise historical prices.
- The cost model is simplified and excludes slippage, market impact, taxes, and liquidity constraints.
- The strategy is long-only and always invested in exactly two names.
