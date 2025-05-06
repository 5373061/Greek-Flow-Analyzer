# ML Implementation for Greek Energy Flow II

This document explains how the machine learning (ML) components have been integrated into the Greek Energy Flow II system to enhance trading decisions based on options Greek analysis and entropy patterns.

## Overview

The ML implementation adds the following capabilities to the Greek Energy Flow II system:

1. **Regime Classification**: ML models trained to predict market regimes (Vanna-Driven, Charm-Dominated, etc.) based on Greek profiles and entropy analysis
2. **Transition Detection**: Algorithms that detect regime transitions to optimize entry and exit points
3. **Pattern Recognition**: Identification of entropy patterns that precede significant market moves
4. **Enhanced Signal Generation**: Integration of ML predictions with traditional Greek analysis for higher-confidence trade signals
5. **Risk Management**: Dynamic position sizing based on ML prediction confidence and entropy state

## Components

### 1. ML Models (models/ml/regime_classifier.py)

- **GreekRegimeClassifier**: Core ML model that predicts market regimes
- **RegimeTransitionPredictor**: Specialized model for detecting transitions between regimes
- **EntropyPatternRecognizer**: Pattern recognition system for identifying significant entropy patterns

### 2. Trade Execution (models/ml/trade_executor.py)

- **MLTradeExecutor**: Executes trades based on ML-enhanced signals
- **MLEnhancedAnalyzer**: Combines traditional Greek analysis with ML predictions

### 3. Integration Scripts

- **run_ml_regime_analysis.py**: Runs ML-based market regime analysis
- **run_ml_enhanced_trading.py**: Orchestrates the entire ML-enhanced trading workflow

## Installation Requirements

All ML components use scikit-learn, which is already included in the project's requirements.txt.

## Usage

### 1. Training ML Models

To train the ML models on your existing Greek analysis data:

```bash
python run_ml_enhanced_trading.py --tickers AAPL MSFT QQQ SPY LULU TSLA CMG WYNN ZM SPOT --train
```

Alternatively, you can provide a CSV file with ticker symbols:

```bash
python run_ml_enhanced_trading.py --ticker-file your_tickers.csv --train
```

### 2. Running Analysis with ML Predictions

To run the complete analysis pipeline with ML predictions:

```bash
python run_ml_enhanced_trading.py --tickers AAPL MSFT QQQ SPY --analyze --fetch-data
python run_ml_enhanced_trading.py --tickers AAPL MSFT QQQ SPY --predict
```

### 3. Running Trading Simulation

To test the ML-enhanced trading strategy in a simulation:

```bash
python run_ml_enhanced_trading.py --tickers AAPL MSFT QQQ SPY --simulate --simulation-days 10 --interval 60
```

### 4. Live Trading

To run the ML-enhanced live trading system:

```bash
python run_ml_enhanced_trading.py --tickers AAPL MSFT QQQ SPY --live --interval 5 --api-key YOUR_API_KEY
```

## Data Flow

1. **Analysis Phase**:
   - Run traditional Greek analysis pipeline
   - Calculate entropy metrics
   - Generate energy levels and reset points

2. **ML Phase**:
   - Train ML models on Greek and entropy data
   - Predict market regimes
   - Detect regime transitions
   - Recognize entropy patterns

3. **Trading Phase**:
   - Generate ML-enhanced trade signals
   - Validate signals against Greek analysis
   - Execute trades with optimized entries and exits
   - Monitor entropy changes for early exit signals

## Ticker Input Options

The system supports two ways to specify which tickers to analyze:

1. **Command-line arguments**: Use the `--tickers` flag followed by a space-separated list of ticker symbols
2. **CSV/TXT file**: Use the `--ticker-file` flag pointing to a file containing tickers (one per line for TXT, or in the first column for CSV)

## Market Data

The system can either use:

1. **Mock data** (for testing and simulation)
2. **Real market data** via API (for live trading)

To use real market data, obtain an API key and specify it with the `--api-key` parameter. The system will fetch historical data before running analysis if the `--fetch-data` flag is provided.

## ML Model Types

The system supports two types of ML models:

1. **Random Forest** (default): Generally provides better performance but may be slower to train
2. **Gradient Boosting**: May provide better generalization for certain market conditions

Select the model type with the `--model-type` flag.

## Output Files

The ML system generates the following output files:

1. **Trained Models**: Saved in `models/ml/` directory
2. **ML Predictions**: Saved in `results/ml_predictions/` directory
3. **Enhanced Recommendations**: Saved in `results/` directory with `_enhanced_recommendation.json` suffix
4. **Simulation Results**: Saved in `results/ml_simulation/` directory
5. **Live Trading Results**: Saved in `results/ml_live/` directory

## Entropy-Based Exit Signals

A key feature of this implementation is the use of entropy changes to trigger trade exits:

1. The system monitors changes in entropy state for each active position
2. When entropy state transitions (e.g., from "Concentrated Energy" to "Dispersed Energy"), it may trigger an exit
3. The direction of entropy change is considered in context with the trade direction
4. This approach allows for earlier exits before traditional stop-loss or take-profit levels are reached

## Performance Metrics

The system tracks the following performance metrics:

1. **Win Rate**: Percentage of profitable trades
2. **Average Profit/Loss**: Average P&L per trade
3. **Largest Win/Loss**: Maximum profit and loss values
4. **Per-Ticker Performance**: Detailed performance breakdowns by ticker
5. **Regime Accuracy**: Accuracy of regime predictions vs. actual market behavior

## Future Enhancements

Planned enhancements to the ML implementation include:

1. **Deep Learning Models**: Integration of neural networks for improved pattern recognition
2. **Reinforcement Learning**: Development of RL agents for optimizing entry/exit timing
3. **Ensemble Methods**: Combining multiple ML models for more robust predictions
4. **Real-time Entropy Analysis**: Streaming analysis of entropy changes for immediate action
5. **Adaptive Risk Management**: Dynamically adjusting position sizing based on ML confidence
