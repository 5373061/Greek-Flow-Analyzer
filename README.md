# Greek Energy Flow Analysis

A comprehensive Python library for analyzing options Greeks to identify energy flow patterns, market regimes, and potential trading opportunities.

## Features

- **Greek Energy Flow Analysis**: Analyze options Greeks (Delta, Gamma, Vanna, Charm) to identify energy concentration and potential price movement
- **Market Regime Classification**: Identify the current market regime based on Greek configurations
- **Reset Point Detection**: Detect potential price levels where market energy may reset
- **Trade Opportunity Analysis**: Identify potential trading opportunities based on Greek energy flow and price history
- **Stop Loss Calculation**: Calculate appropriate stop loss levels based on Greek energy metrics
- **Machine Learning Integration**: Advanced ML models for pattern recognition and prediction
- **Entropy Analysis**: Measure and visualize the entropy of Greek distributions to identify anomalies
- **Momentum Analysis**: Analyze price momentum and volatility trends
- **Comprehensive Testing**: Extensive test suite with unit and integration tests
- **Pattern Analysis**: Identify and analyze ordinal patterns in Greek metrics to enhance trading decisions
- **Interactive Dashboard**: Visualize analysis results and trade recommendations

## Trade Context System

The Greek Energy Flow Analysis includes a standardized trade context system that provides essential information about market conditions and trade parameters:

### Key Trade Context Components

- **Market Regime**: Primary market regime and volatility environment
- **Dominant Greek**: The Greek currently dominating market behavior
- **Energy State**: Current state of Greek energy in the market
- **Entropy Score**: Measure of market disorder
- **Greek Metrics**: Key Greek values for the underlying
- **Support/Resistance Levels**: Key price levels identified by the analysis
- **Hold Time**: Recommended holding period based on regime and pattern analysis

This trade context information is used throughout the system to enhance trade recommendations and provide a comprehensive view of market conditions.

### SEQUENCE OF STEPS ###

1. **Install the Package**:
   - Clone the repository: `git clone https://github.com/yourusername/greek-energy-flow.git`
   - Navigate to project directory: `cd greek-energy-flow`
   - Install dependencies: `pip install -r requirements.txt`
   - Run tests to verify installation: `pytest`

2. **Prepare Your Data**:
   - Load options data from CSV: `options_data = pd.read_csv('your_options_data.csv')`
   - Or fetch from API: `options_data = fetch_options_data(symbol, date)`
   - Prepare market data dictionary with current price, volatility metrics, etc.
   - Validate data format with required columns

3. **Run Scripts in Sequence**:
   - **Data Acquisition & Processing**: `python run_dashboard.py --mode analysis`
   - **ML Training** (after collecting enough historical data): `python run_ml_regime_analysis.py --train`
   - **ML Prediction & Trade Signals**: `python run_ml_regime_analysis.py --predict`
   - **Live Dashboard & Monitoring**: `python run_dashboard.py --mode dashboard --refresh-interval 300`
   - **Live Tracking** (optional): `python run_live_tracker.py`

4. **View Results**:
   - Analysis results are saved to: `results/{symbol}_analysis_results.json`
   - Trade recommendations: `results/{symbol}_trade_recommendation.json`
   - Enhanced ML recommendations: `results/{symbol}_enhanced_recommendation.json`
   - Visualizations are saved to: `results/charts/{symbol}/`
   - Review market regime classification and reset points
   - Examine energy levels and Greek anomalies

5. **Execute Trades**:
   - Follow entry conditions from recommendation
   - Use suggested position sizing and risk parameters
   - Set up alerts for exit conditions
   - Monitor trade progress against projected outcomes

### END SEQUENCE ###

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/greek-energy-flow.git
cd greek-energy-flow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests to verify installation:
```bash
pytest
```

## Quick Start

```python
from greek_flow import GreekEnergyFlow, get_config
import pandas as pd

# Load your options data
options_data = pd.read_csv('your_options_data.csv')

# Initialize the analyzer
analyzer = GreekEnergyFlow()

# Market data
market_data = {
    'currentPrice': 150.25,
    'impliedVolatility': 0.25,
    'historicalVolatility': 0.20,
    'riskFreeRate': 0.04
}

# Run analysis
results = analyzer.analyze_greek_energy(options_data, market_data)

# Print results
print(f"Market Regime: {results['market_regime']}")
print(f"Reset Points: {results['reset_points']}")
print(f"Energy Levels: {results['energy_levels']}")

# Generate entropy report
report = analyzer.generate_entropy_report()
```

## Using the Pipeline Manager

The Pipeline Manager provides a unified interface for running the full analysis pipeline:

```python
from analysis.pipeline_manager import AnalysisPipeline
import pandas as pd

# Initialize the pipeline
pipeline = AnalysisPipeline()

# Load your data
options_data = pd.read_csv('options_data.csv')
price_data = pd.read_csv('price_data.csv')

# Run the full analysis
results = pipeline.run_full_analysis(
    symbol="AAPL",
    options_data=options_data,
    price_data=price_data,
    current_price=150.25,
    skip_entropy=False
)

# Access the results
print(f"Market Regime: {results['formatted_results']['market_regime']}")
print(f"Reset Points: {results['formatted_results']['reset_points']}")
```

## Command Line Interface

The project includes several command-line scripts for running analyses:

### Run Full Analysis Pipeline

The recommended way to run the full analysis is using the `run_dashboard.py` script, which handles the entire workflow:

```bash
# Run full analysis with default tickers
python run_dashboard.py --mode analysis

# Analyze specific tickers
python run_dashboard.py --tickers AAPL MSFT GOOG --mode analysis

# Use tickers from a file
python run_dashboard.py --tickers-file my_tickers.txt --mode analysis

# Specify output directory
python run_dashboard.py --output-dir results --mode analysis

# Run with debug logging
python run_dashboard.py --debug --mode analysis
```

This script:
1. Fetches and processes data for each ticker
2. Runs Greek energy flow analysis
3. Performs entropy analysis
4. Generates trade recommendations
5. Creates visualizations for each ticker

### Launch Interactive Dashboard

To launch the interactive dashboard with all visualizations:

```bash
# Launch dashboard with auto-refresh (every 5 minutes)
python run_dashboard.py --mode dashboard --refresh-interval 300

# Launch dashboard with specific tickers
python run_dashboard.py --tickers AAPL MSFT --mode dashboard

# Launch dashboard with tickers from file
python run_dashboard.py --tickers-file my_tickers.txt --mode dashboard
```

The dashboard provides:
1. Interactive visualization of all analysis components
2. Auto-refreshing data at specified intervals
3. Trade recommendations based on current market conditions
4. Greek energy flow and entropy visualizations
5. Ordinal pattern analysis and visualization

### Using the Dashboard Fix Script

If you encounter issues with the dashboard, you can use the provided fix script:

```bash
# Fix dashboard issues and launch
python fix_dashboard.py
python -m tools.trade_dashboard
```

Or simply run the batch file:

```bash
# Run the fix and launch script
fix_and_run_dashboard.bat
```

This will:
1. Fix the dashboard implementation
2. Convert all trade recommendations to a dashboard-compatible format
3. Create necessary market regime files
4. Launch the dashboard

### ML Training and Prediction

For machine learning enhanced analysis:

```bash
# Train ML models on historical data
python run_ml_regime_analysis.py --train

# Generate ML predictions
python run_ml_regime_analysis.py --predict

# Run ML enhanced trading analysis
python run_ml_enhanced_trading.py
```

### Other Analysis Scripts

```bash
# Run analysis with Pipeline Manager directly
python run_with_pipeline.py --tickers AAPL --analysis-type both

# Run market regime analysis on existing results
python analysis/market_regime_analyzer.py --results-dir results --validate

# Generate trade recommendations from analysis results
python analysis/trade_recommendations.py

# Run scanner for trading opportunities
python run_scanner.py --sector Technology --min-market-cap 100

# Quick analysis of predefined tickers
python run_my_tickers.py
```

## Project Structure

### Main Scripts

- `run_regime_analysis.py` - **Main entry point** for running the full analysis pipeline
- `run_with_pipeline.py` - Analysis using the Pipeline Manager
- `run_scanner.py` - Trading opportunity scanner
- `run_my_tickers.py` - Quick analysis for a predefined list of tickers
- `analysis/trade_recommendations.py` - Generate trade recommendations
- `run_dashboard.py` - Run analysis and launch dashboard
- `fix_dashboard.py` - Fix dashboard-related issues
- `run_live_tracker.py` - Live tracking of instruments

### Analysis Components

- `analysis/pipeline_manager.py` - Orchestrates the analysis pipeline
- `analysis/symbol_analyzer.py` - Main analysis orchestrator for individual symbols
- `analysis/greek_analyzer.py` - Greek energy flow analysis
- `analysis/market_regime_analyzer.py` - Market regime classification
- `analysis/pattern_analyzer.py` - Pattern recognition for market conditions
- `analysis/pattern_ml_integrator.py` - ML integration for pattern recognition
- `analysis/momentum_analyzer.py` - Price momentum analysis
- `analysis/risk_analyzer.py` - Risk metrics calculation

### Pattern Analysis

- `ordinal_pattern_analyzer.py` - Analyzes ordinal patterns in Greek metrics
- `cross_greek_patterns.py` - Analyzes relationships between patterns in different Greeks
- `pattern_integration.py` - Integrates pattern analysis with the pipeline
- `visualization/pattern_visualizer.py` - Visualizes Greek patterns

For more detailed information about pattern analysis, see [README_PATTERNS.md](README_PATTERNS.md).

### Data Management

- `data/data_loader.py` - Data loading from various sources
- `data/cache_manager.py` - Caching for API data
- `api_fetcher.py` - Fetches options data from external APIs

### Visualization

- `analyzer_visualizations/chart_generator.py` - Charting and visualization
- `analyzer_visualizations/formatters.py` - Report formatting
- `tools/trade_dashboard.py` - Interactive dashboard for visualizing results

### Machine Learning Models

- `greek_flow/ml_models.py` - ML models for Greek analysis
- `greek_flow/lstm_models.py` - LSTM models for time series prediction
- `greek_flow/xgboost_models.py` - XGBoost models for classification and regression

## Analysis Pipeline Flow

The analysis pipeline follows these steps:

1. **Data Loading**: Load options data and price history
2. **Greek Analysis**: Calculate and analyze options Greeks
3. **Market Regime Classification**: Identify the current market regime
4. **Pattern Recognition**: Detect market patterns using rule-based and ML approaches
5. **Entropy Analysis**: Analyze the entropy of Greek distributions to identify anomalies
6. **Momentum Analysis**: Analyze price momentum and volatility trends
7. **Trade Opportunity Analysis**: Identify potential trading opportunities
8. **Visualization**: Generate charts and reports
9. **Market Regime Analysis**: Analyze market regimes across multiple symbols

## Pattern Analysis

The pattern analysis module identifies and analyzes ordinal patterns in Greek metrics:

1. **Ordinal Patterns**: Represent the relative ordering of values in a time series window
2. **Pattern Profitability**: Analyze the profitability of different patterns
3. **Cross-Greek Patterns**: Analyze relationships between patterns in different Greeks
4. **Pattern Library**: Build and save a library of patterns for future reference
5. **Pattern Recognition**: Recognize patterns in current data

For more details, see `README_PATTERNS.md`.

## Configuration

The library uses a centralized configuration system. You can:

1. Use the default configuration
2. Override specific parameters
3. Load a custom configuration file

```python
from greek_flow.config import get_config, update_config

# Get current configuration
config = get_config()

# Update specific parameters
update_config({
    'reset_factors': {
        'gammaFlip': 0.40,  # Increase gamma flip importance
        'vannaPeak': 0.30   # Increase vanna peak importance
    },
    'ml_config': {
        'ensemble_threshold': 0.6,  # Minimum confidence threshold
        'use_weighted_voting': True  # Use confidence-weighted voting
    }
})
```

## API Integration

The library can fetch options data from the Polygon.io API:

```python
from api_fetcher import fetch_options_chain_snapshot, fetch_underlying_snapshot

# Set your API key in config.py or pass it directly
api_key = "YOUR_POLYGON_API_KEY"

# Fetch underlying data
underlying_data = fetch_underlying_snapshot("AAPL", api_key)

# Fetch options chain
options_data = fetch_options_chain_snapshot("AAPL", api_key)

# Process the data
from api_fetcher import preprocess_api_options_data
processed_options = preprocess_api_options_data(options_data)
```

## Trade Recommendations

The library generates detailed trade recommendations based on Greek energy flow analysis:

```python
from analysis.trade_recommendations import generate_trade_recommendation

# Generate recommendation
recommendation = generate_trade_recommendation(
    analysis_results=results,
    entropy_data=entropy_results,
    current_price=150.25
)

# Print recommendation
print(f"Strategy: {recommendation['strategy']['name']}")
print(f"Structure: {recommendation['strategy']['structure']}")
print(f"Direction: {recommendation['strategy']['direction']}")
print(f"Profit Target: {recommendation['risk_management']['profit_target_percent']}%")
print(f"Stop Loss: {recommendation['risk_management']['stop_loss_percent']}%")
```

## Machine Learning Integration

The library integrates several machine learning approaches for advanced pattern recognition and prediction:

### Available ML Models

1. **LightGBM Models**: For market regime classification and pattern recognition
2. **XGBoost Models**: For gamma scalping opportunities and option price prediction
3. **LSTM Models**: For time series prediction of volatility regimes and price movements
4. **Ensemble Methods**: Combines multiple models for improved accuracy

### Using ML for Pattern Recognition

```python
from analysis.pattern_analyzer import PatternRecognizer
from analysis.pattern_ml_integrator import PatternMLIntegrator

# Initialize pattern recognizer with ML integration
recognizer = PatternRecognizer(models_dir='models')

# Predict pattern using ML models
pattern = recognizer.predict_pattern(greek_results, momentum_data, use_ml=True)
print(f"Detected pattern: {pattern['pattern']} (Confidence: {pattern['confidence']})")
print(f"Description: {pattern['description']}")
```

## Testing

The project includes a comprehensive test suite to ensure reliability:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline_manager.py

# Run tests with coverage report
pytest --cov=. --cov-report=html

# Run tests with verbose output
pytest -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure that your code passes all tests and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dashboard Interface

The project includes a comprehensive trade dashboard for visualizing analysis results:

```bash
# Launch the trade dashboard
python -m tools.trade_dashboard
```

### Dashboard Features

- **Trade Recommendations**: View and filter trade recommendations based on strategy, risk level, and market alignment
- **Greek Analysis Visualization**: Visualize options Greeks (Delta, Gamma, Vanna, Charm) and energy flow patterns
- **Market Regime Tracking**: Monitor current market regime and historical regime transitions
- **Trade Structure Details**: View detailed entry/exit criteria, risk parameters, and expected outcomes
- **Duration Analysis**: Analyze typical holding periods and optimal exit timing for different strategies
- **Alignment Indicators**: Identify recommendations aligned with the current market regime
- **Pattern Analysis**: View and analyze ordinal patterns in Greek metrics

The dashboard automatically loads analysis results from the `results` directory and provides an intuitive interface for exploring trading opportunities based on Greek energy flow analysis.

### Strategy Types

The dashboard supports multiple strategy types:

1. **Greek Flow**: Strategies based on options Greek energy flow analysis
2. **Momentum**: Strategies based on price momentum
3. **Mean Reversion**: Strategies that capitalize on price reversions to the mean
4. **Volatility Expansion**: Strategies that profit from volatility increases
5. **ML Enhanced**: Strategies enhanced with machine learning predictions
6. **Ordinal**: Strategies based on ordinal pattern analysis

For more detailed information about the dashboard, see [README_DASHBOARD.md](README_DASHBOARD.md).

### Troubleshooting Dashboard Issues

If you encounter issues with the dashboard:

1. Run the dashboard fix script:
   ```bash
   python fix_dashboard.py
   ```

2. Or use the provided batch file:
   ```bash
   fix_and_run_dashboard.bat
   ```

3. For ML-enhanced recommendations:
   ```bash
   fix_ml_and_run_dashboard.bat
   ```

### Data Requirements

The dashboard is designed to work with the data files generated by the analysis pipeline. It will:

1. First extract all needed data directly from recommendation files (e.g., `SYMBOL_enhanced_recommendation.json`)
2. Fall back to loading separate analysis files only if needed (e.g., `SYMBOL_analysis.json`)
3. Display whatever data is available, gracefully handling missing components

For optimal dashboard functionality, run one of these commands to generate the necessary data:

```bash
# Full analysis (generates all data for complete dashboard functionality)
python run_regime_analysis.py --tickers SYMBOL1 SYMBOL2

# Quick analysis (faster, generates essential data for basic dashboard functionality)
python run_regime_analysis.py --tickers SYMBOL1 SYMBOL2 --analysis-type greek --skip-entropy
```

The dashboard will work with whatever data is available, but more complete data will enable more visualizations and insights.

### Optional: Loading Historical Data

For regime transition visualization, you can load historical regime data:

1. Launch the dashboard
2. Click "Load Tracker Data" in the menu
3. Select the directory containing `regime_history.json`

This enables visualization of regime transitions over time, but is not required for basic dashboard functionality.

## Dashboard Setup

The dashboard requires properly formatted data files to display trade recommendations and market context:

1. **Market Regime Files**: The dashboard looks for market regime information in these locations:
   - `results/market_regime/current_regime.json` (primary location)
   - `results/market_regime_summary.json`
   - `results/regime_validation.json`
   - `results/market_bias.json`

2. **Creating Market Regime Files**: You can create the required market regime file using:
   ```bash
   python create_market_regime.py
   ```
   This creates a file with the current market regime classification.

3. **Trade Recommendation Files**: The dashboard loads trade recommendations from:
   - `results/{symbol}_trade_recommendation.json`
   - `results/{symbol}_enhanced_recommendation.json`

4. **Launching the Dashboard**:
   ```bash
   # Quick setup and launch
   fix_and_run_dashboard.bat
   
   # Or run directly
   python -m tools.trade_dashboard
   ```

For ML-enhanced recommendations, use:
```bash
fix_ml_and_run_dashboard.bat
```




