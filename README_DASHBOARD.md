# Greek Energy Flow Analysis - Dashboard Guide

## Overview

The Greek Energy Flow Analysis Dashboard provides a comprehensive visualization interface for analyzing options Greeks, market regimes, and trade recommendations. This guide explains how to use the dashboard, understand its components, and interpret the trade context information.

## Getting Started

### Installation

1. Ensure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard using one of the following methods:

   **Standard Dashboard:**
   ```bash
   python run_dashboard.py --mode dashboard
   ```

   **Quick Launch with Fixes:**
   ```bash
   fix_and_run_dashboard.bat
   ```

   **ML-Enhanced Dashboard:**
   ```bash
   fix_ml_and_run_dashboard.bat
   ```

### Dashboard Components

The dashboard consists of several key components:

1. **Recommendation List**: Displays all available trade recommendations
2. **Filter Controls**: Allows filtering recommendations by strategy, risk level, market regime, etc.
3. **Details View**: Shows detailed information about the selected recommendation
4. **Greek Analysis**: Visualizes options Greeks and energy flow patterns
5. **Market Context**: Displays market regime information and context
6. **Pattern Analysis**: Shows ordinal pattern analysis (if available)

## Trade Context Structure

The trade context provides essential information about market conditions and trade parameters. Understanding this structure is crucial for interpreting recommendations.

### Standard Trade Context Fields

| Field | Description |
|-------|-------------|
| `market_regime.primary` | Primary market regime (e.g., "Bullish Trend", "Vanna-Driven") |
| `market_regime.volatility` | Volatility component of market regime |
| `market_regime.confidence` | Confidence score for the market regime classification |
| `volatility_regime` | Current volatility regime (High/Normal/Low) |
| `hold_time_days` | Recommended holding period in days |
| `dominant_greek` | The dominant Greek influencing the market (e.g., "vanna", "charm") |
| `energy_state` | Current energy state of the market |
| `entropy_score` | Entropy score measuring market disorder |
| `anomalies` | List of detected anomalies in Greek metrics |
| `greek_metrics` | Key Greek values for the underlying |
| `support_levels` | Identified support price levels |
| `resistance_levels` | Identified resistance price levels |
| `confidence_score` | Overall confidence in the recommendation |
| `ml_prediction` | Machine learning prediction (if available) |
| `ml_confidence` | Confidence score for ML prediction |
| `pattern_name` | Identified ordinal pattern (if available) |
| `pattern_confidence` | Confidence score for pattern recognition |

### Example Trade Context

```json
{
  "market_regime": {
    "primary": "Vanna-Driven",
    "volatility": "Normal",
    "confidence": 0.85
  },
  "volatility_regime": "Normal",
  "hold_time_days": 7,
  "dominant_greek": "vanna",
  "energy_state": "Accumulation",
  "entropy_score": 0.65,
  "anomalies": ["Gamma Spike", "Vanna Reversal"],
  "greek_metrics": {
    "delta": 0.45,
    "gamma": 0.08,
    "vanna": 0.12,
    "charm": -0.03
  },
  "support_levels": [150.25, 148.75],
  "resistance_levels": [155.50, 158.25],
  "confidence_score": 0.78,
  "ml_prediction": "Bullish",
  "ml_confidence": 0.82,
  "pattern_name": "Vanna Reversal",
  "pattern_confidence": 0.75
}
```

## Ordinal Pattern Trade Context

The dashboard supports displaying ordinal pattern analysis in the trade context. To ensure proper display:

1. **Ordinal Pattern Format**:
   Trade recommendations should include ordinal pattern data in the TradeContext:

   ```json
   "TradeContext": {
     "market_regime": {
       "primary": "Bullish Trend",
       "volatility": "Normal"
     },
     "ordinal_patterns": {
       "delta": {
         "pattern": [0, 1, 2],
         "confidence": 0.85,
         "expected_value": 0.045
       },
       "gamma": {
         "pattern": [1, 0, 2],
         "confidence": 0.75,
         "expected_value": 0.02
       },
       "vanna": {
         "pattern": [0, 2, 1],
         "confidence": 0.65,
         "expected_value": 0.03
       }
     }
   }
   ```

2. **Creating Sample Recommendations**:
   If you don't have any recommendations, you can create sample ones with:

   ```bash
   python create_sample_recommendations.py --num 10
   ```

3. **Ordinal Pattern Visualization**:
   The dashboard will display ordinal patterns in the "Market Context" tab when available.

4. **Pattern Interpretation**:
   - [0,1,2]: Steadily increasing (bullish)
   - [2,1,0]: Steadily decreasing (bearish)
   - [1,0,2]: Down then up (reversal)
   - [2,0,1]: Down sharply then up (reversal)
   - [0,2,1]: Up then down (reversal)
   - [1,2,0]: Up then down sharply (reversal)

## Strategy Types

The dashboard supports multiple strategy types:

1. **Greek Flow**: Strategies based on options Greek energy flow analysis
2. **Momentum**: Strategies based on price momentum
3. **Mean Reversion**: Strategies that capitalize on price reversions to the mean
4. **Volatility Expansion**: Strategies that profit from volatility increases
5. **ML Enhanced**: Strategies enhanced with machine learning predictions
6. **Ordinal**: Strategies based on ordinal pattern analysis

### Ordinal Pattern Strategies

Ordinal pattern strategies identify recurring patterns in Greek metrics to predict future price movements. The dashboard displays:

- Pattern name and confidence level
- Expected direction (Bullish/Bearish)
- Pattern-specific entry criteria
- Historical performance of the pattern

## Using the Dashboard

### Filtering Recommendations

Use the filter controls to narrow down recommendations:

1. **Strategy**: Filter by strategy type (e.g., "Greek Flow", "Ordinal")
2. **Risk Level**: Filter by risk category (Low/Medium/High)
3. **Market Regime**: Filter by current market regime
4. **Aligned with Market**: Show only recommendations aligned with the current market regime

### Viewing Recommendation Details

Click on a recommendation in the list to view its details:

1. **Trade Structure**: Shows the basic trade structure, including entry/exit points
2. **Entry Criteria**: Displays conditions that should be met before entering the trade
3. **Exit Plan**: Shows profit targets, stop loss levels, and time-based exit rules
4. **Milestones**: Displays expected progress milestones for the trade

### Analyzing Market Context

The Market Context tab provides:

1. **Market Regime**: Current market regime and its characteristics
2. **Volatility Regime**: Current volatility environment
3. **Energy State**: State of Greek energy in the market
4. **Support/Resistance**: Key price levels to watch

### Interpreting Greek Analysis

The Greek Analysis tab shows:

1. **Greek Energy Flow**: Visualization of energy flow between Greeks
2. **Dominant Greek**: The Greek currently dominating market behavior
3. **Greek Anomalies**: Unusual patterns in Greek metrics
4. **Energy Concentration**: Areas where Greek energy is concentrated

## Pattern Analysis Tab

The Pattern Analysis tab provides visualization and details about detected ordinal patterns in the Greek metrics. This tab is only visible when pattern data is available for the selected recommendation.

### Pattern Information

- **Pattern Name**: The identified pattern type (e.g., "Rising Vanna", "Gamma Compression")
- **Pattern Confidence**: Confidence score for the pattern identification (0.0-1.0)
- **Pattern Description**: Description of what the pattern indicates
- **Pattern Statistics**: Statistical metrics related to the pattern
- **Pattern Visualization**: Visual representation of the pattern

### Using Pattern Analysis

Pattern analysis provides additional context for trade decisions:

1. **High Confidence Patterns**: Patterns with confidence scores above 0.7 are more reliable
2. **Pattern Duration**: Check the expected duration of the pattern effect
3. **Pattern Correlation**: Some patterns have strong correlations with specific market moves

## Standardizing Trade Recommendations

To ensure your custom trade recommendations work with the dashboard, use the unified trade format utility:

```python
from utils.unified_trade_format import standardize_trade_context

# Your custom recommendation
my_recommendation = {
    "symbol": "AAPL",
    "trade_type": "LONG",
    "entry_price": 150.25,
    "target_price": 165.50,
    "stop_price": 145.00,
    "trade_context": {
        "market_regime": "Bullish",
        "volatility_regime": "Normal"
    }
}

# Standardize the recommendation
standardized_rec = standardize_trade_context(my_recommendation)

# Now it will work with the dashboard
```

You can also batch convert recommendations:

```bash
python -m utils.unified_trade_format --input "your_recs_dir" --batch
```

## Troubleshooting

### Common Issues

1. **Missing Recommendations**: Run analysis scripts to generate recommendations:
   ```bash
   python run_regime_analysis.py --tickers AAPL MSFT
   ```

2. **Dashboard Display Issues**: Run the fix script:
   ```bash
   python fix_dashboard.py
   ```

3. **Trade Context Missing**: Ensure recommendations include trade context:
   ```bash
   python debug_trade_recommendations.py --symbol AAPL
   ```

4. **Pattern Analysis Missing**: Update pattern libraries:
   ```bash
   python update_pattern_library.py
   ```

### Generating Sample Data

If you don't have real data available, the dashboard can generate sample recommendations:

1. Launch the dashboard with the `--sample` flag:
   ```bash
   python run_dashboard.py --mode dashboard --sample
   ```

2. Or use the sample generation script:
   ```bash
   python generate_sample_recommendations.py --output results
   ```

## Advanced Features

### Loading Historical Data

For regime transition visualization:

1. Launch the dashboard
2. Click "Load Tracker Data" in the menu
3. Select the directory containing `regime_history.json`

### Customizing the Dashboard

You can customize the dashboard by editing:

1. **config.py**: Adjust configuration parameters
2. **tools/trade_dashboard.py**: Modify dashboard layout and behavior
3. **unified_trade_format.py**: Change trade recommendation format

## Contributing

Contributions to the dashboard are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dashboard Setup Requirements

### Required Files

The dashboard requires specific files to function properly:

1. **Market Regime Files**:
   - Primary: `results/market_regime/current_regime.json`
   - Format:
     ```json
     {
       "primary_label": "Bullish Trend",
       "secondary_label": "Secondary Classification",
       "volatility_regime": "Normal",
       "dominant_greek": "Delta",
       "timestamp": "2023-05-06T12:00:00"
     }
     ```
   - Create this file using: `python create_market_regime.py`

2. **Trade Recommendation Files**:
   - Located in: `results/` directory
   - Naming: `{symbol}_trade_recommendation.json` or `{symbol}_enhanced_recommendation.json`
   - Must include entry/exit prices and strategy information

### Troubleshooting Dashboard Issues

If the dashboard shows empty data or "Unknown" market regime:

1. **Missing Market Regime**: 
   ```bash
   python create_market_regime.py
   ```

2. **Empty Trade Recommendations**:
   ```bash
   python -m utils.unified_trade_format --input "results" --batch
   ```

3. **Dashboard Display Issues**:
   ```bash
   python fix_dashboard.py
   ```

4. **Complete Reset and Setup**:
   ```bash
   fix_and_run_dashboard.bat
   ```




