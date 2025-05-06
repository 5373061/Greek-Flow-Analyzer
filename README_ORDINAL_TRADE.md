# Greek Energy Flow II - Ordinal Pattern Trading System

## Overview

The Ordinal Pattern Trading System is an advanced options trading system that identifies recurring patterns in Greek metrics (delta, gamma, theta, vega, vanna, charm) and generates actionable entry/exit signals based on these patterns. Unlike traditional value-based analysis, this system focuses on the relative ordering (ordinal patterns) of Greek metrics over time.

## Core Concept

The system converts sequences of Greek values into ordinal patterns by tracking the rank ordering rather than the absolute values. For example, a sequence [0.3, 0.5, 0.2, 0.4] becomes the pattern (2, 0, 3, 1) - representing the relative positions from lowest to highest.

This approach captures the dynamic relationships between Greeks regardless of absolute market conditions, making it effective across different volatility regimes and price levels.

## Key Components

### 1. Ordinal Pattern Analyzer (`ordinal_pattern_analyzer.py`)

Extracts and analyzes patterns in Greek metrics:
- **Pattern Extraction**: Identifies recurring ordinal patterns across moneyness categories (ITM, ATM, OTM, VOL_CRUSH)
- **Profitability Analysis**: Evaluates which patterns historically precede profitable price movements
- **Pattern Library**: Maintains a database of statistically significant patterns and their success rates

### 2. Cross-Greek Pattern Analyzer (`cross_greek_patterns.py`)

Identifies relationships between patterns in different Greeks:
- **Cross-Pattern Detection**: Finds when patterns in one Greek (e.g., delta) predict patterns in another (e.g., gamma)
- **Predictive Relationships**: Quantifies how frequently pattern transitions occur across different Greeks
- **Enhanced Signals**: Combines cross-Greek insights with single-Greek patterns for stronger signals

### 3. Trade Signal Generator (`trade_signals.py`)

Converts pattern analysis into actionable trading signals:
- **Entry Signals**: Identifies ENTRY_LONG and ENTRY_SHORT opportunities based on pattern recognition
- **Exit Signals**: Generates EXIT_LONG and EXIT_SHORT signals when patterns indicate position closure
- **Risk Management**: Calculates appropriate take-profit and stop-loss levels based on historical pattern performance
- **Strategy Recommendations**: Suggests specific options strategies (calls, puts, spreads) based on signal strength

### 4. Pipeline Integration (`pattern_integration.py`)

Seamlessly integrates the ordinal pattern system with your existing trading pipeline:
- **Automatic Pattern Recognition**: Processes Greek data in real-time to identify active patterns
- **Recommendation Enhancement**: Combines pattern insights with your existing analysis
- **Position Management**: Monitors existing positions and generates appropriate adjustment signals

## How It Works

### Pattern Recognition Process

1. **Data Collection**: Historical Greek metrics are collected for each symbol
2. **Pattern Extraction**: The system identifies recurring ordinal patterns in 4-period windows
3. **Profitability Analysis**: Each pattern is evaluated based on subsequent price movements
4. **Library Building**: Statistically significant patterns are stored for future recognition
5. **Real-Time Recognition**: Current market data is analyzed to identify active patterns

### Signal Generation Logic

#### Entry Signals
- **Bullish Entry**: Rising delta/gamma patterns with positive historical returns
- **Bearish Entry**: Falling delta/gamma patterns with positive short returns
- **Volatility Entry**: Vanna/charm patterns indicating volatility expansion/contraction

#### Exit Signals
- **Profit Taking**: Patterns that historically precede reversals
- **Stop Loss**: Patterns indicating trend continuation against position
- **Time Exit**: Patterns near expiration suggesting closure

#### Risk Management
- **Take Profit**: Set at 70% of maximum historical return for the pattern
- **Stop Loss**: Set at 50% of maximum historical loss for the pattern
- **Position Sizing**: Based on pattern confidence and win rate

## Implementation Example

```python
from pattern_integration import integrate_with_pipeline

# Enable full pattern analysis
enhanced_pipeline = integrate_with_pipeline(
    your_pipeline,
    use_cross_greek=True,
    use_trade_signals=True
)

# Process a symbol
results = enhanced_pipeline.process_symbol(
    symbol="AAPL",
    options_data=options_df,
    price_data=price_df,
    current_price=185.75
)

# Check trade signals
if 'pattern_analysis' in results:
    signals = results['pattern_analysis'].get('trade_signals', [])
    for signal in signals:
        print(f"Signal: {signal['signal_type'].value}")
        print(f"Confidence: {signal['confidence']:.1%}")
        print(f"Strategy: {signal['strategy']}")
        print(f"Target Strike: {signal['target_strike']}")
```

## Pattern Categories

### 1. Moneyness-Based Patterns

- **ITM Patterns**: For delta > 0.7 options, typically used for position exits
- **ATM Patterns**: For 0.4 < delta < 0.6 options, primary entry signals
- **OTM Patterns**: For delta < 0.3 options, high-probability spread strategies
- **VOL_CRUSH Patterns**: For vega < -0.2 changes, volatility-based trades

### 2. Greek-Specific Patterns

- **Delta Patterns**: Directional momentum signals
- **Gamma Patterns**: Acceleration/deceleration indicators
- **Vega Patterns**: Volatility expansion/contraction signals
- **Theta Patterns**: Time decay optimization opportunities
- **Vanna/Charm Patterns**: Second-order effects for sophisticated strategies

## Example Patterns

### Bullish Pattern Example
- Pattern: (0, 1, 2, 3) - Rising sequence
- Description: "Rising → Rising → Rising"
- Historical Win Rate: 72%
- Average Return: 8.3%
- Recommended Strategy: LONG_CALL or BULL_CALL_SPREAD

### Volatility Crush Pattern Example
- Pattern: (3, 2, 1, 0) - Falling vega
- Description: "Falling → Falling → Falling"
- Historical Win Rate: 68%
- Average Return: 12.1%
- Recommended Strategy: SELL_STRADDLE or IRON_CONDOR

## Performance Metrics

The system tracks several key metrics:

1. **Pattern Accuracy**: Percentage of patterns followed by expected price movement
2. **Win Rate**: Percentage of trades that reach profit targets
3. **Average Return**: Mean return per pattern occurrence
4. **Risk-Adjusted Return**: Return relative to maximum drawdown
5. **Pattern Frequency**: How often each pattern occurs

## Integration with Existing Systems

The ordinal pattern system is designed to complement, not replace, your existing trading approach:

1. **Enhancement Layer**: Adds pattern-based confidence to existing signals
2. **Standalone Mode**: Can operate independently with its own signal generation
3. **Comparative Analysis**: Runs in parallel for performance comparison

## Best Practices

### Pattern Recognition
- Use at least 20 historical data points for reliable pattern identification
- Update pattern libraries regularly to capture changing market dynamics
- Focus on patterns with minimum 60% win rate and 5+ occurrences

### Signal Generation
- Combine multiple patterns for higher confidence signals
- Use cross-Greek patterns to confirm single-Greek patterns
- Set appropriate confidence thresholds for different trading styles

### Risk Management
- Always use calculated stop-loss levels
- Size positions based on pattern historical performance
- Monitor pattern degradation over time

## Configuration Options

```python
# Analyze specific pattern categories
patterns = analyzer.extract_patterns(data, moneyness_filter='ATM')

# Customize pattern requirements
analyzer = GreekOrdinalPatternAnalyzer(
    window_size=4,           # Pattern length
    min_occurrences=5,       # Minimum pattern frequency
    min_confidence=0.6,      # Minimum win rate
    top_patterns=3           # Number of patterns per category
)

# Configure signal generation
signal_generator = TradeSignalGenerator(
    pattern_analyzer=analyzer,
    entry_confidence_threshold=0.7,
    exit_confidence_threshold=0.6
)
```

## Troubleshooting

### Common Issues

1. **Insufficient Historical Data**
   - Ensure at least 60 days of data for reliable patterns
   - Check for missing or corrupted Greek values

2. **Low Pattern Recognition**
   - Verify Greek normalization is working correctly
   - Adjust window_size and min_occurrences parameters

3. **Signal Conflicts**
   - Use cross-Greek patterns to resolve conflicting signals
   - Prioritize signals with higher confidence levels

### Debugging Tools

```python
# Check pattern library contents
analyzer.save_pattern_library('debug_library.json')

# Validate pattern recognition
recognized = analyzer.recognize_current_patterns(current_data)

# Inspect signal generation
signals = signal_generator.generate_signals(data, patterns)
```

## Next Steps

1. **Backtesting**: Run historical simulations to validate pattern performance
2. **Paper Trading**: Test signals in a paper trading environment
3. **Parameter Optimization**: Fine-tune pattern detection parameters
4. **Live Integration**: Gradually integrate with live trading systems

## Support

For issues or questions about the ordinal pattern trading system:
1. Check the test files for usage examples
2. Review the complete_example.py for comprehensive demonstrations
3. Consult the inline documentation in each module

Remember: This system is designed to be a powerful addition to your trading toolkit, but all trades should be validated with your existing analysis and risk management procedures.