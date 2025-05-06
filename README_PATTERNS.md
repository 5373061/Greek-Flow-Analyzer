# Greek Energy Flow II - Pattern Analysis

## Overview

The pattern analysis module in Greek Energy Flow II identifies and analyzes ordinal patterns in Greek metrics to enhance trading decisions. This document explains the key concepts and how to use the pattern analysis features.

## Ordinal Patterns

Ordinal patterns represent the relative ordering of values in a time series window. For example, in a window of 3 points, there are 6 possible ordinal patterns:
- (0,1,2): Steadily increasing
- (0,2,1): Up then down
- (1,0,2): Down then up
- (1,2,0): Up then down sharply
- (2,0,1): Down sharply then up
- (2,1,0): Steadily decreasing

These patterns capture the shape of the time series regardless of the absolute values, making them useful for identifying recurring patterns in Greek metrics.

## Key Components

### GreekOrdinalPatternAnalyzer

The main class for analyzing ordinal patterns in Greek metrics. Key methods:

- `extract_patterns()`: Extracts ordinal patterns from Greek data
- `analyze_pattern_profitability()`: Analyzes the profitability of each pattern
- `build_pattern_library()`: Builds a library of patterns with their statistics
- `recognize_current_patterns()`: Recognizes patterns in current market data
- `enhance_trade_recommendation()`: Enhances trade recommendations based on recognized patterns

### CrossGreekPatternAnalyzer

Analyzes relationships between patterns in different Greek metrics:

- `analyze_cross_greek_patterns()`: Analyzes relationships between patterns in different Greeks
- `find_predictive_relationships()`: Finds patterns in one Greek that predict patterns in another
- `enhance_recommendation_with_cross_patterns()`: Enhances trade recommendations with cross-Greek insights

### PatternMetrics

Utility class for calculating metrics on ordinal patterns:

- `calculate_pattern_entropy()`: Calculates the entropy of a sequence of patterns
- `calculate_pattern_complexity()`: Calculates the complexity of a sequence of patterns

## Usage Examples

### Basic Pattern Analysis

```python
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer

# Initialize analyzer
analyzer = GreekOrdinalPatternAnalyzer(window_size=3)

# Extract patterns
patterns = analyzer.extract_patterns(greek_data)

# Analyze pattern profitability
analysis = analyzer.analyze_pattern_profitability(greek_data, patterns)

# Build pattern library
library = analyzer.build_pattern_library(analysis, moneyness='ATM')

# Save library
analyzer.save_pattern_library('patterns/greek_patterns.json')

# Recognize patterns in current data
current_data = greek_data.iloc[-4:].reset_index(drop=True)
recognized = analyzer.recognize_current_patterns(current_data)
```

### Cross-Greek Pattern Analysis

```python
from cross_greek_patterns import CrossGreekPatternAnalyzer

# Initialize cross-Greek analyzer
cross_analyzer = CrossGreekPatternAnalyzer(analyzer)

# Analyze cross-Greek patterns
cross_patterns = cross_analyzer.analyze_cross_greek_patterns(greek_data, forward_period=3)

# Find predictive relationships
predictive = cross_analyzer.find_predictive_relationships(min_occurrences=2)

# Enhance recommendation with cross-Greek insights
enhanced = cross_analyzer.enhance_recommendation_with_cross_patterns(recommendation, recent_data)
```

### Integration with Pipeline

```python
from pattern_integration import integrate_with_pipeline

# Integrate pattern analysis with pipeline
enhanced_pipeline = integrate_with_pipeline(pipeline_manager, use_cross_greek=True)

# Process a symbol (now includes pattern analysis)
results = enhanced_pipeline.process_symbol(symbol, options_data, price_data, current_price)
```

## Pattern Visualization

The `PatternVisualizer` class provides visualization of Greek patterns:

```python
from visualization.pattern_visualizer import PatternVisualizer

# Initialize visualizer
visualizer = PatternVisualizer()

# Plot Greek patterns
visualizer.plot_greek_patterns(greek_data, patterns, save_path='patterns.png')
```

## Dashboard Integration

The pattern analysis module integrates with the dashboard to provide visual insights:

### Viewing Pattern Analysis in the Dashboard

1. Run the pattern analysis to generate pattern libraries:
   ```bash
   python update_pattern_library.py --results-dir results --pattern-dir patterns
   ```

2. Launch the dashboard with pattern integration:
   ```bash
   python fix_dashboard.py
   python -m tools.trade_dashboard
   ```
   
   Or use the batch file:
   ```bash
   fix_and_run_dashboard.bat
   ```

3. In the dashboard, select a symbol to view its pattern analysis:
   - The dashboard will display recognized patterns
   - Pattern profitability statistics
   - Cross-Greek pattern relationships
   - Pattern-based trade recommendations

### Pattern Analysis Features in Dashboard

- **Pattern Recognition**: View patterns recognized in recent Greek data
- **Pattern Statistics**: See win rates and expected values for each pattern
- **Cross-Greek Insights**: Understand relationships between patterns in different Greeks
- **Pattern-Enhanced Recommendations**: Get trade recommendations enhanced with pattern insights

## Updating Pattern Libraries

To keep pattern libraries up-to-date with the latest market data:

```bash
# Update pattern libraries for all symbols with analysis results
python update_pattern_library.py --results-dir results --pattern-dir patterns

# Update and integrate with pipeline
python update_pattern_library.py --results-dir results --pattern-dir patterns --skip-pipeline-integration
```

This will:
1. Extract patterns from recent Greek data
2. Analyze pattern profitability
3. Update pattern libraries for each symbol
4. Create a combined pattern library
5. Integrate with the analysis pipeline (unless skipped)

## Performance Considerations

- Pattern extraction is computationally intensive for large datasets
- Consider using a smaller window size (e.g., 3) for faster analysis
- For real-time applications, pre-compute pattern libraries and use them for recognition
- The `recognize_current_patterns()` method is optimized for real-time use

## Advanced Pattern Analysis

For advanced pattern analysis, consider:

1. **Pattern Entropy**: Measure the entropy of pattern distributions to identify market regime changes
2. **Pattern Complexity**: Calculate the complexity of pattern sequences to identify market structure changes
3. **Pattern Transitions**: Analyze transitions between patterns to identify market dynamics
4. **ML-Enhanced Pattern Recognition**: Use machine learning to improve pattern recognition accuracy

```python
from pattern_metrics import PatternMetrics

# Calculate pattern entropy
entropy = PatternMetrics.calculate_pattern_entropy(patterns)

# Calculate pattern complexity
complexity = PatternMetrics.calculate_pattern_complexity(patterns)

# Use ML for pattern recognition
from pattern_ml_integrator import PatternMLIntegrator
ml_integrator = PatternMLIntegrator()
enhanced_patterns = ml_integrator.enhance_pattern_recognition(patterns, greek_data)
```

## Troubleshooting

If you encounter issues with pattern analysis:

1. **Missing pattern libraries**: Run `update_pattern_library.py` to generate pattern libraries
2. **Pattern recognition errors**: Ensure Greek data has the required columns and sufficient history
3. **Dashboard integration issues**: Run `fix_dashboard.py` to fix dashboard integration
4. **Performance issues**: Reduce window size or use pre-computed pattern libraries

For more help, see the main documentation or open an issue on GitHub.

