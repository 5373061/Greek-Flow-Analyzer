# Greek Energy Flow II - Pattern Analysis Testing

This document provides information on how to run the test scripts for the Greek Energy Flow II pattern analysis integration, including entry/exit signal generation.

## Available Test Scripts

1. **test_ordinal_pattern_analyzer.py**: Unit tests for the `GreekOrdinalPatternAnalyzer` class and `PatternMetrics` utility.
2. **test_cross_greek_patterns.py**: Unit tests for the `CrossGreekPatternAnalyzer` class that analyzes relationships between different Greek metrics.
3. **test_trade_signals.py**: Unit tests for the `TradeSignalGenerator` class that generates entry/exit signals based on patterns.
4. **test_pattern_integration.py**: Integration tests for the pattern analyzer integration with the pipeline.
5. **run_tests.py**: Script to run all tests and generate a report.

## Running the Tests

### Run All Tests

To run all tests at once and generate a detailed report:

```bash
python run_tests.py
```

This will:
- Run all test modules (`test_*.py`)
- Display results in the console
- Save a test report to the `logs` directory
- Filter out common warnings by default

### Command-Line Options

The `run_tests.py` script supports several command-line options:

```bash
# Run with verbose output
python run_tests.py --verbose

# Run tests matching a specific pattern
python run_tests.py --pattern test_ordinal*

# Stop on first failure
python run_tests.py --failfast

# Show all warnings (don't filter)
python run_tests.py --show-warnings
```

You can combine these options as needed:

```bash
python run_tests.py --verbose --pattern test_pattern_integration.py --show-warnings
```

### Run Individual Test Modules

To run a specific test module:

```bash
python -m unittest test_ordinal_pattern_analyzer.py
python -m unittest test_pattern_integration.py
```

### Run Specific Test Cases

To run specific test cases:

```bash
python -m unittest test_ordinal_pattern_analyzer.TestGreekOrdinalPatternAnalyzer.test_extract_patterns
```

## Test Log Files

Test logs are stored in the `logs` directory with timestamps in the filename format:
- `test_report_YYYYMMDD_HHMMSS.log`

The log files contain all test output, including any warnings that might be filtered from the console output.

## Common Warnings

You may see warnings like:

```
Insufficient data for pattern analysis after filtering (needed 3, got 0)
```

These warnings are expected in some test cases where we're testing the behavior with insufficient data. They are filtered from the console output by default but are still recorded in the log file. Use the `--show-warnings` option to see all warnings.

### Filtered Warning Types

The following warning types are filtered from console output by default:

1. "Insufficient data for pattern analysis" - Occurs when there's not enough data for pattern extraction
2. "No patterns found for [moneyness]" - Occurs when no patterns are found for a specific moneyness category
3. "Cannot filter for VOL_CRUSH" - Occurs when required columns for VOL_CRUSH filtering are missing
4. "Delta column missing" - Occurs when the delta column is missing for moneyness filtering

All warnings are still recorded in the log files for reference.

## What the Tests Cover

### Unit Tests (test_ordinal_pattern_analyzer.py)

- **Pattern Extraction**: Tests that ordinal patterns are correctly extracted from Greek metrics
- **Moneyness Filtering**: Tests filtering data based on ITM/ATM/OTM categories
- **Profitability Analysis**: Tests the analysis of pattern profitability
- **Pattern Library**: Tests building, saving, and loading pattern libraries
- **Pattern Recognition**: Tests recognizing patterns in current market data
- **Trade Enhancement**: Tests enhancing trade recommendations based on recognized patterns
- **Pattern Metrics**: Tests entropy and complexity calculations for patterns

### Cross-Greek Pattern Tests (test_cross_greek_patterns.py)

- **Cross-Greek Analysis**: Tests analyzing relationships between patterns in different Greek metrics
- **Predictive Relationships**: Tests finding predictive relationships between Greeks
- **Cross-Greek Enhancement**: Tests enhancing trade recommendations with cross-Greek insights
- **Moneyness Determination**: Tests determining moneyness categories from trade recommendations

### Trade Signal Tests (test_trade_signals.py)

- **Signal Generation**: Tests entry and exit signal generation based on recognized patterns
- **Signal Types**: Tests identification of ENTRY_LONG, ENTRY_SHORT, EXIT_LONG, EXIT_SHORT signals
- **Strategy Recommendation**: Tests recommendation of appropriate options strategies based on signal type and confidence
- **Risk Management**: Tests calculation of take-profit and stop-loss levels
- **Position Management**: Tests handling of existing positions and generating appropriate signals

### Integration Tests (test_pattern_integration.py)

- **Pipeline Integration**: Tests integration with the mock pipeline manager
- **Data Extraction**: Tests extraction of Greek data from analysis results
- **Pattern Enhancement Flow**: Tests the full flow of pattern-based trade enhancement
- **Library Persistence**: Tests saving and loading pattern libraries during integration
- **Cross-Greek Integration**: Tests the integration of cross-Greek pattern analysis with the pipeline
- **Signal Integration**: Tests the integration of trade signal generation with the pipeline

## Adapting Tests for Real Pipeline

To adapt the integration tests for your real pipeline:

1. Replace `MockPipelineManager` with your actual pipeline manager class
2. Update the data extraction functions in `pattern_integration.py` to match your data structure
3. Adjust test expectations to match your pipeline's behavior

## Troubleshooting

If tests fail, check:

1. **Data Format Issues**: Ensure your data structure matches what's expected by the pattern analyzer
2. **Missing Dependencies**: Verify that numpy and pandas are installed
3. **File Permissions**: Check permissions for saving pattern libraries
4. **Path Issues**: Verify path configurations for pattern libraries

For detailed error information, check the test logs in the `logs` directory.

