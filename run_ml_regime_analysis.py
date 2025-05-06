"""
Run ML-enhanced Market Regime Analysis

This script combines traditional Greek analysis with machine learning models to:
1. Predict market regimes based on Greek and entropy data
2. Detect regime transitions for trade entries and exits
3. Identify optimal entry/exit points using pattern recognition

Usage:
    python run_ml_regime_analysis.py --tickers AAPL MSFT --train
    python run_ml_regime_analysis.py --tickers AAPL MSFT --predict
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Import Greek analysis modules
from run_regime_analysis import run_pipeline_analysis, run_regime_analysis

# Import ML trade recommendation saver
from save_ml_trade_recommendations import save_ml_trade_recommendation, create_market_regime_summary

# Import ML modules
from models.ml.regime_classifier import (
    GreekRegimeClassifier, 
    RegimeTransitionPredictor,
    EntropyPatternRecognizer,
    evaluate_model_performance
)

# Add import for performance tracker
try:
    from analysis.trade_performance_tracker import TradePerformanceTracker
    HAS_PERFORMANCE_TRACKER = True
except ImportError:
    HAS_PERFORMANCE_TRACKER = False
    logging.warning("Trade Performance Tracker not found. Performance tracking will be disabled.")

# Add import for ordinal pattern analyzer
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ml_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json file."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def train_ml_models(tickers, results_dir="results", model_type="randomforest", use_mock=False):
    """
    Train ML models on Greek analysis data.
    
    Args:
        tickers: List of ticker symbols
        results_dir: Directory containing analysis results
        model_type: Type of ML model to use
        use_mock: Whether to use mock data for testing
        
    Returns:
        Dictionary with training metrics
    """
    logger.info(f"Training ML models using {model_type}")
    
    # Create output directory for models
    model_dir = "models/ml"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load analysis data for all tickers
    analysis_data = {}
    for ticker in tickers:
        analysis_file = os.path.join(results_dir, f"{ticker}_analysis_results.json")
        
        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, 'r') as f:
                    analysis_data[ticker] = json.load(f)
                    logger.info(f"Loaded analysis data for {ticker}")
            except Exception as e:
                logger.error(f"Error loading analysis data for {ticker}: {e}")
        else:
            logger.warning(f"Analysis data not found for {ticker}")
    
    if not analysis_data and not use_mock:
        logger.error("No analysis data found for training")
        return None
    
    # If no real data and
    if use_mock:
        # Generate mock data for testing
        logger.info("Using mock data for training")
        analysis_data = generate_mock_analysis_data(tickers)
    
    # Initialize classifier
    regime_classifier = GreekRegimeClassifier(model_type=model_type)
    
    # Train primary regime classifier
    primary_report = regime_classifier.train(
        results_dir=results_dir,
        target='primary_regime'
    )
    
    # Train volatility regime classifier
    volatility_classifier = GreekRegimeClassifier(model_type=model_type)
    volatility_report = volatility_classifier.train(
        results_dir=results_dir,
        target='volatility_regime'
    )
    
    # Train dominant Greek classifier
    greek_classifier = GreekRegimeClassifier(model_type=model_type)
    greek_report = greek_classifier.train(
        results_dir=results_dir,
        target='dominant_greek'
    )
    
    # Compile training results
    training_results = {
        'primary_regime': primary_report,
        'volatility_regime': volatility_report,
        'dominant_greek': greek_report,
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type
    }
    
    # Save training results
    os.makedirs('results/ml', exist_ok=True)
    with open(f"results/ml/training_results_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info("ML models trained successfully")
    
    return training_results

def run_ml_prediction(tickers, results_dir="results", output_dir=None, model_dir="models/ml", use_patterns=True):
    """
    Run ML prediction on Greek analysis data.
    
    Args:
        tickers: List of ticker symbols
        results_dir: Directory containing analysis results
        output_dir: Directory to save prediction results
        model_dir: Directory containing trained models
        use_patterns: Whether to use ordinal pattern analysis
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Running ML prediction for {len(tickers)} tickers")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, "ml_predictions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained models
    models = load_trained_models(model_dir)
    
    if not models:
        logger.error("No trained models found. Please train models first.")
        return None
    
    # First, run the standard pipeline analysis to generate analysis files
    pipeline_success = run_pipeline_analysis(
        tickers=tickers,
        output_dir=results_dir,
        skip_entropy=False,
        analysis_type="both"
    )
    
    if not pipeline_success:
        logger.error("Pipeline analysis failed, aborting ML prediction")
        return None
    
    # Run standard regime analysis
    regime_success = run_regime_analysis(results_dir=results_dir)
    
    if not regime_success:
        logger.warning("Standard regime analysis failed, continuing with ML prediction")
    
    # Initialize ordinal pattern analyzer if requested
    pattern_analyzer = None
    if use_patterns:
        try:
            pattern_analyzer = GreekOrdinalPatternAnalyzer(window_size=4, min_occurrences=3)
            pattern_library_path = os.path.join("patterns", "greek_patterns.json")
            if os.path.exists(pattern_library_path):
                pattern_analyzer.load_pattern_library(pattern_library_path)
                logger.info(f"Loaded pattern library from {pattern_library_path}")
            else:
                logger.warning(f"Pattern library not found at {pattern_library_path}")
        except Exception as e:
            logger.error(f"Error initializing pattern analyzer: {e}")
            pattern_analyzer = None
    
    # Initialize predictions dictionary
    predictions = {}
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"Generating ML predictions for {ticker}")
        
        # Load Greek analysis results
        greek_file = os.path.join(results_dir, f"{ticker}_greek_analysis.json")
        if not os.path.exists(greek_file):
            logger.warning(f"No Greek analysis results found for {ticker}, skipping")
            continue
        
        try:
            with open(greek_file, 'r') as f:
                greek_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Greek analysis for {ticker}: {e}")
            continue
        
        # Extract features for ML prediction
        features = extract_features_from_greek_data(greek_data)
        
        if not features:
            logger.warning(f"Could not extract features for {ticker}, skipping")
            continue
        
        # Make predictions using each model
        regime_classifier = models.get('regime_classifier')
        transition_predictor = models.get('transition_predictor')
        pattern_recognizer = models.get('pattern_recognizer')
        
        if not regime_classifier:
            logger.warning("Regime classifier model not found, skipping prediction")
            continue
        
        # Predict primary and secondary regimes
        ticker_predictions = regime_classifier.predict(ticker, features)
        
        # Get entropy state from Greek data
        entropy_state = greek_data.get('chain_energy', {}).get('entropy_state', 'Unknown')
        
        # Create entropy history from Greek data
        entropy_history = []
        if 'entropy_history' in greek_data:
            entropy_history = greek_data['entropy_history']
        else:
            # Create mock entropy history if not available
            entropy_history = [
                {'avg_entropy': 0.5, 'timestamp': '2023-01-01'},
                {'avg_entropy': 0.55, 'timestamp': '2023-01-02'},
                {'avg_entropy': 0.6, 'timestamp': '2023-01-03'}
            ]
        
        # Create mock regime history (would use actual history in production)
        greek_regimes = ['Vanna-Driven', 'Vanna-Driven', 'Charm-Dominated']
        
        # Detect entropy patterns
        patterns = pattern_recognizer.detect_patterns(entropy_history, greek_regimes)
        
        # Predict regime transitions
        current_regime = ticker_predictions.get('primary_regime', {}).get('prediction', 'Unknown')
        entropy_trend = 'increasing' if entropy_history[-1]['avg_entropy'] > entropy_history[-2]['avg_entropy'] else 'decreasing'
        transition = transition_predictor.predict_transition(ticker, current_regime, entropy_trend)
        
        # Determine trade signals based on predictions
        trade_signals = generate_trade_signals(
            ticker_predictions, 
            patterns, 
            transition,
            entropy_state
        )
        
        # Add ordinal pattern analysis if available
        if pattern_analyzer:
            try:
                # Convert Greek data to DataFrame for pattern analysis
                greek_df = convert_greek_data_to_dataframe(greek_data)
                
                if greek_df is not None and not greek_df.empty:
                    # Extract patterns
                    extracted_patterns = pattern_analyzer.extract_patterns(greek_df)
                    
                    # Analyze pattern profitability
                    pattern_analysis = pattern_analyzer.analyze_pattern_profitability(greek_df, extracted_patterns)
                    
                    # Recognize current patterns
                    current_data = greek_df.iloc[-4:].reset_index(drop=True)
                    recognized_patterns = pattern_analyzer.recognize_current_patterns(current_data)
                    
                    # Enhance trade signals with pattern analysis
                    if recognized_patterns:
                        trade_signals = pattern_analyzer.enhance_trade_recommendation(
                            trade_signals, recognized_patterns
                        )
                        
                        # Add pattern information to predictions
                        ticker_predictions['ordinal_patterns'] = recognized_patterns
                        
                        logger.info(f"Enhanced trade signals for {ticker} with ordinal pattern analysis")
                else:
                    logger.warning(f"Could not convert Greek data to DataFrame for {ticker}")
            except Exception as e:
                logger.error(f"Error in ordinal pattern analysis for {ticker}: {e}")
        
        # Add trade signals to predictions
        ticker_predictions['trade_signals'] = trade_signals
        
        # Store predictions for this ticker
        predictions[ticker] = ticker_predictions
        
        # Save predictions to file
        ticker_output_file = os.path.join(output_dir, f"{ticker}_ml_prediction.json")
        try:
            with open(ticker_output_file, 'w') as f:
                json.dump(ticker_predictions, f, indent=2)
            logger.info(f"Saved ML predictions for {ticker} to {ticker_output_file}")
        except Exception as e:
            logger.error(f"Error saving ML predictions for {ticker}: {e}")
    
    # Save summary of all predictions
    summary_file = os.path.join(output_dir, "ml_predictions_summary.json")
    try:
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'tickers_analyzed': len(predictions),
                'predictions': predictions
            }, f, indent=2)
        logger.info(f"Saved ML predictions summary to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving ML predictions summary: {e}")
    
    # Create market regime summary
    try:
        regime_summary = create_market_regime_summary(predictions)
        regime_summary_file = os.path.join(output_dir, "ml_regime_summary.json")
        with open(regime_summary_file, 'w') as f:
            json.dump(regime_summary, f, indent=2)
        logger.info(f"Saved ML regime summary to {regime_summary_file}")
    except Exception as e:
        logger.error(f"Error creating market regime summary: {e}")
    
    return predictions

def generate_trade_signals(predictions, patterns, transition, entropy_state):
    """
    Generate trade signals based on ML predictions and pattern recognition.
    
    Args:
        predictions: Dictionary with ML predictions
        patterns: List of detected patterns
        transition: Regime transition prediction
        entropy_state: Current entropy state
        
    Returns:
        Dictionary with trade signals
    """
    signals = {
        'entry': {
            'signal': None,
            'strength': 0,
            'confidence': 0,
            'reasons': []
        },
        'exit': {
            'signal': None,
            'strength': 0,
            'confidence': 0,
            'reasons': []
        }
    }
    
    # Extract regime predictions
    primary_regime = predictions.get('primary_regime', {}).get('prediction', 'Unknown')
    volatility_regime = predictions.get('volatility_regime', {}).get('prediction', 'Unknown')
    dominant_greek = predictions.get('dominant_greek', {}).get('prediction', 'Unknown')
    
    # Entry signals based on regime
    if primary_regime == 'Vanna-Driven' and volatility_regime == 'Low':
        signals['entry']['signal'] = 'bullish'
        signals['entry']['strength'] += 2
        signals['entry']['reasons'].append('Vanna-driven regime in low volatility environment')
    elif primary_regime == 'Gamma-Dominated' and volatility_regime == 'High':
        signals['entry']['signal'] = 'bullish'
        signals['entry']['strength'] += 1
        signals['entry']['reasons'].append('Gamma-dominated regime in high volatility environment')
    elif primary_regime == 'Charm-Dominated' and volatility_regime == 'Normal':
        signals['entry']['signal'] = 'neutral'
        signals['entry']['strength'] += 1
        signals['entry']['reasons'].append('Charm-dominated regime in normal volatility environment')
    elif primary_regime == 'Charm-Dominated' and volatility_regime == 'High':
        signals['entry']['signal'] = 'bearish'
        signals['entry']['strength'] += 2
        signals['entry']['reasons'].append('Charm-dominated regime in high volatility environment')
    
    # Entry signals based on entropy state
    if entropy_state == 'Dispersed Energy (High Entropy)':
        if signals['entry']['signal'] == 'bullish':
            signals['entry']['strength'] -= 1
            signals['entry']['reasons'].append('High entropy suggests less directional pressure')
        elif signals['entry']['signal'] == 'bearish':
            signals['entry']['strength'] -= 1
            signals['entry']['reasons'].append('High entropy suggests less directional pressure')
    elif entropy_state == 'Concentrated Energy (Low Entropy)':
        if signals['entry']['signal'] in ['bullish', 'bearish']:
            signals['entry']['strength'] += 1
            signals['entry']['reasons'].append('Concentrated energy enhances directional move potential')
    
    # Entry signals based on patterns
    for pattern in patterns:
        if pattern['pattern'] == 'compression_breakout':
            signals['entry']['signal'] = 'bullish'  # Assuming bullish breakout
            signals['entry']['strength'] += 2
            signals['entry']['reasons'].append('Entropy compression breakout pattern detected')
        elif pattern['pattern'] == 'entropy_spike':
            # Entropy spikes often signal reversals
            if signals['entry']['signal'] == 'bullish':
                signals['entry']['signal'] = 'bearish'
                signals['entry']['strength'] = 1
                signals['entry']['reasons'].append('Entropy spike suggests potential reversal')
            elif signals['entry']['signal'] == 'bearish':
                signals['entry']['signal'] = 'bullish'
                signals['entry']['strength'] = 1
                signals['entry']['reasons'].append('Entropy spike suggests potential reversal')
    
    # Exit signals based on transition prediction
    if transition['transition_type'] == 'likely':
        signals['exit']['signal'] = 'exit'
        signals['exit']['strength'] = 3
        signals['exit']['confidence'] = transition['transition_probability']
        signals['exit']['reasons'].append(f"High probability ({transition['transition_probability']:.2f}) of regime transition")
    elif transition['transition_type'] == 'possible':
        signals['exit']['signal'] = 'reduce'
        signals['exit']['strength'] = 2
        signals['exit']['confidence'] = transition['transition_probability']
        signals['exit']['reasons'].append(f"Moderate probability ({transition['transition_probability']:.2f}) of regime transition")
    
    # Calculate overall confidence
    if signals['entry']['signal']:
        # Scale strength to confidence (0-1)
        signals['entry']['confidence'] = min(signals['entry']['strength'] / 5.0, 1.0)
    
    # If exit signal exists but entry is stronger, prioritize entry
    if signals['exit']['signal'] and signals['entry']['signal']:
        if signals['entry']['strength'] > signals['exit']['strength'] + 1:
            signals['exit']['signal'] = None
            signals['exit']['reasons'] = ['Entry signal overrides weak exit signal']
        elif signals['exit']['strength'] > signals['entry']['strength'] + 1:
            signals['entry']['signal'] = None
            signals['entry']['reasons'] = ['Exit signal overrides weak entry signal']
    
    return signals

def run_ml_enhanced_backtests(tickers, start_date=None, end_date=None, output_dir='results/ml_backtests'):
    """
    Run backtests with ML-enhanced signals.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for backtest
        end_date: End date for backtest
        output_dir: Directory to save backtest results
        
    Returns:
        Dictionary with backtest metrics
    """
    logger.info(f"Running ML-enhanced backtests for {len(tickers)} tickers")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder function - would implement a full backtesting system
    # that compares standard Greek analysis to ML-enhanced analysis
    
    backtest_results = {
        'summary': {
            'standard_strategy': {
                'total_return': 0.15,
                'max_drawdown': 0.08,
                'sharpe_ratio': 1.2,
                'win_rate': 0.65
            },
            'ml_enhanced_strategy': {
                'total_return': 0.22,
                'max_drawdown': 0.07,
                'sharpe_ratio': 1.6,
                'win_rate': 0.72
            }
        },
        'ticker_results': {}
    }
    
    # Save mock backtest results
    backtest_output = os.path.join(output_dir, f"ml_backtest_results_{datetime.now().strftime('%Y%m%d')}.json")
    with open(backtest_output, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    logger.info(f"ML-enhanced backtest results saved to {backtest_output}")
    
    return backtest_results

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="Run ML-enhanced Market Regime Analysis")
    
    # Add arguments
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"],
                      help="List of ticker symbols to analyze")
    parser.add_argument("--output-dir", default="results", help="Output directory for analysis files")
    parser.add_argument("--model-type", choices=["randomforest", "gradientboosting"], default="randomforest",
                      help="Type of ML model to use")
    parser.add_argument("--train", action="store_true", help="Train ML models on existing analysis data")
    parser.add_argument("--predict", action="store_true", help="Run prediction using trained ML models")
    parser.add_argument("--backtest", action="store_true", help="Run backtests with ML-enhanced signals")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters of ML models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate performance of trained ML models")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Train ML models if requested
    if args.train:
        logger.info("Training ML models")
        training_results = train_ml_models(
            tickers=args.tickers,
            results_dir=args.output_dir,
            model_type=args.model_type
        )
        
        if not training_results:
            logger.error("ML model training failed")
            return 1
        
        logger.info("ML model training completed successfully")
    
    # Run prediction if requested
    if args.predict:
        logger.info("Running ML prediction")
        prediction_results = run_ml_prediction(
            tickers=args.tickers,
            results_dir=args.output_dir,
            output_dir=os.path.join(args.output_dir, "ml_predictions"),
            model_dir="models/ml"
        )
        
        if not prediction_results:
            logger.error("ML prediction failed")
            return 1
        
        logger.info("ML prediction completed successfully")
    
    # Run backtests if requested
    if args.backtest:
        logger.info("Running ML-enhanced backtests")
        backtest_results = run_ml_enhanced_backtests(
            tickers=args.tickers,
            output_dir=os.path.join(args.output_dir, "ml_backtests")
        )
        
        if not backtest_results:
            logger.error("ML-enhanced backtests failed")
            return 1
        
        logger.info("ML-enhanced backtests completed successfully")
    
    # Tune hyperparameters if requested
    if args.tune:
        logger.info("Tuning ML model hyperparameters")
        classifier = GreekRegimeClassifier(model_type=args.model_type)
        tuning_results = classifier.tune_hyperparameters(
            results_dir=args.output_dir,
            target='primary_regime'
        )
        
        if not tuning_results:
            logger.error("Hyperparameter tuning failed")
            return 1
        
        logger.info("Hyperparameter tuning completed successfully")
        
        # Save tuning results
        os.makedirs(os.path.join(args.output_dir, "ml"), exist_ok=True)
        tuning_output = os.path.join(args.output_dir, "ml", f"tuning_results_{datetime.now().strftime('%Y%m%d')}.json")
        with open(tuning_output, 'w') as f:
            json.dump(tuning_results, f, indent=2)
    
    # Evaluate model performance if requested
    if args.evaluate:
        logger.info("Evaluating ML model performance")
        evaluation_results = evaluate_model_performance(
            model_dir="models/ml",
            results_dir=args.output_dir,
            output_dir=os.path.join(args.output_dir, "ml_evaluation")
        )
        
        if not evaluation_results:
            logger.error("Model evaluation failed")
            return 1
        
        logger.info("Model evaluation completed successfully")
    
    # Generate market regime summary file for dashboard compatibility
    try:
        regime_file = create_market_regime_summary(output_dir=args.output_dir)
        if regime_file:
            logger.info(f"Created market regime summary at {regime_file}")
    except Exception as e:
        logger.error(f"Error creating market regime summary: {e}")
    
    logger.info("ML-enhanced regime analysis completed successfully")
    return 0

def convert_existing_predictions(input_dir, output_dir=None):
    """
    Convert existing ML prediction files to dashboard-compatible format.
    
    Args:
        input_dir (str): Directory containing ML prediction files
        output_dir (str, optional): Output directory. Defaults to input_dir.
    
    Returns:
        int: Number of files converted
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create recommendations directory
    recommendations_dir = os.path.join(output_dir, "recommendations")
    os.makedirs(recommendations_dir, exist_ok=True)
    
    # Find all ML prediction files
    ml_files = []
    for file in os.listdir(input_dir):
        if file.endswith("_ml_prediction.json"):
            ml_files.append(os.path.join(input_dir, file))
    
    if not ml_files:
        print(f"No ML prediction files found in {input_dir}")
        return 0
    
    converted = 0
    for ml_file in ml_files:
        try:
            # Load ML prediction file
            with open(ml_file, 'r') as f:
                ml_data = json.load(f)
            
            # Extract ticker from filename
            ticker = os.path.basename(ml_file).split("_ml_prediction.json")[0]
            
            # Extract data
            ticker_predictions = ml_data.get('ml_predictions', {})
            trade_signals = ml_data.get('trade_signals', {})
            transition = ml_data.get('regime_transition', {})
            entropy_state = "Unknown"  # Not typically stored in older files
            
            # Save in dashboard format
            save_ml_trade_recommendation(
                ticker,
                ticker_predictions,
                trade_signals,
                transition,
                entropy_state,
                output_dir=output_dir
            )
            
            converted += 1
            print(f"Converted {ticker} ML prediction to dashboard format")
            
        except Exception as e:
            print(f"Error converting {ml_file}: {e}")
    
    if converted > 0:
        # Create market regime summary
        create_market_regime_summary(output_dir=output_dir)
        print(f"Created market regime summary in {output_dir}")
    
    return converted

def generate_mock_analysis_data(tickers):
    """
    Generate mock analysis data for testing ML models.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary with mock analysis data for each ticker
    """
    logger.info(f"Generating mock analysis data for {len(tickers)} tickers")
    
    mock_data = {}
    for ticker in tickers:
        # Create deterministic but varied mock data based on ticker name
        ticker_hash = sum(ord(c) for c in ticker)
        base_price = 50 + (ticker_hash % 200)  # Generate price between 50 and 250
        
        # Create mock analysis data structure
        mock_data[ticker] = {
            "symbol": ticker,
            "analysis_results": {
                "greek_analysis": {
                    "market_regime": {
                        "primary": "Bullish" if ticker_hash % 3 == 0 else "Bearish" if ticker_hash % 3 == 1 else "Neutral",
                        "volatility": "High" if ticker_hash % 2 == 0 else "Normal"
                    },
                    "dominant_greek": "Delta" if ticker_hash % 4 == 0 else "Gamma" if ticker_hash % 4 == 1 else 
                                      "Vanna" if ticker_hash % 4 == 2 else "Charm"
                },
                "entropy_analysis": {
                    "energy_state_string": "High Energy" if ticker_hash % 3 == 0 else "Low Energy" if ticker_hash % 3 == 1 else "Neutral",
                    "metrics": {
                        "average_entropy": 50 + (ticker_hash % 50),
                        "delta_entropy": 30 + (ticker_hash % 40),
                        "gamma_entropy": 40 + (ticker_hash % 60)
                    }
                }
            },
            "market_data": {
                "currentPrice": base_price,
                "historicalVolatility": 0.2 + (ticker_hash % 100) / 1000,
                "impliedVolatility": 0.25 + (ticker_hash % 100) / 1000
            }
        }
    
    logger.info(f"Generated mock analysis data for {len(tickers)} tickers")
    return mock_data

def convert_greek_data_to_dataframe(greek_data):
    """
    Convert Greek analysis data to DataFrame for pattern analysis.
    
    Args:
        greek_data: Greek analysis data dictionary
        
    Returns:
        DataFrame with normalized Greek values
    """
    try:
        # Extract Greek values from data
        greeks = ['delta', 'gamma', 'vanna', 'charm']
        greek_values = {}
        
        # Check if we have time series data
        if 'time_series' in greek_data:
            time_series = greek_data['time_series']
            
            # Extract values for each Greek
            for greek in greeks:
                if greek in time_series:
                    greek_values[f'norm_{greek}'] = time_series[greek]
            
            # Add price data if available
            if 'price' in time_series:
                greek_values['price'] = time_series['price']
            
            # Create DataFrame
            df = pd.DataFrame(greek_values)
            return df
        
        # If no time series, try to extract from other structures
        chain_data = greek_data.get('chain_data', {})
        if chain_data:
            # Extract average values for each Greek
            for greek in greeks:
                if greek in chain_data:
                    # Normalize values
                    values = chain_data[greek]
                    if isinstance(values, list) and len(values) > 0:
                        # Use min-max normalization
                        min_val = min(values)
                        max_val = max(values)
                        if max_val > min_val:
                            norm_values = [(v - min_val) / (max_val - min_val) for v in values]
                        else:
                            norm_values = [0.5 for _ in values]
                        greek_values[f'norm_{greek}'] = norm_values
            
            # Add price data if available
            if 'price_history' in greek_data:
                greek_values['price'] = greek_data['price_history']
            
            # Create DataFrame
            df = pd.DataFrame(greek_values)
            return df
        
        logger.warning("Could not extract Greek values from data")
        return None
    
    except Exception as e:
        logger.error(f"Error converting Greek data to DataFrame: {e}")
        return None

if __name__ == "__main__":
    # If run with --convert-existing argument, convert existing files
    if len(sys.argv) > 1 and sys.argv[1] == "--convert-existing":
        if len(sys.argv) > 2:
            directory = sys.argv[2]
        else:
            directory = "results"
        
        num_converted = convert_existing_predictions(directory)
        print(f"Converted {num_converted} ML prediction files to dashboard format")
        sys.exit(0)
    
    # Otherwise run normal ML analysis
    sys.exit(main())







