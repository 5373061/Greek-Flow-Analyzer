"""
pattern_integration.py - Integration with Greek Energy Flow pipeline
"""

import pandas as pd
import logging
import os
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
from cross_greek_patterns import CrossGreekPatternAnalyzer
from trade_signals import TradeSignalGenerator

logger = logging.getLogger(__name__)

def integrate_with_pipeline(pipeline_manager, pattern_library_path="patterns", 
                          use_cross_greek=False, use_trade_signals=True):
    """
    Integrate the ordinal pattern analysis with the existing pipeline.
    
    Args:
        pipeline_manager: The existing pipeline manager instance
        pattern_library_path: Directory path for storing pattern libraries
        use_cross_greek: Whether to use cross-Greek pattern analysis
        use_trade_signals: Whether to generate trade signals
        
    Returns:
        Enhanced pipeline manager
    """
    # Create directory if it doesn't exist
    os.makedirs(pattern_library_path, exist_ok=True)
    
    # Check if pipeline_manager is None
    if pipeline_manager is None:
        logger.error("Pipeline manager is None. Cannot integrate pattern analysis.")
        return None
    
    # Check if process_symbol method exists
    if not hasattr(pipeline_manager, 'process_symbol'):
        logger.error("Pipeline manager does not have process_symbol method.")
        # Try to find an alternative method
        if hasattr(pipeline_manager, 'run') or hasattr(pipeline_manager, 'analyze'):
            logger.info("Found alternative method. Using it instead.")
            original_process_symbol = getattr(pipeline_manager, 'run' if hasattr(pipeline_manager, 'run') else 'analyze')
        else:
            logger.error("No suitable method found in pipeline manager.")
            return pipeline_manager
    else:
        # Store the original process_symbol method
        original_process_symbol = pipeline_manager.process_symbol
    
    # Initialize pattern analyzer and attach to pipeline manager
    pipeline_manager.pattern_analyzer = GreekOrdinalPatternAnalyzer()
    
    # Define the enhanced process_symbol method
    def process_symbol_with_patterns(symbol, options_data, price_data, current_price, **kwargs):
        # Run the original analysis
        results = original_process_symbol(symbol, options_data, price_data, current_price, **kwargs)
        
        try:
            # Extract Greek data for pattern analysis
            greek_data = extract_greek_data(results)
            
            if greek_data.empty:
                logger.warning(f"No Greek data available for pattern analysis for {symbol}")
                return results
            
            # Try to load existing pattern library
            library_path = f"{pattern_library_path}/{symbol}_patterns.json"
            try:
                pipeline_manager.pattern_analyzer.load_pattern_library(library_path)
                logger.info(f"Loaded existing pattern library for {symbol}")
            except FileNotFoundError:
                logger.info(f"Building new pattern library for {symbol}")
                # Extract patterns
                patterns = pipeline_manager.pattern_analyzer.extract_patterns(greek_data)
                
                # Analyze pattern profitability
                analysis = pipeline_manager.pattern_analyzer.analyze_pattern_profitability(greek_data, patterns)
                
                # Build pattern library for each moneyness category
                for moneyness in ['ITM', 'ATM', 'OTM', 'VOL_CRUSH']:
                    pipeline_manager.pattern_analyzer.build_pattern_library(analysis, moneyness)
                
                # Save the library
                pipeline_manager.pattern_analyzer.save_pattern_library(library_path)
            
            # Extract recent Greek data for pattern recognition
            recent_data = extract_recent_greek_data(results)
            
            # Recognize patterns in current data
            recognized_patterns = pipeline_manager.pattern_analyzer.recognize_current_patterns(recent_data)
            
            # Add pattern analysis to results
            if 'pattern_analysis' not in results:
                results['pattern_analysis'] = {}
            
            results['pattern_analysis']['recognized_patterns'] = recognized_patterns
            
            # Enhance trade recommendation if available
            if 'trade_recommendation' in results:
                # Get the current recommendation
                recommendation = results['trade_recommendation']
                
                # Enhance with pattern insights
                enhanced_recommendation = pipeline_manager.pattern_analyzer.enhance_recommendation_with_patterns(
                    recommendation, recognized_patterns
                )
                
                # Update the recommendation
                results['trade_recommendation'] = enhanced_recommendation
                results['trade_recommendation']['enhanced_by_patterns'] = True
            
            # Add cross-Greek pattern analysis if enabled
            if use_cross_greek and len(greek_data) >= 5:
                try:
                    # Initialize cross-Greek analyzer
                    cross_analyzer = CrossGreekPatternAnalyzer(pipeline_manager.pattern_analyzer)
                    
                    # Analyze cross-Greek patterns
                    cross_patterns = cross_analyzer.analyze_cross_greek_patterns(
                        greek_data, forward_period=3
                    )
                    
                    # Find predictive relationships
                    predictive = cross_analyzer.find_predictive_relationships(min_occurrences=2)
                    
                    # Add cross-Greek analysis to results
                    results['pattern_analysis']['cross_greek_patterns'] = predictive
                    
                    # Enhance recommendation with cross-Greek insights if available
                    if 'trade_recommendation' in results:
                        cross_enhanced = cross_analyzer.enhance_recommendation_with_cross_patterns(
                            results['trade_recommendation'], recent_data
                        )
                        results['trade_recommendation'] = cross_enhanced
                
                except Exception as e:
                    logger.error(f"Error in cross-Greek pattern analysis: {e}")
            
            # Generate trade signals if enabled
            if use_trade_signals:
                try:
                    # Initialize signal generator
                    signal_generator = TradeSignalGenerator(
                        pipeline_manager.pattern_analyzer,
                        cross_analyzer if use_cross_greek else None
                    )
                    
                    # Extract current position if available
                    current_position = None
                    if 'current_position' in results:
                        current_position = results['current_position']
                    
                    # Generate trade signals
                    signals = signal_generator.generate_signals(
                        recent_data, 
                        recognized_patterns,
                        current_position
                    )
                    
                    # Add signals to results
                    results['pattern_analysis']['trade_signals'] = signals
                    
                    # If there's no existing trade recommendation, create one from signals
                    if 'trade_recommendation' not in results and signals:
                        # Use the highest confidence signal as the primary recommendation
                        top_signal = signals[0]
                        results['trade_recommendation'] = {
                            'action': top_signal['signal_type'].value,
                            'confidence': top_signal['confidence'],
                            'strategy': top_signal['strategy'],
                            'target_strike': top_signal['target_strike'],
                            'take_profit': top_signal.get('take_profit'),
                            'stop_loss': top_signal.get('stop_loss'),
                            'expiration': top_signal.get('expiration'),
                            'signal_source': 'ordinal_pattern'
                        }
                        
                        logger.info(f"Created trade recommendation from pattern signal for {symbol}: "
                                   f"{top_signal['signal_type'].value} with confidence {top_signal['confidence']:.2f}")
                    
                    # Enhance existing recommendation with signal insights
                    elif 'trade_recommendation' in results and signals:
                        top_signal = signals[0]
                        recommendation = results['trade_recommendation']
                        
                        # Combine confidence levels
                        original_confidence = recommendation.get('confidence', 0.5)
                        new_confidence = 0.6 * original_confidence + 0.4 * top_signal['confidence']
                        recommendation['confidence'] = new_confidence
                        
                        # Add signal details
                        recommendation['supporting_signals'] = signals[:3]  # Top 3 signals
                        recommendation['signal_enhanced'] = True
                        
                        logger.info(f"Enhanced trade recommendation for {symbol} with pattern signals. "
                                   f"Confidence: {original_confidence:.2f} → {new_confidence:.2f}")
                
                except Exception as e:
                    logger.error(f"Error in trade signal generation: {e}")
            
        except Exception as e:
            logger.error(f"Error in pattern analysis for {symbol}: {e}")
        
        return results
    
    # Replace the process_symbol method
    pipeline_manager.process_symbol = process_symbol_with_patterns
    
    return pipeline_manager


def extract_greek_data(results):
    """
    Extract Greek data from analysis results.
    
    Args:
        results: Results from Greek Energy Flow analysis
        
    Returns:
        DataFrame with Greek data
    """
    # Create a DataFrame to store Greek data
    data = []
    
    # Check if historical data is available
    if 'historical_data' in results and results['historical_data']:
        # Extract data from each historical point
        for timestamp, point in results['historical_data'].items():
            if 'greeks' in point:
                greeks = point['greeks']
                
                # Create a data point
                data_point = {
                    'timestamp': timestamp,  # Add timestamp to data point
                    'price': point.get('price', 0)
                }
                
                # Add normalized Greek metrics
                for greek in ['delta', 'gamma', 'theta', 'vega', 'vanna', 'charm']:
                    norm_key = f'{greek}_normalized'
                    if norm_key in greeks:
                        data_point[f'norm_{greek}'] = greeks[norm_key]
                    elif greek in greeks:
                        # Use raw value if normalized not available
                        data_point[f'norm_{greek}'] = greeks[greek]
                    
                    # Add raw Greek metrics
                    if greek in greeks:
                        data_point[greek] = greeks[greek]
                
                # Add vega change for VOL_CRUSH detection
                if 'vega' in greeks and 'vega_prev' in point:
                    data_point['vega_prev'] = point['vega_prev']
                    data_point['vega_change'] = greeks['vega'] - point['vega_prev']
                
                data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure we have the minimum required columns
    required_columns = ['price', 'norm_delta', 'norm_gamma']
    if not all(col in df.columns for col in required_columns):
        logger.warning(f"Missing required columns in Greek data. Available: {df.columns}")
        return pd.DataFrame()
    
    return df


def extract_recent_greek_data(results, lookback_window=4):
    """
    Extract recent Greek data for pattern recognition.
    
    Args:
        results: Results from Greek Energy Flow analysis
        lookback_window: Number of recent data points to include
        
    Returns:
        DataFrame with recent Greek data
    """
    # Extract all Greek data
    all_data = extract_greek_data(results)
    
    # Return the most recent data points
    if len(all_data) > lookback_window:
        return all_data.tail(lookback_window).reset_index(drop=True)
    else:
        return all_data









