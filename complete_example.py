"""
complete_example.py - Complete example demonstrating all pattern analysis features
including entry/exit signal generation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pattern_integration import integrate_with_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_realistic_pipeline():
    """
    Create a more realistic pipeline manager for demonstration.
    """
    class RealisticPipelineManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.current_positions = {}
            
        def process_symbol(self, symbol, options_data, price_data, current_price, **kwargs):
            """
            Process a symbol and return realistic analysis results.
            """
            self.logger.info(f"Processing symbol: {symbol}")
            
            # Generate realistic historical Greek data
            historical_data = self._generate_realistic_historical_data(symbol, current_price)
            
            # Mock current Greek analysis
            current_greeks = self._calculate_current_greeks(symbol, current_price)
            
            # Mock existing analysis results
            results = {
                'symbol': symbol,
                'market_data': {
                    'currentPrice': current_price,
                    'timestamp': datetime.now().isoformat()
                },
                'greek_analysis': {
                    'greek_profiles': {
                        'aggregated_greeks': current_greeks,
                        'atm_greeks': current_greeks,
                        'individual_greeks': current_greeks
                    }
                },
                'historical_data': historical_data,
                'current_position': self.current_positions.get(symbol),
                'volatility_analysis': {
                    'implied_volatility': 0.25 + np.random.normal(0, 0.05),
                    'volatility_skew': 0.15 + np.random.normal(0, 0.03)
                }
            }
            
            # Create a base trade recommendation if not already existing
            if 'trade_recommendation' not in results:
                results['trade_recommendation'] = {
                    'action': 'WAIT',
                    'strategy': 'MONITOR_FOR_CLEARER_SIGNALS',
                    'confidence': 0.5,
                    'option_selection': {
                        'atm_strike': round(current_price / 5) * 5,
                        'otm_strike': round(current_price / 5) * 5 + 5
                    }
                }
            
            return results
        
        def _generate_realistic_historical_data(self, symbol, current_price, days=20):
            """
            Generate realistic historical Greek data for the symbol.
            """
            historical = {}
            base_price = current_price * 0.9  # Start 10% below current price
            
            for i in range(days):
                date = datetime.now() - timedelta(days=days-i)
                timestamp = date.isoformat()
                
                # Generate price movement
                price = base_price * (1 + i * 0.005 + np.random.normal(0, 0.01))
                
                # Generate realistic Greek values based on price and time
                moneyness = price / current_price
                time_decay = (days - i) / 365.0  # Time to expiration in years
                
                # Calculate realistic Delta
                if moneyness > 1.05:  # OTM calls
                    delta = 0.2 + np.random.normal(0, 0.05)
                elif moneyness < 0.95:  # ITM calls  
                    delta = 0.7 + np.random.normal(0, 0.05)
                else:  # ATM calls
                    delta = 0.5 + np.random.normal(0, 0.1)
                
                # Generate other Greeks with realistic relationships
                gamma = (0.1 + np.random.normal(0, 0.02)) / moneyness  # Higher gamma at ATM
                theta = -(0.01 + np.random.normal(0, 0.002)) * time_decay  # Increases as expiration nears
                vega = (0.3 + np.random.normal(0, 0.05)) * time_decay  # Decreases as expiration nears
                
                # Add some patterns based on symbol for predictability
                if symbol == "AAPL":
                    # Bullish pattern for AAPL
                    delta += i * 0.01
                    gamma += i * 0.002
                elif symbol == "MSFT":
                    # Bearish pattern for MSFT
                    delta -= i * 0.01
                    gamma -= i * 0.002
                
                historical[timestamp] = {
                    'price': price,
                    'greeks': {
                        'delta_normalized': delta,
                        'gamma_normalized': gamma,
                        'theta_normalized': theta,
                        'vega_normalized': vega,
                        'vanna_normalized': gamma * vega * 0.1,
                        'charm_normalized': theta * delta * 0.1,
                        'delta': delta,
                        'gamma': gamma,
                        'theta': theta,
                        'vega': vega,
                        'vanna': gamma * vega * 0.1,
                        'charm': theta * delta * 0.1
                    }
                }
            
            return historical
        
        def _calculate_current_greeks(self, symbol, current_price):
            """
            Calculate current Greek values for the symbol.
            """
            # Base values
            delta = 0.5 + np.random.normal(0, 0.1)
            gamma = 0.08 + np.random.normal(0, 0.01)
            theta = -0.05 + np.random.normal(0, 0.01)
            vega = 0.25 + np.random.normal(0, 0.05)
            
            # Add symbol-specific bias
            if symbol == "AAPL":
                delta += 0.1  # More bullish
                gamma += 0.02
            elif symbol == "MSFT":
                delta -= 0.1  # More bearish
                gamma -= 0.02
            
            return {
                'net_delta_normalized': delta,
                'total_gamma_normalized': gamma,
                'total_theta_normalized': theta,
                'total_vega_normalized': vega,
                'total_vanna_normalized': gamma * vega * 0.1,
                'total_charm_normalized': theta * delta * 0.1,
                'net_delta': delta,
                'total_gamma': gamma,
                'total_theta': theta,
                'total_vega': vega,
                'total_vanna': gamma * vega * 0.1,
                'total_charm': theta * delta * 0.1
            }
    
    return RealisticPipelineManager()

def demonstrate_complete_workflow():
    """
    Demonstrate the complete workflow with pattern analysis and signal generation.
    """
    logger.info("Starting complete pattern analysis and signal generation demonstration")
    
    # Create a realistic pipeline
    pipeline = create_realistic_pipeline()
    
    # Integrate all pattern analysis features
    enhanced_pipeline = integrate_with_pipeline(
        pipeline,
        pattern_library_path="patterns",
        use_cross_greek=True,
        use_trade_signals=True
    )
    
    # Test with multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    current_prices = {
        "AAPL": 185.75,
        "MSFT": 332.40,
        "GOOGL": 142.80
    }
    
    # Process each symbol
    all_results = {}
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {symbol} (Current Price: ${current_prices[symbol]})")
        logger.info('='*50)
        
        # Process the symbol
        results = enhanced_pipeline.process_symbol(
            symbol=symbol,
            options_data=pd.DataFrame(),  # Would be real data in production
            price_data=pd.DataFrame(),     # Would be real data in production
            current_price=current_prices[symbol]
        )
        
        all_results[symbol] = results
        
        # Analyze the results
        analyze_results(symbol, results)
    
    # Compare results across symbols
    logger.info(f"\n{'='*50}")
    logger.info("Cross-Symbol Analysis")
    logger.info('='*50)
    
    compare_symbols(all_results)
    
    # Demonstrate position management
    logger.info(f"\n{'='*50}")
    logger.info("Position Management Demonstration")
    logger.info('='*50)
    
    demonstrate_position_management(enhanced_pipeline)

def analyze_results(symbol, results):
    """
    Analyze and display the results for a symbol.
    """
    # Check pattern analysis
    if 'pattern_analysis' in results:
        pattern_results = results['pattern_analysis']
        
        # Display recognized patterns
        if 'recognized_patterns' in pattern_results:
            logger.info("\nRecognized Patterns:")
            for moneyness, greek_patterns in pattern_results['recognized_patterns'].items():
                if greek_patterns:
                    logger.info(f"  {moneyness} patterns:")
                    for greek, pattern_info in greek_patterns.items():
                        stats = pattern_info.get('stats', {})
                        logger.info(f"    {greek}: {pattern_info['description']}")
                        logger.info(f"      Win rate: {stats.get('win_rate', 0):.1%}")
                        logger.info(f"      Avg return: {stats.get('avg_return', 0):.1%}")
        
        # Display cross-Greek patterns
        if 'cross_greek_patterns' in pattern_results:
            logger.info("\nCross-Greek Pattern Relationships:")
            for pair_key, relationships in pattern_results['cross_greek_patterns'].items():
                if relationships:
                    logger.info(f"  {pair_key}:")
                    for rel in relationships[:2]:  # Show top 2
                        logger.info(f"    {rel['source_description']} → {rel['target_description']}")
                        logger.info(f"      Occurrences: {rel['occurrences']}")
        
        # Display trade signals
        if 'trade_signals' in pattern_results:
            logger.info("\nGenerated Trade Signals:")
            for i, signal in enumerate(pattern_results['trade_signals'][:3], 1):  # Show top 3
                logger.info(f"  Signal {i}:")
                logger.info(f"    Type: {signal['signal_type'].value}")
                logger.info(f"    Confidence: {signal['confidence']:.1%}")
                logger.info(f"    Strategy: {signal['strategy']}")
                logger.info(f"    Target Strike: {signal['target_strike']}")
                if 'take_profit' in signal:
                    logger.info(f"    Take Profit: {signal['take_profit']:.1%}")
                if 'stop_loss' in signal:
                    logger.info(f"    Stop Loss: {signal['stop_loss']:.1%}")
    
    # Display final trade recommendation
    if 'trade_recommendation' in results:
        rec = results['trade_recommendation']
        logger.info("\nFinal Trade Recommendation:")
        logger.info(f"  Action: {rec.get('action', 'NONE')}")
        logger.info(f"  Confidence: {rec.get('confidence', 0):.1%}")
        logger.info(f"  Strategy: {rec.get('strategy', 'N/A')}")
        
        # Show enhancement history
        enhancements = []
        if rec.get('pattern_enhanced', False):
            enhancements.append("ordinal patterns")
        if rec.get('cross_pattern_enhanced', False):
            enhancements.append("cross-Greek patterns")
        if rec.get('signal_enhanced', False):
            enhancements.append("trade signals")
        
        if enhancements:
            logger.info(f"  Enhanced by: {', '.join(enhancements)}")

def compare_symbols(all_results):
    """
    Compare analysis results across symbols.
    """
    logger.info("\nSymbol Comparison:")
    
    # Create comparison table
    comparison_data = []
    
    for symbol, results in all_results.items():
        if 'trade_recommendation' in results:
            rec = results['trade_recommendation']
            
            # Count signals
            num_signals = 0
            if 'pattern_analysis' in results and 'trade_signals' in results['pattern_analysis']:
                num_signals = len(results['pattern_analysis']['trade_signals'])
            
            comparison_data.append({
                'Symbol': symbol,
                'Action': rec.get('action', 'WAIT'),
                'Confidence': rec.get('confidence', 0),
                'Strategy': rec.get('strategy', 'N/A'),
                'Signals': num_signals
            })
    
    # Display comparison
    df = pd.DataFrame(comparison_data)
    logger.info("\n" + df.to_string(index=False))
    
    # Find the highest confidence recommendation
    if len(df) > 0:
        best_rec = df.loc[df['Confidence'].idxmax()]
        logger.info(f"\nHighest confidence recommendation: {best_rec['Symbol']} - "
                   f"{best_rec['Action']} ({best_rec['Confidence']:.1%})")

def demonstrate_position_management(enhanced_pipeline):
    """
    Demonstrate position management with pattern-based signals.
    """
    logger.info("\nPosition Management Example:")
    
    # Create a mock position
    current_position = {
        'symbol': 'AAPL',
        'type': 'long',
        'strategy': 'BULL_CALL_SPREAD',
        'entry_date': datetime.now() - timedelta(days=10),
        'expiration': datetime.now() + timedelta(days=15),
        'entry_price': 5.50,
        'current_price': 6.20,
        'strike_long': 180,
        'strike_short': 185,
        'quantity': 10,
        'status': 'active',
        'current_date': datetime.now()
    }
    
    logger.info(f"Current Position: {current_position['strategy']} on {current_position['symbol']}")
    logger.info(f"  Entry: {current_position['entry_date'].strftime('%Y-%m-%d')}")
    logger.info(f"  Expiration: {current_position['expiration'].strftime('%Y-%m-%d')}")
    logger.info(f"  P&L: {((current_position['current_price'] - current_position['entry_price']) / current_position['entry_price']):.1%}")
    
    # Process the symbol with the existing position
    results = enhanced_pipeline.process_symbol(
        symbol=current_position['symbol'],
        options_data=pd.DataFrame(),
        price_data=pd.DataFrame(),
        current_price=182.50
    )
    
    # Add the position to results
    results['current_position'] = current_position
    
    # Check for exit signals
    if 'pattern_analysis' in results and 'trade_signals' in results['pattern_analysis']:
        signals = results['pattern_analysis']['trade_signals']
        exit_signals = [s for s in signals if 'EXIT' in s['signal_type'].value]
        
        if exit_signals:
            logger.info("\nExit Signals Generated:")
            for signal in exit_signals[:2]:  # Show top 2
                logger.info(f"  {signal['signal_type'].value}")
                logger.info(f"    Confidence: {signal['confidence']:.1%}")
                logger.info(f"    Reason: {signal.get('exit_reason', 'Pattern-based')}")
                if 'target_price' in signal:
                    logger.info(f"    Target: ${signal['target_price']}")
    
    # Check for adjustment signals
    if results.get('trade_recommendation', {}).get('signal_enhanced', False):
        supporting_signals = results['trade_recommendation'].get('supporting_signals', [])
        adjustment_signals = [s for s in supporting_signals if s.get('signal_type', '').startswith('ADJUST')]
        
        if adjustment_signals:
            logger.info("\nAdjustment Signals:")
            for signal in adjustment_signals:
                logger.info(f"  {signal.get('signal_type', 'ADJUST')}")
                logger.info(f"    Reason: {signal.get('reason', 'Pattern-based')}")

def main():
    """
    Main function to run the complete demonstration.
    """
    try:
        demonstrate_complete_workflow()
        logger.info("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()