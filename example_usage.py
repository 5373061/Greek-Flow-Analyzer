"""
example_usage.py - Example of how to use the ordinal pattern analyzer with Greek Energy Flow
"""

import logging
import pandas as pd
from pattern_integration import integrate_with_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("greek_energy_flow_patterns.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_mock_pipeline():
    """
    Create a mock pipeline manager for testing.
    
    In your actual code, you would use your real pipeline manager.
    """
    class MockPipelineManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            
        def process_symbol(self, symbol, options_data, price_data, current_price, **kwargs):
            """
            Mock implementation of process_symbol.
            
            Replace this with your actual implementation.
            """
            self.logger.info(f"Processing symbol: {symbol}")
            
            # Mock results
            results = {
                'symbol': symbol,
                'market_data': {
                    'currentPrice': current_price,
                    'timestamp': '2024-05-05 14:30:00'
                },
                'greek_analysis': {
                    'greek_profiles': {
                        'aggregated_greeks': {
                            'net_delta_normalized': 0.45,
                            'total_gamma_normalized': 0.32,
                            'total_theta_normalized': -0.18,
                            'total_vega_normalized': 0.28,
                            'total_charm_normalized': -0.12,
                            'total_vanna_normalized': 0.22
                        }
                    }
                },
                'trade_recommendation': {
                    'symbol': symbol,
                    'current_price': current_price,
                    'action': 'WAIT',
                    'strategy': 'MONITOR_FOR_CLEARER_SIGNALS',
                    'confidence': 0.6,
                    'option_selection': {
                        'atm_strike': round(current_price / 5) * 5,  # Round to nearest 5
                        'otm_strike': round(current_price / 5) * 5 + 5
                    }
                },
                'historical_data': generate_mock_historical_data(symbol, current_price)
            }
            
            return results
    
    return MockPipelineManager()

def generate_mock_historical_data(symbol, current_price):
    """
    Generate mock historical data for testing.
    
    In your actual code, you would use real historical data.
    """
    # Create 10 historical data points
    historical = {}
    price = current_price * 0.9  # Start 10% below current price
    
    for i in range(10):
        timestamp = f"2024-05-{i+1:02d} 14:30:00"
        
        # Create some variation in the metrics
        delta = 0.4 + i * 0.03
        gamma = 0.5 - i * 0.03
        vanna = 0.3 + (i % 3) * 0.05
        charm = 0.1 + (i % 5) * 0.03
        
        historical[timestamp] = {
            'price': price,
            'greeks': {
                'delta_normalized': delta,
                'gamma_normalized': gamma,
                'theta_normalized': -0.15 - (i % 3) * 0.02,
                'vega_normalized': 0.3 - (i % 4) * 0.02,
                'vanna_normalized': vanna,
                'charm_normalized': charm
            }
        }
        
        # Increase price for next data point
        price *= 1.01  # 1% increase
    
    return historical

def main():
    """
    Example of using the pattern integration.
    """
    logger.info("Starting Greek Energy Flow II with Ordinal Pattern Analysis")
    
    # Create mock pipeline (replace with your actual pipeline)
    pipeline = create_mock_pipeline()
    
    # Integrate pattern analysis with cross-Greek patterns enabled
    enhanced_pipeline = integrate_with_pipeline(pipeline, use_cross_greek=True)
    
    # Process a symbol (this should now include pattern analysis)
    symbol = "AAPL"
    options_data = pd.DataFrame()  # Replace with real options data
    price_data = pd.DataFrame()    # Replace with real price data
    current_price = 185.75         # Replace with real current price
    
    # Process the symbol
    results = enhanced_pipeline.process_symbol(
        symbol=symbol,
        options_data=options_data,
        price_data=price_data,
        current_price=current_price
    )
    
    # Check if pattern analysis was performed
    if 'pattern_analysis' in results:
        pattern_results = results['pattern_analysis']
        logger.info(f"Pattern analysis results for {symbol}:")
        
        # Check recognized patterns
        if 'recognized_patterns' in pattern_results:
            patterns = pattern_results['recognized_patterns']
            for moneyness, greek_patterns in patterns.items():
                if greek_patterns:
                    logger.info(f"  {moneyness} patterns:")
                    for greek, pattern_info in greek_patterns.items():
                        logger.info(f"    {greek}: {pattern_info['description']}")
                        logger.info(f"      Win rate: {pattern_info['stats'].get('win_rate', 0):.2f}")
        
        # Check cross-Greek patterns
        if 'cross_greek_patterns' in pattern_results:
            cross_patterns = pattern_results['cross_greek_patterns']
            logger.info(f"Cross-Greek pattern relationships for {symbol}:")
            
            for pair_key, relationships in cross_patterns.items():
                if relationships:
                    greek1, greek2 = pair_key.split('_')
                    logger.info(f"  {greek1} → {greek2} relationships:")
                    
                    for i, rel in enumerate(relationships[:3], 1):  # Show top 3
                        logger.info(f"    {i}. {rel['source_description']} → {rel['target_description']}")
                        logger.info(f"       Occurrences: {rel['occurrences']}")
    
    # Check if trade recommendation was enhanced
    if 'trade_recommendation' in results:
        recommendation = results['trade_recommendation']
        logger.info(f"Trade recommendation for {symbol}:")
        logger.info(f"  Action: {recommendation.get('action', 'NONE')}")
        logger.info(f"  Confidence: {recommendation.get('confidence', 0):.2f}")
        
        # Check for pattern enhancements
        if recommendation.get('pattern_enhanced', False):
            logger.info(f"  Recommendation was enhanced by ordinal pattern analysis")
            if recommendation.get('pattern_action_upgrade', False):
                logger.info(f"  Action was upgraded based on strong pattern signals")
        
        # Check for cross-Greek insights
        if recommendation.get('cross_pattern_enhanced', False):
            logger.info(f"  Recommendation was further enhanced by cross-Greek pattern analysis")
            
            if 'cross_greek_insights' in recommendation:
                logger.info(f"  Cross-Greek insights:")
                for insight in recommendation['cross_greek_insights']:
                    logger.info(f"    {insight['source_greek']} ({insight['source_pattern']}) → "
                              f"{insight['target_greek']} ({insight['target_pattern']})")
                    logger.info(f"      Confidence: {insight['confidence']:.2f}")
    
    # Process another symbol for comparison
    symbol2 = "MSFT"
    current_price2 = 332.40
    
    logger.info(f"\nProcessing {symbol2} for comparison...")
    
    results2 = enhanced_pipeline.process_symbol(
        symbol=symbol2,
        options_data=pd.DataFrame(),
        price_data=pd.DataFrame(),
        current_price=current_price2
    )
    
    # Compare trade recommendations
    if 'trade_recommendation' in results and 'trade_recommendation' in results2:
        rec1 = results['trade_recommendation']
        rec2 = results2['trade_recommendation']
        
        logger.info(f"Trade recommendation comparison:")
        logger.info(f"  {symbol}: {rec1.get('action', 'NONE')} with confidence {rec1.get('confidence', 0):.2f}")
        logger.info(f"  {symbol2}: {rec2.get('action', 'NONE')} with confidence {rec2.get('confidence', 0):.2f}")
    
    logger.info("Example completed")
    
if __name__ == "__main__":
    main()