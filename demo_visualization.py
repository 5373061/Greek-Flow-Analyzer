import logging
from datetime import datetime
import pandas as pd
import numpy as np
from greek_flow.flow import BlackScholesModel, GreekEnergyFlow, GreekEnergyAnalyzer
from pipeline.data_pipeline import OptionsDataPipeline
from visualizations.energy_flow import create_energy_flow_chart

# Market-derived thresholds
THRESHOLD_CONFIG = {
    'volatility': {
        'low': 0.15,    # 15% IV considered low
        'high': 0.30    # 30% IV considered high
    },
    'gamma': {
        'significant': 0.02,  # 2% gamma contribution threshold
        'critical': 0.05     # 5% gamma concentration threshold
    },
    'delta': {
        'neutral_range': (-0.20, 0.20),  # ±20% considered neutral
        'exposure_limit': 1000  # Maximum acceptable delta exposure
    },
    'price_zones': {  # Price zones relative to current price
        'far_below': 0.85,    # <85% of current price
        'below': 0.95,        # 85-95% of current price
        'near': 1.05,         # 95-105% of current price
        'above': 1.15,        # 105-115% of current price
        'far_above': 1.15     # >115% of current price
    }
}

def calculate_expectancy(analysis_df: pd.DataFrame) -> dict:
    """Calculate expectancy metrics with market-derived thresholds"""
    current_price = analysis_df['underlying_price'].iloc[0]
    
    # Create dynamic price buckets relative to current price
    bucket_edges = [
        current_price * 0.85,  # Far below
        current_price * 0.95,  # Below
        current_price * 0.98,  # Near below
        current_price,         # At the money
        current_price * 1.02,  # Near above
        current_price * 1.05,  # Above
        current_price * 1.15   # Far above
    ]
    
    # Create labels for buckets
    labels = ['far_below', 'below', 'near_below', 'near_above', 'above', 'far_above']
    
    # Assign price buckets
    analysis_df['price_bucket'] = pd.cut(
        analysis_df['strike'],
        bins=bucket_edges,
        labels=labels,
        include_lowest=True
    )
    
    # Calculate composite scores
    composites = {}
    for bucket in analysis_df['price_bucket'].dropna().unique():
        bucket_data = analysis_df[analysis_df['price_bucket'] == bucket]
        
        # Calculate components
        gamma_energy = bucket_data['gamma_contribution'].sum()
        volume_weight = bucket_data['openInterest'].sum() / analysis_df['openInterest'].sum()
        
        composites[bucket] = {
            'score': gamma_energy * 0.6 + volume_weight * 0.4,
            'contracts': len(bucket_data),
            'avg_strike': bucket_data['strike'].mean(),
            'energy_concentration': gamma_energy
        }
    
    # Calculate expected move using updated column name
    avg_vol = analysis_df['impliedVolatility'].mean()  # Updated column name
    expected_move = current_price * avg_vol * np.sqrt(30/365)
    
    return {
        'composites': composites,
        'expected_move': expected_move,
        'expected_range': (current_price - expected_move, current_price + expected_move),
        'current_price': current_price
    }

def calculate_trade_metrics(aggregated_greeks: dict, flow_analyzer: GreekEnergyFlow, market_data: dict) -> dict:
    """Calculate trade metrics using aggregated Greeks"""
    results = {}
    
    # Get core metrics from aggregated Greeks
    net_delta = aggregated_greeks['net_delta']
    total_gamma = aggregated_greeks['total_gamma']
    total_vanna = aggregated_greeks['total_vanna']
    total_charm = aggregated_greeks['total_charm']
    current_price = market_data['currentPrice']
    
    # Calculate gamma energy (dollar impact of gamma per 1% move)
    gamma_energy = total_gamma * current_price * 0.01 # 1% move
    
    # Calculate delta exposure in dollars per 1% move
    delta_exposure = net_delta * current_price * 0.01
    
    # Calculate energy concentration (what percentage of total gamma is at current price)
    total_abs_gamma = sum(abs(g['gamma']) for g in aggregated_greeks['gamma_exposure'])
    if total_abs_gamma > 0:
        nearby_gamma = sum(
            abs(g['gamma']) for g in aggregated_greeks['gamma_exposure'] 
            if abs(g['strike'] - current_price) / current_price <= 0.01 # Within 1%
        )
        energy_concentration = (nearby_gamma / total_abs_gamma) * 100
    else:
        energy_concentration = 0.0
    
    # Normalize impact values
    vanna_impact = total_vanna * current_price * 0.01  # Impact per 1% IV change
    charm_decay = total_charm * current_price * (1/365)  # Daily theta impact on delta
    
    # Calculate expected moves
    implied_vol = market_data.get('impliedVolatility', 0.2)
    weekly_move = current_price * implied_vol * np.sqrt(7/365)
    monthly_move = current_price * implied_vol * np.sqrt(30/365)
    
    results = {
        'gamma_energy': gamma_energy,
        'delta_exposure': delta_exposure,
        'energy_concentration': energy_concentration,
        'vanna_impact': vanna_impact,
        'charm_decay': charm_decay,
        'weekly': weekly_move,
        'monthly': monthly_move,
        'current_price': current_price
    }
    
    return results

def format_trade_metrics(metrics: dict) -> dict:
    """Format the trade metrics for display"""
    return {
        'gamma_energy': f"${metrics['gamma_energy']:,.2f}",
        'delta_exposure': f"${metrics['delta_exposure']:,.2f} per 1% move",
        'energy_concentration': f"{metrics['energy_concentration']:.1f}%", 
        'vanna_impact': f"{metrics['vanna_impact']:.4f}",
        'charm_decay': f"{metrics['charm_decay']:.4f}"
    }

def generate_report(analysis_df: pd.DataFrame, market_data: dict) -> str:
    """Generate a simplified analysis report using available data"""
    
    # Get core metrics directly from data
    current_price = market_data['currentPrice']
    implied_vol = market_data['impliedVolatility']
    
    # Calculate basic metrics
    weekly_move = current_price * implied_vol * np.sqrt(7/365)
    monthly_move = current_price * implied_vol * np.sqrt(30/365)
    
    report = [
        "=" * 80,
        "MARKET ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Price: ${current_price:.2f}",
        f"Implied Volatility: {implied_vol:.1%}",
        "",
        "EXPECTED MOVES:",
        f"Weekly: ±${weekly_move:.2f}",
        f"Monthly: ±${monthly_move:.2f}",
        "",
        "=" * 80
    ]
    
    return "\n".join(report)

def format_currency(value: float) -> str:
    """Format large numbers as currency with K/M/B suffixes"""
    if abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    return f"${value:.2f}"

def main():
    try:
        # Import existing implementations
        from config import get_config, DEFAULT_CONFIG
        from config import POLYGON_API_KEY
        from Greek_Energy_FlowII import BlackScholesModel
        
        # Add API key to config
        ANALYSIS_CONFIG = DEFAULT_CONFIG.copy()
        ANALYSIS_CONFIG['POLYGON_API_KEY'] = POLYGON_API_KEY
        
        # Initialize components once at the start
        flow_analyzer = GreekEnergyFlow(config=ANALYSIS_CONFIG)

        # Debug logging to verify loaded components
        logging.info("\nLoaded Configuration:")
        logging.info("-" * 60)
        logging.info(f"API Key Configured: {POLYGON_API_KEY[:8]}...")
        
        # Fetch and prepare data
        symbol = "SPY"
        pipeline = OptionsDataPipeline(config=ANALYSIS_CONFIG)
        analysis_df = pipeline.prepare_analysis_data(symbol)
        
        if analysis_df is None:
            logging.error("Failed to prepare analysis data")
            return
            
        # Calculate days to expiry using expiration column
        analysis_df['dte'] = (analysis_df['expiration'] - pd.Timestamp.now()).dt.days
        
        # Calculate Greeks with better error handling and validation
        greeks_list = []
        bsm = BlackScholesModel()
        
        logging.info("\nCalculating Greeks...")
        for idx, row in analysis_df.iterrows():
            try:
                # Validate inputs with debugging info
                logging.debug(f"Processing strike {row['strike']}: S={row['underlying_price']}, K={row['strike']}, T={row['dte']}/365, sigma={row['implied_volatility']}, type={row['type']}")
                
                # Ensure proper numeric types with validation
                inputs = {
                    'S': max(float(row['underlying_price']), 0.01),
                    'K': max(float(row['strike']), 0.01),
                    'T': max(float(row['dte'])/365, 1e-6),
                    'r': 0.05,  # Risk-free rate
                    'sigma': max(float(row['implied_volatility']), 0.01),
                    'option_type': row['type'].lower()
                }
                
                # Calculate Greeks
                greeks = bsm.calculate(**inputs)
                greeks['row_idx'] = idx
                greeks['strike'] = row['strike']
                greeks_list.append(greeks)
                
            except Exception as e:
                logging.error(f"Failed Greek calculation for strike {row['strike']}: {str(e)}")
                continue
        
        if not greeks_list:
            logging.error("No valid Greek calculations - check input data")
            return
        
        # Create Greeks DataFrame
        greeks_df = pd.DataFrame(greeks_list)
        greeks_df.set_index('row_idx', inplace=True)
        
        # Merge with original data
        analysis_df = pd.concat([
            analysis_df,
            greeks_df.drop('strike', axis=1)
        ], axis=1)
        
        # Debug column names
        logging.info("\nOriginal Columns:")
        logging.info(f"{analysis_df.columns.tolist()}")
        
        # Get market data values
        current_price = float(analysis_df['underlying_price'].iloc[0])
        implied_vol = float(analysis_df['implied_volatility'].mean())
        historical_vol = implied_vol  # Use IV as historical if no other data available
        
        # Prepare proper market data structure
        market_data = {
            'currentPrice': current_price,
            'historicalVolatility': historical_vol,
            'impliedVolatility': implied_vol,
            'riskFreeRate': 0.05,  # Use same rate as in Greek calculations
        }
        
        # Run Greek energy flow analysis
        flow_results = flow_analyzer.analyze_greek_profiles(analysis_df, market_data)
        
        # Generate and print report
        report = generate_report(analysis_df, market_data)
        for line in report.split('\n'):
            logging.info(line)
            
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    main()
