# entropy_analyzer/integration_after_greeks.py
"""
Integration module for adding entropy analysis and advanced risk management
to the Greek Energy Flow analysis pipeline.
"""

from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
from entropy_analyzer.risk_manager import AdvancedRiskManager
import logging

logger = logging.getLogger("greek_flow.integration")

def run_entropy_analysis(options_df, symbol):
    """
    Run entropy analysis on options data.
    
    Args:
        options_df (pd.DataFrame): Options data
        symbol (str): Symbol being analyzed
        
    Returns:
        dict: Entropy analysis results
    """
    try:
        entropy_analyzer = EntropyAnalyzer(options_df)
        entropy_data = entropy_analyzer.analyze_greek_entropy()
        anomaly_data = entropy_analyzer.detect_anomalies()
        entropy_data.update({"anomalies": anomaly_data})
        logger.info(f"{symbol}: Entropy analysis completed")
        return entropy_data
    except Exception as e:
        logger.error(f"{symbol}: Entropy analysis failed: {e}")
        return {}

def run_advanced_risk_management(greek_data, entropy_data, spot_price, config):
    """
    Run advanced risk management analysis.
    
    Args:
        greek_data (dict): Greek analysis results
        entropy_data (dict): Entropy analysis results
        spot_price (float): Current spot price
        config (dict): Configuration settings
        
    Returns:
        dict: Advanced risk management plan
    """
    try:
        risk_manager = AdvancedRiskManager(
            greek_data=greek_data, 
            entropy_data=entropy_data,
            spot_price=spot_price,
            config=config
        )
        advanced_risk = risk_manager.generate_risk_management_plan()
        return advanced_risk
    except Exception as e:
        logger.error(f"Advanced risk management failed: {e}")
        return {}

def format_adaptive_exits(advanced_risk):
    """
    Format adaptive exit strategies as text.
    
    Args:
        advanced_risk (dict): Advanced risk management plan
        
    Returns:
        str: Formatted text of adaptive exit strategies
    """
    if "adaptive_exits" in advanced_risk and advanced_risk["adaptive_exits"]:
        exits_text = "\nADAPTIVE EXIT STRATEGIES:\n"
        for exit_strat in advanced_risk["adaptive_exits"]:
            exits_text += f"- {exit_strat['type']}: {exit_strat['condition']}\n  Reason: {exit_strat['reason']}\n"
        
        return exits_text
    else:
        return ""

def ensure_energy_state_string(entropy_data):
    """
    Ensure energy_state_string is available for trade recommendations.
    
    Args:
        entropy_data (dict): Entropy analysis results
        
    Returns:
        dict: Updated entropy data with energy_state_string
    """
    if "energy_state" in entropy_data and isinstance(entropy_data["energy_state"], dict) and "state" in entropy_data["energy_state"]:
        entropy_data["energy_state_string"] = entropy_data["energy_state"]["state"]
    return entropy_data

def integrate_with_analysis_pipeline(options_df, greek_data, spot_price, config, risk_metrics, symbol):
    """
    Integrate entropy analysis and advanced risk management with the main analysis pipeline.
    
    Args:
        options_df (pd.DataFrame): Options data
        greek_data (dict): Greek analysis results
        spot_price (float): Current spot price
        config (dict): Configuration settings
        risk_metrics (dict): Existing risk metrics
        symbol (str): Symbol being analyzed
        
    Returns:
        tuple: (entropy_data, updated_risk_metrics)
    """
    # Run entropy analysis
    entropy_data = run_entropy_analysis(options_df, symbol)
    
    # Run advanced risk management
    if greek_data and entropy_data and spot_price is not None:
        advanced_risk = run_advanced_risk_management(greek_data, entropy_data, spot_price, config)
        
        # Merge with basic risk metrics
        if risk_metrics is not None:
            risk_metrics.update(advanced_risk)
        else:
            risk_metrics = advanced_risk
        
        # Format adaptive exit strategies
        exits_text = format_adaptive_exits(advanced_risk)
        if exits_text:
            logger.info(f"{symbol}: Generated {len(advanced_risk['adaptive_exits'])} adaptive exit strategies")
    
    # Ensure energy_state_string is available
    entropy_data = ensure_energy_state_string(entropy_data)
    
    return entropy_data, risk_metrics

