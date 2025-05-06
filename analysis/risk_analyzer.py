# analysis/risk_analyzer.py
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_risk_metrics(greek_data, momentum_data, spot_price, symbol):
    """
    Calculate risk management metrics based on Greek and momentum data.
    
    Args:
        greek_data: Greek analysis results
        momentum_data: Momentum analysis results
        spot_price: Current spot price
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with risk metrics
    """
    try:
        # Default risk metrics
        risk_metrics = {
            "risk_reward_ratio": "N/A",
            "position_size": "N/A",
            "stop_loss": "N/A",
            "take_profit": "N/A",
            "volatility_risk": "Low"
        }        
        
        # Extract energy levels
        levels = greek_data.get("energy_levels", [])
        if not levels:
            return risk_metrics
            
        # Find closest support and resistance
        supports = []
        resistances = []
        for lvl in levels:
            try:
                price_str = str(lvl.get("price", "")).replace("$", "")
                price = float(price_str)
                if "support" in str(lvl.get("direction", "")).lower():
                    supports.append(price)
                elif "resistance" in str(lvl.get("direction", "")).lower():
                    resistances.append(price)
            except (ValueError, TypeError):
                continue
        
        if not supports and not resistances:
            return risk_metrics
            
        # Get closest levels
        closest_support = max([s for s in supports if s < spot_price], default=spot_price * 0.95)
        closest_resistance = min([r for r in resistances if r > spot_price], default=spot_price * 1.05)
        
        # Calculate potential risk and reward
        risk = abs(spot_price - closest_support)
        reward = abs(closest_resistance - spot_price)
        
        # Risk/reward ratio (want at least 1:2)
        ratio = reward / risk if risk > 0 else 0
        risk_metrics["risk_reward_ratio"] = f"{ratio:.2f}"
        
        # Stop loss and take profit
        risk_metrics["stop_loss"] = f"${closest_support:.2f}"
        risk_metrics["take_profit"] = f"${closest_resistance:.2f}"
        
        # Position sizing based on account risk (default values, can be overridden from config)
        account_value = 50000  # Default $50,000 if not set
        risk_percent = 0.02    # Default 2% risk if not set
        
        # Calculate account risk and position size
        account_risk = account_value * risk_percent
        shares = int(account_risk / risk) if risk > 0 else 0
        position_value = shares * spot_price
        risk_metrics["position_size"] = f"{shares} shares (${position_value:.2f})"
        
        # Volatility risk assessment
        regime = greek_data.get("market_regime", {})
        vol_regime = regime.get("volatility_regime", "Normal")
        if vol_regime:
            if "high" in str(vol_regime).lower():
                risk_metrics["volatility_risk"] = "High"
            elif "low" in str(vol_regime).lower():
                risk_metrics["volatility_risk"] = "Low"
            else:
                risk_metrics["volatility_risk"] = "Medium"
        
        return risk_metrics
    
    except Exception as e:
        logger.warning(f"{symbol}: Error calculating risk metrics: {e}")
        return {
            "risk_reward_ratio": "Error",
            "position_size": "N/A",
            "stop_loss": "N/A",
            "take_profit": "N/A",
            "volatility_risk": "Unknown"
        }

class RiskAnalyzer:
    """Class for analyzing risk and managing position sizing."""
    
    def __init__(self, account_value=50000, risk_percent=0.02):
        """
        Initialize RiskAnalyzer.
        
        Args:
            account_value (float): Total account value
            risk_percent (float): Risk percentage per trade (0.02 = 2%)
        """
        self.account_value = account_value
        self.risk_percent = risk_percent
        
    def calculate_risk_metrics(self, greek_data, momentum_data, spot_price, symbol):
        """
        Calculate risk management metrics.
        
        Args:
            greek_data (dict): Greek analysis results
            momentum_data (dict): Momentum analysis results
            spot_price (float): Current spot price
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Risk metrics
        """
        # Create base metrics
        metrics = calculate_risk_metrics(greek_data, momentum_data, spot_price, symbol)
        
        # Check if any metrics are using fallbacks
        if "N/A" in metrics.values() or "Error" in metrics.values() or "Unknown" in metrics.values():
            logger.warning(f"{symbol}: FALLBACK RISK METRICS USED - some risk calculations could not be completed with actual data")
        
        # Add more advanced metrics
        metrics.update(self._calculate_advanced_metrics(greek_data, momentum_data, spot_price))
        
        return metrics
        
    def _calculate_advanced_metrics(self, greek_data, momentum_data, spot_price):
        """Calculate advanced risk metrics based on Greeks and momentum."""
        advanced_metrics = {
            "gamma_exposure_risk": "Low",
            "vanna_risk": "Low",
            "charm_risk": "Low",
            "recommended_strategy": "None"
        }
        
        # Extract Greek magnitudes
        market_regime = greek_data.get("market_regime", {})
        magnitudes = market_regime.get("greek_magnitudes", {})
        
        # Calculate gamma exposure risk
        gamma = float(str(magnitudes.get("total_gamma", "0")).replace(',', ''))
        if abs(gamma) > 0.1:
            advanced_metrics["gamma_exposure_risk"] = "High"
        elif abs(gamma) > 0.05:
            advanced_metrics["gamma_exposure_risk"] = "Medium"
            
        # Calculate vanna risk
        vanna = float(str(magnitudes.get("total_vanna", "0")).replace(',', ''))
        if abs(vanna) > 0.1:
            advanced_metrics["vanna_risk"] = "High"
        elif abs(vanna) > 0.05:
            advanced_metrics["vanna_risk"] = "Medium"
            
        # Calculate charm risk
        charm = float(str(magnitudes.get("total_charm", "0")).replace(',', ''))
        if abs(charm) > 0.1:
            advanced_metrics["charm_risk"] = "High"
        elif abs(charm) > 0.05:
            advanced_metrics["charm_risk"] = "Medium"
            
        # Determine recommended strategy based on all factors
        energy_dir = str(momentum_data.get("energy_direction", "")).lower()
        
        if "positive" in energy_dir and gamma > 0:
            advanced_metrics["recommended_strategy"] = "Bullish - Consider Call Options"
        elif "negative" in energy_dir and gamma < 0:
            advanced_metrics["recommended_strategy"] = "Bearish - Consider Put Options"
        elif "positive" in energy_dir and gamma < 0:
            advanced_metrics["recommended_strategy"] = "Mixed Signals - Consider Call Spreads"
        elif "negative" in energy_dir and gamma > 0:
            advanced_metrics["recommended_strategy"] = "Mixed Signals - Consider Put Spreads"
        else:
            advanced_metrics["recommended_strategy"] = "Neutral - Consider Iron Condor or Calendar Spread"
            
        return advanced_metrics
        
    def calculate_position_size(self, entry_price, stop_price):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            entry_price (float): Entry price
            stop_price (float): Stop loss price
            
        Returns:
            dict: Position sizing information
        """
        if entry_price <= 0 or stop_price <= 0:
            return {"shares": 0, "risk_amount": 0, "position_value": 0}
            
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 0.01:  # Ensure minimum risk per share
            risk_per_share = 0.01
            
        # Calculate position size
        account_risk = self.account_value * self.risk_percent
        shares = int(account_risk / risk_per_share)
        position_value = shares * entry_price
        
        return {
            "shares": shares,
            "risk_amount": account_risk,
            "position_value": position_value,
            "percentage_of_account": position_value / self.account_value
        }
