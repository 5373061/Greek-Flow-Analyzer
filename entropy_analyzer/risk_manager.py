# risk_manager.py - Advanced Risk Management Framework

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("GreekFlow.RiskManager")

class AdvancedRiskManager:
    """Advanced risk management based on Greek energy states and entropy."""
    
    def __init__(self, greek_data=None, entropy_data=None, spot_price=None, config=None):
        """Initialize the risk manager.
        
        Args:
            greek_data (dict): Greek analysis results
            entropy_data (dict): Entropy analysis results
            spot_price (float): Current spot price
            config (module): Configuration module
        """
        self.greek_data = greek_data
        self.entropy_data = entropy_data
        self.spot_price = spot_price
        self.config = config
        
        # Default risk parameters
        self.default_risk = {
            "account_value": getattr(config, "ACCOUNT_VALUE", 50000),
            "max_risk_percent": getattr(config, "RISK_PERCENT", 0.02),
            "base_stop_percent": getattr(config, "BASE_STOP_PERCENT", 0.05),
            "volatility_factor": getattr(config, "VOLATILITY_FACTOR", 1.0)
        }
    
    def set_data(self, greek_data=None, entropy_data=None, spot_price=None):
        """Set the data for risk analysis.
        
        Args:
            greek_data (dict, optional): Greek analysis results
            entropy_data (dict, optional): Entropy analysis results
            spot_price (float, optional): Current spot price
        """
        if greek_data is not None:
            self.greek_data = greek_data
        if entropy_data is not None:
            self.entropy_data = entropy_data
        if spot_price is not None:
            self.spot_price = spot_price
    
    def calculate_dynamic_position_size(self):
        """Calculate dynamic position size based on energy states.
        
        Returns:
            dict: Position sizing information
        """
        if not self.greek_data or not self.entropy_data or self.spot_price is None:
            return {
                "position_size": 0,
                "position_value": 0,
                "risk_percent": self.default_risk["max_risk_percent"],
                "sizing_factor": 1.0,
                "reason": "Insufficient data for position sizing"
            }
        
        try:
            # Extract energy state
            energy_state = self.entropy_data.get("energy_state", {})
            avg_entropy = energy_state.get("average_normalized_entropy", 50)
            
            # Scale risk based on entropy
            # Lower entropy = more concentrated energy = more confidence = larger position
            entropy_factor = 1.5 - (avg_entropy / 100)  # 0.5 to 1.5 scaling
            entropy_factor = max(0.5, min(1.5, entropy_factor))  # Clamp to 0.5-1.5 range
            
            # Extract market regime from Greek data
            regime = self.greek_data.get("market_regime", {})
            vol_regime = regime.get("volatility_regime", "Normal").lower()
            
            # Scale based on volatility regime
            vol_factor = 1.0
            if "high" in vol_regime:
                vol_factor = 0.7  # Reduce size in high volatility
            elif "low" in vol_regime:
                vol_factor = 1.2  # Increase size in low volatility
            
            # Adjust for anomalies
            anomaly_data = self.entropy_data.get("anomalies", {})
            anomaly_count = anomaly_data.get("anomaly_count", 0)
            
            # Reduce size when anomalies present
            anomaly_factor = 1.0 - (anomaly_count * 0.1)
            anomaly_factor = max(0.5, anomaly_factor)  # Don't go below 50%
            
            # Calculate combined factor
            combined_factor = entropy_factor * vol_factor * anomaly_factor
            
            # Calculate actual risk percent
            risk_percent = self.default_risk["max_risk_percent"] * combined_factor
            risk_percent = min(risk_percent, self.default_risk["max_risk_percent"] * 1.5)  # Cap at 150% of max
            
            # Calculate position value
            account_value = self.default_risk["account_value"]
            position_value = account_value * risk_percent
            
            # Calculate shares based on stop distance
            stop_distance = self.calculate_dynamic_stop_distance()
            if stop_distance > 0:
                shares = int(position_value / stop_distance)
            else:
                shares = int(position_value / (self.spot_price * self.default_risk["base_stop_percent"]))
            
            reason_parts = []
            if entropy_factor != 1.0:
                direction = "concentrated" if entropy_factor > 1.0 else "dispersed"
                reason_parts.append(f"energy {direction}")
                
            if vol_factor != 1.0:
                reason_parts.append(f"{vol_regime} volatility")
                
            if anomaly_factor != 1.0:
                reason_parts.append(f"{anomaly_count} anomalies")
                
            reason = f"Adjusted for {', '.join(reason_parts)}" if reason_parts else "Standard sizing"
            
            return {
                "position_size": shares,
                "position_value": shares * self.spot_price,
                "risk_percent": risk_percent,
                "sizing_factor": combined_factor,
                "entropy_factor": entropy_factor,
                "volatility_factor": vol_factor,
                "anomaly_factor": anomaly_factor,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error in position sizing: {e}")
            return {
                "position_size": 0,
                "position_value": 0,
                "risk_percent": self.default_risk["max_risk_percent"],
                "sizing_factor": 1.0,
                "reason": f"Error: {str(e)}"
            }
    
    def calculate_dynamic_stop_distance(self):
        """Calculate dynamic stop loss distance based on energy states.
        
        Returns:
            float: Stop loss distance (price units)
        """
        if not self.greek_data or self.spot_price is None:
            return self.spot_price * self.default_risk["base_stop_percent"]
        
        try:
            # Extract energy levels from Greek data
            levels = self.greek_data.get("energy_levels", [])
            
            # Find support levels
            supports = []
            for lvl in levels:
                try:
                    price_str = str(lvl.get("price", "")).replace("$", "")
                    price = float(price_str)
                    if "support" in str(lvl.get("direction", "")).lower() and price < self.spot_price:
                        supports.append((price, lvl.get("strength", 1)))
                except (ValueError, TypeError):
                    continue
            
            # If we have support levels, use the closest strong one
            if supports:
                # Sort by distance (closest first)
                supports.sort(key=lambda x: self.spot_price - x[0])
                
                # Find strongest support within reasonable range
                # Don't use supports that are too far away
                max_distance = self.spot_price * 0.1  # Max 10% away
                
                valid_supports = [(price, strength) for price, strength in supports 
                                if self.spot_price - price <= max_distance]
                
                if valid_supports:
                    # Sort by strength (strongest first)
                    valid_supports.sort(key=lambda x: x[1], reverse=True)
                    best_support = valid_supports[0][0]
                    
                    # Calculate distance
                    distance = self.spot_price - best_support
                    
                    # Add buffer (5% of distance)
                    distance = distance * 1.05
                    
                    return distance
            
            # Fallback to volatility-based stop
            if "aggregated_greeks" in self.greek_data:
                iv = self.greek_data.get("market_data", {}).get("impliedVolatility", 0.3)
                # Scale stop with volatility
                vol_stop = self.spot_price * self.default_risk["base_stop_percent"] * max(1.0, iv / 0.3)
                return vol_stop
            
            # Default fallback
            return self.spot_price * self.default_risk["base_stop_percent"]
            
        except Exception as e:
            logger.error(f"Error in stop distance calculation: {e}")
            return self.spot_price * self.default_risk["base_stop_percent"]
    
    def calculate_dynamic_take_profit(self):
        """Calculate dynamic take profit level based on energy states.
        
        Returns:
            float: Take profit level (price)
        """
        if not self.greek_data or self.spot_price is None:
            return self.spot_price * (1 + 2 * self.default_risk["base_stop_percent"])
        
        try:
            # Extract energy levels from Greek data
            levels = self.greek_data.get("energy_levels", [])
            
            # Find resistance levels
            resistances = []
            for lvl in levels:
                try:
                    price_str = str(lvl.get("price", "")).replace("$", "")
                    price = float(price_str)
                    if "resistance" in str(lvl.get("direction", "")).lower() and price > self.spot_price:
                        resistances.append((price, lvl.get("strength", 1)))
                except (ValueError, TypeError):
                    continue
            
            # If we have resistance levels, use the closest strong one
            if resistances:
                # Sort by distance (closest first)
                resistances.sort(key=lambda x: x[0] - self.spot_price)
                
                # Find strongest resistance within reasonable range
                # Don't use resistances that are too far away
                max_distance = self.spot_price * 0.15  # Max 15% away
                
                valid_resistances = [(price, strength) for price, strength in resistances 
                                   if price - self.spot_price <= max_distance]
                
                if valid_resistances:
                    # Sort by strength (strongest first)
                    valid_resistances.sort(key=lambda x: x[1], reverse=True)
                    best_resistance = valid_resistances[0][0]
                    
                    # Calculate target with slight discount (5% of distance)
                    target = best_resistance - (best_resistance - self.spot_price) * 0.05
                    
                    return target
            
            # Fallback to stop-loss multiple
            stop_distance = self.calculate_dynamic_stop_distance()
            return self.spot_price + (2.0 * stop_distance)  # 2:1 reward:risk ratio
            
        except Exception as e:
            logger.error(f"Error in take profit calculation: {e}")
            return self.spot_price * (1 + 2 * self.default_risk["base_stop_percent"])
    
    def generate_risk_management_plan(self):
        """Generate a comprehensive risk management plan.
        
        Returns:
            dict: Risk management recommendations
        """
        if not self.greek_data or not self.entropy_data or self.spot_price is None:
            logger.warning("Insufficient data for risk management plan")
            return {
                "stop_loss": "N/A",
                "take_profit": "N/A",
                "position_size": "N/A",
                "risk_reward_ratio": "N/A",
                "adaptive_exits": []
            }
        
        try:
            # Calculate dynamic stop and target
            stop_distance = self.calculate_dynamic_stop_distance()
            stop_level = self.spot_price - stop_distance
            
            take_profit = self.calculate_dynamic_take_profit()
            
            # Calculate risk/reward ratio
            risk = stop_distance
            reward = take_profit - self.spot_price
            ratio = reward / risk if risk > 0 else 0
            
            # Calculate position sizing
            position_data = self.calculate_dynamic_position_size()
            
            # Extract energy state for adaptive exits
            energy_state = self.entropy_data.get("energy_state", {})
            
            # Generate adaptive exit recommendations
            adaptive_exits = []
            
            # Time-based exit (based on charm effects)
            if "aggregated_greeks" in self.greek_data:
                total_charm = self.greek_data.get("aggregated_greeks", {}).get("total_charm", 0)
                
                # If charm is significant, suggest time-based exit
                if abs(total_charm) > 0.05:
                    days_to_exit = 1 if total_charm < 0 else 3
                    adaptive_exits.append({
                        "type": "Time-based",
                        "condition": f"Exit after {days_to_exit} trading days",
                        "reason": "Significant charm decay detected"
                    })
            
            # Volatility-based exit
            vega = self.greek_data.get("aggregated_greeks", {}).get("total_vega", 0)
            if abs(vega) > 5:
                # If we have significant vega exposure
                condition = "IV increases by >20%" if vega > 0 else "IV decreases by >20%"
                adaptive_exits.append({
                    "type": "Volatility-based",
                    "condition": condition,
                    "reason": f"Significant {'positive' if vega > 0 else 'negative'} vega exposure"
                })
            
            # Entropy-based exit
            if "average_entropy_gradient" in energy_state:
                gradient = energy_state.get("average_entropy_gradient", 0)
                
                if abs(gradient) > 0.03:
                    # If entropy is changing rapidly
                    direction = "increases" if gradient > 0 else "decreases"
                    adaptive_exits.append({
                        "type": "Structure-based",
                        "condition": f"If energy concentration {direction} significantly",
                        "reason": "Rapid change in energy distribution detected"
                    })
            
            # Partial exit recommendations
            if ratio > 1.5:
                adaptive_exits.append({
                    "type": "Partial exit",
                    "condition": f"Exit 50% at 50% of target (${self.spot_price + (reward * 0.5):.2f})",
                    "reason": "Lock in partial profits on favorable risk/reward"
                })
            
            return {
                "stop_loss": f"${stop_level:.2f}",
                "take_profit": f"${take_profit:.2f}",
                "position_size": position_data["position_size"],
                "position_value": f"${position_data['position_value']:.2f}",
                "risk_reward_ratio": f"{ratio:.2f}",
                "adaptive_exits": adaptive_exits,
                "sizing_reason": position_data["reason"],
                "volatility_risk": "High" if energy_state.get("average_normalized_entropy", 50) > 70 else 
                                 "Low" if energy_state.get("average_normalized_entropy", 50) < 30 else "Medium"
            }
            
        except Exception as e:
            logger.error(f"Error in risk management plan: {e}")
            return {
                "stop_loss": "N/A",
                "take_profit": "N/A",
                "position_size": "N/A",
                "risk_reward_ratio": "N/A",
                "adaptive_exits": [],
                "error": str(e)
            }
