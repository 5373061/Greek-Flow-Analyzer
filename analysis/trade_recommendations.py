"""
Enhanced Trade Recommendation Generator based on Greek Energy Flow analysis.
Provides detailed, actionable options trading strategies with precise parameters.
"""

import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradeRecommendationEngine:
    """Generate actionable trade recommendations based on Greek Energy and Entropy analysis."""
    
    def __init__(self):
        self.strategies = {
            # Vanna-driven strategies
            "BULLISH_VANNA": {
                "name": "Bullish Vanna Play",
                "description": "Capitalize on positive vanna flow in bullish conditions",
                "structure": "Long call options with vanna-optimal expiration",
                "legs": [
                    {"type": "call", "action": "buy", "delta": 0.40, "quantity": 1}
                ],
                "days_to_expiration": {"min": 14, "optimal": 21, "max": 45},
                "profit_target": 0.35,
                "stop_loss": 0.20
            },
            "BEARISH_VANNA": {
                "name": "Bearish Vanna Play",
                "description": "Capitalize on negative vanna flow in bearish conditions",
                "structure": "Long put options with vanna-optimal expiration",
                "legs": [
                    {"type": "put", "action": "buy", "delta": 0.40, "quantity": 1}
                ],
                "days_to_expiration": {"min": 14, "optimal": 21, "max": 45},
                "profit_target": 0.35,
                "stop_loss": 0.20
            },
            "VANNA_REVERSAL": {
                "name": "Vanna Reversal Strategy",
                "description": "Position for reversal when vanna flows are extreme",
                "structure": "Out-of-the-money options in counter-trend direction",
                "legs": [
                    {"type": "counter_trend", "action": "buy", "delta": 0.25, "quantity": 1}
                ],
                "days_to_expiration": {"min": 7, "optimal": 14, "max": 21},
                "profit_target": 0.50,
                "stop_loss": 0.30
            },
            
            # Charm-dominated strategies
            "CHARM_CALENDAR": {
                "name": "Charm-Optimized Calendar Spread",
                "description": "Exploit time decay differential in charm-dominated regimes",
                "structure": "Calendar spread with specific tenors",
                "legs": [
                    {"type": "atm_call", "action": "buy", "delta": 0.50, "expiration": "far", "quantity": 1},
                    {"type": "atm_call", "action": "sell", "delta": 0.50, "expiration": "near", "quantity": 1}
                ],
                "near_expiration": {"days": 7},
                "far_expiration": {"days": 30},
                "profit_target": 0.25,
                "stop_loss": 0.15
            },
            "CHARM_DIAGONAL": {
                "name": "Charm-Optimized Diagonal Spread",
                "description": "Diagonal spread optimized for charm-dominated regimes",
                "structure": "Diagonal spread with strike and expiration differential",
                "legs": [
                    {"type": "otm_call", "action": "buy", "delta": 0.40, "expiration": "far", "quantity": 1},
                    {"type": "atm_call", "action": "sell", "delta": 0.50, "expiration": "near", "quantity": 1}
                ],
                "near_expiration": {"days": 7},
                "far_expiration": {"days": 30},
                "profit_target": 0.30,
                "stop_loss": 0.15
            },
            
            # Volatility-based strategies
            "LONG_VOLATILITY": {
                "name": "Long Volatility Strategy",
                "description": "Position for volatility expansion in high entropy markets",
                "structure": "Long straddle or strangle",
                "legs": [
                    {"type": "atm_call", "action": "buy", "delta": 0.50, "quantity": 1},
                    {"type": "atm_put", "action": "buy", "delta": 0.50, "quantity": 1}
                ],
                "days_to_expiration": {"min": 14, "optimal": 30, "max": 45},
                "profit_target": 0.40,
                "stop_loss": 0.20
            },
            "SHORT_VOLATILITY": {
                "name": "Short Volatility Strategy",
                "description": "Collect premium in low entropy markets with defined risk",
                "structure": "Iron condor with 30-delta short strikes",
                "legs": [
                    {"type": "otm_call", "action": "buy", "delta": 0.15, "quantity": 1},
                    {"type": "otm_call", "action": "sell", "delta": 0.30, "quantity": 1},
                    {"type": "otm_put", "action": "sell", "delta": 0.30, "quantity": 1},
                    {"type": "otm_put", "action": "buy", "delta": 0.15, "quantity": 1}
                ],
                "days_to_expiration": {"min": 21, "optimal": 30, "max": 45},
                "profit_target": 0.20,
                "stop_loss": 0.15
            }
        }
    
    def determine_directional_bias(self, greek_analysis, entropy_profile):
        """
        Determine directional bias from Greek and entropy analysis.
        
        Parameters:
            greek_analysis (dict): Greek analysis results
            entropy_profile (dict): Entropy profile data
            
        Returns:
            str: Directional bias ("Bullish", "Bearish", or "Neutral")
            float: Confidence score (0-1)
        """
        # Extract market regime
        market_regime = self._extract_market_regime(greek_analysis)
        if not market_regime:
            return "Neutral", 0.5
        
        # Extract Greek magnitudes
        magnitudes = market_regime.get("greek_magnitudes", {})
        
        # Calculate directional signals
        delta_signal = self._normalize_value(magnitudes.get("normalized_delta", 0), -1, 1)
        gamma_signal = self._normalize_value(magnitudes.get("total_gamma", 0), -5000, 5000)
        vanna_signal = self._normalize_value(magnitudes.get("total_vanna", 0), -10000, 10000)
        
        # Calculate directional score
        # Positive score = bullish, Negative score = bearish
        direction_score = (
            delta_signal * 0.4 +  # 40% weight to delta
            np.sign(gamma_signal) * abs(gamma_signal) * 0.3 +  # 30% weight to gamma
            np.sign(vanna_signal) * abs(vanna_signal) * 0.3    # 30% weight to vanna
        )
        
        # Determine direction and confidence
        confidence = abs(direction_score)
        
        if direction_score > 0.2:
            return "Bullish", min(confidence, 0.95)
        elif direction_score < -0.2:
            return "Bearish", min(confidence, 0.95)
        else:
            return "Neutral", 0.5
    
    def select_optimal_strategy(self, greek_analysis, entropy_profile, current_price):
        """
        Select optimal trading strategy based on market conditions.
        
        Parameters:
            greek_analysis (dict): Greek analysis results
            entropy_profile (dict): Entropy profile data
            current_price (float): Current price of the underlying
            
        Returns:
            dict: Detailed strategy recommendation
        """
        # Extract key data
        market_regime = self._extract_market_regime(greek_analysis)
        if not market_regime:
            return self._create_wait_recommendation(
                "Insufficient data", 
                greek_analysis, 
                entropy_profile, 
                current_price
            )
        
        regime_type = market_regime.get("primary_label", "Unknown")
        regime_strength = market_regime.get("regime_strength", 0.5)
        vol_regime = market_regime.get("volatility_regime", "Normal")
        dominant_greek = market_regime.get("dominant_greek", "Unknown")
        
        # Extract energy state
        energy_state, entropy_score = self._extract_energy_state(entropy_profile)
        
        # Determine directional bias
        direction, direction_confidence = self.determine_directional_bias(greek_analysis, entropy_profile)
        
        # Get energy levels and reset points
        energy_levels = self._extract_energy_levels(greek_analysis)
        reset_points = self._extract_reset_points(greek_analysis)
        
        # Log key decision factors
        logger.info(f"Strategy Selection: {regime_type} regime ({regime_strength:.2f} strength), {vol_regime} volatility")
        logger.info(f"Dominant Greek: {dominant_greek}, Direction: {direction} ({direction_confidence:.2f} confidence)")
        logger.info(f"Energy State: {energy_state}, Entropy Score: {entropy_score:.2f}")
        
        # Strategy selection matrix
        if "Vanna-Driven" in regime_type:
            if direction == "Bullish" and direction_confidence > 0.6:
                strategy_key = "BULLISH_VANNA"
            elif direction == "Bearish" and direction_confidence > 0.6:
                strategy_key = "BEARISH_VANNA"
            elif entropy_score > 0.6:  # High entropy
                strategy_key = "LONG_VOLATILITY"
            elif abs(energy_levels.get("imbalance", 0)) > 0.7:  # Extreme energy imbalance
                strategy_key = "VANNA_REVERSAL"
            else:
                return self._create_wait_recommendation(
                    "Vanna-driven regime without clear directional bias or volatility opportunity",
                    greek_analysis, 
                    entropy_profile, 
                    current_price
                )
                
        elif "Charm-Dominated" in regime_type:
            if entropy_score > 0.6:  # High entropy
                strategy_key = "CHARM_DIAGONAL"  # More aggressive
            else:
                strategy_key = "CHARM_CALENDAR"  # More conservative
                
        else:  # Unknown or custom regime type
            if entropy_score > 0.7:  # Very high entropy
                strategy_key = "LONG_VOLATILITY"
            elif entropy_score < 0.3 and direction_confidence < 0.4:  # Low entropy, no clear direction
                strategy_key = "SHORT_VOLATILITY"
            else:
                return self._create_wait_recommendation(
                    "Unclear market regime without actionable entropy signal",
                    greek_analysis, 
                    entropy_profile, 
                    current_price
                )
        
        # Generate full strategy recommendation
        return self._generate_detailed_recommendation(
            strategy_key, 
            greek_analysis, 
            entropy_profile, 
            current_price, 
            direction, 
            direction_confidence
        )
    
    def _generate_detailed_recommendation(self, strategy_key, greek_analysis, entropy_profile, current_price, direction, direction_confidence):
        """Generate detailed trading strategy with specific parameters."""
        strategy_template = self.strategies.get(strategy_key)
        if not strategy_template:
            return self._create_wait_recommendation(
                f"Strategy template not found for {strategy_key}",
                greek_analysis, 
                entropy_profile, 
                current_price
            )
        
        # Extract energy state and score
        _, entropy_score = self._extract_energy_state(entropy_profile)
        
        # Calculate optimal expiration dates
        today = datetime.now().date()
        expiration_days = strategy_template.get("days_to_expiration", {"optimal": 30})
        
        # Adjust expiration based on entropy
        if "optimal" in expiration_days:
            if entropy_score > 0.7:  # High entropy = shorter duration
                optimal_days = int(expiration_days["optimal"] * 0.7)
            elif entropy_score < 0.3:  # Low entropy = longer duration
                optimal_days = int(expiration_days["optimal"] * 1.3)
            else:
                optimal_days = expiration_days["optimal"]
            
            expiration_date = (today + timedelta(days=optimal_days)).strftime("%Y-%m-%d")
        else:
            # Handle calendar/diagonal spreads
            near_days = strategy_template.get("near_expiration", {}).get("days", 7)
            far_days = strategy_template.get("far_expiration", {}).get("days", 30)
            
            # Adjust based on entropy
            if entropy_score > 0.7:  # High entropy = tighter spread
                near_days = max(3, int(near_days * 0.7))
                far_days = max(14, int(far_days * 0.8))
            elif entropy_score < 0.3:  # Low entropy = wider spread
                near_days = int(near_days * 1.2)
                far_days = int(far_days * 1.2)
            
            near_date = (today + timedelta(days=near_days)).strftime("%Y-%m-%d")
            far_date = (today + timedelta(days=far_days)).strftime("%Y-%m-%d")
            expiration_date = {"near": near_date, "far": far_date}
        
        # Calculate optimal strikes
        strikes = self._calculate_optimal_strikes(
            strategy_template.get("legs", []),
            greek_analysis,
            current_price,
            direction
        )
        
        # Adjust profit target and stop loss based on entropy
        profit_target = strategy_template.get("profit_target", 0.3)
        stop_loss = strategy_template.get("stop_loss", 0.2)
        
        if entropy_score > 0.7:  # High entropy = higher targets and wider stops
            profit_target = profit_target * 1.3
            stop_loss = stop_loss * 1.2
        elif entropy_score < 0.3:  # Low entropy = lower targets and tighter stops
            profit_target = profit_target * 0.8
            stop_loss = stop_loss * 0.8
        
        # Calculate position size
        position_size = self._calculate_position_size(
            strategy_template.get("legs", []),
            direction_confidence,
            entropy_score,
            stop_loss
        )
        
        # Determine best entry conditions
        entry_conditions = self._determine_entry_conditions(
            greek_analysis,
            entropy_profile,
            current_price
        )
        
        # Create detailed recommendation
        recommendation = {
            "symbol": greek_analysis.get("symbol", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price,
            "strategy": {
                "name": strategy_template["name"],
                "key": strategy_key,
                "description": strategy_template["description"],
                "structure": strategy_template["structure"],
                "direction": direction,
                "direction_confidence": round(direction_confidence, 2)
            },
            "implementation": {
                "legs": [],
                "expiration": expiration_date,
                "position_size": position_size,
                "strikes": strikes
            },
            "risk_management": {
                "profit_target_percent": round(profit_target * 100, 1),
                "stop_loss_percent": round(stop_loss * 100, 1),
                "max_loss_per_contract": self._calculate_max_loss(strategy_template.get("legs", []), strikes, current_price),
                "risk_reward_ratio": round(profit_target / stop_loss, 2)
            },
            "entry_conditions": entry_conditions,
            "exit_conditions": self._determine_exit_conditions(strategy_template, entropy_score),
            "trade_timing": self._determine_optimal_timing(greek_analysis, entropy_profile),
            "analysis_summary": {
                "market_regime": self._extract_market_regime(greek_analysis),
                "energy_state": self._extract_energy_state(entropy_profile)[0],
                "entropy_score": round(entropy_score, 2)
            }
        }
        
        # Build detailed legs instructions
        for i, leg_template in enumerate(strategy_template.get("legs", [])):
            leg = {
                "leg_number": i + 1,
                "option_type": leg_template["type"],
                "action": leg_template["action"],
                "strike": strikes.get(f"leg_{i+1}", "ATM"),
                "quantity": leg_template["quantity"]
            }
            
            # Handle expiration for calendar/diagonal spreads
            if "expiration" in leg_template:
                if isinstance(expiration_date, dict):
                    leg["expiration"] = expiration_date[leg_template["expiration"]]
                else:
                    leg["expiration"] = expiration_date
            else:
                leg["expiration"] = expiration_date if isinstance(expiration_date, str) else expiration_date.get("far", expiration_date.get("near"))
            
            recommendation["implementation"]["legs"].append(leg)
        
        return recommendation
    
    def _create_wait_recommendation(self, reason, greek_analysis, entropy_profile, current_price):
        """Create a recommendation to wait for better conditions."""
        _, entropy_score = self._extract_energy_state(entropy_profile)
        
        return {
            "symbol": greek_analysis.get("symbol", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price,
            "strategy": {
                "name": "Wait for Better Setup",
                "key": "WAIT",
                "description": "Current market conditions do not present a favorable risk/reward opportunity",
                "reason": reason
            },
            "recommendation": "Monitor for changes in market regime or entropy state",
            "monitoring_criteria": {
                "market_regime_shift": True,
                "entropy_threshold": 0.7 if entropy_score < 0.5 else 0.3,
                "price_levels_to_watch": self._extract_key_levels(greek_analysis, current_price)
            },
            "analysis_summary": {
                "market_regime": self._extract_market_regime(greek_analysis),
                "energy_state": self._extract_energy_state(entropy_profile)[0],
                "entropy_score": round(entropy_score, 2)
            }
        }
    
    def _extract_market_regime(self, greek_analysis):
        """Extract market regime data from potentially nested structure."""
        if not greek_analysis:
            return {}
            
        # Check different possible locations
        if "market_regime" in greek_analysis:
            return greek_analysis["market_regime"]
        elif "greek_profiles" in greek_analysis and "market_regime" in greek_analysis["greek_profiles"]:
            return greek_analysis["greek_profiles"]["market_regime"]
        elif "formatted_results" in greek_analysis and "market_regime" in greek_analysis["formatted_results"]:
            return greek_analysis["formatted_results"]["market_regime"]
        
        # Check in greek_analysis parent if it exists
        parent = greek_analysis.get("greek_analysis", {})
        if "market_regime" in parent:
            return parent["market_regime"]
        elif "greek_profiles" in parent and "market_regime" in parent["greek_profiles"]:
            return parent["greek_profiles"]["market_regime"]
        elif "formatted_results" in parent and "market_regime" in parent["formatted_results"]:
            return parent["formatted_results"]["market_regime"]
            
        return {}
    
    def _extract_energy_state(self, entropy_profile):
        """Extract energy state and entropy score from profile."""
        if not entropy_profile:
            return "Unknown", 0.5
            
        # Default value
        energy_state = "Unknown"
        entropy_score = 0.5
        
        # Try to extract energy state
        if "energy_state" in entropy_profile:
            if isinstance(entropy_profile["energy_state"], dict):
                energy_state = entropy_profile["energy_state"].get("state", "Unknown")
                # Normalize entropy score to 0-1 range
                entropy_score = entropy_profile["energy_state"].get("average_normalized_entropy", 50) / 100
            elif isinstance(entropy_profile["energy_state"], str):
                energy_state = entropy_profile["energy_state"]
                # Estimate entropy score from description
                if "High" in energy_state or "Dispersed" in energy_state:
                    entropy_score = 0.8
                elif "Low" in energy_state or "Concentrated" in energy_state:
                    entropy_score = 0.2
                else:
                    entropy_score = 0.5
        elif "energy_state_string" in entropy_profile:
            energy_state = entropy_profile["energy_state_string"]
            # Estimate entropy score from description
            if "High" in energy_state or "Dispersed" in energy_state:
                entropy_score = 0.8
            elif "Low" in energy_state or "Concentrated" in energy_state:
                entropy_score = 0.2
            else:
                entropy_score = 0.5
                
        # Check if normalized_entropy is provided directly
        if "normalized_entropy" in entropy_profile:
            entropy_score = entropy_profile["normalized_entropy"]
            
        return energy_state, entropy_score
    
    def _extract_energy_levels(self, greek_analysis):
        """Extract and analyze energy levels."""
        if not greek_analysis:
            return {"levels": [], "imbalance": 0}
            
        # Look for energy levels in different possible locations
        energy_levels = []
        
        if "energy_levels" in greek_analysis:
            energy_levels = greek_analysis["energy_levels"]
        elif "greek_profiles" in greek_analysis and "energy_levels" in greek_analysis["greek_profiles"]:
            energy_levels = greek_analysis["greek_profiles"]["energy_levels"]
        elif "formatted_results" in greek_analysis and "energy_levels" in greek_analysis["formatted_results"]:
            energy_levels = greek_analysis["formatted_results"]["energy_levels"]
        
        # If not found, check in parent structure
        if not energy_levels and "greek_analysis" in greek_analysis:
            parent = greek_analysis["greek_analysis"]
            if "energy_levels" in parent:
                energy_levels = parent["energy_levels"]
            elif "greek_profiles" in parent and "energy_levels" in parent["greek_profiles"]:
                energy_levels = parent["greek_profiles"]["energy_levels"]
            elif "formatted_results" in parent and "energy_levels" in parent["formatted_results"]:
                energy_levels = parent["formatted_results"]["energy_levels"]
        
        # Calculate support/resistance imbalance if levels exist
        if energy_levels and isinstance(energy_levels, list) and len(energy_levels) > 0:
            # Get current price
            current_price = 0
            if "current_price" in greek_analysis:
                current_price = greek_analysis["current_price"]
            elif "market_data" in greek_analysis and "currentPrice" in greek_analysis["market_data"]:
                current_price = greek_analysis["market_data"]["currentPrice"]
            
            if current_price > 0:
                # Separate levels into support and resistance
                supports = [lvl for lvl in energy_levels if lvl.get("price", 0) < current_price]
                resistances = [lvl for lvl in energy_levels if lvl.get("price", 0) > current_price]
                
                # Calculate total strength
                support_strength = sum(lvl.get("strength", 0) for lvl in supports)
                resistance_strength = sum(lvl.get("strength", 0) for lvl in resistances)
                
                if support_strength + resistance_strength > 0:
                    # Calculate imbalance (-1 to 1, positive means more support)
                    imbalance = (support_strength - resistance_strength) / (support_strength + resistance_strength)
                else:
                    imbalance = 0
            else:
                imbalance = 0
        else:
            imbalance = 0
            
        return {
            "levels": energy_levels if isinstance(energy_levels, list) else [],
            "imbalance": imbalance
        }
    
    def _extract_reset_points(self, greek_analysis):
        """Extract reset points data."""
        if not greek_analysis:
            return []
            
        # Look for reset points in different possible locations
        reset_points = []
        
        if "reset_points" in greek_analysis:
            reset_points = greek_analysis["reset_points"]
        elif "greek_profiles" in greek_analysis and "reset_points" in greek_analysis["greek_profiles"]:
            reset_points = greek_analysis["greek_profiles"]["reset_points"]
        elif "formatted_results" in greek_analysis and "reset_points" in greek_analysis["formatted_results"]:
            reset_points = greek_analysis["formatted_results"]["reset_points"]
        
        # If not found, check in parent structure
        if not reset_points and "greek_analysis" in greek_analysis:
            parent = greek_analysis["greek_analysis"]
            if "reset_points" in parent:
                reset_points = parent["reset_points"]
            elif "greek_profiles" in parent and "reset_points" in parent["greek_profiles"]:
                reset_points = parent["greek_profiles"]["reset_points"]
            elif "formatted_results" in parent and "reset_points" in parent["formatted_results"]:
                reset_points = parent["formatted_results"]["reset_points"]
                
        return reset_points if isinstance(reset_points, list) else []
    
    def _extract_key_levels(self, greek_analysis, current_price):
        """Extract key price levels to watch."""
        energy_levels = self._extract_energy_levels(greek_analysis)["levels"]
        reset_points = self._extract_reset_points(greek_analysis)
        
        key_levels = []
        
        # Add strong energy levels
        if energy_levels:
            sorted_levels = sorted(energy_levels, key=lambda x: x.get("strength", 0), reverse=True)
            for level in sorted_levels[:3]:  # Top 3 strongest levels
                key_levels.append({
                    "price": level.get("price", 0),
                    "type": "energy_level",
                    "strength": level.get("strength", 0),
                    "description": level.get("description", "Strong energy level")
                })
        
        # Add recent reset points
        if reset_points:
            sorted_resets = sorted(reset_points, key=lambda x: x.get("date", ""), reverse=True)
            for point in sorted_resets[:2]:  # 2 most recent reset points
                key_levels.append({
                    "price": point.get("price", 0),
                    "type": "reset_point",
                    "date": point.get("date", ""),
                    "description": point.get("description", "Recent reset point")
                })
        
        # Add psychological levels (round numbers)
        price_magnitude = len(str(int(current_price)))
        round_level_1 = round(current_price, -(price_magnitude - 2))  # Round to nearest 100/10/1
        round_level_2 = round(current_price, -(price_magnitude - 1))  # Round to nearest 1000/100/10
        
        if abs(round_level_1 - current_price) / current_price > 0.01:  # Only add if not too close
            key_levels.append({
                "price": round_level_1,
                "type": "psychological",
                "description": "Psychological round number"
            })
        
        if abs(round_level_2 - current_price) / current_price > 0.03:  # Only add if not too close
            key_levels.append({
                "price": round_level_2,
                "type": "psychological",
                "description": "Major psychological level"
            })
        
        # Sort by distance from current price
        key_levels.sort(key=lambda x: abs(x["price"] - current_price))
        
        return key_levels
    
    def _calculate_optimal_strikes(self, legs, greek_analysis, current_price, direction):
        """Calculate optimal strike prices for strategy legs."""
        energy_levels = self._extract_energy_levels(greek_analysis)["levels"]
        
        # Round to nearest 5
        atm_strike = round(current_price / 5) * 5
        
        strikes = {"atm": atm_strike}
        
        # Calculate standard delta-based strikes
        strikes["call_25delta"] = round((current_price * 1.05) / 5) * 5
        strikes["call_10delta"] = round((current_price * 1.10) / 5) * 5
        strikes["put_25delta"] = round((current_price * 0.95) / 5) * 5
        strikes["put_10delta"] = round((current_price * 0.90) / 5) * 5
        
        # Check if any energy levels align with strike prices
        energy_aligned_strikes = {}
        if energy_levels:
            # Sort by strength
            sorted_levels = sorted(energy_levels, key=lambda x: x.get("strength", 0), reverse=True)
            
            # Find energy levels close to common delta strikes
            for strike_name, strike_price in strikes.items():
                for level in sorted_levels:
                    level_price = level.get("price", 0)
                    if abs(level_price - strike_price) / current_price < 0.02:
                        # Round to nearest 5
                        energy_aligned_strikes[strike_name] = round(level_price / 5) * 5
                        break
        
        # Generate strikes for each leg
        leg_strikes = {}
        for i, leg in enumerate(legs):
            leg_num = i + 1
            leg_type = leg["type"]
            
            if "call" in leg_type.lower() and "atm" in leg_type.lower():
                leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("atm", strikes["atm"])
            elif "put" in leg_type.lower() and "atm" in leg_type.lower():
                leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("atm", strikes["atm"])
            elif "call" in leg_type.lower() and "otm" in leg_type.lower():
                leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("call_25delta", strikes["call_25delta"])
            elif "put" in leg_type.lower() and "otm" in leg_type.lower():
                leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("put_25delta", strikes["put_25delta"])
            elif leg_type == "counter_trend":
                if direction == "Bullish":
                    leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("put_25delta", strikes["put_25delta"])
                else:
                    leg_strikes[f"leg_{leg_num}"] = energy_aligned_strikes.get("call_25delta", strikes["call_25delta"])
            else:
                # Default to ATM
                leg_strikes[f"leg_{leg_num}"] = strikes["atm"]
        
        return leg_strikes
    
    def _calculate_position_size(self, legs, direction_confidence, entropy_score, stop_loss):
        """Calculate appropriate position size based on strategy parameters."""
        # Base allocation percentage based on strategy complexity
        num_legs = len(legs)
        
        if num_legs <= 1:
            base_allocation = 0.05  # 5% for single-leg strategies
        elif num_legs == 2:
            base_allocation = 0.04  # 4% for two-leg strategies
        else:
            base_allocation = 0.03  # 3% for complex multi-leg strategies
        
        # Adjust based on confidence and entropy
        confidence_factor = direction_confidence ** 2  # Square for non-linear scaling
        entropy_factor = 1 - (abs(entropy_score - 0.5) * 0.5)  # Reduce size for extreme entropy
        
        # Calculate final allocation percentage
        adjusted_allocation = base_allocation * confidence_factor * entropy_factor
        
        # Calculate number of contracts based on account size and stop loss
        account_size = 100000  # Default account size for illustration
        risk_per_trade = account_size * adjusted_allocation
        
        # Estimated contract values (would be calculated from actual option prices)
        estimated_contract_value = 500  # Placeholder
        max_contracts = max(1, int(risk_per_trade / estimated_contract_value))
        
        return {
            "account_percentage": round(adjusted_allocation * 100, 2),
            "dollar_allocation": round(risk_per_trade, 2),
            "contracts": max_contracts,
            "sizing_rationale": f"Based on {num_legs}-leg strategy with {direction_confidence:.2f} direction confidence and {entropy_score:.2f} entropy score"
        }
    
    def _calculate_max_loss(self, legs, strikes, current_price):
        """Calculate maximum loss per contract for the strategy."""
        # This is a placeholder that would be calculated from actual option prices
        # and strategy structure. For now, we use simple estimates.
        
        num_legs = len(legs)
        
        if num_legs == 1:
            # Single leg long option - assume 100% loss of premium
            return round(current_price * 0.05 * 100, 2)  # Estimate 5% of price as premium
        elif num_legs == 2:
            # Spread strategies - typically defined risk
            leg_types = [leg["type"] for leg in legs]
            actions = [leg["action"] for leg in legs]
            
            if "buy" in actions and "sell" in actions:
                # Credit or debit spread
                if "call" in leg_types[0] and "call" in leg_types[1]:
                    # Call spread
                    width = abs(strikes.get("leg_1", 0) - strikes.get("leg_2", 0))
                    return round(width * 100 * 0.5, 2)  # 50% of width as max loss
                elif "put" in leg_types[0] and "put" in leg_types[1]:
                    # Put spread
                    width = abs(strikes.get("leg_1", 0) - strikes.get("leg_2", 0))
                    return round(width * 100 * 0.5, 2)  # 50% of width as max loss
                else:
                    # Calendar or diagonal
                    return round(current_price * 0.02 * 100, 2)  # 2% of price
            else:
                # Two long options (straddle/strangle)
                return round(current_price * 0.08 * 100, 2)  # 8% of price
        else:
            # Complex strategy (e.g., iron condor, butterfly)
            return round(current_price * 0.04 * 100, 2)  # 4% of price
    
    def _determine_entry_conditions(self, greek_analysis, entropy_profile, current_price):
        """Determine optimal conditions for trade entry."""
        energy_levels = self._extract_energy_levels(greek_analysis)["levels"]
        _, entropy_score = self._extract_energy_state(entropy_profile)
        
        # Find nearest support/resistance
        nearest_support = None
        nearest_resistance = None
        
        if energy_levels:
            supports = [lvl for lvl in energy_levels if lvl.get("price", 0) < current_price]
            resistances = [lvl for lvl in energy_levels if lvl.get("price", 0) > current_price]
            
            if supports:
                nearest_support = max(supports, key=lambda x: x.get("price", 0))
            
            if resistances:
                nearest_resistance = min(resistances, key=lambda x: x.get("price", 0))
        
        # Calculate entry zones
        if nearest_support and nearest_resistance:
            support_price = nearest_support.get("price", current_price * 0.95)
            resistance_price = nearest_resistance.get("price", current_price * 1.05)
            
            # Calculate zones based on distance and entropy
            support_distance = (current_price - support_price) / current_price
            resistance_distance = (resistance_price - current_price) / current_price
            
            if entropy_score > 0.7:  # High entropy = wider zones
                ideal_entry = current_price
                entry_zone_low = current_price * (1 - min(0.03, support_distance * 0.5))
                entry_zone_high = current_price * (1 + min(0.03, resistance_distance * 0.5))
            elif entropy_score < 0.3:  # Low entropy = tighter zones
                # Prefer entry near support/resistance
                if support_distance < resistance_distance:
                    ideal_entry = support_price * 1.01  # Just above support
                    entry_zone_low = support_price
                    entry_zone_high = support_price * 1.02
                else:
                    ideal_entry = resistance_price * 0.99  # Just below resistance
                    entry_zone_low = resistance_price * 0.98
                    entry_zone_high = resistance_price
            else:  # Medium entropy
                ideal_entry = current_price
                entry_zone_low = current_price * 0.99
                entry_zone_high = current_price * 1.01
        else:
            # No clear support/resistance - use default values
            ideal_entry = current_price
            entry_zone_low = current_price * 0.98
            entry_zone_high = current_price * 1.02
        
        # Determine VIX conditions based on volatility regime
        market_regime = self._extract_market_regime(greek_analysis)
        vol_regime = market_regime.get("volatility_regime", "Normal")
        
        if vol_regime == "High":
            vix_condition = "above_25"
        elif vol_regime == "Low":
            vix_condition = "below_20"
        else:
            vix_condition = "any"
        
        return {
            "ideal_entry_price": round(ideal_entry, 2),
            "entry_zone": {
                "low": round(entry_zone_low, 2),
                "high": round(entry_zone_high, 2)
            },
            "vix_condition": vix_condition,
            "market_conditions": self._determine_market_conditions(greek_analysis, entropy_profile),
            "max_days_to_wait": 5 if entropy_score > 0.7 else 10
        }
    
    def _determine_exit_conditions(self, strategy_template, entropy_score):
        """Determine exit conditions for the trade."""
        # Base exit criteria
        profit_target = strategy_template.get("profit_target", 0.3)
        stop_loss = strategy_template.get("stop_loss", 0.2)
        
        # Adjust based on entropy
        if entropy_score > 0.7:  # High entropy
            profit_target = profit_target * 1.3
            stop_loss = stop_loss * 1.2
            time_stop = 0.5  # Exit after 50% of planned duration if not profitable
        elif entropy_score < 0.3:  # Low entropy
            profit_target = profit_target * 0.8
            stop_loss = stop_loss * 0.8
            time_stop = 0.7  # Exit after 70% of planned duration if not profitable
        else:  # Medium entropy
            time_stop = 0.6  # Exit after 60% of planned duration if not profitable
        
        # Generate milestone targets
        days_to_expiration = strategy_template.get("days_to_expiration", {}).get("optimal", 30)
        
        # Add milestones for trade progress
        milestones = [
            {
                "day": int(days_to_expiration * 0.25),
                "expected_progress": f"{int(profit_target * 100 * 0.3)}% of target profit",
                "action_if_below": "Evaluate for adjustment"
            },
            {
                "day": int(days_to_expiration * 0.5),
                "expected_progress": f"{int(profit_target * 100 * 0.6)}% of target profit",
                "action_if_below": "Consider early exit"
            },
            {
                "day": int(days_to_expiration * 0.75),
                "expected_progress": f"{int(profit_target * 100 * 0.8)}% of target profit",
                "action_if_below": "Exit position"
            }
        ]
        
        return {
            "profit_target_percent": round(profit_target * 100, 1),
            "stop_loss_percent": round(stop_loss * 100, 1),
            "time_stop": {
                "days": int(days_to_expiration * time_stop),
                "action": "Exit if position is not profitable"
            },
            "milestones": milestones,
            "adjustment_triggers": {
                "adverse_move": f"{round(stop_loss * 100 * 0.7, 1)}% against position",
                "vix_spike": "30% increase in VIX",
                "energy_level_breach": "Strong support/resistance level broken"
            },
            "greek_thresholds": {
                "delta_flip": "Exit if delta changes sign",
                "gamma_threshold": "Adjust if gamma exceeds position limits"
            }
        }
    
    def _determine_optimal_timing(self, greek_analysis, entropy_profile):
        """Determine optimal timing for trade execution."""
        _, entropy_score = self._extract_energy_state(entropy_profile)
        
        # Determine time of day preferences
        if entropy_score > 0.7:  # High entropy
            time_of_day = "Mid-session (10:30 AM - 2:00 PM ET)"
            day_of_week = "Tuesday through Thursday"
        elif entropy_score < 0.3:  # Low entropy
            time_of_day = "Morning (9:30 AM - 10:30 AM ET)"
            day_of_week = "Monday or Friday"
        else:  # Medium entropy
            time_of_day = "Any time during market hours"
            day_of_week = "Any trading day"
        
        # Determine events to avoid
        market_regime = self._extract_market_regime(greek_analysis)
        vol_regime = market_regime.get("volatility_regime", "Normal")
        
        if vol_regime == "High":
            avoid_events = ["FOMC announcements", "Major economic reports", "Earnings releases"]
        else:
            avoid_events = ["Earnings releases"]
        
        return {
            "preferred_time_of_day": time_of_day,
            "preferred_day_of_week": day_of_week,
            "avoid_events": avoid_events,
            "volatility_preference": "Decreasing" if entropy_score > 0.6 else "Stable",
            "execution_strategy": "Limit order" if entropy_score < 0.5 else "Multi-leg market order"
        }
    
    def _determine_market_conditions(self, greek_analysis, entropy_profile):
        """Determine ideal market conditions for entry."""
        market_regime = self._extract_market_regime(greek_analysis)
        regime_type = market_regime.get("primary_label", "Unknown")
        
        if "Vanna-Driven" in regime_type:
            return {
                "ideal": ["Vanna-dominant flow", "Directional momentum"],
                "avoid": ["Expiration week", "Sudden volatility spikes"]
            }
        elif "Charm-Dominated" in regime_type:
            return {
                "ideal": ["Low historical volatility", "Stable price action", "High time decay"],
                "avoid": ["Major news catalysts", "Earnings announcements"]
            }
        else:
            return {
                "ideal": ["Clear directional bias", "Defined support/resistance"],
                "avoid": ["Choppy price action", "Major economic announcements"]
            }
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize a value to the range [0, 1]."""
        if isinstance(value, str):
            try:
                value = float(value.replace(',', ''))
            except (ValueError, TypeError):
                return 0
                
        try:
            value = float(value)
            normalized = (value - min_val) / (max_val - min_val)
            return max(0, min(normalized, 1))
        except (ValueError, TypeError, ZeroDivisionError):
            return 0

# Main function to generate a recommendation
def generate_trade_recommendation(analysis_results, entropy_data, current_price):
    """
    Generate trade recommendations based on analysis results and entropy data.
    
    Args:
        analysis_results: Dictionary containing analysis results
        entropy_data: Dictionary containing entropy analysis results
        current_price: Current price of the underlying asset
        
    Returns:
        Dictionary containing trade recommendation
    """
    from datetime import datetime
    
    print(f"DEBUG - Analysis Results Keys: {list(analysis_results.keys())}")
    print(f"DEBUG - Entropy Data Keys: {list(entropy_data.keys())}")
    print(f"DEBUG - Current Price: {current_price}")
    
    # Extract market regime
    greek_analysis = analysis_results.get("greek_analysis", {})
    market_regime = greek_analysis.get("market_regime", {})
    primary_regime = market_regime.get("primary", "Unknown")
    volatility_regime = market_regime.get("volatility", "Unknown")
    
    # Extract ML predictions if available
    ml_predictions = analysis_results.get("ml_predictions", {})
    ml_primary_regime = ml_predictions.get("primary_regime", {}).get("prediction", "Unknown")
    ml_confidence = ml_predictions.get("primary_regime", {}).get("confidence", 0.5)
    
    # Extract entropy metrics
    metrics = entropy_data.get("metrics", {})
    avg_entropy = metrics.get("average_entropy", 50)
    entropy_trend = metrics.get("entropy_trend", "stable")
    
    # Extract anomalies
    anomalies = entropy_data.get("anomalies", [])
    has_anomalies = len(anomalies) > 0
    
    # Extract price projections
    price_projections = greek_analysis.get("price_projections", {})
    upside_target = price_projections.get("upside_target", current_price * 1.1)
    support_level = price_projections.get("support_level", current_price * 0.9)
    
    # Determine energy state
    if avg_entropy > 70:
        energy_state = "High Energy"
    elif avg_entropy < 30:
        energy_state = "Low Energy"
    else:
        energy_state = "Neutral"
        
    if entropy_trend == "increasing":
        energy_state += " (Increasing)"
    elif entropy_trend == "decreasing":
        energy_state += " (Decreasing)"
    
    print(f"DEBUG - Derived energy_state: {energy_state}")
    
    # Generate recommendation based on market regime and energy state
    strategy = {}
    
    if primary_regime == "Bullish" and energy_state.startswith("High Energy"):
        strategy = {
            "name": "Bullish Momentum Strategy",
            "reason": "Bullish market regime with high energy",
            "description": "Consider long calls or call spreads targeting the upside target.",
            "target_price": upside_target,
            "stop_loss": support_level
        }
    elif primary_regime == "Bullish" and energy_state.startswith("Neutral"):
        strategy = {
            "name": "Bullish Income Strategy",
            "reason": "Bullish market regime with neutral energy",
            "description": "Consider selling put spreads or covered calls.",
            "target_price": upside_target,
            "stop_loss": support_level
        }
    elif primary_regime == "Bearish" and energy_state.startswith("High Energy"):
        strategy = {
            "name": "Bearish Momentum Strategy",
            "reason": "Bearish market regime with high energy",
            "description": "Consider long puts or put spreads targeting the support level.",
            "target_price": support_level,
            "stop_loss": upside_target
        }
    elif primary_regime == "Bearish" and energy_state.startswith("Neutral"):
        strategy = {
            "name": "Bearish Income Strategy",
            "reason": "Bearish market regime with neutral energy",
            "description": "Consider selling call spreads or covered puts.",
            "target_price": support_level,
            "stop_loss": upside_target
        }
    else:
        strategy = {
            "name": "Neutral Strategy",
            "reason": f"{primary_regime} market regime with {energy_state}",
            "description": "Consider iron condors or calendar spreads.",
            "target_price": current_price,
            "range_low": support_level,
            "range_high": upside_target
        }
    
    # Add anomaly warning if present
    if has_anomalies:
        strategy["anomaly_warning"] = "Anomalies detected in the options chain. Exercise caution."
    
    # Add ML confidence if available
    if ml_primary_regime != "Unknown":
        strategy["ml_confirmation"] = f"ML model predicts {ml_primary_regime} with {ml_confidence:.0%} confidence."
    
    return {
        "strategy": strategy,
        "timestamp": datetime.now().isoformat()
    }

def generate_pattern_trade_recommendation(symbol, pattern_data, output_dir=None):
    """
    Generate trade recommendations based on ordinal pattern analysis.
    
    Args:
        symbol: Stock symbol
        pattern_data: Pattern analysis data
        output_dir: Directory to save recommendation files
        
    Returns:
        Dictionary with trade recommendation
    """
    # Extract pattern information
    pattern_type = pattern_data.get("primary_pattern", "Unknown")
    pattern_strength = pattern_data.get("pattern_strength", 0.0)
    pattern_direction = pattern_data.get("expected_direction", "neutral")
    current_price = pattern_data.get("current_price", 0.0)
    
    # Default values
    action = "HOLD"
    confidence = pattern_strength if pattern_strength else 0.5
    target_price = current_price
    stop_price = current_price
    
    # Determine action based on pattern direction
    if pattern_direction == "bullish" and pattern_strength > 0.6:
        action = "BUY"
        target_price = current_price * (1 + 0.05 * pattern_strength)
        stop_price = current_price * (1 - 0.03 * pattern_strength)
    elif pattern_direction == "bearish" and pattern_strength > 0.6:
        action = "SELL"
        target_price = current_price * (1 - 0.05 * pattern_strength)
        stop_price = current_price * (1 + 0.03 * pattern_strength)
    
    # Create recommendation
    recommendation = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "current_price": current_price,
        "action": action,
        "confidence": confidence,
        "target_price": target_price,
        "stop_price": stop_price,
        "pattern_type": pattern_type,
        "pattern_strength": pattern_strength,
        "pattern_direction": pattern_direction,
        "strategy": "Pattern-Based",
        "description": f"{pattern_type} pattern detected with {pattern_strength:.1%} strength, suggesting {pattern_direction} movement"
    }
    
    # Save recommendation if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        rec_path = os.path.join(output_dir, f"{symbol}_pattern_recommendation.json")
        try:
            with open(rec_path, 'w') as f:
                json.dump(recommendation, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pattern recommendation for {symbol}: {e}")
    
    return recommendation

# Create a simple ordinal pattern analyzer class if it doesn't exist
class OrdinalPatternAnalyzer:
    """
    Analyzes ordinal patterns in Greek data.
    This is a placeholder implementation - create a proper implementation in analysis/ordinal_pattern_analyzer.py
    """
    
    def __init__(self, cache_dir="cache"):
        """Initialize the analyzer."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def analyze_tickers(self, tickers, use_parallel=True):
        """
        Analyze ordinal patterns for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            use_parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
                future_to_ticker = {executor.submit(self.analyze_ticker, ticker): ticker for ticker in tickers}
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        results[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"Error analyzing {ticker}: {e}")
                        results[ticker] = {"error": str(e)}
        else:
            for ticker in tickers:
                try:
                    results[ticker] = self.analyze_ticker(ticker)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
                    results[ticker] = {"error": str(e)}
        
        return results
    
    def analyze_ticker(self, ticker):
        """
        Analyze ordinal patterns for a single ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with analysis results
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Load Greek data for the ticker
        # 2. Identify ordinal patterns in the data
        # 3. Calculate pattern strength and expected direction
        
        # For now, return random patterns for testing
        import random
        
        patterns = ["Rising Three", "Falling Three", "Inside Bar", "Outside Bar", 
                   "Bullish Engulfing", "Bearish Engulfing", "Doji", "Hammer"]
        
        directions = ["bullish", "bearish", "neutral"]
        weights = [0.4, 0.4, 0.2]  # More likely to be directional
        
        return {
            "primary_pattern": random.choice(patterns),
            "pattern_strength": random.uniform(0.3, 0.9),
            "expected_direction": random.choices(directions, weights=weights)[0],
            "current_price": random.uniform(50, 500),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import os
    import json
    import sys
    import glob
    
    def test_trade_recommendations():
        """Test function for trade recommendations using real data from results directory."""
        # Find all analysis results
        results_dir = "results"
        if not os.path.exists(results_dir):
            print(f"Results directory '{results_dir}' not found.")
            return
            
        # Look for analysis files
        analysis_patterns = [
            f"{results_dir}/*_analysis_results.json",
            f"{results_dir}/*_analysis.json",
            f"{results_dir}/*_greek_analysis.json"
        ]
        
        analysis_files = []
        for pattern in analysis_patterns:
            analysis_files.extend(glob.glob(pattern))
        
        if not analysis_files:
            print("No analysis results found. Please run the pipeline first.")
            return
            
        print(f"Found {len(analysis_files)} analysis results")
        
        # Process each ticker's analysis results
        for analysis_file in analysis_files:
            try:
                # Extract symbol from filename
                base_name = os.path.basename(analysis_file)
                symbol = base_name.split('_')[0]
                
                print(f"\nProcessing {symbol}...")
                
                # Load analysis results
                with open(analysis_file, 'r') as f:
                    analysis_results = json.load(f)
                
                # Extract entropy data
                entropy_data = analysis_results.get("entropy_data", 
                                                 analysis_results.get("entropy_analysis", {}))
                
                # Extract current price
                current_price = analysis_results.get("market_data", {}).get("currentPrice", 
                               analysis_results.get("current_price", 100.0))
                
                print(f"Current price: {current_price}")
                
                # Generate trade recommendation
                recommendation = generate_trade_recommendation(
                    analysis_results,
                    entropy_data,
                    current_price
                )
                
                # Save recommendation
                output_file = f"{results_dir}/{symbol}_enhanced_recommendation.json"
                with open(output_file, "w") as f:
                    json.dump(recommendation, f, indent=2)
                
                # Print summary of recommendation
                strategy_name = recommendation.get("strategy", {}).get("name", "Unknown")
                
                # Print richer summary based on recommendation type
                if "implementation" in recommendation and "legs" in recommendation["implementation"]:
                    legs = recommendation["implementation"]["legs"]
                    legs_summary = ", ".join([f"{leg['action']} {leg['quantity']} {leg['option_type']} @ {leg['strike']}" for leg in legs])
                    print(f"Enhanced recommendation for {symbol}: {strategy_name}")
                    print(f"Structure: {legs_summary}")
                    print(f"Expiration: {recommendation['implementation']['expiration']}")
                    print(f"Position size: {recommendation['implementation']['position_size']['contracts']} contracts ({recommendation['implementation']['position_size']['account_percentage']}% of account)")
                    print(f"Profit target: {recommendation['risk_management']['profit_target_percent']}%, Stop loss: {recommendation['risk_management']['stop_loss_percent']}%")
                else:
                    print(f"Enhanced recommendation for {symbol}: {strategy_name}")
                    if "reason" in recommendation.get("strategy", {}):
                        print(f"Reason: {recommendation['strategy']['reason']}")
                    
                print(f"Saved to {output_file}")
                
            except Exception as e:
                import traceback
                print(f"Error processing {os.path.basename(analysis_file)}: {e}")
                traceback.print_exc()
    
    # If run directly, process all tickers
    if len(sys.argv) > 1:
        # Process a specific symbol if provided
        symbol = sys.argv[1]
        print(f"Processing single symbol: {symbol}")
        
        # Try to load analysis results for this symbol
        results_dir = "results"
        possible_paths = [
            f"{results_dir}/{symbol}_analysis_results.json",
            f"{results_dir}/{symbol}_analysis.json",
            f"{results_dir}/{symbol}_greek_analysis.json"
        ]
        
        analysis_file = None
        for path in possible_paths:
            if os.path.exists(path):
                analysis_file = path
                break
                
        if not analysis_file:
            print(f"No analysis results found for {symbol}. Please run the pipeline first.")
            sys.exit(1)
            
        try:
            # Load analysis results
            with open(analysis_file, 'r') as f:
                analysis_results = json.load(f)
            
            # Extract entropy data
            entropy_data = analysis_results.get("entropy_data", 
                                             analysis_results.get("entropy_analysis", {}))
            
            # Extract current price
            current_price = analysis_results.get("market_data", {}).get("currentPrice", 
                           analysis_results.get("current_price", 100.0))
            
            print(f"Current price: {current_price}")
            
            # Generate trade recommendation
            recommendation = generate_trade_recommendation(
                analysis_results,
                entropy_data,
                current_price
            )
            
            # Save recommendation
            output_file = f"{results_dir}/{symbol}_enhanced_recommendation.json"
            with open(output_file, "w") as f:
                json.dump(recommendation, f, indent=2)
            
            # Print summary
            print(f"\nEnhanced Trade Recommendation for {symbol}:")
            print(f"Strategy: {recommendation['strategy']['name']}")
            
            if "implementation" in recommendation:
                print("\nImplementation Details:")
                for leg in recommendation.get("implementation", {}).get("legs", []):
                    print(f"- {leg['action'].capitalize()} {leg['quantity']} {leg['option_type']} @ strike {leg['strike']}, expiring {leg['expiration']}")
                
                position = recommendation.get("implementation", {}).get("position_size", {})
                print(f"\nPosition Size: {position.get('contracts', 'N/A')} contracts ({position.get('account_percentage', 'N/A')}% of account)")
                
                risk = recommendation.get("risk_management", {})
                print(f"Risk/Reward: {risk.get('risk_reward_ratio', 'N/A')}, Profit Target: {risk.get('profit_target_percent', 'N/A')}%, Stop Loss: {risk.get('stop_loss_percent', 'N/A')}%")
                
                entry = recommendation.get("entry_conditions", {})
                print(f"\nEntry Zone: {entry.get('entry_zone', {}).get('low', 'N/A')} - {entry.get('entry_zone', {}).get('high', 'N/A')}")
                
                timing = recommendation.get("trade_timing", {})
                print(f"Ideal Timing: {timing.get('preferred_time_of_day', 'N/A')} on {timing.get('preferred_day_of_week', 'N/A')}")
            else:
                print(f"\nRecommendation: {recommendation.get('strategy', {}).get('description', 'N/A')}")
                if "reason" in recommendation.get("strategy", {}):
                    print(f"Reason: {recommendation['strategy']['reason']}")
                
                if "monitoring_criteria" in recommendation:
                    print("\nLevels to Watch:")
                    for level in recommendation.get("monitoring_criteria", {}).get("price_levels_to_watch", []):
                        print(f"- {level.get('price', 'N/A')}: {level.get('description', 'N/A')}")
            
            print(f"\nSaved detailed recommendation to {output_file}")
            
        except Exception as e:
            import traceback
            print(f"Error processing {symbol}: {e}")
            traceback.print_exc()
    else:
        # Process all symbols
        test_trade_recommendations()


