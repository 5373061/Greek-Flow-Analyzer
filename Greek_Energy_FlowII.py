# untitled.py
# (Combined script with Config.py logic and modifications for Barchart data)

"""
Greek Energy Flow Analysis Module - Python Implementation

Provides comprehensive analysis of option Greeks with a focus on energy flow dynamics:
- Uses provided Delta and Gamma from input data.
- Calculates Vanna and Charm using Black-Scholes derivatives and provided IV.
- Reset point detection with multi-factor significance scoring (configurable).
- Market regime classification based on Greek configurations (using external thresholds).
- Price level generation for likely energy concentration/release points.
- Logging at key steps for debugging and traceability.
- Includes preprocessing logic for Barchart CSV format.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Optional
import warnings
from dateutil import parser as dateutil_parser # Use robust date parsing
import re
import os

# --- Configuration ---
# Using centralized configuration from greek_flow.config,
# Import directly from config.py at the root level
try:
    # Try to import from config.py
    from config import DEFAULT_CONFIG, get_config
except ImportError:
    # Define minimal DEFAULT_CONFIG if import fails
    DEFAULT_CONFIG = {
        'regime_thresholds': {},
        'reset_factors': {},
        'price_projection': {'range_percent': 0.15, 'steps': 30},
        'plot_defaults': {'figsize': (12, 6), 'color_scheme': {'gamma': 'red', 'vanna': 'blue', 'charm': 'green', 'default': 'purple'}}
    }
    def get_config():
        return DEFAULT_CONFIG.copy()
# --- End Configuration ---


# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BlackScholesModel:
    @staticmethod
    def _calculate_d1_d2_n(S, K, T, r, sigma):
        """Helper to calculate common components."""
        try:
            # Input validation
            epsilon = 1e-7
            T = max(T, epsilon)  # Prevent division by zero
            sigma = max(sigma, epsilon)  # Prevent unstable calculations
            S = max(S, epsilon)
            K = max(K, epsilon)
            
            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return d1, d2, norm.pdf
        except Exception as e:
            logging.error(f"Error in d1/d2 calculation: {str(e)}")
            return np.nan, np.nan, norm.pdf

    def calculate(self, 
                 S: float, 
                 K: float, 
                 T: float, 
                 r: float, 
                 sigma: float, 
                 option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks"""
        is_call = option_type.lower() == 'call'
        d1, d2, n = BlackScholesModel._calculate_d1_d2_n(S, K, T, r, sigma)

        if np.isnan(d1): # Check if helper returned NaN due to bad inputs
             return {
                'price': np.nan, 'delta': np.nan, 'gamma': np.nan, 'theta': np.nan,
                'vega': np.nan, 'rho': np.nan, 'charm': np.nan, 'vanna': np.nan,
                'volga': np.nan, 'speed': np.nan, 'zomma': np.nan, 'color': np.nan
            }

        N = norm.cdf # Standard normal CDF

        # Option price calculation
        if is_call:
            price = S * N(d1) - K * np.exp(-r * T) * N(d2)
        else:
            price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)

        # First-order Greeks
        delta = N(d1) if is_call else N(d1) - 1
        gamma = n(d1) / (S * sigma * np.sqrt(T))
        theta = (-(S * sigma * n(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2)) if is_call \
                else (-(S * sigma * n(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N(-d2))
        vega = S * np.sqrt(T) * n(d1)
        rho = (K * T * np.exp(-r * T) * N(d2)) if is_call else (-K * T * np.exp(-r * T) * N(-d2))

        # Second-order Greeks and others
        # Vanna - derivative of delta with respect to volatility
        vanna = -n(d1) * d2 / sigma # Note: Corrected BSM Vanna formula often includes S term: -S * n(d1) * d2 / sigma
                                    # Let's use the version consistent with many sources:
        vanna = -S * n(d1) * d2 / sigma


        # Charm - derivative of delta with respect to time
        # Using the formula from the original script
        charm_term1 = 2 * (r - sigma**2 / 2) # Assuming r is risk-free, no dividend yield q
        charm_term2 = d2 * sigma / (np.sqrt(T)) # Check Charm derivation: often involves (2rT - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T)) or similar
        # Re-using original script's formula, ensure it's correct:
        # charm = -n(d1) * (2 * (r - sigma**2 / 2) * np.sqrt(T) - d2 * sigma) / (2 * T * sigma * np.sqrt(T)) # Error in original division
        # Corrected Charm Derivation (check source):
        # charm = -n(d1) * ( (r - sigma^2/2)/(sigma*sqrt(T)) - d2/(2*T) ) # Alternative form
        # Let's use the common form: dDelta/dT = -n(d1)[(r - q - sigma^2/2)/(sigma*sqrt(T)) - d2/(2T)]
        # Assuming q=0:
        charm = -n(d1) * ( (r - sigma**2/2)/(sigma*np.sqrt(T)) - d2/(2*T) )


        volga = vega * (d1 * d2 / sigma)
        # Speed calculation can be unstable near T=0 or sigma=0
        if sigma * np.sqrt(T) < 1e-6:
             speed = np.nan
        else:
             speed = -gamma / S * (1 + d1 / (sigma * np.sqrt(T)))

        zomma = gamma * ((d1 * d2 - 1) / sigma)
        # Color calculation can be unstable near T=0 or sigma=0
        if 2 * S * T * sigma * np.sqrt(T) < 1e-6:
             color = np.nan
        else:
             # Original color calculation seems complex, let's simplify or use a known formula
             # dGamma/dT = -gamma * [ (1-d1*d2)/(2T) + (d1*(r-q) + d2*sigma^2/2)/(sigma*sqrt(T)) ]
             # Assuming q=0:
             term_color_1 = (1 - d1 * d2) / (2 * T)
             term_color_2 = (d1 * r + d2 * sigma**2 / 2) / (sigma * np.sqrt(T))
             color = -gamma * (term_color_1 + term_color_2)


        return {
            'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta,
            'vega': vega, 'rho': rho, 'charm': charm, 'vanna': vanna,
            'volga': volga, 'speed': speed, 'zomma': zomma, 'color': color
        }

    def calculate_batch(self,
                       S: np.ndarray,
                       K: np.ndarray,
                       T: np.ndarray,
                       r: np.ndarray,
                       sigma: np.ndarray,
                       option_types: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized calculation for multiple options"""
        results = {
            'price': np.zeros_like(S),
            'delta': np.zeros_like(S),
            'gamma': np.zeros_like(S),
            'theta': np.zeros_like(S),
            'vega': np.zeros_like(S),
            'rho': np.zeros_like(S),
            'charm': np.zeros_like(S),
            'vanna': np.zeros_like(S),
            'volga': np.zeros_like(S)
        }
        
        for i in range(len(S)):
            result = self.calculate(
                S=float(S[i]),
                K=float(K[i]),
                T=float(T[i]),
                r=float(r[i]),
                sigma=float(sigma[i]),
                option_type=str(option_types[i])
            )
            
            for greek in results:
                results[greek][i] = result[greek]
                
        return results


class GreekEnergyFlow:
    """
    Greek Energy Flow Analysis with dynamic configuration and logging.
    Uses provided Delta/Gamma if available, calculates Vanna/Charm.
    """

    def __init__(self, config=None):
        """Initialize the Greek Energy Flow analyzer with configuration."""
        # Default configuration
        default_config = {
            "reset_factors": {
                "gammaFlip": 1.0,
                "vannaPeak": 0.8,
                "charmCrossover": 0.6,
                "openInterest": 0.5
            },
            "regime_thresholds": {
                "highVolatility": 0.3,
                "lowVolatility": 0.15,
                "strongBullish": 0.7,
                "strongBearish": -0.7,
                "neutralZone": 0.2
            },
            "regime_labels": {
                1: "Strong Bullish",
                2: "Bullish",
                3: "Neutral",
                4: "Bearish",
                5: "Strong Bearish",
                6: "Volatility Expansion",
                7: "Volatility Compression",
                8: "Vanna-Driven",
                9: "Charm-Dominated"
            },
            "price_projection": {
                "range_percent": 0.15,
                "steps": 30
            }
        }
        
        # Use provided config or default
        self.config = default_config
        if config:
            # Update default config with provided values
            for key, value in config.items():
                if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Extract specific configs for easier access
        self.regime_thresholds = self.config.get("regime_thresholds")
        self.regime_labels = self.config.get("regime_labels")
        
        logging.info("GreekEnergyFlow initialized with configuration thresholds and reset factors.")

    @staticmethod
    def parse_date(date_str):
        """Robust date parsing."""
        try:
            return dateutil_parser.parse(date_str)
        except Exception as e:
            logging.error(f"Unable to parse date: {date_str} - Error: {e}")
            raise ValueError(f"Unable to parse date: {date_str}")

    def generate_price_points(self, current_price):
        """Generate an array of price points for projections based on configuration."""
        proj_config = self.config.get('price_projection', {})
        range_percent = proj_config.get('range_percent', 0.15)
        steps = proj_config.get('steps', 30)
        min_price = current_price * (1 - range_percent)
        max_price = current_price * (1 + range_percent)
        if steps <= 0: steps = 1 # Avoid division by zero
        step_size = (max_price - min_price) / steps
        price_points = [min_price + i * step_size for i in range(steps + 1)]
        logging.info(f"Generated {len(price_points)} price points from {min_price:.2f} to {max_price:.2f}.")
        return price_points

    def analyze_greek_profiles(self, options_df, market_data):
        """
        Run the full Greek Energy Flow analysis using preprocessed data.

        Parameters:
        -----------
        options_df : pd.DataFrame
            Preprocessed DataFrame containing option chain data. Must include:
            'strike', 'expiration' (datetime), 'openInterest', 'type' (lower),
            'impliedVolatility' (decimal), 'delta', 'gamma'.
        market_data : dict
            Dictionary containing: 'currentPrice', 'historicalVolatility', 'riskFreeRate'.
            Optionally 'impliedVolatility' for fallback.
        """
        logging.info("Starting Greek profile analysis...")

        if not isinstance(options_df, pd.DataFrame):
             raise TypeError("options_df must be a pandas DataFrame")
        required_cols = ['strike', 'expiration', 'openInterest', 'type', 'impliedVolatility', 'delta', 'gamma']
        if not all(col in options_df.columns for col in required_cols):
            raise ValueError(f"options_df missing one or more required columns: {required_cols}")

        # --- Core Analysis Steps ---
        aggregated_greeks = self._aggregate_greeks(options_df, market_data)
        logging.info("Aggregated Greeks computed.")

        reset_points = self._detect_reset_points(aggregated_greeks, options_df, market_data)
        logging.info(f"Detected {len(reset_points)} reset point signal(s).")

        market_regime = self._classify_market_regime(aggregated_greeks, market_data)
        logging.info(f"Market Regime Classified: {market_regime.get('primary_label', 'N/A')}, {market_regime.get('secondary_label', 'N/A')}")

        energy_levels = self._generate_energy_levels(aggregated_greeks, reset_points, market_data)
        logging.info(f"Generated {len(energy_levels)} energy level(s).")

        vanna_projections = self._project_vanna(aggregated_greeks, market_data)
        charm_projections = self._project_charm(aggregated_greeks, market_data)
        logging.info("Vanna and Charm projections generated.")

        greek_anomalies = self._detect_greek_anomalies(aggregated_greeks, market_data)
        logging.info(f"Detected {len(greek_anomalies)} Greek anomalie(s).")

        logging.info("Greek profile analysis completed.")
        return {
            'reset_points': reset_points,
            'market_regime': market_regime,
            'energy_levels': energy_levels,
            'vanna_projections': vanna_projections,
            'charm_projections': charm_projections,
            'greek_anomalies': greek_anomalies,
            'aggregated_greeks': aggregated_greeks # Return for potential deeper dives
        }

    def _aggregate_greeks(self, options_df, market_data):
        """
        Aggregate Greeks across all option chains.
        Uses provided Delta/Gamma, calculates Vanna/Charm.
        """
        current_price = market_data['currentPrice']
        risk_free_rate = market_data['riskFreeRate']
        # Use provided historical vol or fallback market IV only if IV per option is missing
        fallback_volatility = market_data.get('impliedVolatility', market_data['historicalVolatility'])

        aggregated = {
            'total_gamma': 0.0, 'net_delta': 0.0, 'total_vanna': 0.0, 'total_charm': 0.0,
            'gamma_exposure': [], 'vanna_exposure': [], 'charm_exposure': [],
            'expiration_buckets': {}
        }
        current_date = datetime.now()

        for _, option in options_df.iterrows():
            # --- Extract data from the preprocessed row ---
            strike = option['strike']
            expiration_date = option['expiration'] # Already datetime object
            open_interest = option['openInterest']
            option_type = option['type'] # Already lowercase
            # Use provided IV, handle potential NaNs from cleaning
            implied_volatility = option['impliedVolatility']
            if pd.isna(implied_volatility) or implied_volatility <= 0:
                implied_volatility = fallback_volatility # Use market fallback
                # logging.warning(f"Using fallback IV {fallback_volatility:.2%} for strike {strike} exp {expiration_date.date()}")

            # Get provided Greeks from the row
            provided_delta = option['delta']
            provided_gamma = option['gamma']

            # Check if essential provided greeks are valid
            if pd.isna(provided_delta) or pd.isna(provided_gamma):
                # Calculate delta and gamma if not provided
                time_to_expiry = (expiration_date - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
                bs_calculator = BlackScholesModel()
                calculated_greeks = bs_calculator.calculate(
                    S=current_price,
                    K=strike,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=implied_volatility,
                    option_type=option_type
                )
                
                # Use calculated greeks if provided ones are invalid
                delta = calculated_greeks['delta'] if pd.isna(provided_delta) else provided_delta
                gamma = calculated_greeks['gamma'] if pd.isna(provided_gamma) else provided_gamma
                charm = calculated_greeks['charm']
                vanna = calculated_greeks['vanna']
            else:
                # Use provided greeks directly
                delta = provided_delta
                gamma = provided_gamma
                
                # Calculate vanna and charm using BS model since they're rarely provided
                time_to_expiry = (expiration_date - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
                bs_calculator = BlackScholesModel()
                calculated_greeks = bs_calculator.calculate(
                    S=current_price,
                    K=strike,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=implied_volatility,
                    option_type=option_type
                )
                charm = calculated_greeks['charm']
                vanna = calculated_greeks['vanna']

            # Add to aggregated totals
            aggregated['total_gamma'] += gamma * open_interest
            aggregated['net_delta'] += delta * open_interest
            aggregated['total_vanna'] += vanna * open_interest
            aggregated['total_charm'] += charm * open_interest

            # Add to expiration buckets
            expiration_key = expiration_date.strftime('%Y-%m-%d')
            if expiration_key not in aggregated['expiration_buckets']:
                aggregated['expiration_buckets'][expiration_key] = {
                    'total_gamma': 0.0, 'net_delta': 0.0, 'total_vanna': 0.0, 'total_charm': 0.0
                }
            expiration_bucket = aggregated['expiration_buckets'][expiration_key]
            expiration_bucket['total_gamma'] += gamma * open_interest
            expiration_bucket['net_delta'] += delta * open_interest
            expiration_bucket['total_vanna'] += vanna * open_interest
            expiration_bucket['total_charm'] += charm * open_interest

            # Add to exposure lists
            aggregated['gamma_exposure'].append(gamma * open_interest)
            aggregated['vanna_exposure'].append(vanna * open_interest)
            aggregated['charm_exposure'].append(charm * open_interest)

        # Calculate net exposure
        aggregated['net_exposure'] = {
            'gamma': aggregated['total_gamma'],
            'delta': aggregated['net_delta'],
            'vanna': aggregated['total_vanna'],
            'charm': aggregated['total_charm']
        }

        return aggregated

    def _detect_reset_points(self, aggregated_greeks, options_df, market_data):
        """
        Detect potential price reset points based on Greek profiles.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            options_df (DataFrame): Options chain data
            market_data (dict): Market data including current price
        
        Returns:
            list: List of detected reset points with price and description
        """
        reset_points = []
        current_price = market_data.get("currentPrice", 0)
        
        # Extract aggregated Greeks
        total_gamma = aggregated_greeks.get("total_gamma", 0)
        total_vanna = aggregated_greeks.get("total_vanna", 0)
        total_charm = aggregated_greeks.get("total_charm", 0)
        
        # Check for gamma flip point (where gamma concentration changes sign)
        if abs(total_gamma) > 0.01:
            # Find strikes with highest gamma concentration
            if options_df is not None and not options_df.empty and "gamma" in options_df.columns:
                gamma_concentration = options_df.groupby("strike")["gamma"].sum()
                max_gamma_strike = gamma_concentration.idxmax()
                
                # If max gamma strike is different from current price, it's a potential reset point
                if abs(max_gamma_strike - current_price) > current_price * 0.01:
                    reset_points.append({
                        "price": float(max_gamma_strike),
                        "type": "gamma_flip",
                        "strength": abs(total_gamma) * self.config["reset_factors"]["gammaFlip"],
                        "description": "Gamma Flip Point"
                    })
        
        # Check for vanna peak (where vanna changes sign)
        if abs(total_vanna) > 0.01:
            # Simple approximation - vanna peak often occurs at key strikes
            if options_df is not None and not options_df.empty:
                # Find strikes with highest open interest as proxy for vanna concentration
                if "openInterest" in options_df.columns and "strike" in options_df.columns:
                    oi_concentration = options_df.groupby("strike")["openInterest"].sum()
                    max_oi_strike = oi_concentration.idxmax()
                    
                    # If max OI strike is different from current price, it's a potential reset point
                    if abs(max_oi_strike - current_price) > current_price * 0.01:
                        reset_points.append({
                            "price": float(max_oi_strike),
                            "type": "vanna_peak",
                            "strength": abs(total_vanna) * self.config["reset_factors"]["vannaPeak"],
                            "description": "Vanna Peak Point"
                        })
        
        # Check for charm crossover (where charm changes sign)
        if abs(total_charm) > 0.01:
            # Simple approximation for charm crossover
            if options_df is not None and not options_df.empty:
                # Use weighted average of strikes based on delta as proxy for charm crossover
                if "delta" in options_df.columns and "strike" in options_df.columns:
                    # Calculate absolute delta to weight both calls and puts
                    options_df["abs_delta"] = options_df["delta"].abs()
                    weighted_strikes = (options_df["strike"] * options_df["abs_delta"]).sum()
                    total_weight = options_df["abs_delta"].sum()
                    
                    if total_weight > 0:
                        charm_crossover = weighted_strikes / total_weight
                        
                        # If charm crossover is different from current price, it's a potential reset point
                        if abs(charm_crossover - current_price) > current_price * 0.01:
                            reset_points.append({
                                "price": float(charm_crossover),
                                "type": "charm_crossover",
                                "strength": abs(total_charm) * self.config["reset_factors"]["charmCrossover"],
                                "description": "Charm Crossover Point"
                            })
        
        # Sort reset points by strength (descending)
        reset_points.sort(key=lambda x: x["strength"], reverse=True)
        
        return reset_points

    def _classify_market_regime(self, aggregated_greeks, market_data):
        """
        Classify the current market regime based on Greek profiles.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            market_data (dict): Market data including current price
        
        Returns:
            dict: Market regime classification with labels and metrics
        """
        # Default thresholds if not provided in config
        default_thresholds = {
            'highVolatility': 0.3,
            'lowVolatility': 0.15,
            'strongBullish': 0.7,
            'strongBearish': -0.7,
            'neutralZone': 0.2
        }
        
        # Default regime labels
        default_labels = {
            'bullish': 'Bullish Trend',
            'bearish': 'Bearish Trend',
            'neutral': 'Neutral',
            'volatile': 'High Volatility',
            'vanna_driven': 'Vanna-Driven',
            'charm_driven': 'Charm-Dominated'
        }
        
        # Use configured thresholds or defaults
        thresholds = self.regime_thresholds or default_thresholds
        
        # Extract key metrics
        delta = aggregated_greeks.get('net_delta', 0)
        gamma = aggregated_greeks.get('total_gamma', 0)
        vanna = aggregated_greeks.get('total_vanna', 0)
        charm = aggregated_greeks.get('total_charm', 0)
        
        # Normalize delta for comparison
        total_oi = sum(aggregated_greeks.get('gamma_exposure', [0]))
        normalized_delta = delta / max(total_oi, 1)
        
        # Determine primary regime based on dominant Greek
        primary_regime = 3  # Default to neutral
        
        # Strong directional bias
        if abs(normalized_delta) > thresholds['strongBullish']:
            primary_regime = 1 if normalized_delta > 0 else 5  # Strong bullish or bearish
        # Moderate directional bias
        elif abs(normalized_delta) > thresholds['neutralZone']:
            primary_regime = 2 if normalized_delta > 0 else 4  # Moderate bullish or bearish
        # Volatility regime
        elif abs(gamma) > abs(vanna) and abs(gamma) > abs(charm):
            # High volatility vs low volatility
            hist_vol = market_data.get('historicalVolatility', 0.2)
            impl_vol = market_data.get('impliedVolatility', hist_vol)
            primary_regime = 6 if impl_vol > hist_vol * 1.1 else 7  # Expansion vs compression
        # Greek dominance
        elif abs(vanna) > abs(charm):
            primary_regime = 8  # Vanna-driven
        else:
            primary_regime = 9  # Charm-dominated
        
        # Determine volatility regime
        hist_vol = market_data.get('historicalVolatility', 0.2)
        impl_vol = market_data.get('impliedVolatility', 0.2)
        vol_regime = 'High' if impl_vol > hist_vol * 1.1 else 'Low' if impl_vol < hist_vol * 0.9 else 'Normal'
        
        # Construct the regime classification
        return {
            'primary_label': self.regime_labels.get(primary_regime, 'Unknown'),
            'secondary_label': 'Secondary Classification',  # Placeholder
            'volatility_regime': vol_regime,
            'dominant_greek': 'Delta' if primary_regime <= 5 else 
                             'Gamma' if primary_regime <= 7 else
                             'Vanna' if primary_regime == 8 else 'Charm',
            'greek_magnitudes': {
                'normalized_delta': normalized_delta,
                'total_gamma': gamma,
                'total_vanna': vanna,
                'total_charm': charm
            }
        }

    def _generate_energy_levels(self, aggregated_greeks, reset_points, market_data):
        """
        Generate energy levels based on Greek profiles and reset points.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            reset_points (list): List of detected reset points
            market_data (dict): Market data including current price
        
        Returns:
            list: List of energy levels with price, type, and strength
        """
        energy_levels = []
        current_price = market_data.get('currentPrice', 0)
        
        # Convert reset points to energy levels
        for point in reset_points:
            energy_levels.append({
                'price': point['price'],
                'type': f"{point['type']}_energy",
                'strength': point['strength'],
                'description': f"Energy level from {point['description']}",
                'direction': 'support' if point['price'] < current_price else 'resistance'
            })
        
        # Add key strikes with high open interest as potential energy levels
        if 'gamma_exposure' in aggregated_greeks and len(aggregated_greeks['gamma_exposure']) > 0:
            # This is a simplified approach - in a real implementation, we would analyze the
            # distribution of gamma exposure across strikes more thoroughly
            energy_levels.append({
                'price': current_price * 0.95,  # Example: 5% below current price
                'type': 'gamma_energy',
                'strength': 0.5,
                'description': 'Gamma energy level',
                'direction': 'support'
            })
            
            energy_levels.append({
                'price': current_price * 1.05,  # Example: 5% above current price
                'type': 'gamma_energy',
                'strength': 0.5,
                'description': 'Gamma energy level',
                'direction': 'resistance'
            })
        
        # Sort energy levels by strength
        energy_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return energy_levels

    def _project_vanna(self, aggregated_greeks, market_data):
        """
        Project vanna impact across price points.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            market_data (dict): Market data including current price
        
        Returns:
            dict: Vanna projections across price points
        """
        current_price = market_data.get('currentPrice', 0)
        price_points = self.generate_price_points(current_price)
        
        # Simple linear projection of vanna impact
        # In a real implementation, this would be more sophisticated
        total_vanna = aggregated_greeks.get('total_vanna', 0)
        vanna_slope = total_vanna / (len(price_points) / 2)
        
        vanna_projections = []
        for i, price in enumerate(price_points):
            # Simple linear model: vanna impact decreases as we move away from current price
            price_diff_pct = abs(price - current_price) / current_price
            vanna_impact = total_vanna * (1 - price_diff_pct)
            vanna_projections.append(vanna_impact)
        
        return {
            'price_points': price_points,
            'projections': {
                'vanna': vanna_projections
            }
        }

    def _project_charm(self, aggregated_greeks, market_data):
        """
        Project charm impact across price points.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            market_data (dict): Market data including current price
        
        Returns:
            dict: Charm projections across price points
        """
        current_price = market_data.get('currentPrice', 0)
        price_points = self.generate_price_points(current_price)
        
        # Simple linear projection of charm impact
        # In a real implementation, this would be more sophisticated
        total_charm = aggregated_greeks.get('total_charm', 0)
        
        charm_projections = []
        for price in price_points:
            # Simple model: charm impact is strongest at current price
            price_diff_pct = abs(price - current_price) / current_price
            charm_impact = total_charm * (1 - price_diff_pct * 2)  # Steeper falloff than vanna
            charm_projections.append(charm_impact)
        
        return {
            'price_points': price_points,
            'projections': {
                'charm': charm_projections
            }
        }

    def _detect_greek_anomalies(self, aggregated_greeks, market_data):
        """
        Detect anomalies in Greek profiles.
        
        Args:
            aggregated_greeks (dict): Dictionary of aggregated Greek values
            market_data (dict): Market data including current price
        
        Returns:
            list: List of detected anomalies with description and significance
        """
        anomalies = []
        
        # Extract key metrics
        delta = aggregated_greeks.get('net_delta', 0)
        gamma = aggregated_greeks.get('total_gamma', 0)
        vanna = aggregated_greeks.get('total_vanna', 0)
        charm = aggregated_greeks.get('total_charm', 0)
        
        # Check for unusually high gamma
        if abs(gamma) > 1.0:
            anomalies.append({
                'type': 'high_gamma',
                'description': 'Unusually high gamma exposure',
                'significance': min(abs(gamma) / 2, 1.0),
                'value': gamma
            })
        
        # Check for vanna-charm imbalance
        if abs(vanna) > 0 and abs(charm) > 0:
            vanna_charm_ratio = abs(vanna / charm) if abs(charm) > 0.001 else 100
            if vanna_charm_ratio > 5 or vanna_charm_ratio < 0.2:
                anomalies.append({
                    'type': 'vanna_charm_imbalance',
                    'description': 'Significant imbalance between vanna and charm',
                    'significance': min(abs(vanna_charm_ratio - 1) / 5, 1.0),
                    'value': vanna_charm_ratio
                })
        
        # Check for delta-gamma inconsistency
        if (delta > 0 and gamma < -0.5) or (delta < 0 and gamma > 0.5):
            anomalies.append({
                'type': 'delta_gamma_inconsistency',
                'description': 'Delta and gamma have inconsistent signs',
                'significance': min(abs(gamma), 1.0),
                'value': {'delta': delta, 'gamma': gamma}
            })
        
        return anomalies


# ==============================================================================
#  Utility Class for Analysis and Formatting/Plotting (Mostly unchanged)
# ==============================================================================
class GreekEnergyAnalyzer:
    """Utility class for running Greek Energy Flow analysis."""

    @staticmethod
    def analyze(options_df, market_data, config=None):
        """Run analysis using preprocessed data."""
        analyzer = GreekEnergyFlow(config)
        return analyzer.analyze_greek_profiles(options_df, market_data)

    @staticmethod
    def format_results(analysis_results):
        """Format the analysis results for readable output."""
        # This function takes the direct output of analyze_greek_profiles
        formatted = {}

        # Format Reset Points
        rps = analysis_results.get('reset_points', [])
        formatted['reset_points'] = [{
            'price': f"{point['price']:.2f}",
            'type': point['type'],
            'significance': f"{point['significance'] * 100:.1f}%",
            'time_frame': point['time_frame'],
            'factors': ', '.join(f"{k}: {v*100:.1f}%" for k, v in point.get('factors', {}).items())
        } for point in rps]

        # Format Market Regime
        mr = analysis_results.get('market_regime', {})
        if mr:
             formatted['market_regime'] = {
                 'primary': mr.get('primary_label', 'N/A'),
                 'secondary': mr.get('secondary_label', 'N/A'),
                 'volatility': f"{mr.get('volatility_regime', 'N/A')} Volatility",
                 'dominant_greek': mr.get('dominant_greek', 'N/A'),
                 'metrics': {
                     'delta': f"{mr.get('greek_magnitudes', {}).get('normalized_delta', np.nan):.3f}",
                     'gamma': f"{mr.get('greek_magnitudes', {}).get('total_gamma', np.nan):.5f}",
                     'vanna': f"{mr.get('greek_magnitudes', {}).get('total_vanna', np.nan):.5f}",
                     'charm': f"{mr.get('greek_magnitudes', {}).get('total_charm', np.nan):.5f}"
                 }
             }

        # Format Energy Levels
        els = analysis_results.get('energy_levels', [])
        formatted['energy_levels'] = [{
            'price': f"{level['price']:.2f}",
            'type': level['type'],
            'strength': f"{level['strength'] * 100:.1f}%",
            'direction': level['direction'],
            'components': level.get('components', 1)
        } for level in els]

        # Format Greek Anomalies
        gas = analysis_results.get('greek_anomalies', [])
        formatted['greek_anomalies'] = [{
            'type': anomaly['type'],
            'severity': f"{anomaly['severity'] * 100:.1f}%",
            'description': anomaly['description'],
            'implication': anomaly['implication']
        } for anomaly in gas]

        # Format Projections (Vanna)
        vp = analysis_results.get('vanna_projections', {})
        if vp.get('price_points'):
             formatted['vanna_projections'] = {
                 'price_points': [f"{p:.2f}" for p in vp['price_points']],
                 'projections': {
                     key: [f"{v:.6f}" for v in values]
                     for key, values in vp.get('projections', {}).items()
                 }
             }

        # Format Projections (Charm)
        cp = analysis_results.get('charm_projections', {})
        if cp.get('price_points'):
             formatted['charm_projections'] = {
                 'price_points': [f"{p:.2f}" for p in cp['price_points']],
                 'projections': {
                     key: [f"{v:.6f}" for v in values]
                     for key, values in cp.get('projections', {}).items()
                 }
             }

        return formatted

    @staticmethod
    def plot_results(analysis_results, market_data, config=None):
        """Create visualizations of the analysis results."""
        cfg = config if config is not None else get_config()
        plot_cfg = cfg.get('plot_defaults', {})
        figsize = plot_cfg.get('figsize', (12, 6))
        colors = plot_cfg.get('color_scheme', {'gamma': 'red', 'vanna': 'blue', 'charm': 'green', 'default': 'purple'})

        reset_points = analysis_results.get('reset_points', [])
        energy_levels = analysis_results.get('energy_levels', [])
        vanna_projections = analysis_results.get('vanna_projections', {})
        charm_projections = analysis_results.get('charm_projections', {})
        current_price = market_data['currentPrice']

        figures = {}

        # --- Plot 1: Reset Points and Energy Levels ---
        if reset_points or energy_levels:
            fig1, ax1 = plt.subplots(figsize=figsize)
            ax1.axvline(x=current_price, color='black', linestyle='--', lw=1.5, alpha=0.8, label='Current Price')

            # Plot Reset Points
            plotted_rp_labels = set()
            for point in reset_points[:10]: # Limit plotted points for clarity
                price = point['price']
                sig = point['significance']
                ptype = point['type']
                label = f"{ptype} ({sig*100:.0f}%)" if ptype not in plotted_rp_labels else None
                plotted_rp_labels.add(ptype) # Avoid duplicate legend entries for type

                color = colors.get(ptype.split()[0].lower(), colors['default']) # Color by primary Greek type
                ax1.axvline(x=price, color=color, alpha=min(0.9, sig + 0.2), lw=max(1, sig * 4), label=label)
                ax1.text(price, ax1.get_ylim()[1] * 0.95, f"{price:.2f}", rotation=90,
                         va='top', ha='center', fontsize=9, alpha=0.8)

            # Plot Energy Levels
            plotted_el_labels = set()
            for level in energy_levels[:8]: # Limit plotted levels
                 price = level['price']
                 strength = level['strength']
                 direction = level['direction']
                 level_type = level['type'].replace(" Energy", "").replace(" Cluster", "") # Simplify label
                 label = f"{level_type} ({strength*100:.0f}%)" if level_type not in plotted_el_labels else None
                 plotted_el_labels.add(level_type)

                 color = 'lightgreen' if "Support" in direction else 'salmon' if "Resistance" in direction else 'lightyellow'
                 width = current_price * 0.005 # Width of shaded region
                 ax1.axvspan(price - width/2, price + width/2, alpha=strength * 0.4, color=color, label=label)
                 # ax1.text(price, ax1.get_ylim()[1] * 0.85, f"{price:.2f}", rotation=90,
                 #          va='top', ha='right', fontsize=8, color='grey') # Add text for levels too?

            # Set limits dynamically based on plotted points/levels and current price
            all_prices = [current_price] + [p['price'] for p in reset_points] + [l['price'] for l in energy_levels]
            min_p, max_p = min(all_prices), max(all_prices)
            padding = (max_p - min_p) * 0.15
            ax1.set_xlim(min_p - padding, max_p + padding)

            ax1.set_title('Reset Points & Energy Levels', fontsize=14)
            ax1.set_xlabel('Price', fontsize=12)
            ax1.set_ylabel('Implied Significance / Level Strength', fontsize=12)
            ax1.tick_params(axis='y', which='both', left=False, labelleft=False) # Hide Y axis ticks

            # Create cleaner legend
            handles, labels = ax1.get_legend_handles_labels()
            unique_labels = {}
            for h, l in zip(handles, labels):
                 if l and l not in unique_labels: unique_labels[l] = h
            ax1.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
            figures['reset_points_levels'] = fig1

        # --- Plot 2: Vanna Projections ---
        if vanna_projections.get('price_points'):
            fig2, ax2 = plt.subplots(figsize=figsize)
            prices = vanna_projections['price_points']
            ax2.axvline(x=current_price, color='black', linestyle='--', lw=1.5, alpha=0.7, label='Current Price')
            ax2.axhline(y=0, color='grey', linestyle='-', alpha=0.5, lw=0.5)

            plot_count = 0
            max_plots = 6 # Limit expirations plotted for clarity
            # Plot Total first
            if 'Total' in vanna_projections['projections']:
                 total_vals = [float(v) for v            in vanna_projections['projections']['Total']]
                 ax2.plot(prices, total_vals, color=colors['vanna'], linewidth=2.5, label='Total Vanna')
                 plot_count += 1

            # Plot individual expirations
            sorted_expiries = sorted(vanna_projections['projections'].keys(), key=lambda x: (x != 'Total', x)) # Sort, keep Total first if```python
            # Plot individual expirations
            sorted_expiries = sorted(vanna_projections['projections'].keys(), key=lambda x: (x != 'Total', x)) # Sort, keep Total first if present
            for expiry in sorted_expiries:
                 if expiry != 'Total' and plot_count < max_plots:
                     vals = [float(v) for v in vanna_projections['projections'][expiry]]
                     ax2.plot(prices, vals, linestyle='--', alpha=0.7, label=f'Vanna {expiry}')
                     plot_count += 1

            ax2.set_xlim(min(prices), max(prices))
            ax2.set_title('Vanna Projections', fontsize=14)
            ax2.set_xlabel('Price', fontsize=12)
            ax2.set_ylabel('Aggregated Vanna', fontsize=12)
            ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            ax2.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0,0.85, 1])
            figures['vanna_projections'] = fig2

        # --- Plot 3: Charm Projections ---
        if charm_projections.get('price_points'):
            fig3, ax3 = plt.subplots(figsize=figsize)
            prices = charm_projections['price_points']
            ax3.axvline(x=current_price, color='black', linestyle='--', lw=1.5, alpha=0.7, label='Current Price')
            ax3.axhline(y=0, color='grey', linestyle='-', alpha=0.5, lw=0.5)

            plot_count = 0
            # Plot Total first
            if 'Total' in charm_projections['projections']:
                 total_vals = [float(v) for v in charm_projections['projections']['Total']]
                 ax3.plot(prices, total_vals, color=colors['charm'], linewidth=2.5, label='Total Charm')
                 plot_count += 1

            # Plot individual expirations
            sorted_expiries = sorted(charm_projections['projections'].keys(), key=lambda x: (x != 'Total', x))
            for expiry in sorted_expiries:
                if expiry != 'Total' and plot_count < max_plots:
                     vals = [float(v) for v in charm_projections['projections'][expiry]]
                     ax3.plot(prices, vals, linestyle='--', alpha=0.7, label=f'Charm {expiry}')
                     plot_count += 1

            ax3.set_xlim(min(prices), max(prices))
            ax3.set_title('Charm Projections', fontsize=14)
            ax3.set_xlabel('Price', fontsize=12)
            ax3.set_ylabel('Aggregated Charm', fontsize=12)
            ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            ax3.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            figures['charm_projections'] = fig3

        return figures


# ==============================================================================
#  Data Loading and Preprocessing Functions
# ==============================================================================

def extract_expiry_from_filename(filename):
    """Extracts YYYY-MM-DD expiration date from Barchart filename pattern."""
    # Pattern: *-exp-YYYY-MM-DD-*
    match = re.search(r'-exp-(\d{4}-\d{2}-\d{2})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        except ValueError:
            logging.error(f"Invalid date format found in filename: {filename}")
            return None
    else:
        logging.error(f"Could not find expiration date pattern in filename: {filename}")
        return None

def clean_barchart_iv(iv_str):
    """Cleans IV string (e.g., '117.00%') to decimal (e.g., 1.17)."""
    if isinstance(iv_str, (int, float)): # Already numeric
         # Check if it looks like a percentage (e.g., > 1) and convert, otherwise assume decimal
         return iv_str / 100.0 if iv_str > 1.0 else iv_str
    if isinstance(iv_str, str):
        try:
            # Remove % and convert
            return float(iv_str.replace('%', '').strip()) / 100.0
        except ValueError:
            # Handle cases like 'N/A' or other non-numeric strings
            return np.nan
    return np.nan # Handle other types (None, etc.)

def load_and_preprocess_options(filename, config=None):
    """Loads and preprocesses Barchart options CSV data."""
    cfg = config if config is not None else get_config()
    logging.info(f"Preprocessing options file: {filename}")
    handling_cfg = cfg.get('data_handling', {})
    footer_rows = handling_cfg.get('options_footer_rows', 1)

    # 1. Extract Expiry Date
    expiration_date = extract_expiry_from_filename(filename)
    if expiration_date is None:
        raise ValueError(f"Failed to extract expiration date from {filename}")

    # 2. Read CSV
    try:
        df = pd.read_csv(filename,
                         thousands=',',        # Handles commas in numbers like Open Int
                         skipfooter=footer_rows, # Skip footer lines
                         engine='python')      # Often needed with skipfooter
    except FileNotFoundError:
        logging.error(f"Options file not found: {filename}")
        raise
    except Exception as e:
        logging.error(f"Error reading options CSV '{filename}': {e}")
        raise

    # 3. Rename columns for consistency (adjust if CSV headers change)
    rename_map = {
        'Strike': 'strike',
        'Type': 'type',
        '"Open Int"': 'openInterest', # Header might have quotes
        'Open Int': 'openInterest',   # Header might not have quotes
        'IV': 'iv_string',            # Keep original IV string for reference
        'Delta': 'delta',
        'Gamma': 'gamma',
        'Theta': 'theta', # Keep other provided greeks if needed later
        'Vega': 'vega',
        'Rho': 'rho',
        'Volume': 'volume',
        'Last': 'last_price'
    }
    # Only rename columns that actually exist in the DataFrame
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Check if essential columns exist after rename attempt
    required_renamed = ['strike', 'type', 'openInterest', 'iv_string', 'delta', 'gamma']
    missing_renamed = [col for col in required_renamed if col not in df.columns]
    if missing_renamed:
        raise KeyError(f"Essential columns missing after rename attempt: {missing_renamed}. Check rename_map and CSV headers.")

    # 4. Add Expiration Column
    df['expiration'] = expiration_date

    # 5. Clean Data Types and Values
    df['type'] = df['type'].str.lower()
    df['impliedVolatility'] = df['iv_string'].apply(clean_barchart_iv)

    # Convert relevant columns to numeric, coercing errors to NaN
    numeric_cols = ['strike', 'openInterest', 'delta', 'gamma', 'impliedVolatility', 'last_price', 'volume', 'theta', 'vega', 'rho']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Filter out invalid rows
    initial_rows = len(df)
    df.dropna(subset=['strike', 'expiration', 'openInterest', 'type', 'impliedVolatility', 'delta', 'gamma'], inplace=True)
    # Optionally filter based on minimum OI or volume
    # df = df[df['openInterest'] > 0]
    final_rows = len(df)
    if initial_rows > final_rows:
         logging.warning(f"Dropped {initial_rows - final_rows} rows during preprocessing due to NaNs in essential columns.")

    logging.info(f"Finished preprocessing. {final_rows} valid options rows loaded.")
    return df

def calculate_historical_volatility(price_series, window):
    """Calculates annualized historical volatility from a price series."""
    if len(price_series) < window + 1:
        logging.warning(f"Not enough price data ({len(price_series)}) for HV window ({window}). Returning NaN.")
        return np.nan
    log_returns = np.log(price_series / price_series.shift(-1)) # Calculate log returns (make sure series is sorted old to new if using shift(1))
                                                              # Assuming newest first, shift(-1) gets previous day
    # Calculate rolling std dev of log returns
    rolling_std = log_returns.rolling(window=window).std()
    # Annualize (using 252 trading days)
    annualized_hv = rolling_std * np.sqrt(252)
    # Return the most recent calculated HV value
    return annualized_hv.iloc[window-1] # Adjust index based on rolling calculation alignment

def prepare_market_data(price_hist_filename, risk_free_rate, options_df, config=None):
    """Prepares the market_data dictionary."""
    cfg = config if config is not None else get_config()
    logging.info(f"Preparing market data using: {price_hist_filename}")
    handling_cfg = cfg.get('data_handling', {})
    footer_rows = handling_cfg.get('price_footer_rows', 1)
    hv_window = handling_cfg.get('hv_window', 20)

    # 1. Read Price History
    try:
        price_df = pd.read_csv(price_hist_filename,
                               skipfooter=footer_rows,
                               engine='python',
                               parse_dates=['Time']) # Assuming 'Time' column holds date
        price_df.sort_values('Time', ascending=False, inplace=True) # Ensure newest is first
    except FileNotFoundError:
        logging.error(f"Price history file not found: {price_hist_filename}")
        raise
    except Exception as e:
        logging.error(f"Error reading price history CSV '{price_hist_filename}': {e}")
        raise

    if price_df.empty:
        raise ValueError("Price history data is empty.")

    # 2. Get Current Price
    current_price = price_df['Last'].iloc[0]

    # 3. Calculate Historical Volatility
    # Need prices sorted oldest to newest for standard calculation
    price_series_for_hv = price_df.sort_values('Time', ascending=True)['Last']
    historical_vol = calculate_historical_volatility(price_series_for_hv, window=hv_window)
    if pd.isna(historical_vol):
        logging.warning("Historical Volatility calculation failed. Using fallback 0.20")
        historical_vol = 0.20 # Default fallback

    # 4. Calculate Fallback IV (e.g., average from options chain)
    fallback_iv = options_df['impliedVolatility'].mean()
    if pd.isna(fallback_iv):
        logging.warning("Could not calculate average IV from options chain. Using HV as fallback IV.")
        fallback_iv = historical_vol

    market_data = {
        'currentPrice': current_price,
        'historicalVolatility': historical_vol,
        'riskFreeRate': risk_free_rate,
        'impliedVolatility': fallback_iv # Used if option row IV is invalid
    }
    logging.info(f"Market data prepared: Price={current_price:.2f}, HV={historical_vol:.2%}, FallbackIV={fallback_iv:.2%}, RF={risk_free_rate:.2%}")
    return market_data


# ==============================================================================
#  Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    logging.info("Running Greek Energy Flow Analysis Module Standalone.")

    # --- Configuration for Standalone Run ---
    # Define input file paths relative to the script location or use absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OPTIONS_CSV = os.path.join(SCRIPT_DIR, 'pltr-volatility-greeks-exp-2025-04-11-weekly-50-strikes-+_--04-10-2025.csv')
    PRICE_HIST_CSV = os.path.join(SCRIPT_DIR, 'pltr_price-history-04-10-2025.csv')
    # Add other price/overview CSVs if needed

    # Define Risk-Free Rate (replace with actual source if available)
    RISK_FREE_RATE = 0.01 # Example: 1%

    # Get configuration
    config = get_config()

    # --- Load and Preprocess Data ---
    try:
        # Load and clean options data
        options_data = load_and_preprocess_options(OPTIONS_CSV, config=config)

        # Prepare market data (requires options_data for fallback IV)
        market_data = prepare_market_data(PRICE_HIST_CSV, RISK_FREE_RATE, options_data, config=config)

    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        exit() # Exit if essential data loading fails

    # --- Run Analysis ---
    try:
        # Instantiate analyzer with configuration
        analyzer = GreekEnergyFlow(config=config)
        analysis_results = analyzer.analyze_greek_profiles(options_data, market_data)
        logging.info("Analysis results obtained.")

        # --- Output Results ---
        formatted_results = GreekEnergyAnalyzer.format_results(analysis_results)

        print("\n--- Market Regime ---")
        print(formatted_results.get('market_regime', {}))

        print("\n--- Top Reset Points (Signals) ---")
        for rp in formatted_results.get('reset_points', [])[:5]: # Show top 5
            print(rp)

        print("\n--- Top Energy Levels ---")
        for el in formatted_results.get('energy_levels', [])[:5]: # Show top 5
            print(el)

        print("\n--- Top Greek Anomalies ---")
        for ga in formatted_results.get('greek_anomalies', [])[:3]: # Show top 3
            print(ga)

        # --- Plot Results ---
        figures = GreekEnergyAnalyzer.plot_results(analysis_results, market_data, config=config)
        print(f"\nGenerated {len(figures)} plot(s). Displaying...")
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred during analysis or plotting: {e}", exc_info=True) # Log traceback
