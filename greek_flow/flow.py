"""
Greek Energy Flow Analysis Module

Provides comprehensive analysis of option Greeks with a focus on energy flow dynamics:
- Uses provided Delta and Gamma from input data
- Calculates Vanna and Charm using Black-Scholes derivatives and provided IV
- Reset point detection with multi-factor significance scoring
- Market regime classification based on Greek configurations
- Price level generation for likely energy concentration/release points
"""

import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
import json

# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Configuration ---
DEFAULT_CONFIG = {
    'reset_factors': {
        'gammaFlip': 0.35,
        'vannaPeak': 0.25,
        'charmCluster': 0.15,
        'timeDecay': 0.10
    },
    # Add API configuration
    'POLYGON_API_KEY': None,  # Will be set from config.py
    'API_VERSION': 'v2',
    'API_BASE_URL': 'https://api.polygon.io'
}


class BlackScholesModel:
    """
    Core Black-Scholes implementation with enhanced Greek calculations.
    Kept complete calculation, but only Vanna/Charm will be used downstream
    if Delta/Gamma are provided from input data.
    """
    
    @staticmethod
    def calculate(S, K, T, r, sigma, option_type):
        """
        Calculate option price and Greeks using Black-Scholes model.
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
        
        Returns:
        dict: Dictionary containing option price and Greeks
        """
        # Prevent division by zero or negative values
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            return {
                'price': np.nan, 'delta': np.nan, 'gamma': np.nan,
                'theta': np.nan, 'vega': np.nan, 'vanna': np.nan, 'charm': np.nan
            }
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = 0.5 * (1 + math.erf(d1 / np.sqrt(2)))
        N_d2 = 0.5 * (1 + math.erf(d2 / np.sqrt(2)))
        n_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        
        # Option price
        if option_type.lower() == 'call':
            price = S * N_d1 - K * np.exp(-r * T) * N_d2
            delta = N_d1
            theta = -S * sigma * n_d1 / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
        else:  # put
            price = K * np.exp(-r * T) * (1 - N_d2) - S * (1 - N_d1)
            delta = N_d1 - 1
            theta = -S * sigma * n_d1 / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - N_d2)
        
        # Common Greeks
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * n_d1 / 100  # Divided by 100 for 1% change
        
        # Higher-order Greeks
        vanna = -n_d1 * d2 / sigma
        charm = -n_d1 * (r / (sigma * np.sqrt(T)) - d2 / (2 * T))
        
        return {
            'price': price, 'delta': delta, 'gamma': gamma,
            'theta': theta, 'vega': vega, 'vanna': vanna, 'charm': charm
        }


class GreekEnergyFlow:
    """
    Greek Energy Flow Analysis with dynamic configuration and logging.
    Uses provided Delta/Gamma if available, calculates Vanna/Charm.
    """

    def __init__(self, config=None):
        from config import get_config, DEFAULT_CONFIG
        self.config = config if config is not None else get_config()
        self.regime_thresholds = self.config.get('regime_thresholds', {})
        self.reset_factors = self.config.get('reset_factors', {})
        self.regime_labels = {
            1: "Strong Bullish Trend", 2: "Moderate Bullish", 3: "Neutral Choppy",
            4: "Moderate Bearish", 5: "Strong Bearish Trend", 6: "High Volatility Expansion",
            7: "Low Volatility Compression", 8: "Vanna-Driven", 9: "Charm-Dominated"
        }
        logging.info("GreekEnergyFlow initialized with configuration thresholds and reset factors.")

    def analyze_greek_profiles(self, options_df, market_data):
        """
        Analyze option chain data to extract Greek energy flow patterns.
        
        Parameters:
        options_df (DataFrame): Options chain data with required columns
        market_data (dict): Market context data including current price, volatility, etc.
        
        Returns:
        dict: Analysis results including reset points, market regime, energy levels
        """
        # Validate inputs
        required_columns = ['strike', 'expiration', 'type', 'openInterest', 'impliedVolatility']
        if not all(col in options_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in options_df.columns]
            raise ValueError(f"Missing required columns in options data: {missing}")
        
        if not all(key in market_data for key in ['currentPrice', 'historicalVolatility', 'riskFreeRate']):
            missing = [key for key in ['currentPrice', 'historicalVolatility', 'riskFreeRate'] if key not in market_data]
            raise ValueError(f"Missing required market data: {missing}")
        
        logging.info(f"Analyzing Greek profiles for {len(options_df)} options contracts.")
        
        # --- Core Analysis Steps ---
        aggregated_greeks = self._aggregate_greeks(options_df, market_data)
        logging.info("Aggregated Greeks computed.")

        reset_points = self._detect_reset_points(aggregated_greeks, options_df, market_data)
        logging.info(f"Detected {len(reset_points)} reset point signal(s).")

        market_regime = self._classify_market_regime(aggregated_greeks, market_data)
        logging.info(f"Market Regime Classified: {market_regime.get('primary_label', 'N/A')}, {market_regime.get('secondary_label', 'N/A')}")

        # Save market regime to file for dashboard
        try:
            os.makedirs(os.path.join("results", "market_regime"), exist_ok=True)
            regime_file = os.path.join("results", "market_regime", "current_regime.json")
            with open(regime_file, 'w') as f:
                json.dump(market_regime, f, indent=2)
            logging.info(f"Market regime saved to {regime_file}")
        except Exception as e:
            logging.error(f"Failed to save market regime: {e}")

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
        
        # Alert if using fallback volatility
        if 'impliedVolatility' not in market_data:
            logging.warning(f"FALLBACK VOLATILITY USED: Using historical volatility {fallback_volatility:.2%} as fallback")

        # Initialize with zero values and empty collections
        aggregated = {
            'total_gamma': 0.0,
            'net_delta': 0.0,
            'total_vanna': 0.0,
            'total_charm': 0.0,
            'total_open_interest': 0,  # Track total OI for proper weighting
            'gamma_exposure': [],
            'vanna_exposure': [],
            'charm_exposure': [],
            'expiration_buckets': {}
        }
        
        current_date = datetime.now()
        
        # Process each option contract
        for _, row in options_df.iterrows():
            # Extract basic option data
            strike = row['strike']
            option_type = row['type'].lower()
            
            # Handle expiration date (could be string or datetime)
            if isinstance(row['expiration'], str):
                try:
                    expiration_date = pd.to_datetime(row['expiration'])
                except:
                    logging.warning(f"Could not parse expiration date: {row['expiration']}")
                    continue
            else:
                expiration_date = row['expiration']
            
            # Calculate time to expiry in years
            time_to_expiry = max((expiration_date - current_date).total_seconds() / (365.25 * 24 * 3600), 0.001)
            
            # Get open interest and implied volatility
            open_interest = row.get('openInterest', 0)
            implied_volatility = row.get('impliedVolatility', fallback_volatility)
            
            # Skip if no open interest
            if open_interest <= 0:
                continue
                
            # --- Use provided Delta/Gamma if available ---
            delta = row.get('delta', None)
            gamma = row.get('gamma', None)
            
            # If Delta/Gamma not provided, calculate them
            if delta is None or gamma is None:
                calculated_greeks = BlackScholesModel.calculate(
                    current_price, strike, time_to_expiry, risk_free_rate, implied_volatility, option_type
                )
                delta = calculated_greeks.get('delta', 0)
                gamma = calculated_greeks.get('gamma', 0)
            
            # --- Calculate Vanna and Charm ---
            # Perform full BSM calculation to get Vanna/Charm consistently
            all_calculated_greeks = BlackScholesModel.calculate(
                current_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                implied_volatility, # Use the specific option's IV
                option_type
            )
            calculated_vanna = all_calculated_greeks.get('vanna', np.nan)
            calculated_charm = all_calculated_greeks.get('charm', np.nan)

            # Check if calculation failed
            if pd.isna(calculated_vanna) or pd.isna(calculated_charm):
                 logging.warning(f"Skipping row due to NaN Vanna/Charm calculation: Strike {strike}, Exp {expiration_date.date()}, IV {implied_volatility:.2%}")
                 continue
            
            # --- Aggregate Greeks ---
            # Weight by open interest
            weighted_delta = delta * open_interest
            weighted_gamma = abs(gamma * open_interest)  # Use absolute gamma for total exposure
            weighted_vanna = calculated_vanna * open_interest
            weighted_charm = calculated_charm * open_interest
            
            # Update aggregated values
            aggregated['net_delta'] += weighted_delta
            aggregated['total_gamma'] += weighted_gamma
            aggregated['total_vanna'] += weighted_vanna
            aggregated['total_charm'] += weighted_charm
            aggregated['total_open_interest'] += open_interest
            
            # Store exposure by strike
            aggregated['gamma_exposure'].append((strike, weighted_gamma))
            aggregated['vanna_exposure'].append((strike, weighted_vanna))
            aggregated['charm_exposure'].append((strike, weighted_charm))
            
            # Group by expiration buckets
            exp_key = expiration_date.strftime('%Y-%m-%d')
            if exp_key not in aggregated['expiration_buckets']:
                aggregated['expiration_buckets'][exp_key] = {
                    'net_delta': 0, 'total_gamma': 0, 'total_vanna': 0, 'total_charm': 0,
                    'days_to_expiry': time_to_expiry * 365.25
                }
            
            bucket = aggregated['expiration_buckets'][exp_key]
            bucket['net_delta'] += weighted_delta
            bucket['total_gamma'] += weighted_gamma
            bucket['total_vanna'] += weighted_vanna
            bucket['total_charm'] += weighted_charm
        
        # Sort exposure arrays by strike
        aggregated['gamma_exposure'].sort(key=lambda x: x[0])
        aggregated['vanna_exposure'].sort(key=lambda x: x[0])
        aggregated['charm_exposure'].sort(key=lambda x: x[0])
        
        # Normalize by total open interest if available
        if aggregated['total_open_interest'] > 0:
            aggregated['net_delta_normalized'] = aggregated['net_delta'] / aggregated['total_open_interest']
            aggregated['total_gamma_normalized'] = aggregated['total_gamma'] / aggregated['total_open_interest']
            aggregated['total_vanna_normalized'] = aggregated['total_vanna'] / aggregated['total_open_interest']
            aggregated['total_charm_normalized'] = aggregated['total_charm'] / aggregated['total_open_interest']
        else:
            aggregated['net_delta_normalized'] = 0
            aggregated['total_gamma_normalized'] = 0
            aggregated['total_vanna_normalized'] = 0
            aggregated['total_charm_normalized'] = 0
        
        return aggregated

    def _detect_reset_points(self, aggregated_greeks, options_df, market_data):
        """
        Detect potential reset points based on Greek configurations.
        
        Parameters:
        aggregated_greeks (dict): Aggregated Greek values
        options_df (DataFrame): Options chain data
        market_data (dict): Market context data
        
        Returns:
        list: Reset points with price, type, significance, factors
        """
        reset_points = []
        current_price = market_data.get('currentPrice', 100.0)
        
        # Extract sorted exposure arrays
        gamma_exposure = aggregated_greeks['gamma_exposure']
        vanna_exposure = aggregated_greeks['vanna_exposure']
        charm_exposure = aggregated_greeks['charm_exposure']
        
        # Skip if insufficient data
        if not gamma_exposure or not vanna_exposure or not charm_exposure:
            return reset_points
        
        # Find gamma concentration points (local maxima)
        gamma_strikes = [x[0] for x in gamma_exposure]
        gamma_values = [x[1] for x in gamma_exposure]
        
        # Find local maxima in gamma exposure
        gamma_peaks = []
        for i in range(1, len(gamma_values) - 1):
            if gamma_values[i] > gamma_values[i-1] and gamma_values[i] > gamma_values[i+1]:
                gamma_peaks.append((gamma_strikes[i], gamma_values[i]))
        
        # Find vanna flip points (sign changes)
        vanna_strikes = [x[0] for x in vanna_exposure]
        vanna_values = [x[1] for x in vanna_exposure]
        
        vanna_flips = []
        for i in range(1, len(vanna_values)):
            if (vanna_values[i-1] < 0 and vanna_values[i] > 0) or (vanna_values[i-1] > 0 and vanna_values[i] < 0):
                # Interpolate the zero-crossing point
                if vanna_values[i] != vanna_values[i-1]:  # Avoid division by zero
                    zero_cross = vanna_strikes[i-1] + (vanna_strikes[i] - vanna_strikes[i-1]) * (0 - vanna_values[i-1]) / (vanna_values[i] - vanna_values[i-1])
                    vanna_flips.append((zero_cross, abs(vanna_values[i] - vanna_values[i-1])))
        
        # Find charm clusters (areas of high charm concentration)
        charm_strikes = [x[0] for x in charm_exposure]
        charm_values = [x[1] for x in charm_exposure]
        
        charm_clusters = []
        for i in range(1, len(charm_values) - 1):
            if abs(charm_values[i]) > abs(charm_values[i-1]) and abs(charm_values[i]) > abs(charm_values[i+1]):
                charm_clusters.append((charm_strikes[i], abs(charm_values[i])))
        
        # Generate reset points by combining signals
        all_potential_points = []
        
        # Add gamma peaks
        for strike, value in gamma_peaks:
            significance = self.reset_factors.get('gammaFlip', 0.35) * (value / max(gamma_values) if gamma_values else 0)
            all_potential_points.append({
                'price': strike,
                'significance': significance,
                'distance': abs(strike - current_price) / current_price,
                'type': 'Gamma Concentration',
                'value': value
            })
        
        # Add vanna flips
        for strike, value in vanna_flips:
            significance = self.reset_factors.get('vannaPeak', 0.25) * (value / max(abs(v) for _, v in vanna_exposure) if vanna_exposure else 0)
            all_potential_points.append({
                'price': strike,
                'significance': significance,
                'distance': abs(strike - current_price) / current_price,
                'type': 'Vanna Flip',
                'value': value
            })
        
        # Add charm clusters
        for strike, value in charm_clusters:
            significance = self.reset_factors.get('charmCluster', 0.15) * (value / max(abs(v) for _, v in charm_exposure) if charm_exposure else 0)
            all_potential_points.append({
                'price': strike,
                'significance': significance,
                'distance': abs(strike - current_price) / current_price,
                'type': 'Charm Cluster',
                'value': value
            })
        
        # Sort potential points by significance
        all_potential_points.sort(key=lambda x: x['significance'], reverse=True)
        
        # Filter out points with low significance
        threshold = self.reset_factors.get('timeDecay', 0.10)
        reset_points = [point for point in all_potential_points if point['significance'] >= threshold]
        
        # Ensure each reset point is a dictionary with required fields
        for i, point in enumerate(reset_points):
            if not isinstance(point, dict):
                # Convert to dictionary if it's not already
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    reset_points[i] = {
                        'price': point[0],
                        'significance': point[1],
                        'type': 'reset',
                        'factors': {}
                    }
                else:
                    # Skip invalid points
                    reset_points[i] = None
        
        # Filter out None values
        reset_points = [p for p in reset_points if p is not None]
        
        return reset_points

    def _classify_market_regime(self, aggregated_greeks, market_data):
        """
        Classify the current market regime based on Greek profiles.
        
        Returns a dict with regime classification and metrics.
        """
        # Default regime thresholds if not provided in config
        default_thresholds = {
            'bullish': 'deltaComponent',
            'bearish': 'deltaComponent',
            'neutral': 'gammaComponent',
            'volatile': 'gammaComponent',
            'vanna_driven': 'vannaComponent',
            'charm_driven': 'charmComponent'
        }
        
        # Use configured thresholds or defaults
        thresholds = self.regime_thresholds or default_thresholds
        
        # Extract key metrics
        delta = aggregated_greeks.get('normalized_delta', 0)
        gamma = aggregated_greeks.get('total_gamma', 0)
        vanna = aggregated_greeks.get('total_vanna', 0)
        charm = aggregated_greeks.get('total_charm', 0)
        
        # Determine primary regime based on dominant Greek
        primary_regime = 'neutral'  # Default
        if abs(delta) > abs(gamma) and abs(delta) > abs(vanna) and abs(delta) > abs(charm):
            primary_regime = 'bullish' if delta > 0 else 'bearish'
        elif abs(gamma) > abs(vanna) and abs(gamma) > abs(charm):
            primary_regime = 'volatile'
        elif abs(vanna) > abs(charm):
            primary_regime = 'vanna_driven'
        else:
            primary_regime = 'charm_driven'
        
        # Get the component key or use a default
        component_key = thresholds.get(primary_regime, 'deltaComponent')
        
        # Default regime labels if not available
        default_labels = {
            'bullish': 'Bullish Trend',
            'bearish': 'Bearish Trend',
            'neutral': 'Neutral Choppy',
            'volatile': 'High Volatility',
            'vanna_driven': 'Vanna-Driven',
            'charm_driven': 'Charm-Dominated'
        }
        
        # Determine volatility regime
        hist_vol = market_data.get('historicalVolatility', 0.2)
        impl_vol = market_data.get('impliedVolatility', 0.2)
        vol_regime = 'High' if impl_vol > hist_vol * 1.1 else 'Low' if impl_vol < hist_vol * 0.9 else 'Normal'
        
        # Construct the regime classification
        return {
            'primary_label': self.regime_labels.get(primary_regime, default_labels.get(primary_regime, 'Unknown')),
            'secondary_label': 'Secondary Classification',  # Placeholder
            'volatility_regime': vol_regime,
            'dominant_greek': primary_regime.capitalize(),
            'greek_magnitudes': {
                'normalized_delta': delta,
                'total_gamma': gamma,
                'total_vanna': vanna,
                'total_charm': charm
            }
        }

    def _generate_energy_levels(self, aggregated_greeks, reset_points, market_data):
        """
        Generate energy levels based on aggregated Greeks and reset points.
        
        Parameters:
        aggregated_greeks (dict): Aggregated Greek values
        reset_points (list): List of reset point dictionaries
        market_data (dict): Market context data
        
        Returns:
        list: Energy levels with price, type, strength, direction
        """
        energy_levels = []
        current_price = market_data.get('currentPrice', 100.0)
        
        # Process reset points as energy levels
        for reset_point in reset_points:
            # Extract price and significance from reset point dictionary
            price = reset_point.get('price', 0.0)
            significance = reset_point.get('significance', 0.0)
            
            # Skip if price is missing or zero
            if not price:
                continue
                
            # Determine direction based on price relative to current
            direction = "support" if price < current_price else "resistance"
            
            # Add to energy levels
            energy_levels.append({
                'price': price,
                'type': reset_point.get('type', 'reset'),
                'strength': significance,
                'direction': direction,
                'components': 1
            })
        
        # Add additional energy levels from Greek aggregations
        # ... rest of the method remains unchanged ...
        
        return energy_levels

    def _project_vanna(self, aggregated_greeks, market_data):
        """
        Project Vanna for different price levels.
        """
        current_price = market_data['currentPrice']
        historical_volatility = market_data['historicalVolatility']
        risk_free_rate = market_data['riskFreeRate']
        
        # Calculate key Vanna indicators
        vanna = aggregated_greeks['total_vanna_normalized']
        gamma = aggregated_greeks['total_gamma_normalized']
        
        # Project Vanna for different price levels
        vanna_projections = []
        for price in np.linspace(current_price * 0.8, current_price * 1.2, 100):
            distance = abs(price - current_price) / current_price
            projected_vanna = vanna * (1 - abs(gamma)) * distance
            vanna_projections.append({
                'price': price,
                'vanna': projected_vanna
            })
        
        return vanna_projections

    def _project_charm(self, aggregated_greeks, market_data):
        """
        Project Charm for different price levels.
        """
        current_price = market_data['currentPrice']
        historical_volatility = market_data['historicalVolatility']
        risk_free_rate = market_data['riskFreeRate']
        
        # Calculate key Charm indicators
        charm = aggregated_greeks['total_charm_normalized']
        gamma = aggregated_greeks['total_gamma_normalized']
        
        # Project Charm for different price levels
        charm_projections = []
        for price in np.linspace(current_price * 0.8, current_price * 1.2, 100):
            distance = abs(price - current_price) / current_price
            projected_charm = charm * (1 - abs(gamma)) * (1 - distance)
            charm_projections.append({
                'price': price,
                'charm': projected_charm
            })
        
        return charm_projections

    def _detect_greek_anomalies(self, aggregated_greeks, market_data):
        """
        Detect anomalies in Greek values.
        """
        current_price = market_data['currentPrice']
        historical_volatility = market_data['historicalVolatility']
        risk_free_rate = market_data['riskFreeRate']
        
        # Calculate key Greek indicators
        gamma = aggregated_greeks['total_gamma_normalized']
        delta = aggregated_greeks['net_delta_normalized']
        vanna = aggregated_greeks['total_vanna_normalized']
        charm = aggregated_greeks['total_charm_normalized']
        
        # Define anomaly thresholds
        anomaly_thresholds = {
            'gamma': 0.5,
            'delta': 0.3,
            'vanna': 0.2,
            'charm': 0.1
        }
        
        # Detect anomalies
        anomalies = []
        if abs(gamma) > anomaly_thresholds['gamma']:
            anomalies.append({
                'greek': 'Gamma',
                'value': gamma,
                'description': f"High Gamma ({gamma:.2f}) indicates potential energy concentration or release."
            })
        if abs(delta) > anomaly_thresholds['delta']:
            anomalies.append({
                'greek': 'Delta',
                'value': delta,
                'description': f"High Delta ({delta:.2f}) suggests strong directional movement."
            })
        if abs(vanna) > anomaly_thresholds['vanna']:
            anomalies.append({
                'greek': 'Vanna',
                'value': vanna,
                'description': f"High Vanna ({vanna:.2f}) indicates sensitivity to volatility changes."
            })
        if abs(charm) > anomaly_thresholds['charm']:
            anomalies.append({
                'greek': 'Charm',
                'value': charm,
                'description': f"High Charm ({charm:.2f}) suggests rapid changes in Delta with price movement."
            })
        
        return anomalies


class GreekEnergyAnalyzer:
    """Utility class for running Greek Energy Flow analysis."""

    @staticmethod
    def analyze(options_df, market_data, config=None):
        """Run analysis using preprocessed data."""
        analyzer = GreekEnergyFlow(config)
        return analyzer.analyze_greek_profiles(options_df, market_data)
        
    @staticmethod
    def format_results(analysis_results):
        """Format analysis results for display."""
        formatted = {}
        
        # Format market regime
        if "market_regime" in analysis_results:
            formatted["market_regime"] = analysis_results["market_regime"]
            
        # Format reset points
        if "reset_points" in analysis_results:
            formatted["reset_points"] = analysis_results["reset_points"]
            
        # Format energy levels
        if "energy_levels" in analysis_results:
            formatted["energy_levels"] = analysis_results["energy_levels"]
            
        return formatted
        
    @staticmethod
    def analyze_chain_energy(options_df, symbol):
        """
        Analyze the energy distribution across an options chain.
        
        Args:
            options_df: Options data DataFrame
            symbol: Stock symbol
            
        Returns:
            Dictionary with chain energy analysis results
        """
        # Implementation
        return {"symbol": symbol, "energy_distribution": {}}
