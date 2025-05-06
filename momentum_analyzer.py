import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyFlowAnalyzer:
    """
    EnergyFlowAnalyzer calculates momentum and energy flow metrics for price data.
    
    This class analyzes OHLCV (Open, High, Low, Close, Volume) data to determine
    the momentum direction and strength of price movements. It uses a concept of
    "energy flow" which combines price changes with volume data to determine market momentum.
    
    Attributes:
        ohlcv_data (DataFrame): Processed OHLCV data
        symbol (str): Symbol/identifier for the asset being analyzed
        energy_values (ndarray): Calculated energy values
        smooth_energy (ndarray): Smoothed energy values after applying Gaussian filter
        gradients (ndarray): Gradient of the smooth energy values
        inelasticity (float): Ratio of absolute price change to volume (elasticity measure)
        config (dict): Configuration parameters for the analyzer
    """
    
    def __init__(self, ohlcv_df, symbol, config=None):
        """
        Initialize the EnergyFlowAnalyzer with OHLCV data and symbol.
        
        Args:
            ohlcv_df (DataFrame): DataFrame containing OHLCV data with columns for
                                timestamp, open, high, low, close, and volume
            symbol (str): Symbol/identifier for the asset being analyzed
            config (dict, optional): Configuration parameters:
                - smoothing_sigma: Sigma value for Gaussian smoothing (default: 1.5)
                - lookback_period: Number of periods for recent metrics (default: 20)
                - strong_threshold_multiplier: Multiplier for "strong" classification (default: 1.0)
                - moderate_threshold_multiplier: Multiplier for "moderate" classification (default: 0.33)
        """
        self.symbol = symbol
        logger.info(f"DEBUG [{symbol}]: EnergyFlowAnalyzer initializing with {len(ohlcv_df)} rows")
        
        # Set configuration with defaults
        default_config = {
            'smoothing_sigma': 1.5,
            'lookback_period': 20,
            'strong_threshold_multiplier': 1.0,
            'moderate_threshold_multiplier': 0.33
        }
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Log the input data structure
        logger.info(f"DEBUG [{symbol}]: Input columns: {ohlcv_df.columns.tolist()}")
        logger.info(f"DEBUG [{symbol}]: Data types: {dict(ohlcv_df.dtypes)}")
        logger.info(f"DEBUG [{symbol}]: First row: {ohlcv_df.iloc[0].to_dict()}")
        
        # Make a copy to avoid modifying the original data
        self.ohlcv_data = ohlcv_df.copy()
        
        # Initialize other attributes
        self.energy_values = None
        self.smooth_energy = None
        self.gradients = None
        self.inelasticity = None
        
        # Preprocess the data
        self._preprocess_data()
        
        logger.info(f"DEBUG [{symbol}]: Initialization successful with {len(self.ohlcv_data)} valid rows")
        logger.info(f"Initialized EnergyFlowAnalyzer for {symbol} with {len(self.ohlcv_data)} rows.")
    
    def _preprocess_data(self):
        """
        Preprocess the OHLCV data: convert types, handle missing values, etc.
        """
        # Process timestamp column
        logger.info(f"DEBUG [{self.symbol}]: Converting timestamp column. Original type: {self.ohlcv_data['timestamp'].dtype}")
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.ohlcv_data['timestamp']):
            self.ohlcv_data['timestamp'] = pd.to_datetime(self.ohlcv_data['timestamp'])
        logger.info(f"DEBUG [{self.symbol}]: After timestamp conversion. Type: {self.ohlcv_data['timestamp'].dtype}")
        logger.info(f"DEBUG [{self.symbol}]: NaN count in timestamp: {self.ohlcv_data['timestamp'].isna().sum()}")
        
        # Process numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            logger.info(f"DEBUG [{self.symbol}]: Converting {col}, original type: {self.ohlcv_data[col].dtype}")
            # Ensure numeric types
            self.ohlcv_data[col] = pd.to_numeric(self.ohlcv_data[col], errors='coerce')
            logger.info(f"DEBUG [{self.symbol}]: After {col} conversion. NaN count: {self.ohlcv_data[col].isna().sum()}")
        
        # Drop rows with NaN in all required columns
        original_rows = len(self.ohlcv_data)
        self.ohlcv_data = self.ohlcv_data.dropna(subset=['timestamp', 'close', 'volume'])
        remaining_rows = len(self.ohlcv_data)
        
        logger.info(f"DEBUG [{self.symbol}]: After dropna, remaining rows: {remaining_rows}")
        
        # Fill remaining NaN values (if any)
        # Note: Using ffill() and bfill() instead of the deprecated method parameter
        self.ohlcv_data = self.ohlcv_data.ffill()
        self.ohlcv_data = self.ohlcv_data.bfill()
        
        # Log a warning if rows were dropped during preprocessing
        if original_rows > remaining_rows:
            dropped_rows = original_rows - remaining_rows
            logger.warning(f"{self.symbol}: Dropped {dropped_rows} rows with invalid OHLCV data.")
        
        logger.info(f"DEBUG [{self.symbol}]: Initialization successful with {remaining_rows} valid rows")
        logger.info(f"Initialized EnergyFlowAnalyzer for {self.symbol} with {remaining_rows} rows.")
    
    def calculate_energy_metrics(self):
        """
        Calculate energy flow metrics based on price changes and volume data.
        
        This method calculates several key metrics:
        1. Raw energy values (price change * volume) - measures the force behind price movements
        2. Smoothed energy values using a Gaussian filter - reduces noise in the raw energy signal
        3. Gradients of the smooth energy - measures the rate of change in energy flow
        4. Inelasticity ratio (average of abs(price change) / volume) - measures price sensitivity
        
        The energy metrics provide insights into market momentum by combining price action with volume,
        which helps identify the strength and direction of market movements.
        
        Returns:
            dict: A dictionary containing calculated energy metrics:
                - 'energy_values': Raw energy values
                - 'smooth_energy': Smoothed energy values
                - 'gradients': Energy gradient values
                - 'inelasticity': Inelasticity ratio
        """
        # Sort data by timestamp to ensure correct sequencing
        self.ohlcv_data = self.ohlcv_data.sort_values('timestamp')
        logger.info(f"DEBUG [{self.symbol}]: Sorting data by timestamp, current shape: {self.ohlcv_data.shape}")
        
        # Calculate price change
        logger.info(f"DEBUG [{self.symbol}]: Calculating price change")
        self.ohlcv_data['price_change'] = self.ohlcv_data['close'].diff()
        
        # Log some samples for debugging
        logger.info(f"DEBUG [{self.symbol}]: Close price sample: {self.ohlcv_data['close'].head(3).tolist()}")
        logger.info(f"DEBUG [{self.symbol}]: Price change sample: {self.ohlcv_data['price_change'].head(3).tolist()}")
        logger.info(f"DEBUG [{self.symbol}]: NaN count in price_change: {self.ohlcv_data['price_change'].isna().sum()}")
        
        # Process volume data
        logger.info(f"DEBUG [{self.symbol}]: Volume before clip, sample: {self.ohlcv_data['volume'].head(3).tolist()}")
        logger.info(f"DEBUG [{self.symbol}]: Any negative volume values: {(self.ohlcv_data['volume'] < 0).any()}")
        
        # Ensure volume values are positive
        self.ohlcv_data['volume'] = np.clip(self.ohlcv_data['volume'], 1, None)
        logger.info(f"DEBUG [{self.symbol}]: Volume after clip, sample: {self.ohlcv_data['volume'].head(3).tolist()}")
        
        # Calculate energy values (price_change * volume)
        logger.info(f"DEBUG [{self.symbol}]: Calculating energy values")
        self.ohlcv_data['energy'] = self.ohlcv_data['price_change'] * self.ohlcv_data['volume']
        self.ohlcv_data['energy'] = self.ohlcv_data['energy'].fillna(0)
        
        # Log energy samples
        logger.info(f"DEBUG [{self.symbol}]: Energy values sample: {self.ohlcv_data['energy'].head(3).tolist()}")
        logger.info(f"DEBUG [{self.symbol}]: NaN count in energy: {self.ohlcv_data['energy'].isna().sum()}")
        logger.info(f"DEBUG [{self.symbol}]: Inf count in energy: {np.isinf(self.ohlcv_data['energy']).sum()}")
        
        # Extract energy values as numpy array for processing
        self.energy_values = self.ohlcv_data['energy'].values
        logger.info(f"DEBUG [{self.symbol}]: Raw energy array shape: {self.energy_values.shape}")
        logger.info(f"DEBUG [{self.symbol}]: Raw energy sample: {self.energy_values[:3]}")
        
        # Apply Gaussian smoothing to reduce noise - use config parameter
        smoothing_sigma = self.config['smoothing_sigma']
        logger.info(f"DEBUG [{self.symbol}]: Applying gaussian filter with sigma={smoothing_sigma}")
        self.smooth_energy = gaussian_filter1d(self.energy_values, sigma=smoothing_sigma)
        logger.info(f"DEBUG [{self.symbol}]: Gaussian filter successful, result shape: {self.smooth_energy.shape}")
        logger.info(f"DEBUG [{self.symbol}]: Smoothed energy sample: {self.smooth_energy[:3]}")
        
        # Calculate the gradient (rate of change of energy)
        logger.info(f"DEBUG [{self.symbol}]: Calculating gradient from smooth energy, length: {len(self.smooth_energy)}")
        self.gradients = np.gradient(self.smooth_energy)
        logger.info(f"DEBUG [{self.symbol}]: Gradient calculation successful, shape: {self.gradients.shape}")
        logger.info(f"DEBUG [{self.symbol}]: Gradient sample: {self.gradients[:3]} ")
        logger.info(f"DEBUG [{self.symbol}]: NaN in gradients: {np.isnan(self.gradients).any()}")
        logger.info(f"DEBUG [{self.symbol}]: Last gradient value: {self.gradients[-1]}")
        
        # Calculate inelasticity ratio (ratio of price change to volume)
        logger.info(f"DEBUG [{self.symbol}]: Calculating inelasticity ratio")
        price_changes = self.ohlcv_data['price_change'].abs().fillna(0)
        volumes = self.ohlcv_data['volume'].fillna(1)
        self.inelasticity = np.mean(price_changes / volumes)
        logger.info(f"DEBUG [{self.symbol}]: Inelasticity calculation successful, avg value: {self.inelasticity:.3f}")
        
        logger.info(f"{self.symbol}: Successfully calculated energy metrics with {len(self.smooth_energy)} values")
        
        # Return the calculated metrics as a dictionary
        return {
            'energy_values': self.energy_values,
            'smooth_energy': self.smooth_energy,
            'gradients': self.gradients,
            'inelasticity': self.inelasticity
        }
    
    def get_current_momentum_state(self):
        """
        Get the current momentum state (direction and strength).
        
        This method analyzes the energy flow metrics to determine the current momentum
        state of the asset. It uses the gradient of the smoothed energy values to
        determine both the direction and strength of the momentum.
        
        The direction is determined by the sign of the latest gradient:
        - Positive gradient → Positive momentum
        - Negative gradient → Negative momentum
        
        The strength is determined by comparing the magnitude of the latest gradient
        to a dynamic threshold based on the standard deviation of recent gradients:
        - |gradient| > threshold → Strong
        - threshold/3 < |gradient| < threshold → Moderate
        - |gradient| < threshold/3 → Weak/Flat
        
        Returns:
            tuple: A tuple containing (direction, state) where:
                direction (str): 'Positive', 'Negative', or 'Flat'
                state (str): Describes momentum strength: 'Strong', 'Moderate', or 'Weak/Flat'
        """
        if self.gradients is None:
            # Calculate energy metrics if not already done
            self.calculate_energy_metrics()
        
        # Handle edge case: empty or insufficient data
        if len(self.gradients) < 2:
            logger.warning(f"{self.symbol}: Insufficient data for momentum analysis")
            return "Insufficient Data", "Insufficient Data"
        
        # Get the latest gradient value (momentum)
        latest_gradient = self.gradients[-1]
        logger.info(f"DEBUG [{self.symbol}]: Latest gradient value: {latest_gradient:.6f}")
        
        # Handle edge case: NaN gradient
        if np.isnan(latest_gradient):
            logger.warning(f"{self.symbol}: NaN gradient detected, using fallback")
            # Try using the second-to-last gradient if available
            if len(self.gradients) > 2 and not np.isnan(self.gradients[-2]):
                latest_gradient = self.gradients[-2]
            else:
                return "Indeterminate", "NaN Gradient"
        
        # Use the standard deviation of recent gradients as a dynamic threshold
        # This adapts to the volatility of the specific asset
        lookback_period = self.config['lookback_period']
        recent_gradients = self.gradients[-lookback_period:] if len(self.gradients) >= lookback_period else self.gradients
        
        # Handle edge case: all recent gradients are the same
        if np.all(recent_gradients == recent_gradients[0]):
            std_dev = 0.001  # Use a small default value
        else:
            std_dev = np.std(recent_gradients)
        
        logger.info(f"DEBUG [{self.symbol}]: Using std dev of recent gradients: {std_dev:.6f}")
        
        # Threshold for determining strength - based on standard deviation of gradients
        threshold = max(std_dev, 0.001)  # Ensure minimum threshold
        logger.info(f"DEBUG [{self.symbol}]: Strength threshold: {threshold:.6f}")
        
        # Check for flat market conditions
        # 1. Look at price volatility
        recent_prices = self.ohlcv_data['close'].values[-lookback_period:] if len(self.ohlcv_data) >= lookback_period else self.ohlcv_data['close'].values
        price_volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        # 2. Look at recent volume
        recent_volumes = self.ohlcv_data['volume'].values[-lookback_period:] if len(self.ohlcv_data) >= lookback_period else self.ohlcv_data['volume'].values
        volume_volatility = np.std(recent_volumes) / np.mean(recent_volumes)
        
        # 3. Check if the gradient is very small relative to the price
        gradient_to_price_ratio = abs(latest_gradient) / np.mean(recent_prices)
        
        # Log these metrics for debugging
        logger.info(f"DEBUG [{self.symbol}]: Price volatility: {price_volatility:.6f}")
        logger.info(f"DEBUG [{self.symbol}]: Volume volatility: {volume_volatility:.6f}")
        logger.info(f"DEBUG [{self.symbol}]: Gradient to price ratio: {gradient_to_price_ratio:.6f}")
        
        # Get threshold multipliers from config
        strong_multiplier = self.config['strong_threshold_multiplier']
        moderate_multiplier = self.config['moderate_threshold_multiplier']
        
        # Determine strength based on gradient magnitude relative to threshold
        # and additional flat market indicators
        is_flat_market = (
            price_volatility < 0.01 or  # Very low price volatility
            (abs(latest_gradient) < threshold * moderate_multiplier and volume_volatility < 0.2) or  # Low gradient and stable volume
            gradient_to_price_ratio < 0.0001  # Extremely small gradient relative to price
        )
        
        if is_flat_market:
            strength = "Weak/Flat"
        elif abs(latest_gradient) > threshold * strong_multiplier:
            strength = "Strong"
        elif abs(latest_gradient) > threshold * moderate_multiplier:
            strength = "Moderate"
        else:
            strength = "Weak/Flat"
        
        logger.info(f"DEBUG [{self.symbol}]: Determined strength: {strength}")
        
        # Determine direction based on gradient sign
        if latest_gradient > 0:
            direction = "Positive"
        else:
            direction = "Negative"
        
        logger.info(f"DEBUG [{self.symbol}]: Determined direction: {direction}")
        
        # Combine direction and strength for state
        state = f"{strength} {direction}"
        
        logger.info(f"{self.symbol}: Momentum state: {direction}, {strength} (gradient: {latest_gradient:.6f})")
        
        return direction, state

    def create_visualization(self, output_path=None, show_plot=False):
        """
        Create a visualization of the energy flow analysis.
        
        This method generates a comprehensive visualization of the energy flow metrics,
        including price, volume, energy values, and momentum indicators.
        
        Args:
            output_path (str, optional): Path to save the visualization image
            show_plot (bool, optional): Whether to display the plot (default: False)
            
        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.gridspec import GridSpec
            
            # Check if metrics have been calculated
            if self.smooth_energy is None:
                self.calculate_energy_metrics()
            
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
            
            # Price chart
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(self.ohlcv_data['timestamp'], self.ohlcv_data['close'], label='Close Price')
            ax1.set_title(f"{self.symbol} - Energy Flow Analysis")
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.bar(self.ohlcv_data['timestamp'], self.ohlcv_data['volume'], alpha=0.5, label='Volume')
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Energy flow chart
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(self.ohlcv_data['timestamp'], self.smooth_energy, label='Smooth Energy', color='green')
            ax3.set_ylabel('Energy Flow')
            ax3.grid(True, alpha=0.3)
            
            # Gradient chart
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            ax4.plot(self.ohlcv_data['timestamp'], self.gradients, label='Gradient', color='purple')
            ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax4.set_ylabel('Gradient')
            ax4.set_xlabel('Date')
            ax4.grid(True, alpha=0.3)
            
            # Format x-axis dates
            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.legend()
            
            # Add momentum state annotation
            direction, state = self.get_current_momentum_state()
            momentum_text = f"Momentum: {direction}, {state}"
            fig.text(0.5, 0.01, momentum_text, ha='center', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            
            # Save figure if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved energy flow visualization to {output_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            return fig, [ax1, ax2, ax3, ax4]
            
        except ImportError:
            logger.warning("Matplotlib not available, visualization skipped")
            return None, None
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    # Alias for backward compatibility
    visualize_energy_flow = create_visualization

    def export_results(self, output_path=None):
        """
        Export the energy flow analysis results to a JSON file.
        
        This method exports the comprehensive results of the energy flow analysis,
        including momentum state, metrics, and other relevant information.
        
        Args:
            output_path (str, optional): Path to save the results JSON file
                If not provided, a default path will be used
            
        Returns:
            str: Path to the saved results file, or None if export failed
        """
        try:
            import json
            from datetime import datetime
            import os
            
            # Check if metrics have been calculated
            if self.smooth_energy is None:
                self.calculate_energy_metrics()
                
            # Get momentum state
            direction, state = self.get_current_momentum_state()
            
            # Create results dictionary
            results = {
                "symbol": self.symbol,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_points": len(self.ohlcv_data),
                "date_range": {
                    "start": self.ohlcv_data['timestamp'].min().strftime("%Y-%m-%d"),
                    "end": self.ohlcv_data['timestamp'].max().strftime("%Y-%m-%d")
                },
                "momentum": {
                    "direction": direction,
                    "state": state,
                    "latest_gradient": float(self.gradients[-1]),
                    "inelasticity": float(self.inelasticity)
                },
                "summary_stats": {
                    "mean_energy": float(np.mean(self.energy_values)),
                    "std_energy": float(np.std(self.energy_values)),
                    "mean_gradient": float(np.mean(self.gradients)),
                    "std_gradient": float(np.std(self.gradients))
                }
            }
            
            # Determine output path if not provided
            if output_path is None:
                output_dir = "results"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{self.symbol}_energy_flow_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Save results to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Exported energy flow analysis results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def detect_divergences(self, window_size=None):
        """
        Detect price-momentum divergences in the data.
        
        This method identifies bullish and bearish divergences between price action
        and momentum indicators, which can signal potential trend reversals.
        
        Args:
            window_size (int, optional): Size of the window to look for divergences
                If not provided, uses the lookback_period from config
            
        Returns:
            dict: Dictionary containing detected bullish and bearish divergences
        """
        # Check if metrics have been calculated
        if self.smooth_energy is None:
            self.calculate_energy_metrics()
        
        # Use provided window size or default from config
        if window_size is None:
            window_size = self.config.get('lookback_period', 20)
        
        # Ensure we have enough data
        if len(self.ohlcv_data) < window_size:
            logger.warning(f"Not enough data for divergence detection (need {window_size}, have {len(self.ohlcv_data)})")
            return {'bullish_divergences': [], 'bearish_divergences': []}
        
        # Get price and momentum data
        prices = self.ohlcv_data['close'].values
        momentum = self.smooth_energy
        
        # Initialize divergence lists
        bullish_divergences = []
        bearish_divergences = []
        
        # Look for local minima and maxima in the recent window
        for i in range(window_size, len(prices)):
            # Get window data
            price_window = prices[i-window_size:i+1]
            momentum_window = momentum[i-window_size:i+1]
            
            # Check for bullish divergence (price makes lower low, momentum makes higher low)
            if (price_window[-1] < min(price_window[:-1]) and 
                momentum_window[-1] > min(momentum_window[:-1])):
                
                bullish_divergences.append({
                    'index': i,
                    'timestamp': self.ohlcv_data['timestamp'].iloc[i],
                    'price': float(price_window[-1]),
                    'momentum': float(momentum_window[-1]),
                    'strength': float(momentum_window[-1] - min(momentum_window[:-1]))
                })
            
            # Check for bearish divergence (price makes higher high, momentum makes lower high)
            if (price_window[-1] > max(price_window[:-1]) and 
                momentum_window[-1] < max(momentum_window[:-1])):
                
                bearish_divergences.append({
                    'index': i,
                    'timestamp': self.ohlcv_data['timestamp'].iloc[i],
                    'price': float(price_window[-1]),
                    'momentum': float(momentum_window[-1]),
                    'strength': float(max(momentum_window[:-1]) - momentum_window[-1])
                })
        
        # Log the results
        logger.info(f"{self.symbol}: Detected {len(bullish_divergences)} bullish and {len(bearish_divergences)} bearish divergences")
        
        return {
            'bullish_divergences': bullish_divergences,
            'bearish_divergences': bearish_divergences
        }

    def detect_momentum_changes(self, threshold_multiplier=None):
        """
        Detect significant changes in momentum.
        
        This method identifies points where the momentum gradient changes significantly,
        which can signal potential trend changes or acceleration/deceleration of trends.
        
        Args:
            threshold_multiplier (float, optional): Multiplier for the threshold
                If not provided, uses the strong_threshold_multiplier from config
            
        Returns:
            list: List of detected momentum change points
        """
        # Check if metrics have been calculated
        if self.gradients is None:
            self.calculate_energy_metrics()
        
        # Use provided threshold or default from config
        if threshold_multiplier is None:
            threshold_multiplier = self.config.get('strong_threshold_multiplier', 1.0)
        
        # Get timestamps
        timestamps = self.ohlcv_data['timestamp'].values
        
        # Calculate gradient changes
        gradient_changes = np.diff(self.gradients)
        
        # Determine significance threshold
        threshold = np.std(gradient_changes) * threshold_multiplier
        
        # Find significant changes
        significant_changes = []
        for i in range(len(gradient_changes)):
            if abs(gradient_changes[i]) > threshold:
                if gradient_changes[i] > 0:
                    direction = "Accelerating Positive"
                else:
                    direction = "Accelerating Negative"
                
                significant_changes.append({
                    'timestamp': timestamps[i+1],
                    'direction': direction,
                    'magnitude': float(gradient_changes[i]),
                    'price': float(self.ohlcv_data['close'].values[i+1])
                })
        
        # Log the results
        logger.info(f"{self.symbol}: Detected {len(significant_changes)} significant momentum changes")
        
        return significant_changes

    def backtest_momentum_signals(self, holding_period=5):
        """
        Simple backtest of momentum signals
        
        This method performs a basic backtest of momentum signals by analyzing
        the performance of positive and negative momentum signals over a specified
        holding period.
        
        Args:
            holding_period (int): Number of periods to hold after a signal
            
        Returns:
            dict: Dictionary with backtest results
        """
        # Calculate metrics if not already done
        if self.gradients is None:
            self.calculate_energy_metrics()
        
        # Initialize results
        results = {
            'signals': [],
            'performance': {
                'positive_momentum': {'win_rate': 0, 'avg_return': 0, 'count': 0},
                'negative_momentum': {'win_rate': 0, 'avg_return': 0, 'count': 0}
            }
        }
        
        # Track positive and negative momentum signals
        pos_returns = []
        neg_returns = []
        
        # Analyze each potential signal (skip the last holding_period to have complete data)
        for i in range(len(self.gradients) - holding_period):
            # Skip the first few periods for warm-up
            if i < 5:
                continue
                
            # Get the current gradient value
            gradient = self.gradients[i]
            
            # Get the current price
            current_price = self.ohlcv_data['close'].values[i]
            
            # Get the future price (after holding period)
            future_price = self.ohlcv_data['close'].values[i + holding_period]
            
            # Calculate return
            future_return = (future_price / current_price - 1) * 100
            
            # Record signal
            signal = {
                'timestamp': self.ohlcv_data['timestamp'].values[i],
                'gradient': float(gradient),
                'price': float(current_price),
                'future_price': float(future_price),
                'return': float(future_return)
            }
            
            # Add to appropriate list
            if gradient > 0:
                signal['direction'] = 'Positive'
                pos_returns.append(future_return)
            else:
                signal['direction'] = 'Negative'
                neg_returns.append(future_return)
                
            results['signals'].append(signal)
        
        # Calculate statistics for positive momentum
        if pos_returns:
            results['performance']['positive_momentum']['count'] = len(pos_returns)
            results['performance']['positive_momentum']['avg_return'] = float(np.mean(pos_returns))
            results['performance']['positive_momentum']['win_rate'] = float(np.sum(np.array(pos_returns) > 0) / len(pos_returns))
        
        # Calculate statistics for negative momentum
        if neg_returns:
            results['performance']['negative_momentum']['count'] = len(neg_returns)
            results['performance']['negative_momentum']['avg_return'] = float(np.mean(neg_returns))
            results['performance']['negative_momentum']['win_rate'] = float(np.sum(np.array(neg_returns) < 0) / len(neg_returns))
        
        logger.info(f"{self.symbol}: Backtest completed with {len(results['signals'])} signals")
        
        return results

    def analyze_multiple_timeframes(self, resample_rules=['1D', '1W', '1M']):
        """
        Analyze momentum across multiple timeframes
        
        This method resamples the OHLCV data to different timeframes and analyzes
        the momentum in each timeframe, which can provide a more comprehensive
        view of market momentum across different time horizons.
        
        Args:
            resample_rules (list): List of pandas resample rules for different timeframes
            
        Returns:
            dict: Dictionary with results for each timeframe
        """
        results = {}
        
        # Analyze each timeframe
        for rule in resample_rules:
            # Resample OHLCV data
            resampled = self.ohlcv_data.copy()
            resampled.set_index('timestamp', inplace=True)
            
            # Resample using appropriate aggregation for each column
            resampled = resampled.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            # Skip if we don't have enough data
            if len(resampled) < 20:
                results[rule] = {"status": "insufficient_data"}
                continue
            
            # Create analyzer for this timeframe
            timeframe_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=resampled,
                symbol=f"{self.symbol}_{rule}"
            )
            
            # Calculate energy metrics
            timeframe_analyzer.calculate_energy_metrics()
            
            # Get momentum state
            direction, state = timeframe_analyzer.get_current_momentum_state()
            
            # Store results
            results[rule] = {
                "status": "analyzed",
                "direction": direction,
                "state": state,
                "latest_gradient": float(timeframe_analyzer.gradients[-1]),
                "data_points": len(resampled)
            }
        
        logger.info(f"{self.symbol}: Multi-timeframe analysis completed for {len(results)} timeframes")
        
        return results

    def optimize_data_size(self, max_points=500):
        """
        Optimize data size for large datasets
        
        This method downsamples the data for large datasets to improve performance
        while maintaining the overall pattern of the data.
        
        Args:
            max_points (int): Maximum number of data points to keep
            
        Returns:
            self: Returns self for method chaining
        """
        # Only optimize if we have more than max_points
        if len(self.ohlcv_data) > max_points:
            # Calculate how many points to skip
            skip = len(self.ohlcv_data) // max_points
            
            # Downsample the data
            self.ohlcv_data = self.ohlcv_data.iloc[::skip].reset_index(drop=True)
            
            # Reset calculated metrics
            self.energy_values = None
            self.smooth_energy = None
            self.gradients = None
            self.inelasticity = None
            
            logger.info(f"{self.symbol}: Optimized data size from {len(self.ohlcv_data)*skip} to {len(self.ohlcv_data)} points")
        
        return self








