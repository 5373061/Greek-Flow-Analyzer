import unittest
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the component to test
from momentum_analyzer import EnergyFlowAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMomentumAnalyzer(unittest.TestCase):
    """Tests for the MomentumAnalyzer component"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high is always >= open and close
        self.price_data['high'] = self.price_data[['open', 'close', 'high']].max(axis=1)
        
        # Ensure low is always <= open and close
        self.price_data['low'] = self.price_data[['open', 'close', 'low']].min(axis=1)
        
        # Symbol for testing
        self.symbol = "TEST"
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly"""
        try:
            # Pass parameters in the correct order: ohlcv_df first, then symbol
            analyzer = EnergyFlowAnalyzer(
                ohlcv_df=self.price_data,
                symbol=self.symbol
            )
            self.assertIsNotNone(analyzer)
            logger.info("EnergyFlowAnalyzer initialized successfully")
        except Exception as e:
            self.fail(f"EnergyFlowAnalyzer initialization failed: {e}")
    
    def test_momentum_calculation(self):
        """Test momentum calculation"""
        try:
            # Pass parameters in the correct order: ohlcv_df first, then symbol
            analyzer = EnergyFlowAnalyzer(
                ohlcv_df=self.price_data,
                symbol=self.symbol
            )
            
            # Calculate energy metrics first
            analyzer.calculate_energy_metrics()
            
            # Get momentum state
            direction, state = analyzer.get_current_momentum_state()
            
            # Check that momentum values are returned
            self.assertIsNotNone(direction)
            self.assertIsNotNone(state)
            
            logger.info(f"Momentum direction: {direction}, state: {state}")
        except Exception as e:
            self.fail(f"Momentum calculation failed: {e}")

    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        try:
            # Create data with NaN values in multiple columns
            test_data = self.price_data.copy()
            test_data.loc[10:15, 'close'] = np.nan
            test_data.loc[20:22, 'open'] = np.nan
            test_data.loc[30:32, 'high'] = np.nan
            test_data.loc[40:42, 'low'] = np.nan
            test_data.loc[50:52, 'volume'] = np.nan
            
            # Initialize analyzer with NaN data
            analyzer = EnergyFlowAnalyzer(
                ohlcv_df=test_data,
                symbol=self.symbol
            )
            
            # Check that NaNs were handled in all columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.assertFalse(analyzer.ohlcv_data[col].isna().any(), 
                                 f"NaN values should be filled in {col} column")
            
            # Verify the data shape is maintained (some rows might be dropped if all required values are NaN)
            self.assertGreater(len(analyzer.ohlcv_data), 80, 
                              "Most rows should be preserved after preprocessing")
            
            logger.info("Data preprocessing successful - all NaN values handled correctly")
        except Exception as e:
            self.fail(f"Data preprocessing failed: {e}")

    def test_momentum_classification(self):
        """Test the momentum state classification logic"""
        try:
            # Create a controlled dataset with a clear trend
            dates = pd.date_range(start='2020-01-01', periods=50)
            
            # Create an uptrend dataset with more pronounced trend
            uptrend_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.linspace(80, 120, 50),  # Stronger uptrend (50% increase)
                'close': np.linspace(80, 120, 50),  # Stronger uptrend
                'high': np.linspace(82, 122, 50),   # Stronger uptrend
                'low': np.linspace(78, 118, 50),    # Stronger uptrend
                'volume': np.linspace(5000, 15000, 50)  # Increasing volume (more pronounced)
            })
            
            # Add minimal noise to maintain the trend
            uptrend_data['open'] += np.random.normal(0, 0.5, 50)  # Reduced noise
            uptrend_data['close'] += np.random.normal(0, 0.5, 50)  # Reduced noise
            uptrend_data['high'] = uptrend_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.5, 50)  # Reduced noise
            uptrend_data['low'] = uptrend_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.5, 50)  # Reduced noise
            uptrend_data['volume'] = uptrend_data['volume'].astype(int)
            
            # Make sure the last few points have a strong positive trend
            for i in range(45, 50):
                uptrend_data.loc[i, 'close'] = uptrend_data.loc[i-1, 'close'] + 1.0 + np.random.uniform(0, 0.2)
            
            # Initialize analyzer with uptrend data
            uptrend_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=uptrend_data,
                symbol="UPTREND"
            )
            
            # Calculate energy metrics
            uptrend_analyzer.calculate_energy_metrics()
            
            # Get momentum state
            up_direction, up_state = uptrend_analyzer.get_current_momentum_state()
            
            # Create a downtrend dataset with more pronounced trend
            downtrend_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.linspace(120, 80, 50),  # Stronger downtrend (33% decrease)
                'close': np.linspace(120, 80, 50),  # Stronger downtrend
                'high': np.linspace(122, 82, 50),   # Stronger downtrend
                'low': np.linspace(118, 78, 50),    # Stronger downtrend
                'volume': np.linspace(5000, 15000, 50)  # Increasing volume
            })
            
            # Add minimal noise to maintain the trend
            downtrend_data['open'] += np.random.normal(0, 0.5, 50)  # Reduced noise
            downtrend_data['close'] += np.random.normal(0, 0.5, 50)  # Reduced noise
            downtrend_data['high'] = downtrend_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.5, 50)  # Reduced noise
            downtrend_data['low'] = downtrend_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.5, 50)  # Reduced noise
            downtrend_data['volume'] = downtrend_data['volume'].astype(int)
            
            # Make sure the last few points have a strong negative trend
            for i in range(45, 50):
                downtrend_data.loc[i, 'close'] = downtrend_data.loc[i-1, 'close'] - 1.0 - np.random.uniform(0, 0.2)
            
            # Initialize analyzer with downtrend data
            downtrend_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=downtrend_data,
                symbol="DOWNTREND"
            )
            
            # Calculate energy metrics
            downtrend_analyzer.calculate_energy_metrics()
            
            # Get momentum state
            down_direction, down_state = downtrend_analyzer.get_current_momentum_state()
            
            # Log the results
            logger.info(f"Uptrend classification: Direction={up_direction}, State={up_state}")
            logger.info(f"Downtrend classification: Direction={down_direction}, State={down_state}")
            
            # Add documentation about the classification rules
            logger.info("\nMomentum Classification Rules:")
            logger.info("1. Direction: Based on the sign of the energy gradient")
            logger.info("   - Positive: Increasing energy (gradient > 0)")
            logger.info("   - Negative: Decreasing energy (gradient < 0)")
            logger.info("   - Flat: No significant energy change (gradient ≈ 0)")
            logger.info("2. Strength: Based on the magnitude of the energy gradient")
            logger.info("   - Strong: |gradient| > threshold")
            logger.info("   - Moderate: threshold/3 < |gradient| < threshold")
            logger.info("   - Weak/Flat: |gradient| < threshold/3")
            logger.info("   where threshold is based on the standard deviation of recent gradients")
            
            # Add assertions to verify the expected momentum states
            self.assertEqual(up_direction, "Positive", "Uptrend should be classified as Positive direction")
            self.assertEqual(down_direction, "Negative", "Downtrend should be classified as Negative direction")
            
            # Verify that the states contain the direction words
            self.assertIn("Positive", up_state, "Uptrend state should contain 'Positive'")
            self.assertIn("Negative", down_state, "Downtrend state should contain 'Negative'")
        
        except Exception as e:
            logger.error(f"Error in momentum classification test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Momentum classification test failed: {e}")

    def test_flat_market_classification(self):
        """Test the momentum classification for flat/sideways markets"""
        try:
            # Create a dataset with sideways movement
            dates = pd.date_range(start='2020-01-01', periods=50)
            
            # Create a more truly flat dataset with minimal trend and very small oscillations
            np.random.seed(42)  # Set seed for reproducibility
            base_price = 100.0
            
            # Create oscillating prices with very small amplitude around the base price
            oscillations = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.5  # Small amplitude sine wave
            
            flat_data = pd.DataFrame({
                'timestamp': dates,
                'open': base_price + oscillations + np.random.normal(0, 0.2, 50),  # Very small noise
                'close': base_price + oscillations + np.random.normal(0, 0.2, 50),  # Very small noise
                'high': base_price + oscillations + 0.3 + np.random.normal(0, 0.1, 50),  # Slightly higher
                'low': base_price + oscillations - 0.3 + np.random.normal(0, 0.1, 50),   # Slightly lower
                'volume': np.ones(50) * 1000  # Constant low volume
            })
            
            # Ensure high is always >= open and close
            flat_data['high'] = flat_data[['open', 'close', 'high']].max(axis=1)
            
            # Ensure low is always <= open and close
            flat_data['low'] = flat_data[['open', 'close', 'low']].min(axis=1)
            
            # Initialize analyzer with flat data
            flat_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=flat_data,
                symbol="FLAT"
            )
            
            # Calculate energy metrics
            flat_analyzer.calculate_energy_metrics()
            
            # Get momentum state
            flat_direction, flat_state = flat_analyzer.get_current_momentum_state()
            
            # Log the results
            logger.info(f"Flat market classification: Direction={flat_direction}, State={flat_state}")
            
            # Verify that the state indicates a flat or weak momentum
            self.assertIn("Weak", flat_state, "Flat market should be classified with 'Weak' strength")
        
        except Exception as e:
            logger.error(f"Error in flat market classification test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Flat market classification test failed: {e}")

if __name__ == "__main__":
    unittest.main()







