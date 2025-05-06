"""
test_trade_signals.py - Tests for the trade signal generation module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trade_signals import TradeSignalGenerator, SignalType
from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
from cross_greek_patterns import CrossGreekPatternAnalyzer

class TestTradeSignalGenerator(unittest.TestCase):
    """
    Unit tests for the TradeSignalGenerator class.
    """
    
    def setUp(self):
        """Set up test environment before each test case."""
        # Create test data
        self.test_data = pd.DataFrame({
            'norm_delta': [0.2, 0.25, 0.3, 0.35, 0.4],
            'norm_gamma': [0.5, 0.5, 0.48, 0.45, 0.4],
            'norm_vanna': [0.3, 0.32, 0.35, 0.4, 0.38],
            'norm_theta': [-0.1, -0.12, -0.15, -0.18, -0.22],
            'price': [100, 102, 105, 107, 110],
            'delta': [0.5, 0.52, 0.55, 0.6, 0.65]
        })
        
        # Initialize analyzers
        self.pattern_analyzer = GreekOrdinalPatternAnalyzer(
            window_size=3,
            min_occurrences=2,
            min_confidence=0.5
        )
        
        self.cross_analyzer = CrossGreekPatternAnalyzer(self.pattern_analyzer)
        
        # Build a simple pattern library for testing
        patterns = self.pattern_analyzer.extract_patterns(self.test_data)
        analysis = self.pattern_analyzer.analyze_pattern_profitability(self.test_data, patterns)
        self.pattern_analyzer.build_pattern_library(analysis, 'ATM')
        
        # Initialize signal generator
        self.signal_generator = TradeSignalGenerator(
            self.pattern_analyzer, 
            self.cross_analyzer
        )
    
    def test_generate_entry_signals(self):
        """Test generation of entry signals."""
        # Recognize patterns
        recognized = self.pattern_analyzer.recognize_current_patterns(self.test_data.iloc[-3:])
        
        # Add some mock pattern data to ensure signals are generated
        for moneyness in recognized.keys():
            recognized[moneyness]['delta'] = {
                'pattern': (0, 1, 2),
                'description': 'rising',
                'stats': {
                    'count': 5,
                    'win_rate': 0.7,
                    'avg_return': 0.05,
                    'max_return': 0.1,
                    'min_return': -0.02
                }
            }
        
        # Generate signals without active position
        signals = self.signal_generator.generate_signals(
            self.test_data.iloc[-3:], 
            recognized,
            current_position=None
        )
        
        # Check that signals were generated
        self.assertTrue(len(signals) > 0)
        
        # Check signal structure
        for signal in signals:
            self.assertIn('signal_type', signal)
            self.assertIn('confidence', signal)
            self.assertIn('target_strike', signal)
            self.assertIn('strategy', signal)
            self.assertIn('take_profit', signal)
            self.assertIn('stop_loss', signal)
            
            # Verify signal type is ENTRY_LONG or ENTRY_SHORT
            self.assertIn(signal['signal_type'], [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT])
    
    def test_generate_exit_signals(self):
        """Test generation of exit signals."""
        # Create a mock active position
        current_position = {
            'type': 'long',
            'moneyness': 'ATM',
            'status': 'active',
            'entry_price': 5.0,
            'strike': 105,
            'expiration': datetime.now() + timedelta(days=30),
            'current_date': datetime.now()
        }
        
        # Create pattern data that triggers exit
        recognized = {
            'ATM': {
                'delta': {
                    'pattern': (2, 1, 0),  # Falling pattern
                    'description': 'falling',
                    'stats': {
                        'count': 4,
                        'win_rate': 0.3,
                        'avg_return': -0.05,
                        'max_return': 0.02,
                        'min_return': -0.1
                    }
                }
            },
            'ITM': {},
            'OTM': {},
            'VOL_CRUSH': {}
        }
        
        # Generate signals with active position
        signals = self.signal_generator.generate_signals(
            self.test_data.iloc[-3:], 
            recognized,
            current_position=current_position
        )
        
        # Check for exit signals
        exit_signals = [s for s in signals if s['signal_type'] in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]]
        self.assertTrue(len(exit_signals) > 0)
        
        # Check exit signal structure
        for signal in exit_signals:
            self.assertIn('exit_reason', signal)
            self.assertIn('target_price', signal)
            self.assertIn('partial_exit', signal)
    
    def test_signal_confidence_calculation(self):
        """Test the calculation of signal confidence."""
        test_stats = {
            'win_rate': 0.7,
            'count': 10,
            'avg_return': 0.05
        }
        
        pattern_info = {
            'stats': test_stats,
            'pattern': (0, 1, 2),
            'description': 'rising'
        }
        
        confidence = self.signal_generator._calculate_signal_confidence(test_stats, pattern_info)
        
        # Check that confidence is between 0 and 1
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # Higher win rate should give higher confidence
        test_stats_high = {
            'win_rate': 0.9,
            'count': 20,
            'avg_return': 0.1
        }
        pattern_info_high = {
            'stats': test_stats_high,
            'pattern': (0, 1, 2),
            'description': 'rising'
        }
        
        confidence_high = self.signal_generator._calculate_signal_confidence(test_stats_high, pattern_info_high)
        self.assertGreater(confidence_high, confidence)
    
    def test_target_strike_determination(self):
        """Test determination of target strike prices."""
        # Test OTM long call
        strike = self.signal_generator._determine_target_strike(
            SignalType.ENTRY_LONG, 'OTM', self.test_data
        )
        current_price = self.test_data['price'].iloc[-1]
        self.assertGreater(strike, current_price)  # OTM call should be above current price
        
        # Test ATM long call
        strike_atm = self.signal_generator._determine_target_strike(
            SignalType.ENTRY_LONG, 'ATM', self.test_data
        )
        self.assertAlmostEqual(strike_atm, round(current_price / 5) * 5, places=0)
        
        # Test ITM long call
        strike_itm = self.signal_generator._determine_target_strike(
            SignalType.ENTRY_LONG, 'ITM', self.test_data
        )
        self.assertLess(strike_itm, current_price)  # ITM call should be below current price
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation based on signal and confidence."""
        # High confidence long signal
        strategy = self.signal_generator._recommend_strategy(
            SignalType.ENTRY_LONG, 'OTM', 0.8
        )
        self.assertEqual(strategy, "LONG_CALL")
        
        # Medium confidence short signal
        strategy = self.signal_generator._recommend_strategy(
            SignalType.ENTRY_SHORT, 'ATM', 0.7
        )
        self.assertEqual(strategy, "BEAR_PUT_SPREAD")
        
        # Low confidence signal
        strategy = self.signal_generator._recommend_strategy(
            SignalType.ENTRY_LONG, 'ATM', 0.6
        )
        self.assertEqual(strategy, "IRON_CONDOR")
    
    def test_take_profit_stop_loss_calculation(self):
        """Test calculation of take profit and stop loss levels."""
        test_stats = {
            'avg_return': 0.05,
            'max_return': 0.15,
            'min_return': -0.08
        }
        
        # Test take profit
        take_profit = self.signal_generator._calculate_take_profit(
            SignalType.ENTRY_LONG, test_stats, self.test_data
        )
        self.assertGreaterEqual(take_profit, 0.05)  # Should be at least 5%
        self.assertLessEqual(take_profit, 0.3)      # Should not exceed 30%
        
        # Test stop loss
        stop_loss = self.signal_generator._calculate_stop_loss(
            SignalType.ENTRY_LONG, test_stats, self.test_data
        )
        self.assertLessEqual(stop_loss, -0.05)      # Should be at least -5%
        self.assertGreaterEqual(stop_loss, -0.2)    # Should not exceed -20%
    
    def test_pattern_identification(self):
        """Test pattern identification functions."""
        rising_pattern = (0, 1, 2, 3)
        falling_pattern = (3, 2, 1, 0)
        flat_pattern = (1, 0, 2, 1)
        
        # Test rising pattern detection
        self.assertTrue(self.signal_generator._is_rising_pattern(rising_pattern))
        self.assertFalse(self.signal_generator._is_rising_pattern(falling_pattern))
        
        # Test falling pattern detection
        self.assertTrue(self.signal_generator._is_falling_pattern(falling_pattern))
        self.assertFalse(self.signal_generator._is_falling_pattern(rising_pattern))
    
    def test_position_management(self):
        """Test position management functions."""
        # Test active position detection
        active_position = {
            'status': 'active',
            'expiration': datetime.now() + timedelta(days=30),
            'current_date': datetime.now()
        }
        self.assertTrue(self.signal_generator._has_active_position(active_position))
        
        # Test inactive position
        inactive_position = {
            'status': 'closed'
        }
        self.assertFalse(self.signal_generator._has_active_position(inactive_position))
        
        # Test roll requirement
        expiring_position = {
            'expiration': datetime.now() + timedelta(days=5),
            'current_date': datetime.now()
        }
        self.assertTrue(self.signal_generator._needs_rolling(expiring_position, self.test_data))


if __name__ == '__main__':
    unittest.main()