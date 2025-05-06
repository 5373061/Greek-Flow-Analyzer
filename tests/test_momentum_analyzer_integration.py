import unittest
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta
import tempfile

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components for integration testing
from analysis.symbol_analyzer import SymbolAnalyzer
from momentum_analyzer import EnergyFlowAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMomentumAnalyzerIntegration(unittest.TestCase):
    """Integration tests for MomentumAnalyzer with SymbolAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test symbol
        self.symbol = "TEST"
        
        # Create analyzer instance
        self.analyzer = SymbolAnalyzer(
            cache_dir=self.temp_dir,
            output_dir=self.output_dir,
            use_parallel=False
        )
        
        # Generate test price data
        self.price_data = self._generate_test_price_data()
    
    def tearDown(self):
        """Clean up after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _generate_test_price_data(self):
        """Generate synthetic price data for testing"""
        # Create a DataFrame with price data
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # Create an uptrend dataset
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(90, 110, 100) + np.random.normal(0, 1, 100),
            'high': np.linspace(92, 112, 100) + np.random.normal(0, 1, 100),
            'low': np.linspace(88, 108, 100) + np.random.normal(0, 1, 100),
            'close': np.linspace(90, 110, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high is always >= open and close
        price_data['high'] = price_data[['open', 'close', 'high']].max(axis=1)
        
        # Ensure low is always <= open and close
        price_data['low'] = price_data[['open', 'close', 'low']].min(axis=1)
        
        return price_data
    
    def test_momentum_analyzer_with_symbol_analyzer(self):
        """Test the integration between MomentumAnalyzer and SymbolAnalyzer"""
        try:
            # Save price data to a file
            price_file = os.path.join(self.temp_dir, f"{self.symbol}_price_history.csv")
            self.price_data.to_csv(price_file, index=False)
            logger.info(f"Saved test price history to {price_file}")
            
            # Create a mock implementation class that extends SymbolAnalyzer
            class MockSymbolAnalyzer(SymbolAnalyzer):
                def __init__(self, original, test_price_data):
                    # Copy attributes from original
                    self.__dict__ = original.__dict__.copy()
                    
                    # Store test data
                    self.test_price_data = test_price_data
                    
                    # Check if the method exists before trying to override it
                    if hasattr(self, 'load_price_history'):
                        self._original_load_price_history = self.load_price_history
                        self.load_price_history = self._mock_load_price_history
                    else:
                        # Add the method if it doesn't exist
                        self.load_price_history = self._mock_load_price_history
                    
                    # Also check for other possible method names that might be used
                    for method_name in ['get_price_history', 'fetch_price_data', 'get_price_data']:
                        if hasattr(self, method_name):
                            setattr(self, f"_original_{method_name}", getattr(self, method_name))
                            setattr(self, method_name, self._mock_load_price_history)
                
                def _mock_load_price_history(self, symbol, *args, **kwargs):
                    """Mock for load_price_history method"""
                    logger.info(f"Mock load_price_history called for {symbol}")
                    return self.test_price_data
                
                def analyze_momentum(self, symbol, config=None):
                    """Add a method to analyze momentum using EnergyFlowAnalyzer"""
                    logger.info(f"Analyzing momentum for {symbol}")
                    
                    # Get price history
                    price_data = self.load_price_history(symbol)
                    
                    # Create momentum analyzer
                    momentum_analyzer = EnergyFlowAnalyzer(
                        ohlcv_df=price_data,
                        symbol=symbol,
                        config=config
                    )
                    
                    # Calculate energy metrics
                    metrics = momentum_analyzer.calculate_energy_metrics()
                    
                    # Get momentum state
                    direction, state = momentum_analyzer.get_current_momentum_state()
                    
                    # Detect divergences
                    divergences = momentum_analyzer.detect_divergences()
                    
                    # Detect momentum changes
                    momentum_changes = momentum_analyzer.detect_momentum_changes()
                    
                    # Create visualization
                    viz_path = os.path.join(self.output_dir, f"{symbol}_momentum_viz.png")
                    momentum_analyzer.create_visualization(output_path=viz_path)
                    
                    # Export results
                    results_path = os.path.join(self.output_dir, f"{symbol}_momentum_results.json")
                    momentum_analyzer.export_results(output_path=results_path)
                    
                    # Return comprehensive results
                    return {
                        'symbol': symbol,
                        'direction': direction,
                        'state': state,
                        'metrics': metrics,
                        'divergences': divergences,
                        'momentum_changes': momentum_changes,
                        'visualization_path': viz_path,
                        'results_path': results_path
                    }
            
            # Create a mock analyzer that will use our test data
            mock_analyzer = MockSymbolAnalyzer(
                self.analyzer, 
                self.price_data
            )
            
            # Test the integrated momentum analysis
            try:
                # Define a custom configuration
                momentum_config = {
                    'smoothing_sigma': 2.0,
                    'lookback_period': 15,
                    'strong_threshold_multiplier': 1.2,
                    'moderate_threshold_multiplier': 0.4
                }
                
                # Run the momentum analysis with the mock analyzer
                results = mock_analyzer.analyze_momentum(self.symbol, config=momentum_config)
                
                # Verify results
                self.assertIsNotNone(results)
                self.assertEqual(results['symbol'], self.symbol)
                self.assertIn('direction', results)
                self.assertIn('state', results)
                self.assertIn('metrics', results)
                self.assertIn('divergences', results)
                self.assertIn('momentum_changes', results)
                
                # Check that visualization and results files were created
                self.assertTrue(os.path.exists(results['visualization_path']))
                self.assertTrue(os.path.exists(results['results_path']))
                
                logger.info(f"Integrated momentum analysis - Direction: {results['direction']}, State: {results['state']}")
                logger.info(f"Detected {len(results['divergences']['bullish_divergences'])} bullish and "
                           f"{len(results['divergences']['bearish_divergences'])} bearish divergences")
                logger.info(f"Detected {len(results['momentum_changes'])} significant momentum changes")
                
                # Test passed
                logger.info("MomentumAnalyzer integration test passed")
                
            except Exception as e:
                logger.error(f"Error in momentum analyzer test: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.fail(f"MomentumAnalyzer integration test failed: {e}")
        
        except Exception as e:
            logger.error(f"Error in test setup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Test setup failed: {e}")

    def test_momentum_analyzer_standalone(self):
        """Test the EnergyFlowAnalyzer as a standalone component"""
        try:
            # Create an EnergyFlowAnalyzer instance directly with the test data
            momentum_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=self.price_data,
                symbol=self.symbol
            )
            
            # Calculate energy metrics
            metrics = momentum_analyzer.calculate_energy_metrics()
            
            # Verify that metrics were calculated
            self.assertIsNotNone(metrics)
            self.assertIn('energy_values', metrics)
            self.assertIn('smooth_energy', metrics)
            self.assertIn('gradients', metrics)
            self.assertIn('inelasticity', metrics)
            
            # Get momentum state
            direction, state = momentum_analyzer.get_current_momentum_state()
            
            # Check that momentum values are returned
            self.assertIsNotNone(direction)
            self.assertIsNotNone(state)
            
            logger.info(f"Standalone momentum analysis - Direction: {direction}, State: {state}")
            
            # Test visualization (if matplotlib is available)
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend for testing
                
                # Create a temporary file for the visualization
                viz_file = os.path.join(self.temp_dir, f"{self.symbol}_momentum_viz.png")
                
                # Generate visualization
                fig, axes = momentum_analyzer.visualize_energy_flow(
                    output_path=viz_file,
                    show_plot=False
                )
                
                # Check that visualization was created
                self.assertTrue(os.path.exists(viz_file))
                logger.info(f"Created visualization at {viz_file}")
                
            except ImportError:
                logger.info("Matplotlib not available, skipping visualization test")
            
            # Test export functionality
            export_file = os.path.join(self.temp_dir, f"{self.symbol}_momentum_results.json")
            result_file = momentum_analyzer.export_results(export_file)
            
            # Check that export was successful
            self.assertEqual(result_file, export_file)
            self.assertTrue(os.path.exists(export_file))
            
            # Verify the exported file contains the expected data
            import json
            with open(export_file, 'r') as f:
                results = json.load(f)
            
            self.assertEqual(results['symbol'], self.symbol)
            self.assertIn('momentum', results)
            self.assertEqual(results['momentum']['direction'], direction)
            
            logger.info("Standalone momentum analyzer test passed")
            
        except Exception as e:
            logger.error(f"Error in standalone momentum analyzer test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Standalone momentum analyzer test failed: {e}")

    def test_momentum_analyzer_advanced_features(self):
        """Test the advanced features of EnergyFlowAnalyzer"""
        try:
            # Create an EnergyFlowAnalyzer instance with custom configuration
            custom_config = {
                'smoothing_sigma': 2.0,  # Increased smoothing
                'lookback_period': 15,   # Custom lookback period
                'strong_threshold_multiplier': 1.2,
                'moderate_threshold_multiplier': 0.4
            }
            
            momentum_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=self.price_data,
                symbol=self.symbol,
                config=custom_config
            )
            
            # 1. Test calculate_energy_metrics with custom configuration
            metrics = momentum_analyzer.calculate_energy_metrics()
            
            # Verify that metrics were calculated
            self.assertIsNotNone(metrics)
            self.assertIn('energy_values', metrics)
            self.assertIn('smooth_energy', metrics)
            self.assertIn('gradients', metrics)
            self.assertIn('inelasticity', metrics)
            
            # 2. Test detect_divergences
            divergences = momentum_analyzer.detect_divergences(window_size=20)
            
            # Verify divergence detection
            self.assertIsNotNone(divergences)
            self.assertIn('bullish_divergences', divergences)
            self.assertIn('bearish_divergences', divergences)
            
            logger.info(f"Detected {len(divergences['bullish_divergences'])} bullish and "
                        f"{len(divergences['bearish_divergences'])} bearish divergences")
            
            # 3. Test detect_momentum_changes
            momentum_changes = momentum_analyzer.detect_momentum_changes(threshold_multiplier=1.2)
            
            # Verify momentum change detection
            self.assertIsNotNone(momentum_changes)
            logger.info(f"Detected {len(momentum_changes)} significant momentum changes")
            
            # 4. Test backtest_momentum_signals
            backtest_results = momentum_analyzer.backtest_momentum_signals(holding_period=3)
            
            # Verify backtest results
            self.assertIsNotNone(backtest_results)
            self.assertIn('signals', backtest_results)
            self.assertIn('performance', backtest_results)
            
            # Log performance metrics
            pos_perf = backtest_results['performance']['positive_momentum']
            neg_perf = backtest_results['performance']['negative_momentum']
            
            logger.info(f"Backtest results - Positive momentum: {pos_perf['count']} signals, "
                        f"Win rate: {pos_perf['win_rate']:.2f}, Avg return: {pos_perf['avg_return']:.2f}%")
            logger.info(f"Backtest results - Negative momentum: {neg_perf['count']} signals, "
                        f"Win rate: {neg_perf['win_rate']:.2f}, Avg return: {neg_perf['avg_return']:.2f}%")
            
            # 5. Test analyze_multiple_timeframes
            # Create a longer dataset for multi-timeframe analysis
            dates = pd.date_range(start='2020-01-01', periods=200)
            long_price_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.linspace(90, 110, 200) + np.random.normal(0, 1, 200),
                'high': np.linspace(92, 112, 200) + np.random.normal(0, 1, 200),
                'low': np.linspace(88, 108, 200) + np.random.normal(0, 1, 200),
                'close': np.linspace(90, 110, 200) + np.random.normal(0, 1, 200),
                'volume': np.random.randint(1000, 10000, 200)
            })
            
            # Ensure high is always >= open and close
            long_price_data['high'] = long_price_data[['open', 'close', 'high']].max(axis=1)
            
            # Ensure low is always <= open and close
            long_price_data['low'] = long_price_data[['open', 'close', 'low']].min(axis=1)
            
            # Create analyzer with longer dataset
            long_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=long_price_data,
                symbol=self.symbol
            )
            
            # Test multi-timeframe analysis
            timeframe_results = long_analyzer.analyze_multiple_timeframes(
                resample_rules=['1D', '3D', '7D']
            )
            
            # Verify timeframe results
            self.assertIsNotNone(timeframe_results)
            self.assertIn('1D', timeframe_results)
            
            # Log timeframe results
            for timeframe, result in timeframe_results.items():
                if result.get('status') == 'analyzed':
                    logger.info(f"Timeframe {timeframe}: {result['direction']}, {result['state']}")
                else:
                    logger.info(f"Timeframe {timeframe}: {result['status']}")
            
            # 6. Test optimize_data_size
            # Create a very large dataset
            dates = pd.date_range(start='2020-01-01', periods=1000)
            large_price_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.linspace(90, 110, 1000) + np.random.normal(0, 1, 1000),
                'high': np.linspace(92, 112, 1000) + np.random.normal(0, 1, 1000),
                'low': np.linspace(88, 108, 1000) + np.random.normal(0, 1, 1000),
                'close': np.linspace(90, 110, 1000) + np.random.normal(0, 1, 1000),
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            # Ensure high is always >= open and close
            large_price_data['high'] = large_price_data[['open', 'close', 'high']].max(axis=1)
            
            # Ensure low is always <= open and close
            large_price_data['low'] = large_price_data[['open', 'close', 'low']].min(axis=1)
            
            # Create analyzer with large dataset
            large_analyzer = EnergyFlowAnalyzer(
                ohlcv_df=large_price_data,
                symbol=f"{self.symbol}_LARGE"
            )
            
            # Record original size
            original_size = len(large_analyzer.ohlcv_data)
            
            # Optimize data size
            large_analyzer.optimize_data_size(max_points=200)
            
            # Verify optimization
            optimized_size = len(large_analyzer.ohlcv_data)
            self.assertLess(optimized_size, original_size)
            self.assertLessEqual(optimized_size, 200)
            
            logger.info(f"Data size optimization: {original_size} -> {optimized_size} points")
            
            # Calculate metrics on optimized data
            large_analyzer.calculate_energy_metrics()
            direction, state = large_analyzer.get_current_momentum_state()
            
            # Verify that analysis still works on optimized data
            self.assertIsNotNone(direction)
            self.assertIsNotNone(state)
            
            logger.info(f"Optimized data analysis - Direction: {direction}, State: {state}")
            
            # Test passed
            logger.info("Advanced EnergyFlowAnalyzer features test passed")
            
        except Exception as e:
            logger.error(f"Error in advanced features test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Advanced features test failed: {e}")

    def test_momentum_analyzer_full_pipeline(self):
        """Test the integration of EnergyFlowAnalyzer in a full analysis pipeline"""
        try:
            # Create a mock implementation of a full analysis pipeline
            def run_full_analysis(symbol, price_data, output_dir):
                """Mock full analysis pipeline that includes momentum analysis"""
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Step 1: Basic price analysis
                price_stats = {
                    'symbol': symbol,
                    'start_date': price_data['timestamp'].min(),
                    'end_date': price_data['timestamp'].max(),
                    'start_price': price_data['close'].iloc[0],
                    'end_price': price_data['close'].iloc[-1],
                    'price_change': price_data['close'].iloc[-1] - price_data['close'].iloc[0],
                    'price_change_pct': (price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1) * 100
                }
                
                # Step 2: Momentum analysis
                momentum_analyzer = EnergyFlowAnalyzer(
                    ohlcv_df=price_data,
                    symbol=symbol
                )
                
                # Calculate energy metrics
                metrics = momentum_analyzer.calculate_energy_metrics()
                
                # Get momentum state
                direction, state = momentum_analyzer.get_current_momentum_state()
                
                # Detect divergences
                divergences = momentum_analyzer.detect_divergences()
                
                # Backtest momentum signals
                backtest_results = momentum_analyzer.backtest_momentum_signals(holding_period=5)
                
                # Create visualization
                viz_path = os.path.join(output_dir, f"{symbol}_momentum_viz.png")
                momentum_analyzer.create_visualization(output_path=viz_path)
                
                # Step 3: Multi-timeframe analysis
                timeframe_results = momentum_analyzer.analyze_multiple_timeframes(
                    resample_rules=['1D', '3D', '7D']
                )
                
                # Step 4: Combine results
                combined_results = {
                    'symbol': symbol,
                    'price_analysis': price_stats,
                    'momentum_analysis': {
                        'direction': direction,
                        'state': state,
                        'metrics': {
                            'inelasticity': float(metrics['inelasticity']),
                            'latest_gradient': float(momentum_analyzer.gradients[-1])
                        },
                        'divergences': divergences,
                        'backtest': backtest_results,
                        'timeframes': timeframe_results
                    }
                }
                
                # Step 5: Generate trading signals based on momentum
                signals = []
                
                # Signal based on current momentum
                if direction == "Positive" and "Strong" in state:
                    signals.append({
                        'type': 'LONG',
                        'reason': f'Strong positive momentum: {state}',
                        'confidence': 'HIGH'
                    })
                elif direction == "Negative" and "Strong" in state:
                    signals.append({
                        'type': 'SHORT',
                        'reason': f'Strong negative momentum: {state}',
                        'confidence': 'HIGH'
                    })
                
                # Signals based on divergences
                if divergences['bullish_divergences']:
                    signals.append({
                        'type': 'LONG',
                        'reason': f'Bullish divergence detected',
                        'confidence': 'MEDIUM'
                    })
                
                if divergences['bearish_divergences']:
                    signals.append({
                        'type': 'SHORT',
                        'reason': f'Bearish divergence detected',
                        'confidence': 'MEDIUM'
                    })
                
                # Add signals to results
                combined_results['signals'] = signals
                
                # Step 6: Save results to file
                results_path = os.path.join(output_dir, f"{symbol}_full_analysis.json")
                with open(results_path, 'w') as f:
                    import json
                    json.dump(combined_results, f, indent=2, default=str)
                
                return combined_results
            
            # Run the full analysis pipeline
            output_dir = os.path.join(self.output_dir, "full_pipeline")
            os.makedirs(output_dir, exist_ok=True)
            
            results = run_full_analysis(self.symbol, self.price_data, output_dir)
            
            # Verify results
            self.assertIsNotNone(results)
            self.assertEqual(results['symbol'], self.symbol)
            self.assertIn('price_analysis', results)
            self.assertIn('momentum_analysis', results)
            self.assertIn('signals', results)
            
            # Check momentum analysis results
            momentum = results['momentum_analysis']
            self.assertIn('direction', momentum)
            self.assertIn('state', momentum)
            self.assertIn('metrics', momentum)
            self.assertIn('divergences', momentum)
            self.assertIn('backtest', momentum)
            self.assertIn('timeframes', momentum)
            
            # Check that results file was created
            results_path = os.path.join(output_dir, f"{self.symbol}_full_analysis.json")
            self.assertTrue(os.path.exists(results_path))
            
            # Log key results
            logger.info(f"Full pipeline analysis - Direction: {momentum['direction']}, State: {momentum['state']}")
            logger.info(f"Generated {len(results['signals'])} trading signals")
            
            # Test passed
            logger.info("Full pipeline integration test passed")
            
        except Exception as e:
            logger.error(f"Error in full pipeline test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Full pipeline test failed: {e}")

if __name__ == "__main__":
    unittest.main()




