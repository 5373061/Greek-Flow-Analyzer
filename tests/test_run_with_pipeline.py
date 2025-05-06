"""
Tests for run_with_pipeline.py script.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import io
import subprocess

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
import run_with_pipeline


class TestRunWithPipeline(unittest.TestCase):
    """Test cases for run_with_pipeline.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        # Create subdirectories
        os.makedirs(os.path.join(self.data_dir, "options"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "prices"), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test data
        self.test_tickers = ["TEST"]
        
        # Sample options data
        self.options_data = pd.DataFrame({
            'strike': [100, 110, 120, 90, 80],
            'expiration': ['2023-12-01'] * 5,
            'type': ['c', 'c', 'c', 'p', 'p'],
            'openInterest': [100, 200, 150, 120, 80],
            'impliedVolatility': [0.3, 0.25, 0.2, 0.35, 0.4],
            'delta': [0.6, 0.5, 0.4, -0.4, -0.5],
            'gamma': [0.02, 0.03, 0.01, 0.02, 0.03],
            'theta': [-0.05, -0.04, -0.03, -0.04, -0.05],
            'vega': [0.2, 0.3, 0.1, 0.2, 0.3]
        })
        
        # Sample price data
        self.price_data = pd.DataFrame({
            'date': [pd.Timestamp('2023-11-01'), pd.Timestamp('2023-11-02')],
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000000, 1200000]
        })
        
        # Sample analysis results
        self.analysis_results = {
            "greek_profiles": {
                "delta_exposure": 0.5,
                "gamma_exposure": 0.2,
                "theta_exposure": -0.1,
                "vega_exposure": 0.3
            },
            "formatted_results": {
                "market_regime": "Gamma-Dominated",
                "reset_points": [105.5]
            },
            "chain_energy": {
                "energy_levels": [100, 110],
                "energy_concentration": 0.7
            }
        }
        
        # Sample entropy results
        self.entropy_results = {
            "delta_entropy": 0.8,
            "gamma_entropy": 0.6,
            "theta_entropy": 0.7,
            "vega_entropy": 0.5,
            "anomalies": [
                {"type": "delta", "strike": 100, "score": 0.9}
            ]
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    @patch('run_with_pipeline.AnalysisPipeline')
    @patch('run_with_pipeline.OptionsDataPipeline')
    @patch('api_fetcher.fetch_options_chain_snapshot')
    @patch('api_fetcher.fetch_underlying_snapshot')
    @patch('api_fetcher.get_spot_price_from_snapshot')
    @patch('api_fetcher.preprocess_api_options_data')
    def test_run_pipeline_analysis(self, mock_preprocess, mock_get_price, 
                                  mock_fetch_underlying, mock_fetch_options,
                                  mock_data_pipeline, mock_analysis_pipeline):
        """Test run_pipeline_analysis function."""
        # Set up mocks
        mock_pipeline_instance = mock_analysis_pipeline.return_value
        mock_pipeline_instance.run_full_analysis.return_value = self.analysis_results
        
        mock_fetch_underlying.return_value = {"last": {"price": 100.0}}
        mock_get_price.return_value = 100.0
        mock_fetch_options.return_value = {"results": [{"some": "data"}]}
        mock_preprocess.return_value = self.options_data
        
        # Run the function
        results = run_with_pipeline.run_pipeline_analysis(
            tickers=self.test_tickers,
            api_key="test_api_key",
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        
        # Assertions
        self.assertIsNotNone(results)
        self.assertIn("TEST", results)
        self.assertEqual(results["TEST"]["status"], "success")
        self.assertEqual(results["TEST"]["analysis_results"], self.analysis_results)
        
        # Verify mocks were called
        mock_fetch_underlying.assert_called_once_with("TEST", "test_api_key")
        mock_fetch_options.assert_called_once_with("TEST", "test_api_key")
        mock_pipeline_instance.run_full_analysis.assert_called_once()

    @patch('run_with_pipeline.SymbolAnalyzer')
    def test_run_pattern_analysis(self, mock_symbol_analyzer):
        """Test run_pattern_analysis function."""
        # Set up mocks
        mock_analyzer_instance = mock_symbol_analyzer.return_value
        mock_analyzer_instance.run_analysis.return_value = {
            "TEST": {
                "patterns": ["Double Bottom", "Bull Flag"],
                "confidence": 0.85
            }
        }
        
        # Run the function with HAS_PATTERN_ANALYZER set to True
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', True):
            results = run_with_pipeline.run_pattern_analysis(
                tickers=self.test_tickers,
                output_dir=self.output_dir
            )
            
            # Assertions
            self.assertIsNotNone(results)
            self.assertIn("TEST", results)
            self.assertEqual(results["TEST"]["patterns"][0], "Double Bottom")
            
            # Verify mocks were called
            mock_symbol_analyzer.assert_called_once()
            mock_analyzer_instance.run_analysis.assert_called_once_with(
                self.test_tickers, analysis_date=None
            )
        
        # Test with HAS_PATTERN_ANALYZER set to False
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', False):
            results = run_with_pipeline.run_pattern_analysis(
                tickers=self.test_tickers,
                output_dir=self.output_dir
            )
            
            # Should return False when pattern analyzer is not available
            self.assertFalse(results)

    @patch('run_with_pipeline.argparse.ArgumentParser.parse_args')
    @patch('run_with_pipeline.run_pipeline_analysis')
    @patch('run_with_pipeline.run_pattern_analysis')
    @patch('run_with_pipeline.load_tickers_from_csv')
    def test_main_with_args(self, mock_load_tickers, mock_run_pattern, 
                           mock_run_pipeline, mock_parse_args):
        """Test main function with command line arguments."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.tickers = self.test_tickers
        mock_args.csv = None
        mock_args.analysis_type = "both"
        mock_args.skip_entropy = False
        mock_args.api_key = "test_api_key"
        mock_args.data_dir = self.data_dir
        mock_args.output_dir = self.output_dir
        mock_args.no_parallel = False
        mock_parse_args.return_value = mock_args
        
        mock_run_pipeline.return_value = {"TEST": {"status": "success"}}
        mock_run_pattern.return_value = {"TEST": {"patterns": ["Double Bottom"]}}
        
        # Run main with arguments
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', True):
            with patch.object(sys, 'argv', ['run_with_pipeline.py', '--tickers', 'TEST']):
                result = run_with_pipeline.main()
                
                # Assertions
                self.assertEqual(result, 0)  # Should return 0 for success
                
                # Verify mocks were called
                mock_run_pipeline.assert_called_once_with(
                    tickers=self.test_tickers,
                    api_key="test_api_key",
                    data_dir=self.data_dir,
                    output_dir=self.output_dir,
                    skip_entropy=False
                )
                mock_run_pattern.assert_called_once()

    @patch('run_with_pipeline.run_pipeline_analysis')
    @patch('run_with_pipeline.run_pattern_analysis')
    def test_main_default(self, mock_run_pattern, mock_run_pipeline):
        """Test main function with default values."""
        # Set up mocks
        mock_run_pipeline.return_value = {"AAPL": {"status": "success"}}
        mock_run_pattern.return_value = {"AAPL": {"patterns": ["Double Bottom"]}}
        
        # Run main with no arguments (default case)
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', True):
            with patch.object(sys, 'argv', ['run_with_pipeline.py']):
                result = run_with_pipeline.main()
                
                # Assertions
                self.assertEqual(result, 0)  # Should return 0 for success
                
                # Verify default tickers were used
                mock_run_pipeline.assert_called_once()
                # Extract the first argument (tickers) from the call
                called_tickers = mock_run_pipeline.call_args[1]['tickers']
                self.assertIn("AAPL", called_tickers)
                self.assertIn("MSFT", called_tickers)
                self.assertIn("SPY", called_tickers)

    @patch('run_with_pipeline.pd.read_csv')
    def test_load_tickers_from_csv(self, mock_read_csv):
        """Test load_tickers_from_csv function."""
        # Set up mock
        mock_df = pd.DataFrame({"symbol": ["AAPL", "MSFT", "GOOG"]})
        mock_read_csv.return_value = mock_df
        
        # Test with valid CSV
        tickers = run_with_pipeline.load_tickers_from_csv("test.csv")
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOG"])
        
        # Test with CSV missing symbol column
        mock_read_csv.return_value = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOG"]})
        tickers = run_with_pipeline.load_tickers_from_csv("test.csv")
        self.assertEqual(tickers, [])
        
        # Test with exception
        mock_read_csv.side_effect = Exception("File not found")
        tickers = run_with_pipeline.load_tickers_from_csv("nonexistent.csv")
        self.assertEqual(tickers, [])

    @patch('run_with_pipeline.json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_numpy_encoder(self, mock_file, mock_json_dump):
        """Test NumpyEncoder class."""
        # Create test data with numpy types
        test_data = {
            "integer": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "regular": "string"
        }
        
        # Use the encoder to dump the data
        with open("test.json", "w") as f:
            json.dump(test_data, f, cls=run_with_pipeline.NumpyEncoder)
        
        # Verify the encoder was used
        mock_file.assert_called_once_with("test.json", "w")
        mock_json_dump.assert_called_once()
        
        # Extract the encoder instance from the call
        encoder = mock_json_dump.call_args[1]['cls']
        
        # Test the encoder directly
        self.assertEqual(encoder().default(np.int64(42)), 42)
        self.assertEqual(encoder().default(np.float64(3.14)), 3.14)
        self.assertEqual(encoder().default(np.array([1, 2, 3])), [1, 2, 3])
        
        # Test with non-numpy type (should raise TypeError)
        with self.assertRaises(TypeError):
            encoder().default("string")

    @patch('run_with_pipeline.AnalysisPipeline')
    @patch('run_with_pipeline.OptionsDataPipeline')
    @patch('api_fetcher.fetch_options_chain_snapshot')
    @patch('api_fetcher.fetch_underlying_snapshot')
    @patch('api_fetcher.get_spot_price_from_snapshot')
    @patch('api_fetcher.preprocess_api_options_data')
    def test_run_pipeline_analysis_with_error(self, mock_preprocess, mock_get_price, 
                                             mock_fetch_underlying, mock_fetch_options,
                                             mock_data_pipeline, mock_analysis_pipeline):
        """Test run_pipeline_analysis function with errors."""
        # Set up mocks to simulate an error in fetching underlying data
        mock_fetch_underlying.return_value = None
        
        # Run the function
        results = run_with_pipeline.run_pipeline_analysis(
            tickers=self.test_tickers,
            api_key="test_api_key",
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        
        # Assertions
        self.assertIsNotNone(results)
        self.assertIn("TEST", results)
        self.assertEqual(results["TEST"]["status"], "error")
        self.assertIn("message", results["TEST"])
        
        # Verify mocks were called
        mock_fetch_underlying.assert_called_once_with("TEST", "test_api_key")
        mock_fetch_options.assert_not_called()

    @patch('run_with_pipeline.AnalysisPipeline')
    @patch('run_with_pipeline.OptionsDataPipeline')
    @patch('api_fetcher.fetch_options_chain_snapshot')
    @patch('api_fetcher.fetch_underlying_snapshot')
    @patch('api_fetcher.get_spot_price_from_snapshot')
    @patch('api_fetcher.preprocess_api_options_data')
    def test_run_pipeline_analysis_with_invalid_api_key(self, mock_preprocess, mock_get_price, 
                                                      mock_fetch_underlying, mock_fetch_options,
                                                      mock_data_pipeline, mock_analysis_pipeline):
        """Test run_pipeline_analysis function with invalid API key."""
        # Run the function with an invalid API key
        results = run_with_pipeline.run_pipeline_analysis(
            tickers=self.test_tickers,
            api_key="YOUR_ACTUAL_API_KEY_HERE",  # Invalid API key
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        
        # Assertions - should return False for invalid API key
        self.assertFalse(results)
        
        # Verify mocks were not called
        mock_fetch_underlying.assert_not_called()
        mock_fetch_options.assert_not_called()

    @patch('run_with_pipeline.argparse.ArgumentParser.parse_args')
    @patch('run_with_pipeline.run_pipeline_analysis')
    @patch('run_with_pipeline.run_pattern_analysis')
    @patch('run_with_pipeline.load_tickers_from_csv')
    def test_main_with_csv_file(self, mock_load_tickers, mock_run_pattern, 
                               mock_run_pipeline, mock_parse_args):
        """Test main function with CSV file input."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.tickers = None
        mock_args.csv = "test_tickers.csv"
        mock_args.analysis_type = "greek"
        mock_args.skip_entropy = True
        mock_args.api_key = None
        mock_args.data_dir = None
        mock_args.output_dir = None
        mock_args.no_parallel = True
        mock_parse_args.return_value = mock_args
        
        # Mock the CSV loading
        mock_load_tickers.return_value = ["AAPL", "MSFT", "GOOG"]
        
        # Mock the pipeline analysis
        mock_run_pipeline.return_value = {
            "AAPL": {"status": "success"},
            "MSFT": {"status": "success"},
            "GOOG": {"status": "error", "message": "Some error"}
        }
        
        # Run main with CSV argument
        with patch.object(sys, 'argv', ['run_with_pipeline.py', '--csv', 'test_tickers.csv']):
            result = run_with_pipeline.main()
            
            # Assertions
            self.assertEqual(result, 0)  # Should return 0 for success
            
            # Verify mocks were called
            mock_load_tickers.assert_called_once_with("test_tickers.csv")
            mock_run_pipeline.assert_called_once_with(
                tickers=["AAPL", "MSFT", "GOOG"],
                api_key=run_with_pipeline.POLYGON_API_KEY,  # Should use default
                data_dir=run_with_pipeline.DATA_DIR,  # Should use default
                output_dir=run_with_pipeline.OUTPUT_DIR,  # Should use default
                skip_entropy=True
            )
            mock_run_pattern.assert_not_called()  # Should not be called for greek only

    @patch('run_with_pipeline.argparse.ArgumentParser.parse_args')
    def test_main_with_no_analysis(self, mock_parse_args):
        """Test main function with no analysis type."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.tickers = self.test_tickers
        mock_args.csv = None
        mock_args.analysis_type = "invalid"  # Invalid analysis type
        mock_args.skip_entropy = False
        mock_args.api_key = None
        mock_args.data_dir = None
        mock_args.output_dir = None
        mock_args.no_parallel = False
        mock_parse_args.return_value = mock_args
        
        # Run main with invalid analysis type
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', False):
            with patch.object(sys, 'argv', ['run_with_pipeline.py', '--tickers', 'TEST', '--analysis-type', 'invalid']):
                result = run_with_pipeline.main()
                
                # Assertions
                self.assertEqual(result, 0)  # Should still return 0

    def test_load_tickers_from_csv_with_real_file(self):
        """Test load_tickers_from_csv function with a real file."""
        # Create a test CSV file
        csv_path = os.path.join(self.test_dir, "test_tickers.csv")
        with open(csv_path, "w") as f:
            f.write("symbol,name,sector\n")
            f.write("AAPL,Apple Inc.,Technology\n")
            f.write("MSFT,Microsoft Corporation,Technology\n")
            f.write("GOOG,Alphabet Inc.,Technology\n")
        
        # Load tickers from the file
        tickers = run_with_pipeline.load_tickers_from_csv(csv_path)
        
        # Assertions
        self.assertEqual(len(tickers), 3)
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertIn("GOOG", tickers)
        
        # Create a CSV file without a symbol column
        csv_path = os.path.join(self.test_dir, "test_tickers_no_symbol.csv")
        with open(csv_path, "w") as f:
            f.write("ticker,name,sector\n")
            f.write("AAPL,Apple Inc.,Technology\n")
        
        # Load tickers from the file without symbol column
        tickers = run_with_pipeline.load_tickers_from_csv(csv_path)
        
        # Assertions
        self.assertEqual(tickers, [])

    def test_numpy_encoder_with_real_data(self):
        """Test NumpyEncoder class with real numpy data."""
        # Create test data with numpy types
        test_data = {
            "integer": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "nested": {
                "array": np.array([[1, 2], [3, 4]]),
                "float": np.float32(2.71)
            },
            "regular": "string"
        }
        
        # Use the encoder to dump the data to a real file
        json_path = os.path.join(self.test_dir, "test_data.json")
        with open(json_path, "w") as f:
            json.dump(test_data, f, cls=run_with_pipeline.NumpyEncoder)
        
        # Read the file back
        with open(json_path, "r") as f:
            loaded_data = json.load(f)
        
        # Assertions
        self.assertEqual(loaded_data["integer"], 42)
        self.assertEqual(loaded_data["float"], 3.14)
        self.assertEqual(loaded_data["array"], [1, 2, 3])
        self.assertEqual(loaded_data["nested"]["array"], [[1, 2], [3, 4]])
        
        # For float32, we need to use assertAlmostEqual due to precision issues
        self.assertAlmostEqual(loaded_data["nested"]["float"], 2.71, places=5)
        self.assertEqual(loaded_data["regular"], "string")

    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_main_help(self, mock_stderr, mock_stdout):
        """Test main function with --help option."""
        # Run main with --help
        with patch.object(sys, 'argv', ['run_with_pipeline.py', '--help']):
            with self.assertRaises(SystemExit) as cm:
                run_with_pipeline.main()
        
        # Should exit with code 0
        self.assertEqual(cm.exception.code, 0)
    
        # Check that help text was printed
        output = mock_stdout.getvalue()
        self.assertIn("Run Greek Energy Flow analysis using the Pipeline Manager", output)
        self.assertIn("--tickers", output)
        self.assertIn("--csv", output)

    def test_default_execution_path(self):
        """Test the default execution path when no command line args are provided."""
        # Patch sys.argv to simulate no command line args
        with patch.object(sys, 'argv', ['run_with_pipeline.py']):
            # Patch the functions that would be called
            with patch('run_with_pipeline.run_pipeline_analysis') as mock_run_pipeline:
                with patch('run_with_pipeline.run_pattern_analysis') as mock_run_pattern:
                    # Set HAS_PATTERN_ANALYZER to True to test both analysis types
                    with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', True):
                        # Run the main function
                        result = run_with_pipeline.main()
                        
                        # Assertions
                        self.assertEqual(result, 0)  # Should return 0 for success
                        
                        # Verify the default tickers were used
                        default_tickers = ["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"]
                        mock_run_pipeline.assert_called_once()
                        call_args = mock_run_pipeline.call_args[1]
                        self.assertEqual(call_args['tickers'], default_tickers)
                        self.assertEqual(call_args['api_key'], run_with_pipeline.POLYGON_API_KEY)
                        self.assertEqual(call_args['data_dir'], run_with_pipeline.DATA_DIR)
                        self.assertEqual(call_args['output_dir'], run_with_pipeline.OUTPUT_DIR)
                        self.assertEqual(call_args['skip_entropy'], False)
                        
                        # Verify pattern analysis was called with the same tickers
                        mock_run_pattern.assert_called_once()
                        pattern_call_args = mock_run_pattern.call_args[1]
                        self.assertEqual(pattern_call_args['tickers'], default_tickers)
                        self.assertEqual(pattern_call_args['output_dir'], run_with_pipeline.OUTPUT_DIR)

    @patch('run_with_pipeline.argparse.ArgumentParser.parse_args')
    @patch('run_with_pipeline.run_pipeline_analysis')
    @patch('run_with_pipeline.run_pattern_analysis')
    def test_main_with_both_analysis_types(self, mock_run_pattern, mock_run_pipeline, mock_parse_args):
        """Test main function with both analysis types."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.tickers = ["AAPL", "MSFT"]
        mock_args.csv = None
        mock_args.analysis_type = "both"
        mock_args.skip_entropy = False
        mock_args.api_key = "custom_api_key"
        mock_args.data_dir = "custom_data_dir"
        mock_args.output_dir = "custom_output_dir"
        mock_args.no_parallel = False
        mock_args.date = "2023-05-01"
        mock_args.cache_dir = "custom_cache_dir"
        mock_parse_args.return_value = mock_args
        
        # Set HAS_PATTERN_ANALYZER to True
        with patch.object(run_with_pipeline, 'HAS_PATTERN_ANALYZER', True):
            # Run main with both analysis types
            with patch.object(sys, 'argv', ['run_with_pipeline.py', '--tickers', 'AAPL', 'MSFT', '--analysis-type', 'both']):
                result = run_with_pipeline.main()
                
                # Assertions
                self.assertEqual(result, 0)  # Should return 0 for success
                
                # Verify both analysis types were called
                mock_run_pipeline.assert_called_once()
                mock_run_pattern.assert_called_once()
                
                # Check that the tickers were passed correctly
                pipeline_call_args = mock_run_pipeline.call_args[1]
                self.assertEqual(pipeline_call_args['tickers'], ["AAPL", "MSFT"])
                
                pattern_call_args = mock_run_pattern.call_args[1]
                self.assertEqual(pattern_call_args['tickers'], ["AAPL", "MSFT"])

    def test_numpy_encoder_complex_types(self):
        """Test NumpyEncoder with more complex numpy types."""
        # Create test data with complex numpy types
        test_data = {
            "datetime64": np.datetime64('2023-05-01'),
            "complex": np.complex128(1 + 2j),
            "bool": np.bool_(True),
            "structured": np.array([(1, 2.0)], dtype=[('a', np.int32), ('b', np.float64)])[0],  # Single structured element
            "masked": np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
        }
        
        # Use the encoder to convert to JSON
        json_str = json.dumps(test_data, cls=run_with_pipeline.NumpyEncoder)
        
        # Parse the JSON string back to Python objects
        loaded_data = json.loads(json_str)
        
        # Assertions
        self.assertEqual(loaded_data["datetime64"], "2023-05-01")
        self.assertEqual(loaded_data["complex"], {"real": 1.0, "imag": 2.0})
        self.assertEqual(loaded_data["bool"], True)
        self.assertEqual(loaded_data["structured"], {"a": 1, "b": 2.0})
        self.assertEqual(loaded_data["masked"], [1, None, 3])

    @patch('api_fetcher.fetch_options_chain_snapshot')
    @patch('api_fetcher.fetch_underlying_snapshot')
    @patch('api_fetcher.get_spot_price_from_snapshot')
    @patch('api_fetcher.preprocess_api_options_data')
    def test_end_to_end_integration(self, mock_preprocess, mock_get_price, 
                                   mock_fetch_underlying, mock_fetch_options):
        """Test end-to-end integration with mocked API calls."""
        # Set up mocks
        mock_fetch_underlying.return_value = {"last": {"price": 100.0}}
        mock_get_price.return_value = 100.0
        mock_fetch_options.return_value = {"results": [{"some": "data"}]}
        mock_preprocess.return_value = self.options_data
        
        # Create sample options data file
        options_path = os.path.join(self.data_dir, "options", "TEST_options.csv")
        self.options_data.to_csv(options_path, index=False)
        
        # Create sample price data file
        price_path = os.path.join(self.data_dir, "prices", "TEST_price.csv")
        self.price_data.to_csv(price_path, index=False)
        
        # Create a mock API key
        api_key = "TEST_API_KEY"
        
        # Call the function directly
        results = run_with_pipeline.run_pipeline_analysis(
            tickers=["TEST"],
            api_key=api_key,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            skip_entropy=True
        )
        
        # Check if results were returned
        self.assertIsNotNone(results)
        self.assertIn("TEST", results)
        self.assertEqual(results["TEST"]["status"], "success")
        
        # Check if output files were created
        output_file = os.path.join(self.output_dir, "TEST_analysis.json")
        self.assertTrue(os.path.exists(output_file), f"Output file not created: {output_file}")
        
        # Load and validate the output file
        with open(output_file, 'r') as f:
            file_results = json.load(f)
        
        # Basic validation of results structure
        self.assertIn("status", file_results)
        self.assertEqual(file_results["status"], "success")
        self.assertIn("timestamp", file_results)
        self.assertIn("analysis_results", file_results)
        
        # Check if summary file was created
        summary_files = [f for f in os.listdir(self.output_dir) if f.startswith("analysis_summary_")]
        self.assertTrue(len(summary_files) > 0, "No summary file created")
        
        # Verify mocks were called
        mock_fetch_underlying.assert_called_with("TEST", api_key)
        mock_fetch_options.assert_called_with("TEST", api_key)

if __name__ == "__main__":
    unittest.main()







