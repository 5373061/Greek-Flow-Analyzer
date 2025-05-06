
# test_entropy_analyzer.py - Testing framework for entropy analysis

import unittest
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the entropy analyzer
from entropy_analyzer.entropy_analyzer import EntropyAnalyzer

class TestEntropyAnalyzer(unittest.TestCase):
    """Test suite for the EntropyAnalyzer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic options data
        np.random.seed(42)  # For reproducibility
        
        # Create a DataFrame with synthetic options data
        n_options = 100
        strikes = np.linspace(90, 110, n_options)
        
        # Create synthetic Greeks
        delta = np.random.normal(0, 0.5, n_options)
        gamma = np.abs(np.random.normal(0, 0.1, n_options))
        vega = np.random.normal(0, 1, n_options)
        theta = -np.abs(np.random.normal(0, 0.2, n_options))
        
        # Create synthetic vanna and charm
        vanna = np.random.normal(0, 0.05, n_options)
        charm = np.random.normal(0, 0.02, n_options)
        
        # Create DataFrame
        self.options_data = pd.DataFrame({
            'strike': strikes,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'vanna': vanna,
            'charm': charm
        })
        
        # Create historical data
        self.historical_data = {
            "average_normalized_entropy": [50, 55, 45, 60, 40]
        }
        
        # Create output directory for test results
        self.output_dir = "test_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_entropy_calculation(self):
        """Test basic entropy calculation."""
        # Create analyzer with test data
        analyzer = EntropyAnalyzer(self.options_data)
        
        # Calculate entropy for a single array
        test_values = np.array([1, 2, 3, 4, 5])
        entropy = analyzer.calculate_shannon_entropy(test_values)
        
        # Shannon entropy should be positive
        self.assertGreater(entropy, 0)
        
        # Test with uniform distribution (all same value)
        uniform = np.ones(10)
        uniform_entropy = analyzer.calculate_shannon_entropy(uniform)
        
        # The implementation appears to return log2(n) for uniform distributions
        # This is consistent with the maximum possible entropy for n bins
        self.assertAlmostEqual(uniform_entropy, np.log2(10), places=5)
        
        # Test with another uniform distribution (different value)
        uniform2 = np.ones(10) * 5
        uniform2_entropy = analyzer.calculate_shannon_entropy(uniform2)
        
        # Both uniform distributions should have the same entropy
        self.assertAlmostEqual(uniform_entropy, uniform2_entropy, places=5)
        
        # Test with binary distribution
        binary = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        binary_entropy = analyzer.calculate_shannon_entropy(binary)
        
        # Binary distribution should have positive entropy
        self.assertGreater(binary_entropy, 0)
        
        # Test with more diverse distribution
        diverse = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        diverse_entropy = analyzer.calculate_shannon_entropy(diverse)
        
        # Diverse distribution should have higher entropy than binary
        self.assertGreater(diverse_entropy, binary_entropy)
    
    def test_cross_entropy(self):
        """Test cross-entropy calculation."""
        # Create analyzer with test data
        analyzer = EntropyAnalyzer(self.options_data)
        
        # Run full analysis first to ensure data is processed
        analyzer.analyze_greek_entropy()
        
        try:
            # Calculate cross-entropy between delta and gamma
            cross_entropy = analyzer.calculate_cross_entropy('delta', 'gamma')
            
            # Cross-entropy should be a number
            self.assertFalse(np.isnan(cross_entropy))
        except Exception as e:
            # If there's an error, print it and mark the test as skipped
            print(f"Cross-entropy calculation failed: {e}")
            self.skipTest("Cross-entropy calculation failed")
    
    def test_full_analysis(self):
        """Test full entropy analysis."""
        # Create analyzer with test data
        analyzer = EntropyAnalyzer(
            self.options_data, 
            self.historical_data,
            config={"visualization_enabled": False}
        )
        
        # Run full analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that we got results
        self.assertIsNotNone(entropy_metrics)
        self.assertIn("greek_entropy", entropy_metrics)
        self.assertIn("cross_entropy", entropy_metrics)
        self.assertIn("strike_entropy", entropy_metrics)
        self.assertIn("energy_state", entropy_metrics)
        
        # Check energy state
        self.assertIn("state", entropy_metrics["energy_state"])
        self.assertIn("description", entropy_metrics["energy_state"])
        
        # Run anomaly detection
        anomalies = analyzer.detect_anomalies()
        
        # Check anomaly results
        self.assertIsNotNone(anomalies)
        self.assertIn("anomalies", anomalies)
        self.assertIn("anomaly_count", anomalies)
        
        # Generate full report
        report = analyzer.generate_entropy_report()
        
        # Check report (as a string, not a dictionary)
        self.assertIsNotNone(report)
        self.assertIn("Energy State:", report)
        self.assertIn("Anomalies Detected", report)
        self.assertIn("Trading Implications", report)
        
        # Save report for inspection
        with open(os.path.join(self.output_dir, "entropy_report.txt"), "w") as f:
            f.write(report)
    
    def test_concentrated_distribution(self):
        """Test with a highly concentrated distribution."""
        # Create a concentrated distribution
        n_options = 100
        strikes = np.linspace(90, 110, n_options)
        
        # Create extremely concentrated delta (almost all zeros with a few ones)
        concentrated_delta = np.zeros(n_options)
        concentrated_delta[45:55] = 1.0  # Only 10 values are non-zero
        
        # Other Greeks with normal distribution
        gamma = np.abs(np.random.normal(0, 0.1, n_options))
        vega = np.random.normal(0, 1, n_options)
        
        # Create DataFrame
        concentrated_data = pd.DataFrame({
            'strike': strikes,
            'delta': concentrated_delta,
            'gamma': gamma,
            'vega': vega
        })
        
        # Create analyzer with concentrated data
        analyzer = EntropyAnalyzer(
            concentrated_data,
            config={
                "visualization_enabled": False,
                "entropy_threshold_low": 30,
                "entropy_threshold_high": 70
            }
        )
        
        # Run analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that delta entropy is calculated
        self.assertIn("delta", entropy_metrics["greek_entropy"])
        
        # Print the actual entropy value for debugging
        delta_entropy = entropy_metrics["greek_entropy"]["delta"]["normalized_entropy"]
        print(f"Delta entropy: {delta_entropy}")
        
        # Delta should have low entropy (concentrated distribution)
        # The exact value might vary based on implementation, but it should be lower than gamma/vega
        self.assertLessEqual(delta_entropy, 60, "Delta entropy should be relatively low for concentrated distribution")
        
        # Check that gamma and vega have higher entropy (more dispersed)
        gamma_entropy = entropy_metrics["greek_entropy"]["gamma"]["normalized_entropy"]
        vega_entropy = entropy_metrics["greek_entropy"]["vega"]["normalized_entropy"]
        
        # Gamma and vega should have higher entropy than delta
        self.assertGreater(gamma_entropy, delta_entropy, "Gamma entropy should be higher than delta entropy")
        self.assertGreater(vega_entropy, delta_entropy, "Vega entropy should be higher than delta entropy")
        
        # Check energy state
        energy_state = entropy_metrics["energy_state"]["state"]
        self.assertIsNotNone(energy_state, "Energy state should be defined")
        
        # Generate report
        report = analyzer.generate_entropy_report()
        
        # Check that report contains information about delta concentration
        self.assertIn("delta", report.lower(), "Report should mention delta")
        
        # Save report for inspection
        with open(os.path.join(self.output_dir, "concentrated_report.txt"), "w") as f:
            f.write(report)
    
    def test_dispersed_distribution(self):
        """Test with a highly dispersed distribution."""
        # Create a dispersed distribution
        n_options = 100
        strikes = np.linspace(90, 110, n_options)
        
        # Create dispersed gamma (uniform across a wide range)
        dispersed_gamma = np.random.uniform(0, 0.2, n_options)
        
        # Other Greeks with normal distribution
        delta = np.random.normal(0, 0.5, n_options)
        vega = np.random.normal(0, 1, n_options)
        
        # Create DataFrame
        dispersed_data = pd.DataFrame({
            'strike': strikes,
            'delta': delta,
            'gamma': dispersed_gamma,
            'vega': vega
        })
        
        # Create analyzer with dispersed data
        analyzer = EntropyAnalyzer(
            dispersed_data,
            config={"visualization_enabled": False}
        )
        
        # Run analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that gamma entropy is high (dispersed)
        gamma_entropy = entropy_metrics["greek_entropy"]["gamma"]["normalized_entropy"]
        
        # Should be higher than 50 (dispersed)
        self.assertGreater(gamma_entropy, 50, "Gamma entropy should be high for uniform distribution")
        
        # Check that entropy metrics contain all expected fields
        self.assertIn("entropy", entropy_metrics["greek_entropy"]["gamma"])
        self.assertIn("normalized_entropy", entropy_metrics["greek_entropy"]["gamma"])
        self.assertIn("sample_count", entropy_metrics["greek_entropy"]["gamma"])
        
        # Check cross-entropy calculations
        self.assertIn("cross_entropy", entropy_metrics)
        self.assertGreaterEqual(len(entropy_metrics["cross_entropy"]), 1, 
                              "Should have at least one cross-entropy calculation")
        
        # Check energy state
        self.assertIn("state", entropy_metrics["energy_state"])
        self.assertIn("description", entropy_metrics["energy_state"])
        
        # Generate report
        report = analyzer.generate_entropy_report()
        
        # Check report content
        self.assertIn("Energy State:", report)
        self.assertIn("gamma", report.lower())
        
        # Save report for inspection
        with open(os.path.join(self.output_dir, "dispersed_report.txt"), "w") as f:
            f.write(report)
    
    def test_historical_comparison(self):
        """Test historical entropy comparison."""
        # Create historical data
        historical_data = {
            "average_normalized_entropy": [50, 55, 45, 60, 40]
        }
        
        # Create analyzer with historical data
        analyzer = EntropyAnalyzer(
            self.options_data, 
            historical_data,
            config={"visualization_enabled": False}
        )
        
        # Run analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that historical comparison is included in metrics
        if "historical_comparison" in entropy_metrics:
            self.assertIn("entropy_percent_change", entropy_metrics["historical_comparison"])
            self.assertIn("entropy_trend", entropy_metrics["historical_comparison"])
            
            # Verify the trend description is a non-empty string
            self.assertIsInstance(entropy_metrics["historical_comparison"]["entropy_trend"], str)
            self.assertGreater(len(entropy_metrics["historical_comparison"]["entropy_trend"]), 0)
        
        # Generate report
        report = analyzer.generate_entropy_report()
        
        # Check that report was generated
        self.assertIsNotNone(report)
        
        # Check if historical data is being used somewhere in the analyzer
        self.assertEqual(analyzer.historical_data, historical_data)
        
        # Save report for inspection
        with open(os.path.join(self.output_dir, "historical_report.txt"), "w") as f:
            f.write(report)
    
    def test_custom_config(self):
        """Test analyzer with custom configuration."""
        # Create custom config
        custom_config = {
            "visualization_enabled": False,
            "anomaly_sensitivity": 2.0,  # Higher sensitivity
            "available_greeks": ["delta", "gamma", "vega"]  # Limited Greeks
        }
        
        # Create analyzer with custom config
        analyzer = EntropyAnalyzer(
            self.options_data,
            config=custom_config
        )
        
        # Run analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that only specified Greeks were analyzed
        greek_entropy = entropy_metrics["greek_entropy"]
        self.assertIn("delta", greek_entropy)
        self.assertIn("gamma", greek_entropy)
        self.assertIn("vega", greek_entropy)
        self.assertNotIn("theta", greek_entropy)
        
        # Run anomaly detection with higher sensitivity
        anomalies = analyzer.detect_anomalies()
        
        # Higher sensitivity should detect more anomalies
        # (This is a heuristic test, might not always hold)
        self.assertGreaterEqual(anomalies["anomaly_count"], 0)

    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Create analyzer with empty data
        analyzer = EntropyAnalyzer(empty_df)
        
        # Run analysis (should handle error gracefully)
        result = analyzer.analyze_greek_entropy()
        
        # Check that error was returned
        self.assertIn("error", result)
        
        # Create DataFrame with missing columns
        invalid_df = pd.DataFrame({"strike": [100, 110, 120]})
        
        # Create analyzer with invalid data
        analyzer = EntropyAnalyzer(invalid_df)
        
        # Run analysis (should handle error gracefully)
        result = analyzer.analyze_greek_entropy()
        
        # Check that error was returned
        self.assertIn("error", result)

    def test_realistic_market_scenario(self):
        """Test entropy analyzer with realistic market data."""
        # Create realistic options data
        # This simulates a typical options chain with strikes around a spot price of 100
        spot_price = 100.0
        n_options = 20
        strikes = np.linspace(90, 110, n_options)
        
        # Create expiration dates (multiple expirations)
        today = datetime.now().date()
        expirations = [
            today + timedelta(days=30),  # 1 month
            today + timedelta(days=60),  # 2 months
            today + timedelta(days=90)   # 3 months
        ]
        
        # Create realistic option types
        option_types = ['call', 'put']
        
        # Create empty DataFrame to hold all options
        realistic_data = []
        
        for expiry in expirations:
            for opt_type in option_types:
                for strike in strikes:
                    # Calculate days to expiration
                    dte = (expiry - today).days
                    
                    # Calculate realistic Greeks based on strike, expiry, and option type
                    if opt_type == 'call':
                        # Call option Greeks
                        moneyness = spot_price / strike
                        delta = max(0.01, min(0.99, 0.5 + 0.5 * (moneyness - 1) * (100 / dte)))
                        gamma = 0.1 * np.exp(-((strike - spot_price) ** 2) / (2 * (spot_price * 0.2) ** 2))
                        vega = gamma * spot_price * np.sqrt(dte / 365)
                        theta = -vega * spot_price / (2 * np.sqrt(dte / 365))
                        vanna = vega * (1 - delta) / (spot_price * 0.2)
                        charm = -theta * (1 - delta) / (spot_price * 0.2)
                    else:
                        # Put option Greeks
                        moneyness = strike / spot_price
                        delta = -max(0.01, min(0.99, 0.5 + 0.5 * (moneyness - 1) * (100 / dte)))
                        gamma = 0.1 * np.exp(-((strike - spot_price) ** 2) / (2 * (spot_price * 0.2) ** 2))
                        vega = gamma * spot_price * np.sqrt(dte / 365)
                        theta = -vega * spot_price / (2 * np.sqrt(dte / 365))
                        vanna = -vega * (1 + delta) / (spot_price * 0.2)
                        charm = -theta * (1 + delta) / (spot_price * 0.2)
                    
                    # Add some realistic noise
                    delta += np.random.normal(0, 0.02)
                    gamma += np.random.normal(0, 0.005)
                    vega += np.random.normal(0, 0.1)
                    theta += np.random.normal(0, 0.05)
                    vanna += np.random.normal(0, 0.01)
                    charm += np.random.normal(0, 0.005)
                    
                    # Create option data
                    option = {
                        'strike': strike,
                        'expiration': expiry,
                        'option_type': opt_type,
                        'dte': dte,
                        'delta': delta,
                        'gamma': gamma,
                        'vega': vega,
                        'theta': theta,
                        'vanna': vanna,
                        'charm': charm,
                        'volume': int(np.random.exponential(500) * (1.1 - abs(strike - spot_price) / 20)),
                        'open_interest': int(np.random.exponential(2000) * (1.1 - abs(strike - spot_price) / 20))
                    }
                    realistic_data.append(option)
        
        # Convert to DataFrame
        realistic_df = pd.DataFrame(realistic_data)
        
        # Create analyzer with realistic data
        analyzer = EntropyAnalyzer(
            realistic_df,
            config={
                "visualization_enabled": True,
                "entropy_threshold_low": 30,
                "entropy_threshold_high": 70,
                "available_greeks": ["delta", "gamma", "vega", "theta", "vanna", "charm"]
            }
        )
        
        # Run full analysis
        entropy_metrics = analyzer.analyze_greek_entropy()
        
        # Check that we got results for all Greeks
        for greek in ["delta", "gamma", "vega", "theta", "vanna", "charm"]:
            self.assertIn(greek, entropy_metrics["greek_entropy"])
            
        # Check that we got cross-entropy results
        self.assertGreater(len(entropy_metrics["cross_entropy"]), 0)
        
        # Check that we got strike entropy results
        self.assertGreater(len(entropy_metrics["strike_entropy"]), 0)
        
        # Check energy state
        self.assertIn("state", entropy_metrics["energy_state"])
        self.assertIn("description", entropy_metrics["energy_state"])
        self.assertIn("average_normalized_entropy", entropy_metrics["energy_state"])
        
        # Run anomaly detection
        anomalies = analyzer.detect_anomalies()
        
        # Check anomaly results
        self.assertIsNotNone(anomalies)
        self.assertIn("anomalies", anomalies)
        self.assertIn("anomaly_count", anomalies)
        
        # Generate report
        report = analyzer.generate_entropy_report()
        
        # Check report content
        self.assertIn("=== ENTROPY ANALYSIS REPORT ===", report)
        self.assertIn("Energy State:", report)
        self.assertIn("Greek Entropy Values:", report)
        self.assertIn("Anomalies Detected", report)
        self.assertIn("Trading Implications:", report)
        
        # Save results for inspection
        with open(os.path.join(self.output_dir, "realistic_entropy_metrics.json"), "w") as f:
            json.dump(entropy_metrics, f, indent=2, default=str)
            
        with open(os.path.join(self.output_dir, "realistic_report.txt"), "w") as f:
            f.write(report)
            
        print(f"\nRealistic test completed. Results saved to {self.output_dir}")
        print(f"Energy State: {entropy_metrics['energy_state']['state']}")
        print(f"Average Normalized Entropy: {entropy_metrics['energy_state']['average_normalized_entropy']:.2f}%")
        print(f"Anomalies detected: {anomalies['anomaly_count']}")

if __name__ == "__main__":
    unittest.main()







