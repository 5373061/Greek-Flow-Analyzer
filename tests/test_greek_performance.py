import numpy as np
import pandas as pd
import time
from typing import Dict, List
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.greek_analyzer import GreekEnergyAnalyzer
from bsm_model import BlackScholesModel

logger = logging.getLogger(__name__)

def generate_test_options(n_options: int = 1000) -> Dict[str, List[Dict]]:
    """Generate synthetic options data for testing"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random but realistic option parameters
    S = np.full(n_options, 100.0)  # Underlying price
    K = np.random.normal(100, 20, n_options)  # Strike prices
    T = np.random.uniform(0.1, 2.0, n_options)  # Time to expiry
    r = np.full(n_options, 0.05)  # Risk-free rate
    sigma = np.random.uniform(0.2, 0.8, n_options)  # Implied volatility
    volume = np.random.randint(1, 1000, n_options)  # Trading volume
    
    # Create option chains
    calls = []
    puts = []
    for i in range(n_options):
        option = {
            'underlying_price': S[i],
            'strike': K[i],
            'days_to_expiry': T[i] * 365,
            'interest_rate': r[i],
            'implied_volatility': sigma[i],
            'volume': volume[i]
        }
        calls.append(option.copy())
        puts.append(option.copy())
    
    return {'calls': calls, 'puts': puts}

def benchmark_greek_calculation(n_options: int = 1000, batch_sizes: List[int] = [100, 500, 1000, 2000]):
    """Benchmark Greek calculation performance with different batch sizes"""
    options_chain = generate_test_options(n_options)
    results = {}
    
    logger.info(f"Benchmarking with {n_options} options per type (calls/puts)")
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        analyzer = GreekEnergyAnalyzer(batch_size=batch_size)
        
        start_time = time.time()
        energy_flow = analyzer.analyze_chain_energy(options_chain, "TEST")
        elapsed = time.time() - start_time
        
        results[batch_size] = {
            'time': elapsed,
            'options_per_second': (2 * n_options) / elapsed  # Both calls and puts
        }
        
        logger.info(f"Batch size {batch_size:4d}: {elapsed:.3f}s ({results[batch_size]['options_per_second']:.0f} options/s)")
    
    return results

def verify_greek_accuracy(n_samples: int = 100):
    """Verify the accuracy of vectorized calculations against single-option calculations"""
    # Generate test data
    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.1, 2.0, n_samples)
    r = np.full(n_samples, 0.05)
    sigma = np.random.uniform(0.1, 0.8, n_samples)
    option_types = np.random.choice(['call', 'put'], n_samples)
    
    # Calculate Greeks using both methods
    bsm = BlackScholesModel()
    batch_results = bsm.calculate_batch(S, K, T, r, sigma, option_types)
    
    # Calculate individual results
    individual_results = {
        'delta': np.zeros(n_samples),
        'gamma': np.zeros(n_samples),
        'theta': np.zeros(n_samples),
        'vega': np.zeros(n_samples)
    }
    
    for i in range(n_samples):
        result = bsm.calculate(S[i], K[i], T[i], r[i], sigma[i], option_types[i])
        for greek in individual_results:
            individual_results[greek][i] = result[greek]
    
    # Compare results
    max_diffs = {}
    for greek in individual_results:
        diff = np.abs(batch_results[greek] - individual_results[greek])
        max_diff = np.max(diff)
        max_diffs[greek] = max_diff
        logger.info(f"{greek}: Max difference = {max_diff:.2e}")
    
    return max_diffs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    logger.info("\nRunning performance benchmarks...")
    perf_results = benchmark_greek_calculation(n_options=5000)
    
    # Verify accuracy
    logger.info("\nVerifying calculation accuracy...")
    accuracy_results = verify_greek_accuracy(n_samples=1000)