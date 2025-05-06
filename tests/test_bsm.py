import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Greek_Energy_FlowII import BlackScholesModel
import logging

def validate_greeks(result, moneyness, option_type):
    """Validate Greek values match expected behavior"""
    try:
        # Delta checks
        if option_type == 'call':
            assert 0 <= result['delta'] <= 1, f"Call delta should be between 0 and 1, got {result['delta']}"
            if moneyness == 'ITM':
                assert result['delta'] > 0.5, f"ITM call delta should be > 0.5, got {result['delta']}"
            elif moneyness == 'OTM':
                assert result['delta'] < 0.5, f"OTM call delta should be < 0.5, got {result['delta']}"
        else:  # put
            assert -1 <= result['delta'] <= 0, f"Put delta should be between -1 and 0, got {result['delta']}"
            if moneyness == 'ITM':  # Note: For puts, ITM means S < K
                # Fix: ITM puts should have delta between -1 and -0.5
                assert result['delta'] < -0.5, f"ITM put delta should be < -0.5, got {result['delta']}"
            elif moneyness == 'OTM':  # Note: For puts, OTM means S > K
                # Fix: OTM puts should have delta between -0.5 and 0
                assert result['delta'] > -0.5, f"OTM put delta should be > -0.5, got {result['delta']}"
        
        # Gamma checks (same for calls and puts)
        assert result['gamma'] >= 0, f"Gamma should be non-negative, got {result['gamma']}"
        if moneyness == 'ATM':
            assert result['gamma'] > 0.01, f"ATM gamma should be significant, got {result['gamma']}"
        
        # Theta checks
        if option_type == 'call':
            assert result['theta'] <= 0, f"Call theta should be non-positive, got {result['theta']}"
        else:
            # Put theta can be positive for deep ITM puts
            if moneyness == 'OTM':
                assert result['theta'] <= 0, f"OTM put theta should be non-positive, got {result['theta']}"
        
        # Vega checks (same for calls and puts)
        assert result['vega'] >= 0, f"Vega should be non-negative, got {result['vega']}"
        if moneyness == 'ATM':
            assert result['vega'] > 0, f"ATM vega should be positive, got {result['vega']}"

    except AssertionError as e:
        logging.error(f"Greek validation failed: {str(e)}")
        raise

def print_summary(total_tests, passed_tests):
    print("\nTEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests:  {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("=" * 50)

def test_bsm(test_suites=None):
    # Create test suite with multiple scenarios
    if test_suites is None:
        test_suites = {
            "Standard Options": [
                # Calls
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},  # ATM
                {'S': 110, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},  # ITM
                {'S': 100, 'K': 110, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},  # OTM
                
                # Puts
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'},   # ATM
                {'S': 90, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'},    # ITM
                {'S': 110, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'}    # OTM
            ],
            "Time Decay": [
                {'S': 100, 'K': 100, 'T': 7/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},   # 1 week
                {'S': 100, 'K': 100, 'T': 180/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'}, # 6 months
                {'S': 100, 'K': 100, 'T': 7/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'},    # 1 week
                {'S': 100, 'K': 100, 'T': 180/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'}   # 6 months
            ],
            "Volatility": [
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.1, 'option_type': 'call'},  # Low vol
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.5, 'option_type': 'call'},  # High vol
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.1, 'option_type': 'put'},   # Low vol
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.5, 'option_type': 'put'}    # High vol
            ],
            "Interest Rate": [
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.01, 'sigma': 0.2, 'option_type': 'call'},  # Low rate
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.10, 'sigma': 0.2, 'option_type': 'call'},  # High rate
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.01, 'sigma': 0.2, 'option_type': 'put'},   # Low rate
                {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.10, 'sigma': 0.2, 'option_type': 'put'}    # High rate
            ]
        }

    bsm = BlackScholesModel()
    total_tests = sum(len(suite) for suite in test_suites.values())
    passed_tests = 0
    
    print("\nBlack-Scholes Model Test Results:")
    print("=" * 80)
    
    for suite_name, test_inputs in test_suites.items():
        print(f"\n{suite_name}:")
        print("-" * 50)
        
        for inputs in test_inputs:
            result = bsm.calculate(**inputs)
            if result:
                desc = f"{inputs['option_type'].upper()}"
                if 'Time' in suite_name:
                    desc += f" ({inputs['T']*365:.0f}d)"
                elif 'Volatility' in suite_name:
                    desc += f" ({inputs['sigma']:.1%} vol)"
                elif 'Interest' in suite_name:
                    desc += f" ({inputs['r']:.1%} rate)"
                    
                # Fix: Separate moneyness logic for puts and calls
                if inputs['option_type'] == 'call':
                    moneyness = "ATM" if inputs['S'] == inputs['K'] else "ITM" if inputs['S'] > inputs['K'] else "OTM"
                else:  # put
                    moneyness = "ATM" if inputs['S'] == inputs['K'] else "ITM" if inputs['S'] < inputs['K'] else "OTM"
                
                valid = validate_greeks(result, moneyness, inputs['option_type'])
                if valid:
                    passed_tests += 1
                    
                print(f"\n{desc} ({moneyness}):")  # Added moneyness to output
                print(f"Validation: {'✓' if valid else '✗'}")
                print("-" * 30)
                for greek, value in result.items():
                    print(f"{greek.title():>10}: {value:>10.4f}")
    
    print_summary(total_tests, passed_tests)

def plot_greek_surface(bsm, greek_name, S_range, K_range, option_type='call'):
    """Plot 3D surface of specified Greek"""
    S, K = np.meshgrid(S_range, K_range)
    Z = np.zeros_like(S)
    
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            result = bsm.calculate(
                S=S[i,j], 
                K=K[i,j], 
                T=30/365,
                r=0.05,
                sigma=0.2,
                option_type=option_type
            )
            if result:
                Z[i,j] = result[greek_name.lower()]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S, K, Z, cmap='viridis')
    
    plt.colorbar(surf)
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel(greek_name)
    plt.title(f'{greek_name} Surface for {option_type.upper()}s')
    plt.show()

def test_edge_cases():
    """Test BSM model with edge cases"""
    bsm = BlackScholesModel()
    edge_cases = {
        "Deep ITM/OTM": [
            {'S': 150, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},  # Deep ITM call
            {'S': 50, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},   # Deep OTM call
            {'S': 50, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'},    # Deep ITM put
            {'S': 150, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'put'}    # Deep OTM put
        ],
        "Extreme Time": [
            {'S': 100, 'K': 100, 'T': 1/365, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},   # 1 day
            {'S': 100, 'K': 100, 'T': 2.0, 'r': 0.05, 'sigma': 0.2, 'option_type': 'call'},     # 2 years
        ],
        "Extreme Volatility": [
            {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 0.01, 'option_type': 'call'}, # Very low vol
            {'S': 100, 'K': 100, 'T': 30/365, 'r': 0.05, 'sigma': 1.0, 'option_type': 'call'},  # Very high vol
        ]
    }
    
    return test_bsm(edge_cases)

def main():
    """Main test runner with visualization"""
    bsm = BlackScholesModel()
    
    # Run standard tests
    print("\n=== Running Standard Tests ===")
    test_bsm()
    
    # Run edge cases
    print("\n=== Running Edge Cases ===")
    test_edge_cases()
    
    # Generate Greek surfaces
    print("\n=== Generating Greek Surfaces ===")
    S_range = np.linspace(80, 120, 20)
    K_range = np.linspace(80, 120, 20)
    
    for greek in ['Delta', 'Gamma', 'Theta', 'Vega']:
        plot_greek_surface(bsm, greek, S_range, K_range, 'call')
        plot_greek_surface(bsm, greek, S_range, K_range, 'put')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()