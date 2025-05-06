import numpy as np
from scipy.stats import norm
from typing import Tuple

def calculate_black_scholes_greeks(
    S: float,    # Current stock price
    K: float,    # Strike price
    T: float,    # Time to expiration (in years)
    r: float,    # Risk-free rate
    sigma: float, # Volatility
    option_type: str
) -> Tuple[float, float, float, float]:
    """Calculate Black-Scholes Greeks (delta, gamma, theta, vega)"""
    
    # Input validation
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
        
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate common terms
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)
    
    if option_type.lower() == 'call':
        delta = N_d1
        theta = (-S*sigma*n_d1/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N_d2)
    else:  # put
        delta = N_d1 - 1
        theta = (-S*sigma*n_d1/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*(1-N_d2))
    
    # Greeks same for calls and puts
    gamma = n_d1/(S*sigma*np.sqrt(T))
    vega = S*np.sqrt(T)*n_d1
    
    return delta, gamma, theta, vega