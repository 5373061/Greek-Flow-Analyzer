# bsm_model.py
import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class BlackScholesModel:
    """Vectorized Black-Scholes Model implementation."""
    
    @staticmethod
    def _calculate_d1_d2_n(
        S: np.ndarray, 
        K: np.ndarray, 
        T: np.ndarray, 
        r: np.ndarray, 
        sigma: np.ndarray
    ) -> tuple:
        """
        Calculate d1 and d2 parameters for BS model with vectorization.
        
        Returns:
            tuple: (d1, d2, norm.pdf)
        """
        # Input validation with vectorized operations
        epsilon = 1e-7
        T = np.maximum(T, epsilon)  # Prevent division by zero
        sigma = np.maximum(sigma, epsilon)  # Prevent unstable calculations
        S = np.maximum(S, epsilon)
        K = np.maximum(K, epsilon)
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2, norm.pdf
    
    @staticmethod
    def calculate_batch(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
        option_types: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of all Greeks for a batch of options.
        
        Args:
            S, K, T, r, sigma: numpy arrays of option parameters
            option_types: numpy array of option types ('call' or 'put')
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of Greeks with vectorized results
        """
        try:
            # Convert inputs to numpy arrays if not already
            S = np.asarray(S, dtype=np.float64)
            K = np.asarray(K, dtype=np.float64)
            T = np.asarray(T, dtype=np.float64)
            r = np.asarray(r, dtype=np.float64)
            sigma = np.asarray(sigma, dtype=np.float64)
            is_call = np.asarray([t.lower() == 'call' for t in option_types])
            
            # Calculate d1, d2 for all options at once
            d1, d2, n = BlackScholesModel._calculate_d1_d2_n(S, K, T, r, sigma)
            
            # Vectorized calculations for N(d1), N(d2)
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = n(d1)
            
            # Calculate all Greeks with vectorization
            sqrt_T = np.sqrt(T)
            exp_rT = np.exp(-r * T)
            
            # First order Greeks
            delta = np.where(is_call, N_d1, N_d1 - 1)
            gamma = n_d1 / (S * sigma * sqrt_T)
            
            theta_call = -(S * sigma * n_d1) / (2 * sqrt_T) - r * K * exp_rT * N_d2
            theta_put = -(S * sigma * n_d1) / (2 * sqrt_T) + r * K * exp_rT * (1 - N_d2)
            theta = np.where(is_call, theta_call, theta_put)
            
            vega = S * sqrt_T * n_d1
            rho = np.where(is_call, 
                        K * T * exp_rT * N_d2,
                        -K * T * exp_rT * (1 - N_d2))
            
            # Second order Greeks
            vanna = -S * n_d1 * d2 / sigma
            
            # Vectorized charm calculation
            charm = -n_d1 * ((r - sigma**2/2)/(sigma * sqrt_T) - d2/(2*T))
            
            volga = vega * (d1 * d2 / sigma)
            
            # Handle potential instabilities in higher order Greeks
            speed = np.where(sigma * sqrt_T < 1e-6,
                         np.nan,
                         -gamma / S * (1 + d1 / (sigma * sqrt_T)))
            
            zomma = gamma * ((d1 * d2 - 1) / sigma)
            
            # Color calculation with stability check
            color_denom = 2 * S * T * sigma * sqrt_T
            color = np.where(color_denom < 1e-6,
                         np.nan,
                         -gamma * ((1 - d1 * d2)/(2*T) + 
                                 (d1 * r + d2 * sigma**2/2)/(sigma * sqrt_T)))
            
            return {
                'price': np.where(is_call,
                              S * N_d1 - K * exp_rT * N_d2,
                              K * exp_rT * (1 - N_d2) - S * (1 - N_d1)),
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'charm': charm,
                'vanna': vanna,
                'volga': volga,
                'speed': speed,
                'zomma': zomma,
                'color': color
            }
            
        except Exception as e:
            logger.error(f"Error in vectorized BSM calculation: {str(e)}")
            return {k: np.full_like(S, np.nan) for k in [
                'price', 'delta', 'gamma', 'theta', 'vega', 'rho',
                'charm', 'vanna', 'volga', 'speed', 'zomma', 'color'
            ]}
    
    @staticmethod
    def calculate(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a single option (uses vectorized implementation internally).
        """
        result = BlackScholesModel.calculate_batch(
            np.array([S]),
            np.array([K]),
            np.array([T]),
            np.array([r]),
            np.array([sigma]),
            np.array([option_type])
        )
        return {k: float(v[0]) for k, v in result.items()}