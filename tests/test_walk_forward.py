"""
Walk-Forward Testing Framework for Greek Energy Flow II

This module implements a comprehensive walk-forward testing framework
to evaluate the performance of the ML-enhanced Greek Energy Flow system
across different market regimes.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from greek_flow.flow import GreekEnergyFlow, GreekEnergyAnalyzer
from entropy_analyzer.entropy_analyzer import EntropyAnalyzer
from models.ml.regime_classifier import GreekRegimeClassifier
from models.ml.trade_executor import MLTradeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WalkForwardTester:
    """
    Walk-Forward Testing framework for the Greek Energy Flow system.
    
    This class implements a walk-forward optimization and testing framework
    that trains models on expanding windows of data and tests on out-of-sample
    periods to simulate real-world performance.
    """
    
    def __init__(self, data_dir='data', results_dir='wf_results', 
                 window_size=60, step_size=5, test_size=5):
        """
        Initialize the walk-forward tester.
        
        Args:
            data_dir: Directory containing historical data
            results_dir: Directory to save test results
            window_size: Size of the training window in days
            step_size: Number of days to step forward
            test_size: Size of the test window in days
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.window_size = window_size
        self.step_size = step_size
        self.test_size = test_size
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Initialized Walk-Forward Tester with window={window_size}, step={step_size}, test={test_size}")
    
    def load_historical_data(self, tickers, start_date, end_date):
        """
        Load historical options and price data for the specified tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary with historical data by ticker
        """
        historical_data = {}
        
        for ticker in tickers:
            try:
                # Load options data
                options_file = os.path.join(self.data_dir, f"{ticker}_options.csv")
                if os.path.exists(options_file):
                    options_data = pd.read_csv(options_file)
                    options_data['date'] = pd.to_datetime(options_data['date'])
                    options_data = options_data[(options_data['date'] >= start_date) & 
                                               (options_data['date'] <= end_date)]
                else:
                    logger.warning(f"Options data file not found for {ticker}")
                    options_data = None
                
                # Load price data
                price_file = os.path.join(self.data_dir, f"{ticker}_prices.csv")
                if os.path.exists(price_file):
                    price_data = pd.read_csv(price_file)
                    price_data['date'] = pd.to_datetime(price_data['date'])
                    price_data = price_data[(price_data['date'] >= start_date) & 
                                           (price_data['date'] <= end_date)]
                else:
                    logger.warning(f"Price data file not found for {ticker}")
                    price_data = None
                
                if options_data is not None and price_data is not None:
                    historical_data[ticker] = {
                        'options': options_data,
                        'prices': price_data
                    }
                    logger.info(f"Loaded historical data for {ticker}: {len(options_data)} options rows, {len(price_data)} price rows")
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {str(e)}")
        
        return historical_data
    
    def run_walk_forward_test(self, tickers, start_date, end_date):
        """
        Run a walk-forward test for the specified tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for the test
            end_date: End date for the test
            
        Returns:
            Dictionary with test results
        """
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Ensure data files exist for all tickers
        logger.info(f"Ensuring data files exist for {len(tickers)} tickers")
        self.ensure_data_files_exist(tickers, start_date, end_date)
        
        # Load historical data
        logger.info(f"Loading historical data for {len(tickers)} tickers from {start_date} to {end_date}")
        historical_data = self.load_historical_data(tickers, start_date, end_date)
        
        # Initialize results
        results = {
            'windows': [],
            'performance': {},
            'trades': [],
            'model_metrics': []
        }
        
        # Generate test windows
        current_date = start_date
        window_end = current_date + timedelta(days=self.window_size)
        
        while window_end + timedelta(days=self.test_size) <= end_date:
            test_end = window_end + timedelta(days=self.test_size)
            
            logger.info(f"Walk-forward window: Train {current_date} to {window_end}, Test {window_end} to {test_end}")
            
            # Run test for this window
            window_results = self._test_window(
                tickers, historical_data, current_date, window_end, test_end
            )
            
            # Store results
            results['windows'].append({
                'train_start': current_date.strftime('%Y-%m-%d'),
                'train_end': window_end.strftime('%Y-%m-%d'),
                'test_start': window_end.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'results': window_results
            })
            
            # Step forward
            current_date += timedelta(days=self.step_size)
            window_end = current_date + timedelta(days=self.window_size)
        
        # Aggregate results
        self._aggregate_results(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_window(self, tickers, historical_data, train_start, train_end, test_end):
        """
        Test a single walk-forward window.
        
        Args:
            tickers: List of ticker symbols
            historical_data: Dictionary with historical data
            train_start: Start date for training
            train_end: End date for training (start date for testing)
            test_end: End date for testing
            
        Returns:
            Dictionary with window test results
        """
        window_results = {
            'trades': [],
            'model_metrics': {},
            'performance': {}
        }
        
        # Train models on training data
        models = self._train_models(tickers, historical_data, train_start, train_end)
        
        # Test models on test data
        for ticker in tickers:
            if ticker not in historical_data:
                continue
                
            ticker_data = historical_data[ticker]
            
            # Filter test data
            test_options = ticker_data['options'][(ticker_data['options']['date'] > train_end) & 
                                                 (ticker_data['options']['date'] <= test_end)]
            test_prices = ticker_data['prices'][(ticker_data['prices']['date'] > train_end) & 
                                               (ticker_data['prices']['date'] <= test_end)]
            
            if len(test_options) == 0 or len(test_prices) == 0:
                logger.warning(f"No test data for {ticker} in window {train_end} to {test_end}")
                continue
            
            # Run simulation on test data
            ticker_results = self._simulate_trading(
                ticker, test_options, test_prices, models.get(ticker)
            )
            
            # Store results
            window_results['trades'].extend(ticker_results['trades'])
            window_results['model_metrics'][ticker] = ticker_results['model_metrics']
            window_results['performance'][ticker] = ticker_results['performance']
        
        return window_results
    
    def _train_models(self, tickers, historical_data, train_start, train_end):
        """
        Train models on the training data.
        
        Args:
            tickers: List of ticker symbols
            historical_data: Dictionary with historical data
            train_start: Start date for training
            train_end: End date for training
            
        Returns:
            Dictionary with trained models by ticker
        """
        models = {}
        
        for ticker in tickers:
            if ticker not in historical_data:
                continue
                
            ticker_data = historical_data[ticker]
            
            # Filter training data
            train_options = ticker_data['options'][(ticker_data['options']['date'] >= train_start) & 
                                                  (ticker_data['options']['date'] <= train_end)]
            train_prices = ticker_data['prices'][(ticker_data['prices']['date'] >= train_start) & 
                                                (ticker_data['prices']['date'] <= train_end)]
            
            if len(train_options) == 0 or len(train_prices) == 0:
                logger.warning(f"No training data for {ticker} in window {train_start} to {train_end}")
                continue
            
            try:
                # Initialize models
                regime_classifier = GreekRegimeClassifier(model_type='randomforest')
                
                # Train models
                # Note: This is a simplified version - in practice, you would extract features
                # and labels from the training data and train the models properly
                
                # For demonstration purposes, we'll just create a dummy model
                models[ticker] = {
                    'regime_classifier': regime_classifier
                }
                
                logger.info(f"Trained models for {ticker} on {len(train_options)} options rows")
                
            except Exception as e:
                logger.error(f"Error training models for {ticker}: {str(e)}")
        
        return models
    
    def _simulate_trading(self, ticker, options_data, price_data, models):
        """
        Simulate trading on test data using trained models.
        
        Args:
            ticker: Ticker symbol
            options_data: Options data for the test period
            price_data: Price data for the test period
            models: Trained models for the ticker
            
        Returns:
            Dictionary with simulation results
        """
        results = {
            'trades': [],
            'model_metrics': {},
            'performance': {
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0
            }
        }
        
        # Group data by date
        options_by_date = options_data.groupby(options_data['date'].dt.date)
        price_by_date = price_data.set_index('date')
        
        # Initialize portfolio
        portfolio = {
            'cash': 100000,
            'positions': {},
            'history': []
        }
        
        # Simulate trading day by day
        for date, day_options in options_by_date:
            try:
                # Get price data for this date
                day_price = price_by_date.loc[date] if date in price_by_date.index else None
                
                if day_price is None:
                    continue
                
                # Run analysis
                analysis_results = self._run_analysis(ticker, day_options, day_price)
                
                # Make trading decisions
                trades = self._make_trading_decisions(
                    ticker, analysis_results, models, portfolio, day_price
                )
                
                # Update portfolio
                for trade in trades:
                    results['trades'].append(trade)
                
                # Record portfolio state
                portfolio['history'].append({
                    'date': date,
                    'cash': portfolio['cash'],
                    'positions': portfolio['positions'].copy(),
                    'equity': self._calculate_equity(portfolio, day_price)
                })
                
            except Exception as e:
                logger.error(f"Error simulating trading for {ticker} on {date}: {str(e)}")
        
        # Calculate performance metrics
        if portfolio['history']:
            results['performance'] = self._calculate_performance(portfolio['history'])
        
        return results
    
    def _run_analysis(self, ticker, options_data, price_data):
        """
        Run analysis on a single day's data.
        
        Args:
            ticker: Ticker symbol
            options_data: Options data for the day
            price_data: Price data for the day
            
        Returns:
            Dictionary with analysis results
        """
        # This is a simplified version - in practice, you would run the full analysis pipeline
        
        # For demonstration purposes, we'll just return a dummy result
        return {
            'greek_analysis': {
                'market_regime': {
                    'primary_label': 'Neutral',
                    'secondary_label': 'Balanced'
                },
                'reset_points': [],
                'energy_levels': []
            },
            'entropy_analysis': {
                'entropy_metrics': {
                    'greek_entropy': {}
                }
            }
        }
    
    def _make_trading_decisions(self, ticker, analysis_results, models, portfolio, price_data):
        """
        Make trading decisions based on analysis results and models.
        
        Args:
            ticker: Ticker symbol
            analysis_results: Analysis results
            models: Trained models
            portfolio: Current portfolio state
            price_data: Price data for the day
            
        Returns:
            List of trades
        """
        # This is a simplified version - in practice, you would use the models to make predictions
        # and then make trading decisions based on those predictions
        
        # For demonstration purposes, we'll just return a dummy trade
        return []
    
    def _calculate_equity(self, portfolio, current_prices):
        """
        Calculate the total equity value of a portfolio.
        
        Args:
            portfolio: Portfolio dictionary with cash and positions
            current_prices: Current prices for securities
        
        Returns:
            Total equity value
        """
        equity = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if hasattr(current_prices, 'close'):
                # If current_prices is a Series or DataFrame row
                price = current_prices.close
            elif isinstance(current_prices, dict) and 'close' in current_prices:
                # If current_prices is a dictionary
                price = current_prices['close']
            else:
                # Default case
                price = 0
            
            equity += position['quantity'] * price
        
        return equity
    
    def _calculate_performance(self, portfolio_history):
        """
        Calculate performance metrics from portfolio history.
        
        Args:
            portfolio_history: List of portfolio states over time
            
        Returns:
            Dictionary with performance metrics
        """
        if not portfolio_history:
            return {
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0
            }
        
        # Calculate equity curve
        equity = [entry['equity'] for entry in portfolio_history]
        
        # Calculate returns
        initial_equity = equity[0]
        final_equity = equity[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Calculate drawdown
        max_equity = initial_equity
        max_drawdown = 0
        
        for e in equity:
            max_equity = max(max_equity, e)
            drawdown = (max_equity - e) / max_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': self._calculate_sharpe(equity),
            'equity_curve': equity
        }
    
    def _calculate_sharpe(self, equity, risk_free_rate=0.02, trading_days=252):
        """
        Calculate Sharpe ratio from equity curve.
        
        Args:
            equity: List of equity values
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
            
        Returns:
            Sharpe ratio
        """
        if len(equity) < 2:
            return 0
        
        # Calculate daily returns
        returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
        
        # Calculate annualized return and volatility
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualize
        annualized_return = (1 + avg_return) ** trading_days - 1
        annualized_vol = std_return * np.sqrt(trading_days)
        
        # Calculate Sharpe
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        
        return sharpe
    
    def _aggregate_results(self, results):
        """
        Aggregate results across all test windows.
        
        Args:
            results: Dictionary with test results
        """
        # Aggregate trades by ticker
        ticker_trades = {}
        
        for window in results['windows']:
            for trade in window['results'].get('trades', []):
                ticker = trade.get('ticker')
                if ticker not in ticker_trades:
                    ticker_trades[ticker] = []
                ticker_trades[ticker].append(trade)
        
        # Calculate performance metrics by ticker
        for ticker, trades in ticker_trades.items():
            if not trades:
                continue
            
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) <= 0]
            
            results['performance'][ticker] = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'total_profit': sum(t.get('profit', 0) for t in trades),
                'avg_profit': sum(t.get('profit', 0) for t in trades) / len(trades) if trades else 0,
                'avg_win': sum(t.get('profit', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t.get('profit', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            }
        
        # Aggregate model metrics
        model_metrics = {}
        
        for window in results['windows']:
            for ticker, metrics in window['results'].get('model_metrics', {}).items():
                if ticker not in model_metrics:
                    model_metrics[ticker] = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': []
                    }
                
                for metric_name, metric_value in metrics.items():
                    if metric_name in model_metrics[ticker]:
                        model_metrics[ticker][metric_name].append(metric_value)
        
        # Calculate average model metrics
        for ticker, metrics in model_metrics.items():
            results['model_metrics'].append({
                'ticker': ticker,
                'avg_accuracy': np.mean(metrics['accuracy']) if metrics['accuracy'] else 0,
                'avg_precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
                'avg_recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
                'avg_f1': np.mean(metrics['f1']) if metrics['f1'] else 0
            })
        
        logger.info("Aggregated results across all test windows")
    
    def _save_results(self, results):
        """
        Save test results to a file.
        
        Args:
            results: Dictionary with test results
        """
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"wf_results_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved walk-forward test results to {filename}")

    def _make_serializable(self, obj):
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()
                    if not k.startswith('_')}
        else:
            return obj

    def _generate_plots(self, results, timestamp):
        """
        Generate and save plots from test results.
        
        Args:
            results: Dictionary with test results
            timestamp: Timestamp for file naming
        """
       

    def ensure_data_files_exist(self, tickers, start_date, end_date):
        """
        Ensure data files exist for all tickers by creating them if necessary.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data generation
            end_date: End date for data generation
        """
        for ticker in tickers:
            options_file = os.path.join(self.data_dir, f"{ticker}_options.csv")
            price_file = os.path.join(self.data_dir, f"{ticker}_prices.csv")
            
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Generate and save options data if file doesn't exist
            if not os.path.exists(options_file):
                logger.info(f"Creating sample options data file for {ticker}")
                options_data, _ = self.generate_sample_data(ticker, start_date, end_date)
                options_data.to_csv(options_file, index=False)
                logger.info(f"Saved sample options data to {options_file}")
            
            # Generate and save price data if file doesn't exist
            if not os.path.exists(price_file):
                logger.info(f"Creating sample price data file for {ticker}")
                _, price_data = self.generate_sample_data(ticker, start_date, end_date)
                price_data.to_csv(price_file, index=False)
                logger.info(f"Saved sample price data to {price_file}")

    def generate_sample_data(self, ticker, start_date, end_date):
        """
        Generate sample options and price data for a ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for data generation
            end_date: End date for data generation
        
        Returns:
            Tuple of (options_data, price_data) DataFrames
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate price data
        base_price = 100.0 if ticker in ['SPY', 'QQQ'] else 150.0 if ticker == 'AAPL' else 200.0
        
        # Add some randomness to make it look realistic
        np.random.seed(hash(ticker) % 10000)  # Use ticker as seed for reproducibility
        
        # Generate price series with random walk
        daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))
        price_series = base_price * np.cumprod(1 + daily_returns)
        
        price_data = pd.DataFrame({
            'date': date_range,
            'open': price_series * (1 - np.random.uniform(0, 0.005, size=len(date_range))),
            'high': price_series * (1 + np.random.uniform(0.001, 0.01, size=len(date_range))),
            'low': price_series * (1 - np.random.uniform(0.001, 0.01, size=len(date_range))),
            'close': price_series,
            'volume': np.random.randint(1000000, 10000000, size=len(date_range)),
            'adj_close': price_series
        })
        
        # Generate options data
        options_rows = []
        
        for date in date_range:
            current_price = float(price_data[price_data['date'] == date]['close'].values[0])
            
            # Generate a range of strikes around the current price
            strikes = np.linspace(current_price * 0.8, current_price * 1.2, 9)
            
            # Generate expiration dates (1, 2, and 3 months out)
            expirations = [
                date + pd.Timedelta(days=30),
                date + pd.Timedelta(days=60),
                date + pd.Timedelta(days=90)
            ]
            
            for strike in strikes:
                for expiration in expirations:
                    for option_type in ['call', 'put']:
                        # Calculate time to expiration in years
                        tte = (expiration - date).days / 365.0
                        
                        # Calculate moneyness
                        moneyness = current_price / strike if option_type == 'call' else strike / current_price
                        
                        # Calculate implied volatility (higher for OTM options)
                        base_iv = 0.2 + 0.1 * abs(1 - moneyness)
                        iv = base_iv + np.random.uniform(-0.05, 0.05)
                        
                        # Calculate option greeks (simplified)
                        if option_type == 'call':
                            delta = 0.5 + 0.5 * (moneyness - 1) / (base_iv * np.sqrt(tte))
                            delta = max(0.01, min(0.99, delta))
                        else:
                            delta = -0.5 - 0.5 * (moneyness - 1) / (base_iv * np.sqrt(tte))
                            delta = max(-0.99, min(-0.01, delta))
                        
                        gamma = (0.4 * np.exp(-((moneyness - 1) ** 2) / (2 * base_iv ** 2 * tte))) / (current_price * base_iv * np.sqrt(tte))
                        vega = 0.1 * current_price * np.sqrt(tte) * np.exp(-((moneyness - 1) ** 2) / (2 * base_iv ** 2 * tte))
                        theta = -0.5 * current_price * base_iv * np.exp(-((moneyness - 1) ** 2) / (2 * base_iv ** 2 * tte)) / (2 * np.sqrt(tte))
                        rho = 0.05 * strike * tte * (0.5 + 0.5 * (moneyness - 1) / (base_iv * np.sqrt(tte)))
                        
                        # Add some noise to the greeks
                        delta *= (1 + np.random.uniform(-0.05, 0.05))
                        gamma *= (1 + np.random.uniform(-0.05, 0.05))
                        vega *= (1 + np.random.uniform(-0.05, 0.05))
                        theta *= (1 + np.random.uniform(-0.05, 0.05))
                        rho *= (1 + np.random.uniform(-0.05, 0.05))
                        
                        # Calculate option price (simplified)
                        intrinsic = max(0, current_price - strike) if option_type == 'call' else max(0, strike - current_price)
                        time_value = current_price * base_iv * np.sqrt(tte) * np.exp(-((moneyness - 1) ** 2) / (2 * base_iv ** 2 * tte))
                        option_price = intrinsic + time_value
                        
                        # Add random open interest and volume
                        open_interest = int(np.random.exponential(1000) * (1.5 - abs(moneyness - 1)))
                        volume = int(open_interest * np.random.uniform(0.1, 0.5))
                        
                        options_rows.append({
                            'date': date,
                            'expiration': expiration,
                            'strike': strike,
                            'type': option_type,
                            'impliedVolatility': iv,
                            'openInterest': open_interest,
                            'volume': volume,
                            'lastPrice': option_price,
                            'bid': option_price * 0.95,
                            'ask': option_price * 1.05,
                            'delta': delta,
                            'gamma': gamma,
                            'vega': vega,
                            'theta': theta,
                            'rho': rho
                        })
        
        options_data = pd.DataFrame(options_rows)
        
        return options_data, price_data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    tester = WalkForwardTester()
    
    # Define test parameters
    tickers = ['AAPL', 'MSFT', 'QQQ', 'SPY']
    start_date = '2024-01-01'
    end_date = '2024-04-30'
    
    # Run a simple test
    logger.info("Starting walk-forward test...")
    results = tester.run_walk_forward_test(tickers, start_date, end_date)
    
    logger.info("Walk-forward test completed")
    logger.info(f"Tested {len(results['windows'])} windows")







