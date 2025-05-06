"""
ML-Enhanced Trading System for Greek Energy Flow II

This script orchestrates the complete workflow:
1. Runs the Greek analysis pipeline
2. Trains ML models on the Greek and entropy data
3. Generates regime predictions with ML models
4. Executes trades based on the ML-enhanced signals
5. Monitors and exits positions based on entropy changes

Usage:
    python run_ml_enhanced_trading.py --tickers AAPL MSFT --train
    python run_ml_enhanced_trading.py --tickers AAPL MSFT --live
    python run_ml_enhanced_trading.py --ticker-file tickers.csv --train
"""

import os
import sys
import json
import time
import logging
import argparse
import csv
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import Greek analysis modules
from run_regime_analysis import run_pipeline_analysis, run_regime_analysis
from run_ml_regime_analysis import train_ml_models, run_ml_prediction

# Import ML trading modules
from models.ml.trade_executor import MLTradeExecutor, MLEnhancedAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ml_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json file."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def load_tickers_from_file(file_path):
    """
    Load ticker symbols from a CSV file.
    
    Args:
        file_path: Path to CSV file containing tickers
        
    Returns:
        List of ticker symbols
    """
    tickers = []
    
    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if row and isinstance(row[0], str) and row[0].strip() and not row[0].startswith('#'):
                        tickers.append(row[0].strip().upper())
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                for line in f:
                    ticker = line.strip()
                    if ticker and not ticker.startswith('#'):
                        tickers.append(ticker.upper())
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return []
        
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    
    except Exception as e:
        logger.error(f"Error loading tickers from {file_path}: {e}")
        return []

def fetch_historical_data(ticker, days=60, api_key=None):
    """
    Fetch historical price data for a ticker.
    
    Args:
        ticker: Stock symbol
        days: Number of days of history to fetch
        api_key: API key for data provider
        
    Returns:
        DataFrame with historical price data
    """
    try:
        config = load_config()
        api_config = config.get('api_config', {})
        
        # Use provided API key or from config
        if api_key is None:
            api_key = api_config.get('api_key')
        
        if not api_key:
            logger.warning("No API key provided, using mock data")
            return generate_mock_historical_data(ticker, days)
        
        # API endpoint (example using Alpha Vantage)
        api_url = api_config.get('api_url', 'https://www.alphavantage.co/query')
        
        # Request parameters
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        # Make API request
        response = requests.get(api_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            logger.error(f"Error fetching data for {ticker}: {data.get('Error Message', 'Unknown error')}")
            # Try using Polygon API as fallback
            return fetch_historical_data_polygon(ticker, days, api_key)
        
        # Parse response
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Limit to requested days
        if days > 0:
            start_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= start_date]
        
        logger.info(f"Fetched {len(df)} days of historical data for {ticker}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        # Try using Polygon API as fallback
        return fetch_historical_data_polygon(ticker, days, api_key)

def fetch_historical_data_polygon(ticker, days=60, api_key=None):
    """
    Fetch historical price data for a ticker using Polygon API.
    
    Args:
        ticker: Stock symbol
        days: Number of days of history to fetch
        api_key: API key for Polygon API
        
    Returns:
        DataFrame with historical price data
    """
    try:
        # Import config to get Polygon API key
        try:
            import config
            polygon_api_key = config.POLYGON_API_KEY
        except (ImportError, AttributeError):
            polygon_api_key = None
        
        # Use provided API key or from config
        if api_key is None:
            api_key = polygon_api_key
        
        if not api_key:
            logger.warning("No Polygon API key provided, using mock data")
            return generate_mock_historical_data(ticker, days)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # API endpoint
        api_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
        
        # Request parameters
        params = {
            'apiKey': api_key,
            'sort': 'asc',
            'limit': 5000  # Maximum allowed
        }
        
        # Make API request
        response = requests.get(api_url, params=params)
        data = response.json()
        
        if 'results' not in data or not data['results']:
            logger.error(f"Error fetching data from Polygon for {ticker}: {data.get('error', 'Unknown error')}")
            return generate_mock_historical_data(ticker, days)
        
        # Parse response
        results = data['results']
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Rename columns
        df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)
        
        # Select only needed columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"Fetched {len(df)} days of historical data for {ticker} from Polygon")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data from Polygon for {ticker}: {e}")
        return generate_mock_historical_data(ticker, days)

def generate_mock_historical_data(ticker, days=30):
    """Generate mock historical data for testing."""
    logger.info(f"Generating mock historical data for {ticker}")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize with a base price based on ticker
    ticker_hash = sum(ord(c) for c in ticker)
    base_price = 50 + (ticker_hash % 200)  # Generate price between 50 and 250
    
    # Generate random price movements
    np.random.seed(ticker_hash)  # Deterministic based on ticker
    
    # Generate daily returns with momentum
    returns = np.random.normal(0.0005, 0.015, len(date_range))  # Slight upward bias
    price_multipliers = np.cumprod(1 + returns)
    
    # Calculate prices
    closes = base_price * price_multipliers
    
    # Generate other prices around close
    highs = closes * np.random.uniform(1.01, 1.03, len(date_range))
    lows = closes * np.random.uniform(0.97, 0.99, len(date_range))
    opens = lows + np.random.uniform(0, 1, len(date_range)) * (highs - lows)
    volumes = np.random.randint(100000, 10000000, len(date_range))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=date_range)
    
    return df

def get_current_prices(tickers, api_key=None, use_mock=False):
    """
    Get current prices for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        api_key: API key for data provider
        use_mock: If True, use mock data for testing
        
    Returns:
        Dictionary with ticker -> price mapping
    """
    # If explicitly asked to use mock data, do so
    if use_mock:
        # Generate deterministic mock prices
        prices = {}
        for ticker in tickers:
            ticker_hash = sum(ord(c) for c in ticker)
            base_price = 50 + (ticker_hash % 200)  # Generate price between 50 and 250
            variation = np.sin(time.time() / 3600) * 0.05  # Slight time-based variation
            prices[ticker] = round(base_price * (1 + variation), 2)
        
        return prices
    
    try:
        # Load API configuration
        config = load_config()
        api_config = config.get('api_config', {})
        
        # Use provided API key or from config
        if api_key is None:
            api_key = api_config.get('api_key')
        
        # Try to import config for Polygon API key
        try:
            import config as project_config
            polygon_api_key = getattr(project_config, 'POLYGON_API_KEY', None)
        except (ImportError, AttributeError):
            polygon_api_key = None
        
        # If no API key provided, try Polygon API key
        if not api_key and polygon_api_key:
            logger.info("Using Polygon API key for current prices")
            return get_current_prices_polygon(tickers, polygon_api_key)
        
        if not api_key:
            logger.warning("No API key provided, using mock data")
            return get_current_prices(tickers, use_mock=True)
        
        # API endpoint for current prices (example using Alpha Vantage)
        api_url = api_config.get('api_url', 'https://www.alphavantage.co/query')
        
        prices = {}
        batch_size = 5  # Limit API requests by processing in batches
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            
            for ticker in batch_tickers:
                # Request parameters
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': api_key
                }
                
                # Make API request
                response = requests.get(api_url, params=params)
                data = response.json()
                
                if 'Global Quote' in data and '05. price' in data['Global Quote']:
                    price = float(data['Global Quote']['05. price'])
                    prices[ticker] = price
                else:
                    logger.warning(f"Error fetching price for {ticker} from Alpha Vantage, trying Polygon")
                    # Try Polygon as fallback
                    if polygon_api_key:
                        ticker_price = get_current_prices_polygon([ticker], polygon_api_key)
                        if ticker in ticker_price:
                            prices[ticker] = ticker_price[ticker]
                        else:
                            logger.warning(f"Error fetching price for {ticker} from Polygon, using mock data")
                            ticker_hash = sum(ord(c) for c in ticker)
                            prices[ticker] = 50 + (ticker_hash % 200)
                    else:
                        logger.warning(f"No Polygon API key available, using mock data for {ticker}")
                        ticker_hash = sum(ord(c) for c in ticker)
                        prices[ticker] = 50 + (ticker_hash % 200)
                
                # Respect API rate limits
                time.sleep(0.2)
        
        return prices
    
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        # Try Polygon as fallback
        try:
            import config as project_config
            polygon_api_key = getattr(project_config, 'POLYGON_API_KEY', None)
            if polygon_api_key:
                return get_current_prices_polygon(tickers, polygon_api_key)
        except:
            pass
        
        return get_current_prices(tickers, use_mock=True)

def get_current_prices_polygon(tickers, api_key):
    """
    Get current prices for a list of tickers using Polygon API.
    
    Args:
        tickers: List of ticker symbols
        api_key: API key for Polygon API
        
    Returns:
        Dictionary with ticker -> price mapping
    """
    try:
        prices = {}
        
        for ticker in tickers:
            # API endpoint
            api_url = f"https://api.polygon.io/v2/last/trade/{ticker}"
            
            # Request parameters
            params = {
                'apiKey': api_key
            }
            
            # Make API request
            response = requests.get(api_url, params=params)
            data = response.json()
            
            if 'results' in data and 'p' in data['results']:
                price = float(data['results']['p'])
                prices[ticker] = price
            else:
                logger.warning(f"Error fetching price for {ticker} from Polygon: {data.get('error', 'Unknown error')}")
                # Fall back to mock price
                ticker_hash = sum(ord(c) for c in ticker)
                prices[ticker] = 50 + (ticker_hash % 200)
            
            # Respect API rate limits
            time.sleep(0.1)
        
        return prices
    
    except Exception as e:
        logger.error(f"Error fetching current prices from Polygon: {e}")
        return get_current_prices(tickers, use_mock=True)

def save_price_data(ticker, price_data, data_dir='data/price_history'):
    """
    Save price data to CSV file.
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with price data
        data_dir: Directory to save price data
        
    Returns:
        Path to saved file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(data_dir, f"{ticker}_daily.csv")
        
        # Save to CSV
        price_data.to_csv(file_path)
        
        logger.info(f"Saved price data for {ticker} to {file_path}")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving price data for {ticker}: {e}")
        return None

def fetch_and_save_historical_data(tickers, days=60, api_key=None, data_dir='data/price_history'):
    """
    Fetch and save historical data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history to fetch
        api_key: API key for data provider
        data_dir: Directory to save price data
        
    Returns:
        Dictionary with ticker -> file path mapping
    """
    logger.info(f"Fetching historical data for {len(tickers)} tickers")
    
    file_paths = {}
    
    for ticker in tickers:
        # Fetch historical data
        price_data = fetch_historical_data(ticker, days=days, api_key=api_key)
        
        # Save to file
        if not price_data.empty:
            file_path = save_price_data(ticker, price_data, data_dir=data_dir)
            if file_path:
                file_paths[ticker] = file_path
    
    logger.info(f"Fetched and saved historical data for {len(file_paths)} tickers")
    
    return file_paths

def run_complete_analysis(tickers, output_dir='results', skip_entropy=False, fetch_data=True, days=60, api_key=None):
    """
    Run the complete Greek analysis pipeline.
    
    Args:
        tickers: List of ticker symbols
        output_dir: Output directory for analysis files
        skip_entropy: If True, skip entropy analysis for faster processing
        fetch_data: If True, fetch historical data before analysis
        days: Number of days of history to fetch
        api_key: API key for data provider
        
    Returns:
        True if analysis completed successfully, False otherwise
    """
    logger.info(f"Running complete analysis for {len(tickers)} tickers")
    
    # Fetch historical data if requested
    if fetch_data:
        fetch_and_save_historical_data(tickers, days=days, api_key=api_key)
    
    # Run pipeline analysis
    pipeline_success = run_pipeline_analysis(
        tickers=tickers,
        output_dir=output_dir,
        skip_entropy=skip_entropy,
        analysis_type="both"
    )
    
    if not pipeline_success:
        logger.error("Pipeline analysis failed")
        return False
    
    # Run market regime analysis
    regime_success = run_regime_analysis(results_dir=output_dir)
    
    if not regime_success:
        logger.error("Market regime analysis failed")
        return False
    
    logger.info("Complete analysis finished successfully")
    
    return True

def train_and_validate_models(tickers, results_dir='results', model_type='randomforest'):
    """
    Train and validate ML models.
    
    Args:
        tickers: List of ticker symbols
        results_dir: Directory containing analysis results
        model_type: Type of ML model to use
        
    Returns:
        Dictionary with training metrics
    """
    logger.info(f"Training and validating ML models using {model_type}")
    
    # Train models
    training_results = train_ml_models(
        tickers=tickers,
        results_dir=results_dir,
        model_type=model_type
    )
    
    if not training_results:
        logger.error("ML model training failed")
        return None
    
    # Generate ML predictions
    prediction_results = run_ml_prediction(
        tickers=tickers,
        results_dir=results_dir,
        output_dir=os.path.join(results_dir, "ml_predictions"),
        model_dir="models/ml"
    )
    
    if not prediction_results:
        logger.error("ML prediction failed")
        return training_results
    
    logger.info("ML models trained and validated successfully")
    
    return {
        'training': training_results,
        'prediction': prediction_results
    }

def run_trading_simulation(tickers, days=5, interval_minutes=60, results_dir='results'):
    """
    Run a trading simulation using ML-enhanced signals.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days to simulate
        interval_minutes: Interval between checks in minutes
        results_dir: Directory containing analysis results
        
    Returns:
        Dictionary with simulation results
    """
    logger.info(f"Running trading simulation for {len(tickers)} tickers over {days} days")
    
    # First, ensure we have ML predictions for all tickers
    logger.info("Generating ML predictions for all tickers before starting simulation")
    ml_predictions_dir = os.path.join(results_dir, "ml_predictions")
    os.makedirs(ml_predictions_dir, exist_ok=True)
    
    # Run ML prediction for all tickers
    prediction_results = run_ml_prediction(
        tickers=tickers,
        results_dir=results_dir,
        output_dir=ml_predictions_dir,
        model_dir="models/ml"
    )
    
    if not prediction_results:
        logger.warning("Failed to generate ML predictions, simulation may have limited functionality")
    
    # Load configuration
    config = load_config()
    
    # Initialize trade executor
    executor = MLTradeExecutor(config=config, risk_config=config.get('risk_config', {}))
    
    # Load active trades and trade history
    executor.load_active_trades()
    executor.load_trade_history()
    
    # Initialize enhanced analyzer
    analyzer = MLEnhancedAnalyzer(executor=executor)
    
    # Initialize simulation variables
    simulation_start = datetime.now()
    iterations = days * 24 * 60 // interval_minutes
    
    # Dictionary to track simulation metrics
    simulation_metrics = {
        'tickers': tickers,
        'start_time': simulation_start.isoformat(),
        'iterations': iterations,
        'interval_minutes': interval_minutes,
        'trades': [],
        'price_history': {},
        'final_positions': {}
    }
    
    # Initialize price history
    for ticker in tickers:
        simulation_metrics['price_history'][ticker] = []
    
    # Run simulation iterations
    for i in range(iterations):
        iteration_time = datetime.now()
        logger.info(f"Simulation iteration {i+1}/{iterations}")
        
        # Generate mock price changes
        current_prices = get_current_prices(tickers, use_mock=True)
        
        # Process each ticker
        for ticker in tickers:
            price = current_prices[ticker]
            
            # Record price in history
            simulation_metrics['price_history'][ticker].append({
                'timestamp': iteration_time.isoformat(),
                'price': price
            })
            
            # Perform enhanced analysis
            analysis = analyzer.analyze_ticker(
                ticker, 
                price, 
                results_dir=results_dir, 
                ml_predictions_dir=os.path.join(results_dir, "ml_predictions")
            )
            
            # Process ticker for trade decisions
            result = executor.process_ticker(
                ticker, 
                price, 
                results_dir=results_dir, 
                ml_predictions_dir=os.path.join(results_dir, "ml_predictions")
            )
            
            # Record trade if action taken
            if result['action'] != 'none':
                # Load active trade details if entry
                if result['action'] == 'entry':
                    active_trade = executor.active_trades.get(ticker, {})
                    trade_record = {
                        'ticker': ticker,
                        'action': 'entry',
                        'price': price,
                        'position_size': active_trade.get('position_size', 0),
                        'stop_loss': active_trade.get('stop_loss', 0),
                        'take_profit': active_trade.get('take_profit', 0),
                        'timestamp': iteration_time.isoformat(),
                        'iteration': i+1
                    }
                # Get trade history for exit
                elif result['action'] == 'exit':
                    # Find the latest exit order in trade history
                    exit_orders = [trade for trade in executor.trade_history 
                                  if trade.get('type') == 'exit' and 
                                  trade.get('order', {}).get('ticker') == ticker]
                    
                    if exit_orders:
                        latest_exit = max(exit_orders, key=lambda x: x.get('timestamp', ''))
                        exit_order = latest_exit.get('order', {})
                        
                        trade_record = {
                            'ticker': ticker,
                            'action': 'exit',
                            'price': price,
                            'exit_reason': exit_order.get('exit_reason', 'unknown'),
                            'profit_loss': exit_order.get('profit_loss', 0),
                            'timestamp': iteration_time.isoformat(),
                            'iteration': i+1
                        }
                    else:
                        trade_record = {
                            'ticker': ticker,
                            'action': 'exit',
                            'price': price,
                            'timestamp': iteration_time.isoformat(),
                            'iteration': i+1
                        }
                
                simulation_metrics['trades'].append(trade_record)
                logger.info(f"Simulation trade: {trade_record['action']} {ticker} at {price}")
        
        # Mock time passing in simulation
        if i < iterations - 1:
            logger.info(f"Simulation waiting {interval_minutes} minutes (compressed)")
            time.sleep(0.5)  # Compressed wait time for simulation
    
    # Record final positions
    simulation_metrics['final_positions'] = executor.active_trades
    
    # Calculate simulation results
    simulation_results = calculate_simulation_results(simulation_metrics, executor.trade_history)
    
    # Save simulation results
    save_simulation_results(simulation_results, results_dir)
    
    logger.info("Trading simulation completed")
    
    return simulation_results

def calculate_simulation_results(simulation_metrics, trade_history):
    """Calculate metrics from simulation results."""
    results = {
        'summary': {
            'total_trades': len(simulation_metrics['trades']),
            'total_entries': sum(1 for trade in simulation_metrics['trades'] if trade['action'] == 'entry'),
            'total_exits': sum(1 for trade in simulation_metrics['trades'] if trade['action'] == 'exit'),
            'profit_loss': 0.0,
            'win_rate': 0.0,
            'avg_win_size': 0.0,
            'avg_loss_size': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        },
        'ticker_performance': {},
        'trades': simulation_metrics['trades'],
        'final_positions': simulation_metrics['final_positions']
    }
    
    # Calculate profit/loss metrics
    profitable_trades = []
    losing_trades = []
    ticker_trades = {}
    
    for trade in simulation_metrics['trades']:
        ticker = trade['ticker']
        
        if ticker not in ticker_trades:
            ticker_trades[ticker] = []
        
        ticker_trades[ticker].append(trade)
        
        if trade['action'] == 'exit' and 'profit_loss' in trade:
            pnl = trade['profit_loss']
            results['summary']['profit_loss'] += pnl
            
            if pnl > 0:
                profitable_trades.append(pnl)
                results['summary']['largest_win'] = max(results['summary']['largest_win'], pnl)
            else:
                losing_trades.append(pnl)
                results['summary']['largest_loss'] = min(results['summary']['largest_loss'], pnl)
    
    # Calculate win rate and average sizes
    total_closed_trades = len(profitable_trades) + len(losing_trades)
    if total_closed_trades > 0:
        results['summary']['win_rate'] = len(profitable_trades) / total_closed_trades
    
    if profitable_trades:
        results['summary']['avg_win_size'] = sum(profitable_trades) / len(profitable_trades)
    
    if losing_trades:
        results['summary']['avg_loss_size'] = sum(losing_trades) / len(losing_trades)
    
    # Calculate per-ticker metrics
    for ticker, trades in ticker_trades.items():
        ticker_entries = sum(1 for trade in trades if trade['action'] == 'entry')
        ticker_exits = sum(1 for trade in trades if trade['action'] == 'exit')
        ticker_pnl = sum(trade.get('profit_loss', 0) for trade in trades if trade['action'] == 'exit' and 'profit_loss' in trade)
        
        price_changes = simulation_metrics['price_history'].get(ticker, [])
        first_price = price_changes[0]['price'] if price_changes else 0
        last_price = price_changes[-1]['price'] if price_changes else 0
        price_change_pct = (last_price - first_price) / first_price * 100 if first_price > 0 else 0
        
        results['ticker_performance'][ticker] = {
            'total_trades': ticker_entries + ticker_exits,
            'entries': ticker_entries,
            'exits': ticker_exits,
            'profit_loss': ticker_pnl,
            'first_price': first_price,
            'last_price': last_price,
            'price_change_pct': price_change_pct
        }
    
    # Add time metrics
    if simulation_metrics['trades']:
        first_trade = min(simulation_metrics['trades'], key=lambda x: x.get('timestamp', ''))
        last_trade = max(simulation_metrics['trades'], key=lambda x: x.get('timestamp', ''))
        
        results['summary']['first_trade_time'] = first_trade.get('timestamp')
        results['summary']['last_trade_time'] = last_trade.get('timestamp')
    
    # Add active positions value
    active_positions_value = 0
    for ticker, position in results['final_positions'].items():
        price_history = simulation_metrics['price_history'].get(ticker, [])
        last_price = price_history[-1]['price'] if price_history else 0
        
        position_size = position.get('position_size', 0)
        position_value = position_size * last_price
        active_positions_value += position_value
    
    results['summary']['active_positions_value'] = active_positions_value
    
    return results

def save_simulation_results(results, output_dir):
    """Save simulation results to file."""
    try:
        # Create output directory
        sim_dir = os.path.join(output_dir, "ml_simulation")
        os.makedirs(sim_dir, exist_ok=True)
        
        # Create filename with timestamp
        filename = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(sim_dir, filename)
        
        # Save results to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Simulation results saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving simulation results: {str(e)}")

def run_live_trading(tickers, interval_minutes=5, results_dir='results', api_key=None):
    """
    Run live trading using ML-enhanced signals.
    
    Args:
        tickers: List of ticker symbols
        interval_minutes: Interval between checks in minutes
        results_dir: Directory containing analysis results
        api_key: API key for data provider
        
    Returns:
        Dictionary with live trading results
    """
    logger.info(f"Starting live trading for {len(tickers)} tickers")
    
    # Load configuration
    config = load_config()
    
    # Initialize trade executor
    executor = MLTradeExecutor(config=config, risk_config=config.get('risk_config', {}))
    
    # Load active trades and trade history
    executor.load_active_trades()
    executor.load_trade_history()
    
    # Initialize enhanced analyzer
    analyzer = MLEnhancedAnalyzer(executor=executor)
    
    # Dictionary to track live trading metrics
    live_metrics = {
        'tickers': tickers,
        'start_time': datetime.now().isoformat(),
        'interval_minutes': interval_minutes,
        'trades': [],
        'price_history': {},
        'active_positions': {}
    }
    
    # Initialize price history
    for ticker in tickers:
        live_metrics['price_history'][ticker] = []
    
    # Create output directory for live trading
    live_dir = os.path.join(results_dir, "ml_live")
    os.makedirs(live_dir, exist_ok=True)
    
    # Run live trading loop
    try:
        iteration = 0
        while True:
            iteration += 1
            current_time = datetime.now()
            logger.info(f"Live trading iteration {iteration} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run complete analysis if first iteration or every 4 hours
            if iteration == 1 or iteration % (4 * 60 // interval_minutes) == 0:
                logger.info("Running complete analysis")
                run_complete_analysis(
                    tickers, 
                    output_dir=results_dir, 
                    fetch_data=True, 
                    days=60, 
                    api_key=api_key
                )
            
            # Run ML prediction every hour
            if iteration == 1 or iteration % (60 // interval_minutes) == 0:
                logger.info("Running ML prediction")
                run_ml_prediction(
                    tickers=tickers,
                    results_dir=results_dir,
                    output_dir=os.path.join(results_dir, "ml_predictions"),
                    model_dir="models/ml"
                )
            
            # Get current prices
            current_prices = get_current_prices(tickers, api_key=api_key)
            
            # Process each ticker
            for ticker in tickers:
                price = current_prices[ticker]
                
                # Record price in history
                live_metrics['price_history'][ticker].append({
                    'timestamp': current_time.isoformat(),
                    'price': price
                })
                
                # Limit price history length
                if len(live_metrics['price_history'][ticker]) > 1000:
                    live_metrics['price_history'][ticker] = live_metrics['price_history'][ticker][-1000:]
                
                # Perform enhanced analysis
                analysis = analyzer.analyze_ticker(
                    ticker, 
                    price, 
                    results_dir=results_dir, 
                    ml_predictions_dir=os.path.join(results_dir, "ml_predictions")
                )
                
                # Process ticker for trade decisions
                result = executor.process_ticker(
                    ticker, 
                    price, 
                    results_dir=results_dir, 
                    ml_predictions_dir=os.path.join(results_dir, "ml_predictions")
                )
                
                # Record trade if action taken
                if result['action'] != 'none':
                    # Create trade record
                    if result['action'] == 'entry':
                        active_trade = executor.active_trades.get(ticker, {})
                        trade_record = {
                            'ticker': ticker,
                            'action': 'entry',
                            'price': price,
                            'position_size': active_trade.get('position_size', 0),
                            'stop_loss': active_trade.get('stop_loss', 0),
                            'take_profit': active_trade.get('take_profit', 0),
                            'timestamp': current_time.isoformat(),
                            'iteration': iteration
                        }
                    else:  # exit
                        # Find the latest exit order in trade history
                        exit_orders = [trade for trade in executor.trade_history 
                                      if trade.get('type') == 'exit' and 
                                      trade.get('order', {}).get('ticker') == ticker]
                        
                        if exit_orders:
                            latest_exit = max(exit_orders, key=lambda x: x.get('timestamp', ''))
                            exit_order = latest_exit.get('order', {})
                            
                            trade_record = {
                                'ticker': ticker,
                                'action': 'exit',
                                'price': price,
                                'exit_reason': exit_order.get('exit_reason', 'unknown'),
                                'profit_loss': exit_order.get('profit_loss', 0),
                                'timestamp': current_time.isoformat(),
                                'iteration': iteration
                            }
                        else:
                            trade_record = {
                                'ticker': ticker,
                                'action': 'exit',
                                'price': price,
                                'timestamp': current_time.isoformat(),
                                'iteration': iteration
                            }
                    
                    # Add trade to metrics
                    live_metrics['trades'].append(trade_record)
                    
                    # Log trade
                    logger.info(f"Live trade: {trade_record['action']} {ticker} at {price}")
                    
                    # Save trade record
                    trade_file = os.path.join(live_dir, f"trade_{ticker}_{result['action']}_{current_time.strftime('%Y%m%d_%H%M%S')}.json")
                    with open(trade_file, 'w') as f:
                        json.dump(trade_record, f, indent=2)
            
            # Save active positions
            live_metrics['active_positions'] = executor.active_trades
            
            # Save live metrics
            metrics_file = os.path.join(live_dir, "live_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(live_metrics, f, indent=2)
            
            # Wait for next iteration
            logger.info(f"Waiting {interval_minutes} minutes until next check")
            
            # Calculate wait time (adjusted for processing time)
            elapsed = (datetime.now() - current_time).total_seconds()
            wait_time = max(0, interval_minutes * 60 - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
    
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Error in live trading: {e}")
    
    # Save final metrics
    final_metrics_file = os.path.join(live_dir, f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(final_metrics_file, 'w') as f:
        json.dump(live_metrics, f, indent=2)
    
    logger.info(f"Final metrics saved to {final_metrics_file}")
    
    return live_metrics

def generate_daily_trade_recommendations(tickers, results_dir='results', confidence_threshold=0.75):
    """
    Generate high-conviction trade recommendations based on ML analysis.
    
    Args:
        tickers: List of ticker symbols
        results_dir: Directory containing analysis results
        confidence_threshold: Minimum confidence threshold for recommendations
        
    Returns:
        Dictionary with trade recommendations
    """
    logger.info(f"Generating daily trade recommendations for {len(tickers)} tickers")
    
    # Load configuration
    config = load_config()
    
    # Initialize trade executor and analyzer
    executor = MLTradeExecutor(config=config, risk_config=config.get('risk_config', {}))
    analyzer = MLEnhancedAnalyzer(executor=executor)
    
    # Get current prices
    current_prices = get_current_prices(tickers, use_mock=False, api_key=config.get('api_config', {}).get('api_key'))
    
    # Initialize recommendations
    recommendations = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'entry_signals': [],
        'exit_signals': [],
        'watchlist': []
    }
    
    # Process each ticker
    for ticker in tickers:
        price = current_prices.get(ticker)
        if not price:
            logger.warning(f"No price data for {ticker}, skipping")
            continue
        
        # Perform enhanced analysis
        analysis = analyzer.analyze_ticker(
            ticker, 
            price, 
            results_dir=results_dir, 
            ml_predictions_dir=os.path.join(results_dir, "ml_predictions")
        )
        
        if not analysis:
            logger.warning(f"Analysis failed for {ticker}, skipping")
            continue
        
        # Extract recommendation
        recommendation = analysis.get('recommendation', {})
        action = recommendation.get('action', 'none')
        confidence = recommendation.get('confidence', 0)
        
        # Check if ticker is in active trades
        in_position = ticker in executor.active_trades
        
        # Process recommendation based on action and confidence
        if action == 'entry' and confidence >= confidence_threshold and not in_position:
            # High-conviction entry signal
            entry_rec = {
                'ticker': ticker,
                'price': price,
                'confidence': confidence,
                'reasoning': recommendation.get('reasoning', []),
                'price_targets': recommendation.get('price_targets', {}),
                'risk_assessment': recommendation.get('risk_assessment', {}),
                'position_size': recommendation.get('position_size', 0),
                'stop_loss': recommendation.get('price_targets', {}).get('stop_loss', 0),
                'take_profit': recommendation.get('price_targets', {}).get('take_profit', 0),
                'ml_confidence': recommendation.get('ml_confidence', {})
            }
            
            # Calculate risk-reward ratio
            risk = abs(price - entry_rec['stop_loss']) if entry_rec['stop_loss'] > 0 else 0
            reward = abs(entry_rec['take_profit'] - price) if entry_rec['take_profit'] > 0 else 0
            
            if risk > 0:
                entry_rec['risk_reward_ratio'] = reward / risk
            else:
                entry_rec['risk_reward_ratio'] = 0
            
            recommendations['entry_signals'].append(entry_rec)
            
        elif action == 'exit' and confidence >= confidence_threshold and in_position:
            # High-conviction exit signal
            exit_rec = {
                'ticker': ticker,
                'price': price,
                'confidence': confidence,
                'reasoning': recommendation.get('reasoning', []),
                'position_details': recommendation.get('position_details', {}),
                'exit_type': recommendation.get('exit_type', 'signal'),
                'profit_loss': recommendation.get('profit_loss', 0),
                'ml_confidence': recommendation.get('ml_confidence', {})
            }
            
            recommendations['exit_signals'].append(exit_rec)
            
        elif confidence >= 0.5 and confidence < confidence_threshold:
            # Medium-conviction signal for watchlist
            watch_rec = {
                'ticker': ticker,
                'price': price,
                'action': action,
                'confidence': confidence,
                'reasoning': recommendation.get('reasoning', [])[:2],  # Limit to top 2 reasons
                'ml_confidence': recommendation.get('ml_confidence', {})
            }
            
            recommendations['watchlist'].append(watch_rec)
    
    # Sort recommendations by confidence
    recommendations['entry_signals'] = sorted(
        recommendations['entry_signals'], 
        key=lambda x: (x.get('confidence', 0), x.get('risk_reward_ratio', 0)), 
        reverse=True
    )
    
    recommendations['exit_signals'] = sorted(
        recommendations['exit_signals'], 
        key=lambda x: x.get('confidence', 0), 
        reverse=True
    )
    
    recommendations['watchlist'] = sorted(
        recommendations['watchlist'], 
        key=lambda x: x.get('confidence', 0), 
        reverse=True
    )
    
    # Save recommendations to file
    output_file = os.path.join(results_dir, f"daily_recommendations_{datetime.now().strftime('%Y%m%d')}.json")
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"Generated {len(recommendations['entry_signals'])} entry signals, "
                f"{len(recommendations['exit_signals'])} exit signals, and "
                f"{len(recommendations['watchlist'])} watchlist items")
    
    return recommendations

def print_trade_recommendations(recommendations):
    """
    Print trade recommendations in a readable format.
    
    Args:
        recommendations: Dictionary with trade recommendations
    """
    print("\n" + "="*80)
    print(f"DAILY TRADE RECOMMENDATIONS - {recommendations['date']}")
    print("="*80)
    
    # Print entry signals
    if recommendations['entry_signals']:
        print("\nENTRY SIGNALS:")
        print("-"*80)
        for i, rec in enumerate(recommendations['entry_signals'], 1):
            print(f"{i}. {rec['ticker']} - BUY @ ${rec['price']:.2f}")
            print(f"   Confidence: {rec['confidence']:.2f} | Risk-Reward: {rec.get('risk_reward_ratio', 0):.2f}")
            print(f"   Stop Loss: ${rec['stop_loss']:.2f} | Take Profit: ${rec['take_profit']:.2f}")
            print(f"   Position Size: {rec['position_size']}")
            print(f"   ML Regime: {rec.get('ml_confidence', {}).get('primary_regime', 0):.2f}")
            print(f"   Reasoning: {', '.join(rec['reasoning'][:3])}")
            print()
    else:
        print("\nNo entry signals today.")
    
    # Print exit signals
    if recommendations['exit_signals']:
        print("\nEXIT SIGNALS:")
        print("-"*80)
        for i, rec in enumerate(recommendations['exit_signals'], 1):
            print(f"{i}. {rec['ticker']} - SELL @ ${rec['price']:.2f}")
            print(f"   Confidence: {rec['confidence']:.2f} | Exit Type: {rec['exit_type']}")
            print(f"   Profit/Loss: ${rec['profit_loss']:.2f}")
            print(f"   Reasoning: {', '.join(rec['reasoning'][:3])}")
            print()
    else:
        print("\nNo exit signals today.")
    
    # Print watchlist
    if recommendations['watchlist']:
        print("\nWATCHLIST:")
        print("-"*80)
        for i, rec in enumerate(recommendations['watchlist'], 1):
            print(f"{i}. {rec['ticker']} - {rec['action'].upper()} WATCH @ ${rec['price']:.2f}")
            print(f"   Confidence: {rec['confidence']:.2f}")
            print(f"   Reasoning: {', '.join(rec['reasoning'])}")
            print()
    else:
        print("\nNo watchlist items today.")
    
    print("="*80 + "\n")

def run_rainbow_analysis(tickers, timeframes=['1d', '4h', '1h'], output_dir='results/rainbow'):
    """
    Run rainbow analysis across multiple timeframes.
    
    Args:
        tickers: List of ticker symbols
        timeframes: List of timeframes to analyze
        output_dir: Directory to save results
        
    Returns:
        Dictionary with rainbow analysis results
    """
    logger.info(f"Running rainbow analysis for {len(tickers)} tickers across {len(timeframes)} timeframes")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    rainbow_results = {
        'timestamp': datetime.now().isoformat(),
        'tickers': {},
        'timeframe_alignment': {}
    }
    
    # Run analysis for each ticker and timeframe
    for ticker in tickers:
        ticker_results = {}
        
        for timeframe in timeframes:
            # Run analysis for this timeframe
            logger.info(f"Analyzing {ticker} on {timeframe} timeframe")
            
            try:
                # Run pipeline analysis for this timeframe
                tf_output_dir = os.path.join(output_dir, timeframe)
                os.makedirs(tf_output_dir, exist_ok=True)

                # Add logging for directory creation
                logger.debug(f"Created output directory: {tf_output_dir}")
                
                # Run analysis with pipeline
                cmd = [
                    "python", "run_with_pipeline.py",
                    "--tickers", ticker,
                    "--output-dir", tf_output_dir,
                    "--timeframe", timeframe,
                    "--analysis-type", "both"
                ]
                
                # Set environment variable to use non-interactive backend
                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"  # Use non-interactive backend
                
                # Run command
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                
                if result.returncode != 0:
                    logger.error(f"Analysis failed for {ticker} on {timeframe} with code {result.returncode}")
                    logger.error(f"Error: {result.stderr}")
                    continue
                
                # Load analysis results
                analysis_file = os.path.join(tf_output_dir, f"{ticker}_analysis.json")
                if not os.path.exists(analysis_file):
                    logger.warning(f"Analysis file not found for {ticker} on {timeframe}")
                    continue
                
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract key metrics
                ticker_results[timeframe] = {
                    'market_regime': analysis_data.get('greek_analysis', {}).get('market_regime', {}),
                    'entropy_state': analysis_data.get('entropy_data', {}).get('energy_state', {}),
                    'momentum': analysis_data.get('momentum_data', {}).get('momentum_score', 0)
                }
                
                logger.info(f"Completed analysis for {ticker} on {timeframe}")
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker} on {timeframe}: {str(e)}")
        
        # Store results for this ticker
        rainbow_results['tickers'][ticker] = ticker_results
        
        # Calculate timeframe alignment
        if ticker_results:
            rainbow_results['timeframe_alignment'][ticker] = calculate_timeframe_alignment(ticker_results)
    
    # Save rainbow results
    output_file = os.path.join(output_dir, f"rainbow_analysis_{datetime.now().strftime('%Y%m%d')}.json")
    with open(output_file, 'w') as f:
        json.dump(rainbow_results, f, indent=2)
    
    logger.info(f"Rainbow analysis completed for {len(rainbow_results['tickers'])} tickers")
    
    return rainbow_results

def calculate_timeframe_alignment(timeframe_results):
    """
    Calculate alignment across timeframes.
    
    Args:
        timeframe_results: Dictionary with results for each timeframe
        
    Returns:
        Dictionary with alignment metrics
    """
    # Initialize alignment metrics
    alignment = {
        'regime_alignment': 0,
        'direction_alignment': 0,
        'entropy_alignment': 0,
        'overall_alignment': 0,
        'aligned_timeframes': [],
        'conflicting_timeframes': []
    }
    
    if not timeframe_results:
        return alignment
    
    # Extract regimes and directions from each timeframe
    regimes = {}
    directions = {}
    entropy_states = {}
    
    for timeframe, results in timeframe_results.items():
        market_regime = results.get('market_regime', {})
        entropy_state = results.get('entropy_state', {})
        
        # Extract regime
        primary_regime = market_regime.get('primary_label', 'Unknown')
        if primary_regime != 'Unknown':
            regimes[timeframe] = primary_regime
        
        # Extract direction
        direction = market_regime.get('direction', 'Neutral')
        if direction != 'Unknown':
            directions[timeframe] = direction
        
        # Extract entropy state
        energy_state = entropy_state.get('primary_state', 'Unknown')
        if energy_state != 'Unknown':
            entropy_states[timeframe] = energy_state
    
    # Calculate regime alignment
    if regimes:
        regime_counts = {}
        for regime in regimes.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Find most common regime
        most_common_regime = max(regime_counts.items(), key=lambda x: x[1])
        alignment['regime_alignment'] = most_common_regime[1] / len(regimes)
        
        # Identify aligned and conflicting timeframes
        for tf, regime in regimes.items():
            if regime == most_common_regime[0]:
                alignment['aligned_timeframes'].append(tf)
            else:
                alignment['conflicting_timeframes'].append(tf)
    
    # Calculate direction alignment
    if directions:
        direction_counts = {}
        for direction in directions.values():
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Find most common direction
        most_common_direction = max(direction_counts.items(), key=lambda x: x[1])
        alignment['direction_alignment'] = most_common_direction[1] / len(directions)
    
    # Calculate entropy alignment
    if entropy_states:
        entropy_counts = {}
        for state in entropy_states.values():
            entropy_counts[state] = entropy_counts.get(state, 0) + 1
        
        # Find most common entropy state
        most_common_entropy = max(entropy_counts.items(), key=lambda x: x[1])
        alignment['entropy_alignment'] = most_common_entropy[1] / len(entropy_states)
    
    # Calculate overall alignment
    weights = {
        'regime_alignment': 0.4,
        'direction_alignment': 0.4,
        'entropy_alignment': 0.2
    }
    
    alignment['overall_alignment'] = (
        alignment['regime_alignment'] * weights['regime_alignment'] +
        alignment['direction_alignment'] * weights['direction_alignment'] +
        alignment['entropy_alignment'] * weights['entropy_alignment']
    )
    
    return alignment

def generate_ml_enhanced_recommendations(tickers, rainbow_results, ml_predictions, confidence_threshold=0.75):
    """
    Generate ML-enhanced trade recommendations with rainbow analysis.
    
    Args:
        tickers: List of ticker symbols
        rainbow_results: Results from rainbow analysis
        ml_predictions: Results from ML predictions
        confidence_threshold: Minimum confidence threshold for recommendations
        
    Returns:
        Dictionary with trade recommendations
    """
    logger.info(f"Generating ML-enhanced recommendations for {len(tickers)} tickers")
    
    # Load configuration
    config = load_config()
    
    # Initialize trade executor and analyzer
    executor = MLTradeExecutor(config=config, risk_config=config.get('risk_config', {}))
    
    # Get current prices
    current_prices = get_current_prices(tickers, use_mock=False, api_key=config.get('api_config', {}).get('api_key'))
    
    # Initialize recommendations
    recommendations = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'entry_signals': [],
        'exit_signals': [],
        'watchlist': []
    }
    
    # Process each ticker
    for ticker in tickers:
        price = current_prices.get(ticker)
        if not price:
            logger.warning(f"No price data for {ticker}, skipping")
            continue
        
        # Get rainbow results for this ticker
        ticker_rainbow = rainbow_results.get('tickers', {}).get(ticker, {})
        if not ticker_rainbow:
            logger.warning(f"No rainbow analysis for {ticker}, skipping")
            continue
        
        # Get ML predictions for this ticker
        ticker_ml = ml_predictions.get(ticker, {})
        
        # Calculate rainbow confidence
        rainbow_alignment = rainbow_results.get('timeframe_alignment', {}).get(ticker, {})
        rainbow_confidence = rainbow_alignment.get('overall_alignment', 0)
        
        # Calculate ML confidence
        ml_confidence = 0
        if 'ml_predictions' in ticker_ml:
            primary_regime = ticker_ml['ml_predictions'].get('primary_regime', {})
            if 'confidence' in primary_regime:
                ml_confidence = primary_regime['confidence']
        
        # Calculate combined confidence
        combined_confidence = (rainbow_confidence * 0.6) + (ml_confidence * 0.4)
        
        # Determine action based on rainbow and ML analysis
        action = 'none'
        reasoning = []
        
        # Check for aligned signals across timeframes
        if rainbow_confidence >= 0.7:
            # Get the dominant regime and direction
            daily_results = ticker_rainbow.get('1d', {})
            daily_regime = daily_results.get('market_regime', {})
            
            primary_regime = daily_regime.get('primary_label', 'Unknown')
            direction = daily_regime.get('direction', 'Neutral')
            
            # Determine action based on regime and direction
            if primary_regime == 'Vanna-Driven' and direction == 'Bullish':
                action = 'entry'
                reasoning.append(f"Vanna-driven bullish regime aligned across {int(rainbow_confidence*100)}% of timeframes")
            elif primary_regime == 'Vanna-Driven' and direction == 'Bearish':
                action = 'entry'
                reasoning.append(f"Vanna-driven bearish regime aligned across {int(rainbow_confidence*100)}% of timeframes")
            elif primary_regime == 'Charm-Dominated':
                action = 'entry'
                reasoning.append(f"Charm-dominated regime aligned across {int(rainbow_confidence*100)}% of timeframes")
        
        # Add ML insights
        if 'trade_signals' in ticker_ml:
            ml_signal = ticker_ml['trade_signals'].get('entry', {})
            if ml_signal.get('signal') in ['bullish', 'bearish'] and ml_signal.get('strength', 0) >= 2:
                action = 'entry'
                reasoning.append(f"ML model predicts {ml_signal['signal']} move with strength {ml_signal['strength']}")
                reasoning.extend(ml_signal.get('reasons', [])[:2])
        
        # Check for regime transitions
        if 'regime_transition' in ticker_ml:
            transition = ticker_ml['regime_transition']
            if transition.get('probability', 0) >= 0.7:
                reasoning.append(f"Potential regime transition to {transition.get('target_regime', 'Unknown')}")
                if transition.get('trade_implication') == 'bullish':
                    action = 'entry'
                    reasoning.append("Regime transition has bullish implications")
                elif transition.get('trade_implication') == 'bearish':
                    action = 'entry'
                    reasoning.append("Regime transition has bearish implications")
        
        # Only proceed if we have an action and sufficient confidence
        if action == 'entry' and combined_confidence >= confidence_threshold:
            # Get the daily timeframe results for detailed analysis
            daily_results = ticker_rainbow.get('1d', {})
            daily_regime = daily_results.get('market_regime', {})
            daily_entropy = daily_results.get('entropy_state', {})
            
            # Determine entry parameters
            entry_price = price
            stop_loss = 0
            take_profit = 0
            
            # Calculate stop loss and take profit based on volatility and support/resistance
            volatility = daily_regime.get('volatility_regime', 'Normal')
            if volatility == 'High':
                stop_pct = 0.05  # 5% for high volatility
            elif volatility == 'Low':
                stop_pct = 0.02  # 2% for low volatility
            else:
                stop_pct = 0.03  # 3% for normal volatility
            
            # Adjust based on direction
            direction = daily_regime.get('direction', 'Neutral')
            if direction == 'Bullish':
                stop_loss = entry_price * (1 - stop_pct)
                take_profit = entry_price * (1 + (stop_pct * 2))  # 2:1 reward-risk ratio
            elif direction == 'Bearish':
                stop_loss = entry_price * (1 + stop_pct)
                take_profit = entry_price * (1 - (stop_pct * 2))  # 2:1 reward-risk ratio
            else:
                # For neutral, use a straddle approach
                stop_loss = entry_price * (1 - (stop_pct / 2))
                take_profit = entry_price * (1 + stop_pct)
            
            # Calculate position size based on risk
            account_size = config.get('risk_config', {}).get('account_size', 100000)
            risk_per_trade = config.get('risk_config', {}).get('risk_per_trade', 0.01)
            max_risk_amount = account_size * risk_per_trade
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                position_size = int(max_risk_amount / risk_per_share)
            else:
                position_size = 0
            
            # Create recommendation
            entry_rec = {
                'ticker': ticker,
                'price': price,
                'action': direction.lower() if direction in ['Bullish', 'Bearish'] else 'neutral',
                'confidence': combined_confidence,
                'rainbow_confidence': rainbow_confidence,
                'ml_confidence': ml_confidence,
                'reasoning': reasoning,
                'price_targets': {
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                },
                'position_size': position_size,
                'risk_assessment': {
                    'volatility': volatility,
                    'risk_amount': max_risk_amount,
                    'risk_percent': risk_per_trade * 100
                },
                'timeframe_alignment': rainbow_alignment,
                'regime': primary_regime
            }
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                entry_rec['risk_reward_ratio'] = reward / risk
            else:
                entry_rec['risk_reward_ratio'] = 0
            
            recommendations['entry_signals'].append(entry_rec)
            
        elif combined_confidence >= 0.5 and combined_confidence < confidence_threshold:
            # Add to watchlist
            watch_rec = {
                'ticker': ticker,
                'price': price,
                'action': action,
                'confidence': combined_confidence,
                'rainbow_confidence': rainbow_confidence,
                'ml_confidence': ml_confidence,
                'reasoning': reasoning[:3]  # Limit to top 3 reasons
            }
            
            recommendations['watchlist'].append(watch_rec)
    
    # Sort recommendations by confidence
    recommendations['entry_signals'] = sorted(
        recommendations['entry_signals'], 
        key=lambda x: (x.get('confidence', 0), x.get('risk_reward_ratio', 0)), 
        reverse=True
    )
    
    recommendations['watchlist'] = sorted(
        recommendations['watchlist'], 
        key=lambda x: x.get('confidence', 0), 
        reverse=True
    )
    
    # Save recommendations to file
    output_file = os.path.join('results', f"rainbow_ml_recommendations_{datetime.now().strftime('%Y%m%d')}.json")
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"Generated {len(recommendations['entry_signals'])} entry signals and "
                f"{len(recommendations['watchlist'])} watchlist items")
    
    return recommendations

def run_ml_enhanced_analysis(tickers, output_dir="results/ml_enhanced", test_mode=False):
    """
    Run ML-enhanced analysis for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        output_dir: Output directory for enhanced analysis
        test_mode: If True, run in test mode
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Running ML-enhanced analysis for {len(tickers)} tickers")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ML-enhanced analyzer
    analyzer = MLEnhancedAnalyzer(test_mode=test_mode)
    
    # Process each ticker
    results = {}
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}...")
            
            # Get current price (would fetch from API in production)
            # For testing, use a dummy price or extract from analysis data
            price = None  # Will be extracted from analysis data
            
            # Run enhanced analysis
            enhanced_analysis = analyzer.analyze_ticker(
                ticker=ticker,
                price=price,
                results_dir="results",
                predictions_dir="results/ml_predictions"
            )
            
            if enhanced_analysis:
                results[ticker] = {
                    "status": "success",
                    "recommendation": enhanced_analysis.get("recommendation", {})
                }
                
                # Save to output directory
                output_path = os.path.join(output_dir, f"{ticker}_enhanced_analysis.json")
                with open(output_path, 'w') as f:
                    json.dump(enhanced_analysis, f, indent=2)
            else:
                results[ticker] = {
                    "status": "error",
                    "message": "Failed to generate enhanced analysis"
                }
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            results[ticker] = {
                "status": "error",
                "message": str(e)
            }
    
    # Save summary results
    summary_path = os.path.join(output_dir, f"ml_enhanced_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main entry point for the ML-enhanced trading system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ML-Enhanced Trading System for Greek Energy Flow II")
    
    # Add arguments
    parser.add_argument("--tickers", nargs="+", help="List of ticker symbols to analyze")
    parser.add_argument("--ticker-file", help="CSV file with ticker symbols")
    parser.add_argument("--output-dir", default="results", help="Output directory for analysis files")
    parser.add_argument("--train", action="store_true", help="Train ML models on existing analysis data")
    parser.add_argument("--predict", action="store_true", help="Run prediction using trained ML models")
    parser.add_argument("--analyze", action="store_true", help="Run enhanced analysis with ML predictions")
    parser.add_argument("--simulate", action="store_true", help="Run trading simulation with ML-enhanced signals")
    parser.add_argument("--live", action="store_true", help="Run live trading with ML-enhanced signals")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes for live trading")
    parser.add_argument("--simulation-days", type=int, default=5, help="Number of days to simulate")
    parser.add_argument("--api-key", help="API key for data provider")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    parser.add_argument("--use-real-data", action="store_true", help="Force use of real market data")
    parser.add_argument("--use-mock-data", action="store_true", help="Force use of mock data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get tickers from args or CSV
    tickers = []
    if args.tickers:
        tickers = args.tickers
    elif args.ticker_file:
        tickers = load_tickers_from_file(args.ticker_file)
    else:
        # Default tickers
        tickers = ["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"]
    
    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Determine whether to use mock data
    use_mock = args.use_mock_data
    if args.use_real_data:
        use_mock = False
    elif args.test_mode:
        use_mock = True
    
    # Train ML models if requested
    if args.train:
        logger.info("Training ML models")
        train_results = train_ml_models(
            tickers=tickers, 
            results_dir=args.output_dir,
            use_mock=use_mock
        )
        
        if not train_results:
            logger.error("ML model training failed")
            return 1
        
        logger.info("ML model training completed successfully")
    
    # Run prediction if requested
    if args.predict:
        logger.info("Running ML prediction")
        prediction_results = run_ml_prediction(
            tickers=tickers,
            results_dir=args.output_dir,
            output_dir=os.path.join(args.output_dir, "ml_predictions"),
            use_mock=use_mock
        )
        
        if not prediction_results:
            logger.error("ML prediction failed")
            return 1
        
        logger.info("ML prediction completed successfully")
    
    # Run analysis if requested
    if args.analyze:
        logger.info("Running enhanced analysis")
        analysis_results = run_complete_analysis(
            tickers=tickers,
            output_dir=args.output_dir,
            fetch_data=True,
            days=60,
            api_key=args.api_key,
            use_mock=use_mock
        )
        
        if not analysis_results:
            logger.error("Enhanced analysis failed")
            return 1
        
        logger.info("Enhanced analysis completed successfully")
    
    # Run simulation if requested
    if args.simulate:
        logger.info(f"Running trading simulation for {args.simulation_days} days")
        simulation_results = run_trading_simulation(
            tickers=tickers,
            days=args.simulation_days,
            interval_minutes=args.interval,
            results_dir=args.output_dir,
            use_mock=use_mock
        )
        
        if not simulation_results:
            logger.error("Trading simulation failed")
            return 1
        
        logger.info("Trading simulation completed successfully")
    
    # Run live trading if requested
    if args.live:
        logger.info(f"Running live trading with {args.interval} minute interval")
        live_results = run_live_trading(
            tickers=tickers,
            interval_minutes=args.interval,
            results_dir=args.output_dir,
            api_key=args.api_key,
            use_mock=use_mock
        )
        
        if not live_results:
            logger.error("Live trading failed")
            return 1
        
        logger.info("Live trading completed successfully")
    
    logger.info("ML-enhanced trading system completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())










