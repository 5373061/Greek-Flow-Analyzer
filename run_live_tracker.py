#!/usr/bin/env python
"""
Live Instrument Tracker for Greek Energy Flow Analysis
For production use with real-time market data
"""

import os
import sys
import logging
import argparse
import time
import json
from datetime import datetime, timedelta
import signal
import requests
import pandas as pd
import numpy as np
from tabulate import tabulate
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/live_tracker_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LiveTracker")

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    logger.info("Received termination signal. Shutting down gracefully...")
    sys.exit(0)

def fetch_fresh_market_data(symbol, api_key=None):
    """
    Fetch fresh market data for a symbol.
    
    Args:
        symbol (str): Ticker symbol
        api_key (str): API key for data provider
        
    Returns:
        dict: Market data
    """
    logger.info(f"Fetching fresh market data for {symbol}")
    
    try:
        # Try Alpha Vantage API
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            price = float(data["Global Quote"]["05. price"])
            change_pct = float(data["Global Quote"]["10. change percent"].strip("%"))
            volume = float(data["Global Quote"]["06. volume"])
            
            return {
                "symbol": symbol,
                "price": price,
                "change_percent": change_pct,
                "volume": volume,
                "timestamp": datetime.now().isoformat(),
                "source": "Alpha Vantage"
            }
        else:
            # Fallback to Yahoo Finance API
            yf_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            yf_response = requests.get(yf_url)
            yf_data = yf_response.json()
            
            if "chart" in yf_data and "result" in yf_data["chart"] and yf_data["chart"]["result"]:
                result = yf_data["chart"]["result"][0]
                price = result["meta"]["regularMarketPrice"]
                prev_close = result["meta"]["previousClose"]
                change_pct = (price - prev_close) / prev_close * 100
                volume = result["meta"].get("regularMarketVolume", 0)
                
                return {
                    "symbol": symbol,
                    "price": price,
                    "change_percent": change_pct,
                    "volume": volume,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Yahoo Finance"
                }
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
    
    # Return placeholder data as last resort
    return {
        "symbol": symbol,
        "price": 100.0,
        "change_percent": 0.0,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat(),
        "source": "placeholder"
    }

def fetch_historical_data(symbol, days=120, api_key=None):
    """
    Fetch historical price data for a symbol.
    
    Args:
        symbol (str): Ticker symbol
        days (int): Number of days of history
        api_key (str): API key for data provider
        
    Returns:
        pd.DataFrame: Historical price data
    """
    logger.info(f"Fetching historical data for {symbol}")
    
    try:
        # Try Alpha Vantage API
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full" if days > 100 else "compact",
            "apikey": api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            # Convert to DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df.tail(days)
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
    
    # Return empty DataFrame if failed
    return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for analysis.
    
    Args:
        df (pd.DataFrame): Historical price data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    if df.empty:
        return df
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
    
    # Momentum
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    return df

def analyze_market_regime(df):
    """
    Analyze market regime based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Market regime analysis
    """
    if df.empty or len(df) < 50:
        return {
            "regime": "unknown",
            "trend": "unknown",
            "volatility": "unknown",
            "momentum": "unknown",
            "confidence": 0.0
        }
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Determine trend
    trend = "neutral"
    trend_confidence = 0.5
    
    if latest['Close'] > latest['SMA50'] > latest['SMA200']:
        trend = "bullish"
        trend_confidence = min(1.0, 0.5 + (latest['Close'] / latest['SMA200'] - 1) * 5)
    elif latest['Close'] < latest['SMA50'] < latest['SMA200']:
        trend = "bearish"
        trend_confidence = min(1.0, 0.5 + (1 - latest['Close'] / latest['SMA200']) * 5)
    
    # Determine volatility
    volatility = "normal"
    recent_atr = df['ATR'].iloc[-20:].mean()
    historical_atr = df['ATR'].mean()
    volatility_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0
    
    if volatility_ratio > 1.5:
        volatility = "high"
    elif volatility_ratio < 0.7:
        volatility = "low"
    
    # Determine momentum
    momentum = "neutral"
    if latest['RSI'] > 70:
        momentum = "overbought"
    elif latest['RSI'] < 30:
        momentum = "oversold"
    elif latest['RSI'] > 50 and latest['MACD'] > 0:
        momentum = "positive"
    elif latest['RSI'] < 50 and latest['MACD'] < 0:
        momentum = "negative"
    
    # Determine overall regime
    regime = "neutral"
    if trend == "bullish" and momentum in ["positive", "neutral"]:
        regime = "bullish"
    elif trend == "bearish" and momentum in ["negative", "neutral"]:
        regime = "bearish"
    elif momentum == "overbought" and trend != "bullish":
        regime = "distribution"
    elif momentum == "oversold" and trend != "bearish":
        regime = "accumulation"
    
    # Calculate confidence
    confidence = trend_confidence
    
    return {
        "regime": regime,
        "trend": trend,
        "volatility": volatility,
        "momentum": momentum,
        "confidence": round(confidence, 2)
    }

def analyze_support_resistance(df, current_price):
    """
    Analyze support and resistance levels.
    
    Args:
        df (pd.DataFrame): Historical price data
        current_price (float): Current price
        
    Returns:
        dict: Support and resistance levels
    """
    if df.empty or len(df) < 30:
        return {
            "support": [round(current_price * 0.95, 2)],
            "resistance": [round(current_price * 1.05, 2)]
        }
    
    # Find pivot points (local highs and lows)
    pivot_high = []
    pivot_low = []
    
    # Use a simple method to identify pivot points
    for i in range(5, len(df) - 5):
        # Check if this is a local high
        if (df['High'].iloc[i] > df['High'].iloc[i-1:i+1].max() and 
            df['High'].iloc[i] > df['High'].iloc[i+1:i+6].max()):
            pivot_high.append(df['High'].iloc[i])
        
        # Check if this is a local low
        if (df['Low'].iloc[i] < df['Low'].iloc[i-5:i].min() and 
            df['Low'].iloc[i] < df['Low'].iloc[i+1:i+6].min()):
            pivot_low.append(df['Low'].iloc[i])
    
    # Find recent support levels below current price
    support = [level for level in pivot_low if level < current_price]
    support.sort(reverse=True)  # Sort from highest to lowest
    
    # Find recent resistance levels above current price
    resistance = [level for level in pivot_high if level > current_price]
    resistance.sort()  # Sort from lowest to highest
    
    # If we don't have enough levels, add some based on percentages
    if len(support) < 2:
        support.append(round(current_price * 0.95, 2))
        support.append(round(current_price * 0.90, 2))
    
    if len(resistance) < 2:
        resistance.append(round(current_price * 1.05, 2))
        resistance.append(round(current_price * 1.10, 2))
    
    # Round values and take top 3
    support = [round(level, 2) for level in support[:3]]
    resistance = [round(level, 2) for level in resistance[:3]]
    
    return {
        "support": support,
        "resistance": resistance
    }

def analyze_risk_reward(current_price, support, resistance):
    """
    Analyze risk/reward ratios for potential trades.
    
    Args:
        current_price (float): Current price
        support (list): Support levels
        resistance (list): Resistance levels
        
    Returns:
        dict: Risk/reward analysis
    """
    # For long positions
    long_entry = current_price
    long_stop = support[0] if support else current_price * 0.95
    long_target1 = resistance[0] if resistance else current_price * 1.05
    long_target2 = resistance[1] if len(resistance) > 1 else current_price * 1.10
    
    long_risk = long_entry - long_stop
    long_reward1 = long_target1 - long_entry
    long_reward2 = long_target2 - long_entry
    
    long_rr1 = round(long_reward1 / long_risk, 2) if long_risk > 0 else 0
    long_rr2 = round(long_reward2 / long_risk, 2) if long_risk > 0 else 0
    
    # For short positions
    short_entry = current_price
    short_stop = resistance[0] if resistance else current_price * 1.05
    short_target1 = support[0] if support else current_price * 0.95
    short_target2 = support[1] if len(support) > 1 else current_price * 0.90
    
    short_risk = short_stop - short_entry
    short_reward1 = short_entry - short_target1
    short_reward2 = short_entry - short_target2
    
    short_rr1 = round(short_reward1 / short_risk, 2) if short_risk > 0 else 0
    short_rr2 = round(short_reward2 / short_risk, 2) if short_risk > 0 else 0
    
    return {
        "long": {
            "entry": round(long_entry, 2),
            "stop": round(long_stop, 2),
            "targets": [round(long_target1, 2), round(long_target2, 2)],
            "risk_reward_ratios": [long_rr1, long_rr2],
            "average_rr": round((long_rr1 + long_rr2) / 2, 2)
        },
        "short": {
            "entry": round(short_entry, 2),
            "stop": round(short_stop, 2),
            "targets": [round(short_target1, 2), round(short_target2, 2)],
            "risk_reward_ratios": [short_rr1, short_rr2],
            "average_rr": round((short_rr1 + short_rr2) / 2, 2)
        }
    }

def generate_options_strategy(symbol, current_price, regime, volatility):
    """
    Generate options strategy recommendations.
    
    Args:
        symbol (str): Ticker symbol
        current_price (float): Current price
        regime (str): Market regime
        volatility (str): Volatility regime
        
    Returns:
        dict: Options strategy recommendations
    """
    strategies = []
    
    # Bullish strategies
    if regime == "bullish":
        if volatility == "high":
            strategies.append({
                "name": "Bull Call Spread",
                "type": "debit",
                "strikes": [
                    round(current_price * 1.00, 1),
                    round(current_price * 1.10, 1)
                ],
                "expiration": "30-45 DTE",
                "max_risk": "Limited to premium paid",
                "max_reward": "Limited to difference between strikes minus premium",
                "notes": "Good for bullish outlook with high volatility"
            })
        else:
            strategies.append({
                "name": "Long Call",
                "type": "debit",
                "strikes": [round(current_price * 1.05, 1)],
                "expiration": "30-60 DTE",
                "max_risk": "Limited to premium paid",
                "max_reward": "Unlimited",
                "notes": "Good for bullish outlook with low/normal volatility"
            })
    
    # Bearish strategies
    elif regime == "bearish":
        if volatility == "high":
            strategies.append({
                "name": "Bear Put Spread",
                "type": "debit",
                "strikes": [
                    round(current_price * 1.00, 1),
                    round(current_price * 0.90, 1)
                ],
                "expiration": "30-45 DTE",
                "max_risk": "Limited to premium paid",
                "max_reward": "Limited to difference between strikes minus premium",
                "notes": "Good for bearish outlook with high volatility"
            })
        else:
            strategies.append({
                "name": "Long Put",
                "type": "debit",
                "strikes": [round(current_price * 0.95, 1)],
                "expiration": "30-60 DTE",
                "max_risk": "Limited to premium paid",
                "max_reward": "Unlimited",
                "notes": "Good for bearish outlook with low/normal volatility"
            })
    
    return strategies

def main():
    """Main entry point for the live tracker."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live Instrument Tracker for Greek Energy Flow Analysis")
    parser.add_argument("--instruments", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ"],
                        help="List of instruments to track")
    parser.add_argument("--interval", type=int, default=15,
                        help="Tracking interval in minutes")
    parser.add_argument("--market-hours-only", action="store_true",
                        help="Only track during market hours")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory for data storage")
    parser.add_argument("--output-dir", default="./results/reports",
                        help="Directory for output reports")
    parser.add_argument("--api-key", help="API key for market data provider")
    parser.add_argument("--fresh-data", action="store_true", 
                        help="Fetch fresh market data instead of using cached data")
    
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting Live Tracker with {len(args.instruments)} instruments")
    logger.info(f"Instruments: {', '.join(args.instruments)}")
    
    try:
        # Import InstrumentTracker
        from tools.instrument_tracker import InstrumentTracker
        
        # Initialize tracker
        tracker = InstrumentTracker(
            instruments=args.instruments,
            data_dir=args.data_dir
        )
        
        # Initial tracking to ensure we have data
        logger.info("Running initial tracking cycle...")
        
        if args.fresh_data:
            # Fetch fresh data and generate recommendations
            logger.info("Fetching fresh market data and generating recommendations...")
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Process each instrument
            for symbol in args.instruments:
                # Fetch fresh market data
                market_data = fetch_fresh_market_data(symbol, args.api_key)
                
                # Fetch historical data
                historical_data = fetch_historical_data(symbol, days=60, api_key=args.api_key)
                
                # Generate recommendation
                # Import the function from the correct module
                from analysis.trade_recommendations import generate_trade_recommendation
                
                # Generate recommendation with the proper parameters
                recommendation = generate_trade_recommendation(
                    analysis_results=market_data,
                    entropy_data=historical_data.get("entropy_data", {}),
                    current_price=market_data.get("price", 0.0)
                )
                
                # Save recommendation to file
                rec_file = os.path.join(args.output_dir, f"{symbol}_recommendation.json")
                with open(rec_file, 'w') as f:
                    json.dump(recommendation, f, indent=2)
                
                logger.info(f"Generated fresh recommendation for {symbol}")
                
                # Print summary to console
                print(f"\n{symbol} - ${market_data['price']:.2f} ({market_data['change_percent']:.2f}%)")
                print(f"Recommendation: {recommendation['action']} (Confidence: {recommendation['confidence']:.2f})")
                print(f"Entry Zone: ${recommendation['entry_zone']['low']} - ${recommendation['entry_zone']['high']}")
                print(f"Targets: {', '.join(['$' + str(t) for t in recommendation['targets']])}")
                print(f"Stop Loss: ${recommendation['stop_loss']}")
                print(f"Notes: {recommendation['notes']}")
                print("-" * 50)
                
                # Avoid hitting API rate limits
                time.sleep(1)
        else:
            # Use the existing tracker
            tracker.safe_track()
        
        # Start continuous monitoring if requested
        if not args.fresh_data:
            logger.info(f"Starting continuous monitoring with {args.interval}-minute intervals")
            if args.market_hours_only:
                tracker.start_market_hours_monitoring()
            else:
                tracker.start_real_time_monitoring(interval_minutes=args.interval)
            
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())



