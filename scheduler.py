"""
Greek Energy Flow Analysis Scheduler

This script provides scheduling capabilities to run the analysis at specified intervals.
It can be configured to run daily, weekly, or at specific times.
"""

import os
import sys
import time
import logging
import argparse
import schedule  # pip install schedule
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"scheduler_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from run_analysis import main as run_analysis_main
    from api_fetcher import fetch_options_chain_snapshot, fetch_underlying_snapshot
    import config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def is_market_open():
    """Check if the US stock market is currently open"""
    now = datetime.now()
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM Eastern Time
    # Note: This is a simplified check and doesn't account for holidays
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def fetch_data_for_symbol(symbol, data_dir="data", timeframe="1d"):
    """Fetch fresh data for a symbol and save to data directory"""
    logger.info(f"Fetching fresh data for {symbol} with {timeframe} timeframe")
    
    # Create directories if they don't exist
    options_dir = os.path.join(data_dir, "options")
    prices_dir = os.path.join(data_dir, "prices")
    os.makedirs(options_dir, exist_ok=True)
    os.makedirs(prices_dir, exist_ok=True)
    
    # Get API key from config
    api_key = getattr(config, "POLYGON_API_KEY", None)
    if not api_key:
        logger.error("API key not found in config")
        return False
    
    try:
        # Fetch options data
        options_data = fetch_options_chain_snapshot(symbol, api_key)
        if not options_data:
            logger.error(f"Failed to fetch options data for {symbol}")
            return False
        
        # Fetch underlying data with specified timeframe
        underlying_data = fetch_underlying_snapshot(symbol, api_key, timeframe=timeframe)
        if not underlying_data:
            logger.error(f"Failed to fetch underlying data for {symbol}")
            return False
        
        # Save options data
        options_file = os.path.join(options_dir, f"{symbol}_options_{datetime.now().strftime('%Y%m%d')}.json")
        with open(options_file, "w") as f:
            import json
            json.dump(options_data, f)
        
        # Save underlying data
        underlying_file = os.path.join(prices_dir, f"{symbol}_underlying_{datetime.now().strftime('%Y%m%d')}.json")
        with open(underlying_file, "w") as f:
            import json
            json.dump(underlying_data, f)
        
        logger.info(f"Successfully fetched and saved data for {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False

def fetch_data_for_all_symbols(instruments_file="data/instruments.csv", data_dir="data", timeframe="1d"):
    """Fetch fresh data for all symbols in the instruments file"""
    logger.info(f"Starting data fetch for all symbols with {timeframe} timeframe")
    
    try:
        # Load instruments
        from utils.instrument_loader import load_instruments_from_csv
        instruments_df = load_instruments_from_csv(instruments_file)
        
        if instruments_df is None or instruments_df.empty:
            logger.error(f"No instruments found in {instruments_file}")
            return False
        
        symbols = instruments_df["symbol"].tolist()
        logger.info(f"Found {len(symbols)} symbols to fetch")
        
        # Fetch data for each symbol
        success_count = 0
        for symbol in symbols:
            if fetch_data_for_symbol(symbol, data_dir, timeframe):
                success_count += 1
            
            # Add a small delay between API calls to avoid rate limits
            time.sleep(1)
        
        logger.info(f"Completed data fetch: {success_count}/{len(symbols)} symbols successful")
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error in fetch_data_for_all_symbols: {e}")
        return False

def run_daily_analysis(instruments_file="data/instruments.csv", output_dir="results", timeframe="1d"):
    """Run the daily analysis on all symbols"""
    logger.info(f"Starting daily analysis with {timeframe} timeframe")
    
    # First, fetch fresh data
    if not fetch_data_for_all_symbols(instruments_file, timeframe=timeframe):
        logger.error("Failed to fetch data, aborting analysis")
        return False
    
    # Create a timestamp for today's analysis
    today = datetime.now().strftime("%Y%m%d")
    today_output_dir = os.path.join(output_dir, today)
    os.makedirs(today_output_dir, exist_ok=True)
    
    # Run the analysis
    try:
        # Prepare arguments for run_analysis_main
        sys.argv = [
            "run_analysis.py",
            "batch",
            instruments_file,
            "--output", today_output_dir,
            "--options-dir", "data/options",
            "--prices-dir", "data/prices",
            "--timeframe", timeframe
        ]
        
        # Run the analysis
        run_analysis_main()
        
        logger.info(f"Daily analysis completed successfully. Results saved to {today_output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error running daily analysis: {e}")
        return False

def run_weekly_summary(output_dir="results"):
    """Generate a weekly summary of analysis results"""
    logger.info("Generating weekly summary")
    
    # Get the date range for this week
    today = datetime.now()
    start_of_week = (today - timedelta(days=today.weekday())).strftime("%Y%m%d")
    end_of_week = (today + timedelta(days=6-today.weekday())).strftime("%Y%m%d")
    
    # Create a weekly summary directory
    weekly_dir = os.path.join(output_dir, f"weekly_{start_of_week}_to_{end_of_week}")
    os.makedirs(weekly_dir, exist_ok=True)
    
    try:
        # Collect all daily results for the week
        all_results = []
        
        # Loop through each day of the week
        for i in range(7):
            day = (today - timedelta(days=today.weekday()) + timedelta(days=i))
            day_str = day.strftime("%Y%m%d")
            day_dir = os.path.join(output_dir, day_str)
            
            if os.path.exists(day_dir):
                # Look for result files
                for file in os.listdir(day_dir):
                    if file.endswith(".json"):
                        try:
                            import json
                            with open(os.path.join(day_dir, file), "r") as f:
                                result = json.load(f)
                                result["date"] = day_str
                                all_results.append(result)
                        except Exception as e:
                            logger.warning(f"Error reading result file {file}: {e}")
        
        # Create a summary report
        if all_results:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(all_results)
            
            # Save the full dataset
            df.to_csv(os.path.join(weekly_dir, "weekly_results.csv"), index=False)
            
            # Generate summary statistics
            summary = {
                "total_symbols_analyzed": df["symbol"].nunique(),
                "total_analyses": len(df),
                "date_range": f"{start_of_week} to {end_of_week}",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save the summary
            with open(os.path.join(weekly_dir, "summary.json"), "w") as f:
                import json
                json.dump(summary, f, indent=2)
            
            logger.info(f"Weekly summary generated successfully. Saved to {weekly_dir}")
            return True
        else:
            logger.warning("No results found for the week")
            return False
    
    except Exception as e:
        logger.error(f"Error generating weekly summary: {e}")
        return False

def schedule_jobs(mode="custom", morning_time="09:00", midday_time="12:30", afternoon_time="16:30"):
    """Schedule jobs based on the specified mode"""
    if mode == "daily":
        # Schedule daily analysis after market close
        schedule.every().monday.at("16:30").do(run_daily_analysis)
        schedule.every().tuesday.at("16:30").do(run_daily_analysis)
        schedule.every().wednesday.at("16:30").do(run_daily_analysis)
        schedule.every().thursday.at("16:30").do(run_daily_analysis)
        schedule.every().friday.at("16:30").do(run_daily_analysis)
        
        # Schedule weekly summary on Friday after daily analysis
        schedule.every().friday.at("18:00").do(run_weekly_summary)
        
        logger.info(f"Scheduled daily analysis at 16:30 on weekdays")
        logger.info("Scheduled weekly summary at 18:00 on Fridays")
    
    elif mode == "weekly":
        # Schedule weekly analysis on Friday after market close
        schedule.every().friday.at("16:30").do(run_daily_analysis)
        
        # Schedule weekly summary after analysis
        schedule.every().friday.at("18:00").do(run_weekly_summary)
        
        logger.info(f"Scheduled weekly analysis at 16:30 on Fridays")
        logger.info(f"Scheduled weekly summary at 18:00 on Fridays")
    
    elif mode == "market_close":
        # Schedule to run at market close (4:00 PM Eastern)
        schedule.every().monday.at("16:00").do(run_daily_analysis)
        schedule.every().tuesday.at("16:00").do(run_daily_analysis)
        schedule.every().wednesday.at("16:00").do(run_daily_analysis)
        schedule.every().thursday.at("16:00").do(run_daily_analysis)
        schedule.every().friday.at("16:00").do(run_daily_analysis)
        
        # Schedule weekly summary on Friday after daily analysis
        schedule.every().friday.at("18:00").do(run_weekly_summary)
        
        logger.info("Scheduled daily analysis at market close (16:00) on weekdays")
        logger.info("Scheduled weekly summary at 18:00 on Fridays")
    
    elif mode == "custom":
        # Schedule pre-market analysis at 9:00 AM
        schedule.every().monday.at(morning_time).do(run_daily_analysis)
        schedule.every().tuesday.at(morning_time).do(run_daily_analysis)
        schedule.every().wednesday.at(morning_time).do(run_daily_analysis)
        schedule.every().thursday.at(morning_time).do(run_daily_analysis)
        schedule.every().friday.at(morning_time).do(run_daily_analysis)
        
        # Schedule mid-day analysis at 12:30 PM
        schedule.every().monday.at(midday_time).do(run_daily_analysis)
        schedule.every().tuesday.at(midday_time).do(run_daily_analysis)
        schedule.every().wednesday.at(midday_time).do(run_daily_analysis)
        schedule.every().thursday.at(midday_time).do(run_daily_analysis)
        schedule.every().friday.at(midday_time).do(run_daily_analysis)
        
        # Schedule post-market analysis at 4:30 PM
        schedule.every().monday.at(afternoon_time).do(run_daily_analysis)
        schedule.every().tuesday.at(afternoon_time).do(run_daily_analysis)
        schedule.every().wednesday.at(afternoon_time).do(run_daily_analysis)
        schedule.every().thursday.at(afternoon_time).do(run_daily_analysis)
        schedule.every().friday.at(afternoon_time).do(run_daily_analysis)
        
        # Schedule weekly summary on Friday after afternoon analysis
        friday_afternoon = datetime.strptime(afternoon_time, "%H:%M")
        summary_time = (friday_afternoon + timedelta(hours=1)).strftime("%H:%M")
        schedule.every().friday.at(summary_time).do(run_weekly_summary)
        
        logger.info(f"Scheduled pre-market analysis at {morning_time} on weekdays")
        logger.info(f"Scheduled mid-day analysis at {midday_time} on weekdays")
        logger.info(f"Scheduled post-market analysis at {afternoon_time} on weekdays")
        logger.info(f"Scheduled weekly summary at {summary_time} on Fridays")
    
    else:
        logger.error(f"Unknown scheduling mode: {mode}")
        return False
    
    return True

def schedule_adaptive_jobs(base_schedule="daily", vix_threshold=25):
    """Schedule jobs adaptively based on market volatility"""
    # Get current VIX level
    try:
        from api_fetcher import fetch_underlying_snapshot
        api_key = getattr(config, "POLYGON_API_KEY", None)
        if api_key:
            vix_data = fetch_underlying_snapshot("VIX", api_key)
            vix_level = vix_data.get("results", {}).get("last", {}).get("price", 20)
        else:
            logger.warning("No API key found, using default VIX level")
            vix_level = 20
    except Exception as e:
        logger.warning(f"Error fetching VIX level: {e}. Using default value.")
        vix_level = 20
    
    if vix_level > vix_threshold:
        # High volatility - schedule more frequent analysis
        logger.info(f"High volatility detected (VIX: {vix_level}). Using frequent schedule.")
        schedule.every().monday.at("09:30").do(run_daily_analysis)
        schedule.every().monday.at("12:30").do(run_daily_analysis)
        schedule.every().monday.at("15:30").do(run_daily_analysis)
        # Repeat for other weekdays...
    else:
        # Normal volatility - use standard schedule
        logger.info(f"Normal volatility (VIX: {vix_level}). Using standard schedule.")
        schedule.every().monday.at("09:30").do(run_daily_analysis)
        schedule.every().monday.at("16:00").do(run_daily_analysis)
        # Repeat for other weekdays...
    
    # Always do weekly summary on Friday
    schedule.every().friday.at("18:00").do(run_weekly_summary)
    
    return True

def schedule_live_tracker_jobs(mode="market_hours", interval_minutes=15, instruments=None):
    """Schedule jobs for the live instrument tracker"""
    if instruments is None:
        instruments = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ"]
    
    instruments_str = " ".join(instruments)
    
    if mode == "market_hours":
        # Schedule to run only during market hours
        cmd = f"python run_live_tracker.py --instruments {instruments_str} --interval {interval_minutes} --market-hours-only"
        
        # Schedule to start at market open (9:30 AM Eastern)
        schedule.every().monday.at("09:25").do(lambda: os.system(cmd))
        schedule.every().tuesday.at("09:25").do(lambda: os.system(cmd))
        schedule.every().wednesday.at("09:25").do(lambda: os.system(cmd))
        schedule.every().thursday.at("09:25").do(lambda: os.system(cmd))
        schedule.every().friday.at("09:25").do(lambda: os.system(cmd))
        
        logger.info(f"Scheduled live tracker to run during market hours with {interval_minutes}-minute intervals")
    
    elif mode == "continuous":
        # Schedule to run continuously
        cmd = f"python run_live_tracker.py --instruments {instruments_str} --interval {interval_minutes}"
        
        # Schedule to start once daily
        schedule.every().day.at("00:01").do(lambda: os.system(cmd))
        
        logger.info(f"Scheduled live tracker to run continuously with {interval_minutes}-minute intervals")
    
    elif mode == "custom":
        # Schedule to run at specific times
        cmd = f"python run_live_tracker.py --instruments {instruments_str} --interval {interval_minutes}"
        
        # Pre-market
        schedule.every().monday.at("08:00").do(lambda: os.system(cmd + " --market-hours-only"))
        schedule.every().tuesday.at("08:00").do(lambda: os.system(cmd + " --market-hours-only"))
        schedule.every().wednesday.at("08:00").do(lambda: os.system(cmd + " --market-hours-only"))
        schedule.every().thursday.at("08:00").do(lambda: os.system(cmd + " --market-hours-only"))
        schedule.every().friday.at("08:00").do(lambda: os.system(cmd + " --market-hours-only"))
        
        logger.info(f"Scheduled live tracker with custom schedule")
    
    return True

def manage_live_tracker_process():
    """Manage the live tracker process - check if it's running and restart if needed"""
    import psutil
    
    # Check if live tracker is running
    tracker_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and 'run_live_tracker.py' in ' '.join(proc.info['cmdline']):
                tracker_running = True
                logger.info(f"Live tracker is running (PID: {proc.info['pid']})")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not tracker_running:
        logger.warning("Live tracker is not running. Attempting to restart...")
        # Start with default settings
        os.system("python run_live_tracker.py --market-hours-only")
        return True
    
    return False

# Add a scheduled job to check on the live tracker process every hour
def schedule_process_monitor():
    """Schedule a job to monitor the live tracker process"""
    schedule.every(1).hours.do(manage_live_tracker_process)
    logger.info("Scheduled process monitor to check live tracker hourly")
    return True

def main():
    """Main entry point for the scheduler"""
    parser = argparse.ArgumentParser(description="Greek Energy Flow Analysis Scheduler")
    
    parser.add_argument("--mode", choices=["daily", "weekly", "market_close", "custom", "live_tracker"], default="custom",
                        help="Scheduling mode: daily, weekly, market_close, custom, or live_tracker")
    parser.add_argument("--morning-time", default="09:00",
                        help="Pre-market time to run the analysis (HH:MM format, 24-hour clock)")
    parser.add_argument("--midday-time", default="12:30",
                        help="Mid-day time to run the analysis (HH:MM format, 24-hour clock)")
    parser.add_argument("--afternoon-time", default="16:30",
                        help="Post-market time to run the analysis (HH:MM format, 24-hour clock)")
    parser.add_argument("--run-now", action="store_true",
                        help="Run the analysis immediately before starting the scheduler")
    parser.add_argument("--instruments", default="data/instruments.csv",
                        help="Path to the instruments CSV file")
    parser.add_argument("--output", default="results",
                        help="Directory to save output files")
    parser.add_argument("--timeframe", default="1d", 
                        help="Timeframe for data analysis (e.g., '1d', '165m')")
    parser.add_argument("--tracker-mode", choices=["market_hours", "continuous", "custom"], default="market_hours",
                        help="Mode for live tracker: market_hours, continuous, or custom")
    parser.add_argument("--tracker-interval", type=int, default=15,
                        help="Interval in minutes for live tracker updates")
    parser.add_argument("--tracker-instruments", nargs="+", 
                        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ"],
                        help="Instruments to track in live tracker mode")

    args = parser.parse_args()
    
    logger.info(f"Starting Greek Energy Flow Analysis Scheduler in {args.mode} mode")
    
    # Run immediately if requested
    if args.run_now:
        logger.info("Running analysis immediately")
        if args.mode == "live_tracker":
            # Start live tracker immediately
            instruments_str = " ".join(args.tracker_instruments)
            cmd = f"python run_live_tracker.py --instruments {instruments_str} --interval {args.tracker_interval}"
            if args.tracker_mode == "market_hours":
                cmd += " --market-hours-only"
            os.system(cmd)
        else:
            run_daily_analysis(args.instruments, args.output, args.timeframe)
    
    # Schedule jobs
    if args.mode == "live_tracker":
        if not schedule_live_tracker_jobs(args.tracker_mode, args.tracker_interval, args.tracker_instruments):
            logger.error("Failed to schedule live tracker jobs")
            return
    elif args.mode == "custom":
        if not schedule_jobs(args.mode, args.morning_time, args.midday_time, args.afternoon_time):
            logger.error("Failed to schedule jobs")
            return
    elif args.mode == "adaptive":
        if not schedule_adaptive_jobs():
            logger.error("Failed to schedule adaptive jobs")
            return
    else:
        if not schedule_jobs(args.mode):
            logger.error("Failed to schedule jobs")
            return
    
    # Schedule process monitor
    schedule_process_monitor()
    
    # Run the scheduler loop
    logger.info("Scheduler is running. Press Ctrl+C to exit.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

if __name__ == "__main__":
    main()




# To run this scheduler:
# 1. Install the schedule package: pip install schedule
# 2. Run this script in the background: python scheduler.py




