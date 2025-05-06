#!/usr/bin/env python
"""
Greek Energy Flow II - Full Analysis Sequence Script

This script runs all analysis modules in the correct sequence:
1. Data acquisition and processing for all tickers
2. ML model training using the analysis data
3. ML predictions and trade signal generation
4. Ordinal pattern analysis and trade recommendations

Just run this script, and it will handle the entire analysis pipeline.
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime

# Try to import ordinal pattern analyzer, but don't fail if it's not available
try:
    from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
    HAS_PATTERN_ANALYZER = True
except ImportError:
    HAS_PATTERN_ANALYZER = False
    print("Warning: GreekOrdinalPatternAnalyzer not found. Pattern analysis will use default implementation.")

# Configure logging with more detailed format
os.makedirs("logs", exist_ok=True)
log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"logs/full_analysis_{log_timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FullAnalysis")
logger.info(f"Starting full analysis sequence. Log file: {log_file}")

def run_command(command, description, timeout=600):  # 10-minute timeout by default
    """Run a command and log its output with improved error handling and timeout."""
    logger.info(f"Running {description}...")
    logger.debug(f"Command: {' '.join(command)}")
    
    try:
        # Set environment variable to use non-interactive backend for matplotlib
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Use non-interactive backend
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # Use a separate thread to read stdout to avoid blocking
        import threading
        import time
        
        def read_output(pipe, log_func):
            for line in iter(pipe.readline, ''):
                log_func(line.strip())
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, logger.info))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, logger.error))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete with timeout
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                process.terminate()
                logger.error(f"{description} timed out after {timeout} seconds")
                return False
            time.sleep(0.1)
        
        # Make sure threads have time to finish reading output
        stdout_thread.join(1)
        stderr_thread.join(1)
        
        if process.returncode == 0:
            logger.info(f"{description} completed successfully")
            return True
        else:
            logger.error(f"{description} failed with return code {process.returncode}")
            return False
            
    except FileNotFoundError:
        logger.error(f"{description} failed: Python interpreter or script not found")
        logger.error(f"Check if the script exists: {command[1]}")
        return False
    except Exception as e:
        logger.error(f"{description} failed with exception: {e}", exc_info=True)
        return False

def check_script_supports_arg(script_path, arg_name):
    """Check if a script supports a specific argument by examining its help output."""
    try:
        help_output = subprocess.check_output(
            ["python", script_path, "--help"], 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return arg_name in help_output
    except Exception as e:
        logger.warning(f"Could not check if {script_path} supports {arg_name}: {e}")
        return False

def main():
    """Run the full analysis sequence with improved error handling."""
    print(f"Starting Greek Energy Flow II Full Analysis... (Log: {log_file})")
    
    # Track success of each step
    step_status = {
        "data_acquisition": False,
        "ml_training": False,
        "ml_prediction": False,
        "pattern_analysis": False,
        "recommendations": False
    }
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run the full Greek Energy Flow II analysis sequence")
        parser.add_argument("--tickers", nargs="+", help="List of tickers to analyze")
        parser.add_argument("--tickers-file", default="my_tickers.txt", help="File containing tickers to analyze")
        parser.add_argument("--output-dir", default="results", help="Directory to store results")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--skip-ml", action="store_true", help="Skip ML training and prediction")
        parser.add_argument("--generate-recommendations", action="store_true", help="Generate trade recommendations")
        parser.add_argument("--use-patterns", action="store_true", default=True, help="Enable ordinal pattern analysis")
        parser.add_argument("--track-performance", action="store_true", default=True, help="Track performance of recommendations")
        
        args = parser.parse_args()
        
        # Set debug level if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Using output directory: {os.path.abspath(args.output_dir)}")
        
        os.makedirs("patterns", exist_ok=True)
        logger.info(f"Using patterns directory: {os.path.abspath('patterns')}")
        
        # Check if tickers file exists, create with default tickers if not
        if not args.tickers and not os.path.exists(args.tickers_file):
            logger.info(f"Tickers file {args.tickers_file} not found. Creating with default tickers.")
            default_tickers = ["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"]
            with open(args.tickers_file, 'w') as f:
                f.write('\n'.join(default_tickers))
            logger.info(f"Created {args.tickers_file} with {len(default_tickers)} default tickers.")
        
        # Prepare ticker arguments
        ticker_args = []
        if args.tickers:
            ticker_args = ["--tickers"] + args.tickers
            logger.info(f"Using command-line tickers: {', '.join(args.tickers)}")
        else:
            ticker_args = ["--tickers-file", args.tickers_file]
            logger.info(f"Using tickers from file: {args.tickers_file}")
            # Log the actual tickers being used
            try:
                with open(args.tickers_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(tickers)} tickers from file: {', '.join(tickers[:10])}" + 
                               (f"... and {len(tickers)-10} more" if len(tickers) > 10 else ""))
            except Exception as e:
                logger.warning(f"Could not read tickers from {args.tickers_file}: {e}")
        
        # Step 1: Run data acquisition and analysis
        logger.info("STEP 1: Running data acquisition and analysis")

        # Check which script to use for data acquisition
        data_acquisition_scripts = ["run_regime_analysis.py", "run_dashboard.py", "run_analysis.py"]
        data_script = None

        for script in data_acquisition_scripts:
            if os.path.exists(script):
                data_script = script
                logger.info(f"Found data acquisition script: {script}")
                break

        if data_script is None:
            logger.error("No data acquisition script found. Please check your installation.")
            return 1

        # Start with a very simple command - just one ticker
        simple_command = [
            "python", data_script,
            "--tickers", "SPY",  # Just one ticker to start
            "--output-dir", args.output_dir
        ]

        if args.debug:
            simple_command.append("--debug")

        logger.info(f"Running simplified analysis command: {' '.join(simple_command)}")
        step_status["data_acquisition"] = run_command(simple_command, "Data acquisition and analysis", timeout=300)  # 5-minute timeout

        # If that fails, try with a different script
        if not step_status["data_acquisition"] and data_script != "run_analysis.py" and os.path.exists("run_analysis.py"):
            logger.warning(f"{data_script} failed. Trying with run_analysis.py...")
            
            alt_command = [
                "python", "run_analysis.py",
                "--tickers", "SPY",
                "--output-dir", args.output_dir
            ]
            
            if args.debug:
                alt_command.append("--debug")
            
            logger.info(f"Running alternative analysis command: {' '.join(alt_command)}")
            step_status["data_acquisition"] = run_command(alt_command, "Alternative data acquisition", timeout=300)

        # If that works, try with a few more tickers
        if step_status["data_acquisition"]:
            logger.info("Single ticker analysis successful. Trying with more tickers...")
            
            # Try with a few more tickers
            if args.tickers:
                more_tickers = args.tickers[:5]  # Limit to 5 tickers
            else:
                try:
                    with open(args.tickers_file, 'r') as f:
                        more_tickers = [line.strip() for line in f if line.strip()][:5]  # First 5 tickers
                except Exception as e:
                    logger.error(f"Could not read tickers from {args.tickers_file}: {e}")
                    more_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]  # Default tickers
            
            more_command = [
                "python", data_script,
                "--tickers"] + more_tickers + [
                "--output-dir", args.output_dir
            ]
            
            if args.debug:
                more_command.append("--debug")
            
            logger.info(f"Running analysis with more tickers: {' '.join(more_command)}")
            run_command(more_command, "Data acquisition with multiple tickers", timeout=600)  # 10-minute timeout

        # Check if data acquisition was successful before proceeding
        if not step_status["data_acquisition"]:
            logger.error("Data acquisition and analysis failed. Cannot proceed with ML training.")
            logger.info("Checking for any existing analysis results to continue with...")
            
            # Check if we have any analysis results
            analysis_files = [f for f in os.listdir(args.output_dir) if f.endswith('_analysis_results.json')]
            if not analysis_files:
                logger.error("No analysis result files found. Cannot proceed with ML training.")
                logger.error("Please fix the data acquisition issues and try again.")
                return 1
            else:
                logger.warning(f"Found {len(analysis_files)} existing analysis files. Will attempt to continue with these.")
        
        # Step 2: Train ML models
        if not args.skip_ml:
            logger.info("STEP 2: Training ML models")
            
            # Check if run_ml_regime_analysis.py exists
            if not os.path.exists("run_ml_regime_analysis.py"):
                logger.error("run_ml_regime_analysis.py not found. Skipping ML steps.")
                args.skip_ml = True
            else:
                # Get help from the script to understand its arguments
                help_command = ["python", "run_ml_regime_analysis.py", "--help"]
                try:
                    help_output = subprocess.check_output(help_command, universal_newlines=True)
                    logger.debug(f"run_ml_regime_analysis.py help output:\n{help_output}")
                except Exception as e:
                    logger.warning(f"Could not get help for run_ml_regime_analysis.py: {e}")
                
                # Build the command with correct arguments
                ml_train_command = [
                    "python", "run_ml_regime_analysis.py",
                    "--train",
                    "--output-dir", args.output_dir  # Use --output-dir instead of --data-dir
                ]
                
                if args.debug:
                    ml_train_command.append("--debug")
                    
                logger.info(f"Running ML training command: {' '.join(ml_train_command)}")
                step_status["ml_training"] = run_command(ml_train_command, "ML model training")
                
                # Check if ML training was successful before proceeding to prediction
                if not step_status["ml_training"]:
                    logger.warning("ML model training failed. Prediction step may not work correctly.")
        else:
            logger.info("ML training skipped as requested.")

        # Step 3: Generate ML predictions (skip if requested)
        if not args.skip_ml:
            logger.info("STEP 3: Generating ML predictions")
            
            # Extract just the tickers without the --tickers or --tickers-file part
            ticker_list = []
            if args.tickers:
                ticker_list = args.tickers
            else:
                # Read tickers from file
                try:
                    with open(args.tickers_file, 'r') as f:
                        ticker_list = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    logger.warning(f"Could not read tickers from {args.tickers_file}: {e}")
            
            # Use the correct format for run_ml_regime_analysis.py
            ml_predict_command = [
                "python", "run_ml_regime_analysis.py",
                "--predict"
            ]
            
            # Add tickers if we have them
            if ticker_list:
                ml_predict_command.extend(["--tickers"] + ticker_list)
            
            # Add output directory
            ml_predict_command.extend(["--output-dir", args.output_dir])
            
            if args.debug:
                ml_predict_command.append("--debug")
                
            if not run_command(ml_predict_command, "ML prediction generation"):
                logger.warning("ML prediction generation failed. Continuing without ML predictions.")
        else:
            logger.info("STEP 3: Skipping ML predictions as requested")

        # Step 3.5: Run market regime analysis
        logger.info("STEP 3.5: Running market regime analysis")
        regime_command = [
            "python", "run_regime_analysis.py",  # Use the script that actually exists
        ]

        # Add tickers directly
        if args.tickers:
            regime_command.extend(["--tickers"] + args.tickers)
        else:
            # Read tickers from file and pass them directly
            try:
                with open(args.tickers_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                    if tickers:
                        regime_command.extend(["--tickers"] + tickers)
                    else:
                        logger.warning(f"No tickers found in {args.tickers_file}")
            except Exception as e:
                logger.warning(f"Could not read tickers from {args.tickers_file}: {e}")

        # Add output directory
        regime_command.extend(["--output-dir", args.output_dir])

        if args.debug:
            regime_command.append("--debug")
            
        if not run_command(regime_command, "Market regime analysis"):
            logger.warning("Market regime analysis failed. Continuing without regime analysis.")

        # Step 3.6: Update ordinal pattern library
        if args.use_patterns and HAS_PATTERN_ANALYZER:
            logger.info("STEP 3.6: Updating ordinal pattern library")
            
            # Check if we have any analysis results before trying to update patterns
            results_files = [f for f in os.listdir(args.output_dir) if f.endswith('_analysis_results.json')]
            if not results_files:
                logger.warning("No analysis result files found. Pattern library update may fail.")
                logger.info(f"Contents of {args.output_dir}: {os.listdir(args.output_dir)[:10]}")
            else:
                logger.info(f"Found {len(results_files)} analysis result files for pattern extraction")
            
            pattern_command = [
                "python", "update_pattern_library.py",
                "--results-dir", args.output_dir,
                "--pattern-dir", "patterns"
            ]
            
            # Add --skip-pipeline-integration flag if it's supported
            if check_script_supports_arg("update_pattern_library.py", "--skip-pipeline-integration"):
                pattern_command.append("--skip-pipeline-integration")
            
            if args.debug:
                pattern_command.append("--debug")
                
            if run_command(pattern_command, "Ordinal pattern library update", timeout=300):
                step_status["pattern_analysis"] = True
            else:
                logger.warning("Pattern library update failed. Continuing...")
        else:
            if not HAS_PATTERN_ANALYZER:
                logger.warning("GreekOrdinalPatternAnalyzer not available. Skipping pattern library update.")
            else:
                logger.info("Pattern analysis disabled. Skipping pattern library update.")

        # Step 4: Generate trade recommendations
        if args.generate_recommendations:
            logger.info("STEP 4: Generating trade recommendations")
            
            # Generate recommendations using the trade_recommendations.py script
            rec_command = [
                "python", "analysis/trade_recommendations.py",  # Use the script that actually exists
                "--results-dir", args.output_dir
            ]
            
            if args.use_patterns:
                rec_command.append("--use-patterns")
            
            if args.debug:
                rec_command.append("--debug")
                
            if not run_command(rec_command, "Trade recommendation generation"):
                logger.warning("Trade recommendation generation failed. Continuing...")

        # Print a summary of what was done
        logger.info("=" * 50)
        logger.info("ANALYSIS SEQUENCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
        
        # Count result files
        analysis_files = [f for f in os.listdir(args.output_dir) if f.endswith('_analysis_results.json')]
        ml_files = [f for f in os.listdir(args.output_dir) if f.endswith('_ml_predictions.json')]
        recommendation_files = [f for f in os.listdir(args.output_dir) if f.endswith('_recommendation.json')]
        
        logger.info(f"Analysis results: {len(analysis_files)} files")
        logger.info(f"ML predictions: {len(ml_files)} files")
        logger.info(f"Trade recommendations: {len(recommendation_files)} files")
        
        # Check pattern library
        pattern_files = [f for f in os.listdir("patterns") if f.endswith('_patterns.json')]
        logger.info(f"Pattern libraries: {len(pattern_files)} files")
        
        # Report on step status
        logger.info("Step completion status:")
        for step, status in step_status.items():
            logger.info(f"  {step.replace('_', ' ').title()}: {'Completed' if status else 'Failed'}")
        
        # Check if any steps succeeded
        if not any(step_status.values()):
            logger.error("All steps failed. Please check the logs for details.")
            return 1
        
        logger.info("Full analysis sequence completed")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception in main function: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
