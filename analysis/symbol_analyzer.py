# analysis/symbol_analyzer.py
import os
import logging
import concurrent.futures
from datetime import date, datetime
from typing import List, Dict, Any, Optional

# Import utilities
from utils.helpers import ensure_directory

# Import data components
from data.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class SymbolAnalyzer:
    """
    Main orchestrator for analyzing stock symbols.
    
    This class coordinates the different analysis components:
    - Data fetching
    - Greek analysis
    - Momentum analysis
    - Pattern recognition
    - Risk analysis
    - Visualization
    """
    
    def __init__(self, cache_dir="cache", output_dir="output", use_parallel=True, max_workers=5):
        """
        Initialize SymbolAnalyzer.
        
        Args:
            cache_dir (str): Directory for caching data
            output_dir (str): Directory for output files
            use_parallel (bool): Whether to use parallel processing
            max_workers (int): Maximum number of parallel workers
        """
        self.cache_dir = ensure_directory(cache_dir)
        self.output_dir = ensure_directory(output_dir)
        self.charts_dir = ensure_directory(os.path.join(output_dir, "charts"))
        self.reports_dir = ensure_directory(os.path.join(output_dir, "reports"))
        
        self.cache_manager = CacheManager(cache_dir)
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
        # Import config lazily to avoid circular imports
        try:
            import config
            self.config = config
            logger.info("Loaded configuration")
        except ImportError:
            logger.warning("Could not import config.py. Using default configuration.")
            self.config = type('Config', (), {})  # Empty config object
            
        # Set default API key
        self.api_key = getattr(self.config, "POLYGON_API_KEY", None)
        if self.api_key is None or "YOUR_API_KEY" in str(self.api_key):
            logger.warning("No valid API key found in config.py")
            
        # Initialize component instances lazily when needed
    
    def run_analysis(self, symbols: List[str], analysis_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run analysis on a list of symbols.
        
        Args:
            symbols (list): List of stock symbols to analyze
            analysis_date (str, optional): Date for analysis (YYYY-MM-DD). Defaults to today.
            
        Returns:
            list: Analysis results for each symbol
        """
        # Validate input
        if not symbols:
            logger.warning("No symbols provided for analysis")
            return []
            
        # Parse date or use today
        if analysis_date:
            try:
                analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Invalid date format: {analysis_date}. Using today's date.")
                analysis_date = date.today()
        else:
            analysis_date = date.today()
        
        logger.info(f"Running analysis for {len(symbols)} symbols as of {analysis_date}")
        
        # Process symbols (parallel or sequential)
        if self.use_parallel and len(symbols) > 1:
            results = self._process_symbols_parallel(symbols, analysis_date)
        else:
            results = self._process_symbols_sequential(symbols, analysis_date)
            
        # Save summary report
        self._save_summary_report(results, analysis_date)
        
        return results
    
    def _process_symbols_parallel(self, symbols: List[str], analysis_date: date) -> List[Dict[str, Any]]:
        """Process multiple symbols in parallel."""
        results = []
        
        logger.info(f"Processing {len(symbols)} symbols in parallel with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.analyze_symbol, symbol, analysis_date): symbol 
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {symbol} in parallel execution: {e}")
                    # Add error record
                    results.append({
                        "symbol": symbol,
                        "date": analysis_date,
                        "error": str(e),
                        "success": False
                    })
                    
        return results
    
    def _process_symbols_sequential(self, symbols: List[str], analysis_date: date) -> List[Dict[str, Any]]:
        """Process symbols sequentially."""
        results = []
        
        logger.info(f"Processing {len(symbols)} symbols sequentially")
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol, analysis_date)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                # Add error record
                results.append({
                    "symbol": symbol,
                    "date": analysis_date,
                    "error": str(e),
                    "success": False
                })
                
        return results
    
    def analyze_symbol(self, symbol: str, analysis_date: date) -> Dict[str, Any]:
        """
        Analyze a single symbol.
        
        Args:
            symbol (str): Stock symbol
            analysis_date (date): Analysis date
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"▶ Processing {symbol}")
        
        # Check cache first
        cached_results = self.cache_manager.get_cached_data(symbol, "analysis")
        if cached_results:
            logger.info(f"{symbol}: Using cached analysis results from {cached_results.get('date', 'unknown date')}")
            self._print_cached_report(symbol, cached_results)
            return cached_results
            
        # Import other components only when needed
        # This avoids circular imports and improves startup time
        try:
            # Import API fetcher
            from api_fetcher import (
                fetch_options_chain_snapshot,
                fetch_underlying_snapshot,
                get_spot_price_from_snapshot,
                preprocess_api_options_data
            )
            
            # Step 1: Fetch options data
            opts_raw = self.cache_manager.get_cached_data(symbol, "options_raw")
            if not opts_raw:
                logger.info(f"{symbol}: Fetching fresh options data")
                opts_raw = fetch_options_chain_snapshot(symbol, self.api_key)
                if opts_raw:
                    self.cache_manager.save_to_cache(opts_raw, symbol, "options_raw")
            
            if opts_raw is None or not opts_raw:
                logger.error(f"{symbol}: Options fetch failed")
                return {"symbol": symbol, "error": "Options fetch failed", "success": False}
                
            # Preprocess options data
            opt_df = preprocess_api_options_data(opts_raw, analysis_date)
            if opt_df.empty:
                logger.error(f"{symbol}: No valid options rows")
                return {"symbol": symbol, "error": "No valid options rows", "success": False}
                
            # Step 2: Fetch underlying data
            under = self.cache_manager.get_cached_data(symbol, "underlying_snapshot")
            if not under:
                logger.info(f"{symbol}: Fetching fresh underlying data")
                under = fetch_underlying_snapshot(symbol, self.api_key)
                if under:
                    self.cache_manager.save_to_cache(under, symbol, "underlying_snapshot")
            
            if under is None:
                logger.error(f"{symbol}: Underlying fetch failed")
                return {"symbol": symbol, "error": "Underlying fetch failed", "success": False}
                
            # Extract spot price
            spot = get_spot_price_from_snapshot(under)
            if spot is None:
                logger.error(f"{symbol}: Spot price extraction failed")
                return {"symbol": symbol, "error": "Spot price extraction failed", "success": False}
                
            # Step 3: Set up market context
            # Import historical IV functions
            from api_fetcher import find_latest_overview_file, load_historical_iv_from_file
            
            # Compute fallback IV
            data_dir = getattr(self.config, "DATA_DIR", "data")
            ov_file = find_latest_overview_file(symbol, data_dir)
            fb_iv = None
            if ov_file:
                hist_iv = load_historical_iv_from_file(ov_file)
                if not hist_iv.empty:
                    fb_iv = hist_iv['Imp Vol'].mean()
            
            # Create market data dictionary
            import numpy as np
            fallback_iv_used = False
            
            if fb_iv is None:
                fb_iv = 0.3
                fallback_iv_used = True
                
            market = {
                "currentPrice": spot,
                "riskFreeRate": getattr(self.config, "RISK_FREE_RATE", 0.04),
                "historicalVolatility": fb_iv,
                "impliedVolatility": fb_iv,
                "analysis_date": analysis_date
            }
            
            if fallback_iv_used:
                logger.warning(f"{symbol}: FALLBACK IMPLIED VOLATILITY USED (0.3) - results may not reflect actual market conditions")
            
            # Step 4: Run Greek analysis
            greek_res = self._process_greeks(opt_df, market, symbol)
            if not greek_res:
                logger.warning(f"{symbol}: Greek analysis did not return results")
                
            # Step 5: Generate charts
            from analyzer_visualizations.chart_generator import ChartGenerator
            try:
                chart_gen = ChartGenerator(output_dir=self.charts_dir)
                chart_gen.plot_reset_points(symbol, greek_res, analysis_date)
                chart_gen.plot_energy_levels(symbol, greek_res, analysis_date)
                chart_gen.plot_price_projections(symbol, greek_res, analysis_date)
                logger.info(f"{symbol}: Charts generated")
            except Exception as e:
                logger.warning(f"{symbol}: Chart generation failed: {e}")
                
            # Step 6: Run momentum analysis
            from momentum_analyzer import EnergyFlowAnalyzer
            from data_loader import load_intraday_data
            
            momentum_data = {}
            try:
                # Try cache first
                price_df = self.cache_manager.get_cached_data(symbol, "price_data")
                if price_df is None or price_df.empty:
                    logger.info(f"{symbol}: Fetching fresh price data")
                    price_df, _ = load_intraday_data(symbol, self.config)
                    if price_df is not None and not price_df.empty:
                        self.cache_manager.save_to_cache(price_df, symbol, "price_data")
                
                energy_dir = "N/A"
                energy_grad = np.nan
                energy_state = "N/A"
                
                if price_df is not None and not price_df.empty:
                    # Prepare DataFrame for momentum analysis
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Try to map columns if needed
                    rename_dict = {
                        'DateTime': 'timestamp',
                        'Date': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }
                    rename_cols = {k: v for k, v in rename_dict.items() if k in price_df.columns}
                    if rename_cols:
                        price_df = price_df.rename(columns=rename_cols)
                    
                    # Check if we have the required columns
                    missing = [c for c in required_cols if c not in price_df.columns]
                    if not missing:
                        try:
                            required_df = price_df[required_cols].copy()
                            required_df = required_df.dropna(subset=['timestamp', 'close']).reset_index(drop=True)
                            
                            if len(required_df) > 5:  # Need minimum data points
                                ef = EnergyFlowAnalyzer(required_df, symbol=symbol)
                                ef.calculate_energy_metrics()
                                energy_dir, energy_state = ef.get_current_momentum_state()
                                # Extract numeric gradient
                                energy_grad = ef.energy_gradients[-1] if hasattr(ef, 'energy_gradients') and len(ef.energy_gradients) > 0 else np.nan
                                logger.info(f"{symbol}: Momentum → {energy_dir}, {energy_state}, Gradient: {energy_grad}")
                            else:
                                logger.warning(f"{symbol}: Not enough price data points after cleaning ({len(required_df)})")
                                energy_dir = "Insufficient Data"
                        except Exception as e:
                            logger.warning(f"{symbol}: Momentum analyzer failed: {e}")
                            energy_dir = "ERROR"
                    else:
                        logger.warning(f"{symbol}: Missing required columns for momentum: {missing}")
                        energy_dir = "Missing Data"
                else:
                    logger.warning(f"{symbol}: No price data available – skipping momentum")
                    energy_dir = "No Data"
            except Exception as e:
                logger.error(f"{symbol}: Error in momentum calculation: {e}")
                energy_dir = "FAILED"
                energy_grad = np.nan
            
            # Create momentum data dictionary
            momentum_data = {
                "energy_direction": energy_dir,
                "energy_state": energy_state if 'energy_state' in locals() else "N/A",
                "energy_gradient": energy_grad if 'energy_grad' in locals() else np.nan
            }
            
            # Step 7: Run pattern recognition
            from analysis.pattern_analyzer import PatternRecognizer
            try:
                pattern_rec = PatternRecognizer()
                pattern_data = pattern_rec.predict_pattern(greek_res, momentum_data)
                logger.info(f"{symbol}: Pattern detected: {pattern_data['pattern']} ({pattern_data['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"{symbol}: Pattern recognition failed: {e}")
                pattern_data = {
                    "pattern": "Unknown",
                    "confidence": 0,
                    "description": "Pattern recognition failed"
                }
                
            # Step 8: Calculate risk metrics
            from analysis.risk_analyzer import RiskAnalyzer
            try:
                risk_analyzer = RiskAnalyzer()
                risk_metrics = risk_analyzer.calculate_risk_metrics(greek_res, momentum_data, spot, symbol)
                logger.info(f"{symbol}: Risk metrics calculated")
            except Exception as e:
                logger.warning(f"{symbol}: Risk metrics calculation failed: {e}")
                risk_metrics = {
                    "risk_reward_ratio": "Error",
                    "position_size": "N/A",
                    "stop_loss": "N/A",
                    "take_profit": "N/A",
                    "volatility_risk": "Unknown"
                }
                
            # Step 9: Generate report
            from analyzer_visualizations.formatters import GreekFormatter, format_levels
            try:
                # Format results
                greek_formatter = GreekFormatter()
                formatted_results = greek_formatter.format_results(greek_res, spot)
                
                # Extract levels for the report
                levels = greek_res.get("energy_levels", [])
                sup_str = format_levels(levels, "support")
                res_str = format_levels(levels, "resistance")
                
                # Full report should be in formatted_results
                full_report = formatted_results.get('full_report', "Report generation failed")
                
                # Save report to file
                report_file = os.path.join(self.reports_dir, f"{symbol}_report_{analysis_date}.txt")
                with open(report_file, 'w') as f:
                    f.write(full_report)
                    f.write("\n\n")
                    f.write(f"Momentum: {energy_dir} | Gradient: {energy_grad}\n")
                    f.write(f"Pattern: {pattern_data['pattern']} | Confidence: {pattern_data['confidence']:.2f}\n")
                    f.write(f"Description: {pattern_data['description']}\n\n")
                    f.write("RISK METRICS:\n")
                    for k, v in risk_metrics.items():
                        f.write(f"{k}: {v}\n")
                
                logger.info(f"{symbol}: Report saved to {report_file}")
                
                # Print report to console
                self._print_report(symbol, analysis_date, full_report, momentum_data, pattern_data)
                
            except Exception as e:
                logger.error(f"{symbol}: Report generation failed: {e}")
                full_report = f"Report generation failed: {e}"
                
            # Step 10: Save snapshot
            from utils.io_logger import append_daily_snapshot
            try:
                # Create combined results
                combined_results = {
                    "symbol": symbol,
                    "spot_price": spot,
                    "date": analysis_date,
                    "greek_results": greek_res,
                    "energy_direction": energy_dir,
                    "energy_gradient": energy_grad,
                    "pattern": pattern_data,
                    "risk_metrics": risk_metrics,
                    "full_report": full_report,
                    "success": True
                }
                
                # Save snapshot
                append_daily_snapshot(greek_res, spot, analysis_date)
                logger.info(f"{symbol}: Daily snapshot saved")
                
                # Cache the results
                self.cache_manager.save_to_cache(combined_results, symbol, "analysis")
                
                return combined_results
                
            except Exception as e:
                logger.error(f"{symbol}: Failed to save snapshot: {e}")
                return {
                    "symbol": symbol,
                    "error": f"Failed to save snapshot: {e}",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"{symbol}: Analysis failed: {e}")
            return {
                "symbol": symbol,
                "error": f"Analysis failed: {e}",
                "success": False
            }
    
    def _print_report(self, symbol, analysis_date, report, momentum_data, pattern_data):
        """Print report to console."""
        print("\n" + "="*12 + f" {symbol} {analysis_date} " + "="*12)
        print(report)
        print(f"Momentum → {momentum_data.get('energy_direction', 'N/A')} | gradient: {momentum_data.get('energy_gradient', 'N/A')}")
        print(f"Pattern → {pattern_data['pattern']} | confidence: {pattern_data['confidence']:.2f}")
    
    def _print_cached_report(self, symbol, cached_results):
        """Print cached report to console."""
        print("\n" + "="*12 + f" {symbol} {cached_results.get('date', 'unknown date')} (CACHED) " + "="*12)
        print(cached_results.get("full_report", "No report available"))
        print(f"Momentum → {cached_results.get('energy_direction', 'N/A')} | gradient: {cached_results.get('energy_gradient', 'N/A')}")
        print(f"Pattern → {cached_results.get('pattern', {}).get('pattern', 'N/A')} | confidence: {cached_results.get('pattern', {}).get('confidence', 0):.2f}")
    
    def _save_summary_report(self, results, analysis_date):
        """Save summary report of all analyzed symbols."""
        try:
            summary_file = os.path.join(self.reports_dir, f"summary_{analysis_date}.txt")
            with open(summary_file, 'w') as f:
                f.write(f"SUMMARY TABLE ({analysis_date})\n")
                f.write("="*80 + "\n")
                f.write(f"{'Symbol':<6} {'Price':>8} {'Direction':<12} {'Pattern':<20} {'Support':<16} {'Resistance':<16}\n")
                f.write("-"*80 + "\n")
                
                for r in results:
                    if not r.get('success', False):
                        f.write(f"{r.get('symbol', ''):<6} {'ERROR':>8} {'ERROR':<12} {'ERROR':<20} {'ERROR':<16} {'ERROR':<16}\n")
                        continue
                        
                    # Extract data
                    symbol = r.get('symbol', '')
                    price = r.get('spot_price', 0)
                    direction = r.get('energy_direction', 'N/A')
                    
                    # Get pattern info
                    pattern_data = r.get('pattern', {})
                    pattern = f"{pattern_data.get('pattern', 'N/A')} ({pattern_data.get('confidence', 0):.2f})"
                    if len(pattern) > 19:
                        pattern = pattern[:16] + "..."
                        
                    # Get support/resistance
                    support = r.get('support_levels', 'N/A')
                    if len(str(support)) > 15:
                        support = str(support)[:12] + "..."
                        
                    resistance = r.get('resistance_levels', 'N/A')
                    if len(str(resistance)) > 15:
                        resistance = str(resistance)[:12] + "..."
                        
                    f.write(f"{symbol:<6} {price:>8.2f} {direction:<12} {pattern:<20} {support:<16} {resistance:<16}\n")
                    
            logger.info(f"Summary report saved to {summary_file}")
            return summary_file
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")
            return None
    
    def display_summary(self, results):
        """Display summary of results to console."""
        if not results:
            print("\nNo results to display")
            return
            
        print("\n" + "="*80)
        print(f"SUMMARY TABLE ({date.today()})")
        print("="*80)
        print(f"{'Symbol':<6} {'Price':>8} {'Direction':<12} {'Pattern':<20} {'Support':<16} {'Resistance':<16}")
        print("-"*80)
        
        for r in results:
            if not r.get('success', False):
                print(f"{r.get('symbol', ''):<6} {'ERROR':>8} {'ERROR':<12} {'ERROR':<20} {'ERROR':<16} {'ERROR':<16}")
                continue
                
            # Extract data
            symbol = r.get('symbol', '')
            price = r.get('spot_price', 0)
            direction = r.get('energy_direction', 'N/A')
            
            # Get pattern info
            pattern_data = r.get('pattern', {})
            pattern = f"{pattern_data.get('pattern', 'N/A')} ({pattern_data.get('confidence', 0):.2f})"
            if len(pattern) > 19:
                pattern = pattern[:16] + "..."
                
            # Get support/resistance
            support = r.get('support_levels', 'N/A')
            if len(str(support)) > 15:
                support = str(support)[:12] + "..."
                
            resistance = r.get('resistance_levels', 'N/A')
            if len(str(resistance)) > 15:
                resistance = str(resistance)[:12] + "..."
                
            print(f"{symbol:<6} {price:>8.2f} {direction:<12} {pattern:<20} {support:<16} {resistance:<16}")
    
    def _process_greeks(self, options_df, market_data, symbol):
        """
        Process options data to calculate Greek values and analyze energy flow.
        
        Args:
            options_df (DataFrame): Options chain data
            market_data (dict): Market context data
            symbol (str): Stock symbol
        
        Returns:
            dict: Greek analysis results
        """
        import pandas as pd
        try:
            # Ensure expiration is datetime
            if 'expiration' in options_df.columns:
                options_df['expiration'] = pd.to_datetime(options_df['expiration'])
            
            # Import Greek analysis components
            from greek_flow.flow import GreekEnergyAnalyzer
            
            # Run analysis
            greek_results = GreekEnergyAnalyzer.analyze(options_df, market_data)
            
            # If the above fails, try using GreekEnergyFlow directly
            if not greek_results:
                logger.info(f"{symbol}: Falling back to GreekEnergyFlow")
                from greek_flow import GreekEnergyFlow
                
                # Format market data for GreekEnergyFlow
                flow_market_data = {
                    'currentPrice': market_data.get('currentPrice'),
                    'historicalVolatility': market_data.get('historicalVolatility', market_data.get('impliedVolatility')),
                    'riskFreeRate': market_data.get('riskFreeRate'),
                    'date': market_data.get('analysis_date').strftime('%Y-%m-%d') if isinstance(market_data.get('analysis_date'), date) else market_data.get('analysis_date')
                }
                
                # Create flow analyzer and run analysis
                flow = GreekEnergyFlow()
                greek_results = flow.analyze_greek_profiles(options_df, flow_market_data)
            
            # Add additional analysis if needed
            greek_results["symbol"] = symbol
            greek_results["options_count"] = len(options_df)
            
            logger.info(f"{symbol}: Greek analysis completed with {len(options_df)} options")
            return greek_results
        except Exception as e:
            logger.error(f"{symbol}: Error in Greek analysis: {e}")
            return None
