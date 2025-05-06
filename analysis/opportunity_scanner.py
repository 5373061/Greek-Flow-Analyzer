import os
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from analysis.pipeline_manager import AnalysisPipeline

logger = logging.getLogger(__name__)

class OpportunityScanner:
    """
    Scans multiple symbols to identify and rank trading opportunities.
    
    This class runs the analysis pipeline across multiple symbols and
    ranks the results to identify the best trading opportunities.
    """
    
    def __init__(self, config=None, max_workers=4):
        """
        Initialize the opportunity scanner.
        
        Args:
            config (dict): Configuration dictionary
            max_workers (int): Maximum number of parallel workers
        """
        self.config = config or {}
        self.max_workers = max_workers
        self.pipeline = AnalysisPipeline(config)
        self.results = {}
        
    def scan_symbols(self, symbols, date=None, use_cached=True):
        """
        Scan multiple symbols for trading opportunities.
        
        Args:
            symbols (list): List of symbols to scan
            date (datetime): Date to analyze (None for latest)
            use_cached (bool): Whether to use cached results
            
        Returns:
            DataFrame: Ranked opportunities
        """
        logger.info(f"Scanning {len(symbols)} symbols for opportunities")
        
        # Run analysis on all symbols
        results = self._analyze_symbols(symbols, date, use_cached)
        
        # Rank opportunities
        ranked_opportunities = self._rank_opportunities(results)
        
        # Store results
        self.results = {
            "scan_date": date or datetime.now(),
            "symbols_analyzed": len(symbols),
            "opportunities": ranked_opportunities
        }
        
        return ranked_opportunities
    
    def _analyze_symbols(self, symbols, date=None, use_cached=True):
        """Run analysis on multiple symbols in parallel."""
        results = []
        
        # Use parallel processing if multiple symbols
        if len(symbols) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all analysis tasks
                future_to_symbol = {
                    executor.submit(self._analyze_single_symbol, symbol, date, use_cached): symbol 
                    for symbol in symbols
                }
                
                # Process results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
        else:
            # Sequential processing
            for symbol in symbols:
                try:
                    result = self._analyze_single_symbol(symbol, date, use_cached)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
        
        logger.info(f"Completed analysis of {len(results)} symbols")
        return results
    
    def _analyze_single_symbol(self, symbol, date=None, use_cached=True):
        """Analyze a single symbol using the pipeline."""
        try:
            # Run the full analysis pipeline
            result = self.pipeline.run(symbol, date, use_cached)
            
            # Add symbol to result
            result["symbol"] = symbol
            
            return result
        except Exception as e:
            logger.error(f"Error in analysis of {symbol}: {e}")
            return None
    
    def _rank_opportunities(self, results):
        """
        Rank trading opportunities based on multiple factors.
        
        This method scores each opportunity based on:
        1. Risk/reward ratio
        2. Energy concentration (entropy)
        3. Market regime alignment
        4. Anomaly presence
        5. Technical pattern strength
        """
        if not results:
            return pd.DataFrame()
        
        # Extract key metrics for ranking
        opportunities = []
        
        for result in results:
            if not result:
                continue
                
            symbol = result.get("symbol")
            
            # Extract risk metrics
            risk_mgmt = result.get("risk_management", {})
            risk_reward = self._parse_float(risk_mgmt.get("risk_reward_ratio", "0"))
            
            # Extract entropy metrics
            entropy_data = result.get("entropy_analysis", {})
            energy_state = entropy_data.get("energy_state", {})
            avg_entropy = energy_state.get("average_normalized_entropy", 50)
            entropy_score = 100 - avg_entropy  # Lower entropy = higher score
            
            # Extract regime data
            greek_data = result.get("greek_analysis", {})
            regime = greek_data.get("market_regime", {})
            regime_strength = regime.get("regime_strength", 0.5)
            
            # Extract anomaly data
            anomalies = entropy_data.get("anomalies", {})
            anomaly_count = anomalies.get("anomaly_count", 0)
            
            # Calculate opportunity score (0-100)
            # Weight factors according to importance
            score_components = {
                "risk_reward": min(risk_reward * 10, 40),  # 40% weight, max 40 points
                "entropy": entropy_score * 0.3,            # 30% weight, max 30 points
                "regime": regime_strength * 20,            # 20% weight, max 20 points
                "anomaly": 10 - (anomaly_count * 2)        # 10% weight, max 10 points
            }
            
            # Ensure anomaly score doesn't go negative
            score_components["anomaly"] = max(0, score_components["anomaly"])
            
            # Calculate total score
            total_score = sum(score_components.values())
            
            # Determine grade based on score
            grade = "A+" if total_score >= 90 else \
                   "A" if total_score >= 80 else \
                   "B+" if total_score >= 75 else \
                   "B" if total_score >= 70 else \
                   "C+" if total_score >= 65 else \
                   "C" if total_score >= 60 else \
                   "D" if total_score >= 50 else "F"
            
            # Extract directional bias
            direction = "Bullish" if regime.get("primary_label", "").lower() == "bullish" else \
                       "Bearish" if regime.get("primary_label", "").lower() == "bearish" else "Neutral"
            
            # Create opportunity record
            opportunity = {
                "symbol": symbol,
                "score": round(total_score, 1),
                "grade": grade,
                "direction": direction,
                "risk_reward": risk_reward,
                "stop_loss": risk_mgmt.get("stop_loss", "N/A"),
                "take_profit": risk_mgmt.get("take_profit", "N/A"),
                "position_size": risk_mgmt.get("position_size", "N/A"),
                "entropy_state": energy_state.get("state", "Unknown"),
                "regime": regime.get("primary_label", "Unknown"),
                "volatility": regime.get("volatility_regime", "Normal"),
                "anomalies": anomaly_count,
                "score_components": score_components
            }
            
            opportunities.append(opportunity)
        
        # Convert to DataFrame
        df = pd.DataFrame(opportunities)
        
        # Sort by score (descending)
        if not df.empty and "score" in df.columns:
            df = df.sort_values("score", ascending=False)
        
        return df
    
    def _parse_float(self, value, default=0.0):
        """Safely parse a float value."""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                # Remove any non-numeric characters except decimal point
                clean_value = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(clean_value)
            except (ValueError, TypeError):
                return default
        
        return default
    
    def get_top_opportunities(self, min_grade="B", direction=None):
        """
        Get top opportunities filtered by grade and direction.
        
        Args:
            min_grade (str): Minimum grade to include (A+, A, B+, B, C+, C, D, F)
            direction (str): Filter by direction (Bullish, Bearish, None for both)
            
        Returns:
            DataFrame: Filtered opportunities
        """
        if not isinstance(self.results.get("opportunities"), pd.DataFrame):
            return pd.DataFrame()
            
        df = self.results["opportunities"]
        
        # Grade ranking
        grade_rank = {"A+": 7, "A": 6, "B+": 5, "B": 4, "C+": 3, "C": 2, "D": 1, "F": 0}
        min_grade_rank = grade_rank.get(min_grade, 0)
        
        # Filter by grade
        mask = df["grade"].apply(lambda g: grade_rank.get(g, 0) >= min_grade_rank)
        
        # Filter by direction if specified
        if direction:
            mask &= (df["direction"] == direction)
            
        return df[mask]
    
    def generate_report(self, output_format="text"):
        """
        Generate a report of the scan results.
        
        Args:
            output_format (str): Format of the report (text, html, json)
            
        Returns:
            str: Formatted report
        """
        if not isinstance(self.results.get("opportunities"), pd.DataFrame) or self.results["opportunities"].empty:
            return "No opportunities found."
            
        df = self.results["opportunities"]
        
        if output_format == "html":
            # Generate HTML report with bootstrap styling
            html = f"""
            <html>
            <head>
                <title>Trading Opportunities Report</title>
                <style>
                    .A\\+ {{ background-color: #28a745; color: white; }}
                    .A {{ background-color: #5cb85c; color: white; }}
                    .B\\+ {{ background-color: #5bc0de; color: white; }}
                    .B {{ background-color: #63c2de; color: white; }}
                    .C\\+ {{ background-color: #ffc107; color: black; }}
                    .C {{ background-color: #f0ad4e; color: black; }}
                    .D {{ background-color: #d9534f; color: white; }}
                    .F {{ background-color: #dc3545; color: white; }}
                    .Bullish {{ color: green; }}
                    .Bearish {{ color: red; }}
                    .Neutral {{ color: gray; }}
                </style>
            </head>
            <body>
                <h1>Trading Opportunities Report</h1>
                <p>Scan Date: {self.results.get('scan_date')}</p>
                <p>Symbols Analyzed: {self.results.get('symbols_analyzed')}</p>
                <table border="1" cellpadding="5">
                    <tr>
                        <th>Symbol</th>
                        <th>Grade</th>
                        <th>Score</th>
                        <th>Direction</th>
                        <th>Risk/Reward</th>
                        <th>Stop Loss</th>
                        <th>Take Profit</th>
                        <th>Entropy State</th>
                        <th>Regime</th>
                    </tr>
            """
            
            for _, row in df.iterrows():
                html += f"""
                    <tr>
                        <td><b>{row['symbol']}</b></td>
                        <td class="{row['grade']}">{row['grade']}</td>
                        <td>{row['score']}</td>
                        <td class="{row['direction']}">{row['direction']}</td>
                        <td>{row['risk_reward']}</td>
                        <td>{row['stop_loss']}</td>
                        <td>{row['take_profit']}</td>
                        <td>{row['entropy_state']}</td>
                        <td>{row['regime']}</td>
                    </tr>
                """
                
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
            
        elif output_format == "json":
            return df.to_json(orient="records")
            
        else:  # text format
            report = [
                "=== Trading Opportunities Report ===",
                f"Scan Date: {self.results.get('scan_date')}",
                f"Symbols Analyzed: {self.results.get('symbols_analyzed')}",
                "\nTop Opportunities:",
                "-" * 80
            ]
            
            # Format header
            header = f"{'Symbol':<8} {'Grade':<4} {'Score':<6} {'Direction':<10} {'R/R':<5} {'Stop':<10} {'Target':<10} {'Entropy':<10} {'Regime':<10}"
            report.append(header)
            report.append("-" * 80)
            
            # Add each opportunity
            for _, row in df.iterrows():
                line = (
                    f"{row['symbol']:<8} {row['grade']:<4} {row['score']:<6.1f} "
                    f"{row['direction']:<10} {row['risk_reward']:<5.2f} "
                    f"{str(row['stop_loss']):<10} {str(row['take_profit']):<10} "
                    f"{row['entropy_state']:<10} {row['regime']:<10}"
                )
                report.append(line)
                
            return "\n".join(report)
    
    def save_report(self, filename=None, output_format="csv"):
        """
        Save the scan results to a file.
        
        Args:
            filename (str): Output filename (None for auto-generated)
            output_format (str): Format to save (csv, html, json)
            
        Returns:
            str: Path to saved file
        """
        if not isinstance(self.results.get("opportunities"), pd.DataFrame) or self.results["opportunities"].empty:
            logger.warning("No opportunities to save.")
            return None
            
        # Generate filename if not provided
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_opportunities_{date_str}.{output_format}"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else ".", exist_ok=True)
        
        # Save in requested format
        df = self.results["opportunities"]
        
        if output_format == "csv":
            df.to_csv(filename, index=False)
        elif output_format == "html":
            with open(filename, "w") as f:
                f.write(self.generate_report(output_format="html"))
        elif output_format == "json":
            df.to_json(filename, orient="records")
        else:
            logger.warning(f"Unsupported output format: {output_format}")
            return None
            
        logger.info(f"Report saved to {filename}")
        return filename

    def scan(self, symbols, min_grade='C', direction='All', limit=None):
        """
        Scan a list of symbols for trading opportunities.
        
        Args:
            symbols: List of symbols to scan
            min_grade: Minimum opportunity grade (A, B, C, D)
            direction: Direction filter ('Bullish', 'Bearish', 'Neutral', 'All')
            limit: Maximum number of symbols to scan
        
        Returns:
            List of opportunity dictionaries
        """
        logger.info(f"Scanning {len(symbols) if limit is None else min(len(symbols), limit)} symbols for opportunities")
        
        # Limit the number of symbols if specified
        if limit is not None and limit > 0:
            symbols = symbols[:limit]
        
        # Create a thread pool to process symbols in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self._analyze_symbol, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:  # Only add non-empty results
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
        
        logger.info(f"Completed analysis of {len(symbols)} symbols")
        
        # Filter and sort opportunities
        opportunities = self._filter_opportunities(results, min_grade, direction)
        
        return opportunities

    def _analyze_symbol(self, symbol):
        """Analyze a single symbol and return opportunity data."""
        try:
            # Run the analysis pipeline
            analysis_results = self.pipeline.run(symbol)
            
            # Check if we have valid results
            if not analysis_results or "status" in analysis_results and analysis_results["status"].startswith("Error"):
                logger.warning(f"{symbol}: Analysis failed - {analysis_results.get('status', 'Unknown error')}")
                return None
            
            # Extract Greek analysis results
            greek_results = analysis_results.get("greek_analysis", {})
            
            # Handle DataFrame truth value ambiguity
            self._process_dataframes(greek_results)
            
            # Create opportunity data
            opportunity = {
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "price": analysis_results.get("market_data", {}).get("currentPrice", 0),
                "grade": "C",  # Default grade
                "direction": "Neutral",  # Default direction
                "reset_points": [],
                "energy_levels": [],
                "market_regime": {},
                "analysis_summary": "No significant patterns detected"
            }
            
            # Extract reset points
            reset_points = greek_results.get("reset_points", [])
            if reset_points:
                opportunity["reset_points"] = reset_points
                
                # Determine grade and direction based on reset points
                strongest_signal = self._get_strongest_signal(reset_points)
                if strongest_signal:
                    opportunity["grade"] = strongest_signal.get("grade", "C")
                    opportunity["direction"] = strongest_signal.get("direction", "Neutral")
                    opportunity["analysis_summary"] = strongest_signal.get("description", "")
            
            # Extract energy levels
            energy_levels = greek_results.get("energy_levels", [])
            if energy_levels:
                opportunity["energy_levels"] = energy_levels
            
            # Extract market regime
            market_regime = greek_results.get("market_regime", {})
            if market_regime:
                opportunity["market_regime"] = market_regime
                
                # If no reset points, use market regime for direction
                if not reset_points and "bias" in market_regime:
                    bias = market_regime["bias"]
                    if bias > 0.3:
                        opportunity["direction"] = "Bullish"
                    elif bias < -0.3:
                        opportunity["direction"] = "Bearish"
            
            return opportunity
        
        except Exception as e:
            logger.error(f"Error in analysis for {symbol}: {e}")
            return None

    def _process_dataframes(self, results):
        """Convert any DataFrames in the results to lists or dictionaries."""
        if not results:
            return
        
        # Process reset points
        if "reset_points" in results:
            reset_points = results["reset_points"]
            if hasattr(reset_points, "empty"):  # It's a DataFrame
                if not reset_points.empty:
                    results["reset_points"] = reset_points.to_dict('records')
                else:
                    results["reset_points"] = []
    
        # Process energy levels
        if "energy_levels" in results:
            energy_levels = results["energy_levels"]
            if hasattr(energy_levels, "empty"):  # It's a DataFrame
                if not energy_levels.empty:
                    results["energy_levels"] = energy_levels.to_dict('records')
                else:
                    results["energy_levels"] = []
    
        # Process market regime
        if "market_regime" in results:
            market_regime = results["market_regime"]
            if hasattr(market_regime, "empty"):  # It's a DataFrame
                if not market_regime.empty:
                    results["market_regime"] = market_regime.iloc[0].to_dict()
                else:
                    results["market_regime"] = {}
    
        # Process aggregated greeks
        if "aggregated_greeks" in results:
            agg_greeks = results["aggregated_greeks"]
            if hasattr(agg_greeks, "empty"):  # It's a DataFrame
                if not agg_greeks.empty:
                    results["aggregated_greeks"] = agg_greeks.to_dict('records')
                else:
                    results["aggregated_greeks"] = {}
