"""
Market Regime Calendar and Instrument Tracker
Tracks Greek Energy Flow metrics over time for chosen instruments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import logging
import time

class InstrumentTracker:
    def __init__(self, instruments=None, data_dir="./data/tracker"):
        """
        Initialize the tracker with a list of instruments to monitor.
        
        Args:
            instruments (list): List of ticker symbols to track
            data_dir (str): Directory to store tracking data
        """
        self.instruments = instruments or []
        self.data_dir = data_dir
        self.regime_history = {}
        self.reset_point_history = {}
        self.energy_level_history = {}
        self.anomaly_history = {}
        self.performance_metrics = {}
        self.performance_history = {}  # Add this line
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(data_dir, 'tracker.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing tracker for {len(instruments)} instruments")
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load existing tracking data if available."""
        history_file = os.path.join(self.data_dir, "regime_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.regime_history = json.load(f)
                self.logger.info(f"Loaded existing regime history for {len(self.regime_history)} instruments")
            except Exception as e:
                self.logger.error(f"Error loading regime history: {e}")
        
        # Load other history files
        for history_type in ["reset_point", "energy_level", "anomaly", "performance"]:
            file_path = os.path.join(self.data_dir, f"{history_type}_history.json")
            target_attr = f"{history_type}_history"
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        setattr(self, target_attr, json.load(f))
                    self.logger.info(f"Loaded existing {history_type} history")
                except Exception as e:
                    self.logger.error(f"Error loading {history_type} history: {e}")
    
    def _save_data(self):
        """Save tracking data to files."""
        for history_type in ["regime", "reset_point", "energy_level", "anomaly", "performance"]:
            file_path = os.path.join(self.data_dir, f"{history_type}_history.json")
            source_attr = f"{history_type}_history"
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(getattr(self, source_attr), f, indent=2)
                self.logger.info(f"Saved {history_type} history")
            except Exception as e:
                self.logger.error(f"Error saving {history_type} history: {e}")
    
    def update_instrument_data(self, symbol, analysis_results):
        """
        Update tracking data for a specific instrument based on analysis results.
        
        Args:
            symbol (str): Instrument symbol
            analysis_results (dict): Results from Greek Energy Flow analysis
        """
        if symbol not in self.instruments:
            self.instruments.append(symbol)
            self.logger.info(f"Added new instrument to tracking: {symbol}")
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Extract key metrics from analysis results
        greek_analysis = analysis_results.get("greek_analysis", {})
        market_regime = greek_analysis.get("market_regime", {})
        
        # Initialize history entries if needed
        for history_dict in [self.regime_history, self.reset_point_history, 
                           self.energy_level_history, self.anomaly_history]:
            if symbol not in history_dict:
                history_dict[symbol] = {}
        
        # Update regime history
        self.regime_history[symbol][today] = {
            "primary_regime": market_regime.get("primary_label", "Unknown"),
            "secondary_regime": market_regime.get("secondary_label", "Unknown"),
            "volatility_regime": market_regime.get("volatility_regime", "Normal"),
            "dominant_greek": market_regime.get("dominant_greek", "Unknown")
        }
        
        # Update reset point history
        reset_points = greek_analysis.get("reset_points", [])
        self.reset_point_history[symbol][today] = {
            "count": len(reset_points),
            "strengths": [rp.get("strength", 0) for rp in reset_points],
            "prices": [rp.get("price", 0) for rp in reset_points]
        }
        
        # Update energy level history
        energy_levels = greek_analysis.get("energy_levels", [])
        self.energy_level_history[symbol][today] = {
            "count": len(energy_levels),
            "strengths": [el.get("strength", 0) for el in energy_levels],
            "prices": [el.get("price", 0) for el in energy_levels]
        }
        
        # Update anomaly history
        anomalies = greek_analysis.get("greek_anomalies", [])
        self.anomaly_history[symbol][today] = {
            "count": len(anomalies),
            "types": [a.get("type", "Unknown") for a in anomalies],
            "strengths": [a.get("strength", 0) for a in anomalies]
        }
        
        self.logger.info(f"Updated tracking data for {symbol}")
        
        # Save updated data
        self._save_data()
    
    def update_performance(self, symbol, price_data, trade_results=None):
        """
        Update performance metrics for a specific instrument.
        
        Args:
            symbol (str): Instrument symbol
            price_data (dict): Recent price data including current price
            trade_results (dict, optional): Results of recent trades
        """
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {}
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Extract current price
        current_price = price_data.get("current_price", 0)
        
        # Initialize performance entry
        performance = {
            "price": current_price,
            "trades": []
        }
        
        # Add trade results if available
        if trade_results:
            performance["trades"] = trade_results
        
        # Store performance data
        self.performance_metrics[symbol][today] = performance
        
        self.logger.info(f"Updated performance metrics for {symbol}")
        
        # Save updated data
        self._save_data()
    
    def get_regime_transitions(self, symbol, days=30):
        """
        Get regime transitions for a specific instrument over the specified number of days.
        
        Args:
            symbol (str): Instrument symbol
            days (int): Number of days to look back
            
        Returns:
            list: List of regime transitions
        """
        if symbol not in self.regime_history:
            return []
        
        # Get regime history for the symbol
        history = self.regime_history[symbol]
        dates = sorted(history.keys())
        
        # Filter to the specified number of days
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent_dates = [d for d in dates if d >= cutoff_date]
        
        if len(recent_dates) < 2:
            return []
        
        # Find transitions
        transitions = []
        for i in range(1, len(recent_dates)):
            prev_date = recent_dates[i-1]
            curr_date = recent_dates[i]
            
            prev_regime = history[prev_date]["primary_regime"]
            curr_regime = history[curr_date]["primary_regime"]
            
            if prev_regime != curr_regime:
                transitions.append({
                    "from_date": prev_date,
                    "to_date": curr_date,
                    "from_regime": prev_regime,
                    "to_regime": curr_regime
                })
        
        return transitions
    
    def get_reset_point_effectiveness(self, symbol, days=30):
        """
        Analyze the effectiveness of reset points as support/resistance levels.
        
        Args:
            symbol (str): Instrument symbol
            days (int): Number of days to look back
            
        Returns:
            dict: Reset point effectiveness metrics
        """
        if symbol not in self.reset_point_history or symbol not in self.performance_metrics:
            return {}
        
        reset_history = self.reset_point_history[symbol]
        price_history = self.performance_metrics[symbol]
        
        dates = sorted([d for d in reset_history.keys() 
                      if d in price_history 
                      and d >= (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")])
        
        if not dates:
            return {}
        
        # Analyze how often price respects reset points
        respect_count = 0
        breach_count = 0
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_price = price_history[prev_date]["price"]
            curr_price = price_history[curr_date]["price"]
            
            # Get reset points from previous day
            reset_prices = reset_history[prev_date]["prices"]
            
            if not reset_prices:
                continue
            
            # Check if price moved between reset points
            for reset_price in reset_prices:
                # Price moved toward reset point but didn't cross
                if (prev_price < reset_price < curr_price or 
                    prev_price > reset_price > curr_price):
                    respect_count += 1
                # Price crossed reset point
                elif (prev_price < reset_price and curr_price > reset_price or
                     prev_price > reset_price and curr_price < reset_price):
                    breach_count += 1
        
        total = respect_count + breach_count
        
        return {
            "respect_rate": respect_count / total if total > 0 else 0,
            "breach_rate": breach_count / total if total > 0 else 0,
            "total_tests": total
        }
    
    def get_optimal_instruments(self, criteria="transitions", top_n=5):
        """
        Get the top N instruments based on specified criteria.
        
        Args:
            criteria (str): Criteria to use for ranking
                - "transitions": Most regime transitions
                - "reset_points": Most reset points
                - "anomalies": Most anomalies
                - "reset_effectiveness": Most effective reset points
            top_n (int): Number of top instruments to return
            
        Returns:
            list: Top N instruments based on criteria
        """
        rankings = {}
        
        if criteria == "transitions":
            for symbol in self.instruments:
                transitions = self.get_regime_transitions(symbol)
                rankings[symbol] = len(transitions)
        
        elif criteria == "reset_points":
            for symbol in self.instruments:
                if symbol in self.reset_point_history:
                    # Average number of reset points over recent history
                    history = self.reset_point_history[symbol]
                    if history:
                        counts = [data["count"] for data in history.values()]
                        rankings[symbol] = sum(counts) / len(counts) if counts else 0
        
        elif criteria == "anomalies":
            for symbol in self.instruments:
                if symbol in self.anomaly_history:
                    # Average number of anomalies over recent history
                    history = self.anomaly_history[symbol]
                    if history:
                        counts = [data["count"] for data in history.values()]
                        rankings[symbol] = sum(counts) / len(counts) if counts else 0
        
        elif criteria == "reset_effectiveness":
            for symbol in self.instruments:
                effectiveness = self.get_reset_point_effectiveness(symbol)
                rankings[symbol] = effectiveness.get("respect_rate", 0)
        
        # Sort by ranking (descending)
        sorted_symbols = sorted(rankings.keys(), key=lambda s: rankings[s], reverse=True)
        
        # Return top N
        return sorted_symbols[:top_n]
    
    def generate_market_regime_report(self, days=30):
        """
        Generate a report on market regimes across all instruments.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            dict: Market regime report
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Collect regime data
        regime_counts = {}
        dominant_greek_counts = {}
        volatility_regime_counts = {}
        
        for symbol, history in self.regime_history.items():
            for date, data in history.items():
                if date < cutoff_date:
                    continue
                
                regime = data["primary_regime"]
                greek = data["dominant_greek"]
                vol_regime = data["volatility_regime"]
                
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
                
                if greek not in dominant_greek_counts:
                    dominant_greek_counts[greek] = 0
                dominant_greek_counts[greek] += 1
                
                if vol_regime not in volatility_regime_counts:
                    volatility_regime_counts[vol_regime] = 0
                volatility_regime_counts[vol_regime] += 1
        
        # Calculate percentages
        total_observations = sum(regime_counts.values())
        regime_percentages = {r: count / total_observations * 100 
                            for r, count in regime_counts.items()}
        
        greek_percentages = {g: count / total_observations * 100 
                           for g, count in dominant_greek_counts.items()}
        
        vol_percentages = {v: count / total_observations * 100 
                         for v, count in volatility_regime_counts.items()}
        
        return {
            "period": f"Last {days} days",
            "total_observations": total_observations,
            "regime_distribution": regime_percentages,
            "dominant_greek_distribution": greek_percentages,
            "volatility_regime_distribution": vol_percentages,
            "most_frequent_regime": max(regime_counts.items(), key=lambda x: x[1])[0],
            "most_frequent_greek": max(dominant_greek_counts.items(), key=lambda x: x[1])[0],
            "most_frequent_volatility": max(volatility_regime_counts.items(), key=lambda x: x[1])[0]
        }
    
    def visualize_market_regime_calendar(self, output_dir="./results/visuals"):
        """
        Generate a visualization of market regime transitions.
        
        Args:
            output_dir (str): Directory to save visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        regime_data = []
        
        for symbol in self.instruments:
            if symbol not in self.regime_history:
                continue
                
            history = self.regime_history[symbol]
            dates = sorted(history.keys())
            
            for date in dates:
                regime_data.append({
                    "symbol": symbol,
                    "date": date,
                    "regime": history[date]["primary_regime"],
                    "dominant_greek": history[date]["dominant_greek"]
                })
        
        if not regime_data:
            self.logger.warning("No regime data available for visualization")
            return
        
        # Convert to DataFrame for easier visualization
        df = pd.DataFrame(regime_data)
        df["date"] = pd.to_datetime(df["date"])
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot regime calendar
        symbols = df["symbol"].unique()
        regimes = df["regime"].unique()
        
        # Create a color map for regimes
        regime_colors = plt.cm.get_cmap("tab10", len(regimes))
        regime_color_map = {regime: regime_colors(i) for i, regime in enumerate(regimes)}
        
        # Plot each symbol
        for i, symbol in enumerate(symbols):
            symbol_data = df[df["symbol"] == symbol]
            
            for _, row in symbol_data.iterrows():
                plt.scatter(row["date"], i, color=regime_color_map[row["regime"]], 
                          s=100, marker="s")
        
        # Set y-ticks to symbol names
        plt.yticks(range(len(symbols)), symbols)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Add legend
        for regime, color in regime_color_map.items():
            plt.scatter([], [], color=color, label=regime, s=100, marker="s")
        plt.legend(title="Market Regimes", loc="upper right")
        
        # Add labels and title
        plt.title("Market Regime Calendar", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Instrument", fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, "market_regime_calendar.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Market regime calendar visualization saved to {output_path}")
    
    def visualize_reset_point_effectiveness(self, output_dir="./results/visuals"):
        """
        Generate a visualization of reset point effectiveness.
        
        Args:
            output_dir (str): Directory to save visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect effectiveness data
        effectiveness_data = []
        
        for symbol in self.instruments:
            effectiveness = self.get_reset_point_effectiveness(symbol)
            if effectiveness:
                effectiveness_data.append({
                    "symbol": symbol,
                    "respect_rate": effectiveness["respect_rate"],
                    "breach_rate": effectiveness["breach_rate"],
                    "total_tests": effectiveness["total_tests"]
                })
        
        if not effectiveness_data:
            self.logger.warning("No reset point effectiveness data available for visualization")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(effectiveness_data)
        
        # Sort by respect rate
        df = df.sort_values("respect_rate", ascending=False)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bar_width = 0.35
        x = np.arange(len(df))
        
        # Plot respect and breach rates
        plt.bar(x - bar_width/2, df["respect_rate"] * 100, 
              width=bar_width, label="Respect Rate (%)", color="green", alpha=0.7)
        plt.bar(x + bar_width/2, df["breach_rate"] * 100, 
              width=bar_width, label="Breach Rate (%)", color="red", alpha=0.7)
        
        # Annotate total tests
        for i, row in enumerate(df.itertuples()):
            plt.annotate(f"Tests: {row.total_tests}", (i, max(row.respect_rate, row.breach_rate) * 100 + 5))
    
    def track_trade_recommendations(self, recommendations_dir="./results"):
        """
        Track trade recommendations and their performance over time.
        
        Args:
            recommendations_dir (str): Directory containing trade recommendation files
        """
        self.logger.info(f"Tracking trade recommendations from {recommendations_dir}")
        
        # Find all recommendation files
        import glob
        rec_files = glob.glob(os.path.join(recommendations_dir, "*_trade_recommendation*.json"))
        
        if not rec_files:
            self.logger.warning(f"No recommendation files found in {recommendations_dir}")
            return
        
        self.logger.info(f"Found {len(rec_files)} recommendation files")
        
        # Process each recommendation file
        for rec_file in rec_files:
            try:
                with open(rec_file, 'r') as f:
                    rec = json.load(f)
                
                # Extract key information
                symbol = rec.get('symbol')
                if not symbol:
                    continue
                    
                strategy = rec.get('strategy', 'Unknown')
                entry_zone = rec.get('entry_criteria', {}).get('price_range', [0, 0])
                target_pct = rec.get('exit_criteria', {}).get('profit_target_percent', 0)
                stop_pct = rec.get('exit_criteria', {}).get('max_loss_percent', 0)
                days_to_hold = rec.get('exit_criteria', {}).get('days_to_hold', 0)
                
                # Get current price if available
                current_price = rec.get('current_price', 0)
                
                # Store recommendation data
                if 'trade_recommendations' not in self.performance_metrics:
                    self.performance_metrics['trade_recommendations'] = {}
                    
                if symbol not in self.performance_metrics['trade_recommendations']:
                    self.performance_metrics['trade_recommendations'][symbol] = []
                
                # Add recommendation with timestamp
                timestamp = os.path.getmtime(rec_file)
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                
                self.performance_metrics['trade_recommendations'][symbol].append({
                    'date': date_str,
                    'strategy': strategy,
                    'entry_zone': entry_zone,
                    'target_pct': target_pct,
                    'stop_pct': stop_pct,
                    'days_to_hold': days_to_hold,
                    'current_price': current_price
                })
                
                self.logger.info(f"Tracked recommendation for {symbol} ({strategy})")
                
            except Exception as e:
                self.logger.error(f"Error processing recommendation file {rec_file}: {e}")
        
        # Save updated data
        self._save_data()

    def analyze_recommendation_effectiveness(self, symbol=None, days=30):
        """
        Analyze the effectiveness of trade recommendations.
        
        Args:
            symbol (str, optional): Specific symbol to analyze, or None for all
            days (int): Number of days to look back
        
        Returns:
            dict: Analysis of recommendation effectiveness
        """
        if 'trade_recommendations' not in self.performance_metrics:
            return {}
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Collect recommendations
        all_recs = []
        
        if symbol:
            # Get recommendations for specific symbol
            if symbol in self.performance_metrics['trade_recommendations']:
                symbol_recs = self.performance_metrics['trade_recommendations'][symbol]
                all_recs.extend([r for r in symbol_recs if r['date'] >= cutoff_date])
        else:
            # Get recommendations for all symbols
            for sym, recs in self.performance_metrics['trade_recommendations'].items():
                all_recs.extend([r for r in recs if r['date'] >= cutoff_date])
        
        if not all_recs:
            return {}
        
        # Analyze by strategy
        strategy_counts = {}
        strategy_targets = {}
        strategy_days = {}
        
        for rec in all_recs:
            strategy = rec['strategy']
            target = rec['target_pct']
            days = rec['days_to_hold']
            
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
                strategy_targets[strategy] = []
                strategy_days[strategy] = []
            
            strategy_counts[strategy] += 1
            strategy_targets[strategy].append(target)
            strategy_days[strategy].append(days)
        
        # Calculate averages
        strategy_analysis = {}
        for strategy, count in strategy_counts.items():
            avg_target = sum(strategy_targets[strategy]) / len(strategy_targets[strategy])
            avg_days = sum(strategy_days[strategy]) / len(strategy_days[strategy])
            
            strategy_analysis[strategy] = {
                'count': count,
                'avg_target_pct': avg_target,
                'avg_days_to_hold': avg_days
            }
        
        return {
            'period': f"Last {days} days",
            'total_recommendations': len(all_recs),
            'strategy_analysis': strategy_analysis,
            'most_recommended_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0]
        }

    def generate_combined_report(self, output_dir="./results/reports", days=30):
        """
        Generate a combined report of market regimes and trade recommendations.
        
        Args:
            output_dir (str): Directory to save the report
            days (int): Number of days to look back
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get market regime report
        regime_report = self.generate_market_regime_report(days)
        
        # Get recommendation effectiveness
        rec_effectiveness = self.analyze_recommendation_effectiveness(days=days)
        
        # Combine reports
        combined_report = {
            'report_date': datetime.now().strftime("%Y-%m-%d"),
            'period': f"Last {days} days",
            'market_regimes': regime_report,
            'trade_recommendations': rec_effectiveness
        }
        
        # Add instrument-specific data
        instrument_data = {}
        for symbol in self.instruments:
            # Get regime transitions
            transitions = self.get_regime_transitions(symbol, days)
            
            # Get reset point effectiveness
            reset_effectiveness = self.get_reset_point_effectiveness(symbol, days)
            
            instrument_data[symbol] = {
                'regime_transitions': transitions,
                'reset_point_effectiveness': reset_effectiveness
            }
        
        combined_report['instrument_data'] = instrument_data
        
        # Save report
        report_file = os.path.join(output_dir, f"combined_report_{datetime.now().strftime('%Y%m%d')}.json")
        try:
            with open(report_file, 'w') as f:
                json.dump(combined_report, f, indent=2)
            self.logger.info(f"Combined report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Error saving combined report: {e}")
        
        return combined_report

    def safe_track(self, max_retries=3):
        """
        Safely track instruments with retry logic for production use.
        
        Args:
            max_retries: Maximum number of retry attempts
        
        Returns:
            bool: Success status
        """
        for attempt in range(max_retries):
            try:
                # Track all instruments
                self.track_all_instruments()
                
                # Track recommendations
                self.track_trade_recommendations()
                
                # Generate report
                self.generate_combined_report()
                
                self.logger.info("Successfully completed tracking cycle")
                return True
                
            except Exception as e:
                self.logger.error(f"Error during tracking cycle (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        self.logger.critical("Failed to complete tracking cycle after maximum retries")
        return False

    def start_real_time_monitoring(self, interval_minutes=15):
        """
        Start real-time monitoring of instruments and recommendations.
        
        Args:
            interval_minutes: Interval between tracking cycles in minutes
        """
        self.logger.info(f"Starting real-time monitoring with {interval_minutes}-minute intervals")
        
        # Initialize status tracking
        self.monitoring_status = {
            'start_time': datetime.now(),
            'cycles_completed': 0,
            'last_cycle_time': None,
            'errors': 0
        }
        
        try:
            while True:
                cycle_start = datetime.now()
                self.logger.info(f"Starting tracking cycle at {cycle_start.strftime('%H:%M:%S')}")
                
                # Run tracking cycle
                success = self.safe_track()
                
                # Update status
                self.monitoring_status['cycles_completed'] += 1
                self.monitoring_status['last_cycle_time'] = datetime.now()
                if not success:
                    self.monitoring_status['errors'] += 1
                
                # Calculate time to next cycle
                elapsed = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(0, interval_minutes * 60 - elapsed)
                
                self.logger.info(f"Cycle completed in {elapsed:.1f} seconds. Next cycle in {wait_time/60:.1f} minutes")
                
                # Save status to file for external monitoring
                self._save_monitoring_status()
                
                # Wait for next cycle
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            self.logger.info("Real-time monitoring stopped by user")
        except Exception as e:
            self.logger.critical(f"Critical error in monitoring loop: {e}")
            # Save final status before exiting
            self._save_monitoring_status()
            raise

    def _save_monitoring_status(self):
        """Save monitoring status to file for external monitoring."""
        try:
            status_file = os.path.join(self.data_dir, "monitoring_status.json")
            
            # Convert datetime objects to strings for JSON serialization
            status_data = {
                'cycles_completed': self.monitoring_status['cycles_completed'],
                'errors': self.monitoring_status['errors'],
                'start_time': self.monitoring_status['start_time'].isoformat(),
                'current_time': datetime.now().isoformat(),
                'uptime_hours': (datetime.now() - self.monitoring_status['start_time']).total_seconds() / 3600
            }
            
            # Add last_cycle_time if it exists
            if self.monitoring_status.get('last_cycle_time'):
                status_data['last_cycle_time'] = self.monitoring_status['last_cycle_time'].isoformat()
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving monitoring status: {e}")

    def is_market_open(self):
        """Check if the US market is currently open."""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check market hours (9:30 AM - 4:00 PM Eastern Time)
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now_eastern = now.astimezone(eastern)
        
        market_open = now_eastern.replace(hour=9, minute=30, second=0)
        market_close = now_eastern.replace(hour=16, minute=0, second=0)
        
        return market_open <= now_eastern <= market_close

    def start_market_hours_monitoring(self):
        """Start monitoring that respects market hours."""
        self.logger.info("Starting market-hours-aware monitoring")
        
        try:
            while True:
                now = datetime.now()
                
                if self.is_market_open():
                    # Market is open - run tracking every 15 minutes
                    self.logger.info("Market is open - running tracking cycle")
                    self.safe_track()
                    
                    # Wait 15 minutes before next check
                    time.sleep(15 * 60)
                else:
                    # Market is closed
                    self.logger.info("Market is closed")
                    
                    # If it's end of day (after 4 PM ET), run end-of-day analysis
                    import pytz
                    eastern = pytz.timezone('US/Eastern')
                    now_eastern = now.astimezone(eastern)
                    
                    if now_eastern.hour == 16 and now_eastern.minute < 15:
                        self.logger.info("Running end-of-day analysis")
                        self.generate_combined_report()
                    
                    # Check again in 30 minutes
                    time.sleep(30 * 60)
                    
        except KeyboardInterrupt:
            self.logger.info("Market hours monitoring stopped by user")
        except Exception as e:
            self.logger.critical(f"Critical error in market hours monitoring: {e}")
            raise

    def track_all_instruments(self):
        """
        Track all instruments in the tracker's list.
        
        Returns:
            dict: Results of tracking for each instrument
        """
        results = {}
        self.logger.info(f"Tracking {len(self.instruments)} instruments")
        
        for symbol in self.instruments:
            try:
                # Get latest data for the instrument
                data = self._get_instrument_data(symbol)
                
                # Run analysis
                analysis_results = self._analyze_instrument(symbol, data)
                
                # Update tracking data
                self.update_instrument_data(symbol, analysis_results)
                
                results[symbol] = {"status": "success"}
                self.logger.info(f"Successfully tracked {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error tracking {symbol}: {e}")
                results[symbol] = {"status": "error", "message": str(e)}
        
        return results

    def _get_instrument_data(self, symbol):
        """
        Get latest market data for an instrument.
        
        Args:
            symbol (str): Instrument symbol
        
        Returns:
            dict: Market data for the instrument
        """
        try:
            # Import API fetcher
            from api_fetcher import fetch_underlying_snapshot, fetch_options_chain_snapshot
            
            # Get API key from config or environment
            api_key = os.environ.get("POLYGON_API_KEY", "YOUR_API_KEY_HERE")
            
            # Fetch underlying data
            self.logger.info(f"Fetching real market data for {symbol}")
            underlying_data = fetch_underlying_snapshot(symbol, api_key)
            
            # Fetch options chain
            options_data = fetch_options_chain_snapshot(symbol, api_key)
            
            # Combine data
            return {
                "symbol": symbol,
                "last_price": underlying_data.get("last", {}).get("price", 0.0),
                "timestamp": datetime.now().isoformat(),
                "underlying_data": underlying_data,
                "options_data": options_data
            }
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            # Return placeholder data as fallback
            return {
                "symbol": symbol,
                "last_price": 100.0,
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_instrument(self, symbol, data):
        """
        Analyze instrument data using Greek Energy Flow analysis.
        
        Args:
            symbol (str): Instrument symbol
            data (dict): Market data for the instrument
        
        Returns:
            dict: Analysis results
        """
        try:
            # Import analysis modules
            from analysis.trade_recommendations import TradeRecommendationEngine
            
            # Create recommendation engine
            engine = TradeRecommendationEngine()
            
            # Extract options data
            options_data = data.get("options_data", {})
            underlying_data = data.get("underlying_data", {})
            current_price = data.get("last_price", 0.0)
            
            # Perform Greek analysis
            # This is a simplified placeholder - you'd need to implement the actual analysis
            greek_analysis = {
                "symbol": symbol,
                "current_price": current_price,
                "market_regime": {
                    "primary": "bullish",
                    "volatility": "normal"
                },
                "dominant_greek": "gamma"
            }
            
            # Generate entropy profile
            entropy_profile = {
                "entropy_score": 0.65,
                "energy_state": "balanced"
            }
            
            # Generate recommendation
            recommendation = engine.select_optimal_strategy(
                greek_analysis, 
                entropy_profile, 
                current_price
            )
            
            # Return complete analysis
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "greek_analysis": greek_analysis,
                "entropy_profile": entropy_profile,
                "recommendation": recommendation
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            # Return basic analysis as fallback
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "greek_analysis": {
                    "regime": "unknown"
                }
            }

    def save_performance_history(self, instrument, performance_data):
        """
        Save performance history for an instrument.
        
        Args:
            instrument (str): Instrument symbol
            performance_data (dict): Performance metrics to save
        """
        if instrument not in self.performance_history:
            self.performance_history[instrument] = []
            
        # Add timestamp to performance data
        performance_data['timestamp'] = datetime.now().isoformat()
        self.performance_history[instrument].append(performance_data)


