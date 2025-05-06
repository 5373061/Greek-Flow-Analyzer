# visualization/chart_generator.py
import os
import matplotlib.pyplot as plt
import logging
from datetime import date
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generates charts for options analysis visualization."""
    
    def __init__(self, output_dir="charts"):
        """
        Initialize ChartGenerator.
        
        Args:
            output_dir (str): Directory to save charts
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_reset_points(self, symbol: str, greek_res: Dict[str, Any], today_date: date):
        """
        Plot reset points (price vs significance) for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            greek_res (dict): Greek analysis results
            today_date (date): Analysis date
            
        Returns:
            str: Path to saved chart or None if failed
        """
        try:
            rps = greek_res.get("reset_points", [])
            if not rps:
                logger.debug(f"No reset points to plot for {symbol}")
                return None

            # Extract prices and significance values (handle numeric or string with '%')
            prices = []
            sigs = []
            for pt in rps:
                try:
                    prices.append(float(pt.get("price", 0)))
                    
                    # Parse significance (could be string with % or float)
                    sig_val = pt.get("significance", 0)
                    if isinstance(sig_val, str):
                        sig_val = sig_val.strip().rstrip('%')
                    sigs.append(float(sig_val))
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse reset point: {pt}")
                    continue

            if not prices or not sigs:
                logger.debug(f"No valid reset points data for {symbol}")
                return None

            plt.figure(figsize=(10, 6))
            plt.scatter(prices, sigs, alpha=0.7)
            plt.title(f"{symbol} Reset Points ({today_date})")
            plt.xlabel("Price")
            plt.ylabel("Significance (%)")
            plt.grid(True, alpha=0.3)

            out_file = os.path.join(self.output_dir, f"{symbol}_reset_points_{today_date}.png")
            plt.savefig(out_file, bbox_inches="tight", dpi=100)
            plt.close()
            
            logger.info(f"Saved Reset Points chart to {os.path.abspath(out_file)}")
            return out_file
            
        except Exception as e:
            logger.error(f"Error plotting reset points for {symbol}: {e}")
            return None

    def plot_energy_levels(self, symbol: str, greek_res: Dict[str, Any], today_date: date):
        """
        Plot energy levels (price vs strength) for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            greek_res (dict): Greek analysis results
            today_date (date): Analysis date
            
        Returns:
            str: Path to saved chart or None if failed
        """
        try:
            els = greek_res.get("energy_levels", [])
            if not els:
                logger.debug(f"No energy levels to plot for {symbol}")
                return None

            prices = []
            strengths = []
            directions = []
            for lvl in els:
                try:
                    prices.append(float(lvl.get("price", 0)))
                    
                    # Parse strength (could be string with % or float)
                    str_val = lvl.get("strength", 0)
                    if isinstance(str_val, str):
                        str_val = str_val.strip().rstrip('%')
                    strengths.append(float(str_val))
                    
                    directions.append(str(lvl.get("direction", "")).lower())
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse energy level: {lvl}")
                    continue

            if not prices or not strengths:
                logger.debug(f"No valid energy levels data for {symbol}")
                return None

            colors = ["green" if "support" in d else "red" if "resistance" in d else "blue" for d in directions]

            plt.figure(figsize=(10, 6))
            plt.bar(prices, strengths, color=colors, alpha=0.7, width=prices[0]*0.015 if prices else 1)
            plt.title(f"{symbol} Energy Levels ({today_date})")
            plt.xlabel("Price")
            plt.ylabel("Strength (%)")
            plt.grid(True, alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Support'),
                Patch(facecolor='red', alpha=0.7, label='Resistance'),
                Patch(facecolor='blue', alpha=0.7, label='Other')
            ]
            plt.legend(handles=legend_elements)

            out_file = os.path.join(self.output_dir, f"{symbol}_energy_levels_{today_date}.png")
            plt.savefig(out_file, bbox_inches="tight", dpi=100)
            plt.close()
            
            logger.info(f"Saved Energy Levels chart to {os.path.abspath(out_file)}")
            return out_file
            
        except Exception as e:
            logger.error(f"Error plotting energy levels for {symbol}: {e}")
            return None
            
    def plot_price_projections(self, symbol: str, greek_res: Dict[str, Any], today_date: date):
        """
        Plot price projections for Vanna and Charm.
        
        Args:
            symbol (str): Stock symbol
            greek_res (dict): Greek analysis results
            today_date (date): Analysis date
            
        Returns:
            list: Paths to saved charts or empty list if failed
        """
        saved_files = []
        
        # Plot Vanna projections
        try:
            vanna_proj = greek_res.get("vanna_projections", {})
            if vanna_proj and "price_points" in vanna_proj and "projections" in vanna_proj:
                prices = vanna_proj["price_points"]
                if isinstance(prices[0], str):
                    prices = [float(p) for p in prices]
                
                plt.figure(figsize=(10, 6))
                
                # Plot all projections
                for label, values in vanna_proj["projections"].items():
                    if label == "Total":
                        # Plot total with thicker line
                        plt.plot(prices, [float(v) for v in values], linewidth=2.5, label=label)
                    else:
                        # Plot others with thinner lines
                        plt.plot(prices, [float(v) for v in values], linewidth=1, alpha=0.7, label=label)
                
                plt.title(f"{symbol} Vanna Projections ({today_date})")
                plt.xlabel("Price")
                plt.ylabel("Vanna Exposure")
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.legend()
                
                out_file = os.path.join(self.output_dir, f"{symbol}_vanna_proj_{today_date}.png")
                plt.savefig(out_file, bbox_inches="tight", dpi=100)
                plt.close()
                
                logger.info(f"Saved Vanna projection chart to {os.path.abspath(out_file)}")
                saved_files.append(out_file)
        except Exception as e:
            logger.error(f"Error plotting Vanna projections for {symbol}: {e}")
        
        # Plot Charm projections
        try:
            charm_proj = greek_res.get("charm_projections", {})
            if charm_proj and "price_points" in charm_proj and "projections" in charm_proj:
                prices = charm_proj["price_points"]
                if isinstance(prices[0], str):
                    prices = [float(p) for p in prices]
                
                plt.figure(figsize=(10, 6))
                
                # Plot all projections
                for label, values in charm_proj["projections"].items():
                    if label == "Total":
                        # Plot total with thicker line
                        plt.plot(prices, [float(v) for v in values], linewidth=2.5, label=label)
                    else:
                        # Plot others with thinner lines
                        plt.plot(prices, [float(v) for v in values], linewidth=1, alpha=0.7, label=label)
                
                plt.title(f"{symbol} Charm Projections ({today_date})")
                plt.xlabel("Price")
                plt.ylabel("Charm Exposure")
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.legend()
                
                out_file = os.path.join(self.output_dir, f"{symbol}_charm_proj_{today_date}.png")
                plt.savefig(out_file, bbox_inches="tight", dpi=100)
                plt.close()
                
                logger.info(f"Saved Charm projection chart to {os.path.abspath(out_file)}")
                saved_files.append(out_file)
        except Exception as e:
            logger.error(f"Error plotting Charm projections for {symbol}: {e}")
            
        return saved_files

def create_pattern_chart(ticker, pattern_data, output_dir="charts"):
    """
    Create a chart visualizing identified ordinal patterns.
    
    Args:
        ticker: Ticker symbol
        pattern_data: Pattern data from GreekOrdinalPatternAnalyzer
        output_dir: Directory to save the chart
    
    Returns:
        Path to the saved chart
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    # Create figure
    fig, axes = plt.subplots(len(pattern_data), 1, figsize=(10, 2*len(pattern_data)))
    if len(pattern_data) == 1:
        axes = [axes]
    
    # Plot each Greek's patterns
    for i, (greek, patterns) in enumerate(pattern_data.items()):
        ax = axes[i]
        
        # Get pattern data
        pattern_tuple = patterns.get('pattern', ())
        description = patterns.get('description', '')
        stats = patterns.get('stats', {})
        
        # Create a visual representation of the pattern
        x = np.arange(len(pattern_tuple))
        y = np.array([p for p in pattern_tuple])
        
        # Plot the pattern
        ax.plot(x, y, 'o-', linewidth=2)
        ax.set_title(f"{greek.capitalize()} Pattern: {description}")
        
        # Add stats
        win_rate = stats.get('win_rate', 0) * 100
        count = stats.get('count', 0)
        avg_return = stats.get('avg_return', 0) * 100
        
        ax.text(0.02, 0.02, 
                f"Win Rate: {win_rate:.1f}% | Count: {count} | Avg Return: {avg_return:.1f}%",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Set axis limits
        ax.set_ylim(-0.5, len(pattern_tuple) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t-{len(pattern_tuple)-i-1}" for i in range(len(pattern_tuple))])
        
    plt.tight_layout()
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{ticker}_patterns.png")
    plt.savefig(chart_path, dpi=100)
    plt.close()
    
    return chart_path
