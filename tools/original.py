import os
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import logging
import sys
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to avoid threading issues

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegratedDashboard:
    """Dashboard that integrates trade recommendations with market regime tracking."""
    
    def __init__(self, root, base_dir=None):
        """Initialize the dashboard."""
        self.root = root
        
        # Set base directory (project root)
        if base_dir is None:
            self.base_dir = os.getcwd()
        else:
            self.base_dir = base_dir
            
        # Default subdirectories
        self.results_dir = os.path.join(self.base_dir, "results")
        self.data_dir = os.path.join(self.base_dir, "data")
        
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Results directory: {self.results_dir}")
        
        # Initialize data structures
        self.recommendations = []
        self.market_regimes = {}
        self.energy_levels = {}
        self.reset_points = {}
        self.regime_transitions = {}
        self.greek_data = {}  # For storing Greek metrics
        self.trade_durations = {}  # For tracking trade durations
        self.selected_recommendation = None
        
        # Print paths and directory contents for debugging
        self._print_debug_info()
        
        # Set up the dashboard UI
        self.setup_ui()
        
        # Load recommendations and tracker data
        self.load_all_data()
        
        # Populate dashboard
        self.populate_dashboard()
        
    def _print_debug_info(self):
        """Print debug information about paths and files."""
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Results directory exists: {os.path.exists(self.results_dir)}")
        
        if os.path.exists(self.results_dir):
            files = os.listdir(self.results_dir)
            logger.info(f"Found {len(files)} files in results directory")
            recommendation_files = [f for f in files if f.endswith("_trade_recommendation.json")]
            logger.info(f"Found {len(recommendation_files)} trade recommendation files")
            
            # Check for market regime files
            regime_files = [f for f in files if "market_regime" in f]
            logger.info(f"Found {len(regime_files)} market regime files")
            
            # Check for enhanced recommendation files
            enhanced_files = [f for f in files if f.endswith("_enhanced_recommendation.json")]
            logger.info(f"Found {len(enhanced_files)} enhanced recommendation files")
            
    def setup_ui(self):
        """Set up the dashboard UI."""
        # Configure the root window
        self.root.title("Integrated Trade Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Create theme settings
        self.create_style()
        
        # Create main layout frames
        self.create_layout()
        
        # Create top controls
        self.create_top_controls()
        
        # Create recommendation list
        self.create_recommendation_list()
        
        # Create details view
        self.create_details_view()
        
        # Create status bar
        self.create_status_bar()
        
        logger.info("UI setup complete")
        
    def create_style(self):
        """Create and configure visual style for the dashboard."""
        self.style = ttk.Style()
        
        # Configure the theme
        available_themes = self.style.theme_names()
        if 'clam' in available_themes:
            self.style.theme_use('clam')
            
        # Configure colors
        bg_color = "#f5f5f5"
        accent_color = "#4a86e8"
        text_color = "#333333"
        
        # Configure styles
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=text_color)
        self.style.configure("TButton", background=accent_color, foreground="white")
        self.style.configure("Accent.TButton", background=accent_color, foreground="white")
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=bg_color, padding=[10, 5])
        
        # Configure Treeview
        self.style.configure("Treeview", 
                         background=bg_color,
                         fieldbackground=bg_color,
                         foreground=text_color)
        self.style.configure("Treeview.Heading", 
                         background=accent_color,
                         foreground="white",
                         font=('Arial', 10, 'bold'))
        
        # Configure tag colors for risk levels
        self.treeview_tags = {
            "LOW": {"background": "#d9ead3"},    # Light green
            "MEDIUM": {"background": "#fff2cc"}, # Light yellow
            "HIGH": {"background": "#f4cccc"},   # Light red
            "selected": {"background": "#c9daf8"}, # Light blue
            "aligned": {"background": "#d0e0e3"}  # Light teal - for regime-aligned trades
        }
        
    def create_layout(self):
        """Create the main layout frames."""
        # Create a PanedWindow for resizable frames
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        
        # Left frame for recommendation list
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Right frame for details and charts
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=2)
        
        # Top frame for filters and controls
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=5, before=self.main_paned)
        
    def create_top_controls(self):
        """Create filter controls and buttons."""
        # Create a frame for filters
        filter_frame = ttk.LabelFrame(self.top_frame, text="Filters")
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Strategy filter
        ttk.Label(filter_frame, text="Strategy:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.strategy_var = tk.StringVar(value="All")
        self.strategy_combo = ttk.Combobox(filter_frame, textvariable=self.strategy_var, width=20)
        self.strategy_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.strategy_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        # Risk filter
        ttk.Label(filter_frame, text="Risk:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.risk_var = tk.StringVar(value="All")
        self.risk_combo = ttk.Combobox(filter_frame, textvariable=self.risk_var, 
                                   values=["All", "LOW", "MEDIUM", "HIGH"], width=15)
        self.risk_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.risk_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        # Market regime filter
        ttk.Label(filter_frame, text="Regime:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.regime_var = tk.StringVar(value="All")
        self.regime_combo = ttk.Combobox(filter_frame, textvariable=self.regime_var, 
                                      values=["All"], width=20)
        self.regime_combo.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        self.regime_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        
        # Show aligned recommendations only
        self.aligned_var = tk.BooleanVar(value=False)
        aligned_check = ttk.Checkbutton(filter_frame, text="Show aligned only", 
                                     variable=self.aligned_var,
                                     command=self.apply_filters)
        aligned_check.grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        
        # Create a frame for buttons
        button_frame = ttk.Frame(self.top_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Refresh button
        self.refresh_btn = ttk.Button(button_frame, text="Refresh Data", 
                                  command=self.refresh_data, style="Accent.TButton")
        self.refresh_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Load tracker data button
        self.load_tracker_btn = ttk.Button(button_frame, text="Load Tracker Data", 
                                        command=self.load_tracker_data)
        self.load_tracker_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
    def create_recommendation_list(self):
        """Create the recommendation list treeview."""
        # Create a frame for the recommendation list
        list_frame = ttk.LabelFrame(self.left_frame, text="Trade Recommendations")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for the recommendations
        columns = ("Symbol", "Strategy", "Entry", "Stop", "Target", "R:R", "Risk")
        self.recommendation_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        
        # Configure column headings
        for col in columns:
            self.recommendation_tree.heading(col, text=col)
        
        # Configure column widths
        self.recommendation_tree.column("Symbol", width=80)
        self.recommendation_tree.column("Strategy", width=120)
        self.recommendation_tree.column("Entry", width=80)
        self.recommendation_tree.column("Stop", width=80)
        self.recommendation_tree.column("Target", width=80)
        self.recommendation_tree.column("R:R", width=50)
        self.recommendation_tree.column("Risk", width=70)
        
        # Configure tags for risk levels
        for tag, config in self.treeview_tags.items():
            self.recommendation_tree.tag_configure(tag.lower(), background=config["background"])
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.recommendation_tree.yview)
        self.recommendation_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.recommendation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.recommendation_tree.bind("<<TreeviewSelect>>", self.on_recommendation_select)
        
    def create_details_view(self):
        """Create the details view."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Entry/Exit tab
        self.entry_exit_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.entry_exit_frame, text="Entry/Exit")
        
        # Greeks tab (new)
        self.greeks_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.greeks_frame, text="Greeks")
        
        # Market Context tab
        self.market_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.market_frame, text="Market Context")
        
        # Regime Analysis tab
        self.regime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.regime_frame, text="Regime Analysis")
        
        # Trade Duration tab (new)
        self.duration_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.duration_frame, text="Trade Duration")
        
        # Initialize the details content
        self.initialize_details_content()
        
    def initialize_details_content(self):
        """Initialize the content for the details tabs."""
        # Summary tab
        summary_content = ttk.Frame(self.summary_frame)
        summary_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header section
        self.header_frame = ttk.Frame(summary_content)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(self.header_frame, text="Select a recommendation", 
                                   font=("Arial", 18, "bold"))
        self.title_label.pack(anchor=tk.W)
        
        self.subtitle_label = ttk.Label(self.header_frame, text="")
        self.subtitle_label.pack(anchor=tk.W)
        
        # Trade structure section
        self.trade_structure_frame = ttk.LabelFrame(summary_content, text="TRADE STRUCTURE")
        self.trade_structure_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.trade_structure_frame, text="Select a recommendation to view trade details").pack(
            anchor=tk.W, padx=10, pady=20)
        
        # Market alignment section (for regime alignment)
        self.alignment_frame = ttk.LabelFrame(summary_content, text="MARKET ALIGNMENT")
        self.alignment_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.alignment_frame, 
                text="Select a recommendation to view market regime alignment").pack(
            anchor=tk.W, padx=10, pady=20)
            
        # Create figure for charts
        chart_frame = ttk.LabelFrame(summary_content, text="Price Analysis")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Entry/Exit tab - split into two frames
        self.entry_frame = ttk.LabelFrame(self.entry_exit_frame, text="ENTRY CRITERIA")
        self.entry_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.entry_frame, text="Select a recommendation to view entry criteria").pack(pady=20)
        
        self.exit_frame = ttk.LabelFrame(self.entry_exit_frame, text="EXIT PLAN")
        self.exit_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.exit_frame, text="Select a recommendation to view exit criteria").pack(pady=20)
        
        # Greeks tab
        self.greeks_info_frame = ttk.LabelFrame(self.greeks_frame, text="Greek Values")
        self.greeks_info_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.greeks_info_frame, text="Select a recommendation to view Greek metrics").pack(pady=20)
        
        # Greek charts frame
        self.greeks_chart_frame = ttk.LabelFrame(self.greeks_frame, text="Greek Visualization")
        self.greeks_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(self.greeks_chart_frame, text="Greek metrics visualization will appear here").pack(pady=20)
        
        # Market Context tab
        self.market_info_frame = ttk.LabelFrame(self.market_frame, text="Current Market Regime")
        self.market_info_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.market_info_frame, text="Loading market regime data...").pack(pady=20)
        
        # Instrument context
        self.instrument_frame = ttk.LabelFrame(self.market_frame, text="Instrument Context")
        self.instrument_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize market regime visualization
        self.market_viz_frame = ttk.LabelFrame(self.regime_frame, text="Market Regime Calendar")
        self.market_viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(self.market_viz_frame, text="Loading regime visualization...").pack(pady=20)
        
        # Initialize trade duration view
        self.duration_info_frame = ttk.LabelFrame(self.duration_frame, text="Trade Duration Analysis")
        self.duration_info_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(self.duration_info_frame, text="Select a recommendation to view duration analysis").pack(pady=20)
        
        # Duration visualization
        self.duration_viz_frame = ttk.LabelFrame(self.duration_frame, text="Duration Performance")
        self.duration_viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(self.duration_viz_frame, text="Duration performance visualization will appear here").pack(pady=20)
        
    def create_status_bar(self):
        """Create status bar at the bottom of the window."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.status_bar, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Tracker status section
        tracker_frame = ttk.Frame(self.status_bar)
        tracker_frame.pack(side=tk.RIGHT)
        
        self.tracker_status = ttk.Label(tracker_frame, text="Tracker: Not loaded", foreground="red")
        self.tracker_status.pack(side=tk.LEFT, padx=5)
        
    def load_all_data(self):
        """Load recommendations and tracker data."""
        # Load trade recommendations
        self.load_recommendations()
        
        # Load market regime data
        self.load_market_regime_data()
        
        # Load energy levels and reset points
        self.load_energy_and_reset_data()
        
        # Load Greek metrics
        self.load_greek_data()
        
        # Load trade duration data
        self.load_trade_duration_data()
        
        # Update tracker status
        if self.market_regimes:
            self.tracker_status.config(text="Tracker: Data loaded", foreground="green")
        
        # Update market regime filter with available regimes
        self.update_regime_filter()
        
    def load_recommendations(self):
        """Load trade recommendations from JSON files."""
        self.recommendations = []
        
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            return
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Loading recommendations...")
        
        try:
            # Look for recommendation files
            for filename in os.listdir(self.results_dir):
                if filename.endswith("_trade_recommendation.json") or filename.endswith("_enhanced_recommendation.json"):
                    try:
                        file_path = os.path.join(self.results_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        # Process the recommendation data
                        if isinstance(data, list):
                            # Multiple recommendations in one file
                            for rec in data:
                                if self._is_valid_recommendation(rec):
                                    self.recommendations.append(rec)
                        elif isinstance(data, dict):
                            # Single recommendation
                            if self._is_valid_recommendation(data):
                                self.recommendations.append(data)
                        
                        logger.info(f"Loaded recommendations from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading recommendations from {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
        
        # Sort recommendations by date (newest first)
        self.recommendations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Loaded {len(self.recommendations)} recommendations")
        
        logger.info(f"Loaded {len(self.recommendations)} recommendations")
        
    def process_recommendation(self, data, file_path):
        """Process and standardize a recommendation."""
        try:
            # Extract symbol
            symbol = data.get("symbol", os.path.basename(file_path).split('_')[0].upper())
            
            # Extract action
            action = data.get("action", "BUY")
            
            # Extract strategy details
            strategy_name = "Unknown"
            if "strategy" in data:
                if isinstance(data["strategy"], dict):
                    strategy_name = data["strategy"].get("name", "Unknown")
                else:
                    strategy_name = data["strategy"]
            
            # Extract risk assessment
            risk_category = "MEDIUM"  # Default
            if "risk_assessment" in data:
                risk_category = data["risk_assessment"].get("risk_category", "MEDIUM")
            
            # Extract price data
            current_price = data.get("current_price", 0)
            
            # Extract entry criteria
            entry_criteria = data.get("entry_criteria", {})
            price_range = entry_criteria.get("price_range", [0, 0])
            if isinstance(price_range, dict):
                price_range = [price_range.get("low", 0), price_range.get("high", 0)]
            
            # Extract exit criteria
            exit_criteria = data.get("exit_criteria", {})
            profit_target = exit_criteria.get("profit_target_percent", 0)
            stop_loss = exit_criteria.get("max_loss_percent", 
                                       exit_criteria.get("stop_loss_percent", 0))
            days_to_hold = exit_criteria.get("days_to_hold", 0)
            
            # Extract ROI data if available
            roi = data.get("roi", 0)
            if not roi and "risk_management" in data:
                # Try to extract ROI from risk management section
                roi = data["risk_management"].get("expected_roi", 0)
            
            # Extract timestamp or use file modification time
            timestamp = data.get("timestamp", "")
            if not timestamp:
                mod_time = os.path.getmtime(file_path)
                timestamp = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            
            # Create standardized recommendation
            standardized = {
                "symbol": symbol,
                "action": action,
                "strategy_name": strategy_name,
                "risk_category": risk_category,
                "current_price": current_price,
                "entry_zone": price_range,
                "profit_target_percent": profit_target,
                "stop_loss_percent": stop_loss,
                "days_to_hold": days_to_hold,
                "roi": roi,
                "timestamp": timestamp,
                "file_path": file_path,
                "raw_data": data  # Store the original data for reference
            }
            
            return standardized
            
        except Exception as e:
            logger.error(f"Error processing recommendation: {e}")
            return None
            
    def load_market_regime_data(self):
        """Load market regime data from regime files."""
        self.market_regimes = {}
        
        # Look for market regime data
        regime_paths = [
            os.path.join(self.results_dir, "market_regime_summary.json"),
            os.path.join(self.results_dir, "market_bias.json"),
            os.path.join(self.results_dir, "market_regime", "current_regime.json"),
            os.path.join(self.results_dir, "regime_validation.json")
        ]
        
        # Try each potential path
        for path in regime_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        regime_data = json.load(f)
                        
                    logger.info(f"Loaded market regime data from {path}")
                    
                    # Process regime data based on format
                    if "market_regime" in regime_data:
                        self.market_regimes = regime_data["market_regime"]
                    elif "primary_regime" in regime_data:
                        self.market_regimes = regime_data
                    elif "regime_distribution" in regime_data:
                        self.market_regimes = {
                            "primary_label": list(regime_data["regime_distribution"].keys())[0],
                            "secondary_label": list(regime_data["regime_distribution"].keys())[1] if len(regime_data["regime_distribution"]) > 1 else "Unknown",
                            "volatility_regime": regime_data.get("volatility_regime", "Normal"),
                            "dominant_greek": regime_data.get("dominant_greek", "Unknown"),
                            "regime_distribution": regime_data["regime_distribution"]
                        }
                    else:
                        # Try using the whole data
                        self.market_regimes = regime_data
                        
                    # Update market summary
                    self.update_market_summary()
                    break
                except Exception as e:
                    logger.error(f"Error loading market regime data from {path}: {e}")
        
        # Check instrument-specific market regimes from analysis files
        self.instrument_regimes = {}  # Initialize the attribute
        
        for symbol in set([rec["symbol"] for rec in self.recommendations]):
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)
                    
                    # Extract market regime data for the instrument
                    if "greek_analysis" in analysis_data and "market_regime" in analysis_data["greek_analysis"]:
                        self.instrument_regimes[symbol] = analysis_data["greek_analysis"]["market_regime"]
                        logger.info(f"Loaded market regime data for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading analysis data for {symbol}: {e}")
        
        # Add energy state if available
        if "energy_state" in self.market_regimes:
            energy_state = "Unknown"
            if isinstance(self.market_regimes["energy_state"], dict):
                energy_state = self.market_regimes["energy_state"].get("state", "Unknown")
            else:
                energy_state = self.market_regimes["energy_state"]
            self.market_regimes["energy_state"] = energy_state
        
        # Update regime transitions
        self.update_regime_transitions()
        
    def update_regime_transitions(self):
        """Update regime transitions based on current market regimes."""
        if not self.market_regimes:
            return
        
        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        
        # Check if today's regime transition already exists
        if current_date in self.regime_transitions:
            return
        
        # Add today's regime transition
        self.regime_transitions[current_date] = {
            "date": current_date,
            "primary_regime": self.market_regimes.get("primary_label", "Unknown"),
            "secondary_regime": self.market_regimes.get("secondary_label", "Unknown"),
            "volatility": self.market_regimes.get("volatility_regime", "Normal"),
            "dominant_greek": self.market_regimes.get("dominant_greek", "Unknown")
        }
        
        # Save regime transitions to a file
        try:
            with open(os.path.join(self.results_dir, "regime_transitions.json"), "w") as f:
                json.dump(self.regime_transitions, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving regime transitions: {e}")
        
    def load_energy_and_reset_data(self):
        """Load energy levels and reset points from the results directory."""
        self.energy_levels = {}
        self.reset_points = {}
        
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found: {self.results_dir}", "error")
            return
        
        # Update status
        self.update_status("Loading energy levels and reset points...")
        
        try:
            # Check for energy and reset data in analysis files
            for symbol in set([rec["symbol"] for rec in self.recommendations]):
                analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
                if os.path.exists(analysis_path):
                    try:
                        with open(analysis_path, "r") as f:
                            analysis_data = json.load(f)
                            
                        # Extract energy levels
                        if "greek_analysis" in analysis_data:
                            greek_analysis = analysis_data["greek_analysis"]
                            
                            # Extract energy levels
                            if "energy_levels" in greek_analysis:
                                self.energy_levels[symbol] = greek_analysis["energy_levels"]
                                
                            # Extract reset points
                            if "reset_points" in greek_analysis:
                                self.reset_points[symbol] = greek_analysis["reset_points"]
                                
                            logger.info(f"Loaded energy and reset data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading energy/reset data for {symbol}: {e}")
            
            # Check if we found any data
            logger.info(f"Loaded energy levels for {len(self.energy_levels)} instruments")
            logger.info(f"Loaded reset points for {len(self.reset_points)} instruments")
            
            # Update status
            self.update_status("Loaded energy and reset data")
        except Exception as e:
            logger.error(f"Error loading energy and reset data: {e}")
            self.update_status(f"Error loading energy and reset data: {e}", "error")
            
    def load_greek_data(self):
        """Load Greek metrics from the results directory."""
        self.greek_data = {}
        
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found: {self.results_dir}", "error")
            return
        
        # Update status
        self.update_status("Loading Greek metrics...")
        
        try:
            for filename in os.listdir(self.results_dir):
                if "_greeks" in filename and filename.endswith(".json"):
                    try:
                        file_path = os.path.join(self.results_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        symbol = data.get("symbol", os.path.basename(file_path).split('_')[0].upper())
                        self.greek_data[symbol] = data
                    except Exception as e:
                        logger.error(f"Error loading Greek metrics from {filename}: {e}")
            
            # Check for Greek data in analysis files
            for symbol in set([rec["symbol"] for rec in self.recommendations]):
                analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
                if os.path.exists(analysis_path):
                    try:
                        with open(analysis_path, "r") as f:
                            analysis_data = json.load(f)
                        
                        # Extract Greek metrics
                        if "greek_analysis" in analysis_data:
                            greek_analysis = analysis_data["greek_analysis"]
                            
                            # Check if Greek data is available
                            if "magnitudes" in greek_analysis:
                                self.greek_data[symbol] = greek_analysis
                                logger.info(f"Loaded Greek data for {symbol}")
                            else:
                                logger.info(f"No Greek data available for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading Greek data for {symbol}: {e}")
            
            # Update status
            self.update_status("Loaded Greek metrics")
            logger.info("Loaded Greek metrics")
        
        except Exception as e:
            logger.error(f"Error loading Greek metrics: {e}")
            self.update_status(f"Error loading Greek metrics: {e}", "error")
            
    def load_trade_duration_data(self):
        """Load trade duration data from the results directory."""
        self.trade_durations = {}
        
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found: {self.results_dir}", "error")
            return
        
        # Update status
        self.update_status("Loading trade duration data...")
        
        try:
            for filename in os.listdir(self.results_dir):
                if "_trade_duration" in filename and filename.endswith(".json"):
                    try:
                        file_path = os.path.join(self.results_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        strategy_name = data.get("strategy_name", os.path.basename(file_path).split('_')[0])
                        self.trade_durations[strategy_name] = data
                    except Exception as e:
                        logger.error(f"Error loading trade duration data from {filename}: {e}")
            
            # Update status
            self.update_status("Loaded trade duration data")
            logger.info("Loaded trade duration data")
        
        except Exception as e:
            logger.error(f"Error loading trade duration data: {e}")
            self.update_status(f"Error loading trade duration data: {e}", "error")
            
    def update_filter_options(self):
        """Update filter options based on loaded recommendations."""
        strategies = set()
        risks = set()
        
        for rec in self.recommendations:
            strategies.add(rec["strategy_name"])
            risks.add(rec["risk_category"])
        
        # Update strategy filter
        self.strategy_combo['values'] = ["All"] + sorted(strategies)
        self.strategy_var.set("All")
        
        # Update risk filter
        self.risk_combo['values'] = ["All"] + sorted(risks)
        self.risk_var.set("All")
        
        # Update regime filter
        self.update_regime_filter()
        
    def update_regime_filter(self):
        """Update the regime filter options based on available regimes."""
        regimes = set()
        
        for rec in self.recommendations:
            if "raw_data" in rec:
                raw_data = rec["raw_data"]
                if "market_regime" in raw_data:
                    regimes.add(raw_data["market_regime"].get("primary_label", "Unknown"))
        
        # Add current market regime if available
        if self.market_regimes:
            regimes.add(self.market_regimes.get("primary_label", "Unknown"))
        
        # Update regime filter
        self.regime_combo['values'] = ["All"] + sorted(regimes)
        self.regime_var.set("All")
        
    def apply_filters(self):
        """Apply the current filters to the recommendation list."""
        filtered_recommendations = self.recommendations[:]
        
        # Apply strategy filter
        if self.strategy_var.get() != "All":
            filtered_recommendations = [rec for rec in filtered_recommendations 
                                       if rec["strategy_name"] == self.strategy_var.get()]
        
        # Apply risk filter
        if self.risk_var.get() != "All":
            filtered_recommendations = [rec for rec in filtered_recommendations 
                                       if rec["risk_category"] == self.risk_var.get()]
        
        # Apply regime filter
        if self.regime_var.get() != "All":
            filtered_recommendations = [rec for rec in filtered_recommendations 
                                       if "raw_data" in rec and "market_regime" in rec["raw_data"] and 
                                       rec["raw_data"]["market_regime"].get("primary_label", "Unknown") == self.regime_var.get()]
        
        # Apply aligned filter
        if self.aligned_var.get():
            filtered_recommendations = [rec for rec in filtered_recommendations 
                                       if "raw_data" in rec and "market_regime" in rec["raw_data"] and 
                                       rec["raw_data"]["market_regime"].get("primary_label", "Unknown") == self.market_regimes.get("primary_label", "Unknown")]
        
        # Update the treeview with filtered recommendations
        self.update_treeview(filtered_recommendations)
        
    def update_treeview(self, recommendations):
        """Update the treeview with the given recommendations."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for rec in recommendations:
            tag = rec["risk_category"]
            if rec.get("raw_data", {}).get("market_regime", {}).get("primary_label", "") == self.market_regimes.get("primary_label", ""):
                tag += ";aligned"
            self.tree.insert("", "end", values=(
                rec["symbol"],
                rec["action"],
                rec["strategy_name"],
                f"${rec['entry_zone'][0]:.2f} - ${rec['entry_zone'][1]:.2f}",
                f"{rec['profit_target_percent']:.1f}%",
                f"{rec['stop_loss_percent']:.1f}%",
                rec["days_to_hold"],
                rec["risk_category"]
            ), tags=tag)
        
        # Update summary labels
        self.update_summary_labels(recommendations)
        
    def update_summary_labels(self, recommendations):
        """Update the summary labels with information from the recommendations."""
        if not recommendations:
            for label in self.summary_labels.values():
                label.config(text="...")
            return
        
        # Get the first recommendation for summary
        rec = recommendations[0]
        raw_data = rec.get("raw_data", {})
        
        # Update current regime
        self.summary_labels["Current Regime"].config(text=self.market_regimes.get("primary_label", "Unknown"))
        
        # Update dominant Greek
        self.summary_labels["Dominant Greek"].config(text=self.market_regimes.get("dominant_greek", "Unknown"))
        
        # Update energy state
        self.summary_labels["Energy State"].config(text=self.market_regimes.get("energy_state", "Unknown"))
        
        # Update regime transitions
        regime_transitions = self.regime_transitions
        if regime_transitions:
            transition_count = len(regime_transitions)
            last_transition = regime_transitions[list(regime_transitions.keys())[-1]]
            transition_info = f"{transition_count} transitions, last on {last_transition['date']}"
        else:
            transition_info = "No transitions recorded"
        self.summary_labels["Regime Transitions"].config(text=transition_info)
        
        # Update aligned trades
        aligned_count = sum(1 for rec in recommendations if "raw_data" in rec and "market_regime" in rec["raw_data"] and 
                           rec["raw_data"]["market_regime"].get("primary_label", "") == self.market_regimes.get("primary_label", ""))
        self.summary_labels["Aligned Trades"].config(text=f"{aligned_count}/{len(recommendations)}")
        
    def on_recommendation_select(self, event):
        """Handle recommendation selection event."""
        try:
            selected_item = self.tree.selection()[0]
            rec = self.tree.item(selected_item, "values")
            rec = {
                "symbol": rec[0],
                "action": rec[1],
                "strategy_name": rec[2],
                "entry_zone": [float(rec[3].split(" - ")[0]), float(rec[3].split(" - ")[1])],
                "profit_target_percent": float(rec[4].strip("%")),
                "stop_loss_percent": float(rec[5].strip("%")),
                "days_to_hold": int(rec[6]),
                "risk_category": rec[7],
                "raw_data": self.get_raw_data(rec[0])
            }
            self.selected_recommendation = rec
            self.update_details_view(rec)
        except Exception as e:
            logger.error(f"Error handling recommendation selection: {e}")
            self.update_status(f"Error handling recommendation selection: {e}", "error")
            
    def get_raw_data(self, symbol):
        """Retrieve raw data for a given symbol."""
        for rec in self.recommendations:
            if rec.get("symbol") == symbol:
                return rec.get("raw_data", {})
        return {}
            
    def update_details_view(self, rec):
        """Update the details view with information from the selected recommendation."""
        try:
            # Update header
            self.title_label.config(text=f"{rec['symbol']} - {rec['action']} {rec['strategy_name']}")
            self.subtitle_label.config(text=f"Current Price: ${rec['current_price']:.2f}")
            
            # Update trade structure
            self.update_trade_structure(rec)
            
            # Update market alignment
            self.update_market_alignment(rec)
            
            # Update price chart
            self.update_chart(rec)
            
            # Update entry/exit criteria
            self.update_entry_exit_criteria(rec)
            
            # Update Greeks tab
            self.update_greeks_view(rec)
            
            # Update Market Context tab
            self.update_market_context(rec)
            
            # Update Regime Analysis tab
            self.update_regime_analysis(rec)
            
            # Update Trade Duration tab
            self.update_duration_view(rec)
            
        except Exception as e:
            logger.error(f"Error updating details: {e}")
            self.update_status(f"Error updating details: {str(e)}", "error")
            
    def update_trade_structure(self, rec):
        """Update the trade structure section with information from the selected recommendation."""
        try:
            # Clear existing widgets
            for widget in self.trade_structure_frame.winfo_children():
                widget.destroy()
            
            # Create labels for trade structure
            ttk.Label(self.trade_structure_frame, text="Action:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=rec['action']).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Strategy:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=rec['strategy_name']).grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Entry Zone:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=f"${rec['entry_zone'][0]:.2f} - ${rec['entry_zone'][1]:.2f}").grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Target %:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=f"{rec['profit_target_percent']:.1f}%").grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Stop %:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=f"{rec['stop_loss_percent']:.1f}%").grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Days to Hold:").grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=str(rec['days_to_hold'])).grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.trade_structure_frame, text="Risk Category:").grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.trade_structure_frame, text=rec['risk_category']).grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
            
        except Exception as e:
            logger.error(f"Error updating trade structure: {e}")
            self.update_status(f"Error updating trade structure: {str(e)}", "error")
            
    def update_market_alignment(self, rec):
        """Update the market alignment section with information from the selected recommendation."""
        try:
            # Clear existing widgets
            for widget in self.alignment_frame.winfo_children():
                widget.destroy()
            
            # Get market regime data
            raw_data = rec.get("raw_data", {})
            market_regime = raw_data.get("market_regime", {})
            
            # Create labels for market alignment
            ttk.Label(self.alignment_frame, text="Primary Regime:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.alignment_frame, text=market_regime.get("primary_label", "Unknown")).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.alignment_frame, text="Secondary Regime:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.alignment_frame, text=market_regime.get("secondary_label", "Unknown")).grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.alignment_frame, text="Volatility Regime:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.alignment_frame, text=market_regime.get("volatility_regime", "Normal")).grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(self.alignment_frame, text="Dominant Greek:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(self.alignment_frame, text=market_regime.get("dominant_greek", "Unknown")).grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
            
            # Add energy state if available
            if "energy_state" in self.market_regimes:
                energy_state = "Unknown"
                if isinstance(self.market_regimes["energy_state"], dict):
                    energy_state = self.market_regimes["energy_state"].get("state", "Unknown")
                else:
                    energy_state = self.market_regimes["energy_state"]
                
                ttk.Label(self.alignment_frame, text="Energy State:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
                ttk.Label(self.alignment_frame, text=energy_state).grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
            
        except Exception as e:
            logger.error(f"Error updating market alignment: {e}")
            self.update_status(f"Error updating market alignment: {str(e)}", "error")
            
    def update_entry_exit_criteria(self, rec):
        """Update the entry/exit criteria section with information from the selected recommendation."""
        try:
            # Clear existing widgets
            for widget in self.entry_frame.winfo_children():
                widget.destroy()
            for widget in self.exit_frame.winfo_children():
                    text=f"Error creating duration visualization: {str(e)}").pack(anchor=tk.W, padx=10, pady=10)
            
    def update_status(self, message, status_type="info"):
        """Update the status bar with a message."""
        try:
            if hasattr(self, 'status_label'):
                # Set color based on status type
                color = {
                    "info": "black",
                    "success": "green",
                    "warning": "orange",
                    "error": "red"
                }.get(status_type, "black")
                
                # Update status label
                self.status_label.config(text=message, foreground=color)
                
                # Log the status message
                if status_type == "error":
                    logger.error(message)
                elif status_type == "warning":
                    logger.warning(message)
                else:
                    logger.info(message)
        except Exception as e:
            logger.error(f"Error updating status: {e}")
                    text=f"Error creating duration visualization: {str(e)}").pack(anchor=tk.W, padx=10, pady=10)
            
    def clear_detail_frames(self):
        """Clear widgets from detail frames for updating."""
        for frame in [self.entry_frame, self.exit_frame, self.greeks_info_frame, 
                   self.greeks_chart_frame, self.market_info_frame, self.instrument_frame, 
                   self.duration_info_frame, self.duration_viz_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
                
    def update_chart(self, rec):
        """Update the chart with price levels, energy levels, and reset points."""
        try:
            self.fig.clear()
            
            # Create GridSpec for multiple subplots
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 1, height_ratios=[3, 1])
            
            # Price chart
            ax1 = self.fig.add_subplot(gs[0])
            
            # Get current price and entry range
            symbol = rec.get('symbol', '')
            current_price = rec.get('current_price', 0)
            entry_range = rec.get('entry_zone', [current_price * 0.95, current_price * 1.05])
            
            # Create x-axis data (30 days)
            import numpy as np
            days = np.arange(-15, 16)
            
            # Simulate a price history with the current price in the middle
            # Use a more realistic simulation for the selected symbol
            if symbol in ["AAPL", "MSFT", "GOOGL"]:  # Tech stocks - more volatile
                volatility = 0.02
            elif symbol in ["SPY", "QQQ"]:  # Index ETFs - less volatile
                volatility = 0.01
            else:
                volatility = 0.015
                
            # Set a random seed based on symbol for consistency
            np.random.seed(sum(ord(c) for c in symbol))
            
            # Generate a semi-realistic price path
            price_history = []
            price = current_price
            for i in range(len(days)):
                if i == 15:  # Current day
                    price = current_price
                elif i < 15:  # Past days
                    change = np.random.normal(0, volatility)
                    price = price / (1 + change)  # Work backwards
                else:  # Future days
                    change = np.random.normal(0, volatility)
                    price = price * (1 + change)
                    
                price_history.append(price)
                
            price_history.reverse()  # Correct the order
            
            # Plot simulated price history
            ax1.plot(days, price_history, 'b-', linewidth=2, label='Price History')
            
            # Plot current price line
            ax1.axhline(y=current_price, color='r', linestyle='-', linewidth=1.5, label='Current Price')
            
            # Plot entry range
            ax1.axhspan(entry_range[0], entry_range[1], alpha=0.2, color='green', label='Entry Range')
            
                            "primary_label": list(regime_data["regime_distribution"].keys())[0],
                            "regime_distribution": regime_data["regime_distribution"]
                        }
                    else:
                        # Try using the whole data
                        self.market_regimes = regime_data
                        
                    # Update market summary
                    self.update_market_summary()
                    break
                        
                except Exception as e:
                    logger.error(f"Error loading market regime data from {path}: {e}")
        
        # Check instrument-specific market regimes from analysis files
        self.instrument_regimes = {}  # Initialize the attribute
        
        for symbol in set([rec["symbol"] for rec in self.recommendations]):
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)
                    
                    # Extract market regime data for the instrument
                    if "greek_analysis" in analysis_data and "market_regime" in analysis_data["greek_analysis"]:
                        self.instrument_regimes[symbol] = analysis_data["greek_analysis"]["market_regime"]
                        logger.info(f"Loaded market regime data for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading analysis data for {symbol}: {e}")
        
    def load_energy_and_reset_data(self):
        """Load energy levels and reset points from analysis files."""
        self.energy_levels = {}
        self.reset_points = {}
        
        # Check for instrument-specific data from analysis files
        for symbol in set([rec["symbol"] for rec in self.recommendations]):
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)
                    
                    # Extract energy levels
                    if "greek_analysis" in analysis_data:
                        greek_analysis = analysis_data["greek_analysis"]
                        
                        # Extract energy levels
                        if "energy_levels" in greek_analysis:
                            self.energy_levels[symbol] = greek_analysis["energy_levels"]
                            
                        # Extract reset points
                        if "reset_points" in greek_analysis:
                            self.reset_points[symbol] = greek_analysis["reset_points"]
                            
                        logger.info(f"Loaded energy and reset data for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading energy/reset data for {symbol}: {e}")
        
        # Check if we found any data
        logger.info(f"Loaded energy levels for {len(self.energy_levels)} instruments")
        logger.info(f"Loaded reset points for {len(self.reset_points)} instruments")
    
    def load_greek_data(self):
        """Load Greek metrics data from analysis files."""
        self.greek_data = {}
        
        # Check for Greek data in analysis files
        for symbol in set([rec["symbol"] for rec in self.recommendations]):
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)
                    
                    # Extract Greek metrics
                    if "greek_analysis" in analysis_data:
                        greek_analysis = analysis_data["greek_analysis"]
                        
                        # Check if Greek data is available
                        if "magnitudes" in greek_analysis:
                            self.greek_data[symbol] = greek_analysis
                            logger.info(f"Loaded Greek data for {symbol}")
                        else:
                            logger.info(f"No Greek data available for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading Greek data for {symbol}: {e}")
    
    def load_trade_duration_data(self):
        """Load or simulate trade duration data."""
        self.trade_durations = {}
        
        # For each strategy, create simulated duration metrics
        strategies = set([rec.get("strategy_name", "Unknown") for rec in self.recommendations])
        
        for strategy in strategies:
            # Simulate duration metrics based on strategy type
            if "VOLATILITY" in strategy:
                # Volatility strategies tend to be shorter
                avg_days = 3.5
                optimal_days = 4
                success_rate = 0.65
            elif "CALENDAR" in strategy:
                # Calendar spreads tend to be longer
                avg_days = 12.4
                optimal_days = 14
                success_rate = 0.72
            elif "VANNA" in strategy:
                # Vanna strategies medium duration
                avg_days = 8.2
                optimal_days = 7
                success_rate = 0.68
            else:
                # Default values
                avg_days = 6.5
                optimal_days = 7
                success_rate = 0.64
            
            # Create duration profile
            self.trade_durations[strategy] = {
                "avg_days": avg_days,
                "optimal_days": optimal_days,
                "success_rate": success_rate,
                "completion_profile": {
                    "day_1": 0.15,  # 15% of profit on day 1
                    "day_2": 0.35,  # 35% of profit on day 2
                    "day_3": 0.60,  # 60% of profit on day 3
                    "day_4": 0.85,  # 85% of profit on day 4
                    "day_5": 0.95   # 95% of profit on day 5
                }
            }
        
        logger.info(f"Created duration profiles for {len(strategies)} strategies")
        
        # Try to load real trade performance data if available
        performance_file = os.path.join(self.results_dir, "trade_performance.json")
        if os.path.exists(performance_file):
            try:
                with open(performance_file, "r") as f:
                    performance_data = json.load(f)
                
                # Process performance data if available
                if "trades" in performance_data:
                    for trade in performance_data["trades"]:
                        strategy = trade.get("strategy", "Unknown")
                        duration = trade.get("duration_days", 0)
                        success = trade.get("success", False)
                        
                        # Update duration statistics
                        if strategy in self.trade_durations:
                            # Update with real data
                            if "real_trades" not in self.trade_durations[strategy]:
                                self.trade_durations[strategy]["real_trades"] = []
                            
                            self.trade_durations[strategy]["real_trades"].append({
                                "duration": duration,
                                "success": success
                            })
                
                logger.info(f"Loaded trade performance data from {performance_file}")
            except Exception as e:
                logger.error(f"Error loading trade performance data: {e}")
            
    def load_tracker_data(self):
        """Load or refresh tracker data."""
        # Show a directory dialog to select tracker data
        tracker_dir = filedialog.askdirectory(
            title="Select Tracker Data Directory",
            initialdir=os.path.join(self.base_dir, "data", "tracker")
        )
        
        if not tracker_dir:
            return
            
        try:
            # Look for regime history file
            history_file = os.path.join(tracker_dir, "regime_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    self.regime_history = json.load(f)
                logger.info(f"Loaded regime history for {len(self.regime_history)} instruments")
                
                # Extract regime transitions
                self.regime_transitions = {}
                for symbol, history in self.regime_history.items():
                    transitions = []
                    dates = sorted(history.keys())
                    
                    if len(dates) >= 2:
                        for i in range(1, len(dates)):
                            prev_date = dates[i-1]
                            curr_date = dates[i]
                            
                            prev_regime = history[prev_date]["primary_regime"]
                            curr_regime = history[curr_date]["primary_regime"]
                            
                            if prev_regime != curr_regime:
                                transitions.append({
                                    "from_date": prev_date,
                                    "to_date": curr_date,
                                    "from_regime": prev_regime,
                                    "to_regime": curr_regime
                                })
                    
                    self.regime_transitions[symbol] = transitions
                
                # Update market regime visualization
                self.create_regime_visualization()
                
                # Update tracker status
                self.tracker_status.config(text="Tracker: Data loaded", foreground="green")
                
                # Update status
                self.update_status(f"Loaded tracker data from {tracker_dir}")
                
            else:
                self.update_status(f"No regime history found in {tracker_dir}", "warning")
                
        except Exception as e:
            logger.error(f"Error loading tracker data: {e}")
            self.update_status(f"Error loading tracker data: {e}", "error")
            
    def update_filter_options(self):
        """Update filter dropdown options based on loaded data."""
        # Get unique strategies
        strategies = ["All"]
        for rec in self.recommendations:
            strategy = rec.get("strategy_name", "Unknown")
            if strategy not in strategies:
                strategies.append(strategy)
                
        # Update strategy combobox
        self.strategy_combo['values'] = strategies
        
    def update_regime_filter(self):
        """Update market regime filter with available regimes."""
        regimes = ["All"]
        
        # Add general market regimes
        if self.market_regimes:
            if "primary_label" in self.market_regimes:
                regimes.append(self.market_regimes["primary_label"])
                
            if "secondary_label" in self.market_regimes:
                regimes.append(self.market_regimes["secondary_label"])
                
            if "regime_distribution" in self.market_regimes:
                for regime in self.market_regimes["regime_distribution"].keys():
                    if regime not in regimes:
                        regimes.append(regime)
        
        # Add instrument-specific regimes if available
        if hasattr(self, "instrument_regimes") and self.instrument_regimes:
            for regime_data in self.instrument_regimes.values():
                if isinstance(regime_data, dict) and "primary_label" in regime_data and regime_data["primary_label"] not in regimes:
                    regimes.append(regime_data["primary_label"])
                    
        # Update regime combobox
        self.regime_combo['values'] = regimes
        
    def update_market_summary(self):
        """Update the market summary display with current regime information."""
        try:
            # Update market regime labels if they exist
            if hasattr(self, 'summary_labels') and self.summary_labels:
                if "Current Regime" in self.summary_labels:
                    self.summary_labels["Current Regime"].config(
                        text=self.market_regimes.get("primary_label", "Unknown"))
                
                if "Dominant Greek" in self.summary_labels:
                    self.summary_labels["Dominant Greek"].config(
                        text=self.market_regimes.get("dominant_greek", "Unknown"))
                
                if "Energy State" in self.summary_labels:
                    self.summary_labels["Energy State"].config(
                        text=self.market_regimes.get("energy_state", "Unknown"))
        except Exception as e:
            logger.error(f"Error updating market summary: {e}")
        
    def populate_dashboard(self):
        """Populate the dashboard with recommendations."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Apply any active filters
        filtered_recs = self.apply_filters(update_ui=False)
        
        # Add recommendations to the tree
        for rec in filtered_recs:
            # Get values
            symbol = rec.get("symbol", "Unknown")
            action = rec.get("action", "Unknown")
            strategy = rec.get("strategy_name", "Unknown")
            
            # Get entry zone for display
            entry_zone = rec.get('entry_zone', [0, 0])
            entry_zone_str = f"${entry_zone[0]:.1f}-${entry_zone[1]:.1f}"
            
            # Get target and stop percentages
            target_pct = f"{rec.get('profit_target_percent', 0):.1f}%"
            stop_pct = f"{rec.get('stop_loss_percent', 0):.1f}%"
            
            # Get days to hold
            days = str(rec.get('days_to_hold', '-'))
            
            # Get risk category
            risk = rec.get("risk_category", "Unknown")
            
            # Determine tags to apply
            tags = [risk]
            
            # Add aligned tag if recommendation aligns with market regime
            if self.is_aligned_with_market_regime(rec):
                tags.append("aligned")
            
            # Add to tree with appropriate tags
            self.tree.insert("", tk.END, values=(
                symbol, action, strategy, entry_zone_str, target_pct, stop_pct, days, risk
            ), tags=tags)
        
        # Select the first item if available
        if self.tree.get_children():
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
            self.on_recommendation_select(None)  # Trigger selection event
        else:
            # No recommendations available
            self.update_status("No recommendations found matching the current filters")
            
        logger.info(f"Populated dashboard with {len(filtered_recs)} recommendations")
        
    def apply_filters(self, update_ui=True):
        """Apply filters and return filtered recommendations."""
        filtered = self.recommendations.copy()
        
        # Apply strategy filter
        if self.strategy_var.get() != "All":
            filtered = [rec for rec in filtered 
                      if rec.get("strategy_name", "Unknown") == self.strategy_var.get()]
        
        # Apply risk filter
        if self.risk_var.get() != "All":
            filtered = [rec for rec in filtered 
                      if rec.get("risk_category", "Unknown") == self.risk_var.get()]
        
        # Apply regime filter
        if self.regime_var.get() != "All":
            # Filter based on alignment with specific regime
            filtered = [rec for rec in filtered 
                      if self.is_aligned_with_specific_regime(rec, self.regime_var.get())]
        
        # Apply aligned filter
        if self.aligned_var.get():
            filtered = [rec for rec in filtered 
                      if self.is_aligned_with_market_regime(rec)]
        
        # Update UI if requested
        if update_ui:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Add filtered recommendations
            for rec in filtered:
                symbol = rec.get("symbol", "Unknown")
                action = rec.get("action", "Unknown")
                strategy = rec.get("strategy_name", "Unknown")
                
                entry_zone = rec.get('entry_zone', [0, 0])
                entry_zone_str = f"${entry_zone[0]:.1f}-${entry_zone[1]:.1f}"
                
                target_pct = f"{rec.get('profit_target_percent', 0):.1f}%"
                stop_pct = f"{rec.get('stop_loss_percent', 0):.1f}%"
                
                days = str(rec.get('days_to_hold', '-'))
                risk = rec.get("risk_category", "Unknown")
                
                # Determine tags to apply
                tags = [risk]
                
                # Add aligned tag if recommendation aligns with market regime
                if self.is_aligned_with_market_regime(rec):
                    tags.append("aligned")
                
                self.tree.insert("", tk.END, values=(
                    symbol, action, strategy, entry_zone_str, target_pct, stop_pct, days, risk
                ), tags=tags)
            
            # Update status
            self.update_status(f"Found {len(filtered)} recommendations matching filters")
        
        return filtered
        
    def is_aligned_with_market_regime(self, rec):
        """Check if recommendation aligns with current market regime."""
        symbol = rec.get("symbol", "")
        strategy = rec.get("strategy_name", "")
        
        # Default alignment check
        if not self.market_regimes:
            return False
            
        # Get current market regime
        current_regime = self.market_regimes.get("primary_label", "Unknown")
        
        # Check instrument-specific regime if available
        instrument_regime = None
        if hasattr(self, "instrument_regimes") and symbol in self.instrument_regimes:
            instrument_regime = self.instrument_regimes[symbol].get("primary_label", None)
            
        # Check if instrument is in current regime
        if instrument_regime and current_regime in instrument_regime:
            return True
            
        # Check recommendation's raw data for regime info
        if "raw_data" in rec and "market_regime" in rec["raw_data"]:
            rec_regime = rec["raw_data"]["market_regime"].get("primary_label", "Unknown")
            return rec_regime == current_regime
            
        return False
        
    def is_aligned_with_specific_regime(self, rec, regime):
        """Check if recommendation aligns with a specific regime."""
        symbol = rec.get("symbol", "")
        strategy = rec.get("strategy_name", "")
        
        # Check for instrument-specific regime if available
        instrument_regime = None
        if hasattr(self, "instrument_regimes") and symbol in self.instrument_regimes:
            instrument_regime = self.instrument_regimes[symbol].get("primary_label", None)
            
        # Check if instrument is in specified regime
        if instrument_regime and regime in instrument_regime:
            return True
            
        # Check recommendation's raw data for regime info
        if "raw_data" in rec and "market_regime" in rec["raw_data"]:
            rec_regime = rec["raw_data"]["market_regime"].get("primary_label", "Unknown")
            return rec_regime == regime
            
        return False
        
    def on_recommendation_select(self, event):
        """Handle selection of a recommendation from the list."""
        # Get selected item
        if not hasattr(self, 'recommendation_tree'):
            return
        
        selection = self.recommendation_tree.selection()
        if not selection:
            return
        
        # Get the selected recommendation
        item = selection[0]
        values = self.recommendation_tree.item(item, "values")
        
        if not values or len(values) < 2:
            return
        
        symbol = values[0]
        strategy = values[1]
        
        # Find the corresponding recommendation
        selected_rec = None
        for rec in self.recommendations:
            if rec.get("symbol") == symbol and rec.get("strategy_name") == strategy:
                selected_rec = rec
                break
        
        if not selected_rec:
            return
        
        # Store the selected recommendation
        self.selected_recommendation = selected_rec
        
        # Update the details view
        self.update_details_view(selected_rec)
        
        # Update status
        self.update_status(f"Selected {symbol} {strategy} recommendation")
    
    def update_details(self, rec):
        """Update the details view with the selected recommendation."""
        try:
            # Clear existing widgets
            self.clear_detail_frames()
            
            # SUMMARY TAB
            # Update header with larger, bold font
            self.title_label.config(text=f"{rec.get('symbol')} - {rec.get('strategy_name')}", 
                                  font=("Arial", 18, "bold"))
            self.subtitle_label.config(
                text=f"Current Price: ${rec.get('current_price', 0):.2f}   Date: {rec.get('timestamp', 'Unknown').split()[0]}",
                font=("Arial", 12))
            
            # Create TRADE STRUCTURE section
            for widget in self.trade_structure_frame.winfo_children():
                widget.destroy()
                
            # Add action line with bold green text
            action_text = f"Action: {rec.get('action', 'BUY')} {rec.get('symbol', 'Unknown')}"
            action_label = ttk.Label(self.trade_structure_frame, text=action_text, 
                               font=("Arial", 12, "bold"), foreground="#006400")
            action_label.pack(anchor=tk.W, padx=10, pady=5)
            
            # Add strategy line
            strategy_text = f"Strategy: {rec.get('strategy_name', 'Unknown')}"
            ttk.Label(self.trade_structure_frame, text=strategy_text, 
                    font=("Arial", 11)).pack(anchor=tk.W, padx=10, pady=5)
            
            # Add ROI if available
            roi = rec.get("roi", 0)
            if roi:
                roi_text = f"Expected ROI: {roi:.1f}%"
                ttk.Label(self.trade_structure_frame, text=roi_text, 
                        font=("Arial", 11, "bold"), foreground="#006400").pack(anchor=tk.W, padx=10, pady=5)
            
            # Add position sizing if available
            raw_data = rec.get("raw_data", {})
            if "implementation" in raw_data and "position_size" in raw_data["implementation"]:
                position_size = raw_data["implementation"]["position_size"]
                position_text = f"Position Size: {position_size.get('contracts', 1)} contracts ({position_size.get('account_percentage', 5)}% of account)"
                ttk.Label(self.trade_structure_frame, text=position_text, 
                        font=("Arial", 11)).pack(anchor=tk.W, padx=10, pady=5)
            
            # Update market alignment section
            for widget in self.alignment_frame.winfo_children():
                widget.destroy()
                
            # Check alignment with market regime
            is_aligned = self.is_aligned_with_market_regime(rec)
            
            # Get current market regime
            current_regime = "Unknown"
            dominant_greek = "Unknown"
            
            if self.market_regimes:
                current_regime = self.market_regimes.get("primary_label", "Unknown")
                dominant_greek = self.market_regimes.get("dominant_greek", "Unknown")
            
            # Check for instrument-specific regime
            instrument_regime = None
            symbol = rec.get("symbol", "")
            
            if hasattr(self, "instrument_regimes") and symbol in self.instrument_regimes:
                instrument_regime = self.instrument_regimes[symbol].get("primary_label", "Unknown")
            
            # Display alignment information
            alignment_color = "green" if is_aligned else "red"
            alignment_text = "Aligned with" if is_aligned else "Not aligned with"
            
            # Create header for alignment
            ttk.Label(self.alignment_frame, text="Market Regime Alignment:", 
                    font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
            
            # Show regime information
            regime_text = f"{alignment_text} current market regime:"
            ttk.Label(self.alignment_frame, text=regime_text, 
                    foreground=alignment_color).pack(anchor=tk.W, padx=10, pady=2)
            
            if instrument_regime:
                ttk.Label(self.alignment_frame, 
                        text=f"• Instrument Regime: {instrument_regime}").pack(anchor=tk.W, padx=20, pady=2)
            
            ttk.Label(self.alignment_frame, 
                    text=f"• Market Regime: {current_regime}").pack(anchor=tk.W, padx=20, pady=2)
                    
            ttk.Label(self.alignment_frame, 
                    text=f"• Dominant Greek: {dominant_greek}").pack(anchor=tk.W, padx=20, pady=2)
            
            # Add energy levels and reset points if available
            if symbol in self.energy_levels or symbol in self.reset_points:
                ttk.Label(self.alignment_frame, text="Key Price Levels:", 
                        font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
                
                # Add energy levels
                if symbol in self.energy_levels and self.energy_levels[symbol]:
                    energy_levels = sorted(self.energy_levels[symbol], 
                                         key=lambda x: abs(x.get("price", 0) - rec.get("current_price", 0)))[:3]
                    
                    for i, level in enumerate(energy_levels):
                        price = level.get("price", 0)
                        strength = level.get("strength", 0)
                        distance = abs(price - rec.get("current_price", 0)) / rec.get("current_price", 0) * 100
                        
                        level_text = f"• Energy Level: ${price:.2f} (Strength: {strength:.1f}, {distance:.1f}% away)"
                        ttk.Label(self.alignment_frame, text=level_text).pack(anchor=tk.W, padx=20, pady=2)
                
                # Add reset points
                if symbol in self.reset_points and self.reset_points[symbol]:
                    reset_points = sorted(self.reset_points[symbol], 
                                       key=lambda x: abs(x.get("price", 0) - rec.get("current_price", 0)))[:2]
                    
                    for i, point in enumerate(reset_points):
                        price = point.get("price", 0)
                        strength = point.get("strength", 0)
                        distance = abs(price - rec.get("current_price", 0)) / rec.get("current_price", 0) * 100
                        
                        point_text = f"• Reset Point: ${price:.2f} (Strength: {strength:.1f}, {distance:.1f}% away)"
                        ttk.Label(self.alignment_frame, text=point_text).pack(anchor=tk.W, padx=20, pady=2)
            
            # Check for recent regime transitions
            if hasattr(self, "regime_transitions") and symbol in self.regime_transitions:
                transitions = self.regime_transitions[symbol]
                if transitions:
                    ttk.Label(self.alignment_frame, text="Recent Regime Transitions:", 
                            font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
                    
                    for i, transition in enumerate(transitions[:2]):
                        from_regime = transition.get("from_regime", "Unknown")
                        to_regime = transition.get("to_regime", "Unknown")
                        date = transition.get("to_date", "Unknown")
                        
                        transition_text = f"• {date}: {from_regime} → {to_regime}"
                        ttk.Label(self.alignment_frame, text=transition_text).pack(anchor=tk.W, padx=20, pady=2)
            
            # Update chart
            self.update_chart(rec)
            
            # ENTRY/EXIT TAB
            # Entry criteria - format similar to the screenshot
            ttk.Label(self.entry_frame, text="ENTRY CRITERIA", 
                    font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
            
            # Entry Zone with larger font and better spacing
            ttk.Label(self.entry_frame, text="Entry Zone:", 
                    font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
            entry_zone = rec.get('entry_zone', [0, 0])
            ttk.Label(self.entry_frame, 
                    text=f"${entry_zone[0]:.2f} - ${entry_zone[1]:.2f}",
                    font=("Arial", 11)).pack(anchor=tk.W, padx=20, pady=5)
            
            # Add VIX condition if available
            raw_data = rec.get("raw_data", {})
            entry_conditions = raw_data.get("entry_conditions", {})
            vix_condition = entry_conditions.get("vix_condition", "any")
            vix_text = "above_25" if vix_condition == "above_25" else "below_20" if vix_condition == "below_20" else "any"
            
            ttk.Label(self.entry_frame, text="VIX Condition:", 
                    font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
            ttk.Label(self.entry_frame, 
                    text=vix_text,
                    font=("Arial", 11)).pack(anchor=tk.W, padx=20, pady=5)
            
            # Add Energy Level if available
            ttk.Label(self.entry_frame, text="Energy Level:", 
                    font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
            energy_text = entry_conditions.get("energy_level", "enter_on_approach")
            ttk.Label(self.entry_frame, 
                    text=energy_text,
                    font=("Arial", 11)).pack(anchor=tk.W, padx=20, pady=5)
            
            # Add ideal entry price if available
            if "ideal_entry_price" in entry_conditions:
                ttk.Label(self.entry_frame, text="Ideal Entry Price:", 
                        font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
                ttk.Label(self.entry_frame, 
                        text=f"${entry_conditions['ideal_entry_price']:.2f}",
                        font=("Arial", 11)).pack(anchor=tk.W, padx=20, pady=5)
            
            # Add market conditions if available
            if "market_conditions" in entry_conditions:
                ttk.Label(self.entry_frame, text="Ideal Market Conditions:", 
                        font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=10, pady=5)
                
                ideal_conditions = entry_conditions["market_conditions"].get("ideal", [])
                avoid_conditions = entry_conditions["market_conditions"].get("avoid", [])
                
                if ideal_conditions:
                    ttk.Label(self.entry_frame, text="Look for:").pack(anchor=tk.W, padx=20, pady=(5, 0))
                    for condition in ideal_conditions:
                        ttk.Label(self.entry_frame, text=f"• {condition}").pack(anchor=tk.W, padx=25, pady=2)
                
                if avoid_conditions:
                    ttk.Label(self.entry_frame, text="Avoid:").pack(anchor=tk.W, padx=20, pady=(5, 0))
                    for condition in avoid_conditions:
                        ttk.Label(self.entry_frame, text=f"• {condition}").pack(anchor=tk.W, padx=25, pady=2)
            
            # Exit criteria - format similar to the screenshot
            ttk.Label(self.exit_frame, text="EXIT PLAN", 
                    font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 5))
            
            # Create a frame for profit target, stop loss, and hold days
            exit_details_frame = ttk.Frame(self.exit_frame)
            exit_details_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Profit Target (in green)
            profit_label = ttk.Label(exit_details_frame, 
                                   text=f"Profit Target: {rec.get('profit_target_percent', 0):.1f}%", 
                                   font=("Arial", 12, "bold"), foreground="green")
            profit_label.pack(side=tk.LEFT
