
# --- START OF FILE fixed_original.py ---

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
        self.instrument_regimes = {} # Initialize instrument regimes
        self.regime_history = {} # Initialize regime history

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
            try:
                files = os.listdir(self.results_dir)
                logger.info(f"Found {len(files)} files in results directory")
                recommendation_files = [f for f in files if f.endswith("_trade_recommendation.json") or f.endswith("_enhanced_recommendation.json")]
                logger.info(f"Found {len(recommendation_files)} trade recommendation files")

                # Check for market regime files
                regime_files = [f for f in files if "market_regime" in f or "market_bias" in f or "regime_validation" in f]
                logger.info(f"Found {len(regime_files)} potential market regime files")

                # Check for enhanced recommendation files
                enhanced_files = [f for f in files if f.endswith("_enhanced_recommendation.json")]
                logger.info(f"Found {len(enhanced_files)} enhanced recommendation files")

                # Check for analysis files
                analysis_files = [f for f in files if f.endswith("_analysis.json")]
                logger.info(f"Found {len(analysis_files)} analysis files")

                # Check for greeks files
                greeks_files = [f for f in files if "_greeks" in f and f.endswith(".json")]
                logger.info(f"Found {len(greeks_files)} Greeks files")

                # Check for duration files
                duration_files = [f for f in files if "_trade_duration" in f and f.endswith(".json")]
                logger.info(f"Found {len(duration_files)} trade duration files")

            except Exception as e:
                logger.error(f"Error listing files in results directory: {e}")
        else:
             logger.warning(f"Results directory not found: {self.results_dir}")


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
        elif 'alt' in available_themes:
            self.style.theme_use('alt')
        elif 'default' in available_themes:
            self.style.theme_use('default')

        # Configure colors
        bg_color = "#f5f5f5"
        accent_color = "#4a86e8"
        text_color = "#333333"
        header_bg = "#e0e0e0" # Light grey for headers

        # Configure styles
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=text_color, font=('Arial', 10))
        self.style.configure("Bold.TLabel", background=bg_color, foreground=text_color, font=('Arial', 10, 'bold'))
        self.style.configure("Header.TLabel", background=header_bg, foreground=text_color, font=('Arial', 12, 'bold'))
        self.style.configure("TButton", background=accent_color, foreground="white", font=('Arial', 10))
        self.style.map("TButton", background=[('active', '#357ae8')]) # Slightly darker blue on hover/press
        self.style.configure("Accent.TButton", background=accent_color, foreground="white", font=('Arial', 10, 'bold'))
        self.style.map("Accent.TButton", background=[('active', '#357ae8')])
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", background=header_bg, padding=[10, 5], font=('Arial', 10))
        self.style.map("TNotebook.Tab", background=[("selected", bg_color)]) # Active tab matches background
        self.style.configure("TLabelframe", background=bg_color, relief="groove", borderwidth=1)
        self.style.configure("TLabelframe.Label", background=bg_color, foreground=text_color, font=('Arial', 11, 'bold'))

        # Configure Treeview
        self.style.configure("Treeview",
                         background="#ffffff", # White background for list
                         fieldbackground="#ffffff",
                         foreground=text_color,
                         rowheight=25,
                         font=('Arial', 10))
        self.style.configure("Treeview.Heading",
                         background=accent_color,
                         foreground="white",
                         font=('Arial', 10, 'bold'),
                         relief="flat")
        self.style.map("Treeview.Heading", relief=[('active','groove'),('pressed','sunken')])

        # Configure tag colors for risk levels and alignment
        self.treeview_tags = {
            "LOW": {"background": "#d9ead3", "foreground": "#38761d"},    # Light green
            "MEDIUM": {"background": "#fff2cc", "foreground": "#b45f06"}, # Light yellow
            "HIGH": {"background": "#f4cccc", "foreground": "#cc0000"},   # Light red
            "selected": {"background": "#c9daf8", "foreground": "#333333"}, # Light blue for selection
            "aligned": {"background": "#cfe2f3", "foreground": "#1c4587"}  # Light teal/blue - for regime-aligned trades
        }
        # Define tags in the Treeview style
        for tag, config in self.treeview_tags.items():
            self.style.configure(f"{tag}.Treeview", background=config["background"], foreground=config["foreground"])

    def create_layout(self):
        """Create the main layout frames."""
        # Top frame for filters and controls - Place it above the PanedWindow
        self.top_frame = ttk.Frame(self.root, padding="5 5 5 0") # Padding bottom 0
        self.top_frame.pack(fill=tk.X, padx=10, pady=(5, 0)) # Reduced bottom pady

        # Create a PanedWindow for resizable frames
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10)) # Padding adjusted

        # Left frame for recommendation list
        self.left_frame = ttk.Frame(self.main_paned, width=400) # Start with a reasonable width
        self.left_frame.pack_propagate(False) # Prevent frame from shrinking
        self.main_paned.add(self.left_frame, weight=1)

        # Right frame for details and charts
        self.right_frame = ttk.Frame(self.main_paned, width=750)
        self.right_frame.pack_propagate(False)
        self.main_paned.add(self.right_frame, weight=2)


    def create_top_controls(self):
        """Create filter controls and buttons."""
        # Create a frame for filters
        filter_frame = ttk.LabelFrame(self.top_frame, text="Filters", padding=5)
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Strategy filter
        ttk.Label(filter_frame, text="Strategy:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.strategy_var = tk.StringVar(value="All")
        self.strategy_combo = ttk.Combobox(filter_frame, textvariable=self.strategy_var, width=20, state="readonly")
        self.strategy_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.strategy_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # Risk filter
        ttk.Label(filter_frame, text="Risk:").grid(row=0, column=2, padx=(10, 5), pady=5, sticky=tk.W)
        self.risk_var = tk.StringVar(value="All")
        self.risk_combo = ttk.Combobox(filter_frame, textvariable=self.risk_var,
                                   values=["All", "LOW", "MEDIUM", "HIGH"], width=10, state="readonly")
        self.risk_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.risk_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # Market regime filter
        ttk.Label(filter_frame, text="Regime:").grid(row=0, column=4, padx=(10, 5), pady=5, sticky=tk.W)
        self.regime_var = tk.StringVar(value="All")
        self.regime_combo = ttk.Combobox(filter_frame, textvariable=self.regime_var,
                                      values=["All"], width=20, state="readonly")
        self.regime_combo.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        self.regime_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        # Show aligned recommendations only
        self.aligned_var = tk.BooleanVar(value=False)
        aligned_check = ttk.Checkbutton(filter_frame, text="Show aligned only",
                                     variable=self.aligned_var,
                                     command=self.apply_filters)
        aligned_check.grid(row=0, column=6, padx=10, pady=5, sticky=tk.W)

        # Create a frame for buttons
        button_frame = ttk.Frame(self.top_frame, padding=5)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Load tracker data button
        self.load_tracker_btn = ttk.Button(button_frame, text="Load Tracker Data",
                                        command=self.load_tracker_data)
        self.load_tracker_btn.pack(side=tk.LEFT, padx=5, pady=0) # Use pady=0 to align with filters

        # Refresh button
        self.refresh_btn = ttk.Button(button_frame, text="Refresh Data",
                                  command=self.refresh_data, style="Accent.TButton")
        self.refresh_btn.pack(side=tk.LEFT, padx=5, pady=0)


    def create_recommendation_list(self):
        """Create the recommendation list treeview."""
        # Create a frame for the recommendation list
        list_frame = ttk.LabelFrame(self.left_frame, text="Trade Recommendations", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0) # Use padding 0 here

        # Create a treeview for the recommendations
        columns = ("Symbol", "Strategy", "Entry", "Stop", "Target", "R:R", "Risk") # R:R might be hard to calculate here simply
        self.recommendation_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")

        # Configure column headings
        for col in columns:
            self.recommendation_tree.heading(col, text=col, anchor=tk.W)

        # Configure column widths and alignment
        self.recommendation_tree.column("Symbol", width=70, anchor=tk.W)
        self.recommendation_tree.column("Strategy", width=120, anchor=tk.W)
        self.recommendation_tree.column("Entry", width=70, anchor=tk.E) # Align numbers right
        self.recommendation_tree.column("Stop", width=70, anchor=tk.E)
        self.recommendation_tree.column("Target", width=70, anchor=tk.E)
        self.recommendation_tree.column("R:R", width=50, anchor=tk.CENTER) # Center R:R
        self.recommendation_tree.column("Risk", width=60, anchor=tk.CENTER) # Center Risk

        # Configure tags for risk levels and alignment (using style configuration now)
        for tag_name in self.treeview_tags:
             self.recommendation_tree.tag_configure(tag_name.lower()) # Ensure tags exist

        # Add scrollbar
        scrollbar_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.recommendation_tree.yview)
        scrollbar_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.recommendation_tree.xview)
        self.recommendation_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Pack the treeview and scrollbar
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.recommendation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        # Bind selection event
        self.recommendation_tree.bind("<<TreeviewSelect>>", self.on_recommendation_select)

    def create_details_view(self):
        """Create the details view."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0) # Use padding 0

        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.summary_frame, text="Summary")

        # Entry/Exit tab
        self.entry_exit_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.entry_exit_frame, text="Entry/Exit")

        # Greeks tab (new)
        self.greeks_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.greeks_frame, text="Greeks")

        # Market Context tab
        self.market_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.market_frame, text="Market Context")

        # Regime Analysis tab
        self.regime_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.regime_frame, text="Regime Analysis")

        # Trade Duration tab (new)
        self.duration_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.duration_frame, text="Trade Duration")

        # Initialize the details content
        self.initialize_details_content()

    def initialize_details_content(self):
        """Initialize the content for the details tabs."""
        # --- Summary tab ---
        summary_content = ttk.Frame(self.summary_frame) # Use inner frame if needed, but can pack directly
        summary_content.pack(fill=tk.BOTH, expand=True)

        # Header section
        self.header_frame = ttk.Frame(summary_content)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = ttk.Label(self.header_frame, text="Select a recommendation",
                                   font=("Arial", 16, "bold")) # Slightly smaller bold
        self.title_label.pack(anchor=tk.W)

        self.subtitle_label = ttk.Label(self.header_frame, text="", font=('Arial', 11))
        self.subtitle_label.pack(anchor=tk.W, pady=(2,0))

        # Trade structure section
        self.trade_structure_frame = ttk.LabelFrame(summary_content, text="TRADE STRUCTURE", padding=10)
        self.trade_structure_frame.pack(fill=tk.X, pady=10)
        ttk.Label(self.trade_structure_frame, text="Select a recommendation to view trade details").pack(
            anchor=tk.W, padx=0, pady=10) # Reduced padding

        # Market alignment section (for regime alignment)
        self.alignment_frame = ttk.LabelFrame(summary_content, text="MARKET ALIGNMENT", padding=10)
        self.alignment_frame.pack(fill=tk.X, pady=10)
        ttk.Label(self.alignment_frame,
                text="Select a recommendation to view market regime alignment").pack(
            anchor=tk.W, padx=0, pady=10)

        # Create figure for charts
        chart_frame = ttk.LabelFrame(summary_content, text="Price Analysis", padding=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Use a placeholder initially for the figure
        self.chart_placeholder = ttk.Frame(chart_frame)
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.chart_placeholder, text="Price chart will appear here.", anchor=tk.CENTER).pack(expand=True)
        self.fig = None
        self.canvas = None


        # --- Entry/Exit tab ---
        self.entry_frame = ttk.LabelFrame(self.entry_exit_frame, text="ENTRY CRITERIA", padding=10)
        self.entry_frame.pack(fill=tk.X, padx=0, pady=(0, 10)) # Use standard padding
        ttk.Label(self.entry_frame, text="Select a recommendation to view entry criteria").pack(pady=10)

        self.exit_frame = ttk.LabelFrame(self.entry_exit_frame, text="EXIT PLAN", padding=10)
        self.exit_frame.pack(fill=tk.X, padx=0, pady=0)
        ttk.Label(self.exit_frame, text="Select a recommendation to view exit criteria").pack(pady=10)

        # --- Greeks tab ---
        self.greeks_info_frame = ttk.LabelFrame(self.greeks_frame, text="Greek Values", padding=10)
        self.greeks_info_frame.pack(fill=tk.X, padx=0, pady=(0, 10))
        ttk.Label(self.greeks_info_frame, text="Select a recommendation to view Greek metrics").pack(pady=10)

        # Greek charts frame
        self.greeks_chart_frame = ttk.LabelFrame(self.greeks_frame, text="Greek Visualization", padding=5)
        self.greeks_chart_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        # Placeholder for Greek chart
        self.greeks_chart_placeholder = ttk.Frame(self.greeks_chart_frame)
        self.greeks_chart_placeholder.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.greeks_chart_placeholder, text="Greek metrics visualization will appear here.", anchor=tk.CENTER).pack(expand=True)
        self.greek_fig = None
        self.greek_canvas = None


        # --- Market Context tab ---
        self.market_info_frame = ttk.LabelFrame(self.market_frame, text="Current Market Regime", padding=10)
        self.market_info_frame.pack(fill=tk.X, padx=0, pady=(0, 10))
        ttk.Label(self.market_info_frame, text="Loading market regime data...").pack(pady=10)

        # Instrument context
        self.instrument_frame = ttk.LabelFrame(self.market_frame, text="Instrument Context", padding=10)
        self.instrument_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        ttk.Label(self.instrument_frame, text="Select recommendation for instrument context.").pack(pady=10)

        # --- Regime Analysis tab ---
        self.market_viz_frame = ttk.LabelFrame(self.regime_frame, text="Market Regime Calendar", padding=5)
        self.market_viz_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        # Placeholder for Regime Viz
        self.regime_viz_placeholder = ttk.Frame(self.market_viz_frame)
        self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.regime_viz_placeholder, text="Loading regime visualization...").pack(pady=10, anchor=tk.CENTER)
        self.regime_fig = None
        self.regime_canvas = None


        # --- Trade Duration tab ---
        self.duration_info_frame = ttk.LabelFrame(self.duration_frame, text="Trade Duration Analysis", padding=10)
        self.duration_info_frame.pack(fill=tk.X, padx=0, pady=(0, 10))
        ttk.Label(self.duration_info_frame, text="Select a recommendation to view duration analysis").pack(pady=10)

        # Duration visualization
        self.duration_viz_frame = ttk.LabelFrame(self.duration_frame, text="Duration Performance", padding=5)
        self.duration_viz_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        # Placeholder for Duration Viz
        self.duration_viz_placeholder = ttk.Frame(self.duration_viz_frame)
        self.duration_viz_placeholder.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.duration_viz_placeholder, text="Duration performance visualization will appear here.").pack(pady=10, anchor=tk.CENTER)
        self.duration_fig = None
        self.duration_canvas = None


    def create_status_bar(self):
        """Create status bar at the bottom of the window."""
        self.status_bar = ttk.Frame(self.root, padding="5 2") # Add padding
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5)) # Adjust padding

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
        logger.info("Starting data load...")
        self.update_status("Loading all data...")

        # Load trade recommendations
        self.load_recommendations()

        # Load market regime data
        self.load_market_regime_data() # Loads general market regime

        # Load energy levels and reset points from analysis files
        self.load_energy_and_reset_data() # Loads instrument specific energy/reset

        # Load Greek metrics from analysis files
        self.load_greek_data() # Loads instrument specific greeks

        # Load trade duration data (simulate or load)
        self.load_trade_duration_data() # Loads strategy specific duration

        # Load Tracker Data (regime history) - Optional, user clicks button
        # self.load_tracker_data() # Don't autoload this

        # Update tracker status based on market regime load
        if self.market_regimes:
            self.tracker_status.config(text="Market Regime: Loaded", foreground="green")
        else:
            self.tracker_status.config(text="Market Regime: Not found", foreground="orange")


        # Update filter options
        self.update_filter_options()

        self.update_status(f"Data loading complete. {len(self.recommendations)} recommendations found.", "success")
        logger.info("Data loading finished.")

    def load_recommendations(self):
        """Load trade recommendations from JSON files."""
        self.recommendations = []

        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found: {self.results_dir}", "warning")
            return

        # Update status
        self.update_status("Loading recommendations...")
        logger.info("Loading recommendations...")
        count = 0
        processed_files = 0

        try:
            # Look for recommendation files
            all_files = os.listdir(self.results_dir)
            rec_files = [f for f in all_files if f.endswith("_trade_recommendation.json") or f.endswith("_enhanced_recommendation.json")]
            logger.info(f"Found {len(rec_files)} potential recommendation files.")

            for filename in rec_files:
                processed_files += 1
                try:
                    file_path = os.path.join(self.results_dir, filename)
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Process the recommendation data
                    processed_rec = self.process_recommendation(data, file_path)
                    if processed_rec:
                        self.recommendations.append(processed_rec)
                        count += 1

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error loading {filename}: {e}")
                except Exception as e:
                    logger.error(f"Error loading or processing recommendation from {filename}: {e}")

        except Exception as e:
            logger.error(f"Error accessing results directory '{self.results_dir}': {e}")
            self.update_status(f"Error accessing results directory: {e}", "error")
            return # Stop loading if directory access fails

        # Sort recommendations by date (newest first)
        try:
            self.recommendations.sort(key=lambda x: x.get("timestamp", "1970-01-01 00:00:00"), reverse=True)
        except Exception as e:
             logger.error(f"Error sorting recommendations: {e}")


        # Update status
        status_msg = f"Loaded {count} recommendations from {processed_files} files."
        self.update_status(status_msg)
        logger.info(status_msg)

    def _is_valid_recommendation(self, rec):
        """Check if a dictionary represents a valid recommendation."""
        # Add more checks as needed
        required_keys = ["symbol", "action", "strategy", "entry_criteria", "exit_criteria"]
        if not isinstance(rec, dict):
            return False
        # Allow for variations like strategy_name
        has_strategy = "strategy" in rec or "strategy_name" in rec
        return all(key in rec for key in ["symbol", "action", "entry_criteria", "exit_criteria"]) and has_strategy


    def process_recommendation(self, data, file_path):
        """Process and standardize a recommendation from raw file data."""
        try:
            if not isinstance(data, dict):
                 logger.warning(f"Skipping non-dict data in {os.path.basename(file_path)}")
                 return None

            # Extract symbol (handle potential variations)
            symbol = data.get("symbol")
            if not symbol and "instrument" in data:
                symbol = data["instrument"]
            if not symbol:
                try:
                    # Try guessing from filename (e.g., AAPL_...)
                    symbol = os.path.basename(file_path).split('_')[0].upper()
                except IndexError:
                    symbol = "Unknown"

            # Extract action
            action = data.get("action", "Unknown").upper()

            # Extract strategy details (more robustly)
            strategy_name = "Unknown"
            if "strategy" in data:
                strategy_info = data["strategy"]
                if isinstance(strategy_info, dict):
                    strategy_name = strategy_info.get("name", "Unknown")
                elif isinstance(strategy_info, str):
                    strategy_name = strategy_info
            elif "strategy_name" in data:
                 strategy_name = data["strategy_name"]

            # Extract risk assessment
            risk_category = "MEDIUM"  # Default
            risk_assessment = data.get("risk_assessment", data.get("risk_management", {})) # Look in both places
            if isinstance(risk_assessment, dict):
                 risk_category = risk_assessment.get("risk_category", risk_assessment.get("level", "MEDIUM")).upper()

            # Extract price data
            current_price = data.get("current_price", data.get("last_price", 0))
            if not isinstance(current_price, (int, float)): current_price = 0

            # Extract entry criteria
            entry_criteria = data.get("entry_criteria", {})
            if not isinstance(entry_criteria, dict): entry_criteria = {}
            # --- Entry Zone ---
            entry_zone = [0, 0]
            price_range = entry_criteria.get("price_range")
            if isinstance(price_range, list) and len(price_range) == 2:
                entry_zone = [float(price_range[0]) if isinstance(price_range[0], (int, float)) else 0,
                              float(price_range[1]) if isinstance(price_range[1], (int, float)) else 0]
            elif isinstance(price_range, dict):
                 low = price_range.get("low", price_range.get("min", 0))
                 high = price_range.get("high", price_range.get("max", 0))
                 entry_zone = [float(low) if isinstance(low, (int, float)) else 0,
                               float(high) if isinstance(high, (int, float)) else 0]
            elif entry_criteria.get("entry_price"): # Single entry price
                ep = entry_criteria.get("entry_price", 0)
                if isinstance(ep, (int, float)):
                    entry_zone = [ep * 0.99, ep * 1.01] # Create small zone around it
                else: entry_zone = [0,0]
            else: # Default based on current price if no entry info
                 entry_zone = [current_price * 0.98, current_price * 1.02] if current_price > 0 else [0, 0]


            # --- Profit Target / Stop Loss ---
            exit_criteria = data.get("exit_criteria", {})
            if not isinstance(exit_criteria, dict): exit_criteria = {}

            profit_target = exit_criteria.get("profit_target_percent", exit_criteria.get("target_roi_percent", 0))
            stop_loss = exit_criteria.get("stop_loss_percent", exit_criteria.get("max_loss_percent", 0))

            # Convert absolute prices to percentages if needed
            if not profit_target and exit_criteria.get("profit_target_price") and entry_zone[0] > 0:
                 pt_price = exit_criteria.get("profit_target_price")
                 if isinstance(pt_price, (int, float)):
                     profit_target = ((pt_price / entry_zone[0]) - 1) * 100 if action == "BUY" else ((entry_zone[0] / pt_price) - 1) * 100

            if not stop_loss and exit_criteria.get("stop_loss_price") and entry_zone[0] > 0:
                 sl_price = exit_criteria.get("stop_loss_price")
                 if isinstance(sl_price, (int, float)):
                     stop_loss = abs(((sl_price / entry_zone[0]) - 1) * 100) # abs for percentage loss

            # Ensure they are floats
            profit_target = float(profit_target) if isinstance(profit_target, (int, float)) else 0.0
            stop_loss = float(stop_loss) if isinstance(stop_loss, (int, float)) else 0.0

            # --- Days to Hold ---
            days_to_hold = exit_criteria.get("days_to_hold", exit_criteria.get("holding_period_days", 0))
            days_to_hold = int(days_to_hold) if isinstance(days_to_hold, (int, float)) else 0

            # Calculate R:R (Reward:Risk ratio) - simplified
            rr_ratio = (profit_target / stop_loss) if stop_loss > 0 else 0
            rr_ratio_str = f"{rr_ratio:.1f}:1" if rr_ratio > 0 else "N/A"

            # Extract ROI data if available
            roi = data.get("expected_roi", data.get("roi", 0))
            if not roi and isinstance(risk_assessment, dict):
                roi = risk_assessment.get("expected_roi", risk_assessment.get("potential_profit_percent", 0))
            roi = float(roi) if isinstance(roi, (int, float)) else 0.0


            # Extract timestamp or use file modification time
            timestamp = data.get("timestamp", data.get("recommendation_time", ""))
            if not timestamp:
                try:
                    mod_time = os.path.getmtime(file_path)
                    timestamp = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


            # Create standardized recommendation
            standardized = {
                "symbol": symbol,
                "action": action,
                "strategy_name": strategy_name,
                "risk_category": risk_category,
                "current_price": float(current_price),
                "entry_zone": entry_zone, # List [low, high]
                "profit_target_percent": profit_target,
                "stop_loss_percent": stop_loss,
                "rr_ratio_str": rr_ratio_str, # Add R:R string
                "days_to_hold": days_to_hold,
                "roi": roi,
                "timestamp": timestamp,
                "file_path": file_path,
                "raw_data": data  # Store the original data for reference
            }

            # Basic validation
            if not symbol or symbol == "Unknown" or not action or action == "Unknown" or not strategy_name or strategy_name == "Unknown":
                 logger.warning(f"Skipping recommendation from {os.path.basename(file_path)} due to missing key info (Symbol/Action/Strategy).")
                 return None

            return standardized

        except Exception as e:
            logger.error(f"Critical error processing recommendation from {os.path.basename(file_path)}: {e}", exc_info=True)
            return None


    def load_market_regime_data(self):
        """Load general market regime data from various potential files."""
        self.market_regimes = {}
        logger.info("Attempting to load market regime data...")
        self.update_status("Loading market regime data...")

        # Look for market regime data in prioritized order
        # results/market_regime/current_regime.json (specific structure)
        # results/market_regime_summary.json
        # results/regime_validation.json
        # results/market_bias.json
        regime_paths = [
            os.path.join(self.results_dir, "market_regime", "current_regime.json"),
            os.path.join(self.results_dir, "market_regime_summary.json"),
            os.path.join(self.results_dir, "regime_validation.json"),
            os.path.join(self.results_dir, "market_bias.json"),
        ]

        loaded_successfully = False
        for path in regime_paths:
            if os.path.exists(path):
                logger.info(f"Found potential regime file: {path}")
                try:
                    with open(path, "r") as f:
                        regime_data_raw = json.load(f)

                    # --- Process loaded data ---
                    processed_regime = {}
                    if isinstance(regime_data_raw, dict):
                        # Common patterns
                        processed_regime['primary_label'] = regime_data_raw.get("primary_regime", regime_data_raw.get("market_regime", {}).get("primary_label", regime_data_raw.get("primary_label")))
                        processed_regime['secondary_label'] = regime_data_raw.get("secondary_regime", regime_data_raw.get("market_regime", {}).get("secondary_label", regime_data_raw.get("secondary_label")))
                        processed_regime['volatility_regime'] = regime_data_raw.get("volatility_regime", regime_data_raw.get("market_regime", {}).get("volatility_regime"))
                        processed_regime['dominant_greek'] = regime_data_raw.get("dominant_greek", regime_data_raw.get("market_regime", {}).get("dominant_greek"))
                        processed_regime['energy_state'] = regime_data_raw.get("energy_state", regime_data_raw.get("market_regime", {}).get("energy_state"))

                        # Handle regime_distribution if present
                        if "regime_distribution" in regime_data_raw:
                             processed_regime["regime_distribution"] = regime_data_raw["regime_distribution"]
                             # Infer primary/secondary if not explicitly set
                             if not processed_regime['primary_label'] and isinstance(regime_data_raw["regime_distribution"], dict):
                                 sorted_regimes = sorted(regime_data_raw["regime_distribution"].items(), key=lambda item: item[1], reverse=True)
                                 if sorted_regimes:
                                     processed_regime['primary_label'] = sorted_regimes[0][0]
                                 if len(sorted_regimes) > 1:
                                     processed_regime['secondary_label'] = sorted_regimes[1][0]

                        # Clean up None values and ensure defaults
                        processed_regime = {k: (v if v is not None else "Unknown") for k, v in processed_regime.items() if v is not None} # Keep distribution if exists
                        processed_regime.setdefault('primary_label', "Unknown")
                        processed_regime.setdefault('secondary_label', "Unknown")
                        processed_regime.setdefault('volatility_regime', "Normal")
                        processed_regime.setdefault('dominant_greek', "Unknown")
                        processed_regime.setdefault('energy_state', "Unknown")

                        # Store the processed data
                        self.market_regimes = processed_regime
                        logger.info(f"Successfully loaded and processed market regime data from {path}")
                        loaded_successfully = True
                        break # Stop after loading the first valid file
                    else:
                        logger.warning(f"Regime data in {path} is not a dictionary, skipping.")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error loading market regime data from {path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing market regime data from {path}: {e}", exc_info=True)
            else:
                 logger.debug(f"Regime file not found: {path}")

        if loaded_successfully:
             self.update_status("Market regime data loaded.", "success")
             self.update_market_context_display() # Update the display immediately
             self.update_regime_transitions() # Record today's regime
        else:
             logger.warning("Could not find or load any valid market regime file.")
             self.update_status("Market regime data not found.", "warning")
             self.update_market_context_display() # Update display to show "Unknown"

        # Note: Instrument-specific regimes are loaded in load_energy_and_reset_data now


    def update_regime_transitions(self):
        """Log regime transitions (simple version - just stores current)."""
        # This is a simplified approach. A real tracker would compare with previous day.
        # For now, it just ensures the self.regime_transitions dict exists for potential future use.
        if not self.market_regimes:
            return

        # We might load historical transitions later from tracker data
        if not self.regime_transitions: # Only initialize if empty
             self.regime_transitions = {}

        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")

        # Store today's regime snapshot if needed (e.g., for visualization)
        # We'll primarily use the loaded regime_history for visualization
        # self.regime_transitions[current_date] = { ... }
        # logger.debug(f"Updated regime transition log for {current_date}")

        # Saving transitions might be better handled by the regime generation script
        # try:
        #     transitions_path = os.path.join(self.results_dir, "regime_transitions_log.json")
        #     # Load existing log if available
        #     if os.path.exists(transitions_path):
        #         with open(transitions_path, "r") as f:
        #             existing_log = json.load(f)
        #         self.regime_transitions.update(existing_log) # Merge, overwriting today if needed
        #
        #     with open(transitions_path, "w") as f:
        #         json.dump(self.regime_transitions, f, indent=4)
        # except Exception as e:
        #     logger.error(f"Error saving regime transitions log: {e}")


    def load_energy_and_reset_data(self):
        """Load energy levels, reset points, and instrument regimes from analysis files."""
        self.energy_levels = {}
        self.reset_points = {}
        self.instrument_regimes = {} # Reset instrument regimes here
        symbols_to_check = set()

        # Get symbols from loaded recommendations if available
        if self.recommendations:
             symbols_to_check = set(rec["symbol"] for rec in self.recommendations if rec and "symbol" in rec)
             logger.info(f"Checking analysis files for {len(symbols_to_check)} symbols from recommendations.")
        else:
             logger.warning("No recommendations loaded, cannot check analysis files by symbol yet.")
             # Could potentially scan *all* analysis files if needed, but might be slow
             # return

        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found, cannot load analysis data.", "warning")
            return

        self.update_status("Loading analysis data (energy, reset, instrument regimes)...")
        loaded_energy_count = 0
        loaded_reset_count = 0
        loaded_instr_regime_count = 0

        for symbol in symbols_to_check:
            if not symbol or symbol == "Unknown": continue
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)

                    if isinstance(analysis_data, dict):
                        # Extract greek_analysis section
                        greek_analysis = analysis_data.get("greek_analysis", analysis_data) # Also check top level

                        if isinstance(greek_analysis, dict):
                            # Extract energy levels
                            if "energy_levels" in greek_analysis and isinstance(greek_analysis["energy_levels"], list):
                                self.energy_levels[symbol] = greek_analysis["energy_levels"]
                                loaded_energy_count += 1

                            # Extract reset points
                            if "reset_points" in greek_analysis and isinstance(greek_analysis["reset_points"], list):
                                self.reset_points[symbol] = greek_analysis["reset_points"]
                                loaded_reset_count += 1

                            # Extract instrument-specific market regime
                            if "market_regime" in greek_analysis and isinstance(greek_analysis["market_regime"], dict):
                                # Basic validation of regime structure
                                if greek_analysis["market_regime"].get("primary_label"):
                                     self.instrument_regimes[symbol] = greek_analysis["market_regime"]
                                     loaded_instr_regime_count += 1
                                else:
                                     logger.warning(f"Market regime found for {symbol} but missing primary_label.")


                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error loading analysis file {analysis_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing analysis data for {symbol} from {analysis_path}: {e}", exc_info=True)
            else:
                 logger.debug(f"Analysis file not found for symbol {symbol}: {analysis_path}")


        logger.info(f"Loaded energy levels for {loaded_energy_count} instruments.")
        logger.info(f"Loaded reset points for {loaded_reset_count} instruments.")
        logger.info(f"Loaded instrument regimes for {loaded_instr_regime_count} instruments.")
        self.update_status("Loaded analysis data.")


    def load_greek_data(self):
        """Load Greek metrics from analysis files or dedicated _greeks.json files."""
        self.greek_data = {}
        symbols_checked = set()

        # Get symbols from loaded recommendations if available
        if self.recommendations:
             symbols_to_check = set(rec["symbol"] for rec in self.recommendations if rec and "symbol" in rec)
        else:
             logger.warning("No recommendations loaded, cannot check greek files by symbol yet.")
             symbols_to_check = set()

        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found, cannot load Greeks.", "warning")
            return

        self.update_status("Loading Greek metrics...")
        loaded_greek_count = 0

        # 1. Check analysis files first (preferred source)
        for symbol in symbols_to_check:
            if not symbol or symbol == "Unknown": continue
            symbols_checked.add(symbol)
            analysis_path = os.path.join(self.results_dir, f"{symbol}_analysis.json")
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, "r") as f:
                        analysis_data = json.load(f)

                    if isinstance(analysis_data, dict):
                        # Extract greek_analysis section
                        greek_analysis = analysis_data.get("greek_analysis", analysis_data) # Also check top level

                        if isinstance(greek_analysis, dict):
                             # Check for common Greek keys or a 'magnitudes' sub-dict
                            if any(g in greek_analysis for g in ['delta', 'gamma', 'theta', 'vega', 'rho', 'vanna', 'charm']) or 'magnitudes' in greek_analysis:
                                self.greek_data[symbol] = greek_analysis # Store the whole greek_analysis dict
                                loaded_greek_count += 1
                                # logger.debug(f"Loaded Greek data for {symbol} from analysis file.")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error loading analysis file {analysis_path} for Greeks: {e}")
                except Exception as e:
                    logger.error(f"Error processing analysis data for Greeks ({symbol}) from {analysis_path}: {e}", exc_info=True)

        # 2. Check dedicated _greeks.json files for symbols not found above
        try:
             all_files = os.listdir(self.results_dir)
             greeks_files = [f for f in all_files if "_greeks" in f and f.endswith(".json")]
             logger.info(f"Found {len(greeks_files)} potential dedicated Greeks files.")

             for filename in greeks_files:
                 try:
                     # Try to extract symbol from filename (e.g., AAPL_greeks.json)
                     potential_symbol = filename.split('_')[0].upper()
                 except IndexError:
                     logger.warning(f"Could not determine symbol from Greeks filename: {filename}")
                     continue

                 # Only load if we haven't already loaded data for this symbol from analysis file
                 if potential_symbol and potential_symbol not in self.greek_data and potential_symbol not in symbols_checked:
                     symbols_checked.add(potential_symbol) # Mark as checked
                     file_path = os.path.join(self.results_dir, filename)
                     try:
                         with open(file_path, "r") as f:
                             data = json.load(f)

                         if isinstance(data, dict):
                             # Assume the file content is the Greek data itself
                              symbol = data.get("symbol", potential_symbol) # Use symbol from data if available
                              if any(g in data for g in ['delta', 'gamma', 'theta', 'vega', 'rho', 'vanna', 'charm']) or 'magnitudes' in data:
                                    self.greek_data[symbol] = data
                                    loaded_greek_count += 1
                                    # logger.debug(f"Loaded Greek data for {symbol} from dedicated file: {filename}")
                              else:
                                   logger.warning(f"Dedicated Greeks file {filename} lacks expected Greek keys.")
                         else:
                              logger.warning(f"Dedicated Greeks file {filename} does not contain a dictionary.")

                     except json.JSONDecodeError as e:
                         logger.error(f"JSON decode error loading dedicated Greeks file {filename}: {e}")
                     except Exception as e:
                         logger.error(f"Error processing dedicated Greeks file {filename}: {e}", exc_info=True)

        except Exception as e:
             logger.error(f"Error listing files in results directory for Greeks files: {e}")


        logger.info(f"Loaded Greek metrics for {loaded_greek_count} instruments from {len(symbols_checked)} checked.")
        self.update_status("Loaded Greek metrics.")


    def load_trade_duration_data(self):
        """Load or simulate trade duration data."""
        self.trade_durations = {}
        strategies_found = set()

        # Get strategies from loaded recommendations if available
        if self.recommendations:
             strategies_found = set(rec["strategy_name"] for rec in self.recommendations if rec and "strategy_name" in rec and rec["strategy_name"] != "Unknown")
             logger.info(f"Found {len(strategies_found)} unique strategies in recommendations.")
        else:
             logger.warning("No recommendations loaded, cannot load duration data by strategy yet.")
             # Fallback: Could try to load *all* duration files found, but might be less useful.
             # return

        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory not found: {self.results_dir}")
            self.update_status(f"Results directory not found, cannot load duration data.", "warning")
            # Proceed to simulation if directory doesn't exist
        else:
            # --- Try loading from files first ---
            self.update_status("Loading trade duration data...")
            loaded_duration_count = 0
            files_checked = 0
            try:
                all_files = os.listdir(self.results_dir)
                duration_files = [f for f in all_files if "_trade_duration" in f and f.endswith(".json")]
                logger.info(f"Found {len(duration_files)} potential duration files.")

                for filename in duration_files:
                    files_checked += 1
                    try:
                        # Try to extract strategy name (might be fragile)
                        # Assumes format like STRATEGYNAME_trade_duration.json
                        potential_strategy = filename.split('_trade_duration')[0]
                    except Exception:
                        logger.warning(f"Could not determine strategy name from duration filename: {filename}")
                        continue

                    # Only load if it matches a strategy we know from recommendations
                    if potential_strategy in strategies_found:
                        file_path = os.path.join(self.results_dir, filename)
                        try:
                            with open(file_path, "r") as f:
                                data = json.load(f)

                            if isinstance(data, dict):
                                # Basic validation: check for expected keys
                                if "avg_days" in data or "optimal_days" in data or "completion_profile" in data:
                                    # Use strategy name from file content if available, else use filename part
                                    strategy_name = data.get("strategy_name", potential_strategy)
                                    self.trade_durations[strategy_name] = data
                                    loaded_duration_count += 1
                                    # logger.debug(f"Loaded duration data for strategy '{strategy_name}' from {filename}")
                                else:
                                    logger.warning(f"Duration file {filename} lacks expected keys (avg_days, optimal_days, completion_profile).")
                            else:
                                logger.warning(f"Duration file {filename} does not contain a dictionary.")

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error loading duration file {filename}: {e}")
                        except Exception as e:
                            logger.error(f"Error processing duration file {filename}: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error listing files in results directory for duration files: {e}")

            logger.info(f"Loaded duration data for {loaded_duration_count} strategies from {files_checked} files checked.")


        # --- Simulate for strategies without loaded data ---
        simulated_count = 0
        for strategy in strategies_found:
            if strategy not in self.trade_durations:
                # Simulate duration metrics based on strategy type keywords
                avg_days, optimal_days, success_rate = 6.5, 7, 0.64 # Defaults
                completion_profile = {f"day_{i+1}": round(min(1.0, (i+1)*0.2), 2) for i in range(5)} # Simple linear profile

                if "VOLATILITY" in strategy.upper() or "VEGA" in strategy.upper():
                    avg_days, optimal_days, success_rate = 3.5, 4, 0.65
                    completion_profile = {"day_1": 0.25, "day_2": 0.60, "day_3": 0.85, "day_4": 0.95, "day_5": 1.0}
                elif "CALENDAR" in strategy.upper() or "THETA" in strategy.upper():
                    avg_days, optimal_days, success_rate = 12.4, 14, 0.72
                    completion_profile = {f"day_{i+1}": round(min(1.0, (i+1)*0.1), 2) for i in range(10)} # Slower profile
                elif "VANNA" in strategy.upper() or "CHARM" in strategy.upper() or "DIRECTIONAL" in strategy.upper():
                    avg_days, optimal_days, success_rate = 8.2, 7, 0.68
                    completion_profile = {"day_1": 0.15, "day_2": 0.35, "day_3": 0.60, "day_4": 0.85, "day_5": 0.95, "day_6": 0.98, "day_7": 1.0}

                self.trade_durations[strategy] = {
                    "strategy_name": strategy,
                    "avg_days": avg_days,
                    "optimal_days": optimal_days,
                    "success_rate": success_rate,
                    "completion_profile": completion_profile,
                    "source": "simulated" # Mark as simulated
                }
                simulated_count += 1
                # logger.debug(f"Simulated duration profile for strategy: {strategy}")

        logger.info(f"Simulated duration profiles for {simulated_count} strategies.")
        total_durations = len(self.trade_durations)
        self.update_status(f"Loaded/Simulated duration data for {total_durations} strategies.")


    def update_filter_options(self):
        """Update filter dropdown options based on loaded data."""
        # Get unique strategies
        strategies = ["All"]
        risks = ["All", "LOW", "MEDIUM", "HIGH"] # Standard risk levels
        regimes = ["All"]

        if self.recommendations:
            unique_strategies = sorted(list(set(rec.get("strategy_name", "Unknown") for rec in self.recommendations if rec and rec.get("strategy_name") != "Unknown")))
            strategies.extend(unique_strategies)

            unique_risks = sorted(list(set(rec.get("risk_category", "MEDIUM") for rec in self.recommendations if rec and rec.get("risk_category"))))
            # Ensure standard risks are present even if not in data
            for r in ["LOW", "MEDIUM", "HIGH"]:
                 if r not in unique_risks: unique_risks.append(r)
            risks = ["All"] + sorted(unique_risks)


        # Populate regime filter from market regime data and instrument regimes
        unique_regimes = set()
        if self.market_regimes and self.market_regimes.get("primary_label") != "Unknown":
            unique_regimes.add(self.market_regimes["primary_label"])
        if self.market_regimes and self.market_regimes.get("secondary_label") != "Unknown":
            unique_regimes.add(self.market_regimes["secondary_label"])

        if self.instrument_regimes:
            for regime_data in self.instrument_regimes.values():
                 if isinstance(regime_data, dict):
                      if regime_data.get("primary_label") and regime_data.get("primary_label") != "Unknown":
                           unique_regimes.add(regime_data["primary_label"])
                      if regime_data.get("secondary_label") and regime_data.get("secondary_label") != "Unknown":
                           unique_regimes.add(regime_data["secondary_label"])

        # Also add regimes mentioned in recommendation raw_data (less reliable)
        # for rec in self.recommendations:
        #     raw_data = rec.get("raw_data", {})
        #     market_regime = raw_data.get("market_regime", {})
        #     if isinstance(market_regime, dict):
        #         if market_regime.get("primary_label") and market_regime.get("primary_label") != "Unknown":
        #             unique_regimes.add(market_regime["primary_label"])


        regimes.extend(sorted(list(unique_regimes)))

        # Update strategy combobox
        self.strategy_combo['values'] = strategies
        if self.strategy_var.get() not in strategies: self.strategy_var.set("All")

        # Update risk combobox
        self.risk_combo['values'] = risks
        if self.risk_var.get() not in risks: self.risk_var.set("All")

        # Update regime combobox
        self.regime_combo['values'] = regimes
        if self.regime_var.get() not in regimes: self.regime_var.set("All")

        logger.info("Filter options updated.")

    def update_regime_filter(self):
        """Update market regime filter options specifically."""
        # This logic is now integrated into update_filter_options
        self.update_filter_options()

    def update_market_context_display(self):
        """Update the 'Current Market Regime' section in the Market Context tab."""
        logger.debug("Updating market context display.")
        # Clear existing widgets
        for widget in self.market_info_frame.winfo_children():
            widget.destroy()

        if self.market_regimes:
            labels = {
                "Primary Regime:": self.market_regimes.get("primary_label", "Unknown"),
                "Secondary Regime:": self.market_regimes.get("secondary_label", "Unknown"),
                "Volatility:": self.market_regimes.get("volatility_regime", "Normal"),
                "Dominant Greek:": self.market_regimes.get("dominant_greek", "Unknown"),
                "Energy State:": self.market_regimes.get("energy_state", "Unknown"),
            }
            row = 0
            for label_text, value_text in labels.items():
                 ttk.Label(self.market_info_frame, text=label_text, font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0,5), pady=2, sticky=tk.W)
                 ttk.Label(self.market_info_frame, text=value_text).grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
                 row += 1

            # Display regime distribution if available
            if "regime_distribution" in self.market_regimes and isinstance(self.market_regimes["regime_distribution"], dict):
                 ttk.Label(self.market_info_frame, text="Distribution:", font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0,5), pady=(5,2), sticky=tk.NW)
                 dist_frame = ttk.Frame(self.market_info_frame)
                 dist_frame.grid(row=row, column=1, padx=5, pady=(5,2), sticky=tk.W)
                 dist_row = 0
                 sorted_dist = sorted(self.market_regimes["regime_distribution"].items(), key=lambda item: item[1], reverse=True)
                 for regime, prob in sorted_dist:
                     ttk.Label(dist_frame, text=f"- {regime}: {prob:.1%}").grid(row=dist_row, column=0, sticky=tk.W)
                     dist_row += 1
                 row += 1

        else:
            ttk.Label(self.market_info_frame, text="Market regime data not loaded or found.").pack(pady=10)

    def populate_dashboard(self):
        """Populate the dashboard recommendations list."""
        logger.info("Populating dashboard...")
        self.update_status("Populating recommendations list...")

        # Clear existing items from the treeview
        if hasattr(self, 'recommendation_tree'):
             for item in self.recommendation_tree.get_children():
                 self.recommendation_tree.delete(item)
        else:
             logger.error("Recommendation treeview not initialized.")
             self.update_status("Error: UI not fully initialized.", "error")
             return

        # Apply filters to get the list to display
        filtered_recs = self.apply_filters(update_ui=False) # Get filtered list without updating UI yet

        # Add recommendations to the tree
        added_count = 0
        for rec in filtered_recs:
            try:
                # Get values needed for the columns
                symbol = rec.get("symbol", "N/A")
                strategy = rec.get("strategy_name", "N/A")

                # Get entry zone for display
                entry_zone = rec.get('entry_zone', [0, 0])
                # Simple entry price display (e.g., target price or mid-point)
                entry_display = f"{entry_zone[0]:.2f}" if entry_zone[0] > 0 else "N/A"
                # entry_display = f"${entry_zone[0]:.1f}-${entry_zone[1]:.1f}" # Alternative range display

                # Calculate target and stop prices based on entry[0] and percentages
                target_pct = rec.get('profit_target_percent', 0)
                stop_pct = rec.get('stop_loss_percent', 0)
                entry_price = entry_zone[0] # Use low end of zone for calculation baseline

                target_price = 0
                stop_price = 0
                if entry_price > 0:
                    if rec.get("action", "BUY").upper() == "BUY":
                         target_price = entry_price * (1 + target_pct / 100)
                         stop_price = entry_price * (1 - stop_pct / 100)
                    else: # SELL action
                         target_price = entry_price * (1 - target_pct / 100)
                         stop_price = entry_price * (1 + stop_pct / 100)

                target_display = f"{target_price:.2f}" if target_price > 0 else "N/A"
                stop_display = f"{stop_price:.2f}" if stop_price > 0 else "N/A"

                # R:R String
                rr_str = rec.get("rr_ratio_str", "N/A")

                # Risk Category
                risk = rec.get("risk_category", "N/A").upper()

                # Determine tags to apply
                tags = []
                if risk in ["LOW", "MEDIUM", "HIGH"]:
                    tags.append(risk.lower()) # Tag name matches style config

                # Add aligned tag if recommendation aligns with market regime
                if self.is_aligned_with_market_regime(rec):
                    tags.append("aligned")

                # Add to tree with appropriate tags
                # Ensure values match the columns defined in create_recommendation_list
                self.recommendation_tree.insert("", tk.END, values=(
                    symbol, strategy, entry_display, stop_display, target_display, rr_str, risk
                ), tags=tuple(tags)) # Pass tags as a tuple
                added_count += 1

            except Exception as e:
                 logger.error(f"Error adding recommendation to tree: {rec.get('symbol')} - {e}", exc_info=False) # Don't need full trace for every item error


        # Select the first item if available
        if self.recommendation_tree.get_children():
            first_item = self.recommendation_tree.get_children()[0]
            self.recommendation_tree.selection_set(first_item)
            self.recommendation_tree.focus(first_item)
            # self.on_recommendation_select(None) # Trigger selection event manually if needed after filters applied
        else:
            # No recommendations available
            self.update_status(f"No recommendations found matching the current filters ({len(filtered_recs)} checked)")
            # Clear details view if no recommendations
            self.clear_details_view()


        logger.info(f"Populated dashboard with {added_count} recommendations.")
        self.update_status(f"Displaying {added_count} recommendations.", "info")


    def apply_filters(self, update_ui=True):
        """Apply filters and return filtered recommendations. Optionally update UI."""
        logger.debug("Applying filters...")
        filtered = self.recommendations.copy() # Start with all recommendations

        # Get filter values
        selected_strategy = self.strategy_var.get()
        selected_risk = self.risk_var.get()
        selected_regime = self.regime_var.get()
        show_aligned_only = self.aligned_var.get()

        # Apply strategy filter
        if selected_strategy != "All":
            filtered = [rec for rec in filtered
                      if rec.get("strategy_name", "Unknown") == selected_strategy]
        initial_count = len(filtered)

        # Apply risk filter
        if selected_risk != "All":
            filtered = [rec for rec in filtered
                      if rec.get("risk_category", "Unknown").upper() == selected_risk]
        logger.debug(f"After Risk filter ({selected_risk}): {len(filtered)} (from {initial_count})")
        initial_count = len(filtered)


        # Apply regime filter (checks if instrument regime OR general market regime matches)
        if selected_regime != "All":
            filtered = [rec for rec in filtered
                      if self.is_aligned_with_specific_regime(rec, selected_regime)]
        logger.debug(f"After Regime filter ({selected_regime}): {len(filtered)} (from {initial_count})")
        initial_count = len(filtered)


        # Apply aligned filter (checks if instrument regime OR general market regime matches CURRENT market regime)
        if show_aligned_only:
            filtered = [rec for rec in filtered
                      if self.is_aligned_with_market_regime(rec)]
        logger.debug(f"After Aligned filter ({show_aligned_only}): {len(filtered)} (from {initial_count})")

        # Update UI if requested
        if update_ui:
            logger.debug("Updating UI based on filters...")
            # Clear existing items from tree
            for item in self.recommendation_tree.get_children():
                 self.recommendation_tree.delete(item)

            # Add filtered recommendations
            added_count = 0
            for rec in filtered:
                 try:
                    symbol = rec.get("symbol", "N/A")
                    strategy = rec.get("strategy_name", "N/A")
                    entry_zone = rec.get('entry_zone', [0, 0])
                    entry_display = f"{entry_zone[0]:.2f}" if entry_zone[0] > 0 else "N/A"
                    target_pct = rec.get('profit_target_percent', 0)
                    stop_pct = rec.get('stop_loss_percent', 0)
                    entry_price = entry_zone[0]
                    target_price = 0
                    stop_price = 0
                    if entry_price > 0:
                        action = rec.get("action", "BUY").upper()
                        target_price = entry_price * (1 + target_pct / 100) if action == "BUY" else entry_price * (1 - target_pct / 100)
                        stop_price = entry_price * (1 - stop_pct / 100) if action == "BUY" else entry_price * (1 + stop_pct / 100)
                    target_display = f"{target_price:.2f}" if target_price > 0 else "N/A"
                    stop_display = f"{stop_price:.2f}" if stop_price > 0 else "N/A"
                    rr_str = rec.get("rr_ratio_str", "N/A")
                    risk = rec.get("risk_category", "N/A").upper()
                    tags = []
                    if risk in ["LOW", "MEDIUM", "HIGH"]: tags.append(risk.lower())
                    if self.is_aligned_with_market_regime(rec): tags.append("aligned")

                    self.recommendation_tree.insert("", tk.END, values=(
                         symbol, strategy, entry_display, stop_display, target_display, rr_str, risk
                    ), tags=tuple(tags))
                    added_count += 1
                 except Exception as e:
                     logger.error(f"Error adding filtered rec to tree: {rec.get('symbol')} - {e}", exc_info=False)

            # Update status
            status_msg = f"Found {len(filtered)} recommendations matching filters."
            self.update_status(status_msg)
            logger.info(status_msg)

             # Select first item or clear details
            if self.recommendation_tree.get_children():
                first_item = self.recommendation_tree.get_children()[0]
                self.recommendation_tree.selection_set(first_item)
                self.recommendation_tree.focus(first_item)
                self.on_recommendation_select(None) # Trigger update
            else:
                self.clear_details_view()


        return filtered # Return the filtered list

    def is_aligned_with_market_regime(self, rec):
        """Check if recommendation aligns with the CURRENT general market regime."""
        if not rec or not isinstance(rec, dict): return False
        symbol = rec.get("symbol", "")

        # Get current market regime (primary label)
        if not self.market_regimes:
            # logger.warning("Cannot check alignment: Market regime data not loaded.")
            return False # Cannot align if market regime is unknown
        current_regime = self.market_regimes.get("primary_label", "Unknown")
        if current_regime == "Unknown":
             # logger.debug(f"Cannot check alignment for {symbol}: Current market regime is Unknown.")
             return False

        # 1. Check instrument-specific regime first (if available)
        instrument_regime_data = self.instrument_regimes.get(symbol)
        if isinstance(instrument_regime_data, dict):
            instrument_primary = instrument_regime_data.get("primary_label", "Unknown")
            if instrument_primary != "Unknown":
                 # logger.debug(f"Align check {symbol}: Instrument regime '{instrument_primary}' vs Market '{current_regime}'")
                 return instrument_primary == current_regime

        # 2. Fallback: Check regime stored within the recommendation's raw data (less reliable)
        # raw_data = rec.get("raw_data", {})
        # market_regime_in_rec = raw_data.get("market_regime", {})
        # if isinstance(market_regime_in_rec, dict):
        #     rec_primary = market_regime_in_rec.get("primary_label", "Unknown")
        #     if rec_primary != "Unknown":
        #          logger.debug(f"Align check {symbol}: Rec raw data regime '{rec_primary}' vs Market '{current_regime}'")
        #          return rec_primary == current_regime

        # 3. If no instrument or raw data regime, cannot determine alignment definitively based on rec info
        # logger.debug(f"Align check {symbol}: No specific regime found for recommendation, cannot determine alignment.")
        return False


    def is_aligned_with_specific_regime(self, rec, target_regime):
        """Check if recommendation aligns with a SPECIFIC target regime."""
        if not rec or not isinstance(rec, dict) or target_regime == "Unknown":
             return False
        symbol = rec.get("symbol", "")

        # 1. Check instrument-specific regime first
        instrument_regime_data = self.instrument_regimes.get(symbol)
        if isinstance(instrument_regime_data, dict):
            instrument_primary = instrument_regime_data.get("primary_label", "Unknown")
            if instrument_primary != "Unknown":
                 # logger.debug(f"Specific align check {symbol}: Instrument '{instrument_primary}' vs Target '{target_regime}'")
                 return instrument_primary == target_regime

        # 2. Fallback: Check regime stored within the recommendation's raw data
        # raw_data = rec.get("raw_data", {})
        # market_regime_in_rec = raw_data.get("market_regime", {})
        # if isinstance(market_regime_in_rec, dict):
        #     rec_primary = market_regime_in_rec.get("primary_label", "Unknown")
        #     if rec_primary != "Unknown":
        #          logger.debug(f"Specific align check {symbol}: Rec raw data '{rec_primary}' vs Target '{target_regime}'")
        #          return rec_primary == target_regime

        # 3. Cannot determine alignment if no info found
        # logger.debug(f"Specific align check {symbol}: No specific regime found for rec, cannot check against '{target_regime}'.")
        return False

    def on_recommendation_select(self, event):
        """Handle selection of a recommendation from the list."""
        if not hasattr(self, 'recommendation_tree'): return

        selection = self.recommendation_tree.selection()
        if not selection:
            self.selected_recommendation = None
            self.clear_details_view() # Clear details if nothing selected
            return

        try:
            item_id = selection[0]
            values = self.recommendation_tree.item(item_id, "values")

            if not values or len(values) < 7: # Check against number of columns
                logger.warning(f"Selected item '{item_id}' has unexpected values: {values}")
                self.selected_recommendation = None
                self.clear_details_view()
                return

            # Identify the recommendation based on values (symbol and strategy are good keys)
            selected_symbol = values[0]
            selected_strategy = values[1]
            # Note: Entry/Stop/Target values in the tree might be formatted prices, not percentages/zones
            # We need to find the original recommendation dict based on symbol/strategy

            found_rec = None
            for rec in self.recommendations:
                 # Add more checks if symbol+strategy isn't unique enough (e.g., timestamp proximity)
                if rec.get("symbol") == selected_symbol and rec.get("strategy_name") == selected_strategy:
                    # Potential match - could add timestamp check if duplicates exist
                    found_rec = rec
                    break # Found the first match

            if not found_rec:
                logger.error(f"Could not find original recommendation data for selected item: {values}")
                self.selected_recommendation = None
                self.clear_details_view()
                self.update_status(f"Error: Could not retrieve details for {selected_symbol} {selected_strategy}", "error")
                return

            # Store the selected recommendation (the full dict)
            self.selected_recommendation = found_rec

            # Update the details view using the full recommendation dict
            self.update_details_view(found_rec)

            # Update status
            self.update_status(f"Selected: {selected_symbol} - {selected_strategy}")
            logger.debug(f"Selected recommendation: {selected_symbol} - {selected_strategy}")

        except IndexError:
             logger.error(f"IndexError retrieving values for selected item: {item_id}")
             self.selected_recommendation = None
             self.clear_details_view()
        except Exception as e:
            logger.error(f"Error handling recommendation selection: {e}", exc_info=True)
            self.update_status(f"Error displaying details: {e}", "error")
            self.selected_recommendation = None
            self.clear_details_view()


    def update_details_view(self, rec):
        """Update all tabs in the details view with the selected recommendation."""
        if not rec or not isinstance(rec, dict):
            logger.warning("update_details_view called with invalid recommendation data.")
            self.clear_details_view()
            return

        logger.debug(f"Updating details view for {rec.get('symbol')}")
        self.update_status(f"Loading details for {rec.get('symbol')}...")

        try:
            # --- Update Summary Tab ---
            self.update_summary_tab(rec)

            # --- Update Entry/Exit Tab ---
            self.update_entry_exit_tab(rec)

            # --- Update Greeks Tab ---
            self.update_greeks_tab(rec)

            # --- Update Market Context Tab ---
            self.update_market_context_tab(rec) # Already updated general, now add instrument

            # --- Update Regime Analysis Tab ---
            self.update_regime_analysis_tab(rec) # Includes visualization

            # --- Update Trade Duration Tab ---
            self.update_trade_duration_tab(rec)

            self.update_status(f"Details loaded for {rec.get('symbol')}", "info")

        except Exception as e:
            logger.error(f"Error updating details view for {rec.get('symbol')}: {e}", exc_info=True)
            self.update_status(f"Error updating details: {str(e)}", "error")
            # Optionally clear view on major error
            # self.clear_details_view()


    def clear_details_view(self):
        """Clear all content from the details tabs and reset placeholders."""
        logger.debug("Clearing details view.")
        self.selected_recommendation = None # Ensure no selection is stored

        # Reset Summary Tab
        self.title_label.config(text="Select a recommendation")
        self.subtitle_label.config(text="")
        for frame in [self.trade_structure_frame, self.alignment_frame]:
             for widget in frame.winfo_children(): widget.destroy()
             ttk.Label(frame, text="-").pack(padx=10, pady=10) # Placeholder

        # Clear Price Chart
        if self.canvas: self.canvas.get_tk_widget().destroy()
        if self.fig: plt.close(self.fig) # Close the figure to release memory
        self.fig, self.canvas = None, None
        # Ensure placeholder is visible
        if not self.chart_placeholder.winfo_ismapped():
            self.chart_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.chart_placeholder, text="Price chart will appear here.", anchor=tk.CENTER).pack(expand=True)


        # Clear Entry/Exit Tab
        for frame in [self.entry_frame, self.exit_frame]:
            for widget in frame.winfo_children(): widget.destroy()
            ttk.Label(frame, text="-").pack(pady=10)

        # Clear Greeks Tab
        for frame in [self.greeks_info_frame]:
             for widget in frame.winfo_children(): widget.destroy()
             ttk.Label(frame, text="-").pack(pady=10)
        # Clear Greeks Chart
        if self.greek_canvas: self.greek_canvas.get_tk_widget().destroy()
        if self.greek_fig: plt.close(self.greek_fig)
        self.greek_fig, self.greek_canvas = None, None
        if not self.greeks_chart_placeholder.winfo_ismapped():
            self.greeks_chart_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.greeks_chart_placeholder, text="Greek metrics visualization will appear here.", anchor=tk.CENTER).pack(expand=True)


        # Clear Market Context Tab (Keep general regime, clear instrument)
        # self.update_market_context_display() # Keep general market info displayed
        for frame in [self.instrument_frame]:
             for widget in frame.winfo_children(): widget.destroy()
             ttk.Label(frame, text="Select recommendation for instrument context.").pack(pady=10)


        # Clear Regime Analysis Tab
        # Clear Regime Viz Chart
        if self.regime_canvas: self.regime_canvas.get_tk_widget().destroy()
        if self.regime_fig: plt.close(self.regime_fig)
        self.regime_fig, self.regime_canvas = None, None
        # Ensure placeholder is visible
        if not self.regime_viz_placeholder.winfo_ismapped():
             self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
             ttk.Label(self.regime_viz_placeholder, text="Regime visualization requires loading Tracker Data.", anchor=tk.CENTER).pack(expand=True)


        # Clear Duration Tab
        for frame in [self.duration_info_frame]:
            for widget in frame.winfo_children(): widget.destroy()
            ttk.Label(frame, text="-").pack(pady=10)
        # Clear Duration Chart
        if self.duration_canvas: self.duration_canvas.get_tk_widget().destroy()
        if self.duration_fig: plt.close(self.duration_fig)
        self.duration_fig, self.duration_canvas = None, None
        if not self.duration_viz_placeholder.winfo_ismapped():
            self.duration_viz_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.duration_viz_placeholder, text="Duration performance visualization will appear here.", anchor=tk.CENTER).pack(expand=True)



    # --- Tab Update Methods ---

    def update_summary_tab(self, rec):
        """Update the Summary tab content."""
        symbol = rec.get('symbol', 'Unknown')
        strategy = rec.get('strategy_name', 'Unknown')
        action = rec.get('action', 'BUY')
        current_price = rec.get('current_price', 0)
        timestamp = rec.get('timestamp', 'Unknown')
        date_str = timestamp.split()[0] if timestamp != 'Unknown' else 'N/A'

        # Update header
        self.title_label.config(text=f"{symbol} - {action} {strategy}")
        self.subtitle_label.config(text=f"Current Price: ${current_price:.2f} (as of {date_str})")

        # Update trade structure
        self.update_trade_structure(rec)

        # Update market alignment
        self.update_market_alignment(rec)

        # Update price chart (includes energy/reset visualization)
        self.update_price_chart(rec)

    def update_entry_exit_tab(self, rec):
        """Update the Entry/Exit tab content."""
        self.update_entry_criteria(rec)
        self.update_exit_plan(rec)

    def update_greeks_tab(self, rec):
        """Update the Greeks tab content."""
        symbol = rec.get('symbol')
        # Update Greek values display
        self.update_greeks_info(symbol)
        # Update Greek visualization
        self.create_greeks_visualization(symbol)

    def update_market_context_tab(self, rec):
        """Update the Market Context tab content."""
        symbol = rec.get('symbol')
        # General market context is updated separately by load_market_regime_data
        # Update instrument-specific context display
        self.update_instrument_context(symbol)

    def update_regime_analysis_tab(self, rec):
        """Update the Regime Analysis tab content."""
        # The main content is the visualization based on loaded tracker data
        self.create_regime_visualization() # Recreate or update the visualization

    def update_trade_duration_tab(self, rec):
        """Update the Trade Duration tab content."""
        strategy_name = rec.get('strategy_name')
        # Update duration info display
        self.update_duration_info(strategy_name)
        # Update duration visualization
        self.create_duration_visualization(strategy_name)

    # --- Helper methods for tab updates ---

    def update_trade_structure(self, rec):
        """Update the TRADE STRUCTURE section in the Summary tab."""
        # Clear existing widgets
        for widget in self.trade_structure_frame.winfo_children(): widget.destroy()

        details = {
            "Action:": f"{rec.get('action', 'N/A')} {rec.get('symbol', 'N/A')}",
            "Strategy:": rec.get('strategy_name', 'N/A'),
            "Entry Zone:": f"${rec.get('entry_zone', [0,0])[0]:.2f} - ${rec.get('entry_zone', [0,0])[1]:.2f}",
            "Target (%):": f"{rec.get('profit_target_percent', 0):.1f}%",
            "Stop Loss (%):": f"{rec.get('stop_loss_percent', 0):.1f}%",
            "Reward:Risk:": rec.get('rr_ratio_str', 'N/A'),
            "Hold (Days):": str(rec.get('days_to_hold', 'N/A')),
            "Risk Level:": rec.get('risk_category', 'N/A'),
            "Expected ROI (%):": f"{rec.get('roi', 0):.1f}%" if rec.get('roi', 0) else "N/A",
        }

        row = 0
        for label, value in details.items():
            if value == "N/A" and label not in ["Action:", "Strategy:"]: continue # Skip N/A details except core ones
            lbl = ttk.Label(self.trade_structure_frame, text=label, font=('Arial', 10, 'bold'))
            lbl.grid(row=row, column=0, padx=(0, 5), pady=1, sticky=tk.W)
            val = ttk.Label(self.trade_structure_frame, text=value, wraplength=250) # Wrap long strategy names
            val.grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
            row += 1

    def update_market_alignment(self, rec):
        """Update the MARKET ALIGNMENT section in the Summary tab."""
        # Clear existing widgets
        for widget in self.alignment_frame.winfo_children(): widget.destroy()

        symbol = rec.get("symbol")
        is_aligned = self.is_aligned_with_market_regime(rec)
        alignment_status = "Aligned" if is_aligned else "Not Aligned"
        alignment_color = "green" if is_aligned else "red"

        # Display Alignment Status
        ttk.Label(self.alignment_frame, text="Alignment Status:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 5), pady=(0, 5), sticky=tk.W)
        ttk.Label(self.alignment_frame, text=alignment_status, foreground=alignment_color, font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=(0, 5), sticky=tk.W)

        # Display Regime Details
        current_market_regime = self.market_regimes.get("primary_label", "Unknown")
        instrument_regime_data = self.instrument_regimes.get(symbol)
        instrument_regime = "N/A"
        if isinstance(instrument_regime_data, dict):
             instrument_regime = instrument_regime_data.get("primary_label", "Unknown")

        ttk.Label(self.alignment_frame, text="Current Market Regime:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=(0, 5), pady=1, sticky=tk.W)
        ttk.Label(self.alignment_frame, text=current_market_regime).grid(row=1, column=1, padx=5, pady=1, sticky=tk.W)

        ttk.Label(self.alignment_frame, text=f"{symbol} Regime:", font=('Arial', 10, 'bold')).grid(row=2, column=0, padx=(0, 5), pady=1, sticky=tk.W)
        ttk.Label(self.alignment_frame, text=instrument_regime).grid(row=2, column=1, padx=5, pady=1, sticky=tk.W)

    def update_price_chart(self, rec):
        """Update the Price Analysis chart in the Summary tab."""
        logger.debug(f"Updating price chart for {rec.get('symbol')}")
        # Clear previous chart and placeholder
        if self.canvas:
            self.canvas.get_tk_widget().destroy() # Use destroy for complete removal
        if self.fig:
            plt.close(self.fig) # Ensure figure is closed
        self.fig, self.canvas = None, None
        for widget in self.chart_placeholder.winfo_children():
            widget.destroy()
        self.chart_placeholder.pack_forget() # Hide placeholder frame


        try:
            symbol = rec.get('symbol', 'Unknown')
            current_price = rec.get('current_price', 0)
            entry_zone = rec.get('entry_zone', [0, 0])
            target_pct = rec.get('profit_target_percent', 0)
            stop_pct = rec.get('stop_loss_percent', 0)
            action = rec.get("action", "BUY").upper()

            # Calculate target/stop prices
            entry_price_low = entry_zone[0]
            entry_price_high = entry_zone[1]
            target_price, stop_price = 0, 0
            if entry_price_low > 0: # Use low end of zone for calc
                target_price = entry_price_low * (1 + target_pct / 100) if action == "BUY" else entry_price_low * (1 - target_pct / 100)
                stop_price = entry_price_low * (1 - stop_pct / 100) if action == "BUY" else entry_price_low * (1 + stop_pct / 100)

            # Get energy levels and reset points for this symbol
            energy = self.energy_levels.get(symbol, [])
            resets = self.reset_points.get(symbol, [])

            # --- Create Plot ---
            self.fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted size
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15) # Fine-tune layout

            # Determine y-axis range dynamically
            all_levels = [current_price, entry_price_low, entry_price_high, target_price, stop_price]
            all_levels.extend([e['price'] for e in energy if isinstance(e.get('price'), (int, float))])
            all_levels.extend([r['price'] for r in resets if isinstance(r.get('price'), (int, float))])
            valid_levels = [lvl for lvl in all_levels if lvl > 0]

            if not valid_levels:
                 logger.warning(f"No valid price levels to plot for {symbol}")
                 # Display a message instead of empty plot
                 ttk.Label(self.chart_placeholder, text=f"No price data to plot for {symbol}.").pack(expand=True)
                 self.chart_placeholder.pack(fill=tk.BOTH, expand=True) # Show placeholder again
                 return


            min_y = min(valid_levels) * 0.95
            max_y = max(valid_levels) * 1.05
            y_range = max_y - min_y
            if y_range == 0: # Handle case where all levels are the same
                min_y *= 0.9
                max_y *= 1.1

            ax.set_ylim(min_y, max_y)
            ax.set_xlim(-1, 1) # Simple x-axis for placing labels

            # Plot horizontal lines
            if current_price > 0: ax.axhline(current_price, color='black', linestyle='-', lw=1.5, label=f'Current: ${current_price:.2f}')
            if entry_price_low > 0 and entry_price_high > 0: ax.axhspan(entry_price_low, entry_price_high, color='green', alpha=0.2, label=f'Entry Zone')
            if target_price > 0: ax.axhline(target_price, color='blue', linestyle='--', lw=1, label=f'Target: ${target_price:.2f}')
            if stop_price > 0: ax.axhline(stop_price, color='red', linestyle='--', lw=1, label=f'Stop: ${stop_price:.2f}')

            # Plot energy levels (lighter colors)
            for e in energy:
                price = e.get('price')
                if isinstance(price, (int, float)) and price > 0:
                    ax.axhline(price, color='orange', linestyle=':', lw=1, alpha=0.7, label=f'Energy: ${price:.2f}' if 'Energy' not in [l.get_label() for l in ax.lines] else "") # Only label first

            # Plot reset points (different style)
            for r in resets:
                price = r.get('price')
                if isinstance(price, (int, float)) and price > 0:
                    ax.axhline(price, color='purple', linestyle='-.', lw=1, alpha=0.7, label=f'Reset: ${price:.2f}' if 'Reset' not in [l.get_label() for l in ax.lines] else "")

            ax.set_title(f'{symbol} Key Price Levels', fontsize=12)
            ax.set_ylabel('Price')
            ax.set_yticks(np.linspace(min_y, max_y, 6)) # Adjust number of ticks
            ax.tick_params(axis='y', labelrotation=0, labelsize=9)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks/labels
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            # Create legend outside the plot area
            # Sort handles and labels to group Energy/Reset if multiple exist
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = {}
            for h, l in zip(handles, labels):
                label_base = l.split(':')[0] # Group by 'Energy', 'Reset', etc.
                if label_base not in unique_labels:
                    unique_labels[label_base] = (h, l) # Store first handle and full label
            sorted_unique = sorted(unique_labels.values(), key=lambda x: x[1]) # Sort by full label text
            unique_handles = [item[0] for item in sorted_unique]
            unique_labels_text = [item[1] for item in sorted_unique]

            ax.legend(unique_handles, unique_labels_text, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            self.fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend


            # Embed chart in Tkinter
            # Get the parent frame for the canvas (which is chart_frame)
            parent_frame = self.chart_placeholder.master
            self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error creating price chart for {rec.get('symbol')}: {e}", exc_info=True)
            # Display error message in the placeholder area
            self.chart_placeholder.pack(fill=tk.BOTH, expand=True) # Ensure placeholder is visible
            ttk.Label(self.chart_placeholder, text=f"Error creating chart: {e}", foreground="red").pack(expand=True)

    def update_entry_criteria(self, rec):
        """Update the ENTRY CRITERIA section in the Entry/Exit tab."""
        # Clear existing widgets
        for widget in self.entry_frame.winfo_children(): widget.destroy()

        entry_criteria = rec.get("raw_data", {}).get("entry_criteria", {})
        entry_conditions = rec.get("raw_data", {}).get("entry_conditions", {}) # Look here too
        if not entry_criteria and entry_conditions: entry_criteria = entry_conditions # Use conditions if criteria empty

        # Entry Zone
        entry_zone = rec.get('entry_zone', [0, 0])
        ttk.Label(self.entry_frame, text="Entry Zone:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 5), pady=2, sticky=tk.W)
        ttk.Label(self.entry_frame, text=f"${entry_zone[0]:.2f} - ${entry_zone[1]:.2f}").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        row = 1
        # Display other criteria if available in raw_data
        if isinstance(entry_criteria, dict):
             details_map = {
                 "ideal_entry_price": "Ideal Entry Price:",
                 "vix_condition": "VIX Condition:",
                 "energy_level_condition": "Energy Level:", # Renamed for clarity
                 "confirmation_signals": "Confirmation Signals:",
                 "timing": "Timing Considerations:",
             }
             for key, label in details_map.items():
                 value = entry_criteria.get(key)
                 if value:
                      # Handle list values nicely
                      if isinstance(value, list): value = ", ".join(map(str, value))
                      if isinstance(value, float): value = f"${value:.2f}"
                      ttk.Label(self.entry_frame, text=label, font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0, 5), pady=2, sticky=tk.NW) # Use NW for lists
                      ttk.Label(self.entry_frame, text=str(value), wraplength=300).grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
                      row += 1

             # Handle market_conditions separately
             market_conditions = entry_criteria.get("market_conditions")
             if isinstance(market_conditions, dict):
                  ideal = market_conditions.get("ideal")
                  avoid = market_conditions.get("avoid")
                  if ideal or avoid:
                     ttk.Label(self.entry_frame, text="Market Conditions:", font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0, 5), pady=(5,2), sticky=tk.NW)
                     mc_frame = ttk.Frame(self.entry_frame)
                     mc_frame.grid(row=row, column=1, padx=5, pady=(5,2), sticky=tk.W)
                     mc_row = 0
                     if ideal:
                          ttk.Label(mc_frame, text="Ideal:", font=('Arial', 9, 'italic')).grid(row=mc_row, column=0, sticky=tk.W)
                          mc_row += 1
                          for cond in ideal: ttk.Label(mc_frame, text=f"- {cond}").grid(row=mc_row, column=0, padx=(10,0), sticky=tk.W); mc_row +=1
                     if avoid:
                          ttk.Label(mc_frame, text="Avoid:", font=('Arial', 9, 'italic')).grid(row=mc_row, column=0, sticky=tk.W)
                          mc_row += 1
                          for cond in avoid: ttk.Label(mc_frame, text=f"- {cond}").grid(row=mc_row, column=0, padx=(10,0), sticky=tk.W); mc_row +=1
                     row +=1

        if row == 1: # Nothing added beyond entry zone
             ttk.Label(self.entry_frame, text="No specific criteria found.").grid(row=row, column=0, columnspan=2, pady=5)

    def update_exit_plan(self, rec):
        """Update the EXIT PLAN section in the Entry/Exit tab."""
        # Clear existing widgets
        for widget in self.exit_frame.winfo_children(): widget.destroy()

        exit_criteria = rec.get("raw_data", {}).get("exit_criteria", {})

        # Key Exit Levels
        profit_target_pct = rec.get('profit_target_percent', 0)
        stop_loss_pct = rec.get('stop_loss_percent', 0)
        days_to_hold = rec.get('days_to_hold', 0)

        # Calculate prices if possible
        entry_price = rec.get('entry_zone', [0, 0])[0]
        action = rec.get("action", "BUY").upper()
        target_price, stop_price = 0, 0
        if entry_price > 0:
             target_price = entry_price * (1 + profit_target_pct / 100) if action == "BUY" else entry_price * (1 - profit_target_pct / 100)
             stop_price = entry_price * (1 - stop_loss_pct / 100) if action == "BUY" else entry_price * (1 + stop_loss_pct / 100)

        # Display Key Levels
        ttk.Label(self.exit_frame, text="Profit Target:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 5), pady=2, sticky=tk.W)
        ttk.Label(self.exit_frame, text=f"{profit_target_pct:.1f}% " + (f"(${target_price:.2f})" if target_price > 0 else ""), foreground="green").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(self.exit_frame, text="Stop Loss:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=(0, 5), pady=2, sticky=tk.W)
        ttk.Label(self.exit_frame, text=f"{stop_loss_pct:.1f}% " + (f"(${stop_price:.2f})" if stop_price > 0 else ""), foreground="red").grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(self.exit_frame, text="Max Hold Time:", font=('Arial', 10, 'bold')).grid(row=2, column=0, padx=(0, 5), pady=2, sticky=tk.W)
        ttk.Label(self.exit_frame, text=f"{days_to_hold} days" if days_to_hold > 0 else "N/A").grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        row = 3
        # Display other criteria if available
        if isinstance(exit_criteria, dict):
             details_map = {
                 "trailing_stop_percent": "Trailing Stop (%):",
                 "profit_taking_levels": "Profit Taking Levels:", # e.g., [50, 75] percent of target
                 "exit_conditions": "Other Exit Conditions:", # e.g., ["Regime change", "VIX spike"]
                 "early_exit_criteria": "Early Exit Criteria:",
             }
             for key, label in details_map.items():
                 value = exit_criteria.get(key)
                 if value:
                      if isinstance(value, list): value = ", ".join(map(str, value))
                      ttk.Label(self.exit_frame, text=label, font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0, 5), pady=2, sticky=tk.NW)
                      ttk.Label(self.exit_frame, text=str(value), wraplength=300).grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
                      row += 1

        if row == 3: # Nothing added beyond key levels
             ttk.Label(self.exit_frame, text="No additional exit criteria found.").grid(row=row, column=0, columnspan=2, pady=5)

    def update_greeks_info(self, symbol):
        """Update the Greek Values section in the Greeks tab."""
         # Clear existing widgets
        for widget in self.greeks_info_frame.winfo_children(): widget.destroy()

        greek_data = self.greek_data.get(symbol)

        if not greek_data or not isinstance(greek_data, dict):
             ttk.Label(self.greeks_info_frame, text=f"No Greek data found for {symbol}.").pack(pady=10)
             return

        # Define the order and labels for Greeks
        # Look for greeks in 'magnitudes' or at the top level
        greeks_to_display = {}
        source_dict = greek_data.get('magnitudes', greek_data) # Check magnitudes first

        greek_order = ['delta', 'gamma', 'theta', 'vega', 'rho', 'vanna', 'charm', 'vomma', 'speed']
        greek_labels = {
            'delta': 'Delta:', 'gamma': 'Gamma:', 'theta': 'Theta:', 'vega': 'Vega:', 'rho': 'Rho:',
            'vanna': 'Vanna:', 'charm': 'Charm:', 'vomma': 'Vomma:', 'speed': 'Speed:'
        }

        row = 0
        found_any = False
        for greek in greek_order:
             value = source_dict.get(greek)
             if value is not None and isinstance(value, (int, float)): # Check if value exists and is numeric
                 label = greek_labels.get(greek, f"{greek.capitalize()}:")
                 ttk.Label(self.greeks_info_frame, text=label, font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=(0, 5), pady=1, sticky=tk.W)
                 ttk.Label(self.greeks_info_frame, text=f"{value:.4f}").grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
                 row += 1
                 found_any = True

        if not found_any:
             ttk.Label(self.greeks_info_frame, text=f"No standard Greek values found in data for {symbol}.").pack(pady=10)


    def create_greeks_visualization(self, symbol):
        """Create or update the Greeks visualization chart."""
        logger.debug(f"Updating Greeks chart for {symbol}")
        # Clear previous chart and placeholder
        if self.greek_canvas: self.greek_canvas.get_tk_widget().destroy()
        if self.greek_fig: plt.close(self.greek_fig)
        self.greek_fig, self.greek_canvas = None, None
        for widget in self.greeks_chart_placeholder.winfo_children(): widget.destroy()
        self.greeks_chart_placeholder.pack_forget()

        greek_data = self.greek_data.get(symbol)
        if not greek_data or not isinstance(greek_data, dict):
            logger.warning(f"No Greek data available to visualize for {symbol}")
            self.greeks_chart_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.greeks_chart_placeholder, text=f"No Greek data found for {symbol}.", anchor=tk.CENTER).pack(expand=True)
            return

        try:
            # Extract relevant Greeks (magnitudes or top-level)
            source_dict = greek_data.get('magnitudes', greek_data)
            greeks = {
                'Delta': source_dict.get('delta', 0),
                'Gamma': source_dict.get('gamma', 0),
                'Theta': source_dict.get('theta', 0),
                'Vega': source_dict.get('vega', 0),
                # Include 2nd/3rd order if available
                'Vanna': source_dict.get('vanna', 0),
                'Charm': source_dict.get('charm', 0),
                'Vomma': source_dict.get('vomma', 0),
                'Speed': source_dict.get('speed', 0),
            }
            # Filter out zero or None values
            greeks_to_plot = {k: v for k, v in greeks.items() if v is not None and v != 0}

            if not greeks_to_plot:
                 logger.warning(f"All Greek values are zero or missing for {symbol}")
                 self.greeks_chart_placeholder.pack(fill=tk.BOTH, expand=True)
                 ttk.Label(self.greeks_chart_placeholder, text=f"No non-zero Greeks to plot for {symbol}.", anchor=tk.CENTER).pack(expand=True)
                 return

            labels = list(greeks_to_plot.keys())
            values = list(greeks_to_plot.values())
            abs_values = [abs(v) for v in values] # Use absolute values for bar height magnitude

            # --- Create Bar Chart ---
            self.greek_fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted size
            self.greek_fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25) # More bottom margin for labels

            bars = ax.bar(labels, abs_values, color=plt.cm.Paired(range(len(labels))))

            ax.set_ylabel('Absolute Magnitude')
            ax.set_title(f'{symbol} Greek Magnitudes', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=9) # Rotate labels
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            # Add value labels on top of bars (showing original signed value)
            for bar, value in zip(bars, values):
                 yval = bar.get_height()
                 ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(abs_values), # Position above bar
                         f'{value:.3f}', ha='center', va='bottom', fontsize=8)


            # Embed chart in Tkinter
            parent_frame = self.greeks_chart_placeholder.master
            self.greek_canvas = FigureCanvasTkAgg(self.greek_fig, master=parent_frame)
            self.greek_canvas_widget = self.greek_canvas.get_tk_widget()
            self.greek_canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.greek_canvas.draw()

        except Exception as e:
            logger.error(f"Error creating Greeks visualization for {symbol}: {e}", exc_info=True)
            self.greeks_chart_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.greeks_chart_placeholder, text=f"Error creating Greeks chart: {e}", foreground="red", anchor=tk.CENTER).pack(expand=True)

    def update_instrument_context(self, symbol):
        """Update the Instrument Context section in the Market Context tab."""
         # Clear existing widgets
        for widget in self.instrument_frame.winfo_children(): widget.destroy()

        # Display Instrument-Specific Regime
        ttk.Label(self.instrument_frame, text="Instrument Regime:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        instr_regime_data = self.instrument_regimes.get(symbol)
        row = 1
        if isinstance(instr_regime_data, dict):
             labels = {
                 "Primary Label:": instr_regime_data.get("primary_label", "Unknown"),
                 "Secondary Label:": instr_regime_data.get("secondary_label", "Unknown"),
                 "Volatility:": instr_regime_data.get("volatility_regime", "Normal"),
                 # Add other relevant fields from instrument regime if they exist
             }
             for label_text, value_text in labels.items():
                 ttk.Label(self.instrument_frame, text=label_text).grid(row=row, column=0, padx=(10, 5), pady=1, sticky=tk.W)
                 ttk.Label(self.instrument_frame, text=value_text).grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
                 row += 1
        else:
             ttk.Label(self.instrument_frame, text="No specific regime data found for this instrument.").grid(row=row, column=0, columnspan=2, padx=10, pady=1)
             row += 1


        # Display Energy/Reset Points Summary
        ttk.Label(self.instrument_frame, text="Key Levels:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)
        row += 1
        energy = self.energy_levels.get(symbol, [])
        resets = self.reset_points.get(symbol, [])

        if energy:
            ttk.Label(self.instrument_frame, text="Energy Levels:").grid(row=row, column=0, padx=(10, 5), pady=1, sticky=tk.NW)
            energy_text = ", ".join([f"${e['price']:.2f}" for e in energy[:3] if isinstance(e.get('price'), (int, float))]) # Show top 3
            ttk.Label(self.instrument_frame, text=energy_text if energy_text else "N/A").grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
            row += 1
        else:
             ttk.Label(self.instrument_frame, text="Energy Levels:").grid(row=row, column=0, padx=(10, 5), pady=1, sticky=tk.W)
             ttk.Label(self.instrument_frame, text="N/A").grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
             row += 1


        if resets:
            ttk.Label(self.instrument_frame, text="Reset Points:").grid(row=row, column=0, padx=(10, 5), pady=1, sticky=tk.NW)
            reset_text = ", ".join([f"${r['price']:.2f}" for r in resets[:3] if isinstance(r.get('price'), (int, float))]) # Show top 3
            ttk.Label(self.instrument_frame, text=reset_text if reset_text else "N/A").grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
            row += 1
        else:
             ttk.Label(self.instrument_frame, text="Reset Points:").grid(row=row, column=0, padx=(10, 5), pady=1, sticky=tk.W)
             ttk.Label(self.instrument_frame, text="N/A").grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
             row += 1


        # Placeholder for other instrument context (e.g., IV Rank, Correlation...)
        # ttk.Label(self.instrument_frame, text="Other Context:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)
        # row += 1
        # ttk.Label(self.instrument_frame, text="IV Rank: ...").grid(row=row, column=0, padx=(10,5), pady=1, sticky=tk.W)


    def create_regime_visualization(self):
        """Create the regime calendar/history visualization."""
        logger.debug("Creating regime visualization.")
        # Clear previous chart and placeholder
        if self.regime_canvas: self.regime_canvas.get_tk_widget().destroy()
        if self.regime_fig: plt.close(self.regime_fig)
        self.regime_fig, self.regime_canvas = None, None
        for widget in self.regime_viz_placeholder.winfo_children(): widget.destroy()
        self.regime_viz_placeholder.pack_forget()

        if not self.regime_history:
             logger.warning("Regime history data not loaded. Cannot create visualization.")
             self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
             ttk.Label(self.regime_viz_placeholder, text="Regime history data not loaded. Use 'Load Tracker Data'.", anchor=tk.CENTER).pack(expand=True)
             return

        try:
            # --- Data Preparation ---
            # Select a symbol (e.g., SPY or the first one available)
            vis_symbol = "SPY" if "SPY" in self.regime_history else list(self.regime_history.keys())[0] if self.regime_history else None

            if not vis_symbol:
                 logger.warning("No symbol found in regime history for visualization.")
                 self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
                 ttk.Label(self.regime_viz_placeholder, text="No symbol data in loaded regime history.", anchor=tk.CENTER).pack(expand=True)
                 return

            history = self.regime_history[vis_symbol]
            dates = sorted(history.keys())
            if not dates:
                  logger.warning(f"No dates found in regime history for {vis_symbol}.")
                  self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
                  ttk.Label(self.regime_viz_placeholder, text=f"No historical data points for {vis_symbol}.", anchor=tk.CENTER).pack(expand=True)
                  return


            # Limit history (e.g., last 60 days)
            start_date_str = (datetime.datetime.strptime(dates[-1], '%Y-%m-%d') - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
            display_dates = [d for d in dates if d >= start_date_str]
            if not display_dates: display_dates = dates[-30:] # Fallback to last 30 points

            regimes = [history[d]['primary_regime'] for d in display_dates]
            datetimes = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in display_dates]

            # --- Create Colormap ---
            unique_regimes = sorted(list(set(regimes)))
            colors = plt.cm.get_cmap('tab10', len(unique_regimes)) # Use a qualitative colormap
            regime_color_map = {regime: colors(i) for i, regime in enumerate(unique_regimes)}

            # --- Create Plot ---
            self.regime_fig, ax = plt.subplots(figsize=(10, 2)) # Wider, shorter figure
            self.regime_fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.1)

            # Plot colored bars for each day
            for i in range(len(datetimes)):
                ax.barh(y=0, width=1, left=i, color=regime_color_map.get(regimes[i], 'grey'), height=1)

            ax.set_yticks([]) # Hide y-axis
            ax.set_xlim(0, len(datetimes))
            ax.set_title(f'{vis_symbol} Market Regime History (Last {len(display_dates)} Days)', fontsize=10)

            # Format x-axis with dates (show fewer labels)
            tick_indices = np.linspace(0, len(datetimes) - 1, num=min(len(datetimes), 7), dtype=int) # Show ~7 dates
            ax.set_xticks([i + 0.5 for i in tick_indices]) # Center ticks in bars
            ax.set_xticklabels([display_dates[i] for i in tick_indices], rotation=30, ha='right', fontsize=8)

            # Add legend for colors
            handles = [plt.Rectangle((0, 0), 1, 1, color=regime_color_map.get(reg, 'grey')) for reg in unique_regimes]
            ax.legend(handles, unique_regimes, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(unique_regimes), 5), fontsize=8, frameon=False)
            self.regime_fig.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout


            # Embed chart in Tkinter
            parent_frame = self.regime_viz_placeholder.master
            self.regime_canvas = FigureCanvasTkAgg(self.regime_fig, master=parent_frame)
            self.regime_canvas_widget = self.regime_canvas.get_tk_widget()
            self.regime_canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.regime_canvas.draw()

        except Exception as e:
             logger.error(f"Error creating regime visualization: {e}", exc_info=True)
             self.regime_viz_placeholder.pack(fill=tk.BOTH, expand=True)
             ttk.Label(self.regime_viz_placeholder, text=f"Error creating regime chart: {e}", foreground="red", anchor=tk.CENTER).pack(expand=True)

    def update_duration_info(self, strategy_name):
        """Update the Trade Duration Analysis section."""
         # Clear existing widgets
        for widget in self.duration_info_frame.winfo_children(): widget.destroy()

        if not strategy_name or strategy_name == "Unknown":
             ttk.Label(self.duration_info_frame, text="Select a recommendation with a valid strategy.").pack(pady=10)
             return

        duration_data = self.trade_durations.get(strategy_name)

        if not duration_data or not isinstance(duration_data, dict):
             ttk.Label(self.duration_info_frame, text=f"No duration data found for strategy: {strategy_name}.").pack(pady=10)
             return

        ttk.Label(self.duration_info_frame, text=f"Strategy: {strategy_name}", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)

        details = {
            "Avg. Hold (Days):": f"{duration_data.get('avg_days', 'N/A'):.1f}",
            "Optimal Hold (Days):": str(duration_data.get('optimal_days', 'N/A')),
            "Success Rate (%):": f"{duration_data.get('success_rate', 0) * 100:.1f}%",
            "Data Source:": duration_data.get("source", "loaded").capitalize() # Show if simulated or loaded
        }
        row = 1
        for label, value in details.items():
            if value == "N/A": continue
            ttk.Label(self.duration_info_frame, text=label).grid(row=row, column=0, padx=(0, 5), pady=1, sticky=tk.W)
            ttk.Label(self.duration_info_frame, text=value).grid(row=row, column=1, padx=5, pady=1, sticky=tk.W)
            row += 1

    def create_duration_visualization(self, strategy_name):
        """Create the duration performance visualization."""
        logger.debug(f"Creating duration chart for strategy: {strategy_name}")
        # Clear previous chart and placeholder
        if self.duration_canvas: self.duration_canvas.get_tk_widget().destroy()
        if self.duration_fig: plt.close(self.duration_fig)
        self.duration_fig, self.duration_canvas = None, None
        for widget in self.duration_viz_placeholder.winfo_children(): widget.destroy()
        self.duration_viz_placeholder.pack_forget()

        if not strategy_name or strategy_name == "Unknown":
             logger.warning("No strategy selected for duration visualization.")
             self.duration_viz_placeholder.pack(fill=tk.BOTH, expand=True)
             ttk.Label(self.duration_viz_placeholder, text="Select a recommendation with a valid strategy.", anchor=tk.CENTER).pack(expand=True)
             return

        duration_data = self.trade_durations.get(strategy_name)
        if not duration_data or not isinstance(duration_data, dict) or "completion_profile" not in duration_data:
            logger.warning(f"No completion profile data found for strategy: {strategy_name}")
            self.duration_viz_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.duration_viz_placeholder, text=f"No duration profile data for {strategy_name}.", anchor=tk.CENTER).pack(expand=True)
            return

        try:
            profile = duration_data["completion_profile"]
            if not isinstance(profile, dict) or not profile:
                 raise ValueError("Completion profile is empty or not a dictionary.")

            # Extract days and percentages, sort by day number
            days = []
            percentages = []
            for day_key, percentage in profile.items():
                 try:
                      day_num = int(day_key.split('_')[-1]) # Extract day number
                      if isinstance(percentage, (int, float)):
                          days.append(day_num)
                          percentages.append(float(percentage))
                 except (ValueError, IndexError):
                      logger.warning(f"Could not parse day key '{day_key}' in duration profile for {strategy_name}")
                      continue

            if not days:
                 raise ValueError("No valid days found in completion profile.")

            # Sort by day number
            sorted_indices = np.argsort(days)
            sorted_days = np.array(days)[sorted_indices]
            sorted_percentages = np.array(percentages)[sorted_indices] * 100 # Convert to percentage

            # --- Create Line Chart ---
            self.duration_fig, ax = plt.subplots(figsize=(7, 4))
            self.duration_fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

            ax.plot(sorted_days, sorted_percentages, marker='o', linestyle='-', color='teal')

            ax.set_xlabel('Holding Days')
            ax.set_ylabel('Profit Target Achieved (%)')
            ax.set_title(f'{strategy_name} - Typical Profit Capture Over Time', fontsize=11)
            ax.set_ylim(0, 105) # Percentage from 0 to 100+
            ax.set_xticks(sorted_days) # Ticks for each day in profile
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add optimal day marker if available
            optimal_days = duration_data.get('optimal_days')
            if optimal_days and isinstance(optimal_days, (int, float)):
                 ax.axvline(x=optimal_days, color='darkorange', linestyle='--', lw=1.5, label=f'Optimal: {optimal_days} days')
                 ax.legend(fontsize=9)


            # Embed chart in Tkinter
            parent_frame = self.duration_viz_placeholder.master
            self.duration_canvas = FigureCanvasTkAgg(self.duration_fig, master=parent_frame)
            self.duration_canvas_widget = self.duration_canvas.get_tk_widget()
            self.duration_canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.duration_canvas.draw()

        except Exception as e:
            logger.error(f"Error creating duration visualization for {strategy_name}: {e}", exc_info=True)
            self.duration_viz_placeholder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(self.duration_viz_placeholder, text=f"Error creating duration chart: {e}", foreground="red", anchor=tk.CENTER).pack(expand=True)

    # --- Action Methods ---

    def refresh_data(self):
        """Reload all data and refresh the dashboard."""
        logger.info("Refreshing data...")
        self.update_status("Refreshing all data...", "info")
        # Clear current selection and details
        self.selected_recommendation = None
        self.clear_details_view()
        # Reload data
        self.load_all_data()
        # Repopulate list (which applies filters)
        self.populate_dashboard()
        # Update status
        self.update_status("Data refreshed.", "success")
        logger.info("Data refresh complete.")

    def load_tracker_data(self):
        """Load historical regime data from tracker files."""
        logger.info("Attempting to load tracker data (regime history)...")
        tracker_dir_guess = os.path.join(self.base_dir, "data", "tracker")
        # Show a directory dialog to select tracker data
        tracker_dir = filedialog.askdirectory(
            title="Select Tracker Data Directory (containing regime_history.json)",
            initialdir=tracker_dir_guess if os.path.exists(tracker_dir_guess) else self.data_dir
        )

        if not tracker_dir:
            logger.info("Tracker data loading cancelled by user.")
            return

        try:
            # Look for regime history file
            history_file = os.path.join(tracker_dir, "regime_history.json")
            if os.path.exists(history_file):
                self.update_status(f"Loading regime history from {history_file}...")
                with open(history_file, "r") as f:
                    loaded_history = json.load(f)

                if isinstance(loaded_history, dict):
                     self.regime_history = loaded_history
                     logger.info(f"Loaded regime history for {len(self.regime_history)} instruments")

                     # Optional: Extract detailed transitions if needed elsewhere
                     # self.regime_transitions = self._extract_transitions_from_history(self.regime_history)

                     # Update market regime visualization immediately
                     self.create_regime_visualization()

                     # Update tracker status in status bar
                     self.tracker_status.config(text="Tracker History: Loaded", foreground="green")
                     self.update_status(f"Loaded tracker history from {tracker_dir}")
                else:
                     logger.error(f"Regime history file {history_file} does not contain a dictionary.")
                     self.update_status(f"Invalid format in regime history file.", "error")
                     self.tracker_status.config(text="Tracker History: Load Error", foreground="red")

            else:
                logger.warning(f"Regime history file not found in {tracker_dir}")
                self.update_status(f"No regime history file found in {tracker_dir}", "warning")
                self.tracker_status.config(text="Tracker History: Not Found", foreground="orange")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {history_file}: {e}")
            self.update_status(f"Error reading tracker history file: {e}", "error")
            self.tracker_status.config(text="Tracker History: Load Error", foreground="red")
        except Exception as e:
            logger.error(f"Error loading tracker data from {tracker_dir}: {e}", exc_info=True)
            self.update_status(f"Error loading tracker data: {e}", "error")
            self.tracker_status.config(text="Tracker History: Load Error", foreground="red")

    def _extract_transitions_from_history(self, history_data):
        """Helper to extract regime transitions from loaded history data."""
        transitions_by_symbol = {}
        for symbol, history in history_data.items():
            if not isinstance(history, dict): continue
            symbol_transitions = []
            dates = sorted(history.keys())

            if len(dates) >= 2:
                for i in range(1, len(dates)):
                    prev_date = dates[i-1]
                    curr_date = dates[i]

                    # Check if data for dates exists and is dict
                    prev_data = history.get(prev_date)
                    curr_data = history.get(curr_date)
                    if not isinstance(prev_data, dict) or not isinstance(curr_data, dict): continue

                    prev_regime = prev_data.get("primary_regime")
                    curr_regime = curr_data.get("primary_regime")

                    if prev_regime and curr_regime and prev_regime != curr_regime:
                        symbol_transitions.append({
                            "from_date": prev_date,
                            "to_date": curr_date,
                            "from_regime": prev_regime,
                            "to_regime": curr_regime
                        })
            transitions_by_symbol[symbol] = symbol_transitions
        logger.info(f"Extracted detailed transitions for {len(transitions_by_symbol)} symbols.")
        return transitions_by_symbol


    def update_status(self, message, status_type="info"):
        """Update the status bar with a message and log it."""
        # Ensure UI elements exist before trying to configure them
        if not hasattr(self, 'status_label') or not self.status_label.winfo_exists():
             print(f"Status update (UI not ready): [{status_type.upper()}] {message}")
             return

        try:
            # Set color based on status type
            color = {
                "info": "black",
                "success": "green",
                "warning": "orange", #"#E69F00", # More distinct orange
                "error": "red"
            }.get(status_type, "black")

            # Update status label
            self.status_label.config(text=message, foreground=color)
            self.root.update_idletasks() # Ensure UI update is processed

            # Log the status message
            if status_type == "error":
                logger.error(message)
            elif status_type == "warning":
                logger.warning(message)
            else: # info, success
                logger.info(message)
        except Exception as e:
            # Log error updating status itself, avoid recursive calls
            logger.error(f"Internal error updating status bar: {e}")


# --- Main execution ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        # Optional: Set base directory if script is not in project root
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # project_base_dir = os.path.dirname(script_dir) # Adjust if needed
        # app = IntegratedDashboard(root, base_dir=project_base_dir)
        app = IntegratedDashboard(root) # Assume script is run from project root
        root.mainloop()
    except Exception as e:
         # Log fatal error if GUI fails to initialize
         logger.critical(f"Fatal error starting the dashboard application: {e}", exc_info=True)
         # Show simple error message if possible
         try:
              messagebox.showerror("Dashboard Error", f"A critical error occurred:\n{e}\n\nPlease check the dashboard.log file for details.")
         except Exception:
              print(f"FATAL ERROR: {e}") # Fallback to console print
         sys.exit(1) # Exit with error code

# --- END OF FILE fixed_original.py ---