import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class PatternVisualizer:
    """
    Visualizes ordinal patterns in Greek metrics.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'delta': '#1f77b4',
            'gamma': '#ff7f0e',
            'theta': '#2ca02c',
            'vega': '#d62728',
            'vanna': '#9467bd',
            'charm': '#8c564b'
        }
    
    def plot_greek_patterns(self, data: pd.DataFrame, patterns: Dict[str, List[Tuple]], 
                          save_path: str = None):
        """
        Plot Greek data with recognized patterns.
        
        Args:
            data: DataFrame with Greek data
            patterns: Dictionary of patterns for each Greek
            save_path: Path to save the plot
        """
        if data.empty:
            logger.warning("No data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(patterns), 1, figsize=self.figsize, sharex=True)
        
        # If only one Greek, convert axes to list
        if len(patterns) == 1:
            axes = [axes]
        
        # Plot each Greek
        for i, (greek, pattern_list) in enumerate(patterns.items()):
            ax = axes[i]
            
            # Check if normalized version exists
            norm_greek = f'norm_{greek}'
            if norm_greek in data.columns:
                column = norm_greek
            elif greek in data.columns:
                column = greek
            else:
                continue
            
            # Plot Greek data
            ax.plot(data[column], color=self.colors.get(greek, 'blue'), label=greek)
            
            # Mark pattern windows
            for idx, pattern in pattern_list:
                # Mark pattern window
                ax.axvspan(idx, idx + 2, alpha=0.2, color='gray')
                
                # Add pattern annotation
                ax.annotate(f"{pattern}", 
                           xy=(idx + 1, data[column].iloc[idx + 1]),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8)
            
            ax.set_ylabel(greek)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Set title and labels
        plt.suptitle('Greek Ordinal Patterns', fontsize=16)
        plt.xlabel('Time')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()