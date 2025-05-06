# entropy_analyzer/entropy_analyzer.py

import numpy as np
import pandas as pd
from scipy import stats
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger("greek_flow.entropy")

class EntropyAnalyzer:
    """Analyzes the entropy of Greek distributions to detect energy states and anomalies."""
    
    def __init__(self, options_data=None, historical_data=None, config=None):
        """Initialize the entropy analyzer with options data.
        
        Args:
            options_data (pd.DataFrame): Options chain data with Greeks
            historical_data (pd.DataFrame, optional): Historical data for comparison
            config (dict, optional): Configuration dictionary
        """
        self.options_data = options_data
        self.historical_data = historical_data
        self.config = config or {}
        self.entropy_metrics = {}
        self.anomalies = {}
        
        # Get configuration parameters
        self.viz_enabled = self.config.get("visualization_enabled", False)
        self.viz_dir = self.config.get("visualization_dir", "entropy_viz")
        self.entropy_low = self.config.get("entropy_threshold_low", 30)
        self.entropy_high = self.config.get("entropy_threshold_high", 70)
        self.anomaly_sensitivity = self.config.get("anomaly_detection_sensitivity", 1.5)
        
        # Create visualization directory if needed
        if self.viz_enabled and not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir, exist_ok=True)
    
    def set_options_data(self, options_data):
        """Set the options data for analysis."""
        self.options_data = options_data
    
    def calculate_shannon_entropy(self, distribution):
        """Calculate Shannon entropy for a distribution of values."""
        # Handle empty or invalid distributions
        if len(distribution) <= 1:
            return 0
            
        # Normalize distribution and avoid zeros
        distribution = np.abs(distribution)
        total = np.sum(distribution)
        if total == 0:
            return 0
            
        probabilities = distribution / total
        # Remove zeros to avoid log(0) errors
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def calculate_relative_entropy(self, dist1, dist2):
        """Calculate KL divergence between two distributions."""
        # Normalize distributions
        dist1 = np.abs(dist1)
        dist2 = np.abs(dist2)
        
        # Ensure non-zero probabilities
        sum1 = np.sum(dist1)
        sum2 = np.sum(dist2)
        
        if sum1 == 0 or sum2 == 0:
            return np.inf
        
        p1 = dist1 / sum1
        p2 = dist2 / sum2
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        p1 = p1 + epsilon
        p2 = p2 + epsilon
        
        # Renormalize
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        
        # Calculate KL divergence: sum(p1 * log(p1/p2))
        kl_div = np.sum(p1 * np.log2(p1 / p2))
        return kl_div
    
    def calculate_cross_entropy(self, dist1, dist2):
        """Calculate cross-entropy between two distributions."""
        # Normalize distributions
        dist1 = np.abs(dist1)
        dist2 = np.abs(dist2)
        
        # Ensure non-zero probabilities
        sum1 = np.sum(dist1)
        sum2 = np.sum(dist2)
        
        if sum1 == 0 or sum2 == 0:
            return np.inf
        
        p1 = dist1 / sum1
        p2 = dist2 / sum2
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        p2 = p2 + epsilon
        
        # Renormalize
        p2 = p2 / np.sum(p2)
        
        # Calculate cross-entropy: -sum(p1 * log(p2))
        cross_ent = -np.sum(p1 * np.log2(p2))
        return cross_ent
    
    def calculate_cross_entropy(self, greek1, greek2):
        """
        Calculate cross-entropy (KL divergence) between two Greek distributions.
        """
        if self.options_data is None:
            return np.nan
        
        if greek1 not in self.options_data.columns or greek2 not in self.options_data.columns:
            return np.nan
        
        # Get values and ensure they're numeric
        dist1 = self.options_data[greek1].dropna().values
        dist2 = self.options_data[greek2].dropna().values
        
        # Ensure both arrays have same length
        min_len = min(len(dist1), len(dist2))
        if min_len == 0:
            return np.nan
        
        dist1 = dist1[:min_len]
        dist2 = dist2[:min_len]
        
        # Ensure values are numeric
        if not np.issubdtype(dist1.dtype, np.number) or not np.issubdtype(dist2.dtype, np.number):
            return np.nan
        
        # Normalize to probability distributions
        dist1 = np.abs(dist1)
        dist2 = np.abs(dist2)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        dist1 = dist1 + epsilon
        dist2 = dist2 + epsilon
        
        # Normalize
        dist1 = dist1 / np.sum(dist1)
        dist2 = dist2 / np.sum(dist2)
        
        # Calculate KL divergence
        kl_div = np.sum(dist1 * np.log2(dist1 / dist2))
        
        return kl_div
    
    def analyze_greek_entropy(self):
        """Analyze the entropy of Greek distributions."""
        if self.options_data is None or len(self.options_data) == 0:
            logger.warning("No options data provided for entropy analysis")
            return {"error": "No options data available"}
        
        try:
            # Get available Greek columns
            available_greeks = self.config.get("available_greeks", 
                                              ["delta", "gamma", "theta", "vega", "vanna", "charm"])
            
            # Filter to only include columns that actually exist in the data
            greek_cols = [col for col in available_greeks if col in self.options_data.columns]
            
            if not greek_cols:
                logger.warning("No Greek columns found in options data")
                return {"error": "No Greek columns found"}
            
            # Calculate entropy for each Greek distribution
            greek_entropy = {}
            for greek in greek_cols:
                values = self.options_data[greek].dropna().values
                if len(values) > 0:
                    entropy = self.calculate_shannon_entropy(values)
                    max_entropy = np.log2(len(values))  # Maximum possible entropy
                    normalized_entropy = (entropy / max_entropy * 100) if max_entropy > 0 else 0
                    
                    greek_entropy[greek] = {
                        "entropy": entropy,
                        "normalized_entropy": normalized_entropy,
                        "sample_count": len(values)
                    }
            
            # Calculate cross-entropy between different Greeks
            cross_entropy = {}
            for i, greek1 in enumerate(greek_cols):
                for greek2 in greek_cols[i+1:]:
                    values1 = self.options_data[greek1].dropna().values
                    values2 = self.options_data[greek2].dropna().values
                    
                    # Ensure both arrays have same length by truncating
                    min_len = min(len(values1), len(values2))
                    if min_len > 0:
                        cross_ent = self.calculate_relative_entropy(
                            values1[:min_len], values2[:min_len]
                        )
                        cross_entropy[f"{greek1}_{greek2}"] = cross_ent
            
            # Calculate entropy by strike price (energy concentration)
            strike_entropy = {}
            if "strike" in self.options_data.columns:
                for greek in greek_cols:
                    # Group by strike
                    strike_groups = self.options_data.groupby("strike")[greek].apply(list)
                    
                    # Calculate entropy for each strike group
                    strike_values = {}
                    for strike, values in strike_groups.items():
                        values = np.array([v for v in values if not np.isnan(v)])
                        if len(values) > 0:
                            strike_values[strike] = self.calculate_shannon_entropy(values)
                    
                    # Calculate entropy gradient across strikes
                    if len(strike_values) > 1:
                        strikes = sorted(strike_values.keys())
                        entropies = [strike_values[s] for s in strikes]
                        
                        # Use linear regression to calculate gradient
                        slope, _, r_value, p_value, _ = stats.linregress(strikes, entropies)
                        
                        strike_entropy[greek] = {
                            "values": strike_values,
                            "gradient": slope,
                            "r_squared": r_value ** 2,
                            "p_value": p_value
                        }
                        
                        # Visualize entropy by strike if enabled
                        if self.viz_enabled:
                            self._visualize_strike_entropy(greek, strikes, entropies, slope)
            
            # Determine energy concentration state based on overall entropy
            avg_norm_entropy = np.mean([v["normalized_entropy"] for v in greek_entropy.values()])
            
            if avg_norm_entropy < self.entropy_low:
                energy_state = "High Energy Concentration (Low Entropy)"
                state_description = "Energy is highly concentrated, indicating potential pressure points"
            elif avg_norm_entropy < self.entropy_high:
                energy_state = "Balanced Energy (Medium Entropy)"
                state_description = "Energy is moderately distributed, indicating a balanced regime"
            else:
                energy_state = "Dispersed Energy (High Entropy)"
                state_description = "Energy is widely dispersed, indicating diffuse pressure"
            
            # Determine energy direction based on entropy gradients
            gradients = [data["gradient"] for data in strike_entropy.values()]
            avg_gradient = np.mean(gradients) if gradients else 0
            
            if avg_gradient < -0.01:
                energy_direction = "Concentrating Energy"
                direction_description = "Energy is becoming more concentrated as prices increase"
            elif avg_gradient > 0.01:
                energy_direction = "Dispersing Energy"
                direction_description = "Energy is becoming more dispersed as prices increase"
            else:
                energy_direction = "Stable Energy Flow"
                direction_description = "Energy concentration is relatively stable across prices"
            
            # Store all entropy metrics
            self.entropy_metrics = {
                "greek_entropy": greek_entropy,
                "cross_entropy": cross_entropy,
                "strike_entropy": strike_entropy,
                "energy_state": {
                    "state": energy_state,
                    "description": state_description,
                    "average_normalized_entropy": avg_norm_entropy
                },
                "energy_direction": {
                    "direction": energy_direction,
                    "description": direction_description,
                    "average_gradient": avg_gradient
                }
            }
            
            # Add a simple energy_state_string for compatibility with trade recommendation system
            self.entropy_metrics["energy_state_string"] = energy_state
            
            # Visualize overall entropy distribution if enabled
            if self.viz_enabled:
                self._visualize_entropy_distribution(greek_entropy)
            
            return self.entropy_metrics
            
        except Exception as e:
            logger.error(f"Error in entropy analysis: {str(e)}")
            return {"error": str(e)}
    
    def detect_anomalies(self):
        """Detect anomalies in Greek entropy distributions."""
        if not self.entropy_metrics:
            logger.warning("No entropy metrics available for anomaly detection")
            return {"anomalies": {}, "anomaly_count": 0}
        
        try:
            # Check for extremely low entropy in any Greek (high concentration)
            anomalies = {}
            greek_entropy = self.entropy_metrics.get("greek_entropy", {})
            
            for greek, data in greek_entropy.items():
                norm_entropy = data["normalized_entropy"]
                if norm_entropy < self.entropy_low / 2:  # Extremely concentrated
                    anomalies[f"{greek}_high_concentration"] = f"Extremely concentrated {greek} energy"
                elif norm_entropy > self.entropy_high + (100 - self.entropy_high) / 2:  # Extremely dispersed
                    anomalies[f"{greek}_high_dispersion"] = f"Extremely dispersed {greek} energy"
            
            # Check for unusual cross-entropy values
            cross_entropy = self.entropy_metrics.get("cross_entropy", {})
            for pair, value in cross_entropy.items():
                if value > 2.0 * self.anomaly_sensitivity:  # High divergence
                    greeks = pair.split("_")
                    anomalies[f"{pair}_divergence"] = f"Unusual relationship between {greeks[0]} and {greeks[1]}"
            
            # Check for extreme entropy gradients
            strike_entropy = self.entropy_metrics.get("strike_entropy", {})
            for greek, data in strike_entropy.items():
                gradient = data["gradient"]
                r_squared = data["r_squared"]
                p_value = data["p_value"]
                
                # Only consider statistically significant gradients
                if r_squared > 0.3 and p_value < 0.05:
                    if gradient < -0.05 * self.anomaly_sensitivity:
                        anomalies[f"{greek}_rapid_concentration"] = f"Rapidly concentrating {greek} energy"
                    elif gradient > 0.05 * self.anomaly_sensitivity:
                        anomalies[f"{greek}_rapid_dispersion"] = f"Rapidly dispersing {greek} energy"
            
            # Store anomalies for reference
            self.anomalies = {
                "anomalies": anomalies,
                "anomaly_count": len(anomalies)
            }
            
            # Visualize anomalies if enabled and anomalies found
            if self.viz_enabled and anomalies:
                self._visualize_anomalies(anomalies)
            
            return self.anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {"anomalies": {}, "anomaly_count": 0, "error": str(e)}
    
    def generate_entropy_report(self):
        """Generate a comprehensive report of entropy analysis."""
        if not self.entropy_metrics:
            return "No entropy analysis results available."
        
        # Create the report
        report = "=== ENTROPY ANALYSIS REPORT ===\n\n"
        
        # Energy state
        energy_state = self.entropy_metrics.get("energy_state", {})
        report += f"Energy State: {energy_state.get('state', 'Unknown')}\n"
        report += f"Description: {energy_state.get('description', 'N/A')}\n"
        report += f"Average Normalized Entropy: {energy_state.get('average_normalized_entropy', 0):.2f}%\n\n"
        
        # Energy direction
        energy_dir = self.entropy_metrics.get("energy_direction", {})
        report += f"Energy Direction: {energy_dir.get('direction', 'Unknown')}\n"
        report += f"Description: {energy_dir.get('description', 'N/A')}\n\n"
        
        # Greek entropy values
        report += "Greek Entropy Values:\n"
        for greek, data in self.entropy_metrics.get("greek_entropy", {}).items():
            report += f"  {greek.capitalize()}: {data.get('normalized_entropy', 0):.2f}% "
            report += f"(sample size: {data.get('sample_count', 0)})\n"
        
        # Strike entropy gradients
        strike_entropy = self.entropy_metrics.get("strike_entropy", {})
        if strike_entropy:
            report += "\nEntropy Gradients Across Strikes:\n"
            for greek, data in strike_entropy.items():
                gradient = data.get("gradient", 0)
                r_squared = data.get("r_squared", 0)
                report += f"  {greek.capitalize()}: {gradient:.6f} "
                report += f"(R²: {r_squared:.2f})\n"
        
        # Anomalies
        if hasattr(self, 'anomalies') and self.anomalies:
            anomalies = self.anomalies.get("anomalies", {})
            if anomalies:
                report += f"\nAnomalies Detected ({len(anomalies)}):\n"
                for key, description in anomalies.items():
                    report += f"  - {description}\n"
            else:
                report += "\nNo significant anomalies detected.\n"
        
        # Cross-entropy insights
        cross_entropy = self.entropy_metrics.get("cross_entropy", {})
        if cross_entropy:
            report += "\nGreek Relationship Insights:\n"
            for pair, value in sorted(cross_entropy.items(), key=lambda x: x[1], reverse=True)[:3]:
                greeks = pair.split("_")
                report += f"  - {greeks[0].capitalize()} and {greeks[1].capitalize()}: "
                
                if value < 0.5:
                    report += f"Highly aligned distributions (KL: {value:.2f})\n"
                elif value < 1.0:
                    report += f"Moderately aligned distributions (KL: {value:.2f})\n"
                else:
                    report += f"Divergent distributions (KL: {value:.2f})\n"
        
        # Trading implications
        report += "\nTrading Implications:\n"
        energy_state_val = energy_state.get("average_normalized_entropy", 50)
        direction = energy_dir.get("direction", "Stable")
        
        if energy_state_val < self.entropy_low:
            if "Concentrating" in direction:
                report += "  - High energy concentration is increasing, suggesting potential breakout point\n"
                report += "  - Consider positions that benefit from accelerated price movement\n"
            else:
                report += "  - High energy concentration indicates potential support/resistance levels\n"
                report += "  - Consider positions that capitalize on strong reaction at these levels\n"
        elif energy_state_val > self.entropy_high:
            report += "  - Dispersed energy suggests range-bound conditions\n"
            report += "  - Consider mean-reversion strategies or positions that benefit from time decay\n"
        else:
            report += "  - Balanced energy state indicates mixed conditions\n"
            report += "  - Consider both directional and non-directional strategies with defined risk\n"
        
        return report
    
    def _visualize_strike_entropy(self, greek, strikes, entropies, slope):
        """Create visualization of entropy distribution across strikes."""
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, entropies, 'o-', label=f'Entropy')
        
        # Add regression line
        x_min, x_max = min(strikes), max(strikes)
        y_min = slope * x_min + (entropies[0] - slope * strikes[0])
        y_max = slope * x_max + (entropies[0] - slope * strikes[0])
        plt.plot([x_min, x_max], [y_min, y_max], 'r--', 
                label=f'Gradient: {slope:.4f}')
        
        plt.title(f'{greek.capitalize()} Entropy Across Strike Prices')
        plt.xlabel('Strike Price')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.viz_dir, f"{greek}_strike_entropy_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Strike entropy visualization saved to {fig_path}")
    
    def _visualize_entropy_distribution(self, greek_entropy):
        """Create visualization of entropy distribution across Greeks."""
        if not greek_entropy:
            return
            
        labels = []
        values = []
        
        for greek, data in greek_entropy.items():
            labels.append(greek.capitalize())
            values.append(data["normalized_entropy"])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color='skyblue')
        
        # Add threshold lines
        plt.axhline(y=self.entropy_low, color='g', linestyle='--', 
                   label=f'Low Entropy Threshold ({self.entropy_low}%)')
        plt.axhline(y=self.entropy_high, color='r', linestyle='--',
                   label=f'High Entropy Threshold ({self.entropy_high}%)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.title('Normalized Entropy Distribution Across Greeks')
        plt.ylabel('Normalized Entropy (%)')
        plt.ylim(0, 105)  # Leave room for text labels
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.viz_dir, f"greek_entropy_distribution_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Entropy distribution visualization saved to {fig_path}")
    
    def _visualize_anomalies(self, anomalies):
        """Create visualization highlighting detected anomalies."""
        if not anomalies:
            return
            
        # Prepare data for visualization
        anomaly_types = {}
        for key, description in anomalies.items():
            parts = key.split('_')
            greek = parts[0]
            
            if greek not in anomaly_types:
                anomaly_types[greek] = []
                
            anomaly_types[greek].append(description)
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        # Create a grid with Greek entropy values as baseline
        greek_entropy = self.entropy_metrics.get("greek_entropy", {})
        labels = []
        values = []
        colors = []
        
        for greek, data in greek_entropy.items():
            labels.append(greek.capitalize())
            values.append(data["normalized_entropy"])
            
            # Color based on anomaly
            if greek in anomaly_types:
                colors.append('red')
            else:
                colors.append('skyblue')
        
        # Create the bar chart
        bars = plt.bar(labels, values, color=colors)
        
        # Add threshold lines
        plt.axhline(y=self.entropy_low, color='g', linestyle='--', 
                   label=f'Low Entropy Threshold ({self.entropy_low}%)')
        plt.axhline(y=self.entropy_high, color='r', linestyle='--',
                   label=f'High Entropy Threshold ({self.entropy_high}%)')
        
        # Add annotations for anomalies
        for i, (label, value) in enumerate(zip(labels, values)):
            greek = label.lower()
            if greek in anomaly_types:
                plt.annotate('\n'.join(anomaly_types[greek]),
                            xy=(i, value),
                            xytext=(i, value + 20),
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.title('Entropy Anomalies Detection')
        plt.ylabel('Normalized Entropy (%)')
        plt.ylim(0, 110)  # Leave room for annotations
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.viz_dir, f"entropy_anomalies_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Anomalies visualization saved to {fig_path}")





