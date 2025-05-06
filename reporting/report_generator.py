from typing import Optional, Dict
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import os

class GreekEnergyReport:
    """Generate detailed reports for Greek energy analysis"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_analysis(self, 
                       analysis_df: pd.DataFrame, 
                       fig: go.Figure,
                       metrics: Dict) -> str:
        """Export complete analysis with ML insights"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = analysis_df['underlying_ticker'].iloc[0]
            base_name = f"{symbol}_analysis_{timestamp}"
            
            # Save visualization
            fig.write_html(os.path.join(self.output_dir, f"{base_name}.html"))
            
            # Export metrics and predictions
            report_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "current_price": float(analysis_df['underlying_price'].iloc[0]),
                "ml_metrics": {
                    "gradient_direction": metrics.get("gradient_direction", "neutral"),
                    "energy_concentration": metrics.get("energy_concentration", 0),
                    "prediction_confidence": metrics.get("confidence", 0)
                },
                "analysis_summary": {
                    "total_gamma": float(analysis_df['gamma_contribution'].sum()),
                    "net_delta_weight": float(analysis_df['delta_weight'].sum()),
                    "strike_coverage": len(analysis_df)
                }
            }
            
            # Save JSON report
            with open(os.path.join(self.output_dir, f"{base_name}.json"), 'w') as f:
                json.dump(report_data, f, indent=2)
                
            return os.path.join(self.output_dir, base_name)
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return ""