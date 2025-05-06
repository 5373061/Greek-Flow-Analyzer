import unittest
import pandas as pd
from datetime import datetime, timedelta
from visualizations.energy_flow import create_energy_flow_chart
import plotly.graph_objects as go

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create sample analyzed data
        self.analyzed_data = pd.DataFrame({
            'strike': [440, 450, 460],
            'gamma_contribution': [0.3, 0.5, 0.2],
            'delta_weight': [-50, 100, 50],
            'expiration': [datetime.now() + timedelta(days=30)] * 3
        })
        
    def test_energy_flow_chart(self):
        """Test creation of energy flow visualization"""
        fig = create_energy_flow_chart(self.analyzed_data)
        
        # Verify figure creation
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
        
        # Verify traces
        self.assertEqual(len(fig.data), 2)  # Should have 2 traces
        self.assertEqual(fig.data[0].type, 'bar')  # Gamma contribution
        self.assertEqual(fig.data[1].type, 'scatter')  # Delta weight

if __name__ == '__main__':
    unittest.main()
