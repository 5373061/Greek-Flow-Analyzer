# visualization/formatters.py
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_full_report(
    formatted_support: str,
    formatted_resistance: str,
    greek_zones: Dict[str, str],
    trade_context: Dict[str, str],
    spot_price: float
) -> str:
    """Assembles all formatted sections into the final text report."""
    parts = []
    parts.append(f"Greek Energy Levels Analysis (Spot Price: ${spot_price:.2f})\n")
    parts.append("=" * 60 + "\n")
    parts.append("--- RESISTANCE LEVELS ---\n")
    parts.append(formatted_resistance + "\n")
    parts.append("-" * 60 + "\n")
    parts.append("--- SUPPORT LEVELS ---\n")
    parts.append(formatted_support + "\n")
    parts.append("-" * 60 + "\n")
    parts.append("--- GREEK CONCENTRATION ZONES ---\n")
    parts.append("GAMMA CONCENTRATION ZONES: " + greek_zones.get('gamma_zones', "N/A") + "\n")
    parts.append("VANNA EXPOSURE (IV Impact): " + greek_zones.get('vanna_exposure', "N/A") + "\n")
    parts.append("CHARM DECAY IMPACT (Time Decay): " + greek_zones.get('charm_decay', "N/A") + "\n")
    parts.append("-" * 60 + "\n")
    parts.append("--- TRADE CONTEXT ---\n")
    parts.append("PRICE ACTION IMPLICATIONS: " + trade_context.get('price_implications', "N/A") + "\n")
    if 'hedging_behavior' in trade_context:
        parts.append("HEDGING BEHAVIOR: " + trade_context.get('hedging_behavior', "N/A") + "\n")
    parts.append("=" * 60 + "\n")
    return "".join(parts)


# Alias for backward compatibility
_generate_full_report = generate_full_report


class GreekFormatter:
    """Utility class for formatting Greek Energy Flow analysis results."""

    @staticmethod
    def format_results(
        analysis_results: Dict[str, Any],
        spot_price_report: float
    ) -> Dict[str, Any]:
        """
        Take the raw analysis_results and a spot_price, and return a dict of:
          - structured tables (reset_points, market_regime, etc.)
          - a full_report text string assembled via generate_full_report
        """
        formatted: Dict[str, Any] = {}
        try:
            # --- Reset Points ---
            rps = analysis_results.get('reset_points', [])
            formatted['reset_points'] = [
                {
                    'price': f"{pt.get('price', np.nan):.2f}",
                    'type': pt.get('type', 'N/A'),
                    'significance': f"{pt.get('significance', 0)*100:.1f}%",
                    'time_frame': pt.get('time_frame', 'N/A'),
                    'factors': ", ".join(
                        f"{k}: {v*100:.1f}%"
                        for k, v in pt.get('factors', {}).items()
                    )
                }
                for pt in rps
            ]

            # --- Market Regime ---
            mr = analysis_results.get('market_regime', {})
            if mr:
                formatted['market_regime'] = {
                    'primary': mr.get('primary_label', 'N/A'),
                    'secondary': mr.get('secondary_label', 'N/A'),
                    'volatility': f"{mr.get('volatility_regime', 'N/A')} Volatility",
                    'dominant_greek': mr.get('dominant_greek', 'N/A'),
                    'metrics': {
                        'delta':    f"{mr.get('greek_magnitudes', {}).get('normalized_delta', np.nan):.3f}",
                        'gamma':    f"{mr.get('greek_magnitudes', {}).get('total_gamma', np.nan):.5f}",
                        'vanna':    f"{mr.get('greek_magnitudes', {}).get('total_vanna', np.nan):.5f}",
                        'charm':    f"{mr.get('greek_magnitudes', {}).get('total_charm', np.nan):.5f}",
                    }
                }

            # --- Energy Levels ---
            els = analysis_results.get('energy_levels', [])
            formatted['energy_levels'] = [
                {
                    'price':      f"{lvl.get('price', np.nan):.2f}",
                    'type':       lvl.get('type','N/A'),
                    'strength':   f"{lvl.get('strength', 0)*100:.1f}%",
                    'direction':  lvl.get('direction','N/A'),
                    'components': lvl.get('components', 1)
                }
                for lvl in els
            ]

            # --- Greek Anomalies ---
            gas = analysis_results.get('greek_anomalies', [])
            formatted['greek_anomalies'] = [
                {
                    'type':        an.get('type','N/A'),
                    'severity':    f"{an.get('severity', 0)*100:.1f}%",
                    'description': an.get('description','N/A'),
                    'implication': an.get('implication','N/A')
                }
                for an in gas
            ]

            # Format projection data if available
            for key in ['vanna_projections', 'charm_projections']:
                proj = analysis_results.get(key, {})
                if proj and 'price_points' in proj:
                    formatted[key] = {
                        'price_points': [f"{p:.2f}" for p in proj['price_points']],
                        'projections': {
                            k: [f"{float(val):.6f}" for val in vs]
                            for k, vs in proj.get('projections', {}).items()
                        }
                    }

            # Format levels for support/resistance
            formatted_support_str = format_levels(els, "support")
            formatted_resistance_str = format_levels(els, "resistance")

            # Get or create zones and context reports
            zones_report = analysis_results.get(
                'greek_zones',
                {'gamma_zones': 'N/A', 'vanna_exposure': 'N/A', 'charm_decay': 'N/A'}
            )
            context_report = analysis_results.get(
                'trade_context',
                {'price_implications': 'N/A'}
            )

            # Generate the full report
            formatted['full_report'] = generate_full_report(
                formatted_support_str,
                formatted_resistance_str,
                zones_report,
                context_report,
                spot_price_report
            )

        except Exception as e:
            logger.error(
                f"Error in GreekFormatter.format_results: {e}",
                exc_info=True
            )
            formatted['full_report'] = "Error generating full text report."

        return formatted


# Class alias for backward compatibility
GreekEnergyAnalyzer = GreekFormatter


def format_levels(levels, direction):
    """Format energy levels for display with specific direction."""
    if not levels:
        return "N/A"
    
    picks = []
    for lvl in levels:
        lvl_dir = str(lvl.get("direction", "")).lower()
        if direction.lower() in lvl_dir:
            price = lvl.get("price", "")
            strength = lvl.get("strength", "")
            picks.append(f"{price} ({strength})")
    
    return ", ".join(picks) if picks else "N/A"
