# utils/io_logger.py
# -------------------------------------------------------------
from pathlib import Path
from datetime import date
import pandas as pd
import json
import logging

def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with standard configuration
    
    Args:
        name: Name of the logger, typically __name__
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if none exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

log = setup_logger(__name__)

# where the daily CSVs live
OUT_DIR = Path("output/daily_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
def append_daily_snapshot(analysis_results: dict,
                          spot_price: float,
                          as_of: date) -> None:
    """
    Append ONE row per day with the key aggregated-Greek metrics.

    Parameters
    ----------
    analysis_results : dict   output of GreekEnergyFlow.analyze_greek_profiles
    spot_price       : float  last price for the underlying
    as_of            : date   today’s date
    """
    agg = analysis_results.get("aggregated_greeks", {})

    row = {
        "date":         as_of.isoformat(),
        "spot":         spot_price,
        "net_delta":    agg.get("net_delta"),
        "total_gamma":  agg.get("total_gamma"),
        "total_vanna":  agg.get("total_vanna"),
        "total_charm":  agg.get("total_charm"),
        # keep the whole blob if you ever want to re-hydrate it later
        "payload":      json.dumps(analysis_results),
    }

    fn = OUT_DIR / f"snapshots_{as_of.year}.csv"
    pd.DataFrame([row]).to_csv(
        fn,
        mode   = "a",
        index  = False,
        header = not fn.exists()          # write header only once
    )
    log.info("📈  snapshot appended → %s", fn)
# ------------------------------------------------------------------

__all__ = ["append_daily_snapshot"]
