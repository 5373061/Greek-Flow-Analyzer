"""
Unified Trade Format Utility

This module provides functions to standardize trade recommendation formats
and ensure they include proper trade context information.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def standardize_trade_context(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize trade context format to ensure compatibility with dashboard.
    
    Args:
        trade_data: The trade recommendation data
        
    Returns:
        Dict with standardized trade context
    """
    # Create a copy to avoid modifying the original
    result = trade_data.copy()
    
    # Check if TradeContext already exists in standardized format
    if 'TradeContext' in result and isinstance(result['TradeContext'], dict):
        # Already in new format, just ensure all fields exist
        context = result['TradeContext']
    elif 'trade_context' in result and isinstance(result['trade_context'], dict):
        # Old format, convert to new format
        context = result['trade_context']
        result['TradeContext'] = context
        # Keep old key for backward compatibility
    else:
        # No context found, create minimal context
        context = {}
        result['TradeContext'] = context
    
    # Ensure market regime exists and is properly formatted
    if 'market_regime' not in context or not isinstance(context['market_regime'], dict):
        # Convert string market regime to dict if needed
        if isinstance(context.get('market_regime'), str):
            primary_regime = context['market_regime']
            context['market_regime'] = {
                'primary': primary_regime,
                'volatility': context.get('volatility_regime', 'Normal'),
                'confidence': 0.5  # Default confidence
            }
        else:
            # Create default market regime
            context['market_regime'] = {
                'primary': 'Unknown',
                'volatility': context.get('volatility_regime', 'Normal'),
                'confidence': 0.0
            }
    
    # Ensure other required fields exist
    required_fields = {
        'volatility_regime': 'Normal',
        'dominant_greek': '',
        'energy_state': '',
        'entropy_score': 0.0,
        'support_levels': [],
        'resistance_levels': [],
        'greek_metrics': {},
        'anomalies': [],
        'hold_time_days': 0,
        'confidence_score': 0.0
    }
    
    for field, default_value in required_fields.items():
        if field not in context:
            context[field] = default_value
    
    return result

def convert_recommendations_to_unified_format(input_file: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert trade recommendations to unified format with standardized trade context.
    
    Args:
        input_file: Path to input JSON file with recommendations
        output_file: Optional path to output JSON file
        
    Returns:
        List of standardized recommendations
    """
    try:
        # Load recommendations
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle both list and single recommendation formats
        if isinstance(data, list):
            recommendations = data
        else:
            # Single recommendation object
            recommendations = [data]
        
        # Standardize each recommendation
        standardized_recs = [standardize_trade_context(rec) for rec in recommendations]
        
        # Save to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(standardized_recs, f, indent=2)
            logger.info(f"Saved {len(standardized_recs)} standardized recommendations to {output_file}")
        
        return standardized_recs
    
    except Exception as e:
        logger.error(f"Error converting recommendations to unified format: {e}", exc_info=True)
        return []

def batch_convert_recommendations(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Batch convert all recommendation files in a directory to unified format.
    
    Args:
        input_dir: Directory containing recommendation JSON files
        output_dir: Optional directory for output files (defaults to input_dir)
        
    Returns:
        Dict with counts of processed files and recommendations
    """
    if not output_dir:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'processed_files': 0,
        'total_recommendations': 0,
        'errors': 0
    }
    
    try:
        # Find all JSON files in input directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        for file in json_files:
            try:
                # Skip files that already have the unified_ prefix
                if file.startswith('unified_'):
                    logger.info(f"Skipping already processed file: {file}")
                    continue
                    
                input_path = os.path.join(input_dir, file)
                output_path = os.path.join(output_dir, f"unified_{file}")
                
                # Convert file
                recs = convert_recommendations_to_unified_format(input_path, output_path)
                
                results['processed_files'] += 1
                results['total_recommendations'] += len(recs)
                
                logger.info(f"Processed {file}: {len(recs)} recommendations")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                results['errors'] += 1
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch conversion: {e}", exc_info=True)
        results['errors'] += 1
        return results

def organize_recommendations_for_dashboard(input_dir: str, output_dir: str = None) -> Dict[str, int]:
    """
    Organize unified recommendations into a dashboard-friendly structure.
    
    Args:
        input_dir: Directory containing unified recommendation files
        output_dir: Optional directory for organized files (defaults to input_dir/recommendations)
        
    Returns:
        Dict with counts of organized files
    """
    if not output_dir:
        output_dir = os.path.join(input_dir, "recommendations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'organized_files': 0,
        'symbols_processed': set()
    }
    
    # Define non-stock symbols to exclude
    excluded_symbols = {"ANALYSIS", "PATTERN", "UNIFIED", "REGIME", "TEST", "MARKET"}
    
    try:
        # Find all unified JSON files
        unified_files = [f for f in os.listdir(input_dir) if f.startswith('unified_') and f.endswith('.json')]
        
        for file in unified_files:
            try:
                input_path = os.path.join(input_dir, file)
                
                # Load the recommendation
                with open(input_path, 'r') as f:
                    recommendations = json.load(f)
                
                # Handle both list and single recommendation formats
                if not isinstance(recommendations, list):
                    recommendations = [recommendations]
                
                # Process each recommendation
                for rec in recommendations:
                    # Extract symbol
                    symbol = rec.get('symbol', '')
                    
                    # Skip excluded symbols
                    if symbol in excluded_symbols:
                        logger.debug(f"Skipping excluded symbol: {symbol}")
                        continue
                    
                    # Skip empty symbols
                    if not symbol:
                        logger.warning(f"Skipping recommendation with empty symbol in file {file}")
                        continue
                    
                    # Create symbol directory
                    symbol_dir = os.path.join(output_dir, symbol)
                    os.makedirs(symbol_dir, exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    rec_filename = f"{symbol}_{timestamp}.json"
                    rec_path = os.path.join(symbol_dir, rec_filename)
                    
                    # Save recommendation
                    with open(rec_path, 'w') as f:
                        json.dump(rec, f, indent=2)
                    
                    results['organized_files'] += 1
                    results['symbols_processed'].add(symbol)
                    
                    logger.info(f"Organized recommendation for {symbol}")
            
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
        
        # Create a summary file with all symbols and their strategies
        create_recommendation_summary(output_dir)
        
        logger.info(f"Organization complete: {results['organized_files']} recommendations organized")
        return results
    
    except Exception as e:
        logger.error(f"Error in organization: {e}", exc_info=True)
        return results

def create_recommendation_summary(output_dir: str) -> None:
    """
    Create a summary file with all symbols and their strategies.
    
    Args:
        output_dir: Directory containing organized recommendation files
    """
    try:
        summary = {
            "symbols": [],
            "strategies": set(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Get all symbol directories
        symbol_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        
        for symbol in symbol_dirs:
            symbol_dir = os.path.join(output_dir, symbol)
            json_files = [f for f in os.listdir(symbol_dir) if f.endswith('.json')]
            
            if not json_files:
                continue
                
            # Get the most recent file
            latest_file = sorted(json_files)[-1]
            file_path = os.path.join(symbol_dir, latest_file)
            
            with open(file_path, 'r') as f:
                rec = json.load(f)
            
            strategy = rec.get('strategy_name', '')
            
            symbol_info = {
                "symbol": symbol,
                "strategy": strategy,
                "risk": rec.get('risk_category', 'MEDIUM'),
                "action": rec.get('action', ''),
                "timestamp": os.path.getmtime(file_path)
            }
            
            summary["symbols"].append(symbol_info)
            if strategy:
                summary["strategies"].add(strategy)
        
        # Convert strategies set to list for JSON serialization
        summary["strategies"] = list(summary["strategies"])
        
        # Sort symbols by symbol name
        summary["symbols"] = sorted(summary["symbols"], key=lambda x: x["symbol"])
        
        # Save summary
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Created recommendation summary with {len(summary['symbols'])} symbols")
    
    except Exception as e:
        logger.error(f"Error creating summary: {e}", exc_info=True)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert trade recommendations to unified format")
    parser.add_argument("--input", "-i", help="Input file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all files in input directory")
    parser.add_argument("--organize", "-g", action="store_true", help="Organize recommendations for dashboard")
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input:
            logger.error("Input directory required for batch mode")
            exit(1)
        
        results = batch_convert_recommendations(args.input, args.output)
        logger.info(f"Batch conversion complete: {results['processed_files']} files, "
                   f"{results['total_recommendations']} recommendations, {results['errors']} errors")
        
        # Organize recommendations if requested
        if args.organize:
            organize_results = organize_recommendations_for_dashboard(args.input if not args.output else args.output)
            logger.info(f"Organization complete: {organize_results['organized_files']} recommendations organized")
    else:
        if not args.input:
            logger.error("Input file required")
            exit(1)
        
        recs = convert_recommendations_to_unified_format(args.input, args.output)
        logger.info(f"Conversion complete: {len(recs)} recommendations")
        
        # Organize if requested and output directory is specified
        if args.organize and args.output:
            output_dir = os.path.dirname(args.output)
            organize_recommendations_for_dashboard(output_dir)
            logger.info("Recommendations organized for dashboard")





