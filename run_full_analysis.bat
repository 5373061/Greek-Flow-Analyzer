@echo off
echo Greek Energy Flow II - Full Analysis Sequence
echo =========================================
echo This script will run:
echo 1. Data acquisition and processing for all tickers
echo 2. ML model training using the analysis data
echo 3. ML predictions and trade signal generation
echo 4. Generate trade recommendations
echo 5. Launch the full-featured dashboard
echo.
echo Press Ctrl+C at any time to cancel
echo.
pause

python run_full_analysis.py --refresh-schedule midday --manual-refresh --generate-recommendations

echo.
echo Analysis complete!
pause

