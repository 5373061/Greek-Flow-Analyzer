@echo off
echo Greek Energy Flow II - Fix ML Recommendations and Run Dashboard
echo ======================================================
echo.
echo This script will:
echo 1. Convert your existing ML-enhanced trade recommendations to a format compatible with the dashboard
echo 2. Create a market regime summary file for the dashboard
echo.
echo This script is designed specifically for the ML-enhanced recommendations from the recent ML upgrade.
echo.
echo Press any key to continue...
pause > nul

echo.
echo Step 1/2: Fixing ML-enhanced trade recommendation files...
python fix_ml_trade_format.py --results-dir "D:\python projects\Greek Energy Flow II\results"

echo.
echo Step 2/2: Launching dashboard with ML-enhanced data...
echo.
echo Press Ctrl+C at any time to close the dashboard.
echo.

python run_dashboard.py --mode dashboard --base-dir "D:\python projects\Greek Energy Flow II\results"

echo.
echo Dashboard closed.
pause
