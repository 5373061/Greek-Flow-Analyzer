@echo off
echo Greek Energy Flow II - Fix and Run Dashboard
echo =========================================
echo.
echo This script will:
echo 1. Fix the dashboard implementation
echo 2. Convert all trade recommendations to a dashboard-compatible format
echo 3. Create necessary market regime files
echo 4. Launch the dashboard
echo.
echo Press any key to continue...
pause > nul

echo.
echo Step 1: Fixing dashboard issues...
python fix_dashboard.py
if %ERRORLEVEL% NEQ 0 (
    echo Error fixing dashboard!
    pause
    exit /b 1
)

echo.
echo Step 2: Launching dashboard...
python -m tools.trade_dashboard

echo.
echo Dashboard closed.
pause

