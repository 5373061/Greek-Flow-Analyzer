@echo off
echo ========================================
echo Greek Energy Flow II - Dashboard Cleaner
echo ========================================
echo.

REM Run the fix script
echo Step 1: Fixing run_dashboard.py...
python fix_run_dashboard.py
echo.

REM Run the cleanup script
echo Step 2: Cleaning up unnecessary dashboard files...
python clean_dashboard_files.py
echo.

REM Run the dashboard
echo Step 3: Launching dashboard...
echo.
python run_dashboard.py
echo.

pause
