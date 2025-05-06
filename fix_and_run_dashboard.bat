@echo off
echo Greek Energy Flow II - Fix and Run Dashboard
echo =========================================
echo.
echo This script will:
echo 1. Create sample recommendations if none exist
echo 2. Create necessary market regime files
echo 3. Fix any issues with recommendation files
echo 4. Launch the dashboard
echo.
echo Press any key to continue...
pause > nul

echo.
echo Step 1: Checking for recommendations...
python -c "import os, glob; print('Found recommendations: ' + str(len(glob.glob('results/*_recommendation.json'))))"
python -c "import os, glob; rec_count = len(glob.glob('results/*_recommendation.json')); exit(1 if rec_count == 0 else 0)"
if %ERRORLEVEL% NEQ 0 (
    echo No recommendations found. Creating sample recommendations...
    python create_sample_recommendations.py --num 10
)

echo.
echo Step 2: Creating market regime file...
python create_market_regime.py
if %ERRORLEVEL% NEQ 0 (
    echo Error creating market regime file!
    pause
    exit /b 1
)

echo.
echo Step 3: Fixing recommendation files...
python tools/debug_dashboard.py --fix
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Issues detected with recommendation files.
    echo Continuing anyway...
)

echo.
echo Step 4: Launching dashboard...
python -m tools.trade_dashboard

echo.
echo Dashboard closed.
pause





