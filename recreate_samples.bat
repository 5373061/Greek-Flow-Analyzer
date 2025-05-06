@echo off
echo Greek Energy Flow II - Recreate Sample Recommendations
echo ===================================================
echo.
echo This script will:
echo 1. Clean only sample recommendations (not real trade recommendations)
echo 2. Create new sample recommendations
echo 3. Fix any issues with recommendation files
echo.
echo Press any key to continue...
pause > nul

echo.
echo Step 1: Cleaning sample recommendations...
python clean_sample_recommendations.py --force
if %ERRORLEVEL% NEQ 0 (
    echo Error cleaning sample recommendations!
    pause
    exit /b 1
)

echo.
echo Step 2: Creating new sample recommendations...
python create_sample_recommendations.py --num 10
if %ERRORLEVEL% NEQ 0 (
    echo Error creating sample recommendations!
    pause
    exit /b 1
)

echo.
echo Step 3: Creating market regime file...
python create_market_regime.py
if %ERRORLEVEL% NEQ 0 (
    echo Error creating market regime file!
    pause
    exit /b 1
)

echo.
echo Step 4: Fixing recommendation files...
python tools/debug_dashboard.py --fix
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Issues detected with recommendation files.
    echo Continuing anyway...
)

echo.
echo Sample recommendations recreated successfully.
echo You can now run the dashboard with: python -m tools.trade_dashboard
echo.
pause