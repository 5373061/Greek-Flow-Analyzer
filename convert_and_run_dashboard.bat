@echo off
echo Greek Energy Flow II - Convert and Run Dashboard
echo =============================================
echo This script will:
echo 1. Convert existing trade recommendations to dashboard format
echo 2. Launch the dashboard with the converted files
echo.
echo Press any key to continue...
pause > nul

echo Converting recommendation files...
python convert_recommendations.py --results-dir "D:\python projects\Greek Energy Flow II\results"

echo.
echo Conversion complete! Running the dashboard...
echo.
echo Press Ctrl+C at any time to close the dashboard.
echo.

run_fixed_dashboard.bat

echo.
echo Conversion and dashboard process complete.
pause
