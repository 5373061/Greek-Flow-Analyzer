@echo off
echo Running ML-Enhanced Greek Regime Analysis
echo =========================================

REM Parse command line arguments
set TICKERS=AAPL MSFT QQQ SPY LULU TSLA CMG WYNN ZM SPOT
set MODE=analyze

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--tickers" (
    set TICKERS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--train" (
    set MODE=train
    shift
    goto :parse_args
)
if /i "%~1"=="--predict" (
    set MODE=predict
    shift
    goto :parse_args
)
if /i "%~1"=="--simulate" (
    set MODE=simulate
    shift
    goto :parse_args
)
if /i "%~1"=="--live" (
    set MODE=live
    shift
    goto :parse_args
)
shift
goto :parse_args

:done_parsing

echo.
echo Mode: %MODE%
echo Tickers: %TICKERS%
echo.

REM Run the appropriate command based on mode
if "%MODE%"=="train" (
    echo Training ML models on existing Greek analysis data...
    python run_ml_enhanced_trading.py --tickers %TICKERS% --train
) else if "%MODE%"=="predict" (
    echo Running ML predictions with existing models...
    python run_ml_enhanced_trading.py --tickers %TICKERS% --predict
) else if "%MODE%"=="simulate" (
    echo Running trading simulation with ML-enhanced signals...
    python run_ml_enhanced_trading.py --tickers %TICKERS% --simulate
) else if "%MODE%"=="live" (
    echo Running live trading with ML-enhanced signals...
    python run_ml_enhanced_trading.py --tickers %TICKERS% --live
) else (
    echo Running standard Greek analysis with ML...
    python run_ml_enhanced_trading.py --tickers %TICKERS% --analyze
)

echo.
echo ML-Enhanced Greek Regime Analysis Complete
echo =========================================
echo.

pause
