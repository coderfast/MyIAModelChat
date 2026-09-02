@echo off
REM Generate HTML training report from CSV metrics
REM Usage: generate_report.bat [csv_path] [--draft]
REM Example: generate_report.bat
REM Example: generate_report.bat checkpoints\chat_model_metrics.csv
REM Example: generate_report.bat checkpoints\draft_model_metrics.csv --draft

setlocal

set CSV_PATH=%~1
set IS_DRAFT=%~2

if "%CSV_PATH%"=="" set CSV_PATH=checkpoints\chat_model_metrics.csv

if not exist "%CSV_PATH%" (
    echo ERROR: CSV file not found: %CSV_PATH%
    echo.
    echo Usage: %~nx0 [csv_path] [--draft]
    echo Example: %~nx0 checkpoints\chat_model_metrics.csv
    pause
    exit /b 1
)

echo Generating report from: %CSV_PATH%
python scripts\generate_report.py "%CSV_PATH%" %IS_DRAFT%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Done! Opening in browser...
    set HTML_PATH=%CSV_PATH:.csv=_report.html%
    start "" "%HTML_PATH%"
) else (
    echo ERROR: Failed to generate report
)

pause
