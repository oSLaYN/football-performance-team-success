@echo off
echo ========================================================
echo Football Performance and Team Success - Dashboard Setup
echo ========================================================
echo.

echo [1/2] Installing Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to Install Dependencies. Verify if Python and/or pip are Correctly Installed.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting Streamlit Dashboard...
echo.
streamlit run dashboard.py
pause
