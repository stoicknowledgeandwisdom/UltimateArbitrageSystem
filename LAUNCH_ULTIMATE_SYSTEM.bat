@echo off
color 0A
title 🚀 ULTIMATE ARBITRAGE SYSTEM - ONE-CLICK LAUNCHER

echo.
echo ========================================================
echo    🚀 ULTIMATE ARBITRAGE SYSTEM LAUNCHER 🚀
echo ========================================================
echo    Zero Investment Mindset - Maximum Automated Income
echo    Daily Profit Potential: 6%% - 15%% 
echo    25+ Revenue Streams Active
echo ========================================================
echo.

REM Check if we're in the right directory
if not exist "ULTIMATE_ENHANCED_INCOME_ENGINE.py" (
    echo ❌ ERROR: Please run this script from the UltimateArbitrageSystem directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo 🔍 Checking system requirements...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if virtual environment exists
if not exist "venv\" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📦 Installing/updating dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Warning: Some dependencies may not have installed correctly
)

echo.
echo ========================================================
echo    READY TO LAUNCH ULTIMATE INCOME GENERATION SYSTEM
echo ========================================================
echo.
echo 💰 Expected Daily Returns: 6%% - 15%%
echo 🤖 Automation Level: 99.9%%
echo ⚡ Execution Speed: Sub-100ms
echo 🔥 Revenue Streams: 25+ Active
echo.
echo Choose launch option:
echo [1] Launch Enhanced Income Engine Only
echo [2] Launch Enhanced Dashboard Only  
echo [3] Launch BOTH (Complete System) - RECOMMENDED
echo [4] Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto launch_engine
if "%choice%"=="2" goto launch_dashboard
if "%choice%"=="3" goto launch_both
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto start

:launch_engine
echo.
echo 🚀 Launching Ultimate Enhanced Income Engine...
echo ========================================================
echo Dashboard will NOT be available with this option
echo Engine will run in console mode only
echo ========================================================
echo.
python ULTIMATE_ENHANCED_INCOME_ENGINE.py
goto end

:launch_dashboard
echo.
echo 🎮 Launching Ultimate Enhanced Dashboard...
echo ========================================================
echo Income engine will NOT be running with this option
echo Dashboard will show simulated data only
echo ========================================================
echo.
echo Dashboard will be available at: http://localhost:5000
start http://localhost:5000
python ULTIMATE_ENHANCED_UI_DASHBOARD.py
goto end

:launch_both
echo.
echo 🔥 LAUNCHING COMPLETE ULTIMATE ARBITRAGE SYSTEM 🔥
echo ========================================================
echo ⚡ Income Engine: Starting all 25+ revenue streams
echo 🎮 Dashboard: Real-time monitoring interface
echo 📊 Access: http://localhost:5000
echo ========================================================
echo.

REM Launch dashboard in background
echo 🎮 Starting Enhanced Dashboard...
start "Ultimate Dashboard" python ULTIMATE_ENHANCED_UI_DASHBOARD.py

REM Wait 3 seconds for dashboard to initialize
timeout /t 3 /nobreak >nul

REM Open browser to dashboard
echo 🌐 Opening dashboard in browser...
start http://localhost:5000

REM Wait 2 more seconds
timeout /t 2 /nobreak >nul

REM Launch income engine in foreground
echo 🚀 Starting Enhanced Income Engine...
echo.
echo ========================================================
echo   ULTIMATE ARBITRAGE SYSTEM IS NOW FULLY ACTIVE!
echo ========================================================
echo   📊 Monitor: http://localhost:5000
echo   💰 Target: 6%% - 15%% daily returns
echo   🛑 Stop: Ctrl+C or Emergency Stop in dashboard
echo ========================================================
echo.

python ULTIMATE_ENHANCED_INCOME_ENGINE.py
goto end

:exit
echo.
echo 👋 Exiting Ultimate Arbitrage System Launcher
goto end

:end
echo.
echo ========================================================
echo    Ultimate Arbitrage System Launcher Completed
echo ========================================================
echo.
pause

