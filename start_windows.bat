@echo off
setlocal
title QuantumML Arena - Startup Script

echo =======================================================
echo          ⚛️  QuantumML Arena - Windows Start
echo =======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.9+ from python.org and check "Add Python to PATH".
    pause
    exit /b
)

:: Check for Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not added to PATH.
    echo Please install Node.js from nodejs.org.
    pause
    exit /b
)

:: Set script directory
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"

echo [1/2] Setting up Backend Server...
cd /d "%BACKEND_DIR%"
if not exist "venv\" (
    echo Creating Python virtual environment...
    python -m venv venv
    
    echo Activating environment and installing dependencies...
    echo (This might take 5-15 minutes on the first run...)
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Starting Backend Server on port 8000...
start "QuantumML Backend" cmd /c "call venv\Scripts\activate.bat && uvicorn main:app --reload --port 8000"


echo.
echo [2/2] Setting up Frontend Server...
cd /d "%FRONTEND_DIR%"
if not exist "node_modules\" (
    echo Installing Node.js dependencies...
    echo (This might take a minute or two...)
    call npm install
)

echo Starting Frontend Server on port 5173...
start "QuantumML Frontend" cmd /c "npm run dev"

echo.
echo =======================================================
echo     ✅  Both servers are starting up!
echo     🌐  A browser window will open shortly...
echo =======================================================
timeout /t 5 >nul
start http://localhost:5173

echo.
echo Leave the two black terminal windows completely open while using the app!
echo Close this window or press any key to exit this launcher.
pause >nul
exit /b
