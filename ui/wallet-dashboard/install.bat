@echo off
echo Installing Wallet Dashboard dependencies...
echo.

:: Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed. Please install Node.js first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

:: Install dependencies
echo Installing npm dependencies...
npm install

if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

:: Create environment file
if not exist .env.local (
    echo Creating .env.local file...
    echo NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws > .env.local
    echo NEXT_PUBLIC_API_URL=http://localhost:8080/api >> .env.local
    echo Environment file created.
)

echo.
echo Installation complete!
echo.
echo To start the development server, run:
echo   npm run dev
echo.
echo To build for production, run:
echo   npm run build
echo.
echo For Tauri desktop app, run:
echo   npm run tauri:dev
echo.
pause

