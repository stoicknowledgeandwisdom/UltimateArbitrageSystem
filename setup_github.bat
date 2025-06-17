@echo off
echo.
echo ============================================================
echo           ULTIMATE ARBITRAGE SYSTEM - GITHUB SETUP
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "ultimate_arbitrage_launcher.py" (
    echo ERROR: Not in UltimateArbitrageSystem directory
    echo Please run this script from the UltimateArbitrageSystem directory
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Initialize Git repository if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo Git repository initialized
) else (
    echo Git repository already exists
)

echo.
echo Creating/updating project files...

REM Create a simple .gitignore
echo # Ultimate Arbitrage System - Git Ignore > .gitignore
echo # ====================================== >> .gitignore
echo. >> .gitignore
echo # Python >> .gitignore
echo __pycache__/ >> .gitignore
echo *.py[cod] >> .gitignore
echo *.log >> .gitignore
echo *.db >> .gitignore
echo *.sqlite >> .gitignore
echo *.sqlite3 >> .gitignore
echo. >> .gitignore
echo # Virtual Environments >> .gitignore
echo venv/ >> .gitignore
echo .venv/ >> .gitignore
echo env/ >> .gitignore
echo .env >> .gitignore
echo arbitrage_env/ >> .gitignore
echo. >> .gitignore
echo # IDE >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo. >> .gitignore
echo # System Files >> .gitignore
echo .DS_Store >> .gitignore
echo Thumbs.db >> .gitignore
echo. >> .gitignore
echo # Analytics and Logs >> .gitignore
echo analytics/*.json >> .gitignore
echo analytics/*.db >> .gitignore
echo logs/ >> .gitignore
echo temp/ >> .gitignore

echo ✅ .gitignore created/updated

REM Add all files
echo.
echo Staging files for commit...
git add .
echo ✅ Files staged

REM Check git status
echo.
echo Git status:
git status --short

REM Create initial commit
echo.
echo Creating initial commit...
git commit -m "🚀 Initial deployment: Ultimate Arbitrage System - Maximum Profit Extraction with Zero-Investment Mindset"
echo ✅ Initial commit created

echo.
echo ============================================================
echo              GITHUB SETUP INSTRUCTIONS
echo ============================================================
echo.
echo 1. Create a new repository on GitHub:
echo    - Go to https://github.com/new
echo    - Repository name: UltimateArbitrageSystem
echo    - Description: Ultimate Arbitrage System - Maximum Profit Extraction
echo    - Make it Public or Private
echo    - DO NOT initialize with README, .gitignore, or license
echo.
echo 2. Add remote and push (replace 'yourusername'):
echo.
echo    git remote add origin https://github.com/yourusername/UltimateArbitrageSystem.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. Your system is ready for GitHub!
echo.
echo ============================================================
echo                  DEPLOYMENT COMPLETE!
echo ============================================================
echo.
echo ✅ Git repository initialized and committed
echo ✅ Ready to push to GitHub
echo ✅ Ultimate Arbitrage System ready for the world!
echo.
echo 💰 Happy Trading with Zero-Investment Mindset! 💰
echo.
pause

