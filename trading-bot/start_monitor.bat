@echo off
echo.
echo ============================================
echo   Crypto Roostoo Account Balance Monitor
echo ============================================
echo.
echo This bot will check your account balance every 5 minutes
echo.
echo Prerequisites:
echo 1. Make sure the crypto-roostoo-api submodule has a .env file
echo 2. Virtual environment and dependencies are already configured
echo 3. API credentials are properly configured in .env file
echo.
echo Press Ctrl+C to stop the bot at any time
echo.
pause
echo.
echo Starting balance monitor bot using virtual environment...
echo Current directory: %CD%
cd ..
echo Changed to: %CD%
".venv\Scripts\python.exe" "trading-bot\monitor_bot.py"
pause