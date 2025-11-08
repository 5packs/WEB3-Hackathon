# WEB3-Hackathon

## Project Structure

This project contains:

- **`crypto-roostoo-api/`** - Git submodule with Roostoo API functions and utilities
- **`trading-bot/`** - Monitor bot that continuously checks exchange info
- **`.venv/`** - Python virtual environment with all dependencies

## Trading Bot

The trading bot continuously monitors your Roostoo account balance by running `test_get_balance` every 5 minutes.

### Quick Start

1. **Navigate to the trading bot folder:**
   ```cmd
   cd trading-bot
   ```

2. **Run the bot:**
   ```cmd
   # Easy way:
   start_monitor.bat
   
   # Or from project root:
   .venv\Scripts\python.exe trading-bot\monitor_bot.py
   ```

### Setup Complete ✅

- ✅ **Virtual Environment:** Python 3.11.9 with all dependencies
- ✅ **API Submodule:** Latest crypto-roostoo-api integrated  
- ✅ **Monitor Bot:** Organized in separate trading-bot folder
- ✅ **Configuration:** API credentials configured in submodule
- ✅ **Data Analysis:** Pandas and NumPy installed for SMA prediction tools

**Installed Packages:**
- `requests` (2.32.5) - HTTP requests for API calls
- `python-dotenv` (1.2.1) - Environment variable loading
- `pandas` (2.3.3) - Data analysis and manipulation
- `numpy` (2.3.4) - Numerical computing
- Plus supporting dependencies (certifi, urllib3, etc.)

### Features

- **Balance monitoring:** Checks account balance and wallet details every 5 minutes
- **Detailed logging:** Logs to both console and file
- **Balance details:** Shows free and locked amounts for each coin
- **Graceful shutdown:** Stop with Ctrl+C
- **Error handling:** Robust error handling and recovery
- **Clean structure:** Organized in separate folders

### Project Layout

```
WEB3-Hackathon/
├── crypto-roostoo-api/     # Git submodule - API functions
│   ├── utilities.py        # Exchange info functions
│   ├── .env               # API credentials
│   └── ...
├── trading-bot/           # Monitor bot
│   ├── monitor_bot.py     # Main bot script
│   ├── start_monitor.bat  # Easy startup
│   └── README.md         # Bot documentation
├── .venv/                # Virtual environment
└── requirements.txt      # Main project dependencies
```