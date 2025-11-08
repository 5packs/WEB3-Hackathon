# Trading Bot

## Crypto Roostoo Account Balance Monitor Bot

This bot continuously monitors your Roostoo account balance by running `test_get_balance` every 5 minutes.

### Features

- **Continuous monitoring:** Runs balance checks every 5 minutes
- **Detailed logging:** Logs to both console and `logs/monitor_bot.log` file
- **Balance details:** Shows free and locked amounts for each coin in your wallet
- **Graceful shutdown:** Press Ctrl+C to stop safely
- **Error handling:** Continues running even if individual checks fail
- **Organized structure:** Self-contained in the trading-bot folder

### Quick Start

1. **Run the bot:**
   ```cmd
   # Easy way - double-click or run:
   start_monitor.bat
   
   # Or from project root:
   .venv\Scripts\python.exe trading-bot\monitor_bot.py
   ```

### What it does

The bot will:
- Connect to the Roostoo API using credentials from `../crypto-roostoo-api/.env`
- Check your account balance and wallet details
- Log balance information every 5 minutes to the `logs/` folder
- Show free and locked amounts for coins in your wallet
- Continue running until stopped with Ctrl+C

### File Structure

```
trading-bot/
├── monitor_bot.py      # Main bot script
├── start_monitor.bat   # Windows startup script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── logs/              # Log files directory
    └── monitor_bot.log # Bot log file (created when bot runs)
```

### Prerequisites

- Virtual environment is set up in parent directory (`.venv/`)
- API credentials configured in `../crypto-roostoo-api/.env`
- Required packages installed in virtual environment