# WEB3-Hackathon

## Project Structure

This project contains:

- **`crypto-roostoo-api/`** - Git submodule with Roostoo API functions and utilities
- **`trading-bot/`** - Monitor bot that continuously checks account balance
- **`sma-prediction/`** - SMA trading strategy optimization and analysis tools
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

## SMA Trading Strategy

The project includes advanced Simple Moving Average (SMA) trading strategy optimization:

### SMA Optimizer Features

- **Multi-currency optimization:** Finds optimal SMA window pairs for multiple cryptocurrencies
- **Parameter persistence:** Saves optimal parameters to JSON files for bot usage
- **Risk assessment:** Analyzes expected returns, Sharpe ratios, and maximum drawdown
- **Risk categorization:** Automatically categorizes currencies by risk level

### Using the SMA System

1. **Generate optimal parameters:**
   ```cmd
   .venv\Scripts\python.exe sma-prediction\multi_cryptocurrency_optimizer.py
   ```

2. **Generated files:**
   - `sma-prediction/optimal_sma_parameters.json` - Detailed parameters with risk metrics
   - `sma-prediction/simple_sma_parameters.json` - Simple parameters for bot usage

3. **Use in trading decisions:**
   ```python
   from trading_strategy import make_optimized_trading_decision
   
   decision = make_optimized_trading_decision(
       'BTC', historical_prices, current_price
   )
   ```

4. **Example usage:**
   ```cmd
   .venv\Scripts\python.exe trading_bot_example.py
   ```

### Project Layout

```
WEB3-Hackathon/
├── crypto-roostoo-api/     # Git submodule - API functions
│   ├── utilities.py        # Exchange info functions
│   ├── balance.py         # Account balance functions  
│   ├── .env               # API credentials
│   └── ...
├── trading-bot/           # Monitor bot
│   ├── monitor_bot.py     # Main bot script
│   ├── start_monitor.bat  # Easy startup
│   ├── logs/             # Bot log files
│   └── README.md         # Bot documentation
├── sma-prediction/        # Trading strategy optimization
│   ├── multi_cryptocurrency_optimizer.py  # Parameter optimization
│   ├── trading_strategy.py               # SMA trading logic
│   ├── optimal_sma_parameters.json       # Detailed parameters
│   └── simple_sma_parameters.json        # Simple parameters
├── trading_bot_example.py # Example of using optimized parameters
├── .venv/                # Virtual environment
└── requirements.txt      # Main project dependencies
```