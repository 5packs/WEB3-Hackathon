#!/usr/bin/env python3
"""
Simple Crypto Roostoo Account Balance Monitor Bot
Runs test_get_balance every 5 minutes continuously
"""
import time
import sys
import signal
import logging
from datetime import datetime
import os

# Configuration
MONITOR_INTERVAL = 300  # 5 minutes in seconds

# Create logs directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE = os.path.join(logs_dir, "monitor_bot.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Setup submodule import - go up one level from trading-bot to find crypto-roostoo-api
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
submodule_path = os.path.join(parent_dir, 'crypto-roostoo-api')
sys.path.insert(0, submodule_path)

# Change to submodule directory for proper .env file loading
original_cwd = os.getcwd()
os.chdir(submodule_path)

try:
    # Import the functions we want to monitor from the submodule
    from balance import test_get_balance, get_balance
    import_success = True
    import_error = None
except ImportError as e:
    import_success = False
    import_error = str(e)
finally:
    # Change back to original directory
    os.chdir(original_cwd)

class MonitorBot:
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown on CTRL+C"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        
    def run_exchange_info_check(self):
        """Run the balance check with error handling"""
        try:
            logger.info("Starting account balance check...")
            
            # Change to submodule directory for API calls
            original_cwd = os.getcwd()
            os.chdir(submodule_path)
            
            try:
                # Also get raw data for logging
                balance = get_balance()
                if balance:
                    spot_wallet = balance.get('SpotWallet', {})
                    coins_count = len(spot_wallet)
                    logger.info(f"Account balance retrieved successfully. Coins in wallet: {coins_count}")
                    
                    # Log some balance details
                    if spot_wallet:
                        for coin, details in list(spot_wallet.items())[:5]:  # Log first 5 coins
                            free = details.get('Free', '0')
                            locked = details.get('Lock', '0')
                            if float(free) > 0 or float(locked) > 0:
                                logger.info(f"  {coin}: Free={free}, Locked={locked}")
                else:
                    logger.warning("Failed to get balance data")
            finally:
                os.chdir(original_cwd)
                
            logger.info("Account balance check completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during balance check: {e}")
            return False
    
    def run(self):
        """Main monitoring loop"""
        if not import_success:
            logger.error("="*60)
            logger.error("IMPORT ERROR: Could not import balance functions from submodule")
            logger.error(f"Error: {import_error}")
            logger.error("Please ensure:")
            logger.error("1. The crypto-roostoo-api submodule is properly initialized")
            logger.error("2. The .env file exists in crypto-roostoo-api/ directory")
            logger.error("3. All required packages are installed")
            logger.error("4. API credentials are properly configured")
            logger.error("="*60)
            return
            
        logger.info("="*60)
        logger.info("Crypto Roostoo Account Balance Monitor Bot Starting")
        logger.info("="*60)
        logger.info(f"Monitor interval: {MONITOR_INTERVAL} seconds ({MONITOR_INTERVAL//60} minutes)")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info(f"Logs directory: {logs_dir}")
        logger.info(f"Submodule path: {submodule_path}")
        
        # Check if submodule directory exists
        if not os.path.exists(submodule_path):
            logger.error(f"Submodule directory not found: {submodule_path}")
            logger.error("Make sure you have initialized the git submodule properly")
            return
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Bot started successfully. Press CTRL+C to stop.")
        
        try:
            while self.running:
                self.cycle_count += 1
                start_time = datetime.now()
                
                logger.info(f"\n--- Cycle #{self.cycle_count} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
                
                # Run the balance check
                success = self.run_exchange_info_check()
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if success:
                    logger.info(f"SUCCESS: Cycle #{self.cycle_count} completed successfully in {duration:.2f} seconds")
                else:
                    logger.error(f"FAILED: Cycle #{self.cycle_count} failed after {duration:.2f} seconds")
                
                if self.running:
                    logger.info(f"Sleeping for {MONITOR_INTERVAL} seconds until next check...")
                    
                    # Sleep in smaller chunks to allow for responsive shutdown
                    for i in range(MONITOR_INTERVAL):
                        if not self.running:
                            break
                        time.sleep(1)
                        
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            logger.info("="*60)
            logger.info(f"Monitor Bot shutting down after {self.cycle_count} cycles")
            logger.info("="*60)

def main():
    """Entry point"""
    bot = MonitorBot()
    bot.run()

if __name__ == "__main__":
    main()