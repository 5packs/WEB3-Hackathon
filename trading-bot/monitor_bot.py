#!/usr/bin/env python3
"""
Simple Crypto Roostoo Account Balance Monitor Bot
Runs test_get_balance every 5 minutes continuously
"""
import time
import sys
import signal
import logging
import json
from datetime import datetime
import os

# Configuration
MONITOR_INTERVAL = 300  # 5 minutes in seconds
PURCHASE_DELAY = 5  # 5 seconds between purchases
STARTUP_WAIT_TIME = 300  # Total wait time (5 minutes) including trades before starting monitoring loop
DRY_RUN_PURCHASES = True  # Set to False for real purchases, True for testing

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

# Try to import purchase_by_value function from the trading-bot directory
try:
    from purchase_by_value import purchase_by_value
    purchase_function_available = True
    logger.info("Purchase by value function imported successfully")
except ImportError as e:
    purchase_function_available = False
    logger.warning(f"Purchase function not available: {e}")
    logger.warning("Automated purchasing will be disabled")

class MonitorBot:
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        self.purchases_executed = False  # Track if automated purchases have been done
        
    def execute_portfolio_purchases(self):
        """Execute automated cryptocurrency purchases based on portfolio allocation"""
        if not purchase_function_available:
            logger.warning("Purchase function not available - skipping automated purchases")
            return False, 0
            
        if self.purchases_executed:
            logger.info("Portfolio purchases already executed - skipping")
            return True, 0
            
        # Start timing the purchase process
        purchase_start_time = time.time()
            
        try:
            # Load the portfolio allocation
            allocation_file = os.path.join(parent_dir, "output", "simple_portfolio_allocation.json")
            
            if not os.path.exists(allocation_file):
                logger.error(f"Portfolio allocation file not found: {allocation_file}")
                return False, 0
                
            with open(allocation_file, 'r') as f:
                portfolio = json.load(f)
                
            logger.info("=" * 60)
            logger.info("STARTING AUTOMATED PORTFOLIO PURCHASES")
            logger.info("=" * 60)
            logger.info(f"Found {len(portfolio)} cryptocurrencies to purchase")
            logger.info(f"Mode: {'DRY RUN (no real purchases)' if DRY_RUN_PURCHASES else 'LIVE TRADING (real purchases)'}")
            
            total_investment = sum(portfolio.values())
            logger.info(f"Total investment amount: ${total_investment:,.2f}")
            
            successful_purchases = 0
            failed_purchases = 0
            
            # Execute purchases with delays
            for i, (coin, amount) in enumerate(portfolio.items(), 1):
                try:
                    logger.info(f"\n--- Purchase {i}/{len(portfolio)}: {coin} ---")
                    logger.info(f"Purchasing ${amount:,.2f} worth of {coin}")
                    
                    # Execute the purchase (configurable dry run)
                    result = purchase_by_value(
                        coin=coin,
                        total_value=amount,
                        order_type="MARKET",
                        dry_run=DRY_RUN_PURCHASES  # Use configuration setting
                    )
                    
                    if result and not result.get('error'):
                        successful_purchases += 1
                        logger.info(f"SUCCESS: Successfully initiated purchase of {coin}")
                        if 'quantity' in result:
                            logger.info(f"   Quantity: {result['quantity']} {coin}")
                    else:
                        failed_purchases += 1
                        error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                        logger.error(f"FAILED: Failed to purchase {coin}: {error_msg}")
                        
                except Exception as e:
                    failed_purchases += 1
                    logger.error(f"ERROR: Exception during {coin} purchase: {e}")
                
                # Sleep between purchases (except after the last one)
                if i < len(portfolio):
                    logger.info(f"Waiting {PURCHASE_DELAY} seconds before next purchase...")
                    time.sleep(PURCHASE_DELAY)
            
            # Calculate total execution time
            purchase_end_time = time.time()
            total_execution_time = purchase_end_time - purchase_start_time
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("PORTFOLIO PURCHASE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total cryptocurrencies: {len(portfolio)}")
            logger.info(f"Successful purchases: {successful_purchases}")
            logger.info(f"Failed purchases: {failed_purchases}")
            logger.info(f"Success rate: {(successful_purchases/len(portfolio)*100):.1f}%")
            logger.info(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes)")
            logger.info("=" * 60)
            
            # Mark as executed regardless of success/failure to prevent retries
            self.purchases_executed = True
            return True, total_execution_time
            
        except Exception as e:
            logger.error(f"Error executing portfolio purchases: {e}")
            purchase_end_time = time.time()
            total_execution_time = purchase_end_time - purchase_start_time
            return False, total_execution_time
        
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
        logger.info(f"Purchase mode: {'DRY RUN (no real purchases)' if DRY_RUN_PURCHASES else 'LIVE TRADING (real purchases)'}")
        logger.info(f"Startup process: Optimization -> Trades -> {STARTUP_WAIT_TIME/60:.1f} min total wait -> Monitoring loop")

        # Run SMA optimizer once at startup (non-blocking to monitoring loop flow)
        try:
            optimizer_dir = os.path.join(parent_dir, 'sma-prediction')
            if os.path.isdir(optimizer_dir):
                # Ensure the optimizer directory is importable
                if optimizer_dir not in sys.path:
                    sys.path.insert(0, optimizer_dir)

                try:
                    import multi_cryptocurrency_optimizer as mco
                    logger.info("Starting one-time SMA optimizer run at startup (may take a while)...")
                    prev_cwd = os.getcwd()
                    os.chdir(optimizer_dir)
                    try:
                        # main() runs the optimization and saves outputs
                        mco.main()
                        logger.info("One-time SMA optimizer run completed")
                        
                        # Execute portfolio purchases after successful optimization
                        logger.info("Starting automated portfolio purchases...")
                        purchase_success, trade_execution_time = self.execute_portfolio_purchases()
                        
                        if purchase_success:
                            logger.info("Automated portfolio purchases completed")
                            
                            # Calculate remaining wait time to reach total 5 minutes
                            remaining_wait_time = STARTUP_WAIT_TIME - trade_execution_time
                            
                            if remaining_wait_time > 0:
                                logger.info(f"Trade execution took {trade_execution_time:.2f} seconds")
                                logger.info(f"Waiting additional {remaining_wait_time:.2f} seconds to complete {STARTUP_WAIT_TIME/60:.1f} minute startup delay...")
                                
                                # Sleep in smaller chunks to allow for responsive shutdown
                                sleep_interval = 1  # 1 second chunks
                                for i in range(int(remaining_wait_time)):
                                    if not self.running:
                                        break
                                    time.sleep(sleep_interval)
                                    
                                # Handle fractional seconds
                                fractional_time = remaining_wait_time - int(remaining_wait_time)
                                if fractional_time > 0 and self.running:
                                    time.sleep(fractional_time)
                                    
                                logger.info("Startup delay completed, entering monitoring loop...")
                            else:
                                logger.info(f"Trade execution took {trade_execution_time:.2f} seconds (longer than {STARTUP_WAIT_TIME/60:.1f} minute target)")
                                logger.info("Proceeding directly to monitoring loop...")
                        else:
                            logger.warning("Automated portfolio purchases failed or were skipped")
                            logger.info(f"Waiting {STARTUP_WAIT_TIME/60:.1f} minutes before starting monitoring loop...")
                            
                            # Full wait if purchases failed
                            for i in range(STARTUP_WAIT_TIME):
                                if not self.running:
                                    break
                                time.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"Error while running SMA optimizer at startup: {e}")
                    finally:
                        os.chdir(prev_cwd)
                except Exception as e:
                    logger.warning(f"Could not import/run SMA optimizer: {e}")
            else:
                logger.debug(f"SMA optimizer directory not present: {optimizer_dir}")
        except Exception as e:
            logger.error(f"Unexpected error when attempting startup optimizer run: {e}")
        
        try:
            while self.running:
                try:
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
                            
                except Exception as cycle_error:
                    # Log the error but continue running - don't let individual cycle errors stop the bot
                    logger.error(f"ERROR in cycle #{self.cycle_count}: {cycle_error}")
                    logger.info("Bot will continue running despite the error...")
                    
                    # Still sleep between cycles even after errors
                    if self.running:
                        logger.info(f"Sleeping for {MONITOR_INTERVAL} seconds before retry...")
                        for i in range(MONITOR_INTERVAL):
                            if not self.running:
                                break
                            time.sleep(1)
                        
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt")
            self.running = False
        except SystemExit:
            logger.info("Received SystemExit signal") 
            self.running = False
        except Exception as fatal_error:
            # Only truly fatal errors that prevent the bot from continuing should stop it
            logger.error(f"FATAL ERROR - Bot cannot continue: {fatal_error}")
            logger.error("This should only happen for critical system-level issues")
            self.running = False
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