#!/usr/bin/env python3
"""
Simple Crypto Roostoo Account Balance Monitor Bot
Runs comprehensive trading cycle every 5 minutes continuously
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
DRY_RUN_PURCHASES = False  # Set to False for real purchases, True for testing

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
    from trades import place_order
    
    # Import trading strategy functions from sma-prediction
    sma_prediction_path = os.path.join(parent_dir, 'sma-prediction')
    sys.path.insert(0, sma_prediction_path)
    from trading_strategy import make_optimized_trading_decision
    from prices import fetch_kraken_ohlc_recent
    
    # Import purchase_by_value from our local module
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from purchase_by_value import purchase_by_value, get_current_price, update_portfolio_allocation_after_purchase
    
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
        
    def recover_missing_currency_allocation(self, coin, portfolio_file, portfolio_data):
        """
        Recovery function for currencies that were marked as purchased but are missing from wallet
        """
        try:
            # Load original allocation file to get the intended allocation
            original_allocation_file = os.path.join(parent_dir, "output", "portfolio_allocation.json")
            
            if os.path.exists(original_allocation_file):
                with open(original_allocation_file, 'r') as f:
                    original_allocation = json.load(f)
                
                if coin in original_allocation:
                    restore_amount = original_allocation[coin]
                    logger.warning(f"RECOVERY: Restoring {coin} allocation from ${portfolio_data[coin]} to ${restore_amount}")
                    
                    # Update the portfolio allocation
                    portfolio_data[coin] = restore_amount
                    
                    # Save the updated portfolio
                    with open(portfolio_file, 'w') as f:
                        json.dump(portfolio_data, f, indent=2)
                    
                    logger.info(f"RECOVERY: {coin} allocation restored to ${restore_amount} for re-purchase attempt")
                    return True
                else:
                    logger.warning(f"RECOVERY: {coin} not found in original allocation file")
            else:
                logger.warning(f"RECOVERY: Original allocation file not found: {original_allocation_file}")
                
        except Exception as e:
            logger.error(f"RECOVERY: Error restoring {coin} allocation: {e}")
            
        return False
        
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
        
        # Sell all currencies before shutdown (only if not already done)
        if not hasattr(self, '_shutdown_sell_completed'):
            self.sell_all_currencies_on_shutdown()
            self._shutdown_sell_completed = True
        
    def sell_all_currencies_on_shutdown(self):
        """Sell all owned currencies when shutting down"""
        try:
            logger.info("=" * 60)
            logger.info("EMERGENCY SELL ALL CURRENCIES ON SHUTDOWN")
            logger.info("=" * 60)
            
            # Change to submodule directory for API calls
            original_cwd = os.getcwd()
            os.chdir(submodule_path)
            
            try:
                # Get current account balance
                logger.info("Fetching current account balance for shutdown sell...")
                balance = get_balance()
                if not balance:
                    logger.error("Failed to get account balance for shutdown sell")
                    return
                
                spot_wallet = balance.get('SpotWallet', {})
                if not spot_wallet:
                    logger.warning("No spot wallet data found for shutdown sell")
                    return
                
                # Filter out USD and get owned currencies with positive balance
                owned_currencies = {}
                for coin, details in spot_wallet.items():
                    if coin.upper() != 'USD':
                        free_amount = float(details.get('Free', 0))
                        if free_amount > 0:
                            owned_currencies[coin] = free_amount
                
                if not owned_currencies:
                    logger.info("No currencies to sell on shutdown")
                    return
                
                logger.info(f"Found {len(owned_currencies)} currencies to sell on shutdown:")
                for coin, amount in owned_currencies.items():
                    logger.info(f"  {coin}: {amount}")
                
                # Sell all owned currencies
                sell_count = 0
                for coin, quantity in owned_currencies.items():
                    try:
                        logger.info(f"SHUTDOWN SELL: Selling {quantity} {coin}")
                        
                        if not DRY_RUN_PURCHASES:
                            # Place immediate market sell order
                            result = place_order(
                                pair_or_coin=coin,
                                side="SELL",
                                quantity=quantity,
                                order_type="MARKET"
                            )
                            
                            logger.info(f"SHUTDOWN SELL: Order placed for {coin}")
                            sell_count += 1
                            
                            # Extract USD value received from the sale (for logging purposes)
                            usd_received = 0
                            if result and isinstance(result, dict):
                                # Parse the result to extract UnitChange (USD received)
                                order_detail = result.get('OrderDetail', {})
                                if order_detail:
                                    usd_received = float(order_detail.get('UnitChange', 0))
                                    logger.info(f"SHUTDOWN SELL: Received ${usd_received:.2f} USD from {coin} sale")
                                else:
                                    logger.warning(f"SHUTDOWN SELL: No OrderDetail found in result: {result}")
                            else:
                                logger.warning(f"SHUTDOWN SELL: Unexpected result format: {result}")
                            
                            # Defensive check: if USD extraction failed, estimate using available price
                            if usd_received <= 0:
                                logger.warning(f"SHUTDOWN SELL: USD received extraction failed for {coin}, attempting to estimate")
                                try:
                                    # Try to get current price and estimate USD value
                                    estimated_price = get_current_price(coin)
                                    if estimated_price and estimated_price > 0:
                                        estimated_usd = estimated_price * quantity
                                        logger.info(f"SHUTDOWN SELL: Estimated USD value for {coin} sale: ${estimated_usd:.2f} ({quantity} * ${estimated_price:.4f})")
                                        usd_received = estimated_usd
                                    else:
                                        logger.error(f"SHUTDOWN SELL: Could not get price for {coin}, preserving original portfolio value")
                                        # Skip updating this coin's portfolio if we can't determine value
                                        continue
                                except Exception as estimate_error:
                                    logger.error(f"SHUTDOWN SELL: Failed to estimate USD value for {coin}: {estimate_error}")
                                    logger.error(f"SHUTDOWN SELL: Will preserve original portfolio value for {coin} instead of setting to 0")
                                    continue
                            
                            # Update portfolio allocation with USD received from sale
                            try:
                                portfolio_file = os.path.join(
                                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "output", 
                                    "simple_portfolio_allocation.json"
                                )
                                
                                logger.info(f"SHUTDOWN SELL: Attempting to update portfolio file: {portfolio_file}")
                                
                                if os.path.exists(portfolio_file):
                                    # Read current portfolio data
                                    with open(portfolio_file, 'r') as f:
                                        portfolio_data = json.load(f)
                                    
                                    logger.info(f"SHUTDOWN SELL: Current portfolio data keys: {list(portfolio_data.keys())}")
                                    logger.info(f"SHUTDOWN SELL: Looking for key: {coin.upper()}")
                                    logger.info(f"SHUTDOWN SELL: Current value for {coin.upper()}: {portfolio_data.get(coin.upper(), 'NOT_FOUND')}")
                                    
                                    if coin.upper() in portfolio_data:
                                        old_value = portfolio_data[coin.upper()]
                                        # Set to USD received from sale (track current position value)
                                        portfolio_data[coin.upper()] = round(usd_received, 2)
                                        
                                        # Write updated data back to file
                                        with open(portfolio_file, 'w') as f:
                                            json.dump(portfolio_data, f, indent=2)
                                        
                                        # Verify the write was successful
                                        with open(portfolio_file, 'r') as f:
                                            verify_data = json.load(f)
                                            new_value = verify_data.get(coin.upper(), 'NOT_FOUND')
                                        
                                        logger.info(f"SHUTDOWN SELL: Updated portfolio allocation for {coin} from ${old_value} to ${usd_received:.2f} (USD received from sale)")
                                        logger.info(f"SHUTDOWN SELL: Verification - {coin.upper()} value after update: {new_value}")
                                    else:
                                        # Add the currency with USD received value (was sold)
                                        portfolio_data[coin.upper()] = round(usd_received, 2)
                                        
                                        with open(portfolio_file, 'w') as f:
                                            json.dump(portfolio_data, f, indent=2)
                                            
                                        logger.warning(f"SHUTDOWN SELL: {coin.upper()} was not in portfolio data, added with ${usd_received:.2f} USD received from sale")
                                        
                                else:
                                    logger.error(f"SHUTDOWN SELL: Portfolio file does not exist: {portfolio_file}")
                                        
                            except Exception as portfolio_error:
                                logger.error(f"Error updating portfolio for {coin}: {portfolio_error}")
                        else:
                            logger.info(f"SHUTDOWN SELL (DRY RUN): Would sell {quantity} {coin}")
                            sell_count += 1
                        
                        # Small delay between orders
                        time.sleep(1)
                        
                    except Exception as sell_error:
                        logger.error(f"Error selling {coin} on shutdown: {sell_error}")
                        continue
                
                logger.info(f"SHUTDOWN SELL COMPLETE: {sell_count} currencies processed")
                
            except Exception as balance_error:
                logger.error(f"Error during shutdown sell process: {balance_error}")
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Critical error during shutdown sell: {e}")
        finally:
            logger.info("=" * 60)
            logger.info("SHUTDOWN SELL PROCESS COMPLETED")
            logger.info("=" * 60)
        
    def run_exchange_info_check(self):
        """Run comprehensive trading cycle with balance check and optimized trading decisions"""
        cycle_start_time = datetime.now()
        
        try:
            logger.info("Starting trading cycle...")
            
            # Create recent_crypto_data folder in project root if it doesn't exist
            recent_data_dir = os.path.join(parent_dir, 'recent_crypto_data')
            os.makedirs(recent_data_dir, exist_ok=True)
            
            # Change to submodule directory for API calls
            original_cwd = os.getcwd()
            os.chdir(submodule_path)
            
            try:
                # Step 1: Get current account balance
                logger.info("Fetching current account balance...")
                balance = get_balance()
                if not balance:
                    logger.error("Failed to get account balance")
                    return False
                
                spot_wallet = balance.get('SpotWallet', {})
                if not spot_wallet:
                    logger.warning("No spot wallet data found")
                    return False
                
                # Filter out USD and get owned currencies with positive balance
                owned_currencies = {}
                for coin, details in spot_wallet.items():
                    if coin.upper() != 'USD':
                        free_amount = float(details.get('Free', 0))
                        if free_amount > 0:
                            owned_currencies[coin] = {
                                'free': free_amount,
                                'locked': float(details.get('Lock', 0))
                            }
                
                logger.info(f"Found {len(owned_currencies)} owned currencies with positive balance")
                for coin, data in owned_currencies.items():
                    logger.info(f"  {coin}: {data['free']} free, {data['locked']} locked")
                
                # Step 2: Load portfolio allocation
                portfolio_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "output", 
                    "simple_portfolio_allocation.json"
                )
                
                if not os.path.exists(portfolio_file):
                    logger.error(f"Portfolio allocation file not found: {portfolio_file}")
                    return False
                
                with open(portfolio_file, 'r') as f:
                    portfolio_allocation = json.load(f)
                
                logger.info(f"Loaded portfolio allocation for {len(portfolio_allocation)} currencies")
                
                # Track trading actions and timing
                trading_start_time = datetime.now()
                sell_actions = 0
                buy_actions = 0
                
                # Step 3: Check owned currencies for SELL decisions
                logger.info("Checking owned currencies for sell decisions...")
                for coin in owned_currencies.keys():
                    if not self.running:  # Check if bot should stop
                        break
                        
                    try:
                        logger.info(f"Making trading decision for owned currency: {coin}")
                        
                        # Get current price for the coin
                        current_price = get_current_price(coin)
                        if not current_price:
                            logger.warning(f"Could not fetch current price for {coin}, skipping...")
                            continue
                        
                        # Get historical prices for the coin (100 recent 5-minute candles)
                        try:
                            # Convert coin symbol to Kraken format if needed
                            kraken_pair = f"{coin}USD" if coin != "BTC" else "XBTUSD"
                            historical_data = fetch_kraken_ohlc_recent(pair=kraken_pair, interval=5, count=100)
                            
                            if not historical_data:
                                logger.warning(f"Could not fetch historical prices for {coin}, skipping...")
                                continue
                            
                            # Save historical data to recent_crypto_data folder (overwrite each run)
                            data_filename = os.path.join(recent_data_dir, f"{coin}_recent_5min_data.json")
                            try:
                                with open(data_filename, 'w') as f:
                                    json.dump({
                                        'coin': coin,
                                        'kraken_pair': kraken_pair,
                                        'interval': 5,
                                        'count': len(historical_data),
                                        'timestamp': datetime.now().isoformat(),
                                        'data': historical_data
                                    }, f, indent=2)
                                logger.info(f"Saved {len(historical_data)} historical data points for {coin} to {data_filename}")
                            except Exception as save_error:
                                logger.warning(f"Could not save historical data for {coin}: {save_error}")
                                
                            # Extract close prices from OHLC data
                            past_prices = [float(candle[4]) for candle in historical_data]  # Index 4 is close price
                            logger.info(f"Fetched {len(past_prices)} historical prices for {coin}")
                            
                        except Exception as price_error:
                            logger.error(f"Error fetching historical prices for {coin}: {price_error}")
                            continue
                        
                        # Make optimized trading decision with current and historical prices
                        # Use absolute path to ensure parameters file is found regardless of working directory
                        parameters_file = os.path.join(parent_dir, "output", "simple_sma_parameters.json")
                        decision = make_optimized_trading_decision(
                            currency=coin,
                            past_prices=past_prices,
                            current_price=current_price,
                            parameters_file=parameters_file
                        )
                        logger.info(f"Trading decision for {coin} at ${current_price:.4f}: {decision}")
                        
                        if decision == "SELL":
                            # Sell the entire quantity
                            quantity = owned_currencies[coin]['free']
                            logger.info(f"Executing SELL order for {quantity} {coin}")
                            
                            if not DRY_RUN_PURCHASES:
                                result = place_order(
                                    pair_or_coin=coin,
                                    side="SELL", 
                                    quantity=quantity,
                                    order_type="MARKET"
                                )
                                logger.info(f"SELL order result for {coin}: {result}")
                                
                                # Extract USD value received from the sale and update portfolio
                                usd_received = 0
                                if result and isinstance(result, dict):
                                    # Parse the result to extract UnitChange (USD received)
                                    order_detail = result.get('OrderDetail', {})
                                    if order_detail:
                                        usd_received = float(order_detail.get('UnitChange', 0))
                                        logger.info(f"MAIN LOOP SELL: Received ${usd_received:.2f} USD from {coin} sale")
                                    else:
                                        logger.warning(f"MAIN LOOP SELL: No OrderDetail found in result: {result}")
                                else:
                                    logger.warning(f"MAIN LOOP SELL: Unexpected result format: {result}")
                                
                                # Defensive check: if USD extraction failed, estimate using current price
                                if usd_received <= 0:
                                    logger.warning(f"MAIN LOOP SELL: USD received extraction failed for {coin}, attempting to estimate using current price")
                                    try:
                                        # Estimate USD value using current price and quantity sold
                                        estimated_usd = current_price * quantity
                                        logger.info(f"MAIN LOOP SELL: Estimated USD value for {coin} sale: ${estimated_usd:.2f} ({quantity} * ${current_price:.4f})")
                                        usd_received = estimated_usd
                                    except Exception as estimate_error:
                                        logger.error(f"MAIN LOOP SELL: Failed to estimate USD value for {coin}: {estimate_error}")
                                        logger.error(f"MAIN LOOP SELL: Will preserve original portfolio value for {coin} instead of setting to 0")
                                        # Don't update portfolio if we can't determine the value
                                        continue
                                
                                # Update portfolio allocation with USD received from sale
                                try:
                                    portfolio_file = os.path.join(
                                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "output", 
                                        "simple_portfolio_allocation.json"
                                    )
                                    
                                    if os.path.exists(portfolio_file):
                                        # Read current portfolio data
                                        with open(portfolio_file, 'r') as f:
                                            portfolio_data = json.load(f)
                                        
                                        if coin.upper() in portfolio_data:
                                            old_value = portfolio_data[coin.upper()]
                                            # Set to USD received from sale (track current position value)
                                            portfolio_data[coin.upper()] = round(usd_received, 2)
                                            
                                            # Write updated data back to file
                                            with open(portfolio_file, 'w') as f:
                                                json.dump(portfolio_data, f, indent=2)
                                            
                                            logger.info(f"MAIN LOOP SELL: Updated portfolio allocation for {coin} from ${old_value} to ${usd_received:.2f} (USD received from sale)")
                                        else:
                                            # Add the currency with USD received value
                                            portfolio_data[coin.upper()] = round(usd_received, 2)
                                            
                                            with open(portfolio_file, 'w') as f:
                                                json.dump(portfolio_data, f, indent=2)
                                            
                                            logger.info(f"MAIN LOOP SELL: Added {coin.upper()} to portfolio with ${usd_received:.2f} USD received from sale")
                                    else:
                                        logger.error(f"MAIN LOOP SELL: Portfolio file does not exist: {portfolio_file}")
                                        
                                except Exception as portfolio_error:
                                    logger.error(f"MAIN LOOP SELL: Error updating portfolio for {coin}: {portfolio_error}")
                            else:
                                logger.info(f"DRY RUN: Would sell {quantity} {coin}")
                                # In dry run, simulate sale by estimating USD value received
                                try:
                                    estimated_usd = current_price * quantity
                                    logger.info(f"DRY RUN SELL: Estimated USD value for {coin} sale: ${estimated_usd:.2f}")
                                    
                                    # Update portfolio with estimated USD (don't use purchase function for sales)
                                    portfolio_file = os.path.join(
                                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "output", 
                                        "simple_portfolio_allocation.json"
                                    )
                                    
                                    if os.path.exists(portfolio_file):
                                        with open(portfolio_file, 'r') as f:
                                            portfolio_data = json.load(f)
                                        
                                        if coin.upper() in portfolio_data:
                                            old_value = portfolio_data[coin.upper()]
                                            portfolio_data[coin.upper()] = round(estimated_usd, 2)
                                            
                                            with open(portfolio_file, 'w') as f:
                                                json.dump(portfolio_data, f, indent=2)
                                            
                                            logger.info(f"DRY RUN SELL: Updated portfolio allocation for {coin} from ${old_value} to ${estimated_usd:.2f} (estimated from dry run sale)")
                                        
                                except Exception as dry_run_error:
                                    logger.error(f"DRY RUN SELL: Error updating portfolio for {coin}: {dry_run_error}")
                                    # Fallback to preserving original value
                                    logger.info(f"DRY RUN SELL: Preserving original portfolio value for {coin}")
                            
                            sell_actions += 1
                            
                        elif decision in ["HOLD", "BUY"]:
                            logger.info(f"Holding {coin} (decision: {decision})")
                        
                        # Wait 5 seconds between actions
                        if sell_actions > 0 and self.running:
                            logger.info("Waiting 5 seconds before next action...")
                            time.sleep(5)
                            
                    except Exception as e:
                        logger.error(f"Error processing sell decision for {coin}: {e}")
                        continue
                
                # Step 4: Check unowned currencies from portfolio for BUY decisions
                logger.info("Checking unowned currencies for buy decisions...")
                
                # Categorize currencies for better debugging
                owned_currency_names = set(owned_currencies.keys())
                portfolio_currency_names = set(portfolio_allocation.keys())
                
                # Find currencies in portfolio but not in wallet (missing from balance)
                potentially_failed_purchases = []
                unowned_with_allocation = []
                
                for coin in portfolio_currency_names:
                    if coin not in owned_currency_names:
                        if portfolio_allocation[coin] > 0:
                            # Has available cash for purchase
                            unowned_with_allocation.append(coin)
                        else:
                            # Allocation = 0 means money was "spent" but currency missing from wallet
                            # This indicates a failed purchase that was incorrectly marked as successful
                            potentially_failed_purchases.append(coin)
                
                # Log detailed currency categorization
                logger.info(f"Currency Analysis:")
                logger.info(f"  • Owned currencies in wallet: {len(owned_currency_names)} - {sorted(owned_currency_names)}")
                logger.info(f"  • Unowned currencies with allocation > 0: {len(unowned_with_allocation)} - {sorted(unowned_with_allocation)}")
                logger.info(f"  • Potentially failed purchases (allocation = 0): {len(potentially_failed_purchases)} - {sorted(potentially_failed_purchases)}")
                
                if potentially_failed_purchases:
                    logger.error(f"ERROR: {len(potentially_failed_purchases)} currencies show as 'spent' (allocation = 0) but are missing from wallet:")
                    for coin in potentially_failed_purchases:
                        logger.error(f"  • {coin}: Money was marked as spent (${portfolio_allocation[coin]}) but currency not in wallet")
                        logger.error(f"    This indicates the purchase failed silently but was marked as successful")
                        logger.error(f"    RECOMMENDED: Restore allocation to allow re-purchase attempt")
                        
                        # Auto-recovery: Restore allocation for failed purchases
                        logger.warning(f"AUTO-RECOVERY: Attempting to restore allocation for {coin}")
                        if self.recover_missing_currency_allocation(coin, portfolio_file, portfolio_allocation):
                            logger.info(f"AUTO-RECOVERY: Successfully restored allocation for {coin}")
                            # Move from failed purchases to unowned with allocation for this cycle
                            unowned_with_allocation.append(coin)
                        else:
                            logger.error(f"AUTO-RECOVERY: Failed to restore allocation for {coin}")
                
                # Process unowned currencies with positive allocation
                unowned_currencies = unowned_with_allocation
                logger.info(f"Found {len(unowned_currencies)} unowned currencies with allocation for buy decisions")
                
                for coin in unowned_currencies:
                    if not self.running:  # Check if bot should stop
                        break
                        
                    try:
                        logger.info(f"Making trading decision for unowned currency: {coin}")
                        
                        # Get current price for the coin
                        current_price = get_current_price(coin)
                        if not current_price:
                            logger.warning(f"Could not fetch current price for {coin}, skipping...")
                            continue
                        
                        # Get historical prices for the coin (100 recent 5-minute candles)
                        try:
                            # Convert coin symbol to Kraken format if needed
                            kraken_pair = f"{coin}USD" if coin != "BTC" else "XBTUSD"
                            historical_data = fetch_kraken_ohlc_recent(pair=kraken_pair, interval=5, count=100)
                            
                            if not historical_data:
                                logger.warning(f"Could not fetch historical prices for {coin}, skipping...")
                                continue
                            
                            # Save historical data to recent_crypto_data folder (overwrite each run)
                            data_filename = os.path.join(recent_data_dir, f"{coin}_recent_5min_data.json")
                            try:
                                with open(data_filename, 'w') as f:
                                    json.dump({
                                        'coin': coin,
                                        'kraken_pair': kraken_pair,
                                        'interval': 5,
                                        'count': len(historical_data),
                                        'timestamp': datetime.now().isoformat(),
                                        'data': historical_data
                                    }, f, indent=2)
                                logger.info(f"Saved {len(historical_data)} historical data points for {coin} to {data_filename}")
                            except Exception as save_error:
                                logger.warning(f"Could not save historical data for {coin}: {save_error}")
                                
                            # Extract close prices from OHLC data
                            past_prices = [float(candle[4]) for candle in historical_data]  # Index 4 is close price
                            logger.info(f"Fetched {len(past_prices)} historical prices for {coin}")
                            
                        except Exception as price_error:
                            logger.error(f"Error fetching historical prices for {coin}: {price_error}")
                            continue
                        
                        # Make optimized trading decision with current and historical prices
                        # Use absolute path to ensure parameters file is found regardless of working directory
                        parameters_file = os.path.join(parent_dir, "output", "simple_sma_parameters.json")
                        decision = make_optimized_trading_decision(
                            currency=coin,
                            past_prices=past_prices,
                            current_price=current_price,
                            parameters_file=parameters_file
                        )
                        logger.info(f"Trading decision for {coin} at ${current_price:.4f}: {decision}")
                        
                        if decision == "BUY":
                            # Buy using value from portfolio allocation
                            allocation_value = portfolio_allocation[coin]
                            logger.info(f"Executing BUY order for ${allocation_value} worth of {coin}")
                            
                            # Use purchase_by_value function
                            result = purchase_by_value(
                                coin=coin,
                                total_value=allocation_value,
                                dry_run=DRY_RUN_PURCHASES
                            )
                            
                            if result:
                                logger.info(f"BUY order successful for {coin}")
                            else:
                                logger.error(f"BUY order failed for {coin}")
                            
                            buy_actions += 1
                            
                        elif decision in ["HOLD", "SELL"]:
                            logger.info(f"Not buying {coin} (decision: {decision})")
                        
                        # Wait 5 seconds between actions
                        if buy_actions > 0 and self.running:
                            logger.info("Waiting 5 seconds before next action...")
                            time.sleep(5)
                            
                    except Exception as e:
                        logger.error(f"Error processing buy decision for {coin}: {e}")
                        continue
                
                # Calculate timing and remaining sleep
                trading_end_time = datetime.now()
                trading_duration = (trading_end_time - trading_start_time).total_seconds()
                
                logger.info(f"Trading cycle completed: {sell_actions} sells, {buy_actions} buys")
                logger.info(f"Trading actions took {trading_duration:.2f} seconds")
                
                return True
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
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
                    
                    # Calculate remaining time to reach 5-minute cycle
                    remaining_time = MONITOR_INTERVAL - duration
                    
                    if remaining_time > 0:
                        logger.info(f"Cycle completed in {duration:.2f} seconds, waiting {remaining_time:.2f} seconds to complete {MONITOR_INTERVAL/60:.1f} minute cycle...")
                        
                        # Sleep in 1-second chunks to allow for responsive shutdown
                        for i in range(int(remaining_time)):
                            if not self.running:
                                break
                            time.sleep(1)
                        
                        # Handle fractional seconds
                        fractional_time = remaining_time - int(remaining_time)
                        if fractional_time > 0 and self.running:
                            time.sleep(fractional_time)
                            
                        logger.info(f"5-minute cycle completed, starting next cycle...")
                    else:
                        logger.warning(f"Cycle took {duration:.2f} seconds (longer than {MONITOR_INTERVAL/60:.1f} minute target), starting next cycle immediately...")
                            
                except Exception as cycle_error:
                    # Log the error but continue running - don't let individual cycle errors stop the bot
                    logger.error(f"ERROR in cycle #{self.cycle_count}: {cycle_error}")
                    logger.info("Bot will continue running despite the error...")
                    
                    # Sleep for reduced time on errors before retrying
                    if self.running:
                        error_sleep_time = 60  # 1 minute on errors
                        logger.info(f"Sleeping for {error_sleep_time} seconds before retry...")
                        for i in range(error_sleep_time):
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
            # Ensure all currencies are sold before final shutdown
            if not hasattr(self, '_shutdown_sell_completed'):
                logger.info("Final shutdown - ensuring all currencies are sold...")
                self.sell_all_currencies_on_shutdown()
                self._shutdown_sell_completed = True
                
            logger.info("="*60)
            logger.info(f"Monitor Bot shutting down after {self.cycle_count} cycles")
            logger.info("="*60)

def main():
    """Entry point"""
    bot = MonitorBot()
    bot.run()

if __name__ == "__main__":
    main()