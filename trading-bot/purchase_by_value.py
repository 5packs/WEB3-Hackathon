#!/usr/bin/env python3
"""
Purchase Cryptocurrency by Total Dollar Value

This module provides functionality to purchase cryptocurrency by specifying
the total dollar amount to spend, rather than the quantity of coins.
It automatically calculates the required quantity based on current market price.
"""

import os
import sys
import logging
import requests
import time
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add crypto-roostoo-api to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
api_path = os.path.join(parent_dir, 'crypto-roostoo-api')
if api_path not in sys.path:
    sys.path.insert(0, api_path)

try:
    from utilities import get_ticker
    from trades import place_order
    primary_api_available = True
except ImportError as e:
    logger.warning(f"Primary API modules not available: {e}")
    logger.warning("Will use backup Kraken API only")
    primary_api_available = False


def get_kraken_price(coin: str) -> Optional[float]:
    """
    Backup function to get cryptocurrency price from Kraken API.
    
    Args:
        coin (str): The cryptocurrency symbol (e.g., "BTC", "ETH")
        
    Returns:
        float: Current price in USD from Kraken, or None if failed
    """
    try:
        # Map common symbols to Kraken format
        kraken_symbol_map = {
            'BTC': 'XBTUSD',
            'ETH': 'ETHUSD', 
            'LTC': 'LTCUSD',
            'XRP': 'XRPUSD',
            'ADA': 'ADAUSD',
            'DOT': 'DOTUSD',
            'LINK': 'LINKUSD',
            'UNI': 'UNIUSD',
            'AVAX': 'AVAXUSD',
            'SOL': 'SOLUSD',
            'AAVE': 'AAVEUSD',
            'FIL': 'FILUSD',
            'ICP': 'ICPUSD',
            'NEAR': 'NEARUSD',
            'TON': 'TONUSD'
        }
        
        coin_upper = coin.upper()
        kraken_pair = kraken_symbol_map.get(coin_upper, f"{coin_upper}USD")
        
        logger.info(f"Fetching price from Kraken API for {kraken_pair}")
        
        # Kraken API endpoint for ticker data
        url = "https://api.kraken.com/0/public/Ticker"
        params = {'pair': kraken_pair}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('error'):
            logger.error(f"Kraken API error: {data['error']}")
            return None
            
        result = data.get('result', {})
        
        # Kraken returns data with the actual pair name as key
        for pair_key, pair_data in result.items():
            if pair_key.replace('X', '').replace('Z', '') == kraken_pair or pair_key == kraken_pair:
                # Get the last trade price (index 0 in 'c' array)
                last_price = pair_data.get('c', [None])[0]
                if last_price:
                    price = float(last_price)
                    logger.info(f"Kraken price for {coin_upper}: ${price:,.4f}")
                    return price
        
        # If exact match not found, try the first result
        if result:
            first_pair = next(iter(result.values()))
            last_price = first_pair.get('c', [None])[0]
            if last_price:
                price = float(last_price)
                logger.info(f"Kraken price for {coin_upper} (fallback): ${price:,.4f}")
                return price
        
        logger.error(f"No price data found in Kraken response for {coin_upper}")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching from Kraken: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error parsing Kraken response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with Kraken API: {e}")
        return None


def get_current_price(coin: str) -> Optional[float]:
    """
    Get the current market price for a cryptocurrency.
    First tries the primary API, then falls back to Kraken API.
    
    Args:
        coin (str): The cryptocurrency symbol (e.g., "BTC", "ETH")
        
    Returns:
        float: Current price in USD, or None if both APIs failed
    """
    # Try primary API first if available
    if primary_api_available:
        try:
            # Ensure coin is in correct format
            if "/" not in coin:
                pair = f"{coin.upper()}/USD"
            else:
                pair = coin.upper()
                
            logger.info(f"Fetching current price for {pair} from primary API")
            ticker_data = get_ticker(pair=pair)
            
            if ticker_data and ticker_data.get('Data'):
                price_data = ticker_data['Data'].get(pair, {})
                last_price = price_data.get('LastPrice')
                
                if last_price:
                    price = float(last_price)
                    logger.info(f"Primary API price for {pair}: ${price:,.4f}")
                    return price
                else:
                    logger.warning(f"LastPrice not found in primary API data for {pair}")
            else:
                logger.warning(f"Failed to get ticker data from primary API for {pair}")
                
        except Exception as e:
            logger.warning(f"Primary API error for {coin}: {e}")
    
    # Fall back to Kraken API
    logger.info(f"Trying backup Kraken API for {coin}")
    kraken_price = get_kraken_price(coin)
    
    if kraken_price:
        return kraken_price
    
    logger.error(f"Both primary API and Kraken backup failed for {coin}")
    return None


def calculate_quantity(total_value: float, price: float, precision: int = 8) -> float:
    """
    Calculate the quantity of cryptocurrency to purchase based on total value and current price.
    
    Args:
        total_value (float): Total USD amount to spend
        price (float): Current price per coin in USD
        precision (int): Number of decimal places to round to (default: 8)
        
    Returns:
        float: Quantity of coins to purchase
    """
    if price <= 0:
        raise ValueError("Price must be greater than 0")
    if total_value <= 0:
        raise ValueError("Total value must be greater than 0")
    
    # Use Decimal for precise calculation
    decimal_value = Decimal(str(total_value))
    decimal_price = Decimal(str(price))
    decimal_precision = Decimal(f"0.{'0' * (precision - 1)}1")
    
    # Calculate quantity and round down to avoid insufficient funds
    quantity = decimal_value / decimal_price
    rounded_quantity = quantity.quantize(decimal_precision, rounding=ROUND_DOWN)
    
    result = float(rounded_quantity)
    logger.info(f"Calculated quantity: {result} coins for ${total_value} at ${price}/coin")
    
    return result


def purchase_by_value(
    coin: str, 
    total_value: float, 
    order_type: str = "MARKET",
    price: Optional[float] = None,
    dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Purchase cryptocurrency by specifying the total dollar amount to spend.
    
    Args:
        coin (str): Cryptocurrency symbol (e.g., "BTC", "ETH")
        total_value (float): Total USD amount to spend
        order_type (str): "MARKET" or "LIMIT" (default: "MARKET")
        price (float, optional): Price for LIMIT orders. Uses current market price if None
        dry_run (bool): If True, only calculate and log, don't place actual order
        
    Returns:
        dict: Order result information, or None if failed
    """
    try:
        logger.info(f"Starting purchase by value: ${total_value} worth of {coin.upper()}")
        
        # Get current market price
        current_price = get_current_price(coin)
        if current_price is None:
            logger.error(f"Failed to get current price for {coin}")
            return None
        
        # Determine the price to use for calculation
        if order_type.upper() == "LIMIT":
            if price is None:
                logger.warning("LIMIT order without specified price, using current market price")
                calculation_price = current_price
            else:
                calculation_price = price
                logger.info(f"Using specified LIMIT price: ${price:,.4f}")
        else:
            calculation_price = current_price
            
        # Calculate required quantity
        try:
            quantity = calculate_quantity(total_value, calculation_price)
        except ValueError as e:
            logger.error(f"Invalid calculation parameters: {e}")
            return None
            
        if quantity <= 0:
            logger.error("Calculated quantity is zero or negative")
            return None
            
        # Prepare order details
        order_info = {
            'coin': coin.upper(),
            'total_value': total_value,
            'price_used': calculation_price,
            'quantity': quantity,
            'order_type': order_type.upper(),
            'estimated_cost': quantity * calculation_price
        }
        
        logger.info(f"Order details:")
        logger.info(f"  Coin: {order_info['coin']}")
        logger.info(f"  Quantity: {order_info['quantity']}")
        logger.info(f"  Order Type: {order_info['order_type']}")
        logger.info(f"  Price Used: ${order_info['price_used']:,.4f}")
        logger.info(f"  Estimated Cost: ${order_info['estimated_cost']:,.2f}")
        
        if dry_run:
            logger.info("DRY RUN: Order not placed")
            return order_info
            
        # Check if primary API is available for placing orders
        if not primary_api_available:
            logger.error("Primary API not available - cannot place orders")
            logger.error("Order placement requires the crypto-roostoo-api modules")
            order_info['error'] = "Primary API not available for order placement"
            return order_info
            
        # Place the actual order
        logger.info("Placing order...")
        
        # Change to API directory for proper .env loading
        original_cwd = os.getcwd()
        os.chdir(api_path)
        
        try:
            if order_type.upper() == "LIMIT" and price is not None:
                result = place_order(
                    pair_or_coin=coin,
                    side="BUY",
                    quantity=quantity,
                    price=price,
                    order_type="LIMIT"
                )
            else:
                result = place_order(
                    pair_or_coin=coin,
                    side="BUY",
                    quantity=quantity,
                    order_type="MARKET"
                )
            
            order_info['api_result'] = result
            logger.info("Order placement attempt completed")
            return order_info
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"Error in purchase_by_value: {e}")
        return None


def test_purchase_by_value():
    """Test the purchase by value functionality with various scenarios."""
    
    print("\n" + "="*60)
    print("TESTING PURCHASE BY VALUE FUNCTIONALITY")
    print("="*60)
    
    # Test 1: Dry run with BTC
    print("\n--- Test 1: Dry Run BTC Purchase ---")
    result1 = purchase_by_value("BTC", 100.0, dry_run=True)
    if result1:
        print(f"✅ Dry run successful: {result1['quantity']} BTC for $100")
    else:
        print("❌ Dry run failed")
    
    # Test 2: Dry run with ETH
    print("\n--- Test 2: Dry Run ETH Purchase ---")
    result2 = purchase_by_value("ETH", 250.0, dry_run=True)
    if result2:
        print(f"✅ Dry run successful: {result2['quantity']} ETH for $250")
    else:
        print("❌ Dry run failed")
    
    # Test 3: Price calculation test
    print("\n--- Test 3: Manual Price Calculation Test ---")
    try:
        manual_quantity = calculate_quantity(500.0, 50000.0)  # $500 worth at $50k/BTC
        print(f"✅ Manual calculation: {manual_quantity} BTC for $500 at $50,000/BTC")
    except Exception as e:
        print(f"❌ Manual calculation failed: {e}")
    
    # Test 4: Price fetching test
    print("\n--- Test 4: Price Fetching Test ---")
    btc_price = get_current_price("BTC")
    if btc_price:
        print(f"✅ BTC price fetch successful: ${btc_price:,.2f}")
    else:
        print("❌ BTC price fetch failed")


def interactive_purchase():
    """Interactive function for manual testing of purchases."""
    
    print("\n" + "="*60)
    print("INTERACTIVE CRYPTOCURRENCY PURCHASE")
    print("="*60)
    
    try:
        # Get user inputs
        coin = input("Enter cryptocurrency symbol (e.g., BTC, ETH): ").strip().upper()
        if not coin:
            print("❌ No coin specified")
            return
            
        total_value_str = input("Enter total USD amount to spend: $").strip()
        try:
            total_value = float(total_value_str)
            if total_value <= 0:
                print("❌ Amount must be greater than 0")
                return
        except ValueError:
            print("❌ Invalid amount format")
            return
            
        order_type = input("Order type (MARKET/LIMIT) [default: MARKET]: ").strip().upper()
        if not order_type:
            order_type = "MARKET"
            
        price = None
        if order_type == "LIMIT":
            price_str = input("Enter LIMIT price per coin: $").strip()
            try:
                price = float(price_str)
                if price <= 0:
                    print("❌ Price must be greater than 0")
                    return
            except ValueError:
                print("❌ Invalid price format")
                return
                
        dry_run_input = input("Dry run only? (y/N) [default: Yes]: ").strip().lower()
        dry_run = dry_run_input != 'n'
        
        # Execute purchase
        print(f"\n{'DRY RUN: ' if dry_run else ''}Purchasing ${total_value} worth of {coin}...")
        
        result = purchase_by_value(
            coin=coin,
            total_value=total_value,
            order_type=order_type,
            price=price,
            dry_run=dry_run
        )
        
        if result:
            print("\n✅ Purchase function completed successfully!")
            if dry_run:
                print("Note: This was a dry run - no actual order was placed")
        else:
            print("\n❌ Purchase function failed")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    """Main entry point for testing the purchase functionality."""
    
    print("Crypto Purchase by Value Tool")
    print("Choose an option:")
    print("1. Run automated tests")
    print("2. Interactive purchase")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            test_purchase_by_value()
        elif choice == "2":
            interactive_purchase()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")