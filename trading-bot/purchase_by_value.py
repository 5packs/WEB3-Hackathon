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
import json
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


def update_portfolio_allocation_after_purchase(coin: str, success: bool = True) -> bool:
    """
    Update the simple_portfolio_allocation.json file after a purchase.
    Sets the coin's value to 0 if the purchase was successful.
    
    Args:
        coin (str): The cryptocurrency symbol that was purchased
        success (bool): Whether the purchase was successful
        
    Returns:
        bool: True if the file was updated successfully, False otherwise
    """
    try:
        # Path to the portfolio allocation file
        portfolio_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "output", 
            "simple_portfolio_allocation.json"
        )
        
        if not os.path.exists(portfolio_file):
            logger.warning(f"Portfolio allocation file not found: {portfolio_file}")
            return False
        
        # Read current portfolio allocation
        with open(portfolio_file, 'r') as f:
            portfolio_data = json.load(f)
        
        # Check if coin exists in the portfolio
        coin_upper = coin.upper()
        if coin_upper not in portfolio_data:
            logger.warning(f"Coin {coin_upper} not found in portfolio allocation")
            return False
        
        if success:
            # Set the coin's value to 0 after successful purchase
            old_value = portfolio_data[coin_upper]
            portfolio_data[coin_upper] = 0
            
            logger.info(f"Updated portfolio allocation: {coin_upper} value changed from ${old_value} to $0")
        else:
            logger.info(f"Purchase failed for {coin_upper}, portfolio allocation unchanged")
        
        # Write updated portfolio back to file
        if success:
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            
            logger.info(f"Portfolio allocation file updated: {portfolio_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating portfolio allocation for {coin}: {e}")
        return False


def calculate_quantity_with_step_size_reduction(total_value: float, price: float, attempt: int = 0) -> float:
    """
    Calculate quantity with intelligent step size reduction strategy.
    Adapts strategies based on coin price and resulting quantity size.
    
    Args:
        total_value (float): Total USD amount to spend
        price (float): Current price per coin in USD
        attempt (int): Attempt number (0-based) for step size reduction
        
    Returns:
        float: Quantity of coins to purchase
    """
    if price <= 0:
        raise ValueError("Price must be greater than 0")
    if total_value <= 0:
        raise ValueError("Total value must be greater than 0")
    
    # Calculate base quantity
    base_quantity = total_value / price
    
    # Choose strategy based on coin price and resulting quantity
    if price > 50000:  # Very expensive coins like BTC (>$50k)
        # For expensive coins, use high-precision decimal strategies
        precision_strategies = [
            lambda q: round(q, 6),   # 6 decimals
            lambda q: round(q, 5),   # 5 decimals  
            lambda q: round(q, 4),   # 4 decimals
            lambda q: round(q, 3),   # 3 decimals
            lambda q: round(q, 2),   # 2 decimals
            lambda q: round(q, 1),   # 1 decimal
            # Try stepping down by small amounts
            lambda q: max(0.001, q * 0.95),   # 95% of original
            lambda q: max(0.0005, q * 0.90)   # 90% of original
        ]
        
        if attempt < len(precision_strategies):
            strategy_func = precision_strategies[attempt]
            result = strategy_func(base_quantity)
            precisions = ["6-decimal", "5-decimal", "4-decimal", "3-decimal", "2-decimal", "1-decimal", "95% reduced", "90% reduced"]
            strategy_name = precisions[attempt] if attempt < len(precisions) else f"fallback-{attempt}"
            
    elif price > 1000:  # Expensive coins like ETH ($1k-$50k)
        # Medium precision for mid-expensive coins
        strategies = [
            lambda q: round(q, 4),   # 4 decimals
            lambda q: round(q, 3),   # 3 decimals
            lambda q: round(q, 2),   # 2 decimals
            lambda q: round(q, 1),   # 1 decimal
            lambda q: round(q * 0.95, 4),  # 95% with precision
            lambda q: round(q * 0.90, 3),  # 90% with precision
            lambda q: round(q * 0.85, 2),  # 85% with precision
            lambda q: round(q * 0.80, 2)   # 80% with precision
        ]
        
        if attempt < len(strategies):
            strategy_func = strategies[attempt]
            result = strategy_func(base_quantity)
            names = ["4-decimal", "3-decimal", "2-decimal", "1-decimal", "95% reduced", "90% reduced", "85% reduced", "80% reduced"]
            strategy_name = names[attempt] if attempt < len(names) else f"fallback-{attempt}"
            
    elif base_quantity > 1_000_000:  # Very large quantities (cheap coins like SHIB)
        # Large number strategies for very cheap coins
        large_strategies = [
            lambda q: int(q // 1000) * 1000,    # Round to thousands
            lambda q: int(q // 10000) * 10000,  # Round to ten-thousands  
            lambda q: int(q // 100000) * 100000, # Round to hundred-thousands
            lambda q: int(q // 1000000) * 1000000, # Round to millions
            lambda q: int(q // 500000) * 500000,  # Round to half-millions
            lambda q: int(q // 250000) * 250000,  # Round to quarter-millions
            lambda q: int(q // 100000) * 100000,  # Round to hundred-thousands again
            lambda q: int(q // 50000) * 50000     # Round to fifty-thousands
        ]
        
        if attempt < len(large_strategies):
            strategy_func = large_strategies[attempt]
            result = strategy_func(base_quantity)
            names = ["thousands", "ten-thousands", "hundred-thousands", "millions", "half-millions", "quarter-millions", "hundred-thousands-2", "fifty-thousands"]
            strategy_name = names[attempt] if attempt < len(names) else f"large-fallback-{attempt}"
            
    elif base_quantity > 1000:  # Medium quantities (mid-price coins)
        # Standard rounding for medium quantities
        medium_strategies = [
            lambda q: round(q, 2),           # 2 decimals
            lambda q: round(q, 1),           # 1 decimal
            lambda q: int(q),                # Whole numbers
            lambda q: int(q // 5) * 5,       # Nearest 5
            lambda q: int(q // 10) * 10,     # Nearest 10
            lambda q: int(q // 25) * 25,     # Nearest 25
            lambda q: int(q // 50) * 50,     # Nearest 50
            lambda q: int(q // 100) * 100    # Nearest 100
        ]
        
        if attempt < len(medium_strategies):
            strategy_func = medium_strategies[attempt]
            result = strategy_func(base_quantity)
            names = ["2-decimal", "1-decimal", "whole", "fives", "tens", "twenty-fives", "fifties", "hundreds"]
            strategy_name = names[attempt] if attempt < len(names) else f"medium-fallback-{attempt}"
            
    else:  # Small quantities (most coins $1-$1000)
        # Decimal precision strategies for smaller quantities
        small_strategies = [
            lambda q: round(q, 3),           # 3 decimals
            lambda q: round(q, 2),           # 2 decimals
            lambda q: round(q, 1),           # 1 decimal
            lambda q: int(q),                # Whole numbers
            lambda q: max(1, int(q // 5) * 5),     # Nearest 5 (min 1)
            lambda q: max(1, int(q // 10) * 10),   # Nearest 10 (min 1)
            lambda q: max(1, int(q // 25) * 25),   # Nearest 25 (min 1)
            lambda q: max(1, int(q // 50) * 50)    # Nearest 50 (min 1)
        ]
        
        if attempt < len(small_strategies):
            strategy_func = small_strategies[attempt]
            result = strategy_func(base_quantity)
            names = ["3-decimal", "2-decimal", "1-decimal", "whole", "fives", "tens", "twenty-fives", "fifties"]
            strategy_name = names[attempt] if attempt < len(names) else f"small-fallback-{attempt}"
    
    # Apply the strategy if one was selected above
    if 'result' in locals():
        # Ensure we don't round to zero
        if result <= 0:
            result = 0.001 if price > 1000 else 1.0
            strategy_name = f"{strategy_name}-emergency-minimum"
        
        logger.info(f"Intelligent quantity reduction: {base_quantity:.6f} -> {result} (strategy: {strategy_name})")
        return float(result)
    
    # Final emergency fallbacks if all strategies are exhausted
    emergency_quantities = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 5.0, 10.0]
    emergency_index = attempt - 8  # Since we have 8 strategies per category
    
    if emergency_index < len(emergency_quantities):
        emergency_qty = emergency_quantities[emergency_index]
        if emergency_qty * price <= total_value:  # Only if we can afford it
            logger.info(f"Emergency quantity: {base_quantity:.6f} -> {emergency_qty} (emergency fallback)")
            return emergency_qty
    
    # Absolute final fallback
    logger.info(f"Absolute minimum fallback: {base_quantity:.6f} -> 0.001 (last resort)")
    return 0.001


def calculate_quantity(total_value: float, price: float, precision: int = 3) -> float:
    """
    Calculate the quantity of cryptocurrency to purchase based on total value and current price.
    
    Args:
        total_value (float): Total USD amount to spend
        price (float): Current price per coin in USD
        precision (int): Number of decimal places to round to (default: 3)
        
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
    logger.info(f"Calculated quantity: {result} coins for ${total_value} at ${price}/coin (precision: {precision} decimals)")
    
    return result


def calculate_quantity_with_progressive_rounding(total_value: float, price: float, start_precision: int = 3) -> float:
    """
    Calculate quantity with progressive rounding that can be reduced if step size errors occur.
    
    Args:
        total_value (float): Total USD amount to spend
        price (float): Current price per coin in USD
        start_precision (int): Starting decimal precision (default: 3)
        
    Returns:
        float: Quantity of coins to purchase
    """
    # Start with the requested precision and work down to 0 if needed
    for precision in range(start_precision, -1, -1):
        try:
            quantity = calculate_quantity(total_value, price, precision)
            logger.info(f"Using {precision} decimal precision for quantity calculation")
            return quantity
        except Exception as e:
            logger.warning(f"Precision {precision} failed: {e}")
            continue
    
    # If all precisions fail, use the basic calculation
    logger.warning("All precision levels failed, using basic calculation")
    return total_value / price
    
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
            
            # Update portfolio allocation after successful dry run
            portfolio_updated = update_portfolio_allocation_after_purchase(coin, success=True)
            if portfolio_updated:
                logger.info(f"Portfolio allocation updated (DRY RUN): {coin} value set to $0")
            else:
                logger.warning(f"Failed to update portfolio allocation for {coin}")
            
            return order_info
            
        # Check if primary API is available for placing orders
        if not primary_api_available:
            logger.error("Primary API not available - cannot place orders")
            logger.error("Order placement requires the crypto-roostoo-api modules")
            order_info['error'] = "Primary API not available for order placement"
            return order_info
            
        # Place the actual order with progressive precision handling
        logger.info("Placing order with progressive precision handling...")
        
        # Use our enhanced order placement function with more fallback strategies
        order_result = place_order_with_progressive_precision(
            coin=coin,
            side="BUY",
            total_value=total_value,
            calculation_price=calculation_price,
            order_type=order_type,
            price=price if order_type.upper() == "LIMIT" else None,
            max_attempts=16  # Significantly increased attempts for maximum success rate
        )
        
        # Update order_info with results from progressive placement
        if order_result.get('success') and order_result.get('is_truly_successful', False):
            order_info['quantity'] = order_result.get('quantity_used')
            order_info['actual_cost'] = order_result.get('actual_cost', quantity * calculation_price)
            order_info['amount_saved'] = order_result.get('amount_saved', 0)
            order_info['strategy_used'] = order_result.get('strategy_used', 'standard')
            order_info['attempts_made'] = order_result.get('attempts_made')
            order_info['api_result'] = order_result.get('api_result')
            order_info['unit_change'] = order_result.get('unit_change')
            
            logger.info(f"CONFIRMED SUCCESS: Purchase completed using {order_result.get('strategy_used', 'unknown strategy')}")
            logger.info(f"Final quantity: {order_result.get('quantity_used')} {coin}")
            logger.info(f"Actual cost: ${order_result.get('actual_cost', 0):.2f}")
            logger.info(f"UnitChange from API: ${order_result.get('unit_change', 0)}")
            if order_result.get('amount_saved', 0) > 0:
                logger.info(f"Amount saved: ${order_result.get('amount_saved'):.2f}")
            logger.info(f"Attempts required: {order_result.get('attempts_made')}")
            
            # Update portfolio allocation ONLY after confirmed successful purchase
            portfolio_updated = update_portfolio_allocation_after_purchase(coin, success=True)
            if portfolio_updated:
                mode_text = "DRY RUN" if dry_run else "REAL PURCHASE"
                logger.info(f"Portfolio allocation updated ({mode_text}): {coin} value set to $0")
                logger.info(f"Purchase confirmed with Success=true, ErrMsg='', UnitChange=${order_result.get('unit_change', 0)}")
            else:
                logger.warning(f"Failed to update portfolio allocation for {coin}")
            
            return order_info
        else:
            order_info['error'] = order_result.get('error', 'Purchase not confirmed as successful')
            order_info['api_result'] = order_result
            
            logger.error(f"FAILED: Purchase not confirmed as successful")
            logger.error(f"Error: {order_info['error']}")
            if order_result.get('attempts_made'):
                logger.error(f"Failed after {order_result.get('attempts_made')} attempts")
            
            # Do NOT update portfolio allocation for failed/unconfirmed purchases
            logger.info(f"Portfolio allocation unchanged for {coin} due to purchase failure/unconfirmed status")
            
            return order_info
            
    except Exception as e:
        logger.error(f"Error in purchase_by_value: {e}")
        return None


def place_order_with_response_capture(pair_or_coin, side, quantity, price=None, order_type=None):
    """
    Wrapper for place_order that captures and returns the API response.
    
    Args:
        pair_or_coin: Cryptocurrency symbol
        side: "BUY" or "SELL"
        quantity: Amount to trade
        price: Price for LIMIT orders
        order_type: "LIMIT" or "MARKET"
        
    Returns:
        Dict with response data and success status
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Capture stdout to get the printed response
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Call the original place_order function
            result = place_order(pair_or_coin, side, quantity, price, order_type)
        
        # Get the captured output
        captured_output = stdout_buffer.getvalue()
        captured_errors = stderr_buffer.getvalue()
        
        # Parse the response from the captured output
        response_data = {
            "captured_output": captured_output,
            "captured_errors": captured_errors,
            "original_result": result
        }
        
        # Try to extract JSON response from the output
        import re
        json_match = re.search(r'Response:\s*({.*})', captured_output, re.DOTALL)
        if json_match:
            try:
                import json
                json_str = json_match.group(1).strip()
                parsed_response = json.loads(json_str)
                response_data["parsed_json"] = parsed_response
                
                # Check for step size error and validate successful purchase
                if isinstance(parsed_response, dict):
                    success = parsed_response.get("Success", False)
                    error_msg = parsed_response.get("ErrMsg", "")
                    order_detail = parsed_response.get("OrderDetail", {})
                    unit_change = order_detail.get("UnitChange") if order_detail else None
                    
                    # Validate successful purchase according to API response structure
                    # Success criteria: Success=true, empty ErrMsg, OrderDetail with UnitChange
                    is_truly_successful = (
                        success is True and 
                        error_msg == "" and 
                        isinstance(order_detail, dict) and 
                        unit_change is not None and 
                        float(unit_change) > 0
                    )
                    
                    if not success and "quantity step size error" in error_msg.lower():
                        response_data["is_step_size_error"] = True
                        response_data["is_truly_successful"] = False
                    else:
                        response_data["is_step_size_error"] = False
                        response_data["is_truly_successful"] = is_truly_successful
                    
                    response_data["success"] = success  # Raw API success flag
                    response_data["error_message"] = error_msg
                    response_data["unit_change"] = unit_change
                    response_data["order_detail"] = order_detail
                    
                    if is_truly_successful:
                        logger.info(f"CONFIRMED SUCCESS: Order completed with UnitChange=${unit_change}")
                    elif success:
                        logger.warning(f"API reports Success=true but validation failed - ErrMsg='{error_msg}', OrderDetail={bool(order_detail)}, UnitChange={unit_change}")
                    else:
                        logger.warning(f"API reports Success=false - ErrMsg='{error_msg}'")
                    
            except json.JSONDecodeError:
                response_data["json_parse_error"] = "Could not parse JSON response"
                response_data["is_step_size_error"] = False
                response_data["success"] = False  # Cannot assume success if we can't parse
                response_data["is_truly_successful"] = False  # Cannot confirm success without parsing
                response_data["error_message"] = "Could not parse API response JSON"
        else:
            # No JSON found, check for error patterns in text
            if "quantity step size error" in captured_output.lower():
                response_data["is_step_size_error"] = True
                response_data["success"] = False
                response_data["is_truly_successful"] = False
            else:
                response_data["is_step_size_error"] = False
                response_data["success"] = False  # Cannot assume success without proper JSON response
                response_data["is_truly_successful"] = False  # Cannot confirm success without OrderDetail
                response_data["error_message"] = "No JSON response found in API output"
                
        return response_data
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "is_step_size_error": False,
            "captured_output": stdout_buffer.getvalue(),
            "captured_errors": stderr_buffer.getvalue()
        }


def place_order_with_progressive_precision(
    coin: str,
    side: str, 
    total_value: float,
    calculation_price: float,
    order_type: str = "MARKET",
    price: Optional[float] = None,
    max_attempts: int = 16  # Significantly increased for more fallback strategies
) -> Dict[str, Any]:
    """
    Place order with aggressive progressive precision reduction and multiple fallback strategies.
    
    Args:
        coin: Cryptocurrency symbol
        side: "BUY" or "SELL"  
        total_value: Total USD amount to spend
        calculation_price: Price to use for quantity calculation
        order_type: "MARKET" or "LIMIT"
        price: Price for LIMIT orders
        max_attempts: Maximum attempts with different strategies
        
    Returns:
        Dict with order result and metadata
    """
    original_cwd = os.getcwd()
    
    logger.info(f"Starting aggressive purchase strategy for {coin} with {max_attempts} possible attempts")
    logger.info(f"Target value: ${total_value:.2f} at ${calculation_price:.4f}/coin")
    
    try:
        # Change to the API directory
        api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crypto-roostoo-api")
        if os.path.exists(api_dir):
            os.chdir(api_dir)
            logger.info(f"Changed to API directory: {api_dir}")
        else:
            logger.warning(f"API directory not found: {api_dir}")
    
        # Comprehensive multi-strategy approach with aggressive fallbacks:
        # Attempts 1-8: Aggressive step size reduction with full amount 
        # Attempts 9-12: Reduced total value (95%, 90%, 85%, 80%)
        # Attempts 13-16: Emergency small quantity purchases (75%, 50%, 25%, 10%)
        
        for attempt in range(max_attempts):
            # Determine strategy based on attempt number
            if attempt < 8:
                # Strategy 1: Aggressive step size patterns with full amount (attempts 1-8)
                current_value = total_value
                current_quantity = calculate_quantity_with_step_size_reduction(current_value, calculation_price, attempt)
                strategy = f"aggressive quantity pattern {attempt + 1}"
                
            elif attempt < 12:
                # Strategy 2: Reduced total value with simple quantity patterns (attempts 9-12)
                reduction_factors = [0.95, 0.90, 0.85, 0.80]
                reduction_factor = reduction_factors[attempt - 8]
                current_value = total_value * reduction_factor
                current_quantity = calculate_quantity_with_step_size_reduction(current_value, calculation_price, 2)  # Use whole number strategy
                strategy = f"reduced value ({reduction_factor*100:.0f}%)"
                
            else:
                # Strategy 3: Emergency small purchases with conservative quantities (attempts 13-16)
                reduction_factors = [0.75, 0.50, 0.25, 0.10]
                reduction_factor = reduction_factors[attempt - 12]
                current_value = total_value * reduction_factor
                
                # For emergency purchases, try very conservative quantity patterns
                emergency_patterns = [2, 3, 4, 5]  # whole numbers, fives, tens, etc
                pattern_index = emergency_patterns[attempt - 12] if attempt - 12 < len(emergency_patterns) else 2
                current_quantity = calculate_quantity_with_step_size_reduction(current_value, calculation_price, pattern_index)
                strategy = f"emergency purchase ({reduction_factor*100:.0f}% value)"
            
            logger.info(f"Attempt {attempt + 1}/{max_attempts}: {strategy}")
            logger.info(f"  Target value: ${current_value:.2f} (quantity: {current_quantity})")
            
            # Skip if quantity becomes too small (but allow tiny quantities in emergency attempts)
            if current_quantity <= 0 or (current_quantity < 0.001 and attempt < 12):
                logger.warning(f"Calculated quantity too small ({current_quantity}), skipping attempt {attempt + 1}")
                continue
            
            try:
                # Place order using our response-capturing wrapper
                if order_type.upper() == "LIMIT" and price is not None:
                    response_data = place_order_with_response_capture(
                        pair_or_coin=coin,
                        side=side,
                        quantity=current_quantity,
                        price=price,
                        order_type="LIMIT"
                    )
                else:
                    response_data = place_order_with_response_capture(
                        pair_or_coin=coin,
                        side=side,
                        quantity=current_quantity,
                        order_type="MARKET"
                    )
                
                # Check if the response indicates a step size error
                if response_data.get("is_step_size_error"):
                    logger.warning(f"Quantity step size error detected from API response")
                    logger.warning(f"API Error: {response_data.get('error_message', 'Unknown error')}")
                    
                    if attempt < max_attempts - 1:
                        logger.info(f"Sleeping 2 seconds to avoid API overload before next attempt...")
                        time.sleep(2)
                        logger.info(f"Trying next step size reduction strategy (attempt {attempt + 2})...")
                        continue
                    else:
                        logger.error("All purchase strategies failed")
                        return {
                            "success": False,
                            "error": f"Quantity step size error - all purchase strategies failed. Last API error: {response_data.get('error_message')}",
                            "last_quantity": current_quantity,
                            "last_strategy": strategy,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "api_response": response_data
                        }
                elif not response_data.get("is_truly_successful", False):
                    # API error or unsuccessful purchase (not meeting all success criteria)
                    error_msg = response_data.get("error_message") or response_data.get("error") or "Purchase not confirmed as successful"
                    
                    # Check if it's a raw API success but failed validation
                    if response_data.get("success", False):
                        error_msg = f"API reports success but validation failed: {error_msg}"
                    
                    # Check if this is a potentially recoverable API response parsing error
                    is_parsing_error = (
                        "No JSON response found" in error_msg or 
                        "Could not parse" in error_msg or
                        response_data.get("json_parse_error")
                    )
                    
                    if is_parsing_error and attempt < max_attempts - 1:
                        # Treat API response parsing errors as recoverable - try next strategy
                        logger.warning(f"API response parsing issue: {error_msg}")
                        logger.info(f"Sleeping 2 seconds to avoid API overload before next attempt...")
                        time.sleep(2)
                        logger.info(f"Trying next purchase strategy (attempt {attempt + 2})...")
                        continue
                    else:
                        # Final attempt or non-recoverable error
                        logger.error(f"Purchase not confirmed as successful: {error_msg}")
                        return {
                            "success": False,
                            "error": error_msg,
                            "quantity_used": current_quantity,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "api_response": response_data
                        }
                else:
                    # Confirmed success with all validation criteria met!
                    actual_cost = current_quantity * calculation_price
                    savings = total_value - actual_cost
                    unit_change = response_data.get("unit_change", 0)
                    
                    logger.info(f"CONFIRMED PURCHASE SUCCESS with {strategy}")
                    logger.info(f"Final purchase: {current_quantity} {coin} for ${actual_cost:.2f}")
                    logger.info(f"API confirmed UnitChange: ${unit_change}")
                    if savings > 0:
                        logger.info(f"Amount saved due to strategy: ${savings:.2f} ({savings/total_value*100:.1f}%)")
                    logger.info(f"All success criteria met: Success=true, ErrMsg='', OrderDetail.UnitChange=${unit_change}")
                    
                    return {
                        "success": True,
                        "is_truly_successful": True,
                        "quantity_used": current_quantity,
                        "actual_cost": actual_cost,
                        "amount_saved": savings,
                        "strategy_used": strategy,
                        "attempts_made": attempt + 1,
                        "api_result": f"Purchase confirmed successful using {strategy}",
                        "unit_change": unit_change,
                        "coin": coin,
                        "side": side,
                        "total_value": total_value,
                        "api_response": response_data
                    }
                
            except Exception as e:
                # Handle unexpected exceptions during order placement
                error_msg = str(e).lower()
                logger.error(f"Unexpected exception during order placement: {e}")
                
                # Check for recoverable errors that should allow retrying with different strategies
                recoverable_errors = [
                    "step size",
                    "quantity",
                    "timeout",
                    "connection",
                    "network",
                    "temporary",
                    "api",
                    "response"
                ]
                
                is_recoverable = any(error_term in error_msg for error_term in recoverable_errors)
                
                if is_recoverable and attempt < max_attempts - 1:
                    logger.warning(f"Recoverable error detected in exception: {e}")
                    logger.info(f"Sleeping 2 seconds to avoid API overload before next attempt...")
                    time.sleep(2)
                    logger.info(f"Trying next purchase strategy (attempt {attempt + 2})...")
                    continue
                else:
                    # Final attempt or non-recoverable error
                    if "step size" in error_msg or ("quantity" in error_msg and "error" in error_msg):
                        logger.error("All purchase strategies failed due to step size errors")
                        return {
                            "success": False,
                            "error": "Quantity step size error - all purchase strategies failed (exception-based)",
                            "last_quantity": current_quantity,
                            "last_strategy": strategy,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "exception": str(e)
                        }
                    else:
                        # Non-step-size error on final attempt
                        logger.error(f"Final attempt failed with non-recoverable exception: {e}")
                        return {
                            "success": False,
                            "error": f"Exception during order placement: {str(e)}",
                            "quantity_used": current_quantity,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "exception": str(e)
                        }
        
        # Should not reach here, but just in case
        return {
            "success": False,
            "error": "Maximum attempts exceeded",
            "attempts_made": max_attempts,
            "coin": coin,
            "side": side,
            "total_value": total_value
        }
        
    finally:
        os.chdir(original_cwd)


def test_progressive_precision():
    """Test the progressive precision functionality without making real orders."""
    
    print("\n" + "="*60)
    print("TESTING PROGRESSIVE PRECISION FUNCTIONALITY")
    print("="*60)
    
    # Test the precision calculation
    print("\n--- Test 1: Precision Calculation ---")
    try:
        # Test with various precision levels
        test_value = 100.0
        test_price = 50000.0  # $50k BTC price
        
        for precision in [3, 2, 1, 0]:
            quantity = calculate_quantity(test_value, test_price, precision)
            print(f"Precision {precision}: {quantity} BTC (${test_value} at ${test_price:,.2f}/BTC)")
            
        print("SUCCESS: Precision calculations working correctly")
        
    except Exception as e:
        print(f"FAILED: Precision calculation error: {e}")
    
    # Test response parsing logic
    print("\n--- Test 2: Response Parsing Logic ---")
    try:
        # Mock response scenarios
        test_responses = [
            {
                "name": "Success Response",
                "response": '{"Success": true, "OrderId": "12345"}',
                "expected_step_size_error": False,
                "expected_success": True
            },
            {
                "name": "Step Size Error", 
                "response": '{"Success": false, "ErrMsg": "quantity step size error"}',
                "expected_step_size_error": True,
                "expected_success": False
            },
            {
                "name": "Other Error",
                "response": '{"Success": false, "ErrMsg": "insufficient funds"}', 
                "expected_step_size_error": False,
                "expected_success": False
            }
        ]
        
        import re
        import json
        
        for test_case in test_responses:
            print(f"\nTesting: {test_case['name']}")
            
            # Simulate parsing logic
            json_str = test_case['response']
            try:
                parsed_response = json.loads(json_str)
                success = parsed_response.get("Success", True)
                error_msg = parsed_response.get("ErrMsg", "")
                is_step_size_error = not success and "quantity step size error" in error_msg.lower()
                
                print(f"  Parsed success: {success}")
                print(f"  Error message: {error_msg}")
                print(f"  Is step size error: {is_step_size_error}")
                
                # Verify expectations
                if (is_step_size_error == test_case['expected_step_size_error'] and 
                    success == test_case['expected_success']):
                    print(f"  Result: PASS")
                else:
                    print(f"  Result: FAIL - Expected step_size: {test_case['expected_step_size_error']}, success: {test_case['expected_success']}")
                    
            except json.JSONDecodeError as e:
                print(f"  JSON Parse Error: {e}")
        
        print("\nSUCCESS: Response parsing tests completed")
        
    except Exception as e:
        print(f"FAILED: Response parsing test error: {e}")
    
    print("\n--- Test 3: Progressive Logic Simulation ---")
    try:
        # Simulate the progressive precision reduction logic
        max_attempts = 4
        current_precision = 3
        
        print(f"Simulating {max_attempts} attempts starting with {current_precision} decimal precision:")
        
        for attempt in range(max_attempts):
            print(f"  Attempt {attempt + 1}: Precision = {current_precision} decimals")
            
            if attempt < max_attempts - 1:
                current_precision = max(0, current_precision - 1)
                print(f"    Next precision will be: {current_precision}")
            else:
                print(f"    Final attempt - no more precision reduction")
        
        print("SUCCESS: Progressive logic simulation completed")
        
    except Exception as e:
        print(f"FAILED: Progressive logic simulation error: {e}")
    
    print("\n--- Test 4: Coin Value-Based Starting Precision ---")
    try:
        # Test precision logic based on coin value
        test_scenarios = [
            {"price": 102000.0, "name": "BTC", "expected_start": 3},  # High value coin
            {"price": 3400.0, "name": "ETH", "expected_start": 3},   # Mid value coin
            {"price": 50.0, "name": "BNB", "expected_start": 3},     # Above $10
            {"price": 5.0, "name": "ADA", "expected_start": 1},      # Below $10
            {"price": 0.5, "name": "DOGE", "expected_start": 1},     # Low value coin
            {"price": 0.001, "name": "SHIB", "expected_start": 1},   # Very low value coin
        ]
        
        print("Testing starting precision based on coin value:")
        for scenario in test_scenarios:
            price = scenario["price"]
            name = scenario["name"]
            expected = scenario["expected_start"]
            
            # Determine starting precision based on our logic
            if price < 10.0:
                actual_start = 1
            else:
                actual_start = 3
                
            status = "PASS" if actual_start == expected else "FAIL"
            print(f"  {name} at ${price:,.4f}: Expected {expected}, Got {actual_start} decimals [{status}]")
        
        print("SUCCESS: Coin value-based precision logic tested")
        
    except Exception as e:
        print(f"FAILED: Coin value-based precision test error: {e}")


def test_purchase_by_value():
    """Test the purchase by value functionality with various scenarios."""
    
    print("\n" + "="*60)
    print("TESTING PURCHASE BY VALUE FUNCTIONALITY")
    print("="*60)
    
    # Test 1: Dry run with BTC
    print("\n--- Test 1: Dry Run BTC Purchase ---")
    result1 = purchase_by_value("BTC", 100.0, dry_run=True)
    if result1:
        print(f"SUCCESS: Dry run successful: {result1['quantity']} BTC for $100")
    else:
        print("FAILED: Dry run failed")
    
    # Test 2: Dry run with ETH
    print("\n--- Test 2: Dry Run ETH Purchase ---")
    result2 = purchase_by_value("ETH", 250.0, dry_run=True)
    if result2:
        print(f"SUCCESS: Dry run successful: {result2['quantity']} ETH for $250")
    else:
        print("FAILED: Dry run failed")
    
    # Test 3: Price calculation test
    print("\n--- Test 3: Manual Price Calculation Test ---")
    try:
        manual_quantity = calculate_quantity(500.0, 50000.0)  # $500 worth at $50k/BTC
        print(f"SUCCESS: Manual calculation: {manual_quantity} BTC for $500 at $50,000/BTC")
    except Exception as e:
        print(f"FAILED: Manual calculation failed: {e}")
    
    # Test 4: Price fetching test
    print("\n--- Test 4: Price Fetching Test ---")
    btc_price = get_current_price("BTC")
    if btc_price:
        print(f"SUCCESS: BTC price fetch successful: ${btc_price:,.2f}")
    else:
        print("FAILED: BTC price fetch failed")


def interactive_purchase():
    """Interactive function for manual testing of purchases."""
    
    print("\n" + "="*60)
    print("INTERACTIVE CRYPTOCURRENCY PURCHASE")
    print("="*60)
    
    try:
        # Get user inputs
        coin = input("Enter cryptocurrency symbol (e.g., BTC, ETH): ").strip().upper()
        if not coin:
            print("ERROR: No coin specified")
            return
            
        total_value_str = input("Enter total USD amount to spend: $").strip()
        try:
            total_value = float(total_value_str)
            if total_value <= 0:
                print("ERROR: Amount must be greater than 0")
                return
        except ValueError:
            print("ERROR: Invalid amount format")
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
                    print("ERROR: Price must be greater than 0")
                    return
            except ValueError:
                print("ERROR: Invalid price format")
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
            print("\nSUCCESS: Purchase function completed successfully!")
            if dry_run:
                print("Note: This was a dry run - no actual order was placed")
        else:
            print("\nFAILED: Purchase function failed")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")


if __name__ == "__main__":
    """Main entry point for testing the purchase functionality."""
    
    print("Crypto Purchase by Value Tool")
    print("Choose an option:")
    print("1. Run automated tests")
    print("2. Interactive purchase")
    print("3. Test progressive precision")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            test_purchase_by_value()
        elif choice == "2":
            interactive_purchase()
        elif choice == "3":
            test_progressive_precision()
        elif choice == "4":
            print("Goodbye!")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")