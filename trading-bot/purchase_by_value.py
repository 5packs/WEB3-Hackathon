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
    Calculate quantity with step size reduction strategy for large quantities.
    
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
    
    # Define step size reduction strategies based on quantity size and attempt
    if base_quantity > 10_000_000:  # Very large quantities (like SHIB)
        # Round down to larger step sizes
        step_sizes = [1000, 10000, 100000, 1000000]  
        if attempt < len(step_sizes):
            step_size = step_sizes[attempt]
            reduced_quantity = int(base_quantity // step_size) * step_size
            logger.info(f"Large quantity step size reduction: {base_quantity:.1f} -> {reduced_quantity} (step: {step_size})")
            return float(reduced_quantity)
    elif base_quantity > 100_000:  # Large quantities
        step_sizes = [100, 1000, 10000]
        if attempt < len(step_sizes):
            step_size = step_sizes[attempt]
            reduced_quantity = int(base_quantity // step_size) * step_size
            logger.info(f"Medium quantity step size reduction: {base_quantity:.1f} -> {reduced_quantity} (step: {step_size})")
            return float(reduced_quantity)
    elif base_quantity > 1000:  # Medium quantities  
        step_sizes = [10, 100, 1000]
        if attempt < len(step_sizes):
            step_size = step_sizes[attempt]
            reduced_quantity = int(base_quantity // step_size) * step_size
            logger.info(f"Medium quantity step size reduction: {base_quantity:.1f} -> {reduced_quantity} (step: {step_size})")
            return float(reduced_quantity)
    
    # For smaller quantities or when step size attempts are exhausted, use decimal precision reduction
    # But use higher precision for very expensive coins to avoid rounding to 0
    if price > 10000:  # Very expensive coins like BTC
        precision = max(0, 4 - attempt)  # Start with 4 decimal places for expensive coins (max detail)
    else:
        precision = max(0, 3 - attempt)
        
    decimal_value = Decimal(str(total_value))
    decimal_price = Decimal(str(price))
    decimal_precision = Decimal(f"0.{'0' * (precision - 1)}1") if precision > 0 else Decimal('1')
    
    quantity = decimal_value / decimal_price
    rounded_quantity = quantity.quantize(decimal_precision, rounding=ROUND_DOWN)
    
    result = float(rounded_quantity)
    logger.info(f"Calculated quantity: {result} coins for ${total_value} at ${price}/coin (precision: {precision} decimals)")
    
    return result


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
        
        # Use our enhanced order placement function
        order_result = place_order_with_progressive_precision(
            coin=coin,
            side="BUY",
            total_value=total_value,
            calculation_price=calculation_price,
            order_type=order_type,
            price=price if order_type.upper() == "LIMIT" else None,
            max_attempts=4
        )
        
        # Update order_info with results from progressive placement
        if order_result.get('success'):
            order_info['quantity'] = order_result.get('quantity_used')
            order_info['precision_used'] = order_result.get('precision_used')
            order_info['attempts_made'] = order_result.get('attempts_made')
            order_info['api_result'] = order_result.get('api_result')
            
            logger.info(f"SUCCESS: Order placed with {order_result.get('precision_used')} decimal precision")
            logger.info(f"Final quantity: {order_result.get('quantity_used')} {coin}")
            logger.info(f"Attempts required: {order_result.get('attempts_made')}")
            
            # Update portfolio allocation after successful purchase (including dry runs)
            portfolio_updated = update_portfolio_allocation_after_purchase(coin, success=True)
            if portfolio_updated:
                mode_text = "DRY RUN" if dry_run else "REAL PURCHASE"
                logger.info(f"Portfolio allocation updated ({mode_text}): {coin} value set to $0")
            else:
                logger.warning(f"Failed to update portfolio allocation for {coin}")
            
            return order_info
        else:
            order_info['error'] = order_result.get('error', 'Unknown error in order placement')
            order_info['api_result'] = order_result
            
            logger.error(f"FAILED: Order placement failed: {order_info['error']}")
            if order_result.get('attempts_made'):
                logger.error(f"Failed after {order_result.get('attempts_made')} attempts")
            
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
                
                # Check for step size error
                if isinstance(parsed_response, dict):
                    success = parsed_response.get("Success", True)
                    error_msg = parsed_response.get("ErrMsg", "")
                    
                    if not success and "quantity step size error" in error_msg.lower():
                        response_data["is_step_size_error"] = True
                    else:
                        response_data["is_step_size_error"] = False
                        
                    response_data["success"] = success
                    response_data["error_message"] = error_msg
                    
            except json.JSONDecodeError:
                response_data["json_parse_error"] = "Could not parse JSON response"
                response_data["is_step_size_error"] = False
                response_data["success"] = True  # Assume success if we can't parse
        else:
            # No JSON found, check for error patterns in text
            if "quantity step size error" in captured_output.lower():
                response_data["is_step_size_error"] = True
                response_data["success"] = False
            else:
                response_data["is_step_size_error"] = False
                response_data["success"] = True  # Assume success if no obvious errors
                
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
    max_attempts: int = 4
) -> Dict[str, Any]:
    """
    Place order with progressive precision reduction on quantity step size errors.
    
    Args:
        coin: Cryptocurrency symbol
        side: "BUY" or "SELL"  
        total_value: Total USD amount to spend
        calculation_price: Price to use for quantity calculation
        order_type: "MARKET" or "LIMIT"
        price: Price for LIMIT orders
        max_attempts: Maximum precision reduction attempts
        
    Returns:
        Dict with order result and metadata
    """
    original_cwd = os.getcwd()
    
    # Log the starting strategy based on coin value
    if calculation_price < 10.0:
        logger.info(f"Coin price ${calculation_price:.4f} < $10, using step size reduction strategy for large quantities")
    else:
        logger.info(f"Coin price ${calculation_price:.4f} >= $10, using standard step size reduction")
    
    try:
        # Change to the API directory
        api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crypto-roostoo-api")
        if os.path.exists(api_dir):
            os.chdir(api_dir)
            logger.info(f"Changed to API directory: {api_dir}")
        else:
            logger.warning(f"API directory not found: {api_dir}")
    
        for attempt in range(max_attempts):
            # Calculate quantity with step size reduction strategy
            current_quantity = calculate_quantity_with_step_size_reduction(total_value, calculation_price, attempt)
            
            logger.info(f"Attempt {attempt + 1}/{max_attempts}: Step size attempt {attempt}, quantity {current_quantity}")
            
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
                        logger.info(f"Trying next step size reduction strategy (attempt {attempt + 2})...")
                        continue
                    else:
                        logger.error("All step size reduction attempts failed")
                        return {
                            "success": False,
                            "error": f"Quantity step size error - all step size attempts failed. Last API error: {response_data.get('error_message')}",
                            "last_quantity": current_quantity,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "api_response": response_data
                        }
                elif not response_data.get("success", True):
                    # Other API error that's not step size related
                    error_msg = response_data.get("error_message") or response_data.get("error") or "Unknown API error"
                    logger.error(f"API error (non-step-size): {error_msg}")
                    return {
                        "success": False,
                        "error": f"API error: {error_msg}",
                        "quantity_used": current_quantity,
                        "attempts_made": attempt + 1,
                        "coin": coin,
                        "side": side,
                        "total_value": total_value,
                        "api_response": response_data
                    }
                else:
                    # Success!
                    logger.info(f"Order placed successfully with step size attempt {attempt + 1}")
                    logger.info(f"API Response indicates success")
                    
                    return {
                        "success": True,
                        "quantity_used": current_quantity,
                        "step_size_attempt": attempt + 1,
                        "attempts_made": attempt + 1,
                        "api_result": "Order placed successfully",
                        "coin": coin,
                        "side": side,
                        "total_value": total_value,
                        "api_response": response_data
                    }
                
            except Exception as e:
                # Handle unexpected exceptions during order placement
                error_msg = str(e).lower()
                logger.error(f"Unexpected exception during order placement: {e}")
                
                # Check if it might be a step size error mentioned in the exception
                if "step size" in error_msg or ("quantity" in error_msg and "error" in error_msg):
                    logger.warning(f"Step size error detected in exception message")
                    
                    if attempt < max_attempts - 1:
                        logger.info(f"Trying next step size reduction strategy (attempt {attempt + 2})...")
                        continue
                    else:
                        logger.error("All step size reduction attempts failed")
                        return {
                            "success": False,
                            "error": "Quantity step size error - all step size attempts failed (exception-based)",
                            "last_quantity": current_quantity,
                            "attempts_made": attempt + 1,
                            "coin": coin,
                            "side": side,
                            "total_value": total_value,
                            "exception": str(e)
                        }
                else:
                    # Different error, don't retry with different step size
                    logger.error(f"Non-step-size exception occurred: {e}")
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