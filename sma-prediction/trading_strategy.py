"""
SMA Trading Strategy Module

Contains the SMA trading decision function and related utilities.
Enhanced with regime-aware trading decisions.
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple

def sma_trading_decision(past_prices: List[float], current_price: float, 
                        short_window: int = 25, long_window: int = 45) -> str:
    """
    Make a buy/sell/hold decision based on SMA crossover strategy.
    
    Args:
        past_prices: List of historical prices (should include at least long_window prices)
        current_price: The current price to evaluate
        short_window: Period for short-term SMA (default: 10)
        long_window: Period for long-term SMA (default: 50)
    
    Returns:
        'BUY', 'SELL', or 'HOLD' decision
    """
    # Combine past prices with current price
    all_prices = past_prices + [current_price]
    all_prices = np.array(all_prices, dtype=float)
    
    # Check if we have enough data
    if len(all_prices) < long_window:
        return 'HOLD'  # Not enough data for reliable signal
    
    # Calculate SMAs
    short_sma = np.mean(all_prices[-short_window:])
    long_sma = np.mean(all_prices[-long_window:])
    
    # Calculate previous SMAs (if we have enough data)
    if len(all_prices) < long_window + 1:
        return 'HOLD'  # Need at least one more data point for crossover detection
    
    prev_short_sma = np.mean(all_prices[-short_window-1:-1])
    prev_long_sma = np.mean(all_prices[-long_window-1:-1])
    
    # Check for crossover signals
    # Buy signal: short SMA crosses above long SMA
    if prev_short_sma <= prev_long_sma and short_sma > long_sma:
        return 'BUY'
    
    # Sell signal: short SMA crosses below long SMA  
    elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
        return 'SELL'
    
    # Additional momentum check - stronger signals
    # If short SMA is significantly above/below long SMA, maintain position
    sma_diff_percent = ((short_sma - long_sma) / long_sma) * 100
    
    if sma_diff_percent > 2.0:  # Short SMA is 2% above long SMA
        return 'BUY'
    elif sma_diff_percent < -2.0:  # Short SMA is 2% below long SMA
        return 'SELL'
    
    return 'HOLD'


def calculate_sma(prices: List[float], window: int) -> float:
    """
    Calculate Simple Moving Average for a given window.
    
    Args:
        prices: List of price values
        window: Number of periods for the moving average
    
    Returns:
        SMA value as float
    """
    if len(prices) < window:
        return np.mean(prices)  # Use available data if less than window
    
    return np.mean(prices[-window:])


def get_sma_signals_info(past_prices: List[float], current_price: float,
                        short_window: int = 10, long_window: int = 50) -> dict:
    """
    Get detailed information about SMA signals and values.
    
    Args:
        past_prices: List of historical prices
        current_price: Current price
        short_window: Short SMA window
        long_window: Long SMA window
    
    Returns:
        Dictionary with SMA values and signal information
    """
    all_prices = past_prices + [current_price]
    all_prices = np.array(all_prices, dtype=float)
    
    if len(all_prices) < long_window + 1:
        return {
            'signal': 'HOLD',
            'short_sma': None,
            'long_sma': None,
            'prev_short_sma': None,
            'prev_long_sma': None,
            'crossover': False,
            'momentum': 0.0,
            'sufficient_data': False
        }
    
    # Calculate current SMAs
    short_sma = np.mean(all_prices[-short_window:])
    long_sma = np.mean(all_prices[-long_window:])
    
    # Calculate previous SMAs
    prev_short_sma = np.mean(all_prices[-short_window-1:-1])
    prev_long_sma = np.mean(all_prices[-long_window-1:-1])
    
    # Detect crossovers
    bullish_crossover = prev_short_sma <= prev_long_sma and short_sma > long_sma
    bearish_crossover = prev_short_sma >= prev_long_sma and short_sma < long_sma
    
    # Calculate momentum
    momentum = ((short_sma - long_sma) / long_sma) * 100
    
    # Get signal
    signal = sma_trading_decision(past_prices, current_price, short_window, long_window)
    
    return {
        'signal': signal,
        'short_sma': short_sma,
        'long_sma': long_sma,
        'prev_short_sma': prev_short_sma,
        'prev_long_sma': prev_long_sma,
        'bullish_crossover': bullish_crossover,
        'bearish_crossover': bearish_crossover,
        'momentum': momentum,
        'sufficient_data': True
    }


def load_optimal_sma_parameters(filepath: str = "output/optimal_sma_parameters.json") -> Dict:
    """
    Load optimal SMA parameters for all currencies from the optimizer output.
    
    Args:
        filepath: Path to the optimal parameters JSON file
        
    Returns:
        Dictionary containing optimal parameters for each currency
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "parameters" in data:
            print(f"[OK] Loaded optimal SMA parameters for {len(data['parameters'])} currencies")
            return data["parameters"]
        else:
            print("[WARNING] Invalid parameters file format")
            return {}
            
    except FileNotFoundError:
        print(f"[ERROR] Parameters file not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in parameters file: {filepath}")
        return {}


def load_simple_sma_parameters(filepath: str = "output/simple_sma_parameters.json") -> Dict:
    """
    Load simplified SMA parameters (just short/long windows) for bot usage.
    
    Args:
        filepath: Path to the simple parameters JSON file
        
    Returns:
        Dictionary with short/long window parameters for each currency
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"[OK] Loaded simplified SMA parameters for {len(data)} currencies from {filepath}")
        return data
            
    except FileNotFoundError:
        print(f"[ERROR] Simple parameters file not found: {filepath}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in simple parameters file: {filepath}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error loading parameters file {filepath}: {e}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}


def get_optimal_parameters_for_currency(currency: str, 
                                       parameters_file: str = "output/simple_sma_parameters.json") -> Tuple[int, int]:
    """
    Get optimal SMA parameters for a specific currency.
    
    Args:
        currency: Currency symbol (e.g., 'BTC', 'ETH')
        parameters_file: Path to parameters file
        
    Returns:
        Tuple of (short_window, long_window). Returns (25, 45) as default if not found.
    """
    params = load_simple_sma_parameters(parameters_file)
    
    if currency in params:
        return params[currency]["short"], params[currency]["long"]
    else:
        print(f"[WARNING] No optimal parameters found for {currency}, using defaults (25, 45)")
        return 25, 45


def make_optimized_trading_decision(currency: str, past_prices: List[float], 
                                  current_price: float,
                                  parameters_file: str = "output/simple_sma_parameters.json") -> str:
    """
    Make trading decision using optimized SMA parameters for the specific currency.
    
    Args:
        currency: Currency symbol (e.g., 'BTC', 'ETH')
        past_prices: List of historical prices
        current_price: Current price
        parameters_file: Path to optimized parameters file
        
    Returns:
        Trading decision: 'BUY', 'SELL', or 'HOLD'
    """
    short_window, long_window = get_optimal_parameters_for_currency(currency, parameters_file)
    return sma_trading_decision(past_prices, current_price, short_window, long_window)


# Example usage and testing
if __name__ == "__main__":
    # Example usage of the SMA trading decision function
    print("SMA Trading Strategy - Example Usage")
    print("-" * 40)
    
    # Simulate some price data
    example_past_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                          111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
                          119, 121, 123, 122, 124, 126, 125, 127, 129, 128,
                          130, 132, 131, 133, 135, 134, 136, 138, 137, 139,
                          141, 140, 142, 144, 143, 145, 147, 146, 148, 150]
    
    current_price = 152.0
    
    # Standard trading decision with default parameters
    decision = sma_trading_decision(example_past_prices, current_price, 
                                  short_window=10, long_window=20)
    
    print(f"Standard Decision (SMA 10/20): {decision}")
    
    # Test optimized trading decision (will use defaults if no file exists)
    optimized_decision = make_optimized_trading_decision("BTC", example_past_prices, current_price)
    print(f"Optimized Decision for BTC: {optimized_decision}")
    
    # Get detailed signal information
    signal_info = get_sma_signals_info(example_past_prices, current_price,
                                     short_window=10, long_window=20)
    
    print(f"\nDetailed Signal Information:")
    print(f"Short SMA (10): {signal_info['short_sma']:.2f}")
    print(f"Long SMA (20): {signal_info['long_sma']:.2f}")
    print(f"Momentum: {signal_info['momentum']:.2f}%")
    print(f"Bullish Crossover: {signal_info['bullish_crossover']}")
    print(f"Bearish Crossover: {signal_info['bearish_crossover']}")
    
    # Test parameter loading functions
    print(f"\nTesting Parameter Loading Functions:")
    print("-" * 40)
    
    # Try to load optimal parameters
    optimal_params = load_optimal_sma_parameters()
    if optimal_params:
        print(f"Found optimal parameters for currencies: {list(optimal_params.keys())[:5]}...")
    
    # Try to load simple parameters  
    simple_params = load_simple_sma_parameters()
    if simple_params:
        print(f"Found simple parameters for currencies: {list(simple_params.keys())[:5]}...")
        
    # Test getting parameters for specific currencies
    for currency in ['BTC', 'ETH', 'XRP']:
        short, long = get_optimal_parameters_for_currency(currency)
        print(f"{currency}: SMA({short}, {long})")