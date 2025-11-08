"""
SMA Trading Strategy Module

Contains the SMA trading decision function and related utilities.
Enhanced with regime-aware trading decisions.
"""

import numpy as np
from typing import List, Dict, Optional

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
    
    # Get trading decision
    decision = sma_trading_decision(example_past_prices, current_price, 
                                  short_window=10, long_window=20)
    
    print(f"Based on past prices and current price of {current_price}")
    print(f"Trading Decision: {decision}")
    
    # Get detailed signal information
    signal_info = get_sma_signals_info(example_past_prices, current_price,
                                     short_window=10, long_window=20)
    
    print(f"\nDetailed Signal Information:")
    print(f"Short SMA (10): {signal_info['short_sma']:.2f}")
    print(f"Long SMA (20): {signal_info['long_sma']:.2f}")
    print(f"Momentum: {signal_info['momentum']:.2f}%")
    print(f"Bullish Crossover: {signal_info['bullish_crossover']}")
    print(f"Bearish Crossover: {signal_info['bearish_crossover']}")