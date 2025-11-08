import requests
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional, Dict, Any


def fetch_kraken_ohlc(pair="XBTUSD", interval=1):
    """
    Fetch OHLC data from Kraken API
    
    Args:
        pair: Trading pair (e.g., "XBTUSD" for Bitcoin)
        interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
    
    Returns:
        List of OHLC data: [timestamp, open, high, low, close, vwap, volume, count]
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": pair,
        "interval": interval,
        "since": 0  # all history, you can pass timestamp for start
    }
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle potential errors from Kraken API
        if data.get("error"):
            raise Exception(f"Kraken API error: {data['error']}")
        
        # Get the first available pair from result (in case pair name differs)
        result_keys = list(data["result"].keys())
        if not result_keys:
            raise Exception("No data returned from Kraken API")
        
        # Remove 'last' key if present and get the pair data
        pair_key = [k for k in result_keys if k != 'last'][0]
        result = data["result"][pair_key]
        return result
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []
    
def fetch_kraken_ohlc_recent(pair="XBTUSD", interval=1, count: int = 100):
    """
    Fetch only the most recent OHLC data from Kraken API
    
    Args:
        pair: Trading pair (e.g., "XBTUSD" for Bitcoin)
        interval: Time interval in minutes
        count: Number of most recent entries to return (default: 100)
    
    Returns:
        List of most recent OHLC data: [timestamp, open, high, low, close, vwap, volume, count]
    """
    # Get all data from the original function
    all_data = fetch_kraken_ohlc(pair, interval)
    
    if not all_data:
        return []
    
    # Return only the most recent 'count' entries
    recent_data = all_data[-count:] if len(all_data) > count else all_data
    
    if len(all_data) > count:
        print(f"[DATA] Returning {count} most recent entries out of {len(all_data)} total for {pair}")
    
    return recent_data


def save_prices_to_csv(ohlc_data: List, filename: str = "price_data.csv", append: bool = False):
    """
    Save OHLC data to CSV file
    
    Args:
        ohlc_data: List of OHLC data from Kraken API
        filename: Name of the CSV file
        append: If True, append to existing file; if False, overwrite
    """
    if not ohlc_data:
        print("No data to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlc_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    
    # Convert timestamp to readable datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Convert price columns to float
    price_columns = ['open', 'high', 'low', 'close', 'vwap']
    for col in price_columns:
        df[col] = df[col].astype(float)
    
    df['volume'] = df['volume'].astype(float)
    df['count'] = df['count'].astype(int)
    
    # Reorder columns for better readability
    df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']]
    
    # Save to CSV
    mode = 'a' if append and os.path.exists(filename) else 'w'
    header = not (append and os.path.exists(filename))
    
    df.to_csv(filename, mode=mode, header=header, index=False)
    print(f"Saved {len(df)} records to {filename}")


def load_prices_from_csv(filename: str = "price_data.csv") -> Optional[pd.DataFrame]:
    """
    Load price data from CSV file
    
    Args:
        filename: Name of the CSV file
    
    Returns:
        DataFrame with price data or None if file doesn't exist
    """
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return None
    
    try:
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} records from {filename}")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def get_latest_timestamp(filename: str = "price_data.csv") -> Optional[int]:
    """
    Get the latest timestamp from existing CSV file
    
    Args:
        filename: Name of the CSV file
    
    Returns:
        Latest timestamp or None if file doesn't exist or is empty
    """
    df = load_prices_from_csv(filename)
    if df is not None and not df.empty:
        return int(df['timestamp'].max())
    return None


def fetch_and_store_prices(pair: str = "XBTUSD", interval: int = 5, 
                          filename: str = "price_data.csv", count: int = 0):
    """
    Fetch prices from Kraken and store them in CSV
    
    Args:
        pair: Trading pair
        interval: Time interval in minutes
        filename: CSV filename
        update_mode: If True, only fetch new data since last stored timestamp
    """
    if (count != 0):
        print(f"Fetching {count} data entries")
        # Modify the fetch function to accept 'since' parameter
        ohlc_data = fetch_kraken_ohlc_recent(pair, interval, count)
        save_prices_to_csv(ohlc_data, filename, append=True)
    else:
        print("Fetching all available data")
        ohlc_data = fetch_kraken_ohlc(pair, interval)
        save_prices_to_csv(ohlc_data, filename, append=False)


def get_close_prices_for_backtest(filename: str = "price_data.csv") -> List[float]:
    """
    Extract close prices from CSV for backtesting
    
    Args:
        filename: CSV filename
    
    Returns:
        List of close prices as floats
    """
    df = load_prices_from_csv(filename)
    if df is not None and not df.empty:
        return df['close'].tolist()
    return []


# Example usage
if __name__ == "__main__":
    # Fetch and store initial data
    print("Fetching Bitcoin price data...")
    fetch_and_store_prices(pair="XBTUSD", interval=5, filename="btc_5m_data.csv")
    
    # Load data for backtesting
    prices = get_close_prices_for_backtest("btc_5m_data.csv")
    print(f"Loaded {len(prices)} prices for backtesting")
    
    # Display first few prices
    if prices:
        print(f"First 5 prices: {prices[:5]}")
        print(f"Last 5 prices: {prices[-5:]}")