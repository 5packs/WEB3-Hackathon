import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime
import sys
import os

# Import the SMA trading decision function from trading_strategy.py
try:
    from trading_strategy import sma_trading_decision
except ImportError:
    # If import fails, define a basic version here
    def sma_trading_decision(past_prices: List[float], current_price: float, 
                            short_window: int = 10, long_window: int = 50) -> str:
        import numpy as np
        all_prices = past_prices + [current_price]
        all_prices = np.array(all_prices, dtype=float)
        
        if len(all_prices) < long_window + 1:
            return 'HOLD'
        
        short_sma = np.mean(all_prices[-short_window:])
        long_sma = np.mean(all_prices[-long_window:])
        prev_short_sma = np.mean(all_prices[-short_window-1:-1])
        prev_long_sma = np.mean(all_prices[-long_window-1:-1])
        
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            return 'BUY'
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            return 'SELL'
        
        sma_diff_percent = ((short_sma - long_sma) / long_sma) * 100
        if sma_diff_percent > 2.0:
            return 'BUY'
        elif sma_diff_percent < -2.0:
            return 'SELL'
        
        return 'HOLD'


class SMABacktester:
    """
    A comprehensive backtesting framework for SMA trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital in USD
            commission: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset all tracking variables."""
        self.capital = self.initial_capital
        self.position = 0.0  # Number of shares/units held
        self.cash = self.initial_capital
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        self.returns = []
        
    def load_data_from_csv(self, filename: str = "price_data.csv") -> pd.DataFrame:
        """
        Load price data from CSV file.
        
        Args:
            filename: Path to CSV file containing price data
            
        Returns:
            DataFrame with price data
        """
        try:
            df = pd.read_csv(filename)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            print(f"Loaded {len(df)} price records from {filename}")
            return df
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def execute_trade(self, price: float, signal: str, timestamp: str = None, index: int = None):
        """
        Execute a trade based on the signal.
        
        Args:
            price: Current price
            signal: 'BUY', 'SELL', or 'HOLD'
            timestamp: Optional timestamp for the trade
            index: Index position in the price series
        """
        if signal == 'BUY' and self.cash > 0:
            # Buy as many shares as possible with available cash
            commission_cost = self.cash * self.commission
            available_cash = self.cash - commission_cost
            shares_to_buy = available_cash / price
            
            if shares_to_buy > 0:
                self.position += shares_to_buy
                self.cash = 0  # All cash used
                
                self.trades.append({
                    'timestamp': timestamp,
                    'index': index,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'commission': commission_cost,
                    'portfolio_value': self.position * price
                })
                
        elif signal == 'SELL' and self.position > 0:
            # Sell all shares
            sale_value = self.position * price
            commission_cost = sale_value * self.commission
            self.cash = sale_value - commission_cost
            
            self.trades.append({
                'timestamp': timestamp,
                'index': index,
                'action': 'SELL',
                'price': price,
                'shares': self.position,
                'commission': commission_cost,
                'portfolio_value': self.cash
            })
            
            self.position = 0
    
    def calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        return self.cash + (self.position * current_price)
    
    def run_backtest(self, df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> Dict:
        """
        Run the backtest on price data.
        
        Args:
            df: DataFrame with price data (must have 'close' column)
            short_window: Short SMA window
            long_window: Long SMA window
            
        Returns:
            Dictionary with backtest results
        """
        if df.empty or 'close' not in df.columns:
            print("Error: Invalid data format. Need 'close' column.")
            return {}
        
        self.reset()
        prices = df['close'].values
        timestamps = df['datetime'].values if 'datetime' in df.columns else range(len(prices))
        
        # Need enough data for SMA calculation
        if len(prices) < long_window + 1:
            print(f"Error: Need at least {long_window + 1} data points")
            return {}
        
        print(f"Running backtest with {len(prices)} price points...")
        print(f"Short SMA: {short_window}, Long SMA: {long_window}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Run the backtest
        for i in range(long_window, len(prices)):
            current_price = prices[i]
            past_prices = prices[:i].tolist()  # All prices up to current point
            
            # Get trading signal
            signal = sma_trading_decision(past_prices, current_price, short_window, long_window)
            
            # Execute trade
            self.execute_trade(current_price, signal, timestamps[i], index=i)
            
            # Track portfolio value and signal
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            self.signals.append(signal)
            
            # Calculate returns
            if i == long_window:
                self.returns.append(0)  # First return is 0
            else:
                prev_value = self.portfolio_values[-2]
                if prev_value > 0:
                    return_pct = (portfolio_value - prev_value) / prev_value
                    self.returns.append(return_pct)
                else:
                    self.returns.append(0)
        
        return self.generate_results(prices, timestamps)
    
    def generate_results(self, prices: np.array, timestamps) -> Dict:
        """Generate comprehensive backtest results."""
        final_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics
        returns_array = np.array(self.returns)
        
        # Remove any infinite or NaN values
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            max_drawdown = self.calculate_max_drawdown()
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Calculate Sortino ratio (risk-adjusted return using downside deviation)
            sortino_ratio = self.calculate_sortino_ratio(returns_array)
            
            # Calculate Calmar ratio (risk-adjusted return using max drawdown)
            calmar_ratio = self.calculate_calmar_ratio(returns_array, max_drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            volatility = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        # Buy and hold benchmark
        buy_hold_return = (prices[-1] - prices[0]) / prices[0]
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return_pct': buy_hold_return * 100,
            'alpha': (total_return - buy_hold_return) * 100,
            'num_trades': len(self.trades),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'volatility': volatility * 100,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'signals': self.signals
        }
        
        return results
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.portfolio_values:
            return 0
        
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown) if len(drawdown) > 0 else 0
    
    def calculate_sortino_ratio(self, returns_array: np.ndarray) -> float:
        """
        Calculate Sortino ratio - risk-adjusted return using downside deviation.
        Only considers negative returns (downside risk) in the denominator.
        """
        if len(returns_array) == 0:
            return 0
        
        mean_return = np.mean(returns_array)
        negative_returns = returns_array[returns_array < 0]
        
        if len(negative_returns) == 0:
            # No downside volatility, return high value but not infinite
            return mean_return * np.sqrt(252) if mean_return > 0 else 0
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0
        
        sortino_ratio = mean_return / downside_deviation * np.sqrt(252)
        return sortino_ratio
    
    def calculate_calmar_ratio(self, returns_array: np.ndarray, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio - annualized return divided by maximum drawdown.
        Measures return per unit of downside risk.
        """
        if len(returns_array) == 0:
            return 0
        
        # Annualized return (assuming daily returns)
        annualized_return = np.mean(returns_array) * 252
        
        # Use absolute value of max drawdown to avoid division by negative
        abs_max_drawdown = abs(max_drawdown)
        
        if abs_max_drawdown == 0:
            # No drawdown, return high value but not infinite
            return annualized_return if annualized_return > 0 else 0
        
        calmar_ratio = annualized_return / abs_max_drawdown
        return calmar_ratio
    
    def print_results(self, results: Dict):
        """Print formatted backtest results."""
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        print(f"Initial Capital:      ${results['initial_capital']:,.2f}")
        print(f"Final Value:          ${results['final_value']:,.2f}")
        print(f"Total Return:         {results['total_return_pct']:.2f}%")
        print(f"Buy & Hold Return:    {results['buy_hold_return_pct']:.2f}%")
        print(f"Alpha (vs B&H):       {results['alpha']:+.2f}%")
        print(f"Number of Trades:     {results['num_trades']}")
        print(f"Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:        {results['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:         {results['calmar_ratio']:.2f}")
        print(f"Max Drawdown:         {results['max_drawdown']:.2f}%")
        print(f"Volatility:           {results['volatility']:.2f}%")
        
        if results['num_trades'] > 0:
            print(f"\nFirst Trade:          {results['trades'][0]['action']} at ${results['trades'][0]['price']:.2f}")
            print(f"Last Trade:           {results['trades'][-1]['action']} at ${results['trades'][-1]['price']:.2f}")
    
    def plot_results(self, df: pd.DataFrame, results: Dict):
        """Plot backtest results."""
        if df.empty or not results:
            return
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Price and trading signals
        prices = df['close'].values
        timestamps = range(len(prices))
        
        ax1.plot(timestamps, prices, label='Price', linewidth=1, alpha=0.7)
        
        # Mark buy and sell trades
        buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in results['trades'] if t['action'] == 'SELL']
        
        if buy_trades:
            buy_indices = [t['index'] for t in buy_trades if t.get('index') is not None]
            buy_prices = [t['price'] for t in buy_trades if t.get('index') is not None]
            if buy_indices:
                ax1.scatter(buy_indices, buy_prices, color='green', marker='^', s=50, label='Buy', zorder=5)
        
        if sell_trades:
            sell_indices = [t['index'] for t in sell_trades if t.get('index') is not None]
            sell_prices = [t['price'] for t in sell_trades if t.get('index') is not None]
            if sell_indices:
                ax1.scatter(sell_indices, sell_prices, color='red', marker='v', s=50, label='Sell', zorder=5)
        
        ax1.set_title('Price and Trading Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value
        portfolio_timestamps = range(len(results['portfolio_values']))
        ax2.plot(portfolio_timestamps, results['portfolio_values'], label='Portfolio Value', color='blue')
        ax2.axhline(y=results['initial_capital'], color='red', linestyle='--', label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        if results['portfolio_values']:
            portfolio_values = np.array(results['portfolio_values'])
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak * 100
            
            ax3.fill_between(portfolio_timestamps, drawdown, 0, alpha=0.3, color='red')
            ax3.plot(portfolio_timestamps, drawdown, color='red')
            ax3.set_title('Drawdown (%)')
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_xlabel('Time')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_sma_backtest(csv_filename: str = "price_data.csv", 
                     short_window: int = 10, long_window: int = 50,
                     initial_capital: float = 10000.0, commission: float = 0.001):
    """
    Convenience function to run a complete SMA backtest.
    
    Args:
        csv_filename: Path to CSV file with price data
        short_window: Short SMA period
        long_window: Long SMA period
        initial_capital: Starting capital
        commission: Commission rate per trade
    """
    # Create backtester
    backtester = SMABacktester(initial_capital=initial_capital, commission=commission)
    
    # Load data
    df = backtester.load_data_from_csv(csv_filename)
    if df.empty:
        return None
    
    # Run backtest
    results = backtester.run_backtest(df, short_window=short_window, long_window=long_window)
    
    if results:
        # Print results
        backtester.print_results(results)
        
        # Plot results
        backtester.plot_results(df, results)
        
        return results
    else:
        print("Backtest failed to run.")
        return None


# Example usage and parameter optimization
def optimize_sma_parameters(csv_filename: str = "price_data.csv", 
                           short_range: Tuple[int, int] = (5, 20),
                           long_range: Tuple[int, int] = (20, 100),
                           step: int = 5):
    """
    Optimize SMA parameters by testing different combinations.
    
    Args:
        csv_filename: Path to CSV file
        short_range: Range for short SMA (min, max)
        long_range: Range for long SMA (min, max)  
        step: Step size for parameter search
    """
    backtester = SMABacktester()
    df = backtester.load_data_from_csv(csv_filename)
    
    if df.empty:
        return None
    
    best_return = -float('inf')
    best_params = None
    results_list = []
    
    print("Optimizing SMA parameters...")
    
    for short in range(short_range[0], short_range[1] + 1, step):
        for long in range(long_range[0], long_range[1] + 1, step):
            if short >= long:
                continue
                
            backtester.reset()
            results = backtester.run_backtest(df, short_window=short, long_window=long)
            
            if results and results['total_return'] > best_return:
                best_return = results['total_return']
                best_params = (short, long)
            
            results_list.append({
                'short': short,
                'long': long,
                'return': results['total_return'] if results else -1,
                'sharpe': results['sharpe_ratio'] if results else 0
            })
    
    print(f"\nBest parameters: Short={best_params[0]}, Long={best_params[1]}")
    print(f"Best return: {best_return*100:.2f}%")
    
    return results_list, best_params


if __name__ == "__main__":
    # Example usage
    print("SMA Strategy Backtesting Framework")
    print("-" * 40)
    
    # Run backtest with default parameters
    results = run_sma_backtest(
        csv_filename="btc_5m_data.csv",  # Use your specific CSV file
        short_window=10,
        long_window=30,
        initial_capital=10000.0,
        commission=0.001
    )
    
    # Uncomment to run parameter optimization
    # print("\nRunning parameter optimization...")
    # optimization_results, best_params = optimize_sma_parameters("price_data.csv")