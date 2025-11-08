"""
Multi-Cryptocurrency Portfolio Optimizer
Fetches data for multiple cryptos and optimizes SMA parameters for each.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
import time

from prices import fetch_kraken_ohlc, save_prices_to_csv, load_prices_from_csv
from backtest_sma import SMABacktester
from trading_strategy import sma_trading_decision


@dataclass
class CryptoResult:
    """Results for a single cryptocurrency."""
    symbol: str
    best_params: Tuple[int, int]
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    data_points: int


class MultiCryptoOptimizer:
    """
    Multi-cryptocurrency portfolio optimizer.
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, 
                 data_dir: str = "crypto_data"):
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Extended cryptocurrency mapping for Kraken API
        # Format: 'DISPLAY_NAME': 'KRAKEN_PAIR'
        self.crypto_pairs = {
            # Major cryptocurrencies (high liquidity)
            'BTC': 'XBTUSD',
            'ETH': 'ETHUSD', 
            'XRP': 'XRPUSD',
            'ADA': 'ADAUSD',
            'DOT': 'DOTUSD',
            'SOL': 'SOLUSD',
            'AVAX': 'AVAXUSD',
            'LINK': 'LINKUSD',
            'UNI': 'UNIUSD',
            'LTC': 'LTCUSD',
            'DOGE': 'DOGEUSD',
            'SHIB': 'SHIBUSD',
            'TRX': 'TRXUSD',
            'TON': 'TONUSD',
            'ICP': 'ICPUSD',
            'FIL': 'FILUSD',
            'AAVE': 'AAVEUSD',
            'CRV': 'CRVUSD',
            'ARB': 'ARBUSD',
            'PEPE': 'PEPEUSD',
            'FLOKI': 'FLOKIUSD',
            'BONK': 'BONKUSD',
            'XLM': 'XLMUSD',
            'ZEC': 'ZECUSD',
            'NEAR': 'NEARUSD',
            'FET': 'FETUSD',
            'HBAR': 'HBARUSD',
            'APT': 'APTUSD',
            'SEI': 'SEIUSD',
            'ENA': 'ENAUSD',
            'ONDO': 'ONDOUSD',
            'TAO': 'TAOUSD',
            
            # Note: Many tokens from your list are not available on Kraken
            # Kraken focuses on major, established cryptocurrencies
            # For tokens like PENDLE, OMNI, LINEA, etc., you'd need a different exchange API
        }
        
        # Prioritize cryptocurrencies by market cap and liquidity
        self.priority_tiers = {
            'tier_1': ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK'],  # Top tier
            'tier_2': ['UNI', 'LTC', 'AAVE', 'FIL', 'NEAR', 'ICP', 'TON', 'ARB'],  # High cap
            'tier_3': ['DOGE', 'SHIB', 'TRX', 'CRV', 'FET', 'APT', 'HBAR', 'SEI'], # Mid cap  
            'tier_4': ['PEPE', 'FLOKI', 'BONK', 'XLM', 'ZEC', 'ENA', 'ONDO', 'TAO'] # Speculative
        }
        
        self.results = {}
    
    def get_data_filename(self, symbol: str, interval: int = 5) -> str:
        """
        Generate the complete file path for cryptocurrency data.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Time interval in minutes
            
        Returns:
            Complete file path for the data file
        """
        filename = f"{symbol.lower()}_{interval}m_data.csv"
        return os.path.join(self.data_dir, filename)
    
    def get_available_cryptos(self) -> List[str]:
        """Get list of all available cryptocurrencies."""
        return list(self.crypto_pairs.keys())
    
    def select_top_cryptos(self, count: int = 20, include_tiers: List[str] = None) -> List[str]:
        """
        Select top cryptocurrencies for optimization based on tiers.
        
        Args:
            count: Maximum number of cryptos to select
            include_tiers: List of tiers to include (default: ['tier_1', 'tier_2'])
        
        Returns:
            List of selected cryptocurrency symbols
        """
        if include_tiers is None:
            include_tiers = ['tier_1', 'tier_2']
        
        selected = []
        
        # Add cryptos from specified tiers in priority order
        for tier in include_tiers:
            if tier in self.priority_tiers:
                for crypto in self.priority_tiers[tier]:
                    if crypto in self.crypto_pairs and len(selected) < count:
                        selected.append(crypto)
        
        # Fill remaining slots with tier_3 if needed
        if len(selected) < count and 'tier_3' not in include_tiers:
            for crypto in self.priority_tiers['tier_3']:
                if crypto in self.crypto_pairs and crypto not in selected and len(selected) < count:
                    selected.append(crypto)
        
        print(f"🎯 Selected {len(selected)} cryptocurrencies for optimization:")
        for i, crypto in enumerate(selected, 1):
            tier = self.get_crypto_tier(crypto)
            print(f"   {i:2d}. {crypto:<6} ({tier})")
        
        return selected
    
    def get_crypto_tier(self, symbol: str) -> str:
        """Get the tier classification for a cryptocurrency."""
        for tier, cryptos in self.priority_tiers.items():
            if symbol in cryptos:
                return tier
        return 'unknown'
    
    def quick_data_check(self, symbols: List[str]) -> List[str]:
        """
        Quickly check which cryptocurrencies have sufficient data.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            List of symbols with adequate data
        """
        valid_symbols = []
        
        print(f"🔍 Checking data availability for {len(symbols)} cryptocurrencies...")
        
        for symbol in symbols:
            if symbol not in self.crypto_pairs:
                continue
                
            filename = self.get_data_filename(symbol, 5)
            try:
                backtester = SMABacktester(self.initial_capital, self.commission)
                df = backtester.load_data_from_csv(filename)
                
                if not df.empty and len(df) >= 200:  # Need sufficient data for optimization
                    valid_symbols.append(symbol)
                    print(f"✅ {symbol}: {len(df)} data points")
                else:
                    print(f"⚠️ {symbol}: Insufficient data ({len(df) if not df.empty else 0} points)")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error loading data - {e}")
        
        print(f"\n📊 {len(valid_symbols)}/{len(symbols)} cryptocurrencies have sufficient data")
        return valid_symbols
        
    def fetch_crypto_data(self, symbols: List[str] = None, interval: int = 5):
        """
        Fetch price data for specified cryptocurrencies.
        
        Args:
            symbols: List of crypto symbols to fetch (default: all)
            interval: Time interval in minutes
        """
        if symbols is None:
            symbols = list(self.crypto_pairs.keys())
            
        print(f"📊 Fetching data for {len(symbols)} cryptocurrencies...")
        
        for symbol in symbols:
            if symbol not in self.crypto_pairs:
                print(f"⚠️ Unknown symbol: {symbol}")
                continue
                
            kraken_pair = self.crypto_pairs[symbol]
            filename = self.get_data_filename(symbol, interval)
            
            print(f"Fetching {symbol} ({kraken_pair})...")
            
            try:
                ohlc_data = fetch_kraken_ohlc(kraken_pair, interval)
                
                if ohlc_data and len(ohlc_data) > 100:  # Ensure enough data
                    save_prices_to_csv(ohlc_data, filename, append=False)
                    print(f"✅ {symbol}: {len(ohlc_data)} records saved to {filename}")
                else:
                    print(f"❌ {symbol}: Insufficient data ({len(ohlc_data) if ohlc_data else 0} records)")
                    
            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")
                
    def optimize_single_crypto(self, symbol: str, interval: int = 5) -> CryptoResult:
        """
        Optimize SMA parameters for a single cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            interval: Time interval in minutes
            
        Returns:
            CryptoResult with optimization results
        """
        filename = self.get_data_filename(symbol, interval)
        
        print(f"🔍 Optimizing {symbol}...")
        
        # Load data
        backtester = SMABacktester(self.initial_capital, self.commission)
        df = backtester.load_data_from_csv(filename)
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
            
        # Parameter ranges for testing
        short_range = [5, 8, 10, 12, 15, 18, 20, 25]
        long_range = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
        
        best_score = -float('inf')
        best_params = None
        best_results = None
        
        total_combinations = sum(1 for s in short_range for l in long_range if s < l)
        current = 0
        
        for short in short_range:
            for long in long_range:
                if short >= long:
                    continue
                    
                current += 1
                
                backtester.reset()
                try:
                    results = backtester.run_backtest(df, short_window=short, long_window=long)
                    
                    if results:
                        # Multi-objective scoring: return + risk-adjusted measures
                        return_score = results['total_return_pct']
                        sharpe_bonus = max(0, results['sharpe_ratio']) * 2
                        drawdown_penalty = results['max_drawdown'] * 0.3
                        
                        # Prefer moderate number of trades (not too few, not too many)
                        trade_penalty = 0
                        if results['num_trades'] < 5:
                            trade_penalty = -1
                        elif results['num_trades'] > 50:
                            trade_penalty = -2
                            
                        score = return_score + sharpe_bonus - drawdown_penalty + trade_penalty
                        
                        if score > best_score:
                            best_score = score
                            best_params = (short, long)
                            best_results = results
                            
                except Exception as e:
                    continue
        
        if best_params:
            crypto_result = CryptoResult(
                symbol=symbol,
                best_params=best_params,
                return_pct=best_results['total_return_pct'],
                sharpe_ratio=best_results['sharpe_ratio'],
                max_drawdown=best_results['max_drawdown'],
                num_trades=best_results['num_trades'],
                data_points=len(df)
            )
            
            print(f"✅ {symbol}: SMA({best_params[0]},{best_params[1]}) = {crypto_result.return_pct:+.2f}%")
            return crypto_result
        else:
            print(f"❌ {symbol}: Optimization failed")
            return None
            
    def optimize_all_cryptos(self, symbols: List[str] = None, interval: int = 5):
        """
        Optimize parameters for all specified cryptocurrencies.
        
        Args:
            symbols: List of symbols to optimize (default: all)
            interval: Time interval in minutes
        """
        if symbols is None:
            symbols = list(self.crypto_pairs.keys())
            
        print(f"🚀 Optimizing {len(symbols)} cryptocurrencies...")
        
        self.results = {}
        
        for symbol in symbols:
            result = self.optimize_single_crypto(symbol, interval)
            if result:
                self.results[symbol] = result
                
    def create_portfolio_weights(self, min_return: float = -5.0) -> Dict[str, float]:
        """
        Create portfolio weights based on performance.
        
        Args:
            min_return: Minimum return threshold to include in portfolio
            
        Returns:
            Dictionary of portfolio weights
        """
        if not self.results:
            return {}
            
        # Filter out poor performers
        good_performers = {
            symbol: result for symbol, result in self.results.items()
            if result.return_pct > min_return
        }
        
        if not good_performers:
            print("⚠️ No cryptocurrencies meet minimum return threshold")
            return {}
            
        # Calculate weights based on performance
        total_score = 0
        scores = {}
        
        for symbol, result in good_performers.items():
            # Score based on return and risk-adjusted measures
            score = max(0, result.return_pct + result.sharpe_ratio * 2 - result.max_drawdown * 0.2)
            scores[symbol] = score
            total_score += score
            
        # Normalize to portfolio weights
        weights = {}
        if total_score > 0:
            for symbol, score in scores.items():
                weights[symbol] = score / total_score
        else:
            # Equal weights if no positive scores
            weight = 1.0 / len(good_performers)
            for symbol in good_performers:
                weights[symbol] = weight
                
        return weights
        
    def calculate_portfolio_performance(self, weights: Dict[str, float]) -> Dict:
        """
        Calculate expected portfolio performance.
        
        Args:
            weights: Portfolio weights for each cryptocurrency
            
        Returns:
            Dictionary with portfolio metrics
        """
        if not weights or not self.results:
            return {}
            
        weighted_return = sum(weights[symbol] * self.results[symbol].return_pct 
                            for symbol in weights)
        
        weighted_sharpe = sum(weights[symbol] * self.results[symbol].sharpe_ratio 
                            for symbol in weights)
        
        weighted_drawdown = sum(weights[symbol] * self.results[symbol].max_drawdown 
                              for symbol in weights)
        
        total_trades = sum(self.results[symbol].num_trades for symbol in weights)
        
        return {
            'expected_return_pct': weighted_return,
            'weighted_sharpe': weighted_sharpe,
            'weighted_max_drawdown': weighted_drawdown,
            'total_trades': total_trades,
            'num_assets': len(weights)
        }
        
    def generate_report(self) -> str:
        """Generate comprehensive optimization report."""
        if not self.results:
            return "No optimization results available."
            
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-CRYPTOCURRENCY OPTIMIZATION REPORT")
        lines.append("=" * 80)
        
        # Individual results
        lines.append(f"\n📊 INDIVIDUAL CRYPTOCURRENCY RESULTS:")
        lines.append("-" * 70)
        lines.append(f"{'Symbol':<6} {'Parameters':<12} {'Return':<8} {'Sharpe':<7} {'MaxDD':<7} {'Trades':<7}")
        lines.append("-" * 70)
        
        # Sort by return
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].return_pct, 
                              reverse=True)
        
        for symbol, result in sorted_results:
            lines.append(f"{symbol:<6} "
                        f"SMA({result.best_params[0]},{result.best_params[1]})   "
                        f"{result.return_pct:+6.2f}% "
                        f"{result.sharpe_ratio:6.2f} "
                        f"{result.max_drawdown:6.2f}% "
                        f"{result.num_trades:6d}")
        
        # Portfolio allocation
        weights = self.create_portfolio_weights()
        if weights:
            lines.append(f"\n💼 RECOMMENDED PORTFOLIO ALLOCATION:")
            lines.append("-" * 40)
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"{symbol}: {weight:6.1%}")
            
            # Portfolio performance
            portfolio_perf = self.calculate_portfolio_performance(weights)
            lines.append(f"\n📈 EXPECTED PORTFOLIO PERFORMANCE:")
            lines.append("-" * 40)
            lines.append(f"Expected Return:    {portfolio_perf['expected_return_pct']:+7.2f}%")
            lines.append(f"Weighted Sharpe:    {portfolio_perf['weighted_sharpe']:7.2f}")
            lines.append(f"Weighted Max DD:    {portfolio_perf['weighted_max_drawdown']:7.2f}%")
            lines.append(f"Total Trades:       {portfolio_perf['total_trades']:7d}")
            lines.append(f"Diversification:    {portfolio_perf['num_assets']:7d} assets")
        
        return "\n".join(lines)
        
    def save_results(self, filename: str = "multi_crypto_results.json"):
        """Save optimization results to JSON file."""
        if not self.results:
            return
            
        # Convert to serializable format
        serializable_results = {}
        for symbol, result in self.results.items():
            serializable_results[symbol] = {
                'best_params': result.best_params,
                'return_pct': result.return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'num_trades': result.num_trades,
                'data_points': result.data_points
            }
            
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"💾 Results saved to {filename}")


def main():
    """Main function to run comprehensive multi-crypto optimization."""
    print("🚀 COMPREHENSIVE MULTI-CRYPTOCURRENCY OPTIMIZER")
    print("=" * 70)
    print("🎯 Goal: Find most profitable cryptocurrencies for next month")
    print("=" * 70)
    
    optimizer = MultiCryptoOptimizer(initial_capital=10000, commission=0.001, 
                                     data_dir="crypto_data")
    
    print(f"\n📋 Available cryptocurrencies: {len(optimizer.get_available_cryptos())}")
    print("   Tier 1 (Top): " + ", ".join(optimizer.priority_tiers['tier_1']))
    print("   Tier 2 (High): " + ", ".join(optimizer.priority_tiers['tier_2']))
    print("   Tier 3 (Mid): " + ", ".join(optimizer.priority_tiers['tier_3']))
    print("   Tier 4 (Spec): " + ", ".join(optimizer.priority_tiers['tier_4']))
    
    # Step 1: Select top cryptocurrencies for testing
    print(f"\n🎯 Step 1: Selecting cryptocurrencies for optimization")
    
    # Test different strategies
    strategies = [
        ("Conservative (Top 15)", 15, ['tier_1', 'tier_2']),
        ("Aggressive (Top 25)", 25, ['tier_1', 'tier_2', 'tier_3']),
        ("Full Spectrum (Top 30)", 30, ['tier_1', 'tier_2', 'tier_3', 'tier_4'])
    ]
    
    print("Available strategies:")
    for i, (name, count, tiers) in enumerate(strategies, 1):
        print(f"   {i}. {name} - {count} cryptos from {tiers}")
    
    # Use aggressive strategy by default (can be modified)
    selected_strategy = 2  # Aggressive approach
    strategy_name, crypto_count, include_tiers = strategies[selected_strategy]
    
    print(f"\n🎲 Using: {strategy_name}")
    test_cryptos = optimizer.select_top_cryptos(crypto_count, include_tiers)
    
    # Step 2: Check data availability
    print(f"\n� Step 2: Checking data availability")
    valid_cryptos = optimizer.quick_data_check(test_cryptos)
    
    if not valid_cryptos:
        print("❌ No valid cryptocurrencies found. Fetching fresh data...")
        
        # Fetch data for top cryptos
        priority_cryptos = optimizer.priority_tiers['tier_1'][:8]  # Top 8 as backup
        print(f"\n📊 Fetching data for priority cryptocurrencies: {priority_cryptos}")
        optimizer.fetch_crypto_data(priority_cryptos, interval=5)
        
        # Wait to avoid rate limits
        print("⏳ Waiting 30 seconds to avoid API rate limits...")
        time.sleep(30)
        
        # Recheck
        valid_cryptos = optimizer.quick_data_check(priority_cryptos)
    
    if not valid_cryptos:
        print("❌ Unable to get sufficient data. Please check your data files.")
        return None
    
    print(f"\n✅ Proceeding with {len(valid_cryptos)} cryptocurrencies")
    
    # Step 3: Run optimization
    print(f"\n🔍 Step 3: Running optimization (this may take several minutes)")
    print(f"Expected runtime: ~{len(valid_cryptos) * 2} minutes for {len(valid_cryptos)} cryptos")
    
    start_time = time.time()
    optimizer.optimize_all_cryptos(valid_cryptos, interval=5)
    end_time = time.time()
    
    print(f"⏰ Optimization completed in {(end_time - start_time)/60:.1f} minutes")
    
    # Step 4: Analyze results and find most profitable
    if not optimizer.results:
        print("❌ No optimization results available")
        return None
    
    print(f"\n� Step 4: Analyzing results for most profitable cryptocurrencies")
    
    # Sort by profitability
    sorted_results = sorted(optimizer.results.items(), 
                           key=lambda x: x[1].return_pct, 
                           reverse=True)
    
    # Filter for profitable ones
    profitable_cryptos = [(symbol, result) for symbol, result in sorted_results 
                         if result.return_pct > 0]
    
    print(f"\n🏆 MOST PROFITABLE CRYPTOCURRENCIES FOR NEXT MONTH:")
    print("=" * 80)
    print(f"{'Rank':<4} {'Crypto':<6} {'Tier':<8} {'Params':<12} {'Return':<9} {'Sharpe':<7} {'MaxDD':<8} {'Risk':<6}")
    print("-" * 80)
    
    for i, (symbol, result) in enumerate(profitable_cryptos[:15], 1):  # Top 15
        tier = optimizer.get_crypto_tier(symbol)
        risk_level = "Low" if result.max_drawdown > -3 else "Med" if result.max_drawdown > -7 else "High"
        
        print(f"{i:3d}  "
              f"{symbol:<6} "
              f"{tier:<8} "
              f"SMA({result.best_params[0]:2d},{result.best_params[1]:2d})   "
              f"{result.return_pct:+7.2f}% "
              f"{result.sharpe_ratio:6.2f} "
              f"{result.max_drawdown:7.2f}% "
              f"{risk_level:<6}")
    
    # Step 5: Portfolio recommendations
    print(f"\n💼 PORTFOLIO RECOMMENDATIONS:")
    
    if len(profitable_cryptos) >= 3:
        # Top 3 portfolio
        top_3 = profitable_cryptos[:3]
        top_3_return = sum(result.return_pct for _, result in top_3) / 3
        
        print(f"\n🥇 HIGH-RETURN PORTFOLIO (Top 3):")
        for symbol, result in top_3:
            print(f"   • {symbol}: SMA({result.best_params[0]},{result.best_params[1]}) - {result.return_pct:+.2f}%")
        print(f"   Expected Portfolio Return: {top_3_return:+.2f}%")
        
        # Balanced portfolio (top 5-8)
        if len(profitable_cryptos) >= 5:
            balanced = profitable_cryptos[:min(8, len(profitable_cryptos))]
            balanced_return = sum(result.return_pct for _, result in balanced) / len(balanced)
            
            print(f"\n⚖️ BALANCED PORTFOLIO (Top {len(balanced)}):")
            for symbol, result in balanced:
                tier = optimizer.get_crypto_tier(symbol)
                print(f"   • {symbol} ({tier}): {result.return_pct:+.2f}%")
            print(f"   Expected Portfolio Return: {balanced_return:+.2f}%")
    
    # Step 6: Save results
    optimizer.save_results("most_profitable_cryptos.json")
    
    # Step 7: Implementation guide
    if profitable_cryptos:
        best_crypto, best_result = profitable_cryptos[0]
        
        print(f"\n🎯 IMPLEMENTATION GUIDE:")
        print(f"   🥇 Best Single Crypto: {best_crypto}")
        print(f"      • Parameters: SMA({best_result.best_params[0]}, {best_result.best_params[1]})")
        print(f"      • Expected Monthly Return: {best_result.return_pct:+.2f}%")
        print(f"      • Risk Level: {abs(best_result.max_drawdown):.2f}% max drawdown")
        print(f"      • Trade Frequency: {best_result.num_trades} trades expected")
        
        print(f"\n📈 NEXT MONTH PROJECTIONS:")
        capitals = [1000, 5000, 10000, 50000]
        for capital in capitals:
            profit = capital * best_result.return_pct / 100
            print(f"      ${capital:,} → ${profit:+,.0f} profit")
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()