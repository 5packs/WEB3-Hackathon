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
    sortino_ratio: float
    calmar_ratio: float
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
        
        # Set output directory to main project output folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        print(f"Selected {len(selected)} cryptocurrencies for optimization:")
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
        
        print(f"Checking data availability for {len(symbols)} cryptocurrencies...")
        
        for symbol in symbols:
            if symbol not in self.crypto_pairs:
                continue
                
            filename = self.get_data_filename(symbol, 5)
            try:
                backtester = SMABacktester(self.initial_capital, self.commission)
                df = backtester.load_data_from_csv(filename)
                
                if not df.empty and len(df) >= 200:  # Need sufficient data for optimization
                    valid_symbols.append(symbol)
                    print(f"[OK] {symbol}: {len(df)} data points")
                else:
                    print(f"[WARNING] {symbol}: Insufficient data ({len(df) if not df.empty else 0} points)")
                    
            except Exception as e:
                print(f"[ERROR] {symbol}: Error loading data - {e}")
        
        print(f"\n[DATA] {len(valid_symbols)}/{len(symbols)} cryptocurrencies have sufficient data")
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
            
        print(f"[DATA] Fetching data for {len(symbols)} cryptocurrencies...")
        
        for symbol in symbols:
            if symbol not in self.crypto_pairs:
                print(f"[WARNING] Unknown symbol: {symbol}")
                continue
                
            kraken_pair = self.crypto_pairs[symbol]
            filename = self.get_data_filename(symbol, interval)
            
            print(f"Fetching {symbol} ({kraken_pair})...")
            
            try:
                # Fetch all available data from Kraken API (up to 721 entries for backtesting)
                ohlc_data = fetch_kraken_ohlc(kraken_pair, interval)
                
                if ohlc_data and len(ohlc_data) > 100:  # Ensure enough data for SMA
                    save_prices_to_csv(ohlc_data, filename, append=False)
                    print(f"[OK] {symbol}: {len(ohlc_data)} records saved to {filename} (full dataset for backtesting)")
                else:
                    print(f"[ERROR] {symbol}: Insufficient data ({len(ohlc_data) if ohlc_data else 0} records)")
                    
            except Exception as e:
                print(f"[ERROR] Error fetching {symbol}: {e}")
                
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
        
        print(f"Optimizing {symbol}...")
        
        # Load data
        backtester = SMABacktester(self.initial_capital, self.commission)
        df = backtester.load_data_from_csv(filename)
        
        if df.empty:
            print(f"[ERROR] No data for {symbol}")
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
                sortino_ratio=best_results['sortino_ratio'],
                calmar_ratio=best_results['calmar_ratio'],
                max_drawdown=best_results['max_drawdown'],
                num_trades=best_results['num_trades'],
                data_points=len(df)
            )
            
            print(f"[OK] {symbol}: SMA({best_params[0]},{best_params[1]}) = {crypto_result.return_pct:+.2f}%")
            return crypto_result
        else:
            print(f"[ERROR] {symbol}: Optimization failed")
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
            
        print(f"Optimizing {len(symbols)} cryptocurrencies...")
        
        self.results = {}
        
        for symbol in symbols:
            result = self.optimize_single_crypto(symbol, interval)
            if result:
                self.results[symbol] = result
    
    def calculate_composite_risk_score(self, result: CryptoResult) -> float:
        """
        Calculate composite risk-adjusted score using your specified weighting:
        0.4 * Sortino + 0.3 * Calmar + 0.3 * Sharpe
        
        Args:
            result: CryptoResult containing the risk metrics
            
        Returns:
            Composite risk score
        """
        # Handle edge cases where ratios might be NaN or infinite
        sortino = result.sortino_ratio if np.isfinite(result.sortino_ratio) else 0
        calmar = result.calmar_ratio if np.isfinite(result.calmar_ratio) else 0
        sharpe = result.sharpe_ratio if np.isfinite(result.sharpe_ratio) else 0
        
        # Normalize extreme values to prevent one metric from dominating
        # Cap individual ratios at reasonable bounds
        sortino = max(-10, min(10, sortino))
        calmar = max(-10, min(10, calmar))
        sharpe = max(-10, min(10, sharpe))
        
        composite_score = 0.4 * sortino + 0.3 * calmar + 0.3 * sharpe
        return composite_score
                
    def create_portfolio_weights(self, min_return: float = -5.0, return_weight: float = 0.5, 
                                risk_weight: float = 0.5) -> Dict[str, float]:
        """
        Create portfolio weights using multi-objective optimization that balances
        returns and composite risk-adjusted metrics.
        
        Args:
            min_return: Minimum return threshold to include in portfolio
            return_weight: Weight given to returns (default 0.5 for equal balance)
            risk_weight: Weight given to composite risk score (default 0.5 for equal balance)
            
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
            print("[WARNING] No cryptocurrencies meet minimum return threshold")
            return {}
            
        # Calculate normalized scores for both objectives
        returns = [result.return_pct for result in good_performers.values()]
        risk_scores = [self.calculate_composite_risk_score(result) for result in good_performers.values()]
        
        # Normalize returns and risk scores to [0, 1] range
        if max(returns) > min(returns):
            norm_returns = [(r - min(returns)) / (max(returns) - min(returns)) for r in returns]
        else:
            norm_returns = [1.0] * len(returns)
            
        if max(risk_scores) > min(risk_scores):
            norm_risk_scores = [(r - min(risk_scores)) / (max(risk_scores) - min(risk_scores)) for r in risk_scores]
        else:
            norm_risk_scores = [1.0] * len(risk_scores)
            
        # Calculate multi-objective scores combining both objectives
        total_score = 0
        scores = {}
        
        for i, (symbol, result) in enumerate(good_performers.items()):
            # Multi-objective score: weighted combination of normalized returns and risk scores
            multi_objective_score = return_weight * norm_returns[i] + risk_weight * norm_risk_scores[i]
            
            # Ensure non-negative scores
            score = max(0, multi_objective_score)
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
        
        weighted_sortino = sum(weights[symbol] * self.results[symbol].sortino_ratio 
                             for symbol in weights)
        
        weighted_calmar = sum(weights[symbol] * self.results[symbol].calmar_ratio 
                            for symbol in weights)
        
        weighted_drawdown = sum(weights[symbol] * self.results[symbol].max_drawdown 
                              for symbol in weights)
        
        # Calculate portfolio composite risk score
        portfolio_composite_risk = sum(weights[symbol] * self.calculate_composite_risk_score(self.results[symbol]) 
                                     for symbol in weights)
        
        total_trades = sum(self.results[symbol].num_trades for symbol in weights)
        
        return {
            'expected_return_pct': weighted_return,
            'weighted_sharpe': weighted_sharpe,
            'weighted_sortino': weighted_sortino,
            'weighted_calmar': weighted_calmar,
            'portfolio_composite_risk_score': portfolio_composite_risk,
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
        lines.append(f"\nINDIVIDUAL CRYPTOCURRENCY RESULTS:")
        lines.append("-" * 90)
        lines.append(f"{'Symbol':<6} {'Parameters':<12} {'Return':<8} {'Sharpe':<7} {'Sortino':<7} {'Calmar':<7} {'MaxDD':<7} {'Trades':<7}")
        lines.append("-" * 90)
        
        # Sort by return
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].return_pct, 
                              reverse=True)
        
        for symbol, result in sorted_results:
            lines.append(f"{symbol:<6} "
                        f"SMA({result.best_params[0]},{result.best_params[1]})   "
                        f"{result.return_pct:+6.2f}% "
                        f"{result.sharpe_ratio:6.2f} "
                        f"{result.sortino_ratio:6.2f} "
                        f"{result.calmar_ratio:6.2f} "
                        f"{result.max_drawdown:6.2f}% "
                        f"{result.num_trades:6d}")
        
        # Portfolio allocation
        weights = self.create_portfolio_weights()
        if weights:
            lines.append(f"\nRECOMMENDED PORTFOLIO ALLOCATION:")
            lines.append("-" * 40)
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"{symbol}: {weight:6.1%}")
            
            # Portfolio performance
            portfolio_perf = self.calculate_portfolio_performance(weights)
            lines.append(f"\nEXPECTED PORTFOLIO PERFORMANCE:")
            lines.append("-" * 40)
            lines.append(f"Expected Return:    {portfolio_perf['expected_return_pct']:+7.2f}%")
            lines.append(f"Weighted Sharpe:    {portfolio_perf['weighted_sharpe']:7.2f}")
            lines.append(f"Weighted Sortino:   {portfolio_perf['weighted_sortino']:7.2f}")
            lines.append(f"Weighted Calmar:    {portfolio_perf['weighted_calmar']:7.2f}")
            lines.append(f"Composite Risk:     {portfolio_perf['portfolio_composite_risk_score']:7.2f}")
            lines.append(f"Weighted Max DD:    {portfolio_perf['weighted_max_drawdown']:7.2f}%")
            lines.append(f"Total Trades:       {portfolio_perf['total_trades']:7d}")
            lines.append(f"Diversification:    {portfolio_perf['num_assets']:7d} assets")
        
        return "\n".join(lines)
        
    def save_results(self, filename: str = "multi_crypto_results.json"):
        """Save optimization results to JSON file in output directory."""
        if not self.results:
            return
        
        # Ensure output directory exists and save to it
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
            
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
            
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {filepath}")

    def save_optimal_parameters(self, filename: str = "optimal_sma_parameters.json"):
        """
        Save only the optimal SMA parameters for trading bot usage.
        Creates a clean file with just the short and long window values in output directory.
        """
        if not self.results:
            print("[WARNING] No optimization results available to save parameters")
            return
        
        # Ensure output directory exists and save to it
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
            
        # Create clean parameter structure for trading bot
        optimal_params = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_currencies": len(self.results),
                "optimizer_settings": {
                    "initial_capital": self.initial_capital,
                    "commission": self.commission
                }
            },
            "parameters": {}
        }
        
        # Add parameters for each currency, sorted by profitability
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1].return_pct, 
                               reverse=True)
        
        for symbol, result in sorted_results:
            optimal_params["parameters"][symbol] = {
                "short_window": result.best_params[0],
                "long_window": result.best_params[1],
                "expected_return_pct": round(result.return_pct, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 3),
                "max_drawdown_pct": round(result.max_drawdown, 2),
                "risk_level": self._get_risk_level(result.max_drawdown),
                "tier": self.get_crypto_tier(symbol)
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(optimal_params, f, indent=2)
            
        print(f"Optimal trading parameters saved to {filepath}")
        print(f"   [DATA] {len(optimal_params['parameters'])} currencies with optimal SMA parameters")
        
        # Also create a simplified version for quick bot access
        simple_params = {}
        for symbol, result in sorted_results:
            simple_params[symbol] = {
                "short": result.best_params[0],
                "long": result.best_params[1]
            }
            
        simple_filename = "simple_sma_parameters.json"
        simple_filepath = os.path.join(self.output_dir, simple_filename)
        with open(simple_filepath, 'w') as f:
            json.dump(simple_params, f, indent=2)
            
        print(f"🤖 Simplified parameters for bot saved to {simple_filepath}")
        
    def save_portfolio_allocation(self, initial_investment: float = 49000, filename: str = "portfolio_allocation.json"):
        """
        Generate optimal portfolio allocation with risk-reward optimization.
        
        Args:
            initial_investment: Total amount to invest (default: $49,000)
            filename: Output filename for portfolio allocation
        """
        if not self.results:
            print("[WARNING] No optimization results available to generate portfolio")
            return
            
        # Save to output directory in main project folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_output_dir = os.path.join(project_root, "output")
        os.makedirs(main_output_dir, exist_ok=True)
        filepath = os.path.join(main_output_dir, filename)
        
        # Filter profitable cryptocurrencies (positive returns)
        profitable_results = {
            symbol: result for symbol, result in self.results.items()
            if result.return_pct > 0 and result.sharpe_ratio > 0
        }
        
        if not profitable_results:
            print("[WARNING] No profitable cryptocurrencies found for portfolio allocation")
            return
            
        # Calculate risk-adjusted scores for each cryptocurrency
        scores = {}
        total_score = 0
        
        for symbol, result in profitable_results.items():
            # Risk-adjusted score: Return + Sharpe bonus - Drawdown penalty
            return_score = max(0, result.return_pct)
            sharpe_bonus = max(0, result.sharpe_ratio) * 3  # Weight Sharpe ratio heavily
            drawdown_penalty = abs(result.max_drawdown) * 0.5  # Penalize high drawdown
            
            # Bonus for tier 1 and tier 2 cryptocurrencies (more stable)
            tier = self.get_crypto_tier(symbol)
            tier_bonus = 2 if tier == 'tier_1' else 1 if tier == 'tier_2' else 0
            
            score = return_score + sharpe_bonus - drawdown_penalty + tier_bonus
            scores[symbol] = max(0, score)  # Ensure non-negative scores
            total_score += scores[symbol]
        
        if total_score == 0:
            print("[WARNING] No valid scores for portfolio allocation")
            return
            
        # Calculate allocation percentages and amounts
        allocations = {}
        total_allocated = 0
        
        for symbol, score in scores.items():
            percentage = score / total_score
            allocation_amount = initial_investment * percentage
            
            # Minimum allocation of $500, maximum of 25% of total
            min_allocation = 500
            max_allocation = initial_investment * 0.25
            
            allocation_amount = max(min_allocation, min(allocation_amount, max_allocation))
            
            allocations[symbol] = {
                "allocation_amount": round(allocation_amount, 2),
                "percentage": round(percentage * 100, 2),
                "score": round(score, 2)
            }
            total_allocated += allocation_amount
        
        # Normalize allocation to use full initial investment
        # First, handle case where total exceeds investment (scale down)
        if total_allocated > initial_investment:
            adjustment_factor = initial_investment / total_allocated
            for symbol in allocations:
                allocations[symbol]["allocation_amount"] = round(
                    allocations[symbol]["allocation_amount"] * adjustment_factor, 2
                )
                allocations[symbol]["percentage"] = round(
                    (allocations[symbol]["allocation_amount"] / initial_investment) * 100, 2
                )
            total_allocated = sum(allocations[symbol]["allocation_amount"] for symbol in allocations)
        
        # Second, handle case where total is less than investment (scale up proportionally)
        elif total_allocated < initial_investment:
            remaining_funds = initial_investment - total_allocated
            
            # Calculate total score of allocated cryptocurrencies for proportional distribution
            allocated_score_total = sum(scores[symbol] for symbol in allocations)
            
            # Distribute remaining funds proportionally based on scores
            for symbol in allocations:
                if allocated_score_total > 0:
                    additional_allocation = remaining_funds * (scores[symbol] / allocated_score_total)
                    allocations[symbol]["allocation_amount"] = round(
                        allocations[symbol]["allocation_amount"] + additional_allocation, 2
                    )
                    allocations[symbol]["percentage"] = round(
                        (allocations[symbol]["allocation_amount"] / initial_investment) * 100, 2
                    )
            
            total_allocated = sum(allocations[symbol]["allocation_amount"] for symbol in allocations)
            
            # Final adjustment to ensure exact total (handle rounding errors)
            if total_allocated != initial_investment:
                difference = initial_investment - total_allocated
                # Add difference to the largest allocation
                largest_symbol = max(allocations.keys(), key=lambda x: allocations[x]["allocation_amount"])
                allocations[largest_symbol]["allocation_amount"] = round(
                    allocations[largest_symbol]["allocation_amount"] + difference, 2
                )
                allocations[largest_symbol]["percentage"] = round(
                    (allocations[largest_symbol]["allocation_amount"] / initial_investment) * 100, 2
                )
                total_allocated = initial_investment
        
        # Calculate expected portfolio performance
        expected_portfolio_return = sum(
            (allocations[symbol]["allocation_amount"] / initial_investment) * self.results[symbol].return_pct
            for symbol in allocations
        )
        
        weighted_sharpe = sum(
            (allocations[symbol]["allocation_amount"] / initial_investment) * self.results[symbol].sharpe_ratio
            for symbol in allocations
        )
        
        weighted_drawdown = sum(
            (allocations[symbol]["allocation_amount"] / initial_investment) * self.results[symbol].max_drawdown
            for symbol in allocations
        )
        
        # Create portfolio structure
        portfolio_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "initial_investment": initial_investment,
                "total_allocated": round(total_allocated, 2),
                "remaining_cash": round(initial_investment - total_allocated, 2),
                "num_cryptocurrencies": len(allocations),
                "allocation_strategy": "Risk-Adjusted Return Optimization"
            },
            "portfolio_performance": {
                "expected_monthly_return_pct": round(expected_portfolio_return, 2),
                "expected_monthly_profit": round(initial_investment * expected_portfolio_return / 100, 2),
                "weighted_sharpe_ratio": round(weighted_sharpe, 3),
                "weighted_max_drawdown_pct": round(weighted_drawdown, 2),
                "risk_level": self._get_portfolio_risk_level(weighted_drawdown)
            },
            "allocations": {}
        }
        
        # Sort allocations by amount (largest first)
        sorted_allocations = sorted(allocations.items(), 
                                   key=lambda x: x[1]["allocation_amount"], 
                                   reverse=True)
        
        for symbol, allocation in sorted_allocations:
            result = self.results[symbol]
            portfolio_data["allocations"][symbol] = {
                "allocation_amount": allocation["allocation_amount"],
                "percentage_of_portfolio": allocation["percentage"],
                "expected_monthly_return_pct": result.return_pct,
                "expected_monthly_profit": round(allocation["allocation_amount"] * result.return_pct / 100, 2),
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown,
                "risk_level": self._get_risk_level(result.max_drawdown),
                "tier": self.get_crypto_tier(symbol),
                "optimal_sma_params": {
                    "short_window": result.best_params[0],
                    "long_window": result.best_params[1]
                },
                "score": allocation["score"]
            }
        
        # Save detailed portfolio allocation to file
        with open(filepath, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
            
        # Create and save simple portfolio allocation (just coin amounts)
        simple_allocation = {}
        for symbol, allocation in sorted_allocations:
            simple_allocation[symbol] = allocation["allocation_amount"]
        
        # Add summary info
        simple_portfolio_data = {
            "summary": {
                "total_investment": initial_investment,
                "total_allocated": round(total_allocated, 2),
                "remaining_cash": round(initial_investment - total_allocated, 2),
                "number_of_coins": len(allocations)
            },
            "allocations": simple_allocation
        }
        
        # Save simple allocation file
        simple_filename = filename.replace('.json', '_simple.json')
        simple_filepath = os.path.join(main_output_dir, simple_filename)
        with open(simple_filepath, 'w') as f:
            json.dump(simple_portfolio_data, f, indent=2)
            
        print(f"Portfolio allocation saved to {filepath}")
        print(f"Simple allocation saved to {simple_filepath}")
        print(f"   ${initial_investment:,.0f} allocated across {len(allocations)} cryptocurrencies")
        print(f"   Expected monthly return: {expected_portfolio_return:+.2f}% (${initial_investment * expected_portfolio_return / 100:+,.0f})")
        print(f"   Portfolio Sharpe ratio: {weighted_sharpe:.3f}")
        print(f"   [WARNING] Maximum drawdown: {weighted_drawdown:.2f}%")

        # Also save a minimal simple allocation file (symbol -> allocation_amount) for programmatic use
        try:
            minimal_simple = {symbol: allocation["allocation_amount"] for symbol, allocation in sorted_allocations}
            minimal_path = os.path.join(main_output_dir, "simple_portfolio_allocation.json")
            with open(minimal_path, 'w') as msf:
                json.dump(minimal_simple, msf, indent=2)
            print(f"Minimal simple allocation saved to {minimal_path}")
        except Exception as e:
            print(f"[WARNING] Failed to write minimal simple allocation: {e}")
        
    def _get_portfolio_risk_level(self, weighted_drawdown: float) -> str:
        """Classify portfolio risk level based on weighted maximum drawdown."""
        if weighted_drawdown >= -5:
            return "low"
        elif weighted_drawdown >= -10:
            return "medium"
        elif weighted_drawdown >= -15:
            return "high"
        else:
            return "very_high"
        
    def _get_risk_level(self, max_drawdown: float) -> str:
        """Classify risk level based on maximum drawdown."""
        if max_drawdown >= -3:
            return "low"
        elif max_drawdown >= -7:
            return "medium"
        else:
            return "high"


def main():
    """Main function to run comprehensive multi-crypto optimization."""
    print("COMPREHENSIVE MULTI-CRYPTOCURRENCY OPTIMIZER")
    print("=" * 70)
    print("Goal: Find most profitable cryptocurrencies for next month")
    print("=" * 70)
    
    optimizer = MultiCryptoOptimizer(initial_capital=10000, commission=0.001, 
                                     data_dir="crypto_data")
    
    print(f"\n[DATA] Available cryptocurrencies: {len(optimizer.get_available_cryptos())}")
    print("   Tier 1 (Top): " + ", ".join(optimizer.priority_tiers['tier_1']))
    print("   Tier 2 (High): " + ", ".join(optimizer.priority_tiers['tier_2']))
    print("   Tier 3 (Mid): " + ", ".join(optimizer.priority_tiers['tier_3']))
    print("   Tier 4 (Spec): " + ", ".join(optimizer.priority_tiers['tier_4']))
    
    # Step 1: Select top cryptocurrencies for testing
    print(f"\nStep 1: Selecting cryptocurrencies for optimization")
    
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
    selected_strategy = 1
    strategy_name, crypto_count, include_tiers = strategies[selected_strategy]
    
    print(f"\nUsing: {strategy_name}")
    test_cryptos = optimizer.select_top_cryptos(crypto_count, include_tiers)
    
    # Step 2: Always fetch fresh data for all selected cryptocurrencies
    print(f"\n[DATA] Step 2: Fetching fresh data for selected cryptocurrencies")
    print(f"Fetching latest data for {len(test_cryptos)} cryptocurrencies...")
    optimizer.fetch_crypto_data(test_cryptos, interval=5)
    
    # Check data availability after fresh fetch
    valid_cryptos = optimizer.quick_data_check(test_cryptos)
    
    if not valid_cryptos:
        print("[ERROR] No valid cryptocurrencies found. Fetching fresh data...")
        
        # Fetch data for ALL available cryptocurrencies as backup
        all_cryptos = list(optimizer.crypto_pairs.keys())
        print(f"\n[DATA] Fetching fresh data for ALL {len(all_cryptos)} available cryptocurrencies...")
        optimizer.fetch_crypto_data(all_cryptos, interval=5)
        
        # Recheck with all available cryptocurrencies
        valid_cryptos = optimizer.quick_data_check(all_cryptos)
    
    if not valid_cryptos:
        print("[ERROR] Unable to get sufficient data. Please check your data files.")
        return None
    
    print(f"\n[OK] Proceeding with {len(valid_cryptos)} cryptocurrencies")
    
    # Step 3: Run optimization
    print(f"\nStep 3: Running optimization (this may take several minutes)")
    print(f"Expected runtime: ~{len(valid_cryptos) * 2} minutes for {len(valid_cryptos)} cryptos")
    
    start_time = time.time()
    optimizer.optimize_all_cryptos(valid_cryptos, interval=5)
    end_time = time.time()
    
    print(f"⏰ Optimization completed in {(end_time - start_time)/60:.1f} minutes")
    
    # Step 4: Analyze results and find most profitable
    if not optimizer.results:
        print("[ERROR] No optimization results available")
        return None
    
    print(f"\n� Step 4: Analyzing results for most profitable cryptocurrencies")
    
    # Sort by profitability
    sorted_results = sorted(optimizer.results.items(), 
                           key=lambda x: x[1].return_pct, 
                           reverse=True)
    
    # Filter for profitable ones
    profitable_cryptos = [(symbol, result) for symbol, result in sorted_results 
                         if result.return_pct > 0]
    
    print(f"\nMOST PROFITABLE CRYPTOCURRENCIES FOR NEXT MONTH:")
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
    print(f"\nPORTFOLIO RECOMMENDATIONS:")
    
    if len(profitable_cryptos) >= 3:
        # Top 3 portfolio
        top_3 = profitable_cryptos[:3]
        top_3_return = sum(result.return_pct for _, result in top_3) / 3
        
        print(f"\nHIGH-RETURN PORTFOLIO (Top 3):")
        for symbol, result in top_3:
            print(f"   • {symbol}: SMA({result.best_params[0]},{result.best_params[1]}) - {result.return_pct:+.2f}%")
        print(f"   Expected Portfolio Return: {top_3_return:+.2f}%")
        
        # Balanced portfolio (top 5-8)
        if len(profitable_cryptos) >= 5:
            balanced = profitable_cryptos[:min(8, len(profitable_cryptos))]
            balanced_return = sum(result.return_pct for _, result in balanced) / len(balanced)
            
            print(f"\nBALANCED PORTFOLIO (Top {len(balanced)}):")
            for symbol, result in balanced:
                tier = optimizer.get_crypto_tier(symbol)
                print(f"   • {symbol} ({tier}): {result.return_pct:+.2f}%")
            print(f"   Expected Portfolio Return: {balanced_return:+.2f}%")
    
    # Step 6: Save results
    optimizer.save_results("most_profitable_cryptos.json")
    
    # Save optimal parameters for trading bot usage
    optimizer.save_optimal_parameters("optimal_sma_parameters.json")
    
    # Generate portfolio allocation for $49,000 investment
    optimizer.save_portfolio_allocation(30000, "portfolio_allocation.json")
    
    # Step 7: Implementation guide
    if profitable_cryptos:
        best_crypto, best_result = profitable_cryptos[0]
        
        print(f"\nIMPLEMENTATION GUIDE:")
        print(f"   Best Single Crypto: {best_crypto}")
        print(f"      • Parameters: SMA({best_result.best_params[0]}, {best_result.best_params[1]})")
        print(f"      • Expected Monthly Return: {best_result.return_pct:+.2f}%")
        print(f"      • Risk Level: {abs(best_result.max_drawdown):.2f}% max drawdown")
        print(f"      • Trade Frequency: {best_result.num_trades} trades expected")
        
        print(f"\nNEXT MONTH PROJECTIONS:")
        capitals = [1000, 5000, 10000, 50000]
        for capital in capitals:
            profit = capital * best_result.return_pct / 100
            print(f"      ${capital:,} → ${profit:+,.0f} profit")
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()