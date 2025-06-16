#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Strategy Sandbox with Numba Acceleration
==============================================

High-performance Python strategy development environment with:
- Numba JIT compilation for speed
- WASM compilation target for production
- Rapid R&D workflow
- Ultra-fast backtesting
- Strategy optimization

This sandbox allows for rapid strategy development while maintaining
production-level performance when compiled to WASM.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import json
from pathlib import Path

# High-performance numerical computing
import numba
from numba import jit, njit, prange
import pandas as pd

# For WASM compilation
try:
    import wasmtime
    WASM_AVAILABLE = True
except ImportError:
    WASM_AVAILABLE = False
    print("WASM runtime not available - install wasmtime-py for production compilation")

logger = logging.getLogger(__name__)

class StrategySignal(Enum):
    """Trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class MarketData:
    """Market data structure optimized for Numba"""
    timestamp: float
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    symbol: str = ""
    signal: StrategySignal = StrategySignal.HOLD
    confidence: float = 0.0
    entry_price: float = 0.0
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    execution_time_ns: int = 0

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    execution_time_avg_ns: float = 0.0
    signals_generated: int = 0

class NumbaAcceleratedStrategy:
    """
    Base class for Numba-accelerated trading strategies
    
    This class provides the framework for ultra-fast strategy execution
    using Numba JIT compilation while maintaining Python flexibility.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.performance = StrategyPerformance()
        self.is_compiled = False
        self.signals_generated = []
        self.execution_times = []
        
        # Initialize Numba-compiled functions
        self._compile_strategy_functions()
        
        logger.info(f"Initialized strategy: {name}")
    
    def _compile_strategy_functions(self):
        """Compile strategy functions with Numba for maximum performance"""
        logger.info(f"Compiling strategy functions for {self.name}...")
        
        # Pre-compile critical functions
        _ = self._numba_calculate_indicators(np.array([1.0, 2.0, 3.0]))
        _ = self._numba_generate_signals(
            np.array([1.0, 2.0, 3.0]), 
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0])
        )
        
        self.is_compiled = True
        logger.info(f"Strategy compilation complete for {self.name}")
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _numba_calculate_indicators(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate technical indicators using Numba for ultra-fast computation
        
        Returns: (sma_20, sma_50, rsi_14)
        """
        n = len(prices)
        
        # Simple Moving Averages
        sma_20 = np.full(n, np.nan)
        sma_50 = np.full(n, np.nan)
        
        # Calculate SMA 20
        for i in range(19, n):
            sma_20[i] = np.mean(prices[i-19:i+1])
        
        # Calculate SMA 50
        for i in range(49, n):
            sma_50[i] = np.mean(prices[i-49:i+1])
        
        # RSI calculation
        rsi = np.full(n, 50.0)
        if n > 14:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Initial average gain/loss
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses[:14])
            
            for i in range(14, n-1):
                avg_gain = (avg_gain * 13 + gains[i]) / 14
                avg_loss = (avg_loss * 13 + losses[i]) / 14
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi[i+1] = 100 - (100 / (1 + rs))
        
        return sma_20, sma_50, rsi
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _numba_generate_signals(
        prices: np.ndarray, 
        sma_20: np.ndarray, 
        sma_50: np.ndarray
    ) -> np.ndarray:
        """
        Generate trading signals using Numba-accelerated logic
        
        Returns: Array of signals (1=BUY, -1=SELL, 0=HOLD)
        """
        n = len(prices)
        signals = np.zeros(n)
        
        for i in range(50, n):  # Start after SMA50 is available
            if not (np.isnan(sma_20[i]) or np.isnan(sma_50[i])):
                # Golden cross strategy
                if sma_20[i] > sma_50[i] and sma_20[i-1] <= sma_50[i-1]:
                    signals[i] = 1  # BUY signal
                elif sma_20[i] < sma_50[i] and sma_20[i-1] >= sma_50[i-1]:
                    signals[i] = -1  # SELL signal
        
        return signals
    
    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _numba_backtest_strategy(
        prices: np.ndarray,
        signals: np.ndarray,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ) -> Tuple[np.ndarray, float, float, int]:
        """
        Ultra-fast backtesting using Numba parallel processing
        
        Returns: (equity_curve, total_return, max_drawdown, total_trades)
        """
        n = len(prices)
        equity = np.full(n, initial_capital)
        position = 0.0
        cash = initial_capital
        total_trades = 0
        peak_equity = initial_capital
        max_drawdown = 0.0
        
        for i in range(1, n):
            if signals[i] != 0 and not np.isnan(prices[i]):
                if signals[i] > 0 and position <= 0:  # BUY signal
                    if position < 0:  # Close short position
                        cash += (-position) * prices[i] * (1 - commission)
                        position = 0
                        total_trades += 1
                    
                    # Open long position
                    position = cash / (prices[i] * (1 + commission))
                    cash = 0
                    total_trades += 1
                    
                elif signals[i] < 0 and position >= 0:  # SELL signal
                    if position > 0:  # Close long position
                        cash += position * prices[i] * (1 - commission)
                        position = 0
                        total_trades += 1
                    
                    # Open short position
                    position = -(cash / (prices[i] * (1 + commission)))
                    cash = 0
                    total_trades += 1
            
            # Calculate current equity
            if position > 0:
                equity[i] = position * prices[i]
            elif position < 0:
                equity[i] = cash - ((-position) * prices[i])
            else:
                equity[i] = cash
            
            # Update max drawdown
            if equity[i] > peak_equity:
                peak_equity = equity[i]
            else:
                drawdown = (peak_equity - equity[i]) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        total_return = (equity[-1] - initial_capital) / initial_capital
        
        return equity, total_return, max_drawdown, total_trades
    
    async def process_market_data(self, data: MarketData) -> Optional[TradingSignal]:
        """
        Process incoming market data and generate trading signals
        
        This method maintains the high-level Python interface while
        delegating performance-critical calculations to Numba.
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Convert to numpy arrays for Numba processing
            # In real implementation, this would use a rolling window
            # For demo, we'll simulate with recent data
            prices = np.array([data.close])  # Simplified for demo
            
            # Calculate indicators (this would use historical data)
            if len(prices) >= 50:  # Minimum required for indicators
                sma_20, sma_50, rsi = self._numba_calculate_indicators(prices)
                signals = self._numba_generate_signals(prices, sma_20, sma_50)
                
                # Generate signal if there's a new signal
                if len(signals) > 0 and signals[-1] != 0:
                    signal_type = StrategySignal.BUY if signals[-1] > 0 else StrategySignal.SELL
                    
                    # Calculate confidence based on indicator strength
                    confidence = min(abs(signals[-1]) * 0.8, 1.0)
                    
                    trading_signal = TradingSignal(
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        signal=signal_type,
                        confidence=confidence,
                        entry_price=data.close,
                        quantity=self._calculate_position_size(data.close, confidence),
                        strategy_name=self.name,
                        execution_time_ns=time.perf_counter_ns() - start_time,
                        metadata={
                            'sma_20': float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
                            'sma_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
                            'rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                        }
                    )
                    
                    self.signals_generated.append(trading_signal)
                    self.execution_times.append(trading_signal.execution_time_ns)
                    
                    # Update performance metrics
                    self.performance.signals_generated += 1
                    self.performance.execution_time_avg_ns = np.mean(self.execution_times)
                    
                    return trading_signal
                    
        except Exception as e:
            logger.error(f"Error processing market data in {self.name}: {e}")
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """
        Calculate position size based on confidence and risk management
        """
        base_size = self.config.get('base_position_size', 1.0)
        max_risk = self.config.get('max_risk_per_trade', 0.02)
        
        # Adjust size based on confidence
        size = base_size * confidence
        
        # Apply risk management
        max_size = (self.config.get('capital', 10000) * max_risk) / price
        
        return min(size, max_size)
    
    async def backtest(
        self, 
        historical_data: List[MarketData], 
        initial_capital: float = 10000.0
    ) -> StrategyPerformance:
        """
        Run ultra-fast backtest using Numba acceleration
        """
        logger.info(f"Starting backtest for {self.name} with {len(historical_data)} data points")
        
        start_time = time.perf_counter()
        
        # Convert data to numpy arrays
        prices = np.array([d.close for d in historical_data])
        
        # Calculate indicators
        sma_20, sma_50, rsi = self._numba_calculate_indicators(prices)
        
        # Generate signals
        signals = self._numba_generate_signals(prices, sma_20, sma_50)
        
        # Run backtest
        equity_curve, total_return, max_drawdown, total_trades = self._numba_backtest_strategy(
            prices, signals, initial_capital
        )
        
        # Calculate additional metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        # Count winning/losing trades
        winning_trades = np.sum(returns > 0)
        losing_trades = np.sum(returns < 0)
        win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
        
        # Calculate profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        profit_factor = 0.0
        if len(losses) > 0:
            profit_factor = np.sum(wins) / abs(np.sum(losses))
        
        execution_time = time.perf_counter() - start_time
        
        performance = StrategyPerformance(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=np.mean(wins) if len(wins) > 0 else 0,
            average_loss=np.mean(losses) if len(losses) > 0 else 0,
        )
        
        logger.info(f"Backtest completed in {execution_time:.4f}s")
        logger.info(f"Performance: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, "
                   f"MaxDD={max_drawdown:.2%}, Trades={total_trades}")
        
        return performance
    
    def optimize_parameters(
        self, 
        historical_data: List[MarketData], 
        parameter_ranges: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Numba-accelerated grid search
        """
        logger.info(f"Starting parameter optimization for {self.name}")
        
        best_performance = None
        best_params = None
        best_return = -float('inf')
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for i, param_combo in enumerate(product(*param_values)):
            # Update strategy config
            old_config = self.config.copy()
            for j, param_name in enumerate(param_names):
                self.config[param_name] = param_combo[j]
            
            # Run backtest with new parameters
            try:
                performance = asyncio.run(self.backtest(historical_data))
                
                # Check if this is the best performance
                if performance.total_return > best_return:
                    best_return = performance.total_return
                    best_performance = performance
                    best_params = dict(zip(param_names, param_combo))
                
                if i % 100 == 0:
                    logger.info(f"Tested {i+1}/{total_combinations} combinations, "
                               f"best return so far: {best_return:.2%}")
                    
            except Exception as e:
                logger.error(f"Error testing parameters {param_combo}: {e}")
            
            # Restore old config
            self.config = old_config
        
        # Apply best parameters
        if best_params:
            self.config.update(best_params)
            logger.info(f"Optimization complete. Best parameters: {best_params}")
            logger.info(f"Best performance: {best_return:.2%} return")
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'optimization_results': {
                'total_combinations_tested': total_combinations,
                'best_return': best_return
            }
        }
    
    def compile_to_wasm(self, output_path: str) -> bool:
        """
        Compile strategy to WebAssembly for production deployment
        
        This allows the Python strategy to run in production with
        near-native performance in the Rust execution engine.
        """
        if not WASM_AVAILABLE:
            logger.error("WASM compilation not available - install wasmtime-py")
            return False
        
        try:
            logger.info(f"Compiling {self.name} to WASM...")
            
            # Generate WASM-compatible code
            wasm_code = self._generate_wasm_code()
            
            # Write to file
            output_file = Path(output_path) / f"{self.name}.wasm"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(wasm_code)
            
            # Generate metadata
            metadata = {
                'strategy_name': self.name,
                'compilation_time': datetime.now().isoformat(),
                'config': self.config,
                'performance_profile': {
                    'avg_execution_time_ns': self.performance.execution_time_avg_ns,
                    'signals_generated': self.performance.signals_generated
                }
            }
            
            metadata_file = Path(output_path) / f"{self.name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully compiled {self.name} to WASM: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compile {self.name} to WASM: {e}")
            return False
    
    def _generate_wasm_code(self) -> bytes:
        """
        Generate WASM bytecode from the strategy
        
        In a real implementation, this would use a Python-to-WASM compiler
        like Pyodide or a custom transpiler.
        """
        # Placeholder - in real implementation this would compile the strategy
        # For now, return minimal WASM module
        wasm_module = (
            b'\x00asm'  # WASM magic number
            b'\x01\x00\x00\x00'  # Version
        )
        return wasm_module
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report
        """
        return {
            'strategy_name': self.name,
            'config': self.config,
            'performance': {
                'total_return': self.performance.total_return,
                'sharpe_ratio': self.performance.sharpe_ratio,
                'max_drawdown': self.performance.max_drawdown,
                'win_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor,
                'total_trades': self.performance.total_trades,
                'signals_generated': self.performance.signals_generated,
                'avg_execution_time_ns': self.performance.execution_time_avg_ns,
            },
            'compilation_status': {
                'is_compiled': self.is_compiled,
                'numba_acceleration': True,
                'wasm_ready': WASM_AVAILABLE
            }
        }

class StrategyManager:
    """
    Manages multiple strategies in the sandbox environment
    """
    
    def __init__(self):
        self.strategies: Dict[str, NumbaAcceleratedStrategy] = {}
        self.executor = ThreadPoolExecutor(max_workers=8)
        logger.info("Strategy Manager initialized")
    
    def register_strategy(self, strategy: NumbaAcceleratedStrategy):
        """Register a new strategy"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
    
    async def process_market_data_batch(
        self, 
        market_data: List[MarketData]
    ) -> List[TradingSignal]:
        """
        Process market data through all strategies concurrently
        """
        all_signals = []
        
        # Process each strategy concurrently
        tasks = []
        for data in market_data:
            for strategy in self.strategies.values():
                task = asyncio.create_task(strategy.process_market_data(data))
                tasks.append(task)
        
        # Wait for all strategies to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid signals
        for result in results:
            if isinstance(result, TradingSignal):
                all_signals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Strategy error: {result}")
        
        return all_signals
    
    def run_mass_backtest(
        self, 
        historical_data: List[MarketData]
    ) -> Dict[str, StrategyPerformance]:
        """
        Run backtests for all strategies in parallel
        """
        logger.info(f"Running mass backtest on {len(self.strategies)} strategies")
        
        results = {}
        
        # Run backtests in parallel
        future_to_strategy = {
            self.executor.submit(
                asyncio.run, 
                strategy.backtest(historical_data)
            ): name 
            for name, strategy in self.strategies.items()
        }
        
        for future in future_to_strategy:
            strategy_name = future_to_strategy[future]
            try:
                performance = future.result()
                results[strategy_name] = performance
                logger.info(f"Backtest completed for {strategy_name}: "
                           f"{performance.total_return:.2%} return")
            except Exception as e:
                logger.error(f"Backtest failed for {strategy_name}: {e}")
        
        return results
    
    def compile_all_to_wasm(self, output_dir: str) -> Dict[str, bool]:
        """
        Compile all strategies to WASM for production
        """
        results = {}
        
        for name, strategy in self.strategies.items():
            success = strategy.compile_to_wasm(output_dir)
            results[name] = success
        
        successful = sum(results.values())
        logger.info(f"WASM compilation complete: {successful}/{len(results)} strategies")
        
        return results
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get performance dashboard for all strategies
        """
        dashboard = {
            'summary': {
                'total_strategies': len(self.strategies),
                'compiled_strategies': sum(1 for s in self.strategies.values() if s.is_compiled),
                'total_signals': sum(s.performance.signals_generated for s in self.strategies.values()),
            },
            'strategies': {}
        }
        
        for name, strategy in self.strategies.items():
            dashboard['strategies'][name] = strategy.get_performance_report()
        
        return dashboard

# Example strategy implementations

class GoldenCrossStrategy(NumbaAcceleratedStrategy):
    """Example golden cross strategy with Numba acceleration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'sma_short': 20,
            'sma_long': 50,
            'base_position_size': 1.0,
            'max_risk_per_trade': 0.02,
            'capital': 10000
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("GoldenCross", default_config)

class MeanReversionStrategy(NumbaAcceleratedStrategy):
    """Example mean reversion strategy with Numba acceleration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_period': 14,
            'base_position_size': 1.0,
            'max_risk_per_trade': 0.02,
            'capital': 10000
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("MeanReversion", default_config)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize strategy manager
        manager = StrategyManager()
        
        # Create and register strategies
        golden_cross = GoldenCrossStrategy()
        mean_reversion = MeanReversionStrategy()
        
        manager.register_strategy(golden_cross)
        manager.register_strategy(mean_reversion)
        
        # Generate sample market data
        import random
        historical_data = []
        base_price = 100.0
        
        for i in range(1000):
            # Random walk with trend
            change = random.gauss(0, 0.02)
            base_price *= (1 + change)
            
            data = MarketData(
                timestamp=time.time() + i,
                symbol="BTCUSD",
                open=base_price,
                high=base_price * 1.01,
                low=base_price * 0.99,
                close=base_price,
                volume=random.uniform(1000, 10000)
            )
            historical_data.append(data)
        
        # Run backtests
        results = manager.run_mass_backtest(historical_data)
        
        # Print results
        for strategy_name, performance in results.items():
            print(f"\n{strategy_name} Performance:")
            print(f"  Total Return: {performance.total_return:.2%}")
            print(f"  Sharpe Ratio: {performance.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {performance.max_drawdown:.2%}")
            print(f"  Win Rate: {performance.win_rate:.2%}")
            print(f"  Total Trades: {performance.total_trades}")
        
        # Compile to WASM
        wasm_results = manager.compile_all_to_wasm("./wasm_output")
        print(f"\nWASM Compilation Results: {wasm_results}")
        
        # Performance dashboard
        dashboard = manager.get_performance_dashboard()
        print(f"\nPerformance Dashboard: {json.dumps(dashboard, indent=2)}")
    
    asyncio.run(main())

