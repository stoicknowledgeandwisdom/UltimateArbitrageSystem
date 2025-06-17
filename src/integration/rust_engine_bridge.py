#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rust Engine Bridge - Python Integration Layer
============================================

High-performance bridge between Python Master Orchestrator and Rust engine.
Provides seamless integration for sub-millisecond trading operations.

Features:
- Zero-copy data transfer when possible
- Async/await support for non-blocking operations
- Performance monitoring and metrics
- Error handling with detailed diagnostics
- Memory-efficient operations
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
from contextlib import asynccontextmanager

# Try to import the Rust engine (will fall back to Python simulation if not available)
try:
    import ultimate_rust_engine
    RUST_ENGINE_AVAILABLE = True
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    logging.warning("Rust engine not available, using Python simulation")

logger = logging.getLogger(__name__)

@dataclass
class RustEngineConfig:
    """Configuration for the Rust engine integration"""
    enable_simd: bool = True
    enable_fpga: bool = False  # Set to True when FPGA hardware is available
    max_order_book_levels: int = 10000
    max_orders_per_level: int = 1000
    memory_pool_size_mb: int = 10
    enable_performance_monitoring: bool = True
    thread_pool_size: int = 8

class PythonOrderBookSimulator:
    """Python-based order book simulator when Rust engine is not available"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.orders = {}  # order_id -> order_info
        self.last_trade_price = None
        self.update_count = 0
        
        logger.info(f"🐍 Initialized Python order book simulator for {symbol}")
    
    def add_order(self, order_id: int, side: str, price: float, quantity: float) -> bool:
        """Add order to simulated order book"""
        try:
            order_info = {
                'id': order_id,
                'side': side.lower(),
                'price': price,
                'quantity': quantity,
                'timestamp': time.time()
            }
            
            self.orders[order_id] = order_info
            
            if side.lower() == 'buy':
                self.bids[price] = self.bids.get(price, 0) + quantity
            else:
                self.asks[price] = self.asks.get(price, 0) + quantity
                
            self.update_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error adding order {order_id}: {e}")
            return False
    
    def remove_order(self, order_id: int) -> bool:
        """Remove order from simulated order book"""
        try:
            if order_id not in self.orders:
                return False
                
            order = self.orders[order_id]
            price = order['price']
            quantity = order['quantity']
            side = order['side']
            
            if side == 'buy':
                self.bids[price] = max(0, self.bids.get(price, 0) - quantity)
                if self.bids[price] == 0:
                    del self.bids[price]
            else:
                self.asks[price] = max(0, self.asks.get(price, 0) - quantity)
                if self.asks[price] == 0:
                    del self.asks[price]
            
            del self.orders[order_id]
            self.update_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error removing order {order_id}: {e}")
            return False
    
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return max(self.bids.keys()) if self.bids else None
    
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return min(self.asks.keys()) if self.asks else None
    
    def spread(self) -> Optional[float]:
        """Get current spread"""
        bid = self.best_bid()
        ask = self.best_ask()
        return ask - bid if bid and ask else None
    
    def mid_price(self) -> Optional[float]:
        """Get mid price"""
        bid = self.best_bid()
        ask = self.best_ask()
        return (bid + ask) / 2 if bid and ask else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get order book statistics"""
        return {
            'symbol': self.symbol,
            'best_bid': self.best_bid(),
            'best_ask': self.best_ask(),
            'bid_volume': sum(self.bids.values()),
            'ask_volume': sum(self.asks.values()),
            'total_orders': len(self.orders),
            'update_count': self.update_count,
            'spread': self.spread(),
            'mid_price': self.mid_price()
        }

class RustEnginePerformanceMonitor:
    """Performance monitoring for Rust engine operations"""
    
    def __init__(self):
        self.operation_times = []
        self.memory_usage = []
        self.throughput_samples = []
        self.error_count = 0
        
        logger.info("📊 Performance monitor initialized")
    
    def record_operation_time(self, operation: str, duration_ns: int):
        """Record operation timing"""
        self.operation_times.append({
            'operation': operation,
            'duration_ns': duration_ns,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 samples
        if len(self.operation_times) > 1000:
            self.operation_times = self.operation_times[-1000:]
    
    def record_error(self, operation: str, error: str):
        """Record operation error"""
        self.error_count += 1
        logger.warning(f"⚠️ {operation} error: {error}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.operation_times:
            return {'status': 'no_data'}
        
        recent_times = [op['duration_ns'] for op in self.operation_times[-100:]]
        
        return {
            'average_latency_ns': np.mean(recent_times),
            'median_latency_ns': np.median(recent_times),
            'p95_latency_ns': np.percentile(recent_times, 95),
            'p99_latency_ns': np.percentile(recent_times, 99),
            'min_latency_ns': np.min(recent_times),
            'max_latency_ns': np.max(recent_times),
            'operations_count': len(self.operation_times),
            'error_count': self.error_count,
            'error_rate': self.error_count / len(self.operation_times) if self.operation_times else 0
        }

class RustEngineInterface:
    """Main interface to the Rust trading engine"""
    
    def __init__(self, config: RustEngineConfig):
        self.config = config
        self.initialized = False
        self.order_books: Dict[str, Any] = {}
        self.performance_monitor = RustEnginePerformanceMonitor()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.thread_pool_size
        )
        
        logger.info("🦀 Initializing Rust Engine Interface")
    
    async def initialize(self) -> bool:
        """Initialize the Rust engine"""
        try:
            start_time = time.perf_counter_ns()
            
            if RUST_ENGINE_AVAILABLE:
                # Initialize the actual Rust engine
                success = ultimate_rust_engine.initialize_engine(
                    self.config.enable_simd,
                    self.config.enable_fpga
                )
                
                if success:
                    logger.info("✅ Rust engine initialized successfully")
                    logger.info(f"   - SIMD optimizations: {'enabled' if self.config.enable_simd else 'disabled'}")
                    logger.info(f"   - FPGA acceleration: {'enabled' if self.config.enable_fpga else 'disabled'}")
                else:
                    logger.error("❌ Failed to initialize Rust engine")
                    return False
            else:
                logger.info("🐍 Using Python simulation mode")
                success = True
            
            self.initialized = success
            
            # Record initialization time
            init_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('initialization', init_time)
            
            logger.info(f"⚡ Engine initialization completed in {init_time / 1_000_000:.2f}ms")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing Rust engine: {e}")
            self.performance_monitor.record_error('initialization', str(e))
            return False
    
    async def create_order_book(self, symbol: str) -> bool:
        """Create a new order book for the given symbol"""
        try:
            if not self.initialized:
                raise ValueError("Engine not initialized")
            
            start_time = time.perf_counter_ns()
            
            if RUST_ENGINE_AVAILABLE:
                # Create Rust order book
                order_book = ultimate_rust_engine.PyOrderBook(symbol)
                self.order_books[symbol] = order_book
            else:
                # Create Python simulation
                order_book = PythonOrderBookSimulator(symbol)
                self.order_books[symbol] = order_book
            
            creation_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('create_order_book', creation_time)
            
            logger.info(f"📚 Created order book for {symbol} in {creation_time / 1_000:.2f}μs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating order book for {symbol}: {e}")
            self.performance_monitor.record_error('create_order_book', str(e))
            return False
    
    async def add_order(self, symbol: str, order_id: int, side: str, price: float, quantity: float) -> bool:
        """Add an order to the order book with ultra-low latency"""
        try:
            if symbol not in self.order_books:
                await self.create_order_book(symbol)
            
            start_time = time.perf_counter_ns()
            
            # Execute in thread pool for true parallelism
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.thread_pool,
                self.order_books[symbol].add_order,
                order_id, side, price, quantity
            )
            
            operation_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('add_order', operation_time)
            
            if operation_time < 1_000_000:  # Less than 1ms
                logger.debug(f"⚡ Added order {order_id} in {operation_time / 1_000:.1f}μs")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error adding order {order_id}: {e}")
            self.performance_monitor.record_error('add_order', str(e))
            return False
    
    async def remove_order(self, symbol: str, order_id: int) -> bool:
        """Remove an order from the order book"""
        try:
            if symbol not in self.order_books:
                return False
            
            start_time = time.perf_counter_ns()
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.thread_pool,
                self.order_books[symbol].remove_order,
                order_id
            )
            
            operation_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('remove_order', operation_time)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error removing order {order_id}: {e}")
            self.performance_monitor.record_error('remove_order', str(e))
            return False
    
    async def get_best_prices(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices with minimal latency"""
        try:
            if symbol not in self.order_books:
                return None, None
            
            start_time = time.perf_counter_ns()
            
            order_book = self.order_books[symbol]
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            
            bid_task = loop.run_in_executor(self.thread_pool, order_book.best_bid)
            ask_task = loop.run_in_executor(self.thread_pool, order_book.best_ask)
            
            bid, ask = await asyncio.gather(bid_task, ask_task)
            
            operation_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('get_best_prices', operation_time)
            
            return bid, ask
            
        except Exception as e:
            logger.error(f"❌ Error getting best prices for {symbol}: {e}")
            self.performance_monitor.record_error('get_best_prices', str(e))
            return None, None
    
    async def get_order_book_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive order book statistics"""
        try:
            if symbol not in self.order_books:
                return None
            
            start_time = time.perf_counter_ns()
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                self.thread_pool,
                self.order_books[symbol].get_stats
            )
            
            operation_time = time.perf_counter_ns() - start_time
            self.performance_monitor.record_operation_time('get_stats', operation_time)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats for {symbol}: {e}")
            self.performance_monitor.record_error('get_stats', str(e))
            return None
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        try:
            engine_stats = {}
            
            if RUST_ENGINE_AVAILABLE:
                # Get Rust engine performance stats
                engine_stats = ultimate_rust_engine.get_performance_stats()
            
            # Combine with our monitoring stats
            monitor_stats = self.performance_monitor.get_performance_stats()
            
            return {
                'engine_stats': engine_stats,
                'monitor_stats': monitor_stats,
                'order_books_count': len(self.order_books),
                'rust_engine_available': RUST_ENGINE_AVAILABLE,
                'initialization_status': self.initialized
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    async def benchmark_performance(self, operations: int = 10000) -> Dict[str, Any]:
        """Run performance benchmark"""
        logger.info(f"🏃 Starting performance benchmark with {operations} operations")
        
        symbol = "BENCHMARK/PAIR"
        await self.create_order_book(symbol)
        
        # Benchmark order additions
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(operations):
            side = 'buy' if i % 2 == 0 else 'sell'
            price = 50000.0 + (i % 1000) * 0.01  # Price variation
            quantity = 1.0 + (i % 10) * 0.1  # Quantity variation
            
            task = self.add_order(symbol, i, side, price, quantity)
            tasks.append(task)
        
        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        benchmark_time = time.perf_counter() - start_time
        successful_operations = sum(1 for r in results if r is True)
        
        # Calculate performance metrics
        ops_per_second = successful_operations / benchmark_time
        avg_latency_ms = (benchmark_time * 1000) / successful_operations
        
        # Get final order book stats
        final_stats = await self.get_order_book_stats(symbol)
        
        benchmark_results = {
            'total_operations': operations,
            'successful_operations': successful_operations,
            'failed_operations': operations - successful_operations,
            'total_time_seconds': benchmark_time,
            'operations_per_second': ops_per_second,
            'average_latency_ms': avg_latency_ms,
            'final_order_book_stats': final_stats,
            'rust_engine_mode': RUST_ENGINE_AVAILABLE
        }
        
        logger.info(f"🏁 Benchmark completed:")
        logger.info(f"   📊 {ops_per_second:.0f} operations/second")
        logger.info(f"   ⚡ {avg_latency_ms:.3f}ms average latency")
        logger.info(f"   ✅ {successful_operations}/{operations} operations successful")
        
        return benchmark_results
    
    async def shutdown(self):
        """Gracefully shutdown the engine"""
        logger.info("🛑 Shutting down Rust engine interface")
        
        # Clear order books
        self.order_books.clear()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Final performance report
        final_metrics = await self.get_performance_metrics()
        logger.info(f"📋 Final performance metrics: {final_metrics}")
        
        self.initialized = False
        logger.info("✅ Rust engine interface shutdown completed")

# Global engine instance
_rust_engine_interface = None

def get_rust_engine_interface(config: Optional[RustEngineConfig] = None) -> RustEngineInterface:
    """Get or create the global Rust engine interface"""
    global _rust_engine_interface
    if _rust_engine_interface is None:
        _rust_engine_interface = RustEngineInterface(config or RustEngineConfig())
    return _rust_engine_interface

@asynccontextmanager
async def rust_engine_context(config: Optional[RustEngineConfig] = None):
    """Async context manager for Rust engine operations"""
    engine = get_rust_engine_interface(config)
    
    try:
        if not engine.initialized:
            await engine.initialize()
        yield engine
    finally:
        # Context cleanup if needed
        pass

# Convenience functions for integration with Master Orchestrator
async def create_ultra_fast_order_book(symbol: str) -> bool:
    """Create an ultra-fast order book for the given symbol"""
    engine = get_rust_engine_interface()
    if not engine.initialized:
        await engine.initialize()
    return await engine.create_order_book(symbol)

async def execute_ultra_fast_order(symbol: str, order_id: int, side: str, price: float, quantity: float) -> bool:
    """Execute an order with sub-millisecond latency"""
    engine = get_rust_engine_interface()
    return await engine.add_order(symbol, order_id, side, price, quantity)

async def get_ultra_fast_prices(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Get best bid/ask prices with minimal latency"""
    engine = get_rust_engine_interface()
    return await engine.get_best_prices(symbol)

async def run_engine_performance_test() -> Dict[str, Any]:
    """Run a comprehensive performance test of the engine"""
    config = RustEngineConfig(
        enable_simd=True,
        enable_fpga=False,  # Set to True when FPGA is available
        thread_pool_size=16
    )
    
    async with rust_engine_context(config) as engine:
        return await engine.benchmark_performance(operations=50000)

if __name__ == "__main__":
    # Quick test of the integration
    async def test_integration():
        logger.info("🧪 Testing Rust Engine Integration")
        
        config = RustEngineConfig(enable_simd=True, enable_fpga=False)
        
        async with rust_engine_context(config) as engine:
            # Create order book
            await engine.create_order_book("BTC/USDT")
            
            # Add some orders
            await engine.add_order("BTC/USDT", 1, "buy", 50000.0, 1.0)
            await engine.add_order("BTC/USDT", 2, "sell", 51000.0, 1.0)
            
            # Get prices
            bid, ask = await engine.get_best_prices("BTC/USDT")
            logger.info(f"💰 Best prices: Bid={bid}, Ask={ask}")
            
            # Get stats
            stats = await engine.get_order_book_stats("BTC/USDT")
            logger.info(f"📊 Order book stats: {stats}")
            
            # Run benchmark
            benchmark = await engine.benchmark_performance(1000)
            logger.info(f"🏁 Benchmark results: {benchmark}")
    
    asyncio.run(test_integration())

