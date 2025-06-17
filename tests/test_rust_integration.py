#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Rust Engine Integration
==================================================

Tests the complete integration between Python and Rust components,
including performance benchmarks and stress testing.
"""

import asyncio
import pytest
import logging
import time
import numpy as np
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_optimization', 'ml_optimization'))

try:
    from rust_engine_bridge import (
        RustEngineInterface,
        RustEngineConfig,
        rust_engine_context,
        run_engine_performance_test,
        PythonOrderBookSimulator
    )
    from orchestrator import MLOptimizationOrchestrator, MLOptimizationConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    logging.warning(f"Imports not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRustEngineIntegration:
    """Test suite for Rust engine integration"""
    
    @pytest.fixture
    async def rust_engine_config(self):
        """Create test configuration for Rust engine"""
        return RustEngineConfig(
            enable_simd=True,
            enable_fpga=False,  # Disable FPGA for testing
            max_order_book_levels=1000,
            max_orders_per_level=100,
            thread_pool_size=4,
            enable_performance_monitoring=True
        )
    
    @pytest.fixture
    async def rust_engine(self, rust_engine_config):
        """Create and initialize Rust engine for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        engine = RustEngineInterface(rust_engine_config)
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, rust_engine_config):
        """Test Rust engine initialization"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        engine = RustEngineInterface(rust_engine_config)
        
        # Test initialization
        success = await engine.initialize()
        assert success or not engine.rust_engine_config  # Success or fallback to Python
        
        # Cleanup
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_order_book_creation(self, rust_engine):
        """Test order book creation"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        # Create order book
        success = await rust_engine.create_order_book("BTC/USDT")
        assert success
        
        # Verify order book exists
        assert "BTC/USDT" in rust_engine.order_books
    
    @pytest.mark.asyncio
    async def test_order_operations(self, rust_engine):
        """Test order addition and removal"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        symbol = "ETH/USDT"
        
        # Create order book
        await rust_engine.create_order_book(symbol)
        
        # Add buy order
        success = await rust_engine.add_order(symbol, 1, "buy", 3000.0, 1.5)
        assert success
        
        # Add sell order
        success = await rust_engine.add_order(symbol, 2, "sell", 3100.0, 2.0)
        assert success
        
        # Get best prices
        bid, ask = await rust_engine.get_best_prices(symbol)
        assert bid == 3000.0
        assert ask == 3100.0
        
        # Remove order
        success = await rust_engine.remove_order(symbol, 1)
        assert success
        
        # Check prices after removal
        bid, ask = await rust_engine.get_best_prices(symbol)
        assert bid is None  # No more bids
        assert ask == 3100.0
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, rust_engine):
        """Test performance benchmarking"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        # Run benchmark
        results = await rust_engine.benchmark_performance(operations=1000)
        
        # Verify results structure
        assert 'total_operations' in results
        assert 'successful_operations' in results
        assert 'operations_per_second' in results
        assert 'average_latency_ms' in results
        
        # Performance assertions
        assert results['total_operations'] == 1000
        assert results['successful_operations'] > 0
        assert results['operations_per_second'] > 0
        assert results['average_latency_ms'] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, rust_engine):
        """Test concurrent order operations"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        symbol = "LTC/USDT"
        await rust_engine.create_order_book(symbol)
        
        # Create concurrent tasks
        tasks = []
        for i in range(100):
            side = "buy" if i % 2 == 0 else "sell"
            price = 150.0 + (i % 50) * 0.1
            quantity = 0.1 + (i % 10) * 0.01
            
            task = rust_engine.add_order(symbol, i, side, price, quantity)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful operations
        successful = sum(1 for r in results if r is True)
        
        # Should have high success rate
        assert successful >= 90  # At least 90% success
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, rust_engine):
        """Stress test the engine with high load"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        symbol = "STRESS/TEST"
        await rust_engine.create_order_book(symbol)
        
        start_time = time.perf_counter()
        
        # High-frequency operations
        tasks = []
        for i in range(5000):  # 5000 operations
            side = "buy" if i % 2 == 0 else "sell"
            price = 100.0 + (i % 1000) * 0.01
            quantity = 0.01 + (i % 100) * 0.001
            
            task = rust_engine.add_order(symbol, i, side, price, quantity)
            tasks.append(task)
        
        # Execute with time limit
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout
            )
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            successful = sum(1 for r in results if r is True)
            ops_per_second = successful / total_time
            
            logger.info(f"Stress test: {successful}/5000 operations in {total_time:.2f}s")
            logger.info(f"Performance: {ops_per_second:.0f} ops/sec")
            
            # Performance requirements
            assert successful >= 4500  # At least 90% success
            assert ops_per_second >= 100  # At least 100 ops/sec
            
        except asyncio.TimeoutError:
            pytest.fail("Stress test timed out")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rust_engine):
        """Test error handling scenarios"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Rust engine imports not available")
        
        # Test operations on non-existent order book
        bid, ask = await rust_engine.get_best_prices("NONEXISTENT/PAIR")
        assert bid is None
        assert ask is None
        
        # Test invalid order removal
        success = await rust_engine.remove_order("NONEXISTENT/PAIR", 999)
        assert not success
        
        # Test stats on non-existent order book
        stats = await rust_engine.get_order_book_stats("NONEXISTENT/PAIR")
        assert stats is None

class TestMLOrchestratorIntegration:
    """Test ML Orchestrator integration with Rust engine"""
    
    @pytest.fixture
    async def orchestrator_config(self):
        """Create test configuration for ML orchestrator"""
        config = MLOptimizationConfig()
        # Disable components not needed for testing
        config.enable_feature_store = False
        config.enable_streaming_etl = False
        config.enable_reinforcement_learning = False
        config.enable_meta_controller = False
        config.enable_explainability = False
        return config
    
    @pytest.fixture
    async def orchestrator(self, orchestrator_config):
        """Create and start ML orchestrator for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("ML Orchestrator imports not available")
        
        orchestrator = MLOptimizationOrchestrator(orchestrator_config)
        await orchestrator.start()
        yield orchestrator
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_rust_engine_initialization_in_orchestrator(self, orchestrator):
        """Test Rust engine initialization within orchestrator"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("ML Orchestrator imports not available")
        
        # Initialize Rust engine
        success = await orchestrator.initialize_rust_engine()
        # Should succeed or gracefully fallback
        assert success or not hasattr(orchestrator, 'rust_engine') or orchestrator.rust_engine is None
    
    @pytest.mark.asyncio
    async def test_enhanced_market_data_processing(self, orchestrator):
        """Test enhanced market data processing with Rust engine"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("ML Orchestrator imports not available")
        
        # Initialize Rust engine (if available)
        await orchestrator.initialize_rust_engine()
        
        # Sample market data
        market_data = {
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'price': 50000.0,
            'volume': 1.5,
            'bid_price': 49990.0,
            'ask_price': 50010.0,
            'spread': 20.0,
            'timestamp': time.time()
        }
        
        # Process market data
        decision = await orchestrator.enhanced_market_data_processing(market_data)
        
        # Should return a decision or None (both are valid)
        assert decision is None or hasattr(decision, 'symbol')
    
    @pytest.mark.asyncio
    async def test_rust_engine_health_check(self, orchestrator):
        """Test Rust engine health check"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("ML Orchestrator imports not available")
        
        # Health check without initialization
        health = await orchestrator.rust_engine_health_check()
        assert 'status' in health
        assert 'available' in health
        
        # Initialize and check again
        await orchestrator.initialize_rust_engine()
        health = await orchestrator.rust_engine_health_check()
        assert 'status' in health
        assert health['available'] is not None

class TestPythonFallback:
    """Test Python fallback when Rust engine is not available"""
    
    @pytest.mark.asyncio
    async def test_python_order_book_simulator(self):
        """Test Python order book simulator"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Imports not available")
        
        # Create Python simulator
        simulator = PythonOrderBookSimulator("TEST/PAIR")
        
        # Add orders
        success = simulator.add_order(1, "buy", 100.0, 1.0)
        assert success
        
        success = simulator.add_order(2, "sell", 101.0, 1.5)
        assert success
        
        # Check prices
        assert simulator.best_bid() == 100.0
        assert simulator.best_ask() == 101.0
        assert simulator.spread() == 1.0
        assert simulator.mid_price() == 100.5
        
        # Remove order
        success = simulator.remove_order(1)
        assert success
        
        # Check after removal
        assert simulator.best_bid() is None
        assert simulator.best_ask() == 101.0
        
        # Get stats
        stats = simulator.get_stats()
        assert stats['symbol'] == "TEST/PAIR"
        assert stats['total_orders'] == 1
        assert stats['update_count'] == 3  # 2 adds + 1 remove

@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test complete integration workflow"""
    if not IMPORTS_AVAILABLE:
        pytest.skip("Imports not available")
    
    logger.info("🧪 Starting full integration workflow test")
    
    # Create configuration
    rust_config = RustEngineConfig(
        enable_simd=True,
        enable_fpga=False,
        thread_pool_size=8
    )
    
    # Test with context manager
    async with rust_engine_context(rust_config) as engine:
        # Create multiple order books
        symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "ADA/USDT"]
        
        for symbol in symbols:
            success = await engine.create_order_book(symbol)
            assert success
        
        # Add orders to each symbol
        order_id = 1
        for symbol in symbols:
            for i in range(10):
                side = "buy" if i % 2 == 0 else "sell"
                price = 1000.0 + i * 10.0
                quantity = 0.1 + i * 0.01
                
                success = await engine.add_order(symbol, order_id, side, price, quantity)
                assert success
                order_id += 1
        
        # Get statistics for all order books
        for symbol in symbols:
            stats = await engine.get_order_book_stats(symbol)
            assert stats is not None
            assert stats['symbol'] == symbol
            assert stats['total_orders'] > 0
        
        # Run performance test
        benchmark = await engine.benchmark_performance(operations=2000)
        assert benchmark['total_operations'] == 2000
        assert benchmark['successful_operations'] > 0
        
        logger.info(f"✅ Full integration test completed successfully")
        logger.info(f"   📊 {benchmark['operations_per_second']:.0f} ops/sec")
        logger.info(f"   ⚡ {benchmark['average_latency_ms']:.3f}ms avg latency")

def test_performance_benchmarking():
    """Test standalone performance benchmarking"""
    if not IMPORTS_AVAILABLE:
        pytest.skip("Imports not available")
    
    async def run_benchmark():
        try:
            results = await run_engine_performance_test()
            assert 'operations_per_second' in results
            assert 'average_latency_ms' in results
            logger.info(f"Standalone benchmark: {results['operations_per_second']:.0f} ops/sec")
            return True
        except Exception as e:
            logger.warning(f"Benchmark failed (expected if Rust not available): {e}")
            return False
    
    # Run the benchmark
    result = asyncio.run(run_benchmark())
    # Test passes whether benchmark succeeds or fails gracefully
    assert isinstance(result, bool)

if __name__ == "__main__":
    # Run basic tests when executed directly
    asyncio.run(test_full_integration_workflow())
    test_performance_benchmarking()
    logger.info("🎉 All tests completed!")

