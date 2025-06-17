#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Ultimate Master Orchestrator
=======================================================

Testing framework for the Day 1 implementation of the Master Orchestration Engine.
Validates all core components for maximum profit generation and autonomous operation.

Test Categories:
- Signal Fusion Engine Tests
- Performance Optimizer Tests  
- Health Monitor Tests
- Execution Coordinator Tests
- Integration Tests
- Performance Benchmarks
- Stress Tests
"""

import asyncio
import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ultimate_master_orchestrator import (
    UltimateMasterOrchestrator,
    get_ultimate_orchestrator,
    TradingSignal,
    ComponentHealth,
    SystemMetrics,
    SignalType,
    ExecutionPriority,
    ComponentStatus,
    SignalFusionEngine,
    PerformanceOptimizer,
    AdvancedHealthMonitor,
    ExecutionCoordinator
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSignalFusionEngine:
    """Test suite for the Signal Fusion Engine"""
    
    @pytest.fixture
    def fusion_engine(self):
        return SignalFusionEngine()
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals for testing"""
        signals = []
        
        # Arbitrage signals (high confidence)
        signals.append(TradingSignal(
            signal_id="arb_001",
            signal_type=SignalType.ARBITRAGE,
            asset="BTC/USDT",
            action="buy",
            confidence=0.9,
            expected_profit=2.5,
            risk_score=0.1,
            urgency=ExecutionPriority.CRITICAL,
            timestamp=datetime.now(),
            source_component="arbitrage_detector",
            metadata={"exchange_spread": 2.3}
        ))
        
        # Momentum signals
        signals.append(TradingSignal(
            signal_id="mom_001",
            signal_type=SignalType.MOMENTUM,
            asset="BTC/USDT",
            action="buy",
            confidence=0.8,
            expected_profit=1.8,
            risk_score=0.2,
            urgency=ExecutionPriority.HIGH,
            timestamp=datetime.now(),
            source_component="momentum_strategy",
            metadata={"trend_strength": 0.85}
        ))
        
        # Conflicting signal
        signals.append(TradingSignal(
            signal_id="rev_001",
            signal_type=SignalType.MEAN_REVERSION,
            asset="BTC/USDT",
            action="sell",
            confidence=0.6,
            expected_profit=1.2,
            risk_score=0.3,
            urgency=ExecutionPriority.MEDIUM,
            timestamp=datetime.now(),
            source_component="reversion_strategy",
            metadata={"overbought_level": 0.75}
        ))
        
        return signals
    
    @pytest.mark.asyncio
    async def test_signal_fusion_basic(self, fusion_engine, sample_signals):
        """Test basic signal fusion functionality"""
        fused_signal = await fusion_engine.fuse_signals(sample_signals)
        
        assert fused_signal is not None
        assert fused_signal.asset == "BTC/USDT"
        assert fused_signal.action in ["buy", "sell", "hold"]
        assert 0 <= fused_signal.confidence <= 1
        assert fused_signal.signal_type == SignalType.ARBITRAGE  # Default for fused
        assert "input_signals" in fused_signal.metadata
        
        logger.info(f"✅ Fused signal: {fused_signal.action} {fused_signal.asset} "
                   f"(confidence: {fused_signal.confidence:.2f}, profit: {fused_signal.expected_profit:.2f}%)")
    
    @pytest.mark.asyncio
    async def test_signal_fusion_consensus_threshold(self, fusion_engine):
        """Test that consensus threshold is properly enforced"""
        # Create signals with low consensus
        conflicting_signals = [
            TradingSignal(
                signal_id=f"conflict_{i}",
                signal_type=SignalType.SENTIMENT,
                asset="ETH/USDT",
                action="buy" if i % 2 == 0 else "sell",
                confidence=0.5,
                expected_profit=1.0,
                risk_score=0.4,
                urgency=ExecutionPriority.LOW,
                timestamp=datetime.now(),
                source_component=f"strategy_{i}",
                metadata={}
            ) for i in range(4)
        ]
        
        fused_signal = await fusion_engine.fuse_signals(conflicting_signals)
        
        # Should return None due to lack of consensus
        assert fused_signal is None
        logger.info("✅ Consensus threshold properly enforced - no signal generated")
    
    @pytest.mark.asyncio
    async def test_signal_fusion_performance(self, fusion_engine):
        """Test signal fusion performance under load"""
        # Generate large number of signals
        signals = []
        for i in range(100):
            signals.append(TradingSignal(
                signal_id=f"perf_{i}",
                signal_type=SignalType.ARBITRAGE,
                asset=f"PAIR_{i % 10}",
                action="buy",
                confidence=np.random.uniform(0.6, 0.9),
                expected_profit=np.random.uniform(1.0, 3.0),
                risk_score=np.random.uniform(0.1, 0.3),
                urgency=ExecutionPriority.HIGH,
                timestamp=datetime.now(),
                source_component="perf_test",
                metadata={}
            ))
        
        start_time = time.time()
        fused_signal = await fusion_engine.fuse_signals(signals)
        fusion_time = (time.time() - start_time) * 1000
        
        assert fusion_time < 100  # Should complete in under 100ms
        logger.info(f"✅ Fusion performance test: {len(signals)} signals processed in {fusion_time:.2f}ms")

class TestPerformanceOptimizer:
    """Test suite for the Performance Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer()
    
    @pytest.fixture
    def mock_components(self):
        """Create mock component health data"""
        return {
            "strategy_engine": ComponentHealth(
                component_name="strategy_engine",
                status=ComponentStatus.HEALTHY,
                cpu_usage=45.0,
                memory_usage=60.0,
                latency_ms=25.0,
                error_rate=0.01,
                uptime_seconds=3600,
                last_heartbeat=datetime.now(),
                performance_score=0.85,
                alerts=[]
            ),
            "data_fetcher": ComponentHealth(
                component_name="data_fetcher",
                status=ComponentStatus.WARNING,
                cpu_usage=80.0,  # High CPU
                memory_usage=85.0,  # High memory
                latency_ms=150.0,  # High latency
                error_rate=0.03,
                uptime_seconds=3600,
                last_heartbeat=datetime.now(),
                performance_score=0.45,
                alerts=["High CPU usage", "High memory usage"]
            )
        }
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, optimizer, mock_components):
        """Test performance analysis functionality"""
        analysis = await optimizer._analyze_performance(mock_components)
        
        assert "overall_health" in analysis
        assert "avg_cpu" in analysis
        assert "avg_memory" in analysis
        assert "avg_latency" in analysis
        assert "bottlenecks" in analysis
        
        assert 0 <= analysis["overall_health"] <= 100
        assert len(analysis["bottlenecks"]) > 0  # Should detect bottlenecks
        
        logger.info(f"✅ Performance analysis: {analysis['overall_health']:.1f}% health, "
                   f"{len(analysis['bottlenecks'])} bottlenecks detected")
    
    @pytest.mark.asyncio
    async def test_optimization_execution(self, optimizer, mock_components):
        """Test optimization strategy execution"""
        results = await optimizer.optimize_performance(mock_components)
        
        assert "applied_optimizations" in results
        assert "improvement_percentage" in results
        assert isinstance(results["applied_optimizations"], list)
        assert results["improvement_percentage"] >= 0
        
        logger.info(f"✅ Optimization executed: {len(results['applied_optimizations'])} strategies applied, "
                   f"{results['improvement_percentage']}% improvement")

class TestAdvancedHealthMonitor:
    """Test suite for the Advanced Health Monitor"""
    
    @pytest.fixture
    def health_monitor(self):
        return AdvancedHealthMonitor()
    
    @pytest.fixture
    def mock_health_callback(self):
        """Mock health callback that returns varying health data"""
        async def health_callback():
            return {
                'cpu_usage': np.random.uniform(10, 50),
                'memory_usage': np.random.uniform(20, 60),
                'error_rate': np.random.uniform(0, 0.02),
                'uptime_seconds': 3600
            }
        return health_callback
    
    @pytest.mark.asyncio
    async def test_component_registration(self, health_monitor, mock_health_callback):
        """Test component registration functionality"""
        success = await health_monitor.register_component("test_component", mock_health_callback)
        
        assert success is True
        assert "test_component" in health_monitor.component_registry
        
        logger.info("✅ Component registration successful")
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, health_monitor, mock_health_callback):
        """Test health check execution"""
        await health_monitor.register_component("test_component", mock_health_callback)
        
        health_results = await health_monitor.check_all_components()
        
        assert "test_component" in health_results
        component_health = health_results["test_component"]
        
        assert isinstance(component_health, ComponentHealth)
        assert component_health.component_name == "test_component"
        assert component_health.status in [status for status in ComponentStatus]
        assert 0 <= component_health.performance_score <= 1
        
        logger.info(f"✅ Health check completed: {component_health.status.value}, "
                   f"score: {component_health.performance_score:.2f}")

class TestExecutionCoordinator:
    """Test suite for the Execution Coordinator"""
    
    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()
    
    @pytest.fixture
    def sample_signal(self):
        return TradingSignal(
            signal_id="exec_test_001",
            signal_type=SignalType.ARBITRAGE,
            asset="BTC/USDT",
            action="buy",
            confidence=0.85,
            expected_profit=2.0,
            risk_score=0.15,
            urgency=ExecutionPriority.HIGH,
            timestamp=datetime.now(),
            source_component="test_strategy",
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_signal_execution(self, coordinator, sample_signal):
        """Test basic signal execution"""
        result = await coordinator.execute_signal(sample_signal)
        
        assert "success" in result
        assert "execution_id" in result
        assert isinstance(result["success"], bool)
        
        if result["success"]:
            assert "profit_realized" in result
            assert "execution_price" in result
            logger.info(f"✅ Execution successful: ${result['profit_realized']:.2f} profit")
        else:
            assert "error" in result
            logger.info(f"✅ Execution failed as expected: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_execution_performance(self, coordinator):
        """Test execution performance under load"""
        signals = []
        for i in range(10):
            signals.append(TradingSignal(
                signal_id=f"perf_exec_{i}",
                signal_type=SignalType.ARBITRAGE,
                asset="BTC/USDT",
                action="buy",
                confidence=0.8,
                expected_profit=1.5,
                risk_score=0.2,
                urgency=ExecutionPriority.HIGH,
                timestamp=datetime.now(),
                source_component="perf_test",
                metadata={}
            ))
        
        start_time = time.time()
        results = await asyncio.gather(*[coordinator.execute_signal(signal) for signal in signals])
        execution_time = (time.time() - start_time) * 1000
        
        successful_executions = sum(1 for result in results if result.get("success"))
        
        assert len(results) == len(signals)
        assert execution_time < 1000  # Should complete in under 1 second
        
        logger.info(f"✅ Execution performance: {len(signals)} signals in {execution_time:.2f}ms, "
                   f"{successful_executions} successful")

class TestUltimateMasterOrchestrator:
    """Integration tests for the Ultimate Master Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        return UltimateMasterOrchestrator()
    
    @pytest.fixture
    def mock_health_callback(self):
        async def health_callback():
            return {
                'cpu_usage': np.random.uniform(20, 40),
                'memory_usage': np.random.uniform(30, 50),
                'error_rate': np.random.uniform(0, 0.01),
                'uptime_seconds': 3600
            }
        return health_callback
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.version == "1.0.0"
        assert orchestrator.running is False
        assert orchestrator.cycle_count == 0
        assert orchestrator.total_profit == 0.0
        
        logger.info("✅ Orchestrator initialization successful")
    
    @pytest.mark.asyncio
    async def test_component_registration(self, orchestrator, mock_health_callback):
        """Test component registration with orchestrator"""
        success = await orchestrator.register_component(
            "test_strategy",
            health_callback=mock_health_callback
        )
        
        assert success is True
        assert "test_strategy" in orchestrator.registered_components
        
        logger.info("✅ Component registration with orchestrator successful")
    
    @pytest.mark.asyncio
    async def test_orchestration_lifecycle(self, orchestrator, mock_health_callback):
        """Test full orchestration lifecycle"""
        # Register component
        await orchestrator.register_component("test_strategy", health_callback=mock_health_callback)
        
        # Start orchestration
        start_success = await orchestrator.start_orchestration()
        assert start_success is True
        assert orchestrator.running is True
        
        # Let it run briefly
        await asyncio.sleep(1)
        
        # Submit test signals
        for i in range(3):
            signal = TradingSignal(
                signal_id=f"lifecycle_test_{i}",
                signal_type=SignalType.ARBITRAGE,
                asset="BTC/USDT",
                action="buy",
                confidence=0.8,
                expected_profit=1.5,
                risk_score=0.2,
                urgency=ExecutionPriority.HIGH,
                timestamp=datetime.now(),
                source_component="test_strategy",
                metadata={"test": True}
            )
            
            submit_success = await orchestrator.submit_signal(signal)
            assert submit_success is True
        
        # Let signals process
        await asyncio.sleep(2)
        
        # Check system status
        status = orchestrator.get_system_status()
        assert "orchestrator" in status
        assert "performance_metrics" in status
        assert "component_health" in status
        
        # Stop orchestration
        stop_success = await orchestrator.stop_orchestration()
        assert stop_success is True
        assert orchestrator.running is False
        
        logger.info(f"✅ Full orchestration lifecycle completed successfully")
        logger.info(f"   - Cycles executed: {orchestrator.cycle_count}")
        logger.info(f"   - Signals processed: {len(orchestrator.processed_signals)}")
        logger.info(f"   - Total profit: ${orchestrator.total_profit:.2f}")

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_signal_processing_throughput(self):
        """Test signal processing throughput"""
        orchestrator = UltimateMasterOrchestrator()
        
        # Create large batch of signals
        signals = []
        for i in range(1000):
            signals.append(TradingSignal(
                signal_id=f"throughput_{i}",
                signal_type=SignalType.ARBITRAGE,
                asset=f"PAIR_{i % 20}",
                action="buy",
                confidence=np.random.uniform(0.7, 0.9),
                expected_profit=np.random.uniform(1.0, 3.0),
                risk_score=np.random.uniform(0.1, 0.3),
                urgency=ExecutionPriority.HIGH,
                timestamp=datetime.now(),
                source_component="throughput_test",
                metadata={}
            ))
        
        # Process signals through fusion engine
        start_time = time.time()
        
        # Process in batches to simulate real-world usage
        batch_size = 50
        fused_signals = []
        
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i+batch_size]
            fused_signal = await orchestrator.signal_fusion_engine.fuse_signals(batch)
            if fused_signal:
                fused_signals.append(fused_signal)
        
        processing_time = time.time() - start_time
        throughput = len(signals) / processing_time
        
        assert throughput > 100  # Should process at least 100 signals per second
        
        logger.info(f"✅ Signal processing throughput: {throughput:.1f} signals/second")
        logger.info(f"   - Processed {len(signals)} signals in {processing_time:.2f} seconds")
        logger.info(f"   - Generated {len(fused_signals)} fused signals")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        orchestrator = UltimateMasterOrchestrator()
        
        # Process many signals to test memory usage
        for batch in range(10):
            signals = []
            for i in range(500):
                signals.append(TradingSignal(
                    signal_id=f"memory_{batch}_{i}",
                    signal_type=SignalType.ARBITRAGE,
                    asset="BTC/USDT",
                    action="buy",
                    confidence=0.8,
                    expected_profit=1.5,
                    risk_score=0.2,
                    urgency=ExecutionPriority.HIGH,
                    timestamp=datetime.now(),
                    source_component="memory_test",
                    metadata={"batch": batch, "index": i}
                ))
            
            # Process batch
            await orchestrator.signal_fusion_engine.fuse_signals(signals)
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Should not increase by more than 100MB
        
        logger.info(f"✅ Memory efficiency test completed")
        logger.info(f"   - Initial memory: {initial_memory:.1f} MB")
        logger.info(f"   - Final memory: {final_memory:.1f} MB")
        logger.info(f"   - Memory increase: {memory_increase:.1f} MB")

# Utility functions for running tests
def run_all_tests():
    """Run all test suites"""
    logger.info("🚀 Starting Ultimate Master Orchestrator Test Suite")
    logger.info("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

if __name__ == "__main__":
    run_all_tests()

