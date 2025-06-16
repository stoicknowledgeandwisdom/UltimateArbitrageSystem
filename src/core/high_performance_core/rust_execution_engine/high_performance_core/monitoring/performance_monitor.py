#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Low Latency Performance Monitor
====================================

Comprehensive performance monitoring system for the high-performance trading engine.
Monitors and profiles:
- Latency (p50, p95, p99, p99.9)
- Throughput (messages/sec)
- Memory usage and allocation patterns
- CPU utilization and thermal throttling
- Network latency and packet loss
- Disk I/O and storage performance
- NUMA node utilization

Integrates with:
- Prometheus for metrics collection
- Grafana for visualization
- perf for CPU profiling
- valgrind for memory profiling
- flamegraph for performance visualization
"""

import asyncio
import logging
import time
import psutil
import subprocess
import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

# Prometheus client
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    start_http_server
)

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Latency measurement statistics"""
    min_ns: int = 0
    max_ns: int = 0
    mean_ns: float = 0.0
    p50_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0
    p999_ns: int = 0
    samples: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ThroughputMetrics:
    """Throughput measurement statistics"""
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    peak_throughput: float = 0.0
    average_throughput: float = 0.0
    samples: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemMetrics:
    """System resource utilization metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_bytes: int = 0
    memory_available_bytes: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    load_average: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class PerformanceCollector:
    """
    High-frequency performance data collector
    
    Collects performance metrics at microsecond precision for
    ultra-low latency monitoring.
    """
    
    def __init__(self, max_samples: int = 1000000):
        self.max_samples = max_samples
        self.latency_samples = deque(maxlen=max_samples)
        self.throughput_samples = deque(maxlen=max_samples)
        self.system_samples = deque(maxlen=max_samples)
        
        # Thread-safe locks
        self.latency_lock = threading.Lock()
        self.throughput_lock = threading.Lock()
        self.system_lock = threading.Lock()
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        
        # Latency histograms with microsecond buckets
        self.latency_histogram = Histogram(
            'trading_latency_microseconds',
            'Trading operation latency in microseconds',
            buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
            registry=self.registry
        )
        
        # Throughput metrics
        self.throughput_gauge = Gauge(
            'trading_throughput_messages_per_second',
            'Trading system throughput in messages per second',
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.network_throughput = Gauge(
            'system_network_throughput_bytes_per_second',
            'Network throughput in bytes per second',
            ['direction'],
            registry=self.registry
        )
        
        # Trading-specific metrics
        self.orders_processed = Counter(
            'trading_orders_processed_total',
            'Total number of orders processed',
            ['status'],
            registry=self.registry
        )
        
        self.market_data_events = Counter(
            'trading_market_data_events_total',
            'Total market data events processed',
            ['symbol', 'event_type'],
            registry=self.registry
        )
        
        logger.info("Performance collector initialized")
    
    def record_latency(self, latency_ns: int, operation: str = "default"):
        """Record latency measurement in nanoseconds"""
        latency_us = latency_ns / 1000.0  # Convert to microseconds
        
        with self.latency_lock:
            self.latency_samples.append((time.time(), latency_ns, operation))
        
        # Update Prometheus histogram
        self.latency_histogram.observe(latency_us)
        
        # Log extreme latencies
        if latency_us > 100:  # > 100 microseconds
            logger.warning(f"High latency detected: {latency_us:.2f}μs for {operation}")
    
    def record_throughput(self, messages: int, duration_s: float):
        """Record throughput measurement"""
        throughput = messages / duration_s if duration_s > 0 else 0
        
        with self.throughput_lock:
            self.throughput_samples.append((time.time(), throughput))
        
        # Update Prometheus gauge
        self.throughput_gauge.set(throughput)
        
        # Check if we're meeting throughput targets
        if throughput < 1_500_000:  # Below 1.5M msg/s
            logger.warning(f"Low throughput detected: {throughput:.0f} msg/s")
    
    def record_system_metrics(self):
        """Record current system resource utilization"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Disk metrics
            disk_io = psutil.disk_io_counters()
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_bytes=memory.used,
                memory_available_bytes=memory.available,
                network_bytes_sent=network_io.bytes_sent,
                network_bytes_recv=network_io.bytes_recv,
                disk_read_bytes=disk_io.read_bytes,
                disk_write_bytes=disk_io.write_bytes,
                load_average=list(load_avg)
            )
            
            with self.system_lock:
                self.system_samples.append(metrics)
            
            # Update Prometheus metrics
            self.cpu_usage.set(cpu_percent)
            self.memory_usage.set(memory.used)
            self.network_throughput.labels(direction='sent').set(network_io.bytes_sent)
            self.network_throughput.labels(direction='recv').set(network_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_latency_stats(self, window_seconds: int = 60) -> LatencyMetrics:
        """Get latency statistics for the specified time window"""
        cutoff_time = time.time() - window_seconds
        
        with self.latency_lock:
            recent_samples = [
                sample[1] for sample in self.latency_samples 
                if sample[0] >= cutoff_time
            ]
        
        if not recent_samples:
            return LatencyMetrics()
        
        samples_array = np.array(recent_samples)
        
        return LatencyMetrics(
            min_ns=int(np.min(samples_array)),
            max_ns=int(np.max(samples_array)),
            mean_ns=float(np.mean(samples_array)),
            p50_ns=int(np.percentile(samples_array, 50)),
            p95_ns=int(np.percentile(samples_array, 95)),
            p99_ns=int(np.percentile(samples_array, 99)),
            p999_ns=int(np.percentile(samples_array, 99.9)),
            samples=len(recent_samples)
        )
    
    def get_throughput_stats(self, window_seconds: int = 60) -> ThroughputMetrics:
        """Get throughput statistics for the specified time window"""
        cutoff_time = time.time() - window_seconds
        
        with self.throughput_lock:
            recent_samples = [
                sample[1] for sample in self.throughput_samples 
                if sample[0] >= cutoff_time
            ]
        
        if not recent_samples:
            return ThroughputMetrics()
        
        return ThroughputMetrics(
            messages_per_second=recent_samples[-1] if recent_samples else 0,
            peak_throughput=max(recent_samples),
            average_throughput=np.mean(recent_samples),
            samples=len(recent_samples)
        )
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class SystemProfiler:
    """
    Advanced system profiling using perf, valgrind, and other tools
    """
    
    def __init__(self, output_dir: str = "./profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"System profiler initialized, output: {self.output_dir}")
    
    async def profile_cpu_with_perf(self, duration: int = 30, pid: Optional[int] = None) -> str:
        """Profile CPU usage using perf"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"perf_profile_{timestamp}.data"
        
        cmd = ["perf", "record", "-g", "-o", str(output_file)]
        
        if pid:
            cmd.extend(["-p", str(pid)])
        else:
            cmd.extend(["-a"])  # All processes
        
        cmd.extend(["sleep", str(duration)])
        
        try:
            logger.info(f"Starting CPU profiling with perf for {duration}s")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"CPU profile saved to {output_file}")
                return str(output_file)
            else:
                logger.error(f"perf failed: {stderr.decode()}")
                return ""
                
        except Exception as e:
            logger.error(f"Error running perf: {e}")
            return ""
    
    async def generate_flamegraph(self, perf_data_file: str) -> str:
        """Generate flamegraph from perf data"""
        if not perf_data_file or not Path(perf_data_file).exists():
            logger.error("Invalid perf data file")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        flamegraph_file = self.output_dir / f"flamegraph_{timestamp}.svg"
        
        try:
            # Convert perf data to folded format
            cmd1 = ["perf", "script", "-i", perf_data_file]
            cmd2 = ["stackcollapse-perf.pl"]
            cmd3 = ["flamegraph.pl"]
            
            logger.info("Generating flamegraph from perf data")
            
            # Run the pipeline: perf script | stackcollapse | flamegraph
            proc1 = await asyncio.create_subprocess_exec(
                *cmd1, stdout=asyncio.subprocess.PIPE
            )
            
            proc2 = await asyncio.create_subprocess_exec(
                *cmd2, 
                stdin=proc1.stdout,
                stdout=asyncio.subprocess.PIPE
            )
            
            with open(flamegraph_file, 'w') as f:
                proc3 = await asyncio.create_subprocess_exec(
                    *cmd3,
                    stdin=proc2.stdout,
                    stdout=f
                )
                
                await proc3.communicate()
            
            logger.info(f"Flamegraph saved to {flamegraph_file}")
            return str(flamegraph_file)
            
        except Exception as e:
            logger.error(f"Error generating flamegraph: {e}")
            return ""
    
    async def profile_memory_with_valgrind(self, command: List[str], duration: int = 60) -> str:
        """Profile memory usage with valgrind massif"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"massif_{timestamp}.out"
        
        cmd = [
            "valgrind",
            "--tool=massif",
            f"--massif-out-file={output_file}",
            "--time-unit=B",  # Use bytes allocated as time unit
            "--detailed-freq=1"
        ] + command
        
        try:
            logger.info(f"Starting memory profiling with valgrind massif")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Let it run for specified duration
            await asyncio.sleep(duration)
            
            # Terminate the process
            process.terminate()
            await process.communicate()
            
            if output_file.exists():
                logger.info(f"Memory profile saved to {output_file}")
                return str(output_file)
            else:
                logger.error("Memory profiling failed")
                return ""
                
        except Exception as e:
            logger.error(f"Error running valgrind: {e}")
            return ""
    
    def analyze_numa_topology(self) -> Dict[str, Any]:
        """Analyze NUMA topology and memory allocation"""
        try:
            # Get NUMA topology
            result = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True,
                text=True
            )
            
            numa_info = {
                "topology": result.stdout if result.returncode == 0 else "unavailable",
                "nodes": [],
                "memory_policy": ""
            }
            
            # Get current memory policy
            result = subprocess.run(
                ["numactl", "--show"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                numa_info["memory_policy"] = result.stdout
            
            # Get per-node memory statistics
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                numa_info["system_memory"] = meminfo
            except Exception:
                pass
            
            return numa_info
            
        except Exception as e:
            logger.error(f"Error analyzing NUMA topology: {e}")
            return {"error": str(e)}

class PerformanceMonitor:
    """
    Main performance monitoring orchestrator
    
    Coordinates all performance monitoring activities and provides
    a unified interface for performance analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collector = PerformanceCollector()
        self.profiler = SystemProfiler(config.get('profile_output_dir', './profiles'))
        
        # Monitoring state
        self.is_running = False
        self.monitoring_tasks = []
        
        # Performance targets
        self.latency_target_ns = config.get('latency_target_ns', 10_000)  # 10μs
        self.throughput_target = config.get('throughput_target', 2_000_000)  # 2M msg/s
        
        logger.info("Performance monitor initialized")
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        logger.info("Starting performance monitoring")
        
        # Start Prometheus metrics server
        prometheus_port = self.config.get('prometheus_port', 9091)
        start_http_server(prometheus_port, registry=self.collector.registry)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
        ]
        
        # Start periodic profiling if enabled
        if self.config.get('enable_periodic_profiling', False):
            self.monitoring_tasks.append(
                asyncio.create_task(self._periodic_profiling_loop())
            )
        
        logger.info("All monitoring tasks started")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping performance monitoring")
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Performance monitoring stopped")
    
    async def _system_metrics_loop(self):
        """Continuously collect system metrics"""
        interval = self.config.get('system_metrics_interval', 1.0)  # 1 second
        
        while self.is_running:
            try:
                self.collector.record_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
                await asyncio.sleep(1)
    
    async def _performance_analysis_loop(self):
        """Continuously analyze performance metrics"""
        interval = self.config.get('analysis_interval', 10.0)  # 10 seconds
        
        while self.is_running:
            try:
                await self._analyze_current_performance()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}")
                await asyncio.sleep(1)
    
    async def _alert_monitoring_loop(self):
        """Monitor for performance alerts"""
        interval = self.config.get('alert_interval', 5.0)  # 5 seconds
        
        while self.is_running:
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _periodic_profiling_loop(self):
        """Perform periodic system profiling"""
        interval = self.config.get('profiling_interval', 300)  # 5 minutes
        
        while self.is_running:
            try:
                logger.info("Starting periodic profiling")
                
                # CPU profiling
                perf_file = await self.profiler.profile_cpu_with_perf(duration=30)
                if perf_file:
                    await self.profiler.generate_flamegraph(perf_file)
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in periodic profiling loop: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_current_performance(self):
        """Analyze current performance and log insights"""
        latency_stats = self.collector.get_latency_stats(window_seconds=60)
        throughput_stats = self.collector.get_throughput_stats(window_seconds=60)
        
        if latency_stats.samples > 0:
            logger.info(
                f"Latency (1m): p50={latency_stats.p50_ns/1000:.1f}μs, "
                f"p99={latency_stats.p99_ns/1000:.1f}μs, "
                f"samples={latency_stats.samples}"
            )
            
            # Check latency targets
            if latency_stats.p99_ns > self.latency_target_ns:
                logger.warning(
                    f"P99 latency ({latency_stats.p99_ns/1000:.1f}μs) "
                    f"exceeds target ({self.latency_target_ns/1000:.1f}μs)"
                )
        
        if throughput_stats.samples > 0:
            logger.info(
                f"Throughput (1m): current={throughput_stats.messages_per_second:.0f} msg/s, "
                f"peak={throughput_stats.peak_throughput:.0f} msg/s"
            )
            
            # Check throughput targets
            if throughput_stats.messages_per_second < self.throughput_target:
                logger.warning(
                    f"Throughput ({throughput_stats.messages_per_second:.0f} msg/s) "
                    f"below target ({self.throughput_target:.0f} msg/s)"
                )
    
    async def _check_performance_alerts(self):
        """Check for performance alert conditions"""
        latency_stats = self.collector.get_latency_stats(window_seconds=30)
        throughput_stats = self.collector.get_throughput_stats(window_seconds=30)
        
        alerts = []
        
        # Critical latency alert
        if latency_stats.p99_ns > self.latency_target_ns * 5:  # 5x target
            alerts.append({
                'type': 'CRITICAL_LATENCY',
                'message': f'P99 latency {latency_stats.p99_ns/1000:.1f}μs is 5x target',
                'value': latency_stats.p99_ns
            })
        
        # Critical throughput alert
        if throughput_stats.messages_per_second < self.throughput_target * 0.5:  # 50% of target
            alerts.append({
                'type': 'CRITICAL_THROUGHPUT',
                'message': f'Throughput {throughput_stats.messages_per_second:.0f} msg/s is 50% below target',
                'value': throughput_stats.messages_per_second
            })
        
        # System resource alerts
        if hasattr(self.collector, 'system_samples') and self.collector.system_samples:
            latest_system = self.collector.system_samples[-1]
            
            if latest_system.cpu_percent > 90:
                alerts.append({
                    'type': 'HIGH_CPU',
                    'message': f'CPU usage {latest_system.cpu_percent:.1f}% is critically high',
                    'value': latest_system.cpu_percent
                })
            
            if latest_system.memory_percent > 95:
                alerts.append({
                    'type': 'HIGH_MEMORY',
                    'message': f'Memory usage {latest_system.memory_percent:.1f}% is critically high',
                    'value': latest_system.memory_percent
                })
        
        # Send alerts if any
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send performance alert"""
        logger.error(f"PERFORMANCE ALERT [{alert['type']}]: {alert['message']}")
        
        # Here you would integrate with your alerting system
        # (Slack, PagerDuty, email, etc.)
        
        # For now, just log the alert
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert': alert,
            'system': 'ultra-trading-engine'
        }
        
        # Save alert to file for external processing
        alert_file = Path("./alerts") / f"alert_{int(time.time())}.json"
        alert_file.parent.mkdir(exist_ok=True)
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        window_seconds = hours * 3600
        
        latency_stats = self.collector.get_latency_stats(window_seconds)
        throughput_stats = self.collector.get_throughput_stats(window_seconds)
        numa_info = self.profiler.analyze_numa_topology()
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'time_window_hours': hours,
            'performance_targets': {
                'latency_target_ns': self.latency_target_ns,
                'throughput_target_msg_per_s': self.throughput_target
            },
            'latency_analysis': {
                'p50_ns': latency_stats.p50_ns,
                'p95_ns': latency_stats.p95_ns,
                'p99_ns': latency_stats.p99_ns,
                'p999_ns': latency_stats.p999_ns,
                'samples': latency_stats.samples,
                'target_met': latency_stats.p99_ns <= self.latency_target_ns
            },
            'throughput_analysis': {
                'current_msg_per_s': throughput_stats.messages_per_second,
                'peak_msg_per_s': throughput_stats.peak_throughput,
                'average_msg_per_s': throughput_stats.average_throughput,
                'samples': throughput_stats.samples,
                'target_met': throughput_stats.messages_per_second >= self.throughput_target
            },
            'system_info': {
                'numa_topology': numa_info
            }
        }
        
        return report
    
    async def run_comprehensive_profile(self, duration: int = 300) -> Dict[str, str]:
        """Run comprehensive system profiling"""
        logger.info(f"Starting comprehensive profiling for {duration} seconds")
        
        results = {}
        
        # CPU profiling with perf
        perf_file = await self.profiler.profile_cpu_with_perf(duration=duration)
        if perf_file:
            results['perf_data'] = perf_file
            
            # Generate flamegraph
            flamegraph_file = await self.profiler.generate_flamegraph(perf_file)
            if flamegraph_file:
                results['flamegraph'] = flamegraph_file
        
        # Memory profiling would be done on the Rust process
        # This is a placeholder for integration with the Rust engine
        results['memory_profile'] = "Not implemented - requires Rust process integration"
        
        logger.info(f"Comprehensive profiling complete: {results}")
        return results

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configuration
        config = {
            'prometheus_port': 9091,
            'latency_target_ns': 10_000,  # 10 microseconds
            'throughput_target': 2_000_000,  # 2M messages/second
            'system_metrics_interval': 1.0,
            'analysis_interval': 10.0,
            'alert_interval': 5.0,
            'enable_periodic_profiling': True,
            'profiling_interval': 300,
            'profile_output_dir': './profiles'
        }
        
        # Initialize monitor
        monitor = PerformanceMonitor(config)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate some performance data
        logger.info("Simulating performance data...")
        
        for i in range(100):
            # Simulate varying latencies
            latency_ns = np.random.gamma(2, 5000) + 1000  # Gamma distribution
            monitor.collector.record_latency(int(latency_ns), "test_operation")
            
            # Simulate throughput
            throughput = np.random.normal(1_800_000, 200_000)  # Around 1.8M with variance
            monitor.collector.record_throughput(int(throughput), 1.0)
            
            await asyncio.sleep(0.1)
        
        # Generate report
        report = monitor.generate_performance_report(hours=1)
        logger.info(f"Performance report: {json.dumps(report, indent=2)}")
        
        # Run for a while
        await asyncio.sleep(30)
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(main())

