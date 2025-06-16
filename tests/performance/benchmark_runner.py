"""Performance Benchmarking Framework

Integrated k6, Locust, and profiling for comprehensive performance testing.
"""

import asyncio
import json
import subprocess
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import statistics
import numpy as np
import pandas as pd
from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import git
import matplotlib.pyplot as plt
import seaborn as sns
from py_spy import SpyProfiler
import memory_profiler
from concurrent.futures import ThreadPoolExecutor
import yaml
import requests


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    name: str
    description: str
    target_url: str
    
    # Load testing parameters
    users: int = 10
    spawn_rate: float = 2.0
    duration_seconds: int = 300
    
    # K6 specific
    k6_script_path: Optional[str] = None
    k6_options: Dict[str, Any] = field(default_factory=dict)
    
    # Locust specific  
    locust_script_path: Optional[str] = None
    locust_host: str = "http://localhost:8000"
    
    # Performance thresholds
    max_response_time_ms: int = 1000
    max_error_rate: float = 0.01  # 1%
    min_rps: float = 100.0
    
    # Profiling
    enable_profiling: bool = True
    profile_duration_seconds: int = 60
    
    # Environment
    warmup_seconds: int = 30
    cooldown_seconds: int = 30


@dataclass
class BenchmarkResult:
    """Results from performance benchmark"""
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    
    # Performance metrics
    total_requests: int
    failed_requests: int
    average_response_time: float
    percentile_95_response_time: float
    percentile_99_response_time: float
    requests_per_second: float
    error_rate: float
    
    # System metrics
    cpu_usage_percent: List[float]
    memory_usage_mb: List[float]
    
    # Detailed data
    response_times: List[float]
    timestamps: List[datetime]
    
    # Profiling data
    flame_graph_path: Optional[str] = None
    memory_profile_path: Optional[str] = None
    
    # Threshold violations
    threshold_violations: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return len(self.threshold_violations) == 0


class K6Runner:
    """K6 load testing runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_k6_script(self, config: BenchmarkConfig) -> str:
        """Generate K6 test script"""
        script_content = f'''
import http from 'k6/http';
import {{ check, sleep }} from 'k6';
import {{ Rate }} from 'k6/metrics';

export let errorRate = new Rate('errors');

export let options = {{
  stages: [
    {{ duration: '30s', target: {config.users // 3} }}, // Ramp up
    {{ duration: '{config.duration_seconds - 60}s', target: {config.users} }}, // Stay at load
    {{ duration: '30s', target: 0 }}, // Ramp down
  ],
  thresholds: {{
    'http_req_duration': ['p(95)<{config.max_response_time_ms}'],
    'http_req_failed': ['rate<{config.max_error_rate}'],
    'errors': ['rate<{config.max_error_rate}'],
  }},
}};

export default function() {{
  // Test various endpoints
  let responses = http.batch([
    ['GET', '{config.target_url}/api/v1/health'],
    ['GET', '{config.target_url}/api/v1/portfolio'],
    ['GET', '{config.target_url}/api/v1/opportunities'],
    ['POST', '{config.target_url}/api/v1/orders', JSON.stringify({{
      symbol: 'BTC/USDT',
      side: 'buy',
      amount: 0.001,
      type: 'market'
    }}), {{
      headers: {{ 'Content-Type': 'application/json' }}
    }}],
  ]);
  
  for (let response of responses) {{
    check(response, {{
      'status is 200': (r) => r.status === 200,
      'response time < {config.max_response_time_ms}ms': (r) => r.timings.duration < {config.max_response_time_ms},
    }});
    
    errorRate.add(response.status !== 200);
  }}
  
  sleep(1);
}}
'''
        return script_content
    
    async def run_k6_test(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run K6 performance test"""
        self.logger.info(f"Starting K6 test: {config.name}")
        
        # Create script if not provided
        if config.k6_script_path:
            script_path = config.k6_script_path
        else:
            script_content = self.create_k6_script(config)
            script_path = f"temp_k6_script_{int(time.time())}.js"
            with open(script_path, 'w') as f:
                f.write(script_content)
        
        try:
            # Run K6
            cmd = [
                'k6', 'run',
                '--out', 'json=k6_results.json',
                script_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"K6 test failed: {stderr.decode()}")
            
            # Parse results
            return self._parse_k6_results('k6_results.json')
            
        finally:
            # Cleanup temporary script
            if not config.k6_script_path and Path(script_path).exists():
                Path(script_path).unlink()
    
    def _parse_k6_results(self, results_file: str) -> Dict[str, Any]:
        """Parse K6 JSON results"""
        metrics = {
            'http_req_duration': [],
            'http_reqs': 0,
            'http_req_failed': 0,
            'errors': 0
        }
        
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('type') == 'Point':
                        metric_name = data.get('metric')
                        value = data.get('data', {}).get('value', 0)
                        
                        if metric_name == 'http_req_duration':
                            metrics['http_req_duration'].append(value)
                        elif metric_name == 'http_reqs':
                            metrics['http_reqs'] += value
                        elif metric_name == 'http_req_failed':
                            metrics['http_req_failed'] += value
            
            # Calculate summary statistics
            if metrics['http_req_duration']:
                durations = metrics['http_req_duration']
                return {
                    'total_requests': metrics['http_reqs'],
                    'failed_requests': metrics['http_req_failed'],
                    'average_response_time': statistics.mean(durations),
                    'percentile_95': np.percentile(durations, 95),
                    'percentile_99': np.percentile(durations, 99),
                    'error_rate': metrics['http_req_failed'] / max(metrics['http_reqs'], 1),
                    'response_times': durations
                }
        
        except Exception as e:
            self.logger.error(f"Failed to parse K6 results: {e}")
        
        return {}


class ArbitrageLocustUser(HttpUser):
    """Locust user for arbitrage system testing"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        # Login or setup session if needed
        pass
    
    @task(3)
    def get_portfolio(self):
        """Get portfolio status"""
        with self.client.get("/api/v1/portfolio", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(5)
    def get_opportunities(self):
        """Get arbitrage opportunities"""
        with self.client.get("/api/v1/opportunities", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get('opportunities', [])) > 0:
                    response.success()
                else:
                    response.failure("No opportunities found")
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(2)
    def get_market_data(self):
        """Get market data"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        symbol = self.environment.parsed_options.symbol if hasattr(self.environment, 'parsed_options') else 'BTC/USDT'
        
        with self.client.get(f"/api/v1/market/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(1)
    def place_order(self):
        """Place test order"""
        order_data = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.001,
            'type': 'market'
        }
        
        with self.client.post("/api/v1/orders", json=order_data, catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Order failed with status {response.status_code}")


class LocustRunner:
    """Locust load testing runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def run_locust_test(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run Locust performance test"""
        self.logger.info(f"Starting Locust test: {config.name}")
        
        # Setup Locust environment
        env = Environment(
            user_classes=[ArbitrageLocustUser],
            events=None
        )
        
        # Start stats printer
        gevent_statslog = threading.Thread(target=stats_printer(env.stats))
        gevent_statslog.start()
        
        # Start stats history
        gevent_history = threading.Thread(target=stats_history, args=(env.runner,))
        gevent_history.start()
        
        try:
            # Start load test
            env.create_local_runner()
            env.runner.start(
                user_count=config.users,
                spawn_rate=config.spawn_rate,
                host=config.locust_host
            )
            
            # Wait for test duration
            await asyncio.sleep(config.duration_seconds)
            
            # Stop test
            env.runner.stop()
            
            # Collect results
            stats = env.runner.stats
            
            return {
                'total_requests': stats.total.num_requests,
                'failed_requests': stats.total.num_failures,
                'average_response_time': stats.total.avg_response_time,
                'percentile_95': stats.total.get_response_time_percentile(0.95),
                'percentile_99': stats.total.get_response_time_percentile(0.99),
                'requests_per_second': stats.total.total_rps,
                'error_rate': stats.total.fail_ratio,
                'response_times': [entry.response_time for entry in stats.entries.values()]
            }
            
        finally:
            if hasattr(env, 'runner') and env.runner:
                env.runner.quit()


class SystemMonitor:
    """System resource monitoring during tests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.cpu_data = []
        self.memory_data = []
        self.timestamps = []
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start system monitoring"""
        self.monitoring = True
        self.cpu_data = []
        self.memory_data = []
        self.timestamps = []
        
        def monitor():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / 1024 / 1024
                    
                    self.cpu_data.append(cpu_percent)
                    self.memory_data.append(memory_mb)
                    self.timestamps.append(datetime.now())
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get collected metrics"""
        return {
            'cpu_usage_percent': self.cpu_data.copy(),
            'memory_usage_mb': self.memory_data.copy(),
            'timestamps': self.timestamps.copy()
        }


class PerformanceProfiler:
    """Performance profiler using py-spy and memory_profiler"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def profile_cpu(self, target_pid: int, duration_seconds: int, output_path: str) -> str:
        """Profile CPU using py-spy"""
        try:
            cmd = [
                'py-spy', 'record',
                '-o', output_path,
                '-d', str(duration_seconds),
                '-p', str(target_pid),
                '-f', 'svg'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"CPU flame graph saved to {output_path}")
                return output_path
            else:
                self.logger.error(f"py-spy profiling failed")
                return None
                
        except Exception as e:
            self.logger.error(f"CPU profiling error: {e}")
            return None
    
    def profile_memory(self, func: Callable, output_path: str) -> str:
        """Profile memory usage"""
        try:
            @memory_profiler.profile(stream=open(output_path, 'w+'))
            def wrapped_func():
                return func()
            
            wrapped_func()
            self.logger.info(f"Memory profile saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Memory profiling error: {e}")
            return None


class PerformanceDeltaAnalyzer:
    """Analyze performance deltas between commits"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.logger = logging.getLogger(__name__)
        
        # Storage for historical results
        self.results_db_path = self.repo_path / 'performance_results.json'
        self.load_historical_results()
    
    def load_historical_results(self):
        """Load historical performance results"""
        if self.results_db_path.exists():
            with open(self.results_db_path, 'r') as f:
                self.historical_results = json.load(f)
        else:
            self.historical_results = {}
    
    def save_historical_results(self):
        """Save historical performance results"""
        with open(self.results_db_path, 'w') as f:
            json.dump(self.historical_results, f, indent=2, default=str)
    
    def record_benchmark_result(self, commit_hash: str, benchmark_name: str, result: BenchmarkResult):
        """Record benchmark result for a commit"""
        if commit_hash not in self.historical_results:
            self.historical_results[commit_hash] = {}
        
        self.historical_results[commit_hash][benchmark_name] = {
            'timestamp': result.start_time.isoformat(),
            'total_requests': result.total_requests,
            'average_response_time': result.average_response_time,
            'percentile_95_response_time': result.percentile_95_response_time,
            'requests_per_second': result.requests_per_second,
            'error_rate': result.error_rate,
            'passed': result.passed,
            'cpu_usage_avg': statistics.mean(result.cpu_usage_percent) if result.cpu_usage_percent else 0,
            'memory_usage_avg': statistics.mean(result.memory_usage_mb) if result.memory_usage_mb else 0
        }
        
        self.save_historical_results()
    
    def compare_with_baseline(self, current_result: BenchmarkResult, 
                            benchmark_name: str,
                            baseline_commits: int = 5) -> Dict[str, Any]:
        """Compare current result with baseline"""
        # Get recent commits
        recent_commits = list(self.repo.iter_commits(max_count=baseline_commits + 1))
        
        baseline_data = []
        for commit in recent_commits[1:]:  # Skip current commit
            commit_hash = commit.hexsha
            if (commit_hash in self.historical_results and 
                benchmark_name in self.historical_results[commit_hash]):
                baseline_data.append(self.historical_results[commit_hash][benchmark_name])
        
        if not baseline_data:
            return {'comparison': 'no_baseline_data'}
        
        # Calculate baseline averages
        baseline_response_time = statistics.mean([d['average_response_time'] for d in baseline_data])
        baseline_rps = statistics.mean([d['requests_per_second'] for d in baseline_data])
        baseline_error_rate = statistics.mean([d['error_rate'] for d in baseline_data])
        
        # Calculate deltas
        response_time_delta = ((current_result.average_response_time / baseline_response_time) - 1) * 100
        rps_delta = ((current_result.requests_per_second / baseline_rps) - 1) * 100
        error_rate_delta = current_result.error_rate - baseline_error_rate
        
        return {
            'comparison': 'completed',
            'baseline_commits': len(baseline_data),
            'deltas': {
                'response_time_percent': response_time_delta,
                'rps_percent': rps_delta,
                'error_rate_absolute': error_rate_delta
            },
            'baseline_metrics': {
                'average_response_time': baseline_response_time,
                'requests_per_second': baseline_rps,
                'error_rate': baseline_error_rate
            },
            'current_metrics': {
                'average_response_time': current_result.average_response_time,
                'requests_per_second': current_result.requests_per_second,
                'error_rate': current_result.error_rate
            },
            'regression_detected': (
                response_time_delta > 10 or  # 10% slower
                rps_delta < -10 or           # 10% fewer RPS
                error_rate_delta > 0.01      # 1% more errors
            )
        }


class BenchmarkRunner:
    """Main benchmark runner orchestrating all components"""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.k6_runner = K6Runner()
        self.locust_runner = LocustRunner()
        self.system_monitor = SystemMonitor()
        self.profiler = PerformanceProfiler()
        
        if repo_path:
            self.delta_analyzer = PerformanceDeltaAnalyzer(repo_path)
        else:
            self.delta_analyzer = None
    
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run complete benchmark with monitoring and profiling"""
        self.logger.info(f"Starting benchmark: {config.name}")
        start_time = datetime.now()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Warmup period
            if config.warmup_seconds > 0:
                self.logger.info(f"Warming up for {config.warmup_seconds} seconds")
                await asyncio.sleep(config.warmup_seconds)
            
            # Start profiling if enabled
            profiling_tasks = []
            if config.enable_profiling:
                # Get target process PID (simplified - would need actual process detection)
                target_pid = None
                for proc in psutil.process_iter(['pid', 'name']):
                    if 'python' in proc.info['name'].lower():
                        target_pid = proc.info['pid']
                        break
                
                if target_pid:
                    flame_graph_path = f"flame_graph_{config.name}_{int(time.time())}.svg"
                    profiling_tasks.append(
                        self.profiler.profile_cpu(target_pid, config.profile_duration_seconds, flame_graph_path)
                    )
            
            # Run load test (K6 or Locust based on configuration)
            if config.k6_script_path or not config.locust_script_path:
                load_test_results = await self.k6_runner.run_k6_test(config)
            else:
                load_test_results = await self.locust_runner.run_locust_test(config)
            
            # Wait for profiling to complete
            flame_graph_path = None
            if profiling_tasks:
                flame_graph_path = await profiling_tasks[0]
            
            # Cooldown period
            if config.cooldown_seconds > 0:
                await asyncio.sleep(config.cooldown_seconds)
            
        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()
        
        end_time = datetime.now()
        system_metrics = self.system_monitor.get_metrics()
        
        # Create result object
        result = BenchmarkResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_requests=load_test_results.get('total_requests', 0),
            failed_requests=load_test_results.get('failed_requests', 0),
            average_response_time=load_test_results.get('average_response_time', 0),
            percentile_95_response_time=load_test_results.get('percentile_95', 0),
            percentile_99_response_time=load_test_results.get('percentile_99', 0),
            requests_per_second=load_test_results.get('requests_per_second', 0),
            error_rate=load_test_results.get('error_rate', 0),
            cpu_usage_percent=system_metrics['cpu_usage_percent'],
            memory_usage_mb=system_metrics['memory_usage_mb'],
            response_times=load_test_results.get('response_times', []),
            timestamps=system_metrics['timestamps'],
            flame_graph_path=flame_graph_path
        )
        
        # Check thresholds
        self._check_thresholds(result)
        
        # Record result for delta analysis
        if self.delta_analyzer:
            current_commit = self.delta_analyzer.repo.head.commit.hexsha
            self.delta_analyzer.record_benchmark_result(current_commit, config.name, result)
        
        self.logger.info(f"Benchmark completed: {config.name} - {'PASSED' if result.passed else 'FAILED'}")
        return result
    
    def _check_thresholds(self, result: BenchmarkResult):
        """Check if result meets performance thresholds"""
        config = result.config
        violations = []
        
        if result.average_response_time > config.max_response_time_ms:
            violations.append(
                f"Average response time {result.average_response_time:.1f}ms "
                f"exceeds threshold {config.max_response_time_ms}ms"
            )
        
        if result.error_rate > config.max_error_rate:
            violations.append(
                f"Error rate {result.error_rate:.3f} exceeds threshold {config.max_error_rate:.3f}"
            )
        
        if result.requests_per_second < config.min_rps:
            violations.append(
                f"RPS {result.requests_per_second:.1f} below threshold {config.min_rps}"
            )
        
        result.threshold_violations = violations
    
    async def run_benchmark_suite(self, configs: List[BenchmarkConfig]) -> List[BenchmarkResult]:
        """Run multiple benchmarks"""
        results = []
        
        for config in configs:
            result = await self.run_benchmark(config)
            results.append(result)
            
            # Wait between benchmarks
            await asyncio.sleep(30)
        
        return results
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        
        report = {
            'summary': {
                'total_benchmarks': total_tests,
                'passed_benchmarks': passed_tests,
                'failure_rate': (total_tests - passed_tests) / total_tests if total_tests > 0 else 0,
                'execution_time': sum((r.end_time - r.start_time).total_seconds() for r in results)
            },
            'benchmark_results': [],
            'performance_trends': [],
            'recommendations': []
        }
        
        for result in results:
            benchmark_data = {
                'name': result.config.name,
                'passed': result.passed,
                'total_requests': result.total_requests,
                'average_response_time': result.average_response_time,
                'p95_response_time': result.percentile_95_response_time,
                'requests_per_second': result.requests_per_second,
                'error_rate': result.error_rate,
                'violations': result.threshold_violations
            }
            
            # Add delta analysis if available
            if self.delta_analyzer:
                comparison = self.delta_analyzer.compare_with_baseline(
                    result, result.config.name
                )
                benchmark_data['delta_analysis'] = comparison
            
            report['benchmark_results'].append(benchmark_data)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        return report
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check for consistent threshold violations
        violation_types = {}
        for result in results:
            for violation in result.threshold_violations:
                if 'response time' in violation:
                    violation_types['response_time'] = violation_types.get('response_time', 0) + 1
                elif 'error rate' in violation:
                    violation_types['error_rate'] = violation_types.get('error_rate', 0) + 1
                elif 'RPS' in violation:
                    violation_types['rps'] = violation_types.get('rps', 0) + 1
        
        for violation_type, count in violation_types.items():
            if count > 1:
                recommendations.append(
                    f"Multiple {violation_type} violations detected. "
                    f"Consider optimizing {violation_type} performance."
                )
        
        # Check resource usage
        high_cpu_tests = [r for r in results if r.cpu_usage_percent and max(r.cpu_usage_percent) > 80]
        if high_cpu_tests:
            recommendations.append(
                f"{len(high_cpu_tests)} tests had high CPU usage (>80%). "
                "Consider CPU optimization or scaling."
            )
        
        high_memory_tests = [r for r in results if r.memory_usage_mb and max(r.memory_usage_mb) > 8000]
        if high_memory_tests:
            recommendations.append(
                f"{len(high_memory_tests)} tests had high memory usage (>8GB). "
                "Consider memory optimization or scaling."
            )
        
        return recommendations
    
    def load_benchmark_configs(self, config_file: Path) -> List[BenchmarkConfig]:
        """Load benchmark configurations from YAML file"""
        with open(config_file, 'r') as f:
            configs_data = yaml.safe_load(f)
        
        configs = []
        for config_data in configs_data.get('benchmarks', []):
            config = BenchmarkConfig(
                name=config_data['name'],
                description=config_data.get('description', ''),
                target_url=config_data['target_url'],
                users=config_data.get('users', 10),
                spawn_rate=config_data.get('spawn_rate', 2.0),
                duration_seconds=config_data.get('duration_seconds', 300),
                max_response_time_ms=config_data.get('max_response_time_ms', 1000),
                max_error_rate=config_data.get('max_error_rate', 0.01),
                min_rps=config_data.get('min_rps', 100.0),
                enable_profiling=config_data.get('enable_profiling', True)
            )
            configs.append(config)
        
        return configs

