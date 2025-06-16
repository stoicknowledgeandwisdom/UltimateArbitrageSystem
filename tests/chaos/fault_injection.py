"""Chaos Engineering Fault Injection Framework

Integration with Chaos Mesh and custom fault injection for resilience testing.
"""

import asyncio
import random
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import aiohttp
import subprocess
from contextlib import asynccontextmanager
import kubernetes
from kubernetes import client, config
import numpy as np


class FaultType(Enum):
    """Types of faults that can be injected"""
    NETWORK_LATENCY = "network_latency"
    PACKET_LOSS = "packet_loss"
    API_ERROR = "api_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"
    DISK_IO = "disk_io"
    CLOCK_SKEW = "clock_skew"
    DNS_FAILURE = "dns_failure"


class SeverityLevel(Enum):
    """Severity levels for fault injection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FaultPolicy:
    """Configuration for fault injection"""
    name: str
    fault_type: FaultType
    severity: SeverityLevel
    target_services: List[str]
    parameters: Dict[str, Any]
    duration_minutes: int = 5
    probability: float = 1.0  # Probability of fault occurring (0-1)
    schedule: Optional[str] = None  # Cron schedule
    enabled: bool = True
    
    # SLO verification
    slo_checks: List[Dict[str, Any]] = field(default_factory=list)
    max_acceptable_failure_rate: float = 0.05  # 5% max failure rate
    max_response_time_ms: int = 5000


@dataclass
class FaultInjectionResult:
    """Result of fault injection test"""
    policy: FaultPolicy
    start_time: datetime
    end_time: datetime
    success: bool
    metrics: Dict[str, Any]
    slo_violations: List[Dict[str, Any]]
    error_logs: List[str]
    recovery_time_seconds: Optional[float] = None


class ChaosMeshClient:
    """Client for interacting with Chaos Mesh"""
    
    def __init__(self, namespace: str = "chaos-testing"):
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi()
    
    async def inject_network_latency(self, policy: FaultPolicy) -> str:
        """Inject network latency using Chaos Mesh"""
        chaos_spec = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {
                "name": f"latency-{policy.name}-{int(time.time())}",
                "namespace": self.namespace
            },
            "spec": {
                "action": "delay",
                "mode": "all",
                "selector": {
                    "labelSelectors": {
                        "app": policy.target_services[0]
                    }
                },
                "delay": {
                    "latency": f"{policy.parameters.get('latency_ms', 100)}ms",
                    "correlation": str(policy.parameters.get('correlation', 0)),
                    "jitter": f"{policy.parameters.get('jitter_ms', 10)}ms"
                },
                "duration": f"{policy.duration_minutes}m"
            }
        }
        
        response = self.custom_api.create_namespaced_custom_object(
            group="chaos-mesh.org",
            version="v1alpha1",
            namespace=self.namespace,
            plural="networkchaos",
            body=chaos_spec
        )
        
        return response['metadata']['name']
    
    async def inject_packet_loss(self, policy: FaultPolicy) -> str:
        """Inject packet loss using Chaos Mesh"""
        chaos_spec = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {
                "name": f"packet-loss-{policy.name}-{int(time.time())}",
                "namespace": self.namespace
            },
            "spec": {
                "action": "loss",
                "mode": "all",
                "selector": {
                    "labelSelectors": {
                        "app": policy.target_services[0]
                    }
                },
                "loss": {
                    "loss": f"{policy.parameters.get('loss_percentage', 10)}%",
                    "correlation": str(policy.parameters.get('correlation', 0))
                },
                "duration": f"{policy.duration_minutes}m"
            }
        }
        
        response = self.custom_api.create_namespaced_custom_object(
            group="chaos-mesh.org",
            version="v1alpha1",
            namespace=self.namespace,
            plural="networkchaos",
            body=chaos_spec
        )
        
        return response['metadata']['name']
    
    async def inject_pod_failure(self, policy: FaultPolicy) -> str:
        """Inject pod failures using Chaos Mesh"""
        chaos_spec = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {
                "name": f"pod-failure-{policy.name}-{int(time.time())}",
                "namespace": self.namespace
            },
            "spec": {
                "action": "pod-failure",
                "mode": "one",
                "selector": {
                    "labelSelectors": {
                        "app": policy.target_services[0]
                    }
                },
                "duration": f"{policy.duration_minutes}m"
            }
        }
        
        response = self.custom_api.create_namespaced_custom_object(
            group="chaos-mesh.org",
            version="v1alpha1",
            namespace=self.namespace,
            plural="podchaos",
            body=chaos_spec
        )
        
        return response['metadata']['name']
    
    async def cleanup_chaos(self, chaos_name: str, chaos_type: str) -> None:
        """Clean up chaos experiment"""
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural=f"{chaos_type.lower()}chaos",
                name=chaos_name
            )
        except Exception as e:
            self.logger.warning(f"Failed to cleanup chaos {chaos_name}: {e}")


class ApplicationFaultInjector:
    """Application-level fault injection without Kubernetes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_faults: Dict[str, Dict[str, Any]] = {}
        self.fault_callbacks: Dict[FaultType, List[Callable]] = {}
    
    def register_fault_handler(self, fault_type: FaultType, handler: Callable):
        """Register handler for specific fault type"""
        if fault_type not in self.fault_callbacks:
            self.fault_callbacks[fault_type] = []
        self.fault_callbacks[fault_type].append(handler)
    
    @asynccontextmanager
    async def inject_api_errors(self, policy: FaultPolicy):
        """Context manager for API error injection"""
        fault_id = f"api_error_{int(time.time())}"
        
        # Store fault configuration
        self.active_faults[fault_id] = {
            'type': FaultType.API_ERROR,
            'policy': policy,
            'start_time': datetime.now(),
            'error_rate': policy.parameters.get('error_rate', 0.1),
            'status_codes': policy.parameters.get('status_codes', [500, 502, 503]),
            'affected_endpoints': policy.parameters.get('endpoints', ['*'])
        }
        
        try:
            self.logger.info(f"Injecting API errors: {fault_id}")
            yield fault_id
        finally:
            # Cleanup
            if fault_id in self.active_faults:
                del self.active_faults[fault_id]
            self.logger.info(f"Stopped API error injection: {fault_id}")
    
    @asynccontextmanager
    async def inject_latency(self, policy: FaultPolicy):
        """Context manager for latency injection"""
        fault_id = f"latency_{int(time.time())}"
        
        self.active_faults[fault_id] = {
            'type': FaultType.NETWORK_LATENCY,
            'policy': policy,
            'start_time': datetime.now(),
            'latency_ms': policy.parameters.get('latency_ms', 100),
            'jitter_ms': policy.parameters.get('jitter_ms', 10)
        }
        
        try:
            self.logger.info(f"Injecting latency: {fault_id}")
            yield fault_id
        finally:
            if fault_id in self.active_faults:
                del self.active_faults[fault_id]
            self.logger.info(f"Stopped latency injection: {fault_id}")
    
    def should_inject_fault(self, fault_type: FaultType, endpoint: str = None) -> bool:
        """Check if fault should be injected for current request"""
        for fault_id, fault_info in self.active_faults.items():
            if fault_info['type'] == fault_type:
                policy = fault_info['policy']
                
                # Check probability
                if random.random() > policy.probability:
                    continue
                
                # Check endpoint filter for API errors
                if fault_type == FaultType.API_ERROR and endpoint:
                    affected_endpoints = fault_info.get('affected_endpoints', ['*'])
                    if '*' not in affected_endpoints and endpoint not in affected_endpoints:
                        continue
                
                return True
        
        return False
    
    def get_injected_latency(self) -> float:
        """Get additional latency to inject"""
        total_latency = 0.0
        
        for fault_info in self.active_faults.values():
            if fault_info['type'] == FaultType.NETWORK_LATENCY:
                base_latency = fault_info.get('latency_ms', 0)
                jitter = fault_info.get('jitter_ms', 0)
                
                # Add jitter
                actual_latency = base_latency + random.uniform(-jitter, jitter)
                total_latency += max(0, actual_latency)
        
        return total_latency / 1000.0  # Convert to seconds
    
    def get_injected_error(self) -> Optional[int]:
        """Get HTTP status code to inject"""
        for fault_info in self.active_faults.values():
            if fault_info['type'] == FaultType.API_ERROR:
                if self.should_inject_fault(FaultType.API_ERROR):
                    status_codes = fault_info.get('status_codes', [500])
                    return random.choice(status_codes)
        
        return None


class SLOMonitor:
    """Monitor SLOs during chaos testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, List[float]] = {
            'response_times': [],
            'error_rates': [],
            'throughput': [],
            'availability': []
        }
        self.violations: List[Dict[str, Any]] = []
    
    async def check_slo(self, policy: FaultPolicy) -> List[Dict[str, Any]]:
        """Check SLO compliance during fault injection"""
        violations = []
        
        for slo_check in policy.slo_checks:
            check_type = slo_check['type']
            threshold = slo_check['threshold']
            
            if check_type == 'response_time':
                if self.metrics['response_times']:
                    avg_response_time = np.mean(self.metrics['response_times'][-100:])  # Last 100 requests
                    if avg_response_time > threshold:
                        violations.append({
                            'type': 'response_time',
                            'threshold': threshold,
                            'actual': avg_response_time,
                            'timestamp': datetime.now().isoformat()
                        })
            
            elif check_type == 'error_rate':
                if self.metrics['error_rates']:
                    current_error_rate = np.mean(self.metrics['error_rates'][-10:])  # Last 10 measurements
                    if current_error_rate > threshold:
                        violations.append({
                            'type': 'error_rate',
                            'threshold': threshold,
                            'actual': current_error_rate,
                            'timestamp': datetime.now().isoformat()
                        })
            
            elif check_type == 'availability':
                if self.metrics['availability']:
                    current_availability = np.mean(self.metrics['availability'][-10:])
                    if current_availability < threshold:
                        violations.append({
                            'type': 'availability',
                            'threshold': threshold,
                            'actual': current_availability,
                            'timestamp': datetime.now().isoformat()
                        })
        
        return violations
    
    def record_response_time(self, response_time_ms: float):
        """Record response time measurement"""
        self.metrics['response_times'].append(response_time_ms)
        
        # Keep only recent measurements
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-500:]
    
    def record_error_rate(self, error_rate: float):
        """Record error rate measurement"""
        self.metrics['error_rates'].append(error_rate)
        
        if len(self.metrics['error_rates']) > 100:
            self.metrics['error_rates'] = self.metrics['error_rates'][-50:]
    
    def record_availability(self, availability: float):
        """Record availability measurement"""
        self.metrics['availability'].append(availability)
        
        if len(self.metrics['availability']) > 100:
            self.metrics['availability'] = self.metrics['availability'][-50:]


class ChaosEngineeringFramework:
    """Main chaos engineering framework"""
    
    def __init__(self, use_chaos_mesh: bool = True):
        self.use_chaos_mesh = use_chaos_mesh
        self.logger = logging.getLogger(__name__)
        
        if use_chaos_mesh:
            self.chaos_mesh = ChaosMeshClient()
        
        self.app_injector = ApplicationFaultInjector()
        self.slo_monitor = SLOMonitor()
        
        # Active experiments
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
    
    async def run_fault_injection_test(self, policy: FaultPolicy) -> FaultInjectionResult:
        """Run complete fault injection test"""
        start_time = datetime.now()
        self.logger.info(f"Starting fault injection test: {policy.name}")
        
        try:
            # Start fault injection
            if self.use_chaos_mesh and policy.fault_type in [FaultType.NETWORK_LATENCY, FaultType.PACKET_LOSS]:
                chaos_id = await self._inject_kubernetes_fault(policy)
            else:
                chaos_id = await self._inject_application_fault(policy)
            
            # Monitor for the duration
            await self._monitor_during_fault(policy)
            
            # Check SLOs
            violations = await self.slo_monitor.check_slo(policy)
            
            # Calculate recovery time
            recovery_time = await self._measure_recovery_time(policy)
            
            end_time = datetime.now()
            
            # Cleanup
            await self._cleanup_fault(chaos_id, policy)
            
            return FaultInjectionResult(
                policy=policy,
                start_time=start_time,
                end_time=end_time,
                success=len(violations) == 0,
                metrics=self.slo_monitor.metrics.copy(),
                slo_violations=violations,
                error_logs=[],
                recovery_time_seconds=recovery_time
            )
        
        except Exception as e:
            self.logger.error(f"Fault injection test failed: {e}")
            return FaultInjectionResult(
                policy=policy,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                metrics={},
                slo_violations=[],
                error_logs=[str(e)]
            )
    
    async def _inject_kubernetes_fault(self, policy: FaultPolicy) -> str:
        """Inject fault using Chaos Mesh"""
        if policy.fault_type == FaultType.NETWORK_LATENCY:
            return await self.chaos_mesh.inject_network_latency(policy)
        elif policy.fault_type == FaultType.PACKET_LOSS:
            return await self.chaos_mesh.inject_packet_loss(policy)
        elif policy.fault_type == FaultType.SERVICE_UNAVAILABLE:
            return await self.chaos_mesh.inject_pod_failure(policy)
        else:
            raise ValueError(f"Unsupported Kubernetes fault type: {policy.fault_type}")
    
    async def _inject_application_fault(self, policy: FaultPolicy) -> str:
        """Inject fault at application level"""
        if policy.fault_type == FaultType.API_ERROR:
            async with self.app_injector.inject_api_errors(policy) as fault_id:
                # Wait for duration
                await asyncio.sleep(policy.duration_minutes * 60)
                return fault_id
        elif policy.fault_type == FaultType.NETWORK_LATENCY:
            async with self.app_injector.inject_latency(policy) as fault_id:
                await asyncio.sleep(policy.duration_minutes * 60)
                return fault_id
        else:
            raise ValueError(f"Unsupported application fault type: {policy.fault_type}")
    
    async def _monitor_during_fault(self, policy: FaultPolicy):
        """Monitor system during fault injection"""
        duration_seconds = policy.duration_minutes * 60
        monitoring_interval = 10  # seconds
        
        for _ in range(0, duration_seconds, monitoring_interval):
            # Simulate monitoring measurements
            # In real implementation, this would call actual monitoring APIs
            
            # Simulate response time measurement
            base_response_time = 100  # ms
            if policy.fault_type == FaultType.NETWORK_LATENCY:
                base_response_time += policy.parameters.get('latency_ms', 100)
            
            response_time = base_response_time + random.uniform(-20, 50)
            self.slo_monitor.record_response_time(response_time)
            
            # Simulate error rate
            base_error_rate = 0.01  # 1%
            if policy.fault_type == FaultType.API_ERROR:
                base_error_rate += policy.parameters.get('error_rate', 0.1)
            
            error_rate = min(1.0, base_error_rate + random.uniform(-0.005, 0.02))
            self.slo_monitor.record_error_rate(error_rate)
            
            # Simulate availability
            base_availability = 0.999  # 99.9%
            if policy.fault_type == FaultType.SERVICE_UNAVAILABLE:
                base_availability = 0.95  # 95%
            
            availability = max(0.0, base_availability + random.uniform(-0.01, 0.005))
            self.slo_monitor.record_availability(availability)
            
            await asyncio.sleep(monitoring_interval)
    
    async def _measure_recovery_time(self, policy: FaultPolicy) -> float:
        """Measure time to recover after fault injection stops"""
        start_time = time.time()
        
        # Wait for system to recover (simplified)
        await asyncio.sleep(5)  # Assume 5 second recovery time
        
        return time.time() - start_time
    
    async def _cleanup_fault(self, chaos_id: str, policy: FaultPolicy):
        """Clean up fault injection"""
        if self.use_chaos_mesh and policy.fault_type in [FaultType.NETWORK_LATENCY, FaultType.PACKET_LOSS]:
            chaos_type = "network" if policy.fault_type in [FaultType.NETWORK_LATENCY, FaultType.PACKET_LOSS] else "pod"
            await self.chaos_mesh.cleanup_chaos(chaos_id, chaos_type)
    
    def load_policies_from_file(self, file_path: Path) -> List[FaultPolicy]:
        """Load fault injection policies from YAML file"""
        with open(file_path, 'r') as f:
            policies_data = yaml.safe_load(f)
        
        policies = []
        for policy_data in policies_data.get('policies', []):
            policy = FaultPolicy(
                name=policy_data['name'],
                fault_type=FaultType(policy_data['fault_type']),
                severity=SeverityLevel(policy_data['severity']),
                target_services=policy_data['target_services'],
                parameters=policy_data['parameters'],
                duration_minutes=policy_data.get('duration_minutes', 5),
                probability=policy_data.get('probability', 1.0),
                slo_checks=policy_data.get('slo_checks', [])
            )
            policies.append(policy)
        
        return policies
    
    async def run_chaos_test_suite(self, policies: List[FaultPolicy]) -> List[FaultInjectionResult]:
        """Run complete chaos test suite"""
        results = []
        
        for policy in policies:
            if not policy.enabled:
                continue
            
            self.logger.info(f"Running chaos test: {policy.name}")
            result = await self.run_fault_injection_test(policy)
            results.append(result)
            
            # Wait between tests
            await asyncio.sleep(30)
        
        return results
    
    def generate_chaos_report(self, results: List[FaultInjectionResult]) -> Dict[str, Any]:
        """Generate comprehensive chaos engineering report"""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        
        return {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failure_rate': (total_tests - successful_tests) / total_tests if total_tests > 0 else 0,
                'total_violations': sum(len(r.slo_violations) for r in results)
            },
            'test_results': [
                {
                    'name': result.policy.name,
                    'fault_type': result.policy.fault_type.value,
                    'success': result.success,
                    'duration': (result.end_time - result.start_time).total_seconds(),
                    'violations': len(result.slo_violations),
                    'recovery_time': result.recovery_time_seconds
                }
                for result in results
            ],
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[FaultInjectionResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for consistent failures
        failure_types = {}
        for result in results:
            if not result.success:
                fault_type = result.policy.fault_type.value
                if fault_type not in failure_types:
                    failure_types[fault_type] = 0
                failure_types[fault_type] += 1
        
        for fault_type, count in failure_types.items():
            if count > 1:
                recommendations.append(
                    f"Multiple failures detected for {fault_type}. "
                    f"Consider improving resilience for this fault type."
                )
        
        # Check recovery times
        long_recovery_times = [r for r in results if r.recovery_time_seconds and r.recovery_time_seconds > 30]
        if long_recovery_times:
            recommendations.append(
                f"{len(long_recovery_times)} tests had recovery times > 30 seconds. "
                "Consider implementing faster recovery mechanisms."
            )
        
        return recommendations

