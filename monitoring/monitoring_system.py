#!/usr/bin/env python3
"""
Comprehensive Monitoring System for Ultimate Arbitrage System

This module provides:
- Metrics collection with Prometheus integration
- Structured logging with OpenTelemetry
- Distributed tracing with Jaeger
- Real-time anomaly detection
- Business KPI monitoring
- Security event monitoring
- Time-travel debugging capabilities
"""

import os
import time
import json
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from functools import wraps
from contextlib import contextmanager

# Prometheus metrics
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum,
    CollectorRegistry, multiprocess, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)

# OpenTelemetry
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Logging
import structlog
from pythonjsonlogger import jsonlogger

# Scientific computing for anomaly detection
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Security and networking
import psutil
import socket
import requests
from cryptography.fernet import Fernet

# ClickHouse for time-travel debugging
from clickhouse_driver import Client as ClickHouseClient


@dataclass
class MetricEvent:
    """Structured metric event for ClickHouse storage"""
    timestamp: datetime
    metric_name: str
    metric_type: str
    value: float
    labels: Dict[str, str]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    service: str = "arbitrage-system"
    environment: str = "production"


@dataclass
class SecurityEvent:
    """Security event structure"""
    timestamp: datetime
    event_type: str
    severity: str
    source_ip: str
    user_agent: Optional[str]
    description: str
    threat_score: float
    action_taken: str
    trace_id: Optional[str] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    timestamp: datetime
    metric_name: str
    current_value: float
    expected_value: float
    deviation_score: float
    is_anomaly: bool
    confidence: float
    context: Dict[str, Any]


class TradingMetrics:
    """Trading-specific metrics collection"""
    
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        
        # Trading performance metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders placed',
            ['exchange', 'symbol', 'side', 'status'],
            registry=registry
        )
        
        self.order_latency = Histogram(
            'trading_order_latency_seconds',
            'Order placement latency',
            ['exchange', 'symbol'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=registry
        )
        
        self.fill_rate = Gauge(
            'trading_fill_rate',
            'Order fill rate percentage',
            ['exchange', 'symbol'],
            registry=registry
        )
        
        self.arbitrage_opportunities = Counter(
            'arbitrage_opportunities_total',
            'Total arbitrage opportunities detected',
            ['symbol', 'exchange_pair'],
            registry=registry
        )
        
        self.arbitrage_executed = Counter(
            'arbitrage_executed_total',
            'Total arbitrage opportunities executed',
            ['symbol', 'exchange_pair', 'result'],
            registry=registry
        )
        
        self.pnl_current = Gauge(
            'trading_pnl_current',
            'Current profit and loss',
            ['strategy', 'symbol'],
            registry=registry
        )
        
        self.pnl_daily = Gauge(
            'trading_pnl_daily',
            'Daily profit and loss',
            ['strategy'],
            registry=registry
        )
        
        self.margin_used = Gauge(
            'trading_margin_used',
            'Currently used margin',
            ['exchange'],
            registry=registry
        )
        
        self.margin_available = Gauge(
            'trading_margin_available',
            'Available margin',
            ['exchange'],
            registry=registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_total',
            'Total portfolio value',
            ['currency'],
            registry=registry
        )
        
        self.risk_score = Gauge(
            'risk_score_current',
            'Current risk score',
            ['risk_type'],
            registry=registry
        )
        
        self.exchange_connectivity = Gauge(
            'exchange_connectivity_status',
            'Exchange connectivity status (1=connected, 0=disconnected)',
            ['exchange'],
            registry=registry
        )
        
        self.api_rate_limit = Gauge(
            'exchange_api_rate_limit_remaining',
            'Remaining API rate limit',
            ['exchange', 'endpoint'],
            registry=registry
        )


class SystemMetrics:
    """System-level metrics collection"""
    
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=registry
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['database', 'operation'],
            registry=registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=registry
        )
        
        # Business metrics
        self.business_events_total = Counter(
            'business_events_total',
            'Total business events',
            ['event_type', 'status'],
            registry=registry
        )
        
        self.slo_violations_total = Counter(
            'slo_violations_total',
            'Total SLO violations',
            ['slo_type', 'severity'],
            registry=registry
        )


class AnomalyDetector:
    """Real-time anomaly detection system"""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.models = {}
        self.scalers = {}
        self.last_trained = {}
        self.training_interval = timedelta(minutes=30)
        
    def add_data_point(self, metric_name: str, value: float, context: Dict[str, Any] = None) -> Optional[AnomalyDetection]:
        """Add a data point and check for anomalies"""
        current_time = datetime.utcnow()
        context = context or {}
        
        # Add to window
        self.data_windows[metric_name].append((current_time, value, context))
        
        # Check if we need to retrain the model
        if (metric_name not in self.last_trained or 
            current_time - self.last_trained[metric_name] > self.training_interval):
            self._train_model(metric_name)
        
        # Detect anomaly
        return self._detect_anomaly(metric_name, value, current_time, context)
    
    def _train_model(self, metric_name: str):
        """Train anomaly detection model for a metric"""
        window = self.data_windows[metric_name]
        if len(window) < 20:  # Need minimum data points
            return
        
        # Prepare data
        values = np.array([point[1] for point in window]).reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Train isolation forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(scaled_values)
        
        # Store model and scaler
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        self.last_trained[metric_name] = datetime.utcnow()
    
    def _detect_anomaly(self, metric_name: str, value: float, timestamp: datetime, context: Dict[str, Any]) -> Optional[AnomalyDetection]:
        """Detect if a value is anomalous"""
        if metric_name not in self.models:
            return None
        
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]
        
        # Scale the value
        scaled_value = scaler.transform([[value]])
        
        # Predict anomaly
        anomaly_score = model.decision_function(scaled_value)[0]
        is_anomaly = model.predict(scaled_value)[0] == -1
        
        # Calculate expected value (median of recent data)
        recent_values = [point[1] for point in list(self.data_windows[metric_name])[-10:]]
        expected_value = np.median(recent_values) if recent_values else value
        
        # Calculate deviation score
        if recent_values:
            std_dev = np.std(recent_values)
            deviation_score = abs(value - expected_value) / (std_dev + 1e-6)
        else:
            deviation_score = 0.0
        
        return AnomalyDetection(
            timestamp=timestamp,
            metric_name=metric_name,
            current_value=value,
            expected_value=expected_value,
            deviation_score=deviation_score,
            is_anomaly=is_anomaly,
            confidence=abs(anomaly_score),
            context=context
        )


class SecurityMonitor:
    """Security event monitoring and threat detection"""
    
    def __init__(self):
        self.threat_patterns = {
            'sql_injection': [
                r'union.*select', r'drop.*table', r'insert.*into',
                r'delete.*from', r'update.*set', r"'.*or.*'.*='.*'"
            ],
            'xss': [
                r'<script.*>', r'javascript:', r'onload=',
                r'onerror=', r'onclick='
            ],
            'directory_traversal': [
                r'\.\./.*\.\./.*', r'\\.*\\.*', r'/etc/passwd',
                r'/etc/shadow', r'\\windows\\system32'
            ],
            'command_injection': [
                r';.*rm.*', r';.*cat.*', r';.*ls.*', r'\|.*nc.*',
                r'&&.*wget.*', r'`.*`'
            ]
        }
        
        self.geo_anomaly_threshold = 0.8
        self.rate_limit_threshold = 100  # requests per minute
        self.request_counts = defaultdict(list)
    
    def analyze_request(self, ip: str, user_agent: str, url: str, headers: Dict[str, str], body: str = "") -> Optional[SecurityEvent]:
        """Analyze an HTTP request for security threats"""
        threats = []
        threat_score = 0.0
        
        # Check for attack patterns
        full_content = f"{url} {body}".lower()
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_content, re.IGNORECASE):
                    threats.append(threat_type)
                    threat_score += 0.3
        
        # Check rate limiting
        current_time = datetime.utcnow()
        self.request_counts[ip] = [
            t for t in self.request_counts[ip]
            if current_time - t < timedelta(minutes=1)
        ]
        self.request_counts[ip].append(current_time)
        
        if len(self.request_counts[ip]) > self.rate_limit_threshold:
            threats.append('rate_limit_exceeded')
            threat_score += 0.5
        
        # Geographic anomaly detection (placeholder)
        # In production, integrate with GeoIP service
        if self._is_geo_anomaly(ip):
            threats.append('geographic_anomaly')
            threat_score += 0.2
        
        # Check for suspicious user agents
        if self._is_suspicious_user_agent(user_agent):
            threats.append('suspicious_user_agent')
            threat_score += 0.1
        
        if threats:
            action_taken = 'blocked' if threat_score > 0.7 else 'logged'
            severity = 'critical' if threat_score > 0.8 else 'warning' if threat_score > 0.4 else 'info'
            
            return SecurityEvent(
                timestamp=current_time,
                event_type=','.join(threats),
                severity=severity,
                source_ip=ip,
                user_agent=user_agent,
                description=f"Security threat detected: {', '.join(threats)}",
                threat_score=threat_score,
                action_taken=action_taken
            )
        
        return None
    
    def _is_geo_anomaly(self, ip: str) -> bool:
        """Check if IP address is from an unusual location"""
        # Placeholder for GeoIP integration
        # In production, use MaxMind GeoIP or similar service
        return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_patterns = [
            'bot', 'crawler', 'spider', 'scraper', 'scanner',
            'nikto', 'sqlmap', 'nmap', 'masscan'
        ]
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)


class MonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self, 
                 clickhouse_host: str = 'localhost',
                 clickhouse_port: int = 9000,
                 jaeger_endpoint: str = 'http://localhost:14268/api/traces',
                 prometheus_port: int = 9999):
        
        # Initialize registry
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self.trading_metrics = TradingMetrics(self.registry)
        self.system_metrics = SystemMetrics(self.registry)
        
        # Initialize anomaly detection
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize security monitoring
        self.security_monitor = SecurityMonitor()
        
        # Initialize ClickHouse client
        self.clickhouse_client = ClickHouseClient(
            host=clickhouse_host,
            port=clickhouse_port,
            database='monitoring'
        )
        
        # Initialize OpenTelemetry
        self._setup_opentelemetry(jaeger_endpoint)
        
        # Initialize structured logging
        self._setup_logging()
        
        # Start metrics server
        self.prometheus_port = prometheus_port
        self._start_metrics_server()
        
        # Setup ClickHouse tables
        self._setup_clickhouse_tables()
        
        # Background tasks
        self.running = True
        self.background_tasks = []
        self._start_background_tasks()
    
    def _setup_opentelemetry(self, jaeger_endpoint: str):
        """Setup OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": "arbitrage-system",
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer = trace.get_tracer(__name__)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument common libraries
        RequestsInstrumentor().instrument()
        Psycopg2Instrumentor().instrument()
        RedisInstrumentor().instrument()
    
    def _setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging
        logging.basicConfig(
            format='%(message)s',
            level=logging.INFO
        )
        
        # Add JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            timestamp=True
        )
        
        # File handlers for different log types
        log_dir = '/app/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        handlers = {
            'trading': logging.FileHandler(f'{log_dir}/trading.log'),
            'security': logging.FileHandler(f'{log_dir}/security.log'),
            'system': logging.FileHandler(f'{log_dir}/system.log'),
            'anomaly': logging.FileHandler(f'{log_dir}/anomaly.log')
        }
        
        for name, handler in handlers.items():
            handler.setFormatter(formatter)
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        start_http_server(self.prometheus_port, registry=self.registry)
        print(f"Metrics server started on port {self.prometheus_port}")
    
    def _setup_clickhouse_tables(self):
        """Setup ClickHouse tables for time-travel debugging"""
        tables = {
            'metrics_events': '''
                CREATE TABLE IF NOT EXISTS metrics_events (
                    timestamp DateTime64(3) CODEC(Delta, LZ4),
                    metric_name LowCardinality(String),
                    metric_type LowCardinality(String),
                    value Float64 CODEC(Gorilla, LZ4),
                    labels Map(String, String),
                    trace_id String,
                    span_id String,
                    service LowCardinality(String),
                    environment LowCardinality(String)
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMMDD(timestamp)
                ORDER BY (timestamp, metric_name, service)
                TTL timestamp + INTERVAL 90 DAY
                SETTINGS index_granularity = 8192
            ''',
            'security_events': '''
                CREATE TABLE IF NOT EXISTS security_events (
                    timestamp DateTime64(3) CODEC(Delta, LZ4),
                    event_type LowCardinality(String),
                    severity LowCardinality(String),
                    source_ip IPv4,
                    user_agent String,
                    description String,
                    threat_score Float32,
                    action_taken LowCardinality(String),
                    trace_id String
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMMDD(timestamp)
                ORDER BY (timestamp, severity, event_type)
                TTL timestamp + INTERVAL 365 DAY
                SETTINGS index_granularity = 8192
            ''',
            'anomaly_detections': '''
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    timestamp DateTime64(3) CODEC(Delta, LZ4),
                    metric_name LowCardinality(String),
                    current_value Float64,
                    expected_value Float64,
                    deviation_score Float64,
                    is_anomaly UInt8,
                    confidence Float64,
                    context Map(String, String)
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMMDD(timestamp)
                ORDER BY (timestamp, metric_name, is_anomaly)
                TTL timestamp + INTERVAL 30 DAY
                SETTINGS index_granularity = 8192
            '''
        }
        
        for table_name, create_sql in tables.items():
            try:
                self.clickhouse_client.execute(create_sql)
                print(f"ClickHouse table '{table_name}' ready")
            except Exception as e:
                print(f"Error creating ClickHouse table '{table_name}': {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        tasks = [
            self._system_metrics_collector,
            self._anomaly_processor,
            self._clickhouse_batch_inserter
        ]
        
        for task in tasks:
            thread = threading.Thread(target=task, daemon=True)
            thread.start()
            self.background_tasks.append(thread)
    
    def _system_metrics_collector(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                # CPU and memory metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Database connections (placeholder)
                # In production, integrate with actual database pool
                db_connections = 10  # Example value
                
                # Update Prometheus metrics
                self.system_metrics.business_events_total.labels(
                    event_type='system_check',
                    status='success'
                ).inc()
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(5)
    
    def _anomaly_processor(self):
        """Process anomaly detections"""
        while self.running:
            try:
                # This would be triggered by metrics updates
                # For now, simulate with random data
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in anomaly processor: {e}")
                time.sleep(5)
    
    def _clickhouse_batch_inserter(self):
        """Batch insert events to ClickHouse"""
        batch_size = 1000
        batch_interval = 10  # seconds
        
        metrics_batch = []
        security_batch = []
        anomaly_batch = []
        
        while self.running:
            try:
                # In production, this would collect from queues
                # For now, simulate with periodic flushes
                time.sleep(batch_interval)
                
                # Flush batches if they have data
                if metrics_batch:
                    self._flush_metrics_batch(metrics_batch)
                    metrics_batch.clear()
                
                if security_batch:
                    self._flush_security_batch(security_batch)
                    security_batch.clear()
                
                if anomaly_batch:
                    self._flush_anomaly_batch(anomaly_batch)
                    anomaly_batch.clear()
                
            except Exception as e:
                print(f"Error in ClickHouse batch inserter: {e}")
                time.sleep(5)
    
    def _flush_metrics_batch(self, batch: List[MetricEvent]):
        """Flush metrics batch to ClickHouse"""
        if not batch:
            return
        
        data = [
            [
                event.timestamp,
                event.metric_name,
                event.metric_type,
                event.value,
                event.labels,
                event.trace_id or '',
                event.span_id or '',
                event.service,
                event.environment
            ]
            for event in batch
        ]
        
        self.clickhouse_client.execute(
            'INSERT INTO metrics_events VALUES',
            data
        )
    
    def _flush_security_batch(self, batch: List[SecurityEvent]):
        """Flush security batch to ClickHouse"""
        if not batch:
            return
        
        data = [
            [
                event.timestamp,
                event.event_type,
                event.severity,
                event.source_ip,
                event.user_agent or '',
                event.description,
                event.threat_score,
                event.action_taken,
                event.trace_id or ''
            ]
            for event in batch
        ]
        
        self.clickhouse_client.execute(
            'INSERT INTO security_events VALUES',
            data
        )
    
    def _flush_anomaly_batch(self, batch: List[AnomalyDetection]):
        """Flush anomaly batch to ClickHouse"""
        if not batch:
            return
        
        data = [
            [
                event.timestamp,
                event.metric_name,
                event.current_value,
                event.expected_value,
                event.deviation_score,
                1 if event.is_anomaly else 0,
                event.confidence,
                event.context
            ]
            for event in batch
        ]
        
        self.clickhouse_client.execute(
            'INSERT INTO anomaly_detections VALUES',
            data
        )
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations"""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(operation_name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span
    
    def record_trading_event(self, event_type: str, **kwargs):
        """Record a trading event with metrics and logging"""
        with self.trace_operation(f"trading.{event_type}", **kwargs) as span:
            # Log the event
            logger = logging.getLogger('trading')
            logger.info(f"Trading event: {event_type}", extra={
                'event_type': event_type,
                'trace_id': format(span.get_span_context().trace_id, '032x'),
                'span_id': format(span.get_span_context().span_id, '016x'),
                **kwargs
            })
            
            # Update metrics based on event type
            if event_type == 'order_placed':
                self.trading_metrics.orders_total.labels(
                    exchange=kwargs.get('exchange', 'unknown'),
                    symbol=kwargs.get('symbol', 'unknown'),
                    side=kwargs.get('side', 'unknown'),
                    status='placed'
                ).inc()
            
            elif event_type == 'arbitrage_opportunity':
                self.trading_metrics.arbitrage_opportunities.labels(
                    symbol=kwargs.get('symbol', 'unknown'),
                    exchange_pair=kwargs.get('exchange_pair', 'unknown')
                ).inc()
    
    def record_security_event(self, event: SecurityEvent):
        """Record a security event"""
        logger = logging.getLogger('security')
        logger.warning("Security event detected", extra=asdict(event))
        
        # Add to batch for ClickHouse (in production, use a queue)
        # For now, insert directly
        try:
            self._flush_security_batch([event])
        except Exception as e:
            print(f"Error recording security event: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the monitoring system"""
        self.running = False
        print("Monitoring system shutting down...")
        
        # Wait for background tasks to finish
        for task in self.background_tasks:
            task.join(timeout=5)
        
        print("Monitoring system shutdown complete")


# Decorators for easy instrumentation
def monitor_execution_time(metric_name: str = None):
    """Decorator to monitor function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                # Record metric (would need access to monitoring system instance)
                print(f"Function {func.__name__} executed in {execution_time:.3f}s")
        return wrapper
    return decorator


def monitor_errors(error_metric: str = None):
    """Decorator to monitor function errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Record error metric
                print(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    monitoring = MonitoringSystem()
    
    try:
        # Simulate some trading events
        monitoring.record_trading_event(
            'order_placed',
            exchange='binance',
            symbol='BTC/USDT',
            side='buy',
            amount=0.1,
            price=50000
        )
        
        # Simulate security event
        security_event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type='suspicious_api_call',
            severity='warning',
            source_ip='192.168.1.100',
            user_agent='curl/7.68.0',
            description='Suspicious API call pattern detected',
            threat_score=0.6,
            action_taken='logged'
        )
        monitoring.record_security_event(security_event)
        
        # Keep running
        print("Monitoring system started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        monitoring.shutdown()

