#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Logging System for Ultimate Master Orchestrator
======================================================

High-performance, structured logging with real-time monitoring
and advanced analytics for the Ultimate Arbitrage System.

Features:
- Structured JSON logging for machine readability
- Real-time log streaming for monitoring
- Performance metrics integration
- Component-specific log levels
- Automated log rotation and archival
- Security and audit logging
"""

import logging
import logging.handlers
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs
    with enhanced metadata for orchestrator monitoring.
    """
    
    def __init__(self):
        super().__init__()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
            'hostname': self.hostname
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add custom fields from extra
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        # Add performance metrics if available
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
        
        # Add component information
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        
        # Add execution context
        if hasattr(record, 'execution_id'):
            log_data['execution_id'] = record.execution_id
        
        if hasattr(record, 'signal_id'):
            log_data['signal_id'] = record.signal_id
        
        # Add trade information
        if hasattr(record, 'asset'):
            log_data['asset'] = record.asset
        
        if hasattr(record, 'action'):
            log_data['action'] = record.action
        
        if hasattr(record, 'profit'):
            log_data['profit'] = record.profit
        
        return json.dumps(log_data, ensure_ascii=False)

class PerformanceFilter(logging.Filter):
    """
    Filter that adds performance metrics to log records
    for enhanced monitoring and analytics.
    """
    
    def __init__(self):
        super().__init__()
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._log_count = 0
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to log record"""
        current_time = time.time()
        
        # Add timing information
        record.timestamp_ms = int(current_time * 1000)
        record.uptime_seconds = current_time - self._start_time
        record.time_since_last_log_ms = (current_time - self._last_log_time) * 1000
        
        # Add log frequency metrics
        self._log_count += 1
        record.log_sequence = self._log_count
        record.logs_per_second = self._log_count / (current_time - self._start_time)
        
        self._last_log_time = current_time
        return True

class ComponentFilter(logging.Filter):
    """
    Filter that enhances logs with component-specific information
    for better traceability and debugging.
    """
    
    def __init__(self, component_name: str):
        super().__init__()
        self.component_name = component_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add component information to log record"""
        record.component = self.component_name
        return True

class OrchestratorLogger:
    """
    Advanced logging system specifically designed for the Ultimate Master Orchestrator
    with real-time monitoring, structured logging, and performance analytics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loggers = {}
        self.handlers = {}
        self.filters = {}
        
        # Create logs directory if it doesn't exist
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize main orchestrator logger
        self._setup_main_logger()
        
        # Initialize component loggers
        self._setup_component_loggers()
        
        # Start log monitoring thread
        self._start_log_monitoring()
    
    def _setup_main_logger(self):
        """Setup the main orchestrator logger with advanced configuration"""
        logger = logging.getLogger('ultimate_orchestrator')
        logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = StructuredFormatter()
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        log_file = self.log_dir / 'orchestrator.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.get('max_file_size_mb', 100) * 1024 * 1024,
            backupCount=self.config.get('backup_count', 5)
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        
        # Performance handler for metrics
        metrics_file = self.log_dir / 'performance_metrics.log'
        metrics_handler = logging.handlers.RotatingFileHandler(
            metrics_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(console_formatter)
        
        # Add filters
        performance_filter = PerformanceFilter()
        console_handler.addFilter(performance_filter)
        file_handler.addFilter(performance_filter)
        metrics_handler.addFilter(performance_filter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(metrics_handler)
        
        # Store references
        self.loggers['main'] = logger
        self.handlers['console'] = console_handler
        self.handlers['file'] = file_handler
        self.handlers['metrics'] = metrics_handler
    
    def _setup_component_loggers(self):
        """Setup specialized loggers for each orchestrator component"""
        components = [
            'signal_fusion',
            'performance_optimizer', 
            'health_monitor',
            'execution_coordinator'
        ]
        
        for component in components:
            logger = logging.getLogger(f'orchestrator.{component}')
            component_level = self.config.get('components', {}).get(component, 'INFO')
            logger.setLevel(getattr(logging, component_level))
            
            # Component-specific file handler
            log_file = self.log_dir / f'{component}.log'
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=25 * 1024 * 1024,  # 25MB
                backupCount=5
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(StructuredFormatter())
            
            # Add component filter
            component_filter = ComponentFilter(component)
            handler.addFilter(component_filter)
            handler.addFilter(PerformanceFilter())
            
            logger.addHandler(handler)
            logger.propagate = True  # Also send to parent logger
            
            self.loggers[component] = logger
            self.handlers[f'{component}_file'] = handler
    
    def _start_log_monitoring(self):
        """Start background thread for log monitoring and analytics"""
        def monitor_logs():
            while True:
                try:
                    # Monitor log file sizes
                    self._check_log_sizes()
                    
                    # Generate log analytics
                    self._generate_log_analytics()
                    
                    # Clean old logs
                    self._cleanup_old_logs()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    # Use basic logging to avoid recursion
                    print(f"Error in log monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
        monitor_thread.start()
    
    def _check_log_sizes(self):
        """Monitor log file sizes and trigger rotation if needed"""
        max_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        
        for log_file in self.log_dir.glob('*.log'):
            if log_file.stat().st_size > max_size:
                logger = self.get_logger('main')
                logger.warning(f"Log file {log_file.name} approaching size limit",
                             extra={'metrics': {'file_size_mb': log_file.stat().st_size / (1024*1024)}})
    
    def _generate_log_analytics(self):
        """Generate analytics from recent log data"""
        # This would analyze recent logs for patterns, errors, performance issues
        # For now, just log basic statistics
        logger = self.get_logger('main')
        
        analytics = {
            'log_files_count': len(list(self.log_dir.glob('*.log'))),
            'total_log_size_mb': sum(f.stat().st_size for f in self.log_dir.glob('*.log')) / (1024*1024),
            'active_loggers': len(self.loggers)
        }
        
        logger.debug("Log analytics generated", extra={'metrics': analytics})
    
    def _cleanup_old_logs(self):
        """Clean up old log files beyond retention period"""
        retention_days = self.config.get('retention_days', 30)
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        for log_file in self.log_dir.glob('*.log.*'):  # Rotated files
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                except Exception as e:
                    print(f"Error deleting old log file {log_file}: {e}")
    
    def get_logger(self, component: str = 'main') -> logging.Logger:
        """Get logger for specific component"""
        return self.loggers.get(component, self.loggers['main'])
    
    def log_signal(self, logger_name: str, level: str, message: str, 
                   signal_id: str = None, asset: str = None, action: str = None,
                   profit: float = None, **kwargs):
        """Log trading signal with enhanced context"""
        logger = self.get_logger(logger_name)
        log_level = getattr(logging, level.upper())
        
        extra_data = kwargs.copy()
        if signal_id:
            extra_data['signal_id'] = signal_id
        if asset:
            extra_data['asset'] = asset
        if action:
            extra_data['action'] = action
        if profit is not None:
            extra_data['profit'] = profit
        
        logger.log(log_level, message, extra={'extra_data': extra_data})
    
    def log_execution(self, execution_id: str, level: str, message: str,
                     signal_id: str = None, success: bool = None, 
                     execution_time_ms: float = None, **kwargs):
        """Log execution with enhanced context"""
        logger = self.get_logger('execution_coordinator')
        log_level = getattr(logging, level.upper())
        
        extra_data = kwargs.copy()
        extra_data['execution_id'] = execution_id
        if signal_id:
            extra_data['signal_id'] = signal_id
        if success is not None:
            extra_data['success'] = success
        if execution_time_ms is not None:
            extra_data['execution_time_ms'] = execution_time_ms
        
        logger.log(log_level, message, extra={'extra_data': extra_data})
    
    def log_performance(self, component: str, metrics: Dict[str, Any], message: str = None):
        """Log performance metrics with structured data"""
        logger = self.get_logger(component)
        
        log_message = message or f"Performance metrics for {component}"
        logger.info(log_message, extra={'metrics': metrics})
    
    def log_health(self, component: str, health_data: Dict[str, Any], status: str):
        """Log health check results with structured data"""
        logger = self.get_logger('health_monitor')
        
        extra_data = {
            'component': component,
            'status': status,
            'health_data': health_data
        }
        
        if status in ['critical', 'offline']:
            logger.error(f"Component {component} health check failed", extra={'extra_data': extra_data})
        elif status == 'warning':
            logger.warning(f"Component {component} health degraded", extra={'extra_data': extra_data})
        else:
            logger.debug(f"Component {component} health check passed", extra={'extra_data': extra_data})
    
    def shutdown(self):
        """Gracefully shutdown logging system"""
        for handler in self.handlers.values():
            handler.close()
        
        # Log final shutdown message
        logger = self.get_logger('main')
        logger.info("Orchestrator logging system shutdown completed")

# Global logger instance
_orchestrator_logger = None

def get_orchestrator_logger(config: Dict[str, Any] = None) -> OrchestratorLogger:
    """Get or create the global orchestrator logger instance"""
    global _orchestrator_logger
    if _orchestrator_logger is None:
        _orchestrator_logger = OrchestratorLogger(config)
    return _orchestrator_logger

def setup_orchestrator_logging(config: Dict[str, Any] = None) -> logging.Logger:
    """Setup orchestrator logging and return main logger"""
    orchestrator_logger = get_orchestrator_logger(config)
    return orchestrator_logger.get_logger('main')

# Convenience functions for common logging patterns
def log_signal(signal_id: str, level: str, message: str, **kwargs):
    """Convenience function for logging signals"""
    logger = get_orchestrator_logger()
    logger.log_signal('signal_fusion', level, message, signal_id=signal_id, **kwargs)

def log_execution(execution_id: str, level: str, message: str, **kwargs):
    """Convenience function for logging executions"""
    logger = get_orchestrator_logger()
    logger.log_execution(execution_id, level, message, **kwargs)

def log_performance(component: str, metrics: Dict[str, Any], message: str = None):
    """Convenience function for logging performance"""
    logger = get_orchestrator_logger()
    logger.log_performance(component, metrics, message)

def log_health(component: str, health_data: Dict[str, Any], status: str):
    """Convenience function for logging health"""
    logger = get_orchestrator_logger()
    logger.log_health(component, health_data, status)

