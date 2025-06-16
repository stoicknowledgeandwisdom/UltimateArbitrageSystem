#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Manager
===============

A comprehensive manager for cryptocurrency arbitrage strategies.

This module provides functionality to:
1. Load and initialize strategies from configuration
2. Register custom strategies at runtime
3. Execute strategies in parallel with advanced scheduling
4. Track detailed performance metrics for each strategy
5. Optimize strategy allocation based on performance
6. Provide unified interfaces for strategy management
7. Handle strategy dependencies and conflicts
8. Implement failover mechanisms for strategy execution
9. Support strategy hot-swapping without system restart
10. Enable dynamic scaling of strategy resources
"""

import importlib
import logging
import threading
import time
import json
import traceback
import os
import sys
import signal
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union, TypeVar, Generic, Iterator, Type
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import multiprocessing as mp
from abc import ABC, abstractmethod
import inspect
import heapq
import itertools
import functools
import weakref

logger = logging.getLogger(__name__)

# Type variable for strategy return types
T = TypeVar('T')

# Constants
DEFAULT_MAX_WORKERS = 10
DEFAULT_EXECUTION_TIMEOUT = 60  # seconds
DEFAULT_OPPORTUNITY_BATCH_SIZE = 10
DEFAULT_METRICS_HISTORY_SIZE = 1000
DEFAULT_PROFIT_THRESHOLD = 0.5  # percentage
DEFAULT_EXECUTION_INTERVAL = 30  # seconds
DEFAULT_HEALTH_CHECK_INTERVAL = 300  # seconds
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_METRICS_PERSISTENCE_PATH = "data/metrics"


class StrategyError(Exception):
    """Base exception for all strategy-related errors."""
    pass


class StrategyValidationError(StrategyError):
    """Raised when a strategy configuration is invalid."""
    pass


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""
    pass


class StrategyTimeoutError(StrategyExecutionError):
    """Raised when strategy execution times out."""
    pass


class StrategyNotFoundError(StrategyError):
    """Raised when a requested strategy is not found."""
    pass


class StrategyStatus(Enum):
    """Status enum for strategy execution state."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"
    DISABLED = "disabled"
    RECOVERING = "recovering"
    OPTIMIZING = "optimizing"
    WARMUP = "warmup"
    COOLDOWN = "cooldown"


class ExecutionPriority(Enum):
    """Priority levels for strategy execution."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceType(Enum):
    """Types of resources that strategies might consume."""
    CPU = auto()
    MEMORY = auto()
    NETWORK = auto()
    EXCHANGE_API = auto()
    DATABASE = auto()


@dataclass
class ResourceUsage:
    """Tracks resource usage for strategies."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    network_requests_per_second: float = 0.0
    exchange_api_calls: Dict[str, int] = field(default_factory=dict)
    database_operations: int = 0


@dataclass
class ExecutionResult:
    """Result of a strategy execution."""
    strategy_id: str
    execution_id: str
    success: bool
    execution_time: float
    opportunities_found: int
    opportunities_executed: int
    profit: float
    timestamp: datetime
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    resource_usage: Optional[ResourceUsage] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpportunityResult:
    """Result of an individual opportunity execution."""
    opportunity_id: str
    strategy_id: str
    execution_id: str
    success: bool
    profit: float
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    exchanges_used: List[str] = field(default_factory=list)
    assets_involved: List[str] = field(default_factory=list)
    trade_volume: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySchedule:
    """Schedule configuration for strategy execution."""
    interval_seconds: int = DEFAULT_EXECUTION_INTERVAL
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    cooldown_after_execution: int = 0  # seconds to wait after execution
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT
    active_hours: Optional[List[Tuple[int, int]]] = None  # [(start_hour, end_hour), ...]
    active_days: Optional[List[int]] = None  # [0,1,2,3,4,5,6] for days of week
    batch_size: int = DEFAULT_OPPORTUNITY_BATCH_SIZE


class StrategyMetrics:
    """Tracks comprehensive performance metrics for an individual strategy."""
    
    def __init__(self, strategy_id: str, max_history: int = DEFAULT_METRICS_HISTORY_SIZE):
        """Initialize metrics tracking for a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            max_history: Maximum number of historical records to keep
        """
        self.strategy_id = strategy_id
        self.max_history = max_history
        
        # Core metrics
        self.execution_count = 0
        self.opportunity_count = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.consecutive_failures = 0
        self.total_profit = 0.0
        self.total_volume_traded = 0.0
        
        # Time tracking
        self.total_execution_time = 0.0
        self.last_execution_time = None
        self.start_time = None
        self.running_time = 0.0
        
        # Profit metrics
        self.last_profit = 0.0
        self.max_profit = 0.0
        self.min_profit = 0.0
        self.avg_profit = 0.0
        self.profit_variance = 0.0
        
        # History for time series analysis
        self.execution_history = deque(maxlen=max_history)
        self.profit_history = deque(maxlen=max_history)
        self.opportunity_history = deque(maxlen=max_history)
        self.execution_time_history = deque(maxlen=max_history)
        
        # Advanced metrics
        self.success_rate = 0.0
        self.opportunity_success_rate = 0.0
        self.profit_per_execution = 0.0
        self.profit_per_opportunity = 0.0
        self.profit_per_second = 0.0
        self.efficiency_score = 0.0
        self.avg_execution_time = 0.0
        self.hourly_performance = defaultdict(float)  # Hour -> Avg profit
        self.daily_performance = defaultdict(float)   # Day of week -> Avg profit
        self.resource_usage_history = deque(maxlen=max_history)
        
    def start_execution(self):
        """Mark the start of a strategy execution."""
        if self.start_time is None:
            self.start_time = datetime.now()
        self.last_execution_time = datetime.now()
        
    def record_execution_result(self, result: ExecutionResult):
        """Record metrics from an execution result.
        
        Args:
            result: Execution result object
        """
        self.execution_count += 1
        self.total_execution_time += result.execution_time
        self.opportunity_count += result.opportunities_found
        self.execution_history.append(result)
        self.execution_time_history.append(result.execution_time)
        
        timestamp = result.timestamp
        hour = timestamp.hour
        day = timestamp.weekday()
        
        # Update time-based performance metrics
        if result.success:
            self.successful_executions += 1
            self.consecutive_failures = 0
            self.hourly_performance[hour] = (self.hourly_performance[hour] * 0.9) + (result.profit * 0.1)
            self.daily_performance[day] = (self.daily_performance[day] * 0.9) + (result.profit * 0.1)
        else:
            self.failed_executions += 1
            self.consecutive_failures += 1
        
        # Record profit if available
        if result.profit != 0:
            self.record_profit(result.profit)
            
        # Record resource usage if available
        if result.resource_usage:
            self.resource_usage_history.append(result.resource_usage)
            
        # Update derived metrics
        self._update_derived_metrics()
        
    def record_profit(self, profit: float):
        """Record profit from a strategy execution.
        
        Args:
            profit: The profit amount (can be negative for losses)
        """
        self.last_profit = profit
        self.total_profit += profit
        self.profit_history.append(profit)
        
        # Update min/max profit
        if profit > self.max_profit:
            self.max_profit = profit
        if self.min_profit == 0 or profit < self.min_profit and profit != 0:
            self.min_profit = profit
            
    def record_opportunity_result(self, result: OpportunityResult):
        """Record the result of an individual opportunity.
        
        Args:
            result: Opportunity result object
        """
        self.opportunity_history.append(result)
        if result.success:
            self.total_volume_traded += result.trade_volume
        
    def update_running_time(self):
        """Update the total running time for the strategy."""
        if self.start_time:
            self.running_time = (datetime.now() - self.start_time).total_seconds()
            
    def _update_derived_metrics(self):
        """Update all derived metrics based on current values."""
        # Success rates
        total_executions = self.successful_executions + self.failed_executions
        self.success_rate = (self.successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        opportunity_success_count = sum(1 for result in self.opportunity_history if result.success)
        total_opportunities = len(self.opportunity_history)
        self.opportunity_success_rate = (opportunity_success_count / total_opportunities * 100) if total_opportunities > 0 else 0
        
        # Profit metrics
        profits = list(self.profit_history)
        if profits:
            self.avg_profit = sum(profits) / len(profits)
            # Calculate variance
            self.profit_variance = sum((p - self.avg_profit) ** 2 for p in profits) / len(profits)
        
        # Efficiency metrics
        self.profit_per_execution = self.total_profit / max(1, self.execution_count)
        self.profit_per_opportunity = self.total_profit / max(1, self.opportunity_count)
        self.profit_per_second = self.total_profit / max(1, self.total_execution_time)
        self.avg_execution_time = self.total_execution_time / max(1, self.execution_count)
        
        # Efficiency score: combines multiple metrics into a single value
        # Higher is better
        self.efficiency_score = (
            (self.success_rate / 100) * 
            (self.profit_per_second * 1000) * 
            (1 / (self.avg_execution_time + 1))
        )
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the strategy metrics.
        
        Returns:
            Dict containing all metrics
        """
        self.update_running_time()
        self._update_derived_metrics()
        
        return {
            "strategy_id": self.strategy_id,
            "execution_count": self.execution_count,
            "opportunity_count": self.opportunity_count,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": self.success_rate,
            "opportunity_success_rate": self.opportunity_success_rate,
            "total_profit": self.total_profit,
            "total_volume_traded": self.total_volume_traded,
            "average_profit": self.avg_profit,
            "profit_variance": self.profit_variance,
            "max_profit": self.max_profit,
            "min_profit": self.min_profit,
            "last_profit": self.last_profit,
            "profit_per_execution": self.profit_per_execution,
            "profit_per_opportunity": self.profit_per_opportunity,
            "profit_per_second": self.profit_per_second,
            "efficiency_score": self.efficiency_score,
            "running_time_seconds": self.running_time,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "avg_execution_time": self.avg_execution_time,
            "best_hour": max(self.hourly_performance.items(), key=lambda x: x[1])[0] if self.hourly_performance else None,
            "best_day": max(self.daily_performance.items(), key=lambda x: x[1])[0] if self.daily_performance else None
        }
        
    def get_time_series_data(self) -> Dict[str, List]:
        """Get time series data for visualization and analysis.
        
        Returns:
            Dict containing time series data
        """
        # Extract time series from execution history
        timestamps = [result.timestamp.isoformat() for result in self.execution_history]
        profits = [result.profit for result in self.execution_history]
        execution_times = [result.execution_time for result in self.execution_history]
        opportunities = [result.opportunities_found for result in self.execution_history]
        
        return {
            "timestamps": timestamps,
            "profits": profits,
            "execution_times": execution_times,
            "opportunities": opportunities,
            "hourly_performance": dict(self.hourly_performance),
            "daily_performance": dict(self.daily_performance)
        }
        
    def calculate_optimal_parameters(self) -> Dict[str, Any]:
        """Calculate optimal parameters based on historical performance.
        
        Returns:
            Dict with recommended parameter adjustments
        """
        recommendations = {}
        
        # Find the best performing hours
        if self.hourly_performance:
            best_hours = sorted(self.hourly_performance.items(), key=lambda x: x[1], reverse=True)
            recommendations["optimal_hours"] = [hour for hour, _ in best_hours[:3]]

