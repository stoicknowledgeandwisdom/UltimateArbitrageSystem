#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standardized Strategy Interface
===============================

A unified interface for arbitrage strategies within the UltimateArbitrageSystem.

This module provides a standardized framework for building arbitrage strategies,
supporting various types from triangular to n-dimensional cross-exchange arbitrage.
The interface is designed to be extensible, allowing for both zero-capital and
capital-based implementations with seamless integration to flash loan providers
and graph-based opportunity detection.

Features:
1. Common base class for all strategy types
2. Full lifecycle management (init, configure, start, execute, stop)
3. Unified profitability calculation with fees and slippage
4. Performance tracking and optimization
5. Dynamic configuration and hot-swapping
6. Inter-component communication protocol
7. Integration hooks for graph detector and flash loan systems
8. Comprehensive logging and metrics collection

The standardized interface ensures consistent behavior across different strategies
and enables easy expansion with new strategy types.
"""

import logging
import time
import json
import uuid
import os
import threading
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import copy
import importlib
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import signal
import weakref
import functools

# Set higher precision for Decimal calculations
getcontext().prec = 28

# Configure logging
logger = logging.getLogger("StrategyInterface")


# Type definitions
T = TypeVar('T')  # Generic type for strategy results


class StrategyType(Enum):
    """Types of arbitrage strategies supported by the system."""
    TRIANGULAR = auto()       # Three-currency cycle on a single exchange
    QUADRANGULAR = auto()     # Four-currency cycle on a single exchange
    CROSS_EXCHANGE = auto()   # Simple cross-exchange between two exchanges
    MULTI_HOP = auto()        # Multiple hops across exchanges
    FLASH_LOAN = auto()       # Zero-capital strategies using flash loans
    HYBRID = auto()           # Combination of multiple strategy types
    CUSTOM = auto()           # Custom strategy implementation
    STATISTICAL = auto()      # Statistical/ML-based strategy
    MARKET_MAKING = auto()    # Market making strategy
    GRID = auto()             # Grid trading strategy


class ExecutionMode(Enum):
    """Execution modes for strategies."""
    SIMULATION = "simulation"  # Log opportunities without trading
    REAL = "real"              # Execute real trades
    BACKTEST = "backtest"      # Run on historical data
    DRY_RUN = "dry_run"        # Test on testnet/sandbox


class CapitalRequirement(Enum):
    """Capital requirements for strategy execution."""
    ZERO_CAPITAL = "zero_capital"     # No capital required (flash loans)
    MINIMAL_CAPITAL = "minimal_capital"  # Small amount of capital required
    SIGNIFICANT_CAPITAL = "significant_capital"  # Significant capital needed
    DYNAMIC = "dynamic"               # Capital requirements vary with opportunity


class RiskProfile(Enum):
    """Risk profile of the strategy."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DYNAMIC = "dynamic"  # Risk level varies with market conditions


class StrategyStatus(Enum):
    """Status enum for strategy execution state."""
    IDLE = "idle"
    INITIALIZING = "initializing"
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


class MetricType(Enum):
    """Types of metrics that can be tracked for strategies."""
    PROFITABILITY = "profitability"
    EXECUTION_SPEED = "execution_speed"
    SUCCESS_RATE = "success_rate"
    VOLUME = "volume"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    CAPITAL_EFFICIENCY = "capital_efficiency"
    SLIPPAGE = "slippage"
    MARKET_IMPACT = "market_impact"
    OPPORTUNITY_COUNT = "opportunity_count"
    EXECUTION_COUNT = "execution_count"
    ERROR_RATE = "error_rate"


class StrategyError(Exception):
    """Base exception for all strategy-related errors."""
    pass


class ConfigurationError(StrategyError):
    """Error in strategy configuration."""
    pass


class ExecutionError(StrategyError):
    """Error during strategy execution."""
    pass


class ValidationError(StrategyError):
    """Error during strategy validation."""
    pass


class InitializationError(StrategyError):
    """Error during strategy initialization."""
    pass


class DependencyError(StrategyError):
    """Error with strategy dependencies."""
    pass


@dataclass
class ProfitCalculation:
    """Results of a profit calculation."""
    total_profit_amount: Decimal
    profit_percentage: Decimal
    volume: Decimal
    fees: Decimal
    slippage: Decimal
    execution_time_ms: Optional[int] = None
    gas_cost: Optional[Decimal] = None
    net_profit: Optional[Decimal] = None
    risk_score: Optional[Decimal] = None
    confidence_score: Optional[Decimal] = None
    price_impact: Optional[Decimal] = None
    expected_execution_paths: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_profit_amount": float(self.total_profit_amount),
            "profit_percentage": float(self.profit_percentage),
            "volume": float(self.volume),
            "fees": float(self.fees),
            "slippage": float(self.slippage),
            "execution_time_ms": self.execution_time_ms,
            "gas_cost": float(self.gas_cost) if self.gas_cost is not None else None,
            "net_profit": float(self.net_profit) if self.net_profit is not None else None,
            "risk_score": float(self.risk_score) if self.risk_score is not None else None,
            "confidence_score": float(self.confidence_score) if self.confidence_score is not None else None,
            "price_impact": float(self.price_impact) if self.price_impact is not None else None,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OpportunityDetails:
    """Detailed information about an arbitrage opportunity."""
    id: str
    strategy_type: StrategyType
    exchanges: List[str]
    symbols: List[str]
    currencies: List[str]
    profit_calculation: ProfitCalculation
    route: List[Dict[str, Any]]  # Detailed route information with exchange/pair/direction
    raw_data: Dict[str, Any]  # Original raw data from opportunity detection
    detection_timestamp: datetime
    expiry_timestamp: Optional[datetime] = None
    validation_time_ms: Optional[int] = None
    is_valid: bool = True
    invalidation_reason: Optional[str] = None
    priority_score: float = 0.0
    execution_difficulty: int = 1  # 1-10 scale
    capital_required: Optional[Decimal] = None
    zero_capital_compatible: bool = False
    flash_loan_compatible: bool = False
    cross_exchange: bool = False
    needs_preprocessing: bool = False
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the opportunity has expired."""
        if self.expiry_timestamp is None:
            return False
        return datetime.now() > self.expiry_timestamp


@dataclass
class ExecutionResult:
    """Result of a strategy execution."""
    opportunity_id: str
    strategy_id: str
    execution_id: str
    success: bool
    profit: Decimal
    volume: Decimal
    fees_paid: Decimal
    slippage: Decimal
    execution_time_ms: int
    status: StrategyStatus
    timestamp: datetime
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    transaction_ids: List[str] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "opportunity_id": self.opportunity_id,
            "strategy_id": self.strategy_id,
            "execution_id": self.execution_id,
            "success": self.success,
            "profit": float(self.profit),
            "volume": float(self.volume),
            "fees_paid": float(self.fees_paid),
            "slippage": float(self.slippage),
            "execution_time_ms": self.execution_time_ms,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "transaction_ids": self.transaction_ids,
            "metrics": self.metrics
        }
        return result


@dataclass
class StrategyMetrics:
    """Comprehensive metrics tracking for a strategy."""
    strategy_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_profit: Decimal = Decimal('0')
    total_loss: Decimal = Decimal('0')
    total_volume: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    total_slippage: Decimal = Decimal('0')
    total_gas_cost: Decimal = Decimal('0')
    total_execution_time_ms: int = 0
    opportunities_detected: int = 0
    opportunities_executed: int = 0
    success_rate: float = 0.0
    average_profit: Decimal = Decimal('0')
    average_loss: Decimal = Decimal('0')
    average_execution_time_ms: int = 0
    best_profit: Decimal = Decimal('0')
    worst_loss: Decimal = Decimal('0')
    profit_variance: float = 0.0
    last_execution_time: Optional[datetime] = None
    first_execution_time: Optional[datetime] = None
    running_time_seconds: int = 0
    last_metrics_update: datetime = field(default_factory=datetime.now)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    hourly_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    daily_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def update_from_execution(self, result: ExecutionResult) -> None:
        """Update metrics from an execution result."""
        self.total_executions += 1
        
        if result.success:
            self.successful_executions += 1
            self.total_profit += result.profit
        else:
            self.failed_executions += 1
            self.total_loss += abs(result.profit) if result.profit < 0 else Decimal('0')
        
        self.total_volume += result.volume
        self.total_fees += result.fees_paid
        self.total_slippage += result.slippage
        self.total_execution_time_ms += result.execution_time_ms
        
        # Update gas cost if available
        if 'gas_cost' in result.metrics:
            gas_cost = Decimal(str(result.metrics['gas_cost']))
            self.total_gas_cost += gas_cost
        
        # Update time-based metrics
        self.last_execution_time = result.timestamp
        if self.first_execution_time is None:
            self.first_execution_time = result.timestamp
        
        # Update success rate
        self.success_rate = self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0
        
        # Update averages
        self.average_profit = self.total_profit / self.successful_executions if self.successful_executions > 0 else Decimal('0')
        self.average_loss = self.total_loss / self.failed_executions if self.failed_executions > 0 else Decimal('0')
        self.average_execution_time_ms = self.total_execution_time_ms // self.total_executions if self.total_executions > 0 else 0
        
        # Update best/worst
        if result.success and result.profit > self.best_profit:
            self.best_profit = result.profit
        elif not result.success and result.profit < 0 and abs(result.profit) > self.worst_loss:
            self.worst_loss = abs(result.profit)
        
        # Update execution history (keep limited history)
        self.execution_history.append(result)
        if len(self.execution_history) > 1000:  # Limit history size
            self.execution_history.pop(0)
        
        # Update hourly metrics
        hour = result.timestamp.hour
        if hour not in self.hourly_metrics:
            self.hourly_metrics[hour] = {
                'executions': 0,
                'successful': 0,
                'profit': Decimal('0'),
                'volume': Decimal('0')
            }
        
        self.hourly_metrics[hour]['executions'] += 1
        if result.success:
            self.hourly_metrics[hour]['successful'] += 1
            self.hourly_metrics[hour]['profit'] += result.profit
        self.hourly_metrics[hour]['volume'] += result.volume
        
        # Update daily metrics
        day = result.timestamp.weekday()
        if day not in self.daily_metrics:
            self.daily_metrics[day] = {
                'executions': 0,
                'successful': 0,
                'profit': Decimal('0'),
                'volume': Decimal('0')
            }
        
        self.daily_metrics[day]['executions'] += 1
        if result.success:
            self.daily_metrics[day]['successful'] += 1
            self.daily_metrics[day]['profit'] += result.profit
        self.daily_metrics[day

