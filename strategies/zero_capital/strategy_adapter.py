#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Adapter Module
======================

A bridge component that connects the QuantumArbitrageStrategy with the core strategy management system.

This adapter serves as an integration layer between the specialized Quantum Arbitrage Strategy
and the generic strategy management infrastructure of the UltimateArbitrageSystem. It handles:

1. Configuration translation between system-level and strategy-specific parameters
2. Event propagation between the strategy and the broader system
3. Standardized error handling and logging
4. Metrics collection and reporting in a format consistent with the core system
5. Lifecycle management (initialization, execution, cleanup)
6. Dependency injection and management
7. Data format transformations for cross-component compatibility

The adapter pattern allows the Quantum Arbitrage Strategy to evolve independently
while maintaining a stable interface with the rest of the system.
"""

import logging
import time
import json
import threading
import uuid
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Type, TypeVar
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import traceback
import inspect
import asyncio
import functools
import copy

# Core system imports
from strategies.strategy_interface import (
    Strategy, StrategyType, ExecutionMode, CapitalRequirement, RiskProfile,
    StrategyStatus, ExecutionPriority, MetricType, ProfitCalculation, 
    OpportunityDetails, ExecutionResult, StrategyMetrics,
    StrategyError, ConfigurationError, ExecutionError, ValidationError, DependencyError
)

# Quantum strategy imports
from strategies.zero_capital.quantum_arbitrage import (
    QuantumArbitrageStrategy, MarketCondition, SafetyLevel, ExecutionStrategy
)

# Graph detector imports
from strategies.zero_capital.graph_detector import (
    MarketGraph, ArbitrageOpportunity, MarketInfo
)

# Flash loan imports
from integrations.defi.flash_loan import (
    FlashLoanProvider, FlashLoanParams, ArbitrageRoute, ArbitrageStep,
    ProtocolType, ChainType, FlashLoanResult, TransactionStatus
)

# Configure logging
logger = logging.getLogger("StrategyAdapter")

# Set higher precision for Decimal calculations
getcontext().prec = 28


class StrategyAdapter(Strategy):
    """
    Adapter class that implements the Strategy interface and delegates to a
    QuantumArbitrageStrategy instance, handling all necessary translations
    between the core system and the specialized strategy.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        """
        Initialize the Strategy Adapter.
        
        Args:
            strategy_id: Unique identifier for the strategy
            config: Configuration dictionary from the system
        """
        self.strategy_id = strategy_id
        self.system_config = config
        
        # Initialize basic attributes
        self.name = config.get("name", "Quantum Arbitrage Adapter")
        self.description = config.get("description", "Adapter for Quantum Arbitrage Strategy")
        self.version = config.get("version", "1.0.0")
        
        # Core components and state tracking
        self.quantum_strategy = None
        self.status = StrategyStatus.IDLE
        self.adapter_lock = threading.RLock()
        self.event_handlers = {}
        self.error_count = 0
        self.last_error_time = None
        self.metrics = StrategyMetrics(strategy_id=strategy_id)
        
        # Configuration and dependency tracking
        self.dependencies = {}
        self.configured = False
        self.initialized = False
        
        # Logging setup
        self._configure_logging(config.get("log_level", "INFO"))
        
        logger.info(f"Strategy Adapter initialized (ID: {strategy_id}, Version: {self.version})")
    
    def _configure_logging(self, log_level: str) -> None:
        """
        Configure adapter-specific logging.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, log_level, logging.INFO)
        logger.setLevel(numeric_level)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"strategy_adapter_{self.strategy_id}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)
    
    def translate_config(self) -> Dict[str, Any]:
        """
        Translate system configuration to quantum strategy configuration.
        
        Returns:
            Dict containing quantum strategy-specific configuration
        """
        # Extract strategy-specific configuration
        strategy_config = self.system_config.get("strategy_config", {})
        
        # Create a new configuration dictionary for the quantum strategy
        quantum_config = {
            "name": self.system_config.get("name", "Quantum Arbitrage"),
            "description": self.system_config.get("description", "Zero-capital multi-dimensional arbitrage strategy"),
            "version": self.system_config.get("version", "1.0.0"),
            
            # Map execution parameters
            "execution_mode": self.system_config.get("execution_mode", "simulation"),
            "max_concurrent_executions": strategy_config.get("max_concurrent_executions", 3),
            "execution_strategy": strategy_config.get("execution_strategy", "adaptive"),
            "safety_level": strategy_config.get("safety_level", "standard"),
            
            # Opportunity detection parameters
            "min_profit_threshold": strategy_config.get("min_profit_threshold", 0.005),
            "max_path_length": strategy_config.get("max_path_length", 5),
            "min_volume": strategy_config.get("min_volume", 100.0),
            "max_execution_time_ms": strategy_config.get("max_execution_time_ms", 5000),
            
            # Flash loan parameters
            "flash_loan_enabled": strategy_config.get("flash_loan_enabled", True),
            "preferred_protocols": strategy_config.get("preferred_protocols", ["aave_v3", "balancer"]),
            "preferred_chains": strategy_config.get("preferred_chains", ["ethereum", "polygon"]),
            "max_flash_loan_fee": strategy_config.get("max_flash_loan_fee", 0.001),
            
            # Routing parameters
            "use_cross_exchange": strategy_config.get("use_cross_exchange", True),
            "use_cross_chain": strategy_config.get("use_cross_chain", False),
            "min_liquidity_threshold": strategy_config.get("min_liquidity_threshold", 0.3),
            
            # Safety parameters
            "enable_fallback_routes": strategy_config.get("enable_fallback_routes", True),
            "max_slippage": strategy_config.get("max_slippage", 0.01),
            "gas_price_buffer": strategy_config.get("gas_price_buffer", 0.3),
            "emergency_stop_loss": strategy_config.get("emergency_stop_loss", 0.05),
            "max_failures_before_pause": strategy_config.get("max_failures_before_pause", 3),
            
            # Performance optimization
            "use_cached_routes": strategy_config.get("use_cached_routes", True),
            "opportunity_expiry_seconds": strategy_config.get("opportunity_expiry_seconds", 10),
            "route_cache_ttl_seconds": strategy_config.get("route_cache_ttl_seconds", 60),
            "prioritize_stable_pairs": strategy_config.get("prioritize_stable_pairs", True),
            
            # Advanced parameters
            "auto_tune_parameters": strategy_config.get("auto_tune_parameters", True),
            "performance_history_length": strategy_config.get("performance_history_length", 1000),
            "parameter_update_interval": strategy_config.get("parameter_update_interval", 3600),
            "market_condition_check_interval": strategy_config.get("market_condition_check_interval", 300),
            "log_level": self.system_config.get("log_level", "INFO"),
        }
        
        # Add any additional custom parameters
        for key, value in strategy_config.items():
            if key not in quantum_config:
                quantum_config[key] = value
        
        return quantum_config
    
    def translate_dependencies(self) -> Dict[str, Any]:
        """
        Translate system dependencies to quantum strategy dependencies.
        
        Returns:
            Dict containing quantum strategy-specific dependencies
        """
        # Map core system dependencies to quantum strategy dependencies
        quantum_dependencies = {}
        
        # Required dependencies mapping
        dependency_map = {
            "graph_detector": "market_graph",
            "flash_loan_manager": "flash_loan_manager",
            "exchange_manager": "exchange_manager",
            "market_data_provider": "market_data_provider",
            "risk_controller": "risk_controller"
        }
        
        # Extract and map dependencies
        for quantum_key, system_key in dependency_map.items():
            if system_key in self.dependencies:
                quantum_dependencies[quantum_key] = self.dependencies[system_key]
            else:
                logger.warning(f"Missing dependency mapping for {quantum_key} from {system_key}")
        
        return quantum_dependencies
    
    def set_dependencies(self, dependencies: Dict[str, Any]) -> None:
        """
        Set dependencies required by the strategy.
        
        Args:
            dependencies: Dictionary of dependency objects
        """
        logger.info(f"Setting dependencies for strategy adapter {self.strategy_id}")
        
        # Store dependencies at adapter level
        self.dependencies = dependencies
        
        # Validate if we have the minimum required dependencies
        required_deps = ["market_graph", "market_data_provider", "exchange_manager"]
        missing_deps = [dep for dep in required_deps if dep not in dependencies]
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            raise DependencyError(error_msg)
        
        # If quantum strategy exists, set its dependencies
        if self.quantum_strategy:
            try:
                quantum_deps = self.translate_dependencies()
                self.quantum_strategy.set_dependencies(quantum_deps)
                logger.info(f"Dependencies set for underlying quantum strategy")
            except Exception as e:
                error_msg = f"Error setting dependencies for quantum strategy: {str(e)}"
                logger.error(error_msg)
                raise DependencyError(error_msg) from e
    
    def initialize(self) -> bool:
        """
        Initialize the strategy, preparing it for execution.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        logger.info(f"Initializing strategy adapter {self.strategy_id}")
        
        try:
            with self.adapter_lock:
                self.status = StrategyStatus.INITIALIZING
                
                # Create quantum strategy if it doesn't exist
                if not self.quantum_strategy:
                    logger.info("Creating quantum strategy instance")
                    quantum_config = self.translate_config()
                    self.quantum_strategy = QuantumArbitrageStrategy(
                        strategy_id=self.strategy_id,
                        config=quantum_config
                    )
                    
                    # Set dependencies on quantum strategy
                    if self.dependencies:
                        quantum_deps = self.translate_dependencies()
                        self.quantum_strategy.set_dependencies(quantum_deps)
                
                # Initialize the quantum strategy
                logger.info("Initializing quantum strategy")
                if not self.quantum_strategy.initialize():
                    logger.error("Failed to initialize quantum strategy")
                    self.status = StrategyStatus.ERROR
                    return False
                
                # Setup event handlers
                self._setup_event_handlers()
                
                self.initialized = True
                self.status = StrategyStatus.IDLE
                logger.info(f"Strategy adapter {self.strategy_id} successfully initialized")
                return True
                
        except Exception as e:
            error_msg = f"Error initializing strategy adapter: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.status = StrategyStatus.ERROR
            self.last_error_time = datetime.now()
            self.error_count += 1
            return False
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for the quantum strategy."""
        if not self.quantum_strategy:
            logger.warning("Cannot set up event handlers: quantum strategy not initialized")
            return
            
        # Add event handlers for important quantum strategy events
        # Note: This assumes the quantum strategy has a mechanism to register event handlers
        # Implementation will depend on how the quantum strategy exposes events
        try:
            if hasattr(self.quantum_strategy, 'register_event_handler'):
                # Register handlers for different event types
                self.quantum_strategy.register_event_handler(
                    'opportunity_detected', 
                    self._on_opportunity_detected
                )
                self.quantum_strategy.register_event_handler(
                    'execution_started', 
                    self._on_execution_started
                )
                self.quantum_strategy.register_event_handler(
                    'execution_completed', 
                    self._on_execution_completed
                )
                self.quantum_strategy.register_event_handler(
                    'execution_error', 
                    self._on_execution_error
                )
                self.quantum_strategy.register_event_handler(
                    'status_changed',
                    self._on_status_changed
                )
                
                logger.info("Event handlers registered with quantum strategy")
            else:
                logger.warning("Quantum strategy does not support event registration")
        except Exception as e:
            logger.error(f"Error setting up event handlers: {str(e)}")
    
    def _on_opportunity_detected(self, opportunity: Any) -> None:
        """
        Handle opportunity detected event from quantum strategy.
        
        Args:
            opportunity: The detected opportunity
        """
        logger.debug(f"Opportunity detected: {opportunity.id if hasattr(opportunity, 'id') else 'Unknown'}")
        
        # Translate opportunity to system format if needed
        system_opportunity = self._translate_opportunity(opportunity)
        
        # Propagate to any registered system handlers
        self._propagate_event('opportunity_detected', system_opportunity)
    
    def _on_execution_started(self, execution_id: str, opportunity_id: str) -> None:
        """
        Handle execution started event from quantum strategy.
        
        Args:
            execution_id: Unique identifier for the execution
            opportunity_id: Identifier of the opportunity being executed
        """
        with self.adapter_lock:
            logger.info(f"Execution started: execution_id={execution_id}, opportunity_id={opportunity_id}")
            
            # Track execution start time and metrics
            start_time = datetime.now()
            execution_info = {
                'execution_id': execution_id,
                'opportunity_id': opportunity_id,
                'start_time': start_time,
                'status': 'in_progress',
                'retry_count': 0,
                'warning_level': 0,  # 0=none, 1=low, 2=medium, 3=high
            }
            
            # Store execution info for later reference (when execution completes or errors)
            # This could be stored in a database or in-memory cache in a production system
            if not hasattr(self, '_active_executions'):
                self._active_executions = {}
            self._active_executions[execution_id] = execution_info
            
            # Perform optional pre-execution risk checks
            risk_assessment = self._assess_execution_risk(opportunity_id)
            execution_info['risk_assessment'] = risk_assessment
            
            # Update metrics
            self._update_metrics(MetricType.EXECUTION_COUNT, 1)
            
            # If risk is too high, log a warning
            if risk_assessment.get('risk_level', 0) > 0.7:  # High risk threshold
                logger.warning(f"High risk execution: execution_id={execution_id}, risk={risk_assessment.get('risk_level')}")
                execution_info['warning_level'] = 3
            
            # Propagate event to registered handlers
            event_data = {
                'execution_id': execution_id,
                'opportunity_id': opportunity_id,
                'timestamp': start_time.isoformat(),
                'risk_assessment': risk_assessment
            }
            self._propagate_event('execution_started', event_data)
    
    def _on_execution_completed(self, execution_id: str, result: Any) -> None:
        """
        Handle execution completed event from quantum strategy.
        
        Args:
            execution_id: Unique identifier for the execution that completed
            result: The execution result object from the quantum strategy
        """
        with self.adapter_lock:
            logger.info(f"Execution completed: execution_id={execution_id}")
            
            # Retrieve stored execution info
            execution_info = self._get_execution_info(execution_id)
            if not execution_info:
                logger.warning(f"Execution info not found for completed execution: {execution_id}")
                execution_info = {'start_time': datetime.now() - timedelta(seconds=1)}
            
            # Calculate execution time
            start_time = execution_info.get('start_time', datetime.now() - timedelta(seconds=1))
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
            
            # Translate result to system format
            system_result = self._translate_execution_result(result, execution_id, execution_time)
            
            # Update execution info
            execution_info['status'] = 'completed'
            execution_info['end_time'] = datetime.now()
            execution_info['result'] = system_result
            
            # Update metrics based on execution result
            self._update_metrics(MetricType.EXECUTION_SPEED, execution_time)
            
            if system_result.success:
                self._update_metrics(MetricType.SUCCESS_RATE, 1)
                self._update_metrics(MetricType.PROFITABILITY, float(system_result.profit))
                self._update_metrics(MetricType.VOLUME, float(system_result.volume))
                logger.info(f"Successful execution: {execution_id}, profit={system_result.profit}, time={execution_time:.2f}ms")
            else:
                self._update_metrics(MetricType.ERROR_RATE, 1)
                logger.warning(f"Execution completed without success: {execution_id}, time={execution_time:.2f}ms")
            
            # Perform advanced performance analysis
            self._analyze_execution_performance(system_result, execution_info)
            
            # Clean up active executions tracking
            if hasattr(self, '_active_executions') and execution_id in self._active_executions:
                del self._active_executions[execution_id]
            
            # Propagate the event to registered handlers
            self._propagate_event('execution_completed', system_result)
    
    def _on_execution_error(self, execution_id: str, error: Any, error_details: Dict[str, Any] = None) -> None:
        """
        Handle execution error event from quantum strategy.
        
        Args:
            execution_id: Unique identifier for the execution that errored
            error: The error object or message
            error_details: Additional details about the error context
        """
        with self.adapter_lock:
            error_details = error_details or {}
            error_msg = str(error)
            error_type = error_details.get('type', 'unknown')
            
            logger.error(f"Execution error: execution_id={execution_id}, type={error_type}, error={error_msg}")
            
            # Retrieve stored execution info
            execution_info = self._get_execution_info(execution_id)
            if not execution_info:
                logger.warning(f"Execution info not found for failed execution: {execution_id}")
                execution_info = {
                    'execution_id': execution_id,
                    'start_time': datetime.now() - timedelta(seconds=5),
                    'retry_count': 0,
                    'status': 'unknown'
                }
            
            # Update execution info
            execution_info['status'] = 'error'
            execution_info['end_time'] = datetime.now()
            execution_info['error'] = error_msg
            execution_info['error_type'] = error_type
            execution_info['error_details'] = error_details
            
            # Calculate execution time
            start_time = execution_info.get('start_time', datetime.now() - timedelta(seconds=5))
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
            
            # Update error metrics
            self._update_metrics(MetricType.ERROR_RATE, 1)
            self.error_count += 1
            self.last_error_time = datetime.now()
            
            # Categorize error and determine if it's recoverable
            is_recoverable = self._categorize_error(error_type, error_msg, error_details)
            execution_info['is_recoverable'] = is_recoverable
            
            # Implement retry logic for recoverable errors if retry count is below threshold
            max_retries = self.system_config.get("max_retries", 3)
            if is_recoverable and execution_info['retry_count'] < max_retries:
                opportunity_id = execution_info.get('opportunity_id')
                if opportunity_id:
                    execution_info['retry_count'] += 1
                    logger.info(f"Retrying execution {execution_id} for opportunity {opportunity_id}, retry {execution_info['retry_count']}/{max_retries}")
                    
                    # Schedule retry with exponential backoff (would be async in production)
                    retry_delay = 2 ** execution_info['retry_count']  # exponential backoff
                    logger.info(f"Scheduling retry in {retry_delay} seconds")
                    
                    # In a real implementation, this would use a task queue or scheduler
                    # For now, just log the intent to retry
                    logger.info(f"Would retry execution {execution_id} in {retry_delay} seconds")
            else:
                # Clean up tracking for non-recoverable errors or max retries exceeded
                if hasattr(self, '_active_executions') and execution_id in self._active_executions:
                    del self._active_executions[execution_id]
            
            # Create execution result for error
            error_result = ExecutionResult(
                opportunity_id=execution_info.get('opportunity_id', 'unknown'),
                strategy_id=self.strategy_id,
                execution_id=execution_id,
                success=False,
                profit=Decimal('0'),
                volume=Decimal('0'),
                fees_paid=Decimal('0'),
                slippage=Decimal('0'),
                execution_time_ms=int(execution_time),
                status=StrategyStatus.ERROR,
                timestamp=datetime.now(),
                error_message=error_msg,
                error_traceback=error_details.get('traceback')
            )
            
            # Check if we need to trigger circuit breaker (too many errors)
            if self._should_trigger_circuit_breaker():
                logger.critical(f"Circuit breaker triggered: too many execution errors ({self.error_count})")
                self.status = StrategyStatus.ERROR
                # In a real system, this might notify operations or pause the strategy
            
            # Propagate the error event
            self._propagate_event('execution_error', error_result)
    
    def _on_status_changed(self, new_status: Any, previous_status: Any = None) -> None:
        """
        Handle status changed event from quantum strategy.
        
        Args:
            new_status: The new status of the quantum strategy
            previous_status: The previous status (if available)
        """
        with self.adapter_lock:
            # Convert quantum status to system status if needed
            if isinstance(new_status, str):
                try:
                    new_status = StrategyStatus(new_status)
                except ValueError:
                    # If the status string doesn't match the enum, use a default mapping
                    status_mapping = {
                        'ready': StrategyStatus.IDLE,
                        'running': StrategyStatus.RUNNING,
                        'error': StrategyStatus.ERROR,
                        'stopping': StrategyStatus.STOPPED,
                        'paused': StrategyStatus.PAUSED
                    }
                    new_status = status_mapping.get(new_status.lower(), StrategyStatus.IDLE)
            
            previous_system_status = self.status
            
            # Validate the status transition
            valid_transition = self._validate_status_transition(previous_system_status, new_status)
            
            if valid_transition:
                logger.info(f"Strategy status changing: {previous_system_status} -> {new_status}")
                self.status = new_status
                
                # Perform actions based on new status
                if new_status == StrategyStatus.ERROR:
                    logger.error("Strategy entered ERROR state")
                    # Additional error handling could be implemented here
                
                elif new_status == StrategyStatus.RUNNING:
                    logger.info("Strategy is now RUNNING")
                    # Reset error counters when transitioning to RUNNING
                    self.consecutive_failures = 0
                
                elif new_status == StrategyStatus.STOPPED:
                    logger.info("Strategy is now STOPPED")
                    # Perform cleanup actions
                    self._cleanup_resources()
                
                # Propagate the status change event
                self._propagate_event('status_changed', {
                    'previous_status': previous_system_status,
                    'new_status': new_status,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.warning(f"Invalid status transition: {previous_system_status} -> {new_status}")
    
    def start(self) -> bool:
        """
        Start the strategy, enabling it to begin identifying and executing
        arbitrage opportunities.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        logger.info(f"Starting strategy adapter {self.strategy_id}")
        
        try:
            with self.adapter_lock:
                # Check if already running
                if self.status == StrategyStatus.RUNNING:
                    logger.warning("Strategy is already running")
                    return True
                
                # Check if initialized
                if not self.initialized:
                    logger.error("Cannot start strategy: not initialized")
                    return False
                
                # Validate dependencies before starting
                if not self._validate_dependencies():
                    logger.error("Cannot start strategy: dependencies not valid")
                    return False
                
                # Set status to starting
                self.status = StrategyStatus.INITIALIZING
                
                # Pre-start warmup
                self._perform_warmup()
                
                # Start the quantum strategy
                if self.quantum_strategy:
                    logger.info("Starting quantum strategy")
                    if hasattr(self.quantum_strategy, 'start'):
                        if not self.quantum_strategy.start():
                            logger.error("Failed to start quantum strategy")
                            self.status = StrategyStatus.ERROR
                            return False
                
                # Set status to running (only if quantum strategy started successfully or has no start method)
                self.status = StrategyStatus.RUNNING
                
                # Notify any listeners that strategy has started
                self._propagate_event('strategy_started', {
                    'strategy_id': self.strategy_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Strategy adapter {self.strategy_id} successfully started")
                return True
                
        except Exception as e:
            error_msg = f"Error starting strategy adapter: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.status = StrategyStatus.ERROR
            self.last_error_time = datetime.now()
            self.error_count += 1
            return False
    
    def stop(self) -> bool:
        """
        Stop the strategy, halting all operations and cleaning up resources.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        logger.info(f"Stopping strategy adapter {self.strategy_id}")
        
        try:
            with self.adapter_lock:
                # Check if already stopped
                if self.status == StrategyStatus.STOPPED:
                    logger.warning("Strategy is already stopped")
                    return True
                
                # Set status to stopping
                self.status = StrategyStatus.STOPPING
                
                # Stop the quantum strategy
                if self.quantum_strategy:
                    logger.info("Stopping quantum strategy")
                    if hasattr(self.quantum_strategy, 'stop'):
                        try:
                            if not self.quantum_strategy.stop():
                                logger.error("Failed to stop quantum strategy cleanly")
                                # Continue anyway to clean up adapter resources
                        except Exception as e:
                            logger.error(f"Error stopping quantum strategy: {str(e)}")
                            # Continue anyway to clean up adapter resources
                
                # Clean up resources
                self._cleanup_resources()
                
                # Set status to stopped
                self.status = StrategyStatus.STOPPED
                
                # Notify any listeners that strategy has stopped
                self._propagate_event('strategy_stopped', {
                    'strategy_id': self.strategy_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Strategy adapter {self.strategy_id} successfully stopped")
                return True
                
        except Exception as e:
            error_msg = f"Error stopping strategy adapter: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # Still try to set status to stopped in case of error
            self.status = StrategyStatus.STOPPED
            self.last_error_time = datetime.now()
            self.error_count += 1
            return False
    
    def execute(self, opportunity: OpportunityDetails) -> ExecutionResult:
        """
        Execute a specific arbitrage opportunity.
        
        Args:
            opportunity: Details of the opportunity to execute
            
        Returns:
            ExecutionResult: Result of the execution
        """
        execution_id = str(uuid.uuid4())
        logger.info(f"Executing opportunity {opportunity.id} with execution ID {execution_id}")
        
        try:
            with self.adapter_lock:
                # Check if strategy is in a state that allows execution
                if self.status != StrategyStatus.RUNNING:
                    error_msg = f"Cannot execute opportunity: strategy is {self.status}, not RUNNING"
                    logger.error(error_msg)
                    return ExecutionResult(
                        opportunity_id=opportunity.id,
                        strategy_id=self.strategy_id,
                        execution_id=execution_id,
                        success=False,
                        profit=Decimal('0'),
                        volume=Decimal('0'),
                        fees_paid=Decimal('0'),
                        slippage=Decimal('0'),
                        execution_time_ms=0,
                        status=self.status,
                        timestamp=datetime.now(),
                        error_message=error_msg
                    )
                
                # Track execution start time and metrics
                start_time = datetime.now()
                execution_info = {
                    'execution_id': execution_id,
                    'opportunity_id': opportunity.id,
                    'start_time': start_time,
                    'status': 'in_progress',
                    'retry_count': 0,
                    'warning_level': 0,
                }
                
                # Store execution info
                if not hasattr(self, '_active_executions'):
                    self._active_executions = {}
                self._active_executions[execution_id] = execution_info
                
                # Perform risk assessment
                risk_assessment = self._assess_execution_risk(opportunity.id)
                execution_info['risk_assessment'] = risk_assessment
                
                # Check if risk is too high
                if risk_assessment.get('risk_level', 0) > 0.9:  # Critical risk threshold
                    error_msg = f"Execution risk too high: {risk_assessment.get('risk_level')}"
                    logger.error(error_msg)
                    
                    # Remove from active executions
                    del self._active_executions[execution_id]
                    
                    return ExecutionResult(
                        opportunity_id=opportunity.id,
                        strategy_id=self.strategy_id,
                        execution_id=execution_id,
                        success=False,
                        profit=Decimal('0'),
                        volume=Decimal('0'),
                        fees_paid=Decimal('0'),
                        slippage=Decimal('0'),
                        execution_time_ms=0,
                        status=StrategyStatus.ERROR,
                        timestamp=datetime.now(),
                        error_message=error_msg
                    )
            
            # Convert opportunity to quantum strategy format
            quantum_opportunity = self._translate_opportunity(opportunity, system_to_quantum=True)
            
            # Execute using the quantum strategy
            if self.quantum_strategy and hasattr(self.quantum_strategy, 'execute'):
                try:
                    # Release lock before calling potentially long-running method
                    # This allows other threads to access the adapter while execution is in progress
                    result = self.quantum_strategy.execute(quantum_opportunity, execution_id)
                    
                    # Re-acquire lock to update execution info
                    with self.adapter_lock:
                        # Calculate execution time
                        execution_time = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
                        
                        # Translate result to system format
                        system_result = self._translate_execution_result(result, execution_id, execution_time)
                        
                        # Update execution info
                        if execution_id in self._active_executions:
                            self._active_executions[execution_id]['status'] = 'completed'
                            self._active_executions[execution_id]['end_time'] = datetime.now()
                            self._active_executions[execution_id]['result'] = system_result
                        
                        # Update metrics
                        self._update_metrics(MetricType.EXECUTION_SPEED, execution_time)
                        if system_result.success:
                            self._update_metrics(MetricType.SUCCESS_RATE, 1)
                            self._update_metrics(MetricType.PROFITABILITY, float(system_result.profit))
                            self._update_metrics(MetricType.VOLUME, float(system_result.volume))
                        else:
                            self._update_metrics(MetricType.ERROR_RATE, 1)
                        
                        # Clean up active executions tracking
                        if execution_id in self._active_executions:
                            del self._active_executions[execution_id]
                        
                        return system_result
                        
                except Exception as e:
                    error_msg = f"Error in quantum strategy execution: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    
                    with self.adapter_lock:
                        # Update execution info
                        if execution_id in self._active_executions:
                            self._active_executions[execution_id]['status'] = 'error'
                            self._active_executions[execution_id]['end_time'] = datetime.now()
                            self._active_executions[execution_id]['error'] = error_msg
                        
                        # Calculate execution time
                        execution_time = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
                        
                        # Update error metrics
                        self._update_metrics(MetricType.ERROR_RATE, 1)
                        self.error_count += 1
                        self.last_error_time = datetime.now()
                        
                        # Clean up active executions tracking
                        if execution_id in self._active_executions:
                            del self._active_executions[execution_id]
                        
                        # Create and return error result
                        return ExecutionResult(
                            opportunity_id=opportunity.id,
                            strategy_id=self.strategy_id,
                            execution_id=execution_id,
                            success=False,
                            profit=Decimal('0'),
                            volume=Decimal('0'),
                            fees_paid=Decimal('0'),
                            slippage=Decimal('0'),
                            execution_time_ms=int(execution_time),
                            status=StrategyStatus.ERROR,
                            timestamp=datetime.now(),
                            error_message=error_msg,
                            error_traceback=traceback.format_exc()
                        )
            else:
                error_msg = "Quantum strategy doesn't have execute method"
                logger.error(error_msg)
                
                with self.adapter_lock:
                    # Clean up active executions tracking
                    if hasattr(self, '_active_executions') and execution_id in self._active_executions:
                        del self._active_executions[execution_id]
                
                return ExecutionResult(
                    opportunity_id=opportunity.id,
                    strategy_id=self.strategy_id,
                    execution_id=execution_id,
                    success=False,
                    profit=Decimal('0'),
                    volume=Decimal('0'),
                    fees_paid=Decimal('0'),
                    slippage=Decimal('0'),
                    execution_time_ms=0,
                    status=StrategyStatus.ERROR,
                    timestamp=datetime.now(),
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Error in adapter during execution: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Clean up active executions tracking
            with self.adapter_lock:
                if hasattr(self, '_active_executions') and execution_id in self._active_executions:
                    del self._active_executions[execution_id]
            
            return ExecutionResult(
                opportunity_id=opportunity.id,
                strategy_id=self.strategy_id,
                execution_id=execution_id,
                success=False,
                profit=Decimal('0'),
                volume=Decimal('0'),
                fees_paid=Decimal('0'),
                slippage=Decimal('0'),
                execution_time_ms=0,
                status=StrategyStatus.ERROR,
                timestamp=datetime.now(),
                error_message=error_msg,
                error_traceback=traceback.format_exc()
            )
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate the strategy configuration.
        
        Returns:
            Tuple containing:
                - bool: True if configuration is valid, False otherwise
                - List[str]: List of validation messages or errors
        """
        logger.info(f"Validating configuration for strategy adapter {self.strategy_id}")
        
        validation_messages = []
        is_valid = True
        
        try:
            with self.adapter_lock:
                # Basic adapter-level validation
                if not self.system_config:
                    validation_messages.append("Missing system configuration")
                    is_valid = False
                
                # Check for required configuration parameters
                required_configs = ["execution_mode", "name"]
                for config_key in required_configs:
                    if config_key not in self.system_config:
                        validation_messages.append(f"Missing required configuration parameter: {config_key}")
                        is_valid = False
                
                # Validate execution mode
                execution_mode = self.system_config.get("execution_mode")
                if execution_mode and execution_mode not in [mode.value for mode in ExecutionMode]:
                    validation_messages.append(f"Invalid execution mode: {execution_mode}")
                    is_valid = False
                
                # If quantum strategy exists, validate its configuration
                if self.quantum_strategy and hasattr(self.quantum_strategy, 'validate_configuration'):
                    # Convert any validation errors from quantum strategy to adapter-level messages
                    quantum_valid, quantum_messages = self.quantum_strategy.validate_configuration()
                    if not quantum_valid:
                        is_valid = False
                        for msg in quantum_messages:
                            validation_messages.append(f"Quantum strategy validation: {msg}")
                
                # Validate dependencies if already set
                if self.dependencies:
                    # Check that all required dependencies are present and valid
                    dependency_valid = self._validate_dependencies()
                    if not dependency_valid:
                        is_valid = False
                        validation_messages.append("Dependencies validation failed")
                
                # Advanced configuration validation
                advanced_validation_result = self._validate_advanced_configuration()
                if not advanced_validation_result[0]:
                    is_valid = False
                    validation_messages.extend(advanced_validation_result[1])
                
                # Log validation results
                if is_valid:
                    logger.info(f"Configuration validation passed for strategy adapter {self.strategy_id}")
                else:
                    logger.warning(f"Configuration validation failed for strategy adapter {self.strategy_id}: {validation_messages}")
                
                return is_valid, validation_messages
                
        except Exception as e:
            error_msg = f"Error validating configuration: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            validation_messages.append(error_msg)
            return False, validation_messages
    
    def _translate_opportunity(self, opportunity: Any, system_to_quantum: bool = False) -> Any:
        """
        Translate between quantum strategy opportunity format and system opportunity format.
        
        Args:
            opportunity: The opportunity to translate
            system_to_quantum: If True, convert from system to quantum format; 
                             otherwise, convert from quantum to system format
        
        Returns:
            Translated opportunity object
        """
        try:
            # From quantum to system format (default)
            if not system_to_quantum:
                # Check if it's already a system opportunity (prevent double conversion)
                if isinstance(opportunity, OpportunityDetails):
                    return opportunity
                
                # Assume it's an ArbitrageOpportunity from the quantum strategy
                if isinstance(opportunity, ArbitrageOpportunity):
                    # Convert quantum opportunity to system opportunity
                    profit_pct = float(opportunity.profit_pct) if hasattr(opportunity, 'profit_pct') else 0.0
                    volume_limit = float(opportunity.volume_limit) if hasattr(opportunity, 'volume_limit') else 0.0
                    
                    # Extract path information
                    path_description = []
                    route_data = []
                    currencies_involved = []
                    
                    if hasattr(opportunity, 'path') and opportunity.path:
                        for step in opportunity.path:
                            if len(step) >= 3:
                                exchange, base, quote = step[0], step[1], step[2]
                                path_description.append(f"{exchange}: {base}->{quote}")
                                currencies_involved.extend([base, quote])
                                
                                # Create route data for arbitrage details
                                route_data.append({
                                    'exchange': exchange,
                                    'base_currency': base,
                                    'quote_currency': quote,
                                    'rate': opportunity.market_rates.get(f"{base}/{quote}", 0.0) if hasattr(opportunity, 'market_rates') else 0.0
                                })
                    
                    # Remove duplicates from currencies list
                    currencies_involved = list(set(currencies_involved))
                    
                    # Create additional metadata
                    metadata = {
                        'confidence_score': float(opportunity.confidence_score) if hasattr(opportunity, 'confidence_score') else 0.0,
                        'slippage_estimate': float(opportunity.slippage_estimate_pct) if hasattr(opportunity, 'slippage_estimate_pct') else 0.0,
                        'estimated_execution_time_ms': opportunity.estimated_execution_time_ms if hasattr(opportunity, 'estimated_execution_time_ms') else 0,
                        'execution_difficulty': opportunity.execution_difficulty if hasattr(opportunity, 'execution_difficulty') else 0,
                        'market_data_timestamp': opportunity.timestamp.isoformat() if hasattr(opportunity, 'timestamp') else datetime.now().isoformat(),
                        'original_quantum_id': opportunity.id if hasattr(opportunity, 'id') else str(uuid.uuid4())
                    }
                    
                    # Create system opportunity
                    system_opportunity = OpportunityDetails(
                        id=getattr(opportunity, 'id', str(uuid.uuid4())),
                        strategy_id=self.strategy_id,
                        timestamp=getattr(opportunity, 'timestamp', datetime.now()),
                        profit_pct=Decimal(str(profit_pct)),
                        volume_limit=Decimal(str(volume_limit)),
                        path=path_description,
                        currencies=currencies_involved,
                        exchanges=getattr(opportunity, 'exchanges_involved', []),
                        priority=ExecutionPriority.HIGH if profit_pct > 0.01 else ExecutionPriority.MEDIUM,
                        route=route_data,
                        metadata=metadata
                    )
                    
                    return system_opportunity
                
                # If the type is unknown, try to extract basic info and create a system opportunity
                logger.warning(f"Unknown opportunity type for translation: {type(opportunity)}")
                
                # Try to extract minimal required information
                opp_id = getattr(opportunity, 'id', str(uuid.uuid4()))
                timestamp = getattr(opportunity, 'timestamp', datetime.now())
                profit = getattr(opportunity, 'profit_pct', Decimal('0'))
                
                return OpportunityDetails(
                    id=opp_id,
                    strategy_id=self.strategy_id,
                    timestamp=timestamp,
                    profit_pct=profit if isinstance(profit, Decimal) else Decimal(str(profit)),
                    volume_limit=Decimal('0'),
                    path=[],
                    currencies=[],
                    exchanges=[],
                    priority=ExecutionPriority.LOW,
                    route=[],
                    metadata={'warning': 'Created from unknown opportunity type'}
                )
            
            # From system to quantum format
            else:
                # Check if it's already a quantum opportunity
                if isinstance(opportunity, ArbitrageOpportunity):
                    return opportunity
                
                # Convert OpportunityDetails to ArbitrageOpportunity format
                if isinstance(opportunity, OpportunityDetails):
                    # Extract path information from route data
                    path = []
                    market_rates = {}
                    exchanges_involved = set()
                    currencies_involved = list(opportunity.currencies) if hasattr(opportunity, 'currencies') else []
                    
                    # Process route data if available
                    if hasattr(opportunity, 'route') and opportunity.route:
                        for step in opportunity.route:
                            if isinstance(step, dict) and 'exchange' in step and 'base_currency' in step and 'quote_currency' in step:
                                exchange = step['exchange']
                                base = step['base_currency']
                                quote = step['quote_currency']
                                rate = step.get('rate', 0.0)
                                
                                path.append((exchange, base, quote))
                                exchanges_involved.add(exchange)
                                market_rates[f"{base}/{quote}"] = Decimal(str(rate))
                    
                    # If no path was constructed from route data, try to parse the path strings
                    if not path and hasattr(opportunity, 'path') and opportunity.path:
                        for step_str in opportunity.path:
                            # Try to parse string like "exchange: base->quote"
                            parts = step_str.split(':')
                            if len(parts) == 2:
                                exchange = parts[0].strip()
                                currency_parts = parts[1].strip().split('->')
                                if len(currency_parts) == 2:
                                    base = currency_parts[0].strip()
                                    quote = currency_parts[1].strip()
                                    path.append((exchange, base, quote))
                                    exchanges_involved.add(exchange)
                    
                    # Extract metadata
                    metadata = opportunity.metadata if hasattr(opportunity, 'metadata') else {}
                    confidence_score = Decimal(str(metadata.get('confidence_score', 0.8)))
                    slippage_estimate = Decimal(str(metadata.get('slippage_estimate', 0.001)))
                    estimated_execution_time_ms = metadata.get('estimated_execution_time_ms', 500)
                    execution_difficulty = metadata.get('execution_difficulty', 5)
                    
                    # Calculate profit with slippage
                    profit_pct = opportunity.profit_pct
                    profit_with_slippage = profit_pct - slippage_estimate
                    
                    # Create ArbitrageOpportunity
                    quantum_opportunity = ArbitrageOpportunity(
                        id=opportunity.id,
                        path=path,
                        profit_pct=profit_pct,
                        volume_limit=opportunity.volume_limit,
                        fee_adjusted_profit=profit_pct * Decimal('0.99'),  # Simple approximation
                        timestamp=opportunity.timestamp,
                        exchanges_involved=list(exchanges_involved),
                        currencies_involved=currencies_involved,
                        path_length=len(path),
                        execution_difficulty=execution_difficulty,
                        confidence_score=confidence_score,
                        estimated_execution_time_ms=estimated_execution_time_ms,
                        slippage_estimate_pct=slippage_estimate,
                        profit_with_slippage=profit_with_slippage,
                        market_rates=market_rates
                    )
                    
                    return quantum_opportunity
                
                # If it's an unknown type, create a minimal quantum opportunity
                logger.warning(f"Unknown system opportunity type for translation to quantum: {type(opportunity)}")
                
                # Create a simple opportunity with minimal data
                return ArbitrageOpportunity(
                    id=getattr(opportunity, 'id', str(uuid.uuid4())),
                    path=[],
                    profit_pct=Decimal('0'),
                    volume_limit=Decimal('0'),
                    fee_adjusted_profit=Decimal('0'),
                    timestamp=datetime.now(),
                    exchanges_involved=[],
                    currencies_involved=[],
                    path_length=0,
                    execution_difficulty=10,  # Mark as difficult since we don't have proper data
                    confidence_score=Decimal('0.1'),  # Low confidence due to incomplete data
                    estimated_execution_time_ms=1000,
                    slippage_estimate_pct=Decimal('0.005'),
                    profit_with_slippage=Decimal('0'),
                    market_rates={}
                )
                
        except Exception as e:
            logger.error(f"Error translating opportunity: {str(e)}\n{traceback.format_exc()}")
            
            # Return a fallback opportunity depending on the direction
            if system_to_quantum:
                # Return minimal quantum opportunity
                return ArbitrageOpportunity(
                    id=str(uuid.uuid4()),
                    path=[],
                    profit_pct=Decimal('0'),
                    volume_limit=Decimal('0'),
                    fee_adjusted_profit=Decimal('0'),
                    timestamp=datetime.now(),
                    exchanges_involved=[],
                    currencies_involved=[],
                    path_length=0,
                    execution_difficulty=10,
                    confidence_score=Decimal('0.1'),
                    estimated_execution_time_ms=1000,
                    slippage_estimate_pct=Decimal('0.005'),
                    profit_with_slippage=Decimal('0'),
                    market_rates={}
                )
            else:
                # Return minimal system opportunity
                return OpportunityDetails(
                    id=str(uuid.uuid4()),
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    profit_pct=Decimal('0'),
                    volume_limit=Decimal('0'),
                    path=[],
                    currencies=[],
                    exchanges=[],
                    priority=ExecutionPriority.LOW,
                    route=[],
                    metadata={'error': 'Error during opportunity translation'}
                )
    
    def _translate_execution_result(self, result: Any, execution_id: str, execution_time_ms: float) -> ExecutionResult:
        """
        Translate quantum strategy execution result to system execution result.
        
        Args:
            result: The execution result from quantum strategy
            execution_id: The unique identifier for the execution
            execution_time_ms: The execution time in milliseconds
            
        Returns:
            ExecutionResult: The translated system execution result
        """
        try:
            # Default values
            opportunity_id = 'unknown'
            success = False
            profit = Decimal('0')
            volume = Decimal('0')
            fees_paid = Decimal('0')
            slippage = Decimal('0')
            status = StrategyStatus.COMPLETED
            error_message = None
            error_traceback = None
            
            # Extract execution info if available
            execution_info = self._get_execution_info(execution_id)
            if execution_info:
                opportunity_id = execution_info.get('opportunity_id', 'unknown')
            
            # Extract result data based on result type
            if hasattr(result, 'success'):
                success = result.success
            else:
                # Try to determine success from other attributes
                if hasattr(result, 'status'):
                    success = result.status in ['completed', 'success', 'executed']
                elif isinstance(result, dict):
                    success = result.get('success', False)
                    if not success and 'status' in result:
                        success = result['status'] in ['completed', 'success', 'executed']
            
            # Extract profit
            if hasattr(result, 'profit'):
                profit_val = result.profit
                profit = profit_val if isinstance(profit_val, Decimal) else Decimal(str(profit_val))
            elif isinstance(result, dict) and 'profit' in result:
                profit_val = result['profit']
                profit = profit_val if isinstance(profit_val, Decimal) else Decimal(str(profit_val))
            
            # Extract volume
            if hasattr(result, 'volume'):
                volume_val = result.volume
                volume = volume_val if isinstance(volume_val, Decimal) else Decimal(str(volume_val))
            elif isinstance(result, dict) and 'volume' in result:
                volume_val = result['volume']
                volume = volume_val if isinstance(volume_val, Decimal) else Decimal(str(volume_val))
            
            # Extract fees and slippage
            if hasattr(result, 'fees_paid'):
                fees_val = result.fees_paid
                fees_paid = fees_val if isinstance(fees_val, Decimal) else Decimal(str(fees_val))
            elif isinstance(result, dict) and 'fees_paid' in result:
                fees_val = result['fees_paid']
                fees_paid = fees_val if isinstance(fees_val, Decimal) else Decimal(str(fees_val))
            
            if hasattr(result, 'slippage'):
                slippage_val = result.slippage
                slippage = slippage_val if isinstance(slippage_val, Decimal) else Decimal(str(slippage_val))
            elif isinstance(result, dict) and 'slippage' in result:
                slippage_val = result['slippage']
                slippage = slippage_val if isinstance(slippage_val, Decimal) else Decimal(str(slippage_val))
            
            # Extract status
            if hasattr(result, 'status') and isinstance(result.status, StrategyStatus):
                status = result.status
            elif hasattr(result, 'status'):
                # Map string status to enum if needed
                status_str = str(result.status).lower()
                status_mapping = {
                    'completed': StrategyStatus.COMPLETED,
                    'running': StrategyStatus.RUNNING,
                    'error': StrategyStatus.ERROR,
                    'failed': StrategyStatus.ERROR,
                    'cancelled': StrategyStatus.STOPPED,
                    'pending': StrategyStatus.INITIALIZING
                }
                status = status_mapping.get(status_str, StrategyStatus.COMPLETED)
            
            # Extract error details
            if not success:
                if hasattr(result, 'error_message'):
                    error_message = result.error_message
                elif isinstance(result, dict) and 'error_message' in result:
                    error_message = result['error_message']
                
                if hasattr(result, 'error_traceback'):
                    error_traceback = result.error_traceback
                elif isinstance(result, dict) and 'error_traceback' in result:
                    error_traceback = result['error_traceback']
                
                # Set status to ERROR if there's an error and status isn't already ERROR
                if error_message and status != StrategyStatus.ERROR:
                    status = StrategyStatus.ERROR
            
            # Create and return execution result
            return ExecutionResult(
                opportunity_id=opportunity_id,
                strategy_id=self.strategy_id,
                execution_id=execution_id,
                success=success,
                profit=profit,
                volume=volume,
                fees_paid=fees_paid,
                slippage=slippage,
                execution_time_ms=int(execution_time_ms),
                status=status,
                timestamp=datetime.now(),
                error_message=error_message,
                error_traceback=error_traceback
            )
            
        except Exception as e:
            error_msg = f"Error translating execution result: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Return a minimal error result
            return ExecutionResult(
                opportunity_id=self._get_execution_info(execution_id).get('opportunity_id', 'unknown') if self._get_execution_info(execution_id) else 'unknown',
                strategy_id=self.strategy_id,
                execution_id=execution_id,
                success=False,
                profit=Decimal('0'),
                volume=Decimal('0'),
                fees_paid=Decimal('0'),
                slippage=Decimal('0'),
                execution_time_ms=int(execution_time_ms),
                status=StrategyStatus.ERROR,
                timestamp=datetime.now(),
                error_message=f"Error translating result: {str(e)}",
                error_traceback=traceback.format_exc()
            )
    
    def _propagate_event(self, event_type: str, event_data: Any) -> None:
        """
        Propagate an event to registered handlers.
        
        Args:
            event_type: Type of the event
            event_data: Data associated with the event
        """
        try:
            if event_type not in self.event_handlers:
                # No handlers for this event type
                return
            
            # Make a copy of handlers to avoid issues if handlers are modified during iteration
            handlers = list(self.event_handlers.get(event_type, []))
            
            # Call each handler
            for handler in handlers:
                try:
                    # Call handler outside of lock to avoid deadlocks
                    # This means handlers must be thread-safe
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {str(e)}\n{traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error propagating {event_type} event: {str(e)}\n{traceback.format_exc()}")
    
    def _update_metrics(self, metric_type: MetricType, value: float) -> None:
        """
        Update strategy metrics.
        
        Args:
            metric_type: Type of metric to update
            value: Value to update the metric with
        """
        try:
            with self.adapter_lock:
                # Update the appropriate metric based on type
                if metric_type == MetricType.EXECUTION_COUNT:
                    self.metrics.total_executions += 1
                    
                elif metric_type == MetricType.SUCCESS_RATE:
                    self.metrics.successful_executions += 1
                    # Recalculate success rate
                    if self.metrics.total_executions > 0:
                        self.metrics.success_rate = self.metrics.successful_executions / self.metrics.total_executions
                    
                elif metric_type == MetricType.ERROR_RATE:
                    self.metrics.failed_executions += 1
                    # Recalculate error rate
                    if self.metrics.total_executions > 0:
                        self.metrics.error_rate = self.metrics.failed_executions / self.metrics.total_executions
                    
                elif metric_type == MetricType.PROFITABILITY:
                    # Add profit to total
                    self.metrics.total_profit += Decimal(str(value))
                    
                    # Update rolling average
                    if not hasattr(self.metrics, 'profit_history'):
                        self.metrics.profit_history = []
                    
                    # Add to history with timestamp
                    self.metrics.profit_history.append((datetime.now(), value))
                    
                    # Keep only the latest N entries (configurable)
                    history_length = self.system_config.get('metrics_history_length', 100)
                    if len(self.metrics.profit_history) > history_length:
                        self.metrics.profit_history = self.metrics.profit_history[-history_length:]
                    
                    # Calculate average of recent profits
                    recent_profits = [p[1] for p in self.metrics.profit_history]
                    if recent_profits:
                        self.metrics.average_profit = sum(recent_profits) / len(recent_profits)
                    
                elif metric_type == MetricType.VOLUME:
                    # Add volume to total
                    self.metrics.total_volume += Decimal(str(value))
                    
                    # Track max volume
                    if Decimal(str(value)) > self.metrics.max_transaction_volume:
                        self.metrics.max_transaction_volume = Decimal(str(value))
                    
                    # Update average volume
                    if self.metrics.total_executions > 0:
                        self.metrics.average_volume = self.metrics.total_volume / self.metrics.total_executions
                    
                elif metric_type == MetricType.EXECUTION_SPEED:
                    # Track execution time (in milliseconds)
                    if not hasattr(self.metrics, 'execution_times'):
                        self.metrics.execution_times = []
                    
                    # Add execution time to history
                    self.metrics.execution_times.append((datetime.now(), value))
                    
                    # Keep only the latest N entries
                    history_length = self.system_config.get('metrics_history_length', 100)
                    if len(self.metrics.execution_times) > history_length:
                        self.metrics.execution_times = self.metrics.execution_times[-history_length:]
                    
                    # Calculate average and max execution times
                    recent_times = [t[1] for t in self.metrics.execution_times]
                    if recent_times:
                        self.metrics.average_execution_time_ms = sum(recent_times) / len(recent_times)
                        self.metrics.max_execution_time_ms = max(recent_times)
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}\n{traceback.format_exc()}")
    
    def _assess_execution_risk(self, opportunity_id: str) -> Dict[str, Any]:
        """
        Assess the risk of executing a specific opportunity.
        
        Args:
            opportunity_id: ID of the opportunity to assess
            
        Returns:
            Dict containing risk assessment details
        """
        try:
            # Initialize risk assessment
            risk_assessment = {
                'risk_level': 0.0,  # 0.0 to 1.0, with 1.0 being highest risk
                'risk_factors': [],
                'warnings': [],
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            # Check if the strategy is in a healthy state
            if self.status != StrategyStatus.RUNNING:
                risk_assessment['risk_level'] += 0.3
                risk_assessment['risk_factors'].append(f"Strategy is in {self.status} state")
                risk_assessment['warnings'].append(f"Strategy state is {self.status}, not RUNNING")
            
            # Check recent error rate
            if hasattr(self.metrics, 'error_rate') and self.metrics.error_rate > 0.2:
                risk_factor = min(self.metrics.error_rate * 2, 0.6)  # Cap at 0.6
                risk_assessment['risk_level'] += risk_factor
                risk_assessment['risk_factors'].append(f"High error rate: {self.metrics.error_rate:.2f}")
                risk_assessment['warnings'].append(f"Strategy has a high error rate of {self.metrics.error_rate:.2f}")
            
            # Check for recent errors
            if self.last_error_time and (datetime.now() - self.last_error_time).total_seconds() < 60:
                risk_assessment['risk_level'] += 0.2
                risk_assessment['risk_factors'].append("Recent error occurred")
                risk_assessment['warnings'].append(f"Recent error at {self.last_error_time.isoformat()}")
            
            # Check market volatility if market data provider is available
            if 'market_data_provider' in self.dependencies:
                try:
                    market_data = self.dependencies['market_data_provider']
                    if hasattr(market_data, 'get_market_volatility'):
                        volatility = market_data.get_market_volatility()
                        if volatility > 0.5:  # High volatility
                            risk_factor = min(volatility * 0.4, 0.4)  # Cap at 0.4
                            risk_assessment['risk_level'] += risk_factor
                            risk_assessment['risk_factors'].append(f"High market volatility: {volatility:.2f}")
                            risk_assessment['warnings'].append(f"Market volatility is high ({volatility:.2f})")
                except Exception as e:
                    logger.warning(f"Error checking market volatility: {str(e)}")
            
            # Circuit breaker check
            if self.error_count > self.system_config.get('max_failures_before_pause', 3):
                risk_assessment['risk_level'] += 0.5
                risk_assessment['risk_factors'].append(f"Error count threshold exceeded: {self.error_count}")
                risk_assessment['warnings'].append(f"Error count ({self.error_count}) exceeds threshold")
            
            # Cap risk level at 1.0
            risk_assessment['risk_level'] = min(risk_assessment['risk_level'], 1.0)
            
            # Add risk category
            if risk_assessment['risk_level'] < 0.3:
                risk_assessment['category'] = 'low'
            elif risk_assessment['risk_level'] < 0.7:
                risk_assessment['category'] = 'medium'
            else:
                risk_assessment['category'] = 'high'
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing execution risk: {str(e)}\n{traceback.format_exc()}")
            # Return a high risk assessment on error
            return {
                'risk_level': 0.8,
                'risk_factors': [f"Error during risk assessment: {str(e)}"],
                'warnings': ["Unable to properly assess risk due to error"],
                'category': 'high',
                'assessment_timestamp': datetime.now().isoformat()
            }
    
    def _get_execution_info(self, execution_id: str) -> Dict[str, Any]:
        """
        Retrieve information about a specific execution.
        
        Args:
            execution_id: ID of the execution to retrieve
            
        Returns:
            Dict containing execution information or None if not found
        """
        try:
            # Check if we have execution tracking
            if not hasattr(self, '_active_executions'):
                logger.debug(f"No active executions tracking for execution ID {execution_id}")
                return None
            
            # Look up execution in tracking dictionary
            if execution_id in self._active_executions:
                return self._active_executions[execution_id]
                
            # Not found
            logger.debug(f"Execution ID {execution_id} not found in active executions")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving execution info: {str(e)}")
            return None
    
    def _analyze_execution_performance(self, result: ExecutionResult, execution_info: Dict[str, Any]) -> None:
        """
        Analyze the performance of an execution for optimization.
        
        Args:
            result: The execution result
            execution_info: Additional execution tracking information
        """
        try:
            # Only analyze successful executions
            if not result.success:
                return
            
            # Calculate execution duration
            if 'start_time' in execution_info and 'end_time' in execution_info:
                duration = (execution_info['end_time'] - execution_info['start_time']).total_seconds()
                logger.debug(f"Execution {result.execution_id} took {duration:.3f} seconds")
            
            # Calculate profit percentage based on volume
            if result.volume > Decimal('0'):
                profit_percentage = (result.profit / result.volume) * Decimal('100')
                logger.info(f"Execution {result.execution_id} profit: {result.profit} ({profit_percentage:.2f}% of volume)")
            
            # Analyze slippage impact
            if result.slippage > Decimal('0'):
                slippage_percentage = (result.slippage / result.volume) * Decimal('100') if result.volume > Decimal('0') else Decimal('0')
                logger.info(f"Execution {result.execution_id} slippage: {result.slippage} ({slippage_percentage:.2f}% of volume)")
            
            # Check if performance warrants tuning of parameters
            # This would be more sophisticated in a production system
            if self.system_config.get("auto_tune_parameters", False):
                if result.execution_time_ms > 2000:  # Execution took longer than 2 seconds
                    logger.info(f"Execution {result.execution_id} was slow ({result.execution_time_ms} ms), considering parameter tuning")
                    # Here you would implement logic to adjust parameters for future executions
                
                if result.profit < Decimal('0.1') and result.volume > Decimal('100'):  # Low profit for reasonable volume
                    logger.info(f"Execution {result.execution_id} had low profit ({result.profit}) for volume {result.volume}, considering parameter tuning")
                    # Here you would implement logic to adjust profit thresholds or other parameters
            
        except Exception as e:
            logger.error(f"Error analyzing execution performance: {str(e)}\n{traceback.format_exc()}")
    
    def _categorize_error(self, error_type: str, error_message: str, error_details: Dict[str, Any]) -> bool:
        """
        Categorize an execution error and determine if it's recoverable.
        
        Args:
            error_type: Type or category of the error
            error_message: Error message text
            error_details: Additional error details
            
        Returns:
            bool: True if the error is recoverable, False otherwise
        """
        try:
            # Default to non-recoverable
            is_recoverable = False
            
            # Check for known recoverable errors
            recoverable_patterns = [
                "timeout", "connection", "network", "temporarily", "retry", 
                "throttled", "rate limit", "overloaded", "busy", "unavailable"
            ]
            
            # Categorization based on error type
            if error_type.lower() in ["network", "timeout", "connection", "rate_limit", "service_unavailable"]:
                is_recoverable = True
                logger.info(f"Categorized error as recoverable based on type: {error_type}")
            
            # Categorization based on error message content
            elif any(pattern in error_message.lower() for pattern in recoverable_patterns):
                is_recoverable = True
                logger.info(f"Categorized error as recoverable based on message: '{error_message}'")
            
            # Special case for flash loan errors which may be recoverable
            elif "flash_loan" in error_message.lower() and "revert" not in error_message.lower():
                is_recoverable = True
                logger.info(f"Categorized flash loan error as recoverable: '{error_message}'")
            
            # Check explicitly specified error properties
            elif error_details.get("is_recoverable", False):
                is_recoverable = True
                logger.info(f"Error marked as recoverable in error details")
            
            # Additional non-recoverable cases
            if error_type.lower() in ["validation", "configuration", "permission", "authentication", "funds", "critical"]:
                is_recoverable = False
                logger.info(f"Categorized error as non-recoverable based on type: {error_type}")
            
            # Fatal errors always override and are not recoverable
            if error_details.get("is_fatal", False) or "fatal" in error_message.lower():
                is_recoverable = False
                logger.info(f"Categorized error as non-recoverable (fatal)")
            
            # Log the categorization
            logger.debug(f"Error categorized as {'recoverable' if is_recoverable else 'non-recoverable'}: {error_type} - {error_message}")
            
            return is_recoverable
            
        except Exception as e:
            logger.error(f"Error categorizing error: {str(e)}\n{traceback.format_exc()}")
            # Default to non-recoverable if we can't determine
            return False
    
    def _should_trigger_circuit_breaker(self) -> bool:
        """
        Determine if a circuit breaker should be triggered due to errors.
        
        Returns:
            bool: True if circuit breaker should be triggered, False otherwise
        """
        try:
            # Get configuration thresholds
            max_errors = self.system_config.get('max_failures_before_pause', 3)
            error_window_seconds = self.system_config.get('error_window_seconds', 300)  # 5 minutes
            max_error_rate = self.system_config.get('max_error_rate', 0.3)  # 30% error rate threshold
            
            # Check for absolute error count threshold
            if self.error_count >= max_errors:
                logger.warning(f"Circuit breaker triggered: error count {self.error_count} >= threshold {max_errors}")
                return True
            
            # Check for error rate threshold
            if hasattr(self.metrics, 'error_rate') and self.metrics.error_rate >= max_error_rate:
                # Only trigger if we have enough executions to be statistically significant
                if self.metrics.total_executions >= 5:
                    logger.warning(f"Circuit breaker triggered: error rate {self.metrics.error_rate:.2f} >= threshold {max_error_rate}")
                    return True
            
            # Check for recent errors within time window
            recent_errors = 0
            if hasattr(self, '_active_executions'):
                now = datetime.now()
                for exec_info in self._active_executions.values():
                    if exec_info.get('status') == 'error' and 'end_time' in exec_info:
                        if (now - exec_info['end_time']).total_seconds() <= error_window_seconds:
                            recent_errors += 1
            
            # Check if recent errors exceed threshold
            recent_error_threshold = max(2, max_errors - 1)  # At least 2 errors
            if recent_errors >= recent_error_threshold:
                logger.warning(f"Circuit breaker triggered: {recent_errors} recent errors in {error_window_seconds}s")
                return True
            
            # No circuit breaker conditions met
            return False
            
        except Exception as e:
            logger.error(f"Error in circuit breaker check: {str(e)}\n{traceback.format_exc()}")
            # Default to triggering circuit breaker if we can't determine (safety first)
            return True
    
    def _validate_status_transition(self, previous_status: StrategyStatus, new_status: StrategyStatus) -> bool:
        """
        Validate if a status transition is allowed.
        
        Args:
            previous_status: The current/previous status
            new_status: The requested new status
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        try:
            # Define valid transitions for each status
            valid_transitions = {
                StrategyStatus.IDLE: [
                    StrategyStatus.INITIALIZING, 
                    StrategyStatus.RUNNING,
                    StrategyStatus.STOPPED,
                    StrategyStatus.ERROR
                ],
                StrategyStatus.INITIALIZING: [
                    StrategyStatus.IDLE,
                    StrategyStatus.RUNNING,
                    StrategyStatus.STOPPED,
                    StrategyStatus.ERROR
                ],
                StrategyStatus.RUNNING: [
                    StrategyStatus.STOPPING,
                    StrategyStatus.PAUSED,
                    StrategyStatus.ERROR
                ],
                StrategyStatus.PAUSED: [
                    StrategyStatus.RUNNING,
                    StrategyStatus.STOPPING,
                    StrategyStatus.STOPPED,
                    StrategyStatus.ERROR
                ],
                StrategyStatus.STOPPING: [
                    StrategyStatus.STOPPED,
                    StrategyStatus.ERROR
                ],
                StrategyStatus.STOPPED: [
                    StrategyStatus.INITIALIZING,
                    StrategyStatus.IDLE
                ],
                StrategyStatus.ERROR: [
                    StrategyStatus.STOPPED,
                    StrategyStatus.IDLE,
                    StrategyStatus.INITIALIZING
                ],
                StrategyStatus.COMPLETED: [
                    StrategyStatus.IDLE,
                    StrategyStatus.STOPPED
                ]
            }
            
            # Special case: same status is always valid
            if previous_status == new_status:
                return True
            
            # Check if the transition is valid
            valid = new_status in valid_transitions.get(previous_status, [])
            
            # Always allow transition to ERROR (safety mechanism)
            if new_status == StrategyStatus.ERROR:
                return True
            
            if not valid:
                logger.warning(f"Invalid status transition: {previous_status} -> {new_status}")
            
            return valid
        
        except Exception as e:
            logger.error(f"Error validating status transition: {str(e)}")
            # Allow the transition if we can't validate (but log the error)
            return True
    
    def _cleanup_resources(self) -> None:
        """
        Clean up any resources or connections held by the adapter.
        """
        logger.info(f"Cleaning up resources for strategy adapter {self.strategy_id}")
        
        try:
            with self.adapter_lock:
                # Clear execution tracking
                if hasattr(self, '_active_executions'):
                    logger.info(f"Clearing {len(self._active_executions)} active executions")
                    self._active_executions.clear()
                
                # Close any open connections or resources
                # This would depend on what resources the adapter manages
                
                # Stop any background tasks
                # For example, monitoring threads or scheduled tasks
                
                # Reset error tracking
                self.error_count = 0
                
                logger.info("Resource cleanup completed")
        
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}\n{traceback.format_exc()}")
    
    def _validate_dependencies(self) -> bool:
        """
        Validate all required dependencies for the strategy.
        
        Returns:
            bool: True if all dependencies are valid, False otherwise
        """
        logger.info(f"Validating dependencies for strategy adapter {self.strategy_id}")
        
        try:
            # List of required dependencies
            required_dependencies = [
                "market_graph",
                "market_data_provider",
                "exchange_manager"
            ]
            
            # Optional but recommended dependencies
            optional_dependencies = [
                "flash_loan_manager",
                "risk_controller",
                "analytics_engine"
            ]
            
            # Check for required dependencies
            missing_deps = []
            for dep in required_dependencies:
                if dep not in self.dependencies:
                    missing_deps.append(dep)
                    logger.error(f"Missing required dependency: {dep}")
            
            if missing_deps:
                return False
            
            # Log warning for missing optional dependencies
            for dep in optional_dependencies:
                if dep not in self.dependencies:
                    logger.warning(f"Missing optional dependency: {dep}")
            
            # Validate interfaces of dependencies
            for dep_name, dep_obj in self.dependencies.items():
                # Validate specific dependencies based on their expected interfaces
                if dep_name == "market_graph":
                    if not hasattr(dep_obj, 'find_arbitrage_opportunities'):
                        logger.error(f"Invalid market_graph dependency: missing find_arbitrage_opportunities method")
                        return False
                        
                elif dep_name == "market_data_provider":
                    if not hasattr(dep_obj, 'get_market_data'):
                        logger.error(f"Invalid market_data_provider dependency: missing get_market_data method")
                        return False
                        
                elif dep_name == "exchange_manager":
                    if not hasattr(dep_obj, 'execute_trade'):
                        logger.error(f"Invalid exchange_manager dependency: missing execute_trade method")
                        return False
            
            logger.info("All dependencies validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating dependencies: {str(e)}\n{traceback.format_exc()}")
            return False
    
    def _perform_warmup(self) -> None:
        """
        Perform warm-up actions before execution to improve performance.
        """
        logger.info(f"Performing warm-up actions for strategy adapter {self.strategy_id}")
        
        try:
            # Initialize metrics tracking if not already done
            if not hasattr(self.metrics, 'execution_times'):
                self.metrics.execution_times = []
            if not hasattr(self.metrics, 'profit_history'):
                self.metrics.profit_history = []
            
            # Initialize execution tracking
            if not hasattr(self, '_active_executions'):
                self._active_executions = {}
            
            # Pre-cache common data if market data provider is available
            if 'market_data_provider' in self.dependencies:
                market_data = self.dependencies['market_data_provider']
                
                # Get configuration for currencies to monitor
                monitored_currencies = self.system_config.get(
                    'monitored_currencies',
                    ['BTC', 'ETH', 'USDT', 'USDC', 'SOL', 'ADA']
                )
                
                logger.info(f"Pre-caching market data for {len(monitored_currencies)} currencies")
                
                # Pre-fetch market data for commonly used currency pairs
                if hasattr(market_data, 'get_market_data'):
                    for base in monitored_currencies:
                        for quote in monitored_currencies:
                            if base != quote:
                                try:
                                    # This would trigger caching in the market data provider
                                    market_data.get_market_data(base, quote)
                                except Exception as e:
                                    logger.debug(f"Error pre-caching {base}/{quote}: {str(e)}")
            
            # Warm up the quantum strategy if it has a warm-up method
            if self.quantum_strategy and hasattr(self.quantum_strategy, 'warm_up'):
                logger.info("Warming up quantum strategy")
                self.quantum_strategy.warm_up()
            
            logger.info("Warm-up completed successfully")
            
        except Exception as e:
            logger.error(f"Error during warm-up: {str(e)}\n{traceback.format_exc()}")
            # Warm-up failures shouldn't prevent the strategy from running
    
    def _validate_advanced_configuration(self) -> Tuple[bool, List[str]]:
        """
        Perform advanced validation of strategy configuration.
        
        Returns:
            Tuple containing:
                - bool: True if configuration is valid, False otherwise
                - List[str]: List of validation messages or errors
        """
        logger.info(f"Validating advanced configuration for strategy adapter {self.strategy_id}")
        
        validation_messages = []
        is_valid = True
        
        try:
            with self.adapter_lock:
                # Get strategy-specific configuration
                strategy_config = self.system_config.get("strategy_config", {})
                
                # Validate profit threshold
                min_profit_threshold = strategy_config.get("min_profit_threshold", 0.005)
                if min_profit_threshold < 0 or min_profit_threshold > 0.1:
                    validation_messages.append(f"Invalid min_profit_threshold: {min_profit_threshold}. Must be between 0 and 0.1")
                    is_valid = False
                
                # Validate path length
                max_path_length = strategy_config.get("max_path_length", 5)
                if not isinstance(max_path_length, int) or max_path_length < 2 or max_path_length > 10:
                    validation_messages.append(f"Invalid max_path_length: {max_path_length}. Must be an integer between 2 and 10")
                    is_valid = False
                
                # Validate min_volume
                min_volume = strategy_config.get("min_volume", 100.0)
                if not isinstance(min_volume, (int, float)) or min_volume <= 0 or min_volume >= 1000000:
                    validation_messages.append(f"Invalid min_volume: {min_volume}. Must be a positive number less than 1,000,000")
                    is_valid = False
                
                # Validate max_execution_time_ms
                max_execution_time_ms = strategy_config.get("max_execution_time_ms", 5000)
                if not isinstance(max_execution_time_ms, int) or max_execution_time_ms < 100 or max_execution_time_ms > 30000:
                    validation_messages.append(f"Invalid max_execution_time_ms: {max_execution_time_ms}. Must be an integer between 100 and 30,000")
                    is_valid = False
                
                # Validate safety_level
                safety_level = strategy_config.get("safety_level", "standard")
                valid_safety_levels = ["low", "standard", "high"]
                if safety_level not in valid_safety_levels:
                    validation_messages.append(f"Invalid safety_level: {safety_level}. Must be one of {valid_safety_levels}")
                    is_valid = False
                
                # Validate market_condition_check_interval
                market_condition_check_interval = strategy_config.get("market_condition_check_interval", 300)
                if not isinstance(market_condition_check_interval, int) or market_condition_check_interval < 60 or market_condition_check_interval > 3600:
                    validation_messages.append(f"Invalid market_condition_check_interval: {market_condition_check_interval}. Must be an integer between 60 and 3600 seconds")
                    is_valid = False
                
        except Exception as e:
            logger.error(f"Error during advanced configuration validation: {str(e)}")
            validation_messages.append(f"Unexpected error during advanced configuration validation: {str(e)}")
            is_valid = False
        
        return is_valid, validation_messages
