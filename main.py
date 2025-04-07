#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate Arbitrage System
A comprehensive cryptocurrency arbitrage system that leverages multiple
strategies for profit generation with sophisticated risk management.

This is the main entry point for the application.
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create required directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data/historical", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configure logging
log_filename = os.path.join("logs", f"arbitrage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("UltimateArbitrageSystem")

# Import system components
try:
    from exchanges.exchange_manager import ExchangeManager
    from strategies.strategy_manager import StrategyManager
    from risk_management.risk_controller import RiskController
    from data.market_data import MarketDataProvider
    from core.arbitrage_core.trading_engine import ArbitrageEngine
except ImportError as e:
    logger.critical(f"Failed to import core components: {e}")
    logger.critical("Please ensure all required modules are installed and in the correct directory structure.")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


class ArbitrageSystem:
    """Main class that coordinates all components of the arbitrage system."""
    
    def __init__(self, config_path: str = "config/system_config.json", test_mode: bool = False):
        """Initialize the arbitrage system.
        
        Args:
            config_path: Path to the system configuration file
            test_mode: If True, run in test mode without executing actual trades
        """
        logger.info("Initializing Ultimate Arbitrage System...")
        logger.info(f"Test mode: {'Enabled' if test_mode else 'Disabled'}")
        
        # Initialize state variables
        self.running = False
        self.test_mode = test_mode
        self.config = self._load_config(config_path)
        self.components_initialized = False
        self.monitoring_thread = None
        self.data_backup_thread = None
        self.health_check_thread = None
        self.exchange_manager = None
        self.exchange_manager = None
        self.market_data = None
        self.risk_controller = None
        self.strategy_manager = None
        self.trading_engine = None
        self.optimizer = None
        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "total_profit": 0.0,
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "open_positions": 0,
            "latency_ms": {},
            "opportunities_found": 0,
            "opportunities_executed": 0,
            "strategy_performance": {}
        }
        
        try:
            # Configure logging level from config
            if "log_level" in self.config:
                log_level = getattr(logging, self.config["log_level"], logging.INFO)
                logger.setLevel(log_level)
                logger.info(f"Log level set to {self.config['log_level']}")
            
            # Add optimizer parameters to config if they don't exist
            if "optimizer_parameters" not in self.config:
                self.config["optimizer_parameters"] = {
                    "reallocation_interval": 3600,  # 1 hour in seconds
                    "min_weight": 0.05,
                    "max_weight": 0.5,
                    "performance_metrics_weight": {
                        "profit": 0.4,
                        "win_rate": 0.2, 
                        "sharpe_ratio": 0.3,
                        "execution_time": 0.1
                    },
                    "market_condition_features": {
                        "volatility": True,
                        "trend": True,
                        "volume": True,
                        "liquidity": True
                    }
                }
                logger.info("Added default optimizer parameters to configuration")
            
            # Initialize system components
            self._initialize_components()
            
        except Exception as e:
            logger.critical(f"Failed to initialize system components: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing exchange manager...")
            self.exchange_manager = ExchangeManager(self.config.get("exchanges", []))
            
            logger.info("Initializing market data provider...")
            self.market_data = MarketDataProvider(self.exchange_manager)
            
            logger.info("Initializing risk controller...")
            risk_params = self.config.get("risk_parameters", {})
            if self.test_mode:
                # Use more conservative risk parameters in test mode
                logger.info("Using conservative risk parameters for test mode")
                for param in ["max_exposure_per_trade", "max_daily_loss"]:
                    if param in risk_params:
                        risk_params[param] = risk_params[param] * 0.5
            self.risk_controller = RiskController(risk_params)
            
            logger.info("Initializing strategy manager...")
            strategies = self.config.get("strategies", [])
            # Only use enabled strategies
            enabled_strategies = [s for s in strategies if s.get("enabled", True)]
            if self.test_mode:
                # In test mode, only use strategies marked as test_safe
                enabled_strategies = [s for s in enabled_strategies if s.get("test_safe", True)]
            
            logger.info(f"Loading {len(enabled_strategies)} enabled strategies...")
            self.strategy_manager = StrategyManager(
                enabled_strategies,
                self.exchange_manager,
                self.market_data,
                self.risk_controller
            )
            
            logger.info("Initializing arbitrage engine...")
            engine_config = {
                "test_mode": self.test_mode,
                "max_parallel_executions": self.config.get("max_parallel_executions", 3),
                "execution_delay": self.config.get("execution_delay", 0.5),
                "profit_threshold_multiplier": 1.5 if self.test_mode else 1.0
            }
            self.trading_engine = ArbitrageEngine(
                self.exchange_manager,
                self.strategy_manager,
                self.risk_controller,
                engine_config
            )
            
            # Initialize the strategy optimizer if available
            if AI_OPTIMIZATION_AVAILABLE:
                logger.info("Initializing AI strategy optimizer...")
                optimizer_config = self.config.get("optimizer_parameters", {})
                self.optimizer = StrategyOptimizer(
                    self.strategy_manager,
                    self.risk_controller,
                    optimizer_config
                )
                logger.info("AI strategy optimizer initialized successfully")
            
            self.components_initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.critical(f"Error during component initialization: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration data
        """
        logger.info(f"Loading configuration from {config_path}")
        
        # Default configuration
        default_config = {
            "system_name": "Ultimate Arbitrage System",
            "version": "1.0.0",
            "log_level": "INFO",
            "max_parallel_executions": 3,
            "execution_delay": 0.5,
            "monitoring_interval": 10,
            "data_backup_interval": 3600,
            "health_check_interval": 60,
            "exchanges": [],
            "strategies": [],
            "risk_parameters": {
                "max_exposure_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_open_positions": 3,
                "min_profit_threshold": 0.005,
                "slippage_tolerance": 0.002,
                "max_position_duration": 3600
            }
        }
        
        try:
            config_file = Path(config_path)
            
            # Check if config file exists
            if not config_file.exists():
                logger.warning(f"Configuration file {config_path} not found")
                logger.warning("Using default configuration")
                return default_config
                
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate essential configuration settings
            self._validate_config(config)
            
            # Merge with default config for any missing values
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    # For nested dicts, add any missing keys
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            logger.warning("Using default configuration")
            return default_config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Using default configuration")
            return default_config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration settings.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required sections
        for section in ["exchanges", "strategies", "risk_parameters"]:
            if section not in config:
                logger.warning(f"Missing required configuration section: {section}")
        
        # Check exchanges configuration
        if "exchanges" in config:
            for i, exchange in enumerate(config["exchanges"]):
                if "id" not in exchange:
                    logger.warning(f"Exchange at index {i} missing 'id' field")
                if "type" not in exchange:
                    logger.warning(f"Exchange at index {i} missing 'type' field")
                if "api_key" not in exchange and not self.test_mode:
                    logger.warning(f"Exchange {exchange.get('id', f'at index {i}')} missing 'api_key' field")
        
        # Check strategies configuration
        if "strategies" in config:
            for i, strategy in enumerate(config["strategies"]):
                if "id" not in strategy:
                    logger.warning(f"Strategy at index {i} missing 'id' field")
                if "type" not in strategy:
                    logger.warning(f"Strategy at index {i} missing 'type' field")
    
    def start(self) -> bool:
        """Start the arbitrage system.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("System is already running")
            return True
        
        if not self.components_initialized:
            logger.error("Cannot start system: components not initialized")
            return False
        
        try:
            logger.info("Starting arbitrage system...")
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Connect to exchanges
            logger.info("Connecting to exchanges...")
            exchange_connect_success = self.exchange_manager.connect_all()
            
            if not exchange_connect_success and not self.test_mode:
                logger.error("Failed to connect to any exchanges")
                return False
            
            # Update total capital in risk controller
            balances = self.exchange_manager.get_balances()
            total_capital = self._calculate_total_capital(balances)
            self.risk_controller.update_total_capital(total_capital)
            
            # Check if we have enough capital to operate
            min_required_capital = self.config.get("min_required_capital", 0)
            if total_capital < min_required_capital and not self.test_mode:
                logger.error(f"Insufficient capital: ${total_capital:.2f} available, ${min_required_capital:.2f} required")
                return False
            
            # Start market data collection
            logger.info("Starting market data collection...")
            self.market_data.start_data_collection()
            
            # Add monitored symbols based on strategies
            self._register_monitored_symbols()
            
            # Initialize thread control
            self.running = True
            self.performance_metrics["start_time"] = datetime.now()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            # Initialize optimizer with current market data if available
            if AI_OPTIMIZATION_AVAILABLE and self.optimizer:
                logger.info("Running initial strategy optimization...")
                self._optimize_strategy_allocation()
                logger.info("Initial strategy optimization completed")
            
            # Start trading engine
            logger.info("Starting trading engine...")
            engine_start_success = self.trading_engine.start()
            
            if not engine_start_success:
                logger.error("Failed to start trading engine")
                self.stop()
                return False
                
            logger.info("Arbitrage system started successfully")
            
            # If not running from a script, keep main thread alive
            if __name__ == "__main__":
                self._main_loop()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start arbitrage system: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.stop()
            return False
    
    def _optimize_strategy_allocation(self) -> None:
        """Run the strategy optimizer to update strategy weights."""
        if not AI_OPTIMIZATION_AVAILABLE or not self.optimizer:
            return
            
        try:
            # Get current market conditions and strategy performance
            market_data = self.market_data.get_latest_market_data()
            
            # Run optimization to get new strategy weights
            new_weights = self.optimizer.get_strategy_weights(market_data)
            
            # Log optimization results
            logger.info(f"Strategy optimizer recommended new weights: {new_weights}")
            
            # Update strategy weights in strategy manager
            if hasattr(self.strategy_manager, 'update_strategy_weights'):
                self.strategy_manager.update_strategy_weights(new_weights)
                logger.info("Strategy weights updated successfully")
            else:
                logger.warning("Strategy manager does not support weight updates")
                
        except Exception as e:
            logger.error(f"Error during strategy optimization: {e}")
            logger.error(traceback.format_exc())
            
    def _main_loop(self) -> None:
        """Main execution loop that keeps the program running."""
        try:
            optimization_interval = self.config.get("optimization_interval", 3600)  # Default 1 hour
            last_optimization = time.time()
            
            while self.running:
                # Periodically run the strategy optimizer
                current_time = time.time()
                if AI_OPTIMIZATION_AVAILABLE and self.optimizer and \
                   (current_time - last_optimization) > optimization_interval:
                    logger.info("Running scheduled strategy optimization...")
                    self._optimize_strategy_allocation()
                    last_optimization = current_time
                    
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.stop()
    
    def _optimizer_loop(self) -> None:
        """Background thread for running the strategy optimizer."""
        try:
            # Get optimization interval from config (in seconds)
            interval = self.config.get("optimizer_parameters", {}).get("reallocation_interval", 3600)
            
            while self.running:
                # Sleep for the specified interval
                time.sleep(interval)
                
                # Skip if system is not fully initialized
                if not self.components_initialized:
                    continue
                    
                # Run optimization
                logger.info(f"Running scheduled strategy optimization (interval: {interval}s)...")
                self._optimize_strategy_allocation()
                
        except Exception as e:
            logger.error(f"Error in optimizer loop: {e}")
            logger.error(traceback.format_exc())
    
    def _start_monitoring_threads(self) -> None:
        """Start all monitoring threads."""
        # Start performance monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MonitoringThread",
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start data backup thread if enabled
        if self.config.get("data_backup_enabled", True):
            self.data_backup_thread = threading.Thread(
                target=self._data_backup_loop,
                name="DataBackupThread",
                daemon=True
            )
            self.data_backup_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthCheckThread",
            daemon=True
        )
        self.health_check_thread.start()
        
        # Start optimizer thread if available
        if AI_OPTIMIZATION_AVAILABLE and self.optimizer:
            self.optimizer_thread = threading.Thread(
                target=self._optimizer_loop,
                name="OptimizerThread",
                daemon=True
            )
            self.optimizer_thread.start()
    
    def stop(self) -> None:
        """Stop the arbitrage system and clean up resources."""
        if not self.running:
            logger.info("System is not running")
            return
        
        logger.info("Stopping arbitrage system...")
        self.running = False
        
        try:
            # Stop trading engine
            if self.trading_engine:
                logger.info("Stopping trading engine...")
                self.trading_engine.stop()
            
            # Stop market data collection
            if self.market_data:
                logger.info("Stopping market data collection...")
                self.market_data.stop_data_collection()
            
            # Disconnect from exchanges
            if self.exchange_manager:
                logger.info("Disconnecting from exchanges...")
                self.exchange_manager.disconnect_all()
            
            # Wait for threads to terminate
            self._wait_for_threads()
            
            # Save performance metrics

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate Arbitrage System
-------------------------
Main entry point for the Ultimate Arbitrage System.
This file initializes all system components and manages the application lifecycle.

Features:
- Loads configuration from system_config.json
- Initializes trading engine, strategies, and risk management
- Provides system monitoring and performance tracking
- Handles graceful shutdown procedures
- Implements AI-enhanced strategy optimization
- Provides data persistence and comprehensive reporting
- Includes advanced error recovery mechanisms
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
import traceback
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

# Add project root to path to enable absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import system components
from core.arbitrage_core.trading_engine import ArbitrageEngine
from strategies.strategy_manager import StrategyManager
from exchanges.exchange_manager import ExchangeManager
from risk_management.risk_controller import RiskController
from data.market_data import MarketDataProvider

# Try to import optional AI components
try:
    from ai.optimizer import StrategyOptimizer
    AI_OPTIMIZATION_AVAILABLE = True
except ImportError:
    AI_OPTIMIZATION_AVAILABLE = False

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, log_rotation: bool = True) -> None:
    """Configure the logging system with specified level and output destination."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        if log_rotation:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
        else:
            file_handler = logging.FileHandler(log_file)
            
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set third-party library log levels to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)

class PerformanceMetrics:
    """Tracks and analyzes system performance metrics."""
    
    def __init__(self, data_directory: str = "data/metrics"):
        """Initialize performance metrics tracking.
        
        Args:
            data_directory: Directory to store performance data
        """
        self.logger = logging.getLogger("PerformanceMetrics")
        self.data_directory = data_directory
        os.makedirs(data_directory, exist_ok=True)
        
        # Initialize metric storage
        self.trades = []
        self.profits = []
        self.trade_times = []
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.errors_by_type = {}
        self.latencies = []
        self.execution_success_rate = 1.0
        
        # Financial metrics
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.peak_capital = 0.0
        self.drawdowns = []
        
        # Strategy performance
        self.strategy_performance = {}
        
        # Exchange performance
        self.exchange_performance = {}
        
        # Timestamp for session
        self.session_start = datetime.datetime.now()
        self.last_snapshot = self.session_start
        
        # Save interval (in seconds)
        self.save_interval = 300  # 5 minutes
        
        self.logger.info("Performance metrics initialized")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a completed trade and update metrics.
        
        Args:
            trade_data: Dictionary with trade details
        """
        self.trades.append(trade_data)
        
        profit = trade_data.get("profit", 0.0)
        self.profits.append(profit)
        
        strategy_id = trade_data.get("strategy_id", "unknown")
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                "trades": 0,
                "profit": 0.0,
                "wins": 0,
                "losses": 0
            }
        
        self.strategy_performance[strategy_id]["trades"] += 1
        self.strategy_performance[strategy_id]["profit"] += profit
        
        if profit > 0:
            self.strategy_performance[strategy_id]["wins"] += 1
        elif profit < 0:
            self.strategy_performance[strategy_id]["losses"] += 1
        
        # Record exchange performance
        exchange_id = trade_data.get("exchange_id", "unknown")
        if exchange_id not in self.exchange_performance:
            self.exchange_performance[exchange_id] = {
                "trades": 0,
                "profit": 0.0,
                "errors": 0
            }
        
        self.exchange_performance[exchange_id]["trades"] += 1
        self.exchange_performance[exchange_id]["profit"] += profit
        
        # Update execution metrics
        self.opportunities_executed += 1
        execution_time = trade_data.get("execution_time", 0)
        if execution_time > 0:
            self.latencies.append(execution_time)
        
        # Update success rate
        if len(self.trades) > 0:
            successful_trades = sum(1 for t in self.trades if not t.get("error"))
            self.execution_success_rate = successful_trades / len(self.trades)
        
        # Calculate drawdown if we have capital info
        if self.current_capital > 0:
            self.current_capital += profit
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            else:
                drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                self.drawdowns.append(drawdown)
        
        # Check if we should save metrics
        now = datetime.datetime.now()
        if (now - self.last_snapshot).total_seconds() > self.save_interval:
            self.save_metrics()
            self.last_snapshot = now
    
    def record_opportunity(self) -> None:
        """Record a detected arbitrage opportunity."""
        self.opportunities_found += 1
    
    def record_error(self, error_type: str, details: Any = None) -> None:
        """Record an error that occurred during system operation.
        
        Args:
            error_type: Type or category of error
            details: Additional error details
        """
        if error_type not in self.errors_by_type:
            self.errors_by_type[error_type] = []
        
        self.errors_by_type[error_type].append({
            "timestamp": datetime.datetime.now(),
            "details": details
        })
        
        # Update exchange error count if applicable
        if isinstance(details, dict) and "exchange_id" in details:
            exchange_id = details["exchange_id"]
            if exchange_id in self.exchange_performance:
                self.exchange_performance[exchange_id]["errors"] += 1
    
    def update_capital(self, current_capital: float) -> None:
        """Update the current capital value.
        
        Args:
            current_capital: Current total capital across all exchanges
        """
        if self.initial_capital == 0:
            self.initial_capital = current_capital
            self.peak_capital = current_capital
        
        self.current_capital = current_capital
        
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        else:
            drawdown = (self.peak_capital - current_capital) / self.peak_capital
            self.drawdowns.append(drawdown)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        total_profit = sum(self.profits)
        win_rate = 0
        if len(self.profits) > 0:
            win_rate = sum(1 for p in self.profits if p > 0) / len(self.profits)
        
        avg_profit = 0
        if len(self.profits) > 0:
            avg_profit = total_profit / len(self.profits)
        
        avg_latency = 0
        if len(self.latencies) > 0:
            avg_latency = sum(self.latencies) / len(self.latencies)
        
        max_drawdown = 0
        if len(self.drawdowns) > 0:
            max_drawdown = max(self.drawdowns)
        
        return {
            "total_trades": len(self.trades),
            "total_profit": total_profit,
            "win_rate": win_rate,
            "avg_profit_per_trade": avg_profit,
            "opportunities_found": self.opportunities_found,
            "opportunities_executed": self.opportunities_executed,
            "execution_rate": self.opportunities_executed / max(1, self.opportunities_found),
            "execution_success_rate": self.execution_success_rate,
            "avg_execution_latency": avg_latency,
            "total_errors": sum(len(errors) for errors in self.errors_by_type.values()),
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "roi_percentage": ((self.current_capital - self.initial_capital) / max(0.01, self.initial_capital)) * 100,
            "max_drawdown": max_drawdown,
            "session_duration": (datetime.datetime.now() - self.session_start).total_seconds() / 3600  # hours
        }
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics broken down by strategy.
        
        Returns:
            Dictionary containing per-strategy metrics
        """
        for strategy_id, data in self.strategy_performance.items():
            if data["trades"] > 0:
                data["win_rate"] = data["wins"] / data["trades"]
                data["avg_profit"] = data["profit"] / data["trades"]
            else:
                data["win_rate"] = 0
                data["avg_profit"] = 0
        
        return self.strategy_performance
    
    def save_metrics(self) -> bool:
        """Save performance metrics to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save summary metrics
            summary = self.get_summary()
            summary_file = os.path.join(self.data_directory, f"summary_{timestamp}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save trade history
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_file = os.path.join(self.data_directory, f"trades_{timestamp}.csv")
                trades_df.to_csv(trades_file, index=False)
            
            # Save strategy performance
            strategy_file = os.path.join(self.data_directory, f"strategy_performance_{timestamp}.json")
            with open(strategy_file, 'w') as f:
                json.dump(self.get_strategy_performance(), f, indent=2)
            
            self.logger.info(f"Performance metrics saved to {self.data_directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")
            return False
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report.
        
        Args:
            output_file: File to save the report to (optional)
            
        Returns:
            str: Path to the generated report file
        """
        try:
            # Save current metrics first
            self.save_metrics()
            
            # Default output file if none provided
            if not output_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.data_directory, f"performance_report_{timestamp}.html")
            
            # Generate HTML report using pandas
            summary = self.get_summary()
            strategy_perf = self.get_strategy_performance()
            
            # Convert to DataFrames
            summary_df = pd.DataFrame([summary])
            strategy_df = pd.DataFrame.from_dict(strategy_perf, orient='index')
            
            # Create HTML components
            html_parts = []
            html_parts.append("<html><head><title>Ultimate Arbitrage System - Performance Report</title>")
            html_parts.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
            html_parts.append("table { border-collapse: collapse; width: 100%; }")
            html_parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_parts.append("th { background-color: #f2f2f2; }")
            html_parts.append("h1, h2 { color: #333366; }")
            html_parts.append("</style></head><body>")
            
            # Header
            html_parts.append(f"<h1>Ultimate Arbitrage System - Performance Report</h1>")
            html_parts.append(f"<p>Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Summary section
            html_parts.append("<h2>Performance Summary</h2>")
            html_parts.append(summary_df.to_html())
            
            # Strategy performance
            html_parts.append("<h2>Strategy Performance</h2>")
            html_parts.append(strategy_df.to_html())
            
            # Error summary

