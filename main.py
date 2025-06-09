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

# Import system components with fallback handling
try:
    from exchanges.exchange_manager import ExchangeManager
except ImportError:
    logger.warning("ExchangeManager not found, creating fallback")
    class ExchangeManager:
        def __init__(self, config):
            self.config = config
        def initialize(self): return True
        def connect_all(self): return True
        def disconnect_all(self): pass
        def get_balances(self): return {}

try:
    from strategies.strategy_manager import StrategyManager
except ImportError:
    logger.warning("StrategyManager not found, creating fallback")
    class StrategyManager:
        def __init__(self, config, exchange_manager, market_data, risk_controller):
            self.config = config
        def initialize(self): return True
        def start_all_strategies(self): return True
        def stop_all_strategies(self): pass
        def get_active_strategies(self): return []

try:
    from risk_management.risk_controller import RiskController
except ImportError:
    logger.warning("RiskController not found, creating fallback")
    class RiskController:
        def __init__(self, config):
            self.config = config
        def initialize(self): return True
        def check_system_health(self): return True
        def get_risk_metrics(self): return {}
        def update_total_capital(self, capital): pass

try:
    from data.market_data import MarketDataProvider
except ImportError:
    logger.warning("MarketDataProvider not found, creating fallback")
    class MarketDataProvider:
        def __init__(self, exchange_manager):
            self.exchange_manager = exchange_manager
        def initialize(self): return True
        def start_data_collection(self): return True
        def stop_data_collection(self): pass
        def get_market_data(self): return {}
        def get_latest_market_data(self): return {}

try:
    from core.arbitrage_core.trading_engine import ArbitrageEngine
except ImportError:
    logger.warning("ArbitrageEngine not found, creating fallback")
    class ArbitrageEngine:
        def __init__(self, exchange_manager, strategy_manager, risk_controller, config):
            self.exchange_manager = exchange_manager
            self.strategy_manager = strategy_manager
            self.risk_controller = risk_controller
            self.config = config
        def initialize(self): return True
        def start(self): return True
        def stop(self): pass
        def get_performance_metrics(self): return {}


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
        self.optimizer_thread = None
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
            
            # Initialize all system components
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
                logger.info("Using conservative risk parameters for test mode")
                for param in ["max_exposure_per_trade", "max_daily_loss"]:
                    if param in risk_params:
                        risk_params[param] = risk_params[param] * 0.5
            self.risk_controller = RiskController(risk_params)
            
            logger.info("Initializing strategy manager...")
            strategies = self.config.get("strategies", [])
            enabled_strategies = [s for s in strategies if s.get("enabled", True)]
            if self.test_mode:
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
            
            self.components_initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.critical(f"Error during component initialization: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration from JSON file."""
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
            
            if not config_file.exists():
                logger.warning(f"Configuration file {config_path} not found")
                logger.warning("Using default configuration")
                return default_config
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with default config for any missing values
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
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
    
    def start(self) -> bool:
        """Start the arbitrage system."""
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
            
            # Start trading engine
            logger.info("Starting trading engine...")
            engine_start_success = self.trading_engine.start()
            
            if not engine_start_success:
                logger.error("Failed to start trading engine")
                self.stop()
                return False
                
            logger.info("Arbitrage system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start arbitrage system: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.stop()
            return False
    
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
            self._save_performance_metrics()
            
            logger.info("System shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.running = False
            self.performance_metrics["end_time"] = datetime.now()
    
    def _wait_for_threads(self):
        """Wait for all background threads to terminate."""
        threads = [self.monitoring_thread, self.data_backup_thread, self.health_check_thread, self.optimizer_thread]
        
        for thread in threads:
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread.name} to terminate...")
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate gracefully")
    
    def _save_performance_metrics(self):
        """Save performance metrics to file."""
        try:
            metrics_file = os.path.join("reports", f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            logger.info(f"Performance metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self.running,
            "test_mode": self.test_mode,
            "components_initialized": self.components_initialized,
            "performance_metrics": self.performance_metrics.copy(),
            "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds() if self.performance_metrics["start_time"] else 0
        }
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        self.stop()
    
    def _calculate_total_capital(self, balances: Dict[str, Any]) -> float:
        """Calculate total capital from exchange balances."""
        # Placeholder implementation - would normally convert all balances to USD
        total = 0.0
        for exchange_id, balance in balances.items():
            if isinstance(balance, dict):
                for currency, amount in balance.items():
                    if isinstance(amount, (int, float)) and amount > 0:
                        # In a real implementation, we'd convert to USD using current rates
                        total += amount  # Simplified for demo
        return total
    
    def _register_monitored_symbols(self):
        """Register symbols to monitor based on active strategies."""
        # Placeholder implementation
        logger.info("Registering monitored symbols...")
        pass
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Starting monitoring loop...")
        while self.running:
            try:
                # Update performance metrics
                if self.trading_engine:
                    metrics = self.trading_engine.get_performance_metrics()
                    self.performance_metrics.update(metrics)
                
                time.sleep(self.config.get("monitoring_interval", 10))
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _data_backup_loop(self):
        """Background data backup loop."""
        logger.info("Starting data backup loop...")
        while self.running:
            try:
                # Backup important data
                self._save_performance_metrics()
                time.sleep(self.config.get("data_backup_interval", 3600))
            except Exception as e:
                logger.error(f"Error in data backup loop: {e}")
                time.sleep(60)
    
    def _health_check_loop(self):
        """Background health check loop."""
        logger.info("Starting health check loop...")
        while self.running:
            try:
                # Perform health checks
                if self.risk_controller:
                    health_ok = self.risk_controller.check_system_health()
                    if not health_ok:
                        logger.warning("System health check failed")
                
                time.sleep(self.config.get("health_check_interval", 60))
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(30)


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    global arbitrage_system
    if arbitrage_system:
        arbitrage_system.stop()
    sys.exit(0)


def main():
    """Main function to run the arbitrage system."""
    global arbitrage_system
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ultimate Arbitrage System")
    parser.add_argument("--config", default="config/system_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode (no actual trades)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Print startup banner
        print("\n" + "=" * 80)
        print("🚀 ULTIMATE ARBITRAGE SYSTEM 🚀")
        print("Advanced Cryptocurrency Arbitrage Platform")
        print("=" * 80)
        print(f"Mode: {'TEST' if args.test_mode else 'LIVE'}")
        print(f"Config: {args.config}")
        print(f"Log Level: {args.log_level}")
        print("=" * 80 + "\n")
        
        # Initialize and start the system
        arbitrage_system = ArbitrageSystem(config_path=args.config, test_mode=args.test_mode)
        
        # Start the system
        if arbitrage_system.start():
            logger.info("System started successfully")
            
            # Keep the main thread alive
            try:
                while arbitrage_system.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
            
        else:
            logger.error("Failed to start the system")
            return 1
            
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        if arbitrage_system:
            arbitrage_system.stop()
    
    return 0


# Global variable for signal handling
arbitrage_system = None

if __name__ == "__main__":
    sys.exit(main())

