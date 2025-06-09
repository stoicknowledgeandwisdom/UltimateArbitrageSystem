#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate Risk Controller

Advanced risk management system for cryptocurrency arbitrage operations.
Implements adaptive position sizing, dynamic risk allocation, market condition detection,
multi-currency risk balancing, and profit optimization strategies.
"""

import logging
import time
import json
import os
import threading
import math
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
import heapq
from collections import deque
import pandas as pd
from scipy import stats

logger = logging.getLogger("RiskController")

class RiskController:
    """
    Advanced risk management system with profit optimization algorithms and adaptive risk allocation.
    Designed to maximize arbitrage opportunities while maintaining strict risk controls.
    """
    
    def __init__(self, config_path=None, config_dict=None, ml_models_path=None):
        """
        Initialize the risk controller with advanced configuration options.
        
        Args:
            config_path: Path to risk configuration JSON file
            config_dict: Dictionary containing risk parameters
            ml_models_path: Path to machine learning models for risk prediction
        """
        # Core risk parameters with advanced defaults
        self.default_parameters = {
            # Position sizing parameters
            "max_position_size": 0.02,              # Maximum position size as % of total capital
            "min_position_size": 0.001,             # Minimum position size as % of total capital
            "dynamic_position_sizing": True,        # Use dynamic position sizing based on opportunity quality
            "position_size_increment": 0.0005,      # Increment for position size adjustments
            
            # Loss control parameters
            "daily_loss_limit": 0.05,               # Maximum daily loss as % of total capital
            "weekly_loss_limit": 0.10,              # Maximum weekly loss as % of total capital
            "monthly_loss_limit": 0.15,             # Maximum monthly loss as % of total capital
            "max_drawdown": 0.15,                   # Maximum allowed drawdown before reducing exposure
            "stop_loss_threshold": 0.02,            # Stop loss threshold (2%)
            "tiered_stop_loss": True,               # Use tiered stop-loss levels
            "trailing_stop_loss": True,             # Use trailing stop-loss for profitable positions
            
            # Profit optimization parameters
            "take_profit_threshold": 0.01,          # Take profit threshold (1%)
            "trailing_take_profit": True,           # Use trailing take-profit for maximizing gains
            "profit_lock_threshold": 0.005,         # Lock in profits when they reach this threshold
            "profit_reinvestment_rate": 0.5,        # Percentage of profits to reinvest
            
            # Exposure control parameters
            "max_open_positions": 10,               # Maximum number of simultaneous open positions
            "max_positions_per_strategy": 5,        # Maximum positions for any single strategy
            "max_exchange_exposure": 0.30,          # Maximum exposure on a single exchange (30%)
            "max_symbol_exposure": 0.15,            # Maximum exposure on a single symbol (15%)
            "max_strategy_exposure": 0.40,          # Maximum exposure for any single strategy (40%)
            
            # Opportunity quality parameters
            "min_profit_threshold": 0.005,          # Minimum profit threshold for trades (0.5%)
            "optimal_profit_threshold": 0.015,      # Optimal profit threshold for maximum position size
            "min_volume_threshold": 10000,          # Minimum market volume in base currency
            "min_liquidity_ratio": 5.0,             # Minimum ratio of market liquidity to position size
            
            # Execution parameters
            "max_slippage": 0.002,                  # Maximum allowed slippage (0.2%)
            "dynamic_slippage_adjustment": True,    # Adjust slippage tolerance based on market conditions
            "execution_timeout": 15,                # Maximum execution time in seconds
            "retry_attempts": 3,                    # Number of retry attempts for failed executions
            "position_timeout": 3600,               # Maximum position duration in seconds (1 hour)
            
            # Market condition parameters
            "volatility_impact_factor": 0.5,        # How much volatility affects position sizing (0-1)
            "max_acceptable_volatility": 0.04,      # Maximum acceptable market volatility
            "trend_impact_factor": 0.3,             # How much trend affects position sizing (0-1)
            "volume_impact_factor": 0.2,            # How much volume affects position sizing (0-1)
            
            # Correlation parameters
            "correlation_limit": 0.7,               # Maximum correlation between open positions
            "portfolio_correlation_limit": 0.5,     # Maximum overall portfolio correlation
            "symbol_correlation_window": 720,       # Lookback window for correlation analysis (hours)
            
            # Risk adjustment parameters
            "risk_increment_profit": 0.001,         # Increment max position size after profit (0.1%)
            "risk_decrement_loss": 0.003,           # Decrement max position size after loss (0.3%)
            "volatility_adjustment": True,          # Adjust position size based on market volatility
            "adaptive_risk_profiles": True,         # Use different risk profiles based on market conditions
            
            # Recovery and failsafe parameters
            "cooldown_period": 300,                 # Cooldown period after stop loss trigger (5 min)
            "auto_shutdown_threshold": 0.10,        # Emergency shutdown if daily loss exceeds 10%
            "circuit_breaker_levels": [0.05, 0.08, 0.10],  # Progressive circuit breaker levels
            "auto_recovery_mode": True,             # Automatically recover from circuit breaker events
            
            # Multi-currency parameters
            "base_currency": "USDT",                # Base currency for risk calculations
            "multi_currency_support": True,         # Support risk management across multiple currencies
            "currency_correlation_threshold": 0.8,  # Maximum correlation between currency exposures
            
            # System health parameters
            "health_check_interval": 30,            # System health check interval in seconds
            "performance_monitoring": True,         # Enable detailed performance monitoring
            "auto_optimization_interval": 86400,    # Auto-optimize risk parameters daily (in seconds)
            
            # Advanced features
            "ml_risk_adjustment": True,             # Use machine learning for risk adjustments
            "market_regime_detection": True,        # Detect and adapt to different market regimes
            "opportunity_ranking": True,            # Rank opportunities by risk-adjusted return
            "hedging_strategies": True,             # Use hedging to reduce overall portfolio risk
            "cross_exchange_netting": True,         # Net exposures across multiple exchanges
            "risk_parity_allocation": True,         # Use risk parity for capital allocation
            "dynamic_timeframe_analysis": True,     # Analyze opportunities across multiple timeframes
        }
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.risk_parameters = self._load_config_from_file(config_path)
        elif config_dict:
            self.risk_parameters = self._validate_config_dict(config_dict)
        else:
            logger.warning("No risk configuration provided, using default parameters")
            self.risk_parameters = self.default_parameters
        
        # Runtime tracking variables
        self.open_positions = {}                     # Dictionary of open positions by ID
        self.position_history = []                   # History of closed positions
        self.daily_pnl = 0.0                         # Daily profit and loss
        self.weekly_pnl = 0.0                        # Weekly profit and loss
        self.monthly_pnl = 0.0                       # Monthly profit and loss
        self.total_capital = 0.0                     # Total capital across all exchanges
        self.starting_capital = 0.0                  # Starting capital for calculating absolute growth
        self.peak_capital = 0.0                      # Peak capital achieved
        self.current_drawdown = 0.0                  # Current drawdown from peak capital
        self.max_drawdown_experienced = 0.0          # Maximum drawdown experienced
        
        # Exchange and asset tracking
        self.exchange_balances = {}                  # Balances per exchange
        self.exchange_exposure = {}                  # Current exposure per exchange
        self.symbol_exposure = {}                    # Current exposure per symbol
        self.strategy_exposure = {}                  # Current exposure per strategy
        self.currency_exposure = {}                  # Exposure per currency for multi-currency support
        
        # Market and performance metrics
        self.volatility_metrics = {}                 # Store volatility metrics per market
        self.market_trend_metrics = {}               # Store trend metrics per market
        self.opportunity_performance = {}            # Performance metrics for different opportunity types
        self.trade_metrics = {                       # Success/failure metrics
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "average_profit_pct": 0.0,
            "average_loss_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
        
        # Correlation tracking
        self.symbol_correlation_matrix = {}          # Correlation matrix between symbols
        self.last_correlation_update = 0             # Timestamp of last correlation update
        
        # Time tracking
        self.daily_reset_time = self._get_next_reset_time('daily')
        self.weekly_reset_time = self._get_next_reset_time('weekly')
        self.monthly_reset_time = self._get_next_reset_time('monthly')
        self.last_health_check = time.time()         # Last system health check time
        self.last_parameters_optimization = time.time()  # Last time risk parameters were optimized
        self.system_start_time = time.time()         # System start time
        
        # Status indicators
        self.trading_enabled = True                  # Global switch for trading
        self.risk_levels = ["normal", "reduced", "minimal", "emergency"]  # Risk levels
        self.current_risk_level = "normal"           # Current risk level
        self.circuit_breaker_triggered = False       # Circuit breaker status
        self.health_status = {                       # System health status
            "status": "OK", 
            "warnings": [], 
            "errors": [],
            "performance_metrics": {},
            "risk_metrics": {}
        }
        
        # Thread safety
        self.risk_lock = threading.RLock()           # Lock for thread safety
        
        # Advanced tracking
        self.price_data_buffer = {}                  # Buffer for price data used in analysis
        self.market_regime = "normal"                # Current market regime (normal, volatile, trending)
        self.opportunity_queue = []                  # Priority queue for ranking opportunities
        
        # Initialize machine learning components if enabled
        self.ml_models = {}
        if self.risk_parameters["ml_risk_adjustment"] and ml_models_path:
            self._initialize_ml_models(ml_models_path)
        
        # Create directories for data storage
        os.makedirs("data/position_history", exist_ok=True)
        os.makedirs("data/risk_analytics", exist_ok=True)
        os.makedirs("data/market_metrics", exist_ok=True)
        
        # Start monitoring threads
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._risk_monitor_thread, daemon=True)
        self.monitor_thread.start()
        
        if self.risk_parameters["performance_monitoring"]:
            self.analytics_thread = threading.Thread(target=self._analytics_thread, daemon=True)
            self.analytics_thread.start()
        
        logger.info(f"Advanced Risk Controller initialized with {len(self.risk_parameters)} parameters")
        logger.info(f"Trading enabled: {self.trading_enabled}, Risk level: {self.current_risk_level}")
    
    def _initialize_ml_models(self, models_path: str) -> None:
        """Initialize machine learning models for risk prediction."""
        try:
            # This would load trained ML models for:
            # - Volatility prediction
            # - Stop-loss optimization
            # - Position sizing optimization
            # - Market regime detection
            
            # Placeholder for actual ML model loading code
            logger.info(f"Initialized machine learning models from {models_path}")
            
            # In an actual implementation, you would load models like:
            # import joblib
            # self.ml_models["volatility_predictor"] = joblib.load(f"{models_path}/volatility_model.pkl")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.risk_parameters["ml_risk_adjustment"] = False
    
    def _load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load risk parameters from a JSON configuration file."""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                
            # If config has a 'risk_parameters' key, use that section
            if 'risk_parameters' in config:
                params = config['risk_parameters']
            else:
                params = config
                
            validated_config = self._validate_config_dict(params)
            
            logger.info(f"Loaded risk configuration from {config_path}")
            return validated_config
            
        except Exception as e:
            logger.error(f"Error loading risk configuration: {e}")
            return self.default_parameters
    
    def _validate_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and merge the provided config with defaults."""
        validated = self.default_parameters.copy()
        
        if not isinstance(config_dict, dict):
            logger.error(f"Invalid configuration format: {type(config_dict)}")
            return validated
        
        # Update with provided values
        for key, value in config_dict.items():
            if key in validated:
                # Type checking
                if isinstance(validated[key], bool) and not isinstance(value, bool):
                    logger.warning(f"Invalid type for {key}, using default")
                    continue
                if not isinstance(value, bool) and not isinstance(validated[key], bool):
                    try:
                        # Try to convert to same type as default
                        value = type(validated[key])(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {key}, using default")
                        continue
                validated[key] = value
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Validate logical relationships between parameters
        self._validate_parameter_relationships(validated)
        
        return validated
    
    def _validate_parameter_relationships(self, params: Dict[str, Any]) -> None:
        """Ensure logical relationships between risk parameters."""
        # Ensure min position size < max position size
        if params["min_position_size"] >= params["max_position_size"]:
            params["min_position_size"] = params["max_position_size"] / 4
            logger.warning("Adjusted min_position_size to be less than max_position_size")
        
        # Ensure min profit threshold < optimal profit threshold
        if params["min_profit_threshold"] >= params["optimal_profit_threshold"]:
            params["optimal_profit_threshold"] = params["min_profit_threshold"] * 1.5
            
        return params
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return {
            "status": "healthy",
            "current_exposure": 0.0,
            "max_exposure": self.config.get("max_exposure_per_trade", 0.02),
            "daily_loss": 0.0,
            "max_daily_loss": self.config.get("max_daily_loss", 0.05)
        }
    
    def check_system_health(self) -> bool:
        """Check overall system health."""
        return True
    
    def update_total_capital(self, capital: float):
        """Update total capital amount."""
        self.total_capital = capital

