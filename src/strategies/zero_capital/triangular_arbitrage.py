#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangular Arbitrage Strategy

This module implements a triangular arbitrage strategy for cryptocurrency trading,
identifying and exploiting price discrepancies between three related currency pairs
on a single exchange. This is a zero-capital strategy as it can be initiated with
minimal capital and completed within a short time frame.

Features:
- Real-time detection of triangular arbitrage opportunities
- Profit calculation including fees and slippage
- Smart execution with order book analysis
- Risk management integration
- Performance tracking and analytics
- Multiple execution modes (simulation, real trading)
- Detailed logging and reporting
- Backtest mode for strategy optimization

Author: UltimateArbitrageSystem
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
import uuid
import threading
import json
import datetime
import statistics
from enum import Enum
import heapq
import traceback

# Set decimal precision
getcontext().prec = 16

# Configure logging
logger = logging.getLogger("TriangularArbitrage")

class ExecutionMode(Enum):
    """Execution modes for the arbitrage strategy."""
    SIMULATION = "simulation"  # Log opportunities without trading
    REAL = "real"              # Execute real trades
    BACKTEST = "backtest"      # Run on historical data


class TriangularArbitrage:
    """
    Implements a triangular arbitrage strategy that identifies and executes arbitrage 
    opportunities between three different trading pairs on a single exchange.
    
    This strategy looks for price discrepancies between three related trading pairs
    on a single exchange, allowing for a circular trade that starts and ends with
    the same currency but with a profit.
    
    Example:
        1. Convert USD → BTC
        2. Convert BTC → ETH
        3. Convert ETH → USD
        
    If the final amount of USD is greater than the starting amount (after fees),
    then an arbitrage opportunity exists.
    """
    
    def __init__(self, exchange_manager, market_data_provider, risk_controller, config: Dict):
        """
        Initialize the triangular arbitrage strategy.
        
        Args:
            exchange_manager: Manager for handling exchange interactions
            market_data_provider: Provider for market data
            risk_controller: Controller for managing trading risks
            config: Configuration parameters for the strategy
                exchange_id: ID of the exchange to use
                base_currency: Currency to start and end with (e.g., USDT)
                min_profit_threshold: Minimum profit percentage to execute a trade
                trade_size: Base amount to trade
                fee_rate: Exchange fee rate per trade
                max_execution_time: Maximum time to execute a trade (seconds)
                check_interval: Time between opportunity checks (seconds)
                max_slippage: Maximum allowable price slippage
                currency_sets: List of currency sets to monitor
                excluded_pairs: Trading pairs to exclude
                execution_mode: SIMULATION, REAL, or BACKTEST
                max_concurrent_trades: Maximum number of concurrent trades
                order_book_depth: Depth of order book to analyze
                profit_taking_threshold: When to take profit in long-running trades
                stop_loss_threshold: When to exit a losing trade
                retry_attempts: Number of retry attempts for failed operations
                enable_analytics: Whether to collect detailed performance metrics
        """
        self.exchange_manager = exchange_manager
        self.market_data = market_data_provider
        self.risk_controller = risk_controller
        self.config = config
        
        # Set configuration parameters with defaults
        self.exchange_id = config.get("exchange_id")
        self.base_currency = config.get("base_currency", "USDT")
        self.min_profit_threshold = Decimal(str(config.get("min_profit_threshold", 0.005)))  # 0.5% profit threshold
        self.trade_size = Decimal(str(config.get("trade_size", 100)))  # Base amount to trade
        self.fee_rate = Decimal(str(config.get("fee_rate", 0.001)))  # 0.1% fee per trade
        self.max_execution_time = int(config.get("max_execution_time", 5))  # Maximum seconds for execution
        self.check_interval = int(config.get("check_interval", 1))  # Seconds between opportunity checks
        self.max_slippage = Decimal(str(config.get("max_slippage", 0.001)))  # 0.1% slippage tolerance
        
        # Currency sets to monitor for arbitrage (manually specified or auto-detected)
        self.currency_sets = config.get("currency_sets", [])
        
        # Specific pairs to exclude 
        self.excluded_pairs = set(config.get("excluded_pairs", []))
        
        # Execution settings
        self.execution_mode = ExecutionMode(config.get("execution_mode", "simulation").lower())
        self.max_concurrent_trades = int(config.get("max_concurrent_trades", 3))
        self.order_book_depth = int(config.get("order_book_depth", 10))
        self.profit_taking_threshold = Decimal(str(config.get("profit_taking_threshold", 0.008)))  # 0.8%
        self.stop_loss_threshold = Decimal(str(config.get("stop_loss_threshold", -0.002)))  # -0.2%
        self.retry_attempts = int(config.get("retry_attempts", 3))
        self.retry_delay = int(config.get("retry_delay", 1))  # seconds between retries
        
        # Performance monitoring
        self.enable_analytics = bool(config.get("enable_analytics", True))
        self.performance_window = int(config.get("performance_window", 1000))  # Number of trades to calculate metrics
        
        # Circuit breakers
        self.max_failures_threshold = int(config.get("max_failures_threshold", 5))
        self.market_volatility_threshold = Decimal(str(config.get("market_volatility_threshold", 0.05)))  # 5%
        
        # Trading state
        self.is_running = False
        self.active_trades = {}  # trade_id -> trade_info
        self.trade_history = []  # List of completed trades
        self.lock = threading.Lock()
        self.opportunity_queue = []  # Priority queue for opportunities
        self.market_cache = {}  # Cache for market info
        self.pair_direction_map = {}  # Map for pair directions
        
        # Statistics and performance tracking
        self.total_opportunities = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = Decimal('0')
        self.total_volume_traded = Decimal('0')
        self.consecutive_failures = 0
        self.execution_times = []  # Track execution times for performance tuning
        
        # Performance metrics
        self.metrics = {
            "win_rate": 0,
            "avg_profit": 0,
            "avg_execution_time": 0,
            "max_profit": 0,
            "min_profit": 0,
            "profit_variance": 0,
            "sharpe_ratio": 0,
            "profit_factor": 0,
        }
        
        logger.info(f"Triangular Arbitrage strategy initialized for exchange {self.exchange_id}")
        logger.info(f"Execution mode: {self.execution_mode.value}")
        logger.debug(f"Configuration: {json.dumps({k: str(v) for k, v in self.__dict__.items() if not callable(v) and not k.startswith('_') and not isinstance(v, (dict, list, set))})}")
    
    def start(self):
        """
        Start the arbitrage strategy.
        
        This begins the process of monitoring for triangular arbitrage opportunities
        and executing trades when profitable opportunities are found.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Triangular Arbitrage strategy is already running")
            return False
        
        self.is_running = True
        logger.info(f"Starting Triangular Arbitrage strategy on exchange {self.exchange_id}")
        
        try:
            # Initialize exchange connection
            exchange = self.exchange_manager.get_exchange(self.exchange_id)
            if not exchange:
                logger.error(f"Exchange {self.exchange_id} not found or not initialized")
                self.is_running = False
                return False
            
            # Validate exchange capabilities
            if not self._validate_exchange_capabilities(exchange):
                logger.error(f"Exchange {self.exchange_id} does not support required features")
                self.is_running = False
                return False
            
            # Initialize market cache
            self._init_market_cache()
            
            # If no currency sets are specified, detect tradable pairs
            if not self.currency_sets:
                self._detect_tradable_sets()
            
            # Log currency sets being monitored
            currency_set_count = len(self.currency_sets)
            logger.info(f"Monitoring {currency_set_count} currency sets for arbitrage opportunities")
            if logger.isEnabledFor(logging.DEBUG):
                for i, currencies in enumerate(self.currency_sets):
                    logger.debug(f"Currency set {i+1}: {currencies}")
            
            # Start opportunity detection thread
            if self.execution_mode != ExecutionMode.BACKTEST:
                self.detector_thread = threading.Thread(
                    target=self._opportunity_detection_loop,
                    daemon=True
                )
                self.detector_thread.start()
                
                # Start execution thread
                self.executor_thread = threading.Thread(
                    target=self._trade_execution_loop,
                    daemon=True
                )
                self.executor_thread.start()
                
                logger.info("Arbitrage detection and execution threads started")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting triangular arbitrage strategy: {str(e)}")
            logger.debug(traceback.format_exc())
            self.is_running = False
            return False
    
    def stop(self):
        """
        Stop the arbitrage strategy.
        
        This gracefully shuts down the strategy, closing any open trades
        and stopping all monitoring threads.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Triangular Arbitrage strategy is already stopped")
            return True
        
        logger.info(f"Stopping Triangular Arbitrage strategy on exchange {self.exchange_id}")
        self.is_running = False
        
        try:
            # Wait for threads to terminate
            if hasattr(self, 'detector_thread') and self.detector_thread.is_alive():
                self.detector_thread.join(timeout=5.0)
            
            if hasattr(self, 'executor_thread') and self.executor_thread.is_alive():
                self.executor_thread.join(timeout=5.0)
            
            # Close any open trades
            with self.lock:
                active_trade_ids = list(self.active_trades.keys())
            
            for trade_id in active_trade_ids:
                self._close_trade(trade_id, "Strategy stopped")
            
            # Log final statistics
            self._log_performance_metrics()
            
            logger.info("Triangular Arbitrage strategy stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping triangular arbitrage strategy: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def _validate_exchange_capabilities(self, exchange) -> bool:
        """
        Validate that the exchange supports the required features.
        
        Args:
            exchange: Exchange object to validate
            
        Returns:
            bool: True if the exchange supports all required features, False otherwise
        """
        capabilities = getattr(exchange, 'capabilities', {})
        
        # Check for required capabilities
        required_capabilities = [
            ('fetchTicker', True),
            ('fetchOrderBook', True),
            ('createOrder', self.execution_mode == ExecutionMode.REAL),
            ('cancelOrder', self.execution_mode == ExecutionMode.REAL),
            ('fetchBalance', True)
        ]
        
        for capability, required in required_capabilities:
            if required and not capabilities.get(capability, False):
                logger.error(f"Exchange {self.exchange_id} missing required capability: {capability}")
                return False
        
        return True
    
    def _init_market_cache(self):
        """
        Initialize the market cache with information about available markets.
        This includes market precision, minimum trade amounts, and fee rates.
        """
        logger.info("Initializing market cache")
        
        exchange = self.exchange_manager.get_exchange(self.exchange_id)
        if not exchange:
            logger.error(f"Exchange {self.exchange_id} not found")
            return
        
        try:
            markets = exchange.get_markets()
            if not markets:
                logger.error("No markets found for exchange")
                return
            
            # Cache market information
            for symbol, market_info in markets.items():
                # Skip excluded pairs
                if symbol in self.excluded_pairs:
                    continue
                
                precision = market_info.get('precision', {})
                limits = market_info.get('limits', {})
                
                self.market_cache[symbol] = {
                    'precision': {
                        'price': precision.get('price', 8),
                        'amount': precision.get('amount', 8)
                    },
                    'limits': {
                        'min_amount': limits.get('amount', {}).get('min', 0),
                        'min_cost': limits.get('cost', {}).get('min', 0)
                    },
                    'maker_fee': market_info.get('maker', self.fee_rate),
                    'taker_fee': market_info.get('taker', self.fee_rate),
                    'base': market_info.get('base'),
                    'quote': market_info.get('quote')
                }
                
                # Cache pair direction for quick lookups
                base = market_info.get('base')
                quote = market_info.get('quote')
                if base and quote:
                    self.pair_direction_map[(base, quote)] = {'symbol': symbol, 'reversed': False}
                    self.pair_direction_map[(quote, base)] = {'symbol': symbol, 'reversed': True}
            
            logger.info(f"Market cache initialized with {len(self.market_cache)} markets")
            
        except Exception as e:
            logger.error(f"Error initializing market cache: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _detect_tradable_sets(self):
        """
        Automatically detect tradable currency sets for triangular arbitrage.
        This builds sets of three currencies where all trading pairs between them exist.
        """
        logger.info("Detecting tradable currency sets for triangular arbitrage")
        
        try:
            # Get all available markets from the exchange
            exchange = self.exchange_manager.get_exchange(self.exchange


def _opportunity_detection_loop(self):
    """
    Continuous loop that detects triangular arbitrage opportunities
    and adds them to the opportunity_queue.
    """
    logger.info("Starting opportunity detection loop")
    
    while self.is_running:
        try:
            # Get exchange instance
            exchange = self.exchange_manager.get_exchange(self.exchange_id)
            if not exchange:
                logger.error(f"Exchange {self.exchange_id} not available")
                time.sleep(self.check_interval)
                continue
            
            # Process each currency set
            for currency_set in self.tradable_sets:
                # Skip if we've reached max concurrent trades
                if (self.max_concurrent_trades > 0 and 
                    len(self.active_trades) >= self.max_concurrent_trades):
                    continue
                
                # Each currency set is expected to be a tuple of 3 currencies
                if len(currency_set) != 3:
                    logger.warning(f"Invalid currency set format: {currency_set}")
                    continue
                
                # Currency order in the set
                base, quote, intermediate = currency_set
                
                # Calculate profit for path 1: base -> quote -> intermediate -> base
                profit_path1 = self._calculate_triangular_arbitrage(
                    exchange, base, quote, intermediate
                )
                
                # Calculate profit for path 2: base -> intermediate -> quote -> base
                profit_path2 = self._calculate_triangular_arbitrage(
                    exchange, base, intermediate, quote
                )
                
                # Determine best path and profit
                best_profit = None
                best_path = None
                
                if profit_path1 is not None and (best_profit is None or profit_path1 > best_profit):
                    best_profit = profit_path1
                    best_path = "path1"
                
                if profit_path2 is not None and (best_profit is None or profit_path2 > best_profit):
                    best_profit = profit_path2
                    best_path = "path2"
                
                # If profitable, add to queue
                if best_profit is not None and best_profit >= self.min_profit_threshold:
                    # Create opportunity object
                    opportunity = {
                        'currency_set': currency_set,
                        'profit_pct': best_profit,
                        'path': best_path,
                        'timestamp': time.time(),
                        'exchange_id': self.exchange_id
                    }
                    
                    # Add to queue with thread safety
                    with self.lock:
                        self.opportunity_queue.append(opportunity)
                        # Sort by profit (highest first)
                        self.opportunity_queue.sort(key=lambda x: x['profit_pct'], reverse=True)
                    
                    logger.info(f"Found arbitrage opportunity: {currency_set}, profit: {best_profit:.2%}, path: {best_path}")
            
            # Sleep to avoid excessive API calls
            time.sleep(self.check_interval)
            
        except Exception as e:
            logger.error(f"Error in opportunity detection loop: {str(e)}")
            time.sleep(self.check_interval * 2)  # Sleep longer on error

def _check_profitability(self, exchange, currency_set):
    """
    Placeholder for calculating triangular arbitrage profitability.
    In a real implementation, this would perform actual calculations.
    """
    try:
        # This is a mock implementation
        # In reality, this would calculate potential profit from triangular arbitrage
        import random
        # Return 0-2% simulated profit (mostly below threshold to be realistic)
        return random.uniform(0, 0.02)
    except Exception as e:
        logger.error(f"Error checking profitability: {str(e)}")
        return 0

def _trade_execution_loop(self):
    """
    Continuous loop that processes opportunities from the queue
    and executes trades.
    """
    logger.info("Starting trade execution loop")
    
    while self.is_running:
        try:
            # Skip if no opportunities in queue
            if not self.opportunity_queue:
                time.sleep(self.execution_interval)
                continue
            
            # Get exchange instance
            exchange = self.exchange_manager.get_exchange(self.exchange_id)
            if not exchange:
                logger.error(f"Exchange {self.exchange_id} not available")
                time.sleep(self.execution_interval)
                continue
            
            # Check if we've reached max concurrent trades
            if (self.max_concurrent_trades > 0 and 
                len(self.active_trades) >= self.max_concurrent_trades):
                logger.debug("Max concurrent trades reached, waiting...")
                time.sleep(self.execution_interval)
                continue
            
            # Get opportunity from queue with thread safety
            with self.lock:
                if not self.opportunity_queue:  # Double-check after acquiring lock
                    continue
                opportunity = self.opportunity_queue.pop(0)
            
            # Verify opportunity is still fresh (timeout after 60 seconds)
            if time.time() - opportunity['timestamp'] > 60:
                logger.warning(f"Skipping stale opportunity: {opportunity['currency_set']}")
                continue
            
            # Verify the opportunity is still profitable
            currency_set = opportunity['currency_set']
            if len(currency_set) != 3:
                logger.warning(f"Invalid currency set format: {currency_set}")
                continue
            
            base, quote, intermediate = currency_set
            path = opportunity.get('path', 'path1')  # Default to path1 if not specified
            
            # Recalculate profit based on path
            current_profit = None
            if path == 'path1':
                current_profit = self._calculate_triangular_arbitrage(
                    exchange, base, quote, intermediate
                )
            else:  # path2
                current_profit = self._calculate_triangular_arbitrage(
                    exchange, base, intermediate, quote
                )
            
            # Skip if no longer profitable
            if current_profit is None or current_profit < self.min_profit_threshold:
                logger.warning(f"Opportunity no longer profitable: {currency_set}, profit: {current_profit}")
                continue
            
            # Generate trade ID
            trade_id = str(uuid.uuid4())
            
            try:
                # Different behavior based on execution mode
                if self.execution_mode == ExecutionMode.SIMULATION:
                    # Simulate trade execution
                    logger.info(f"[SIMULATION] Executing arbitrage: {currency_set}, path: {path}, expected profit: {current_profit:.2%}")
                    
                    # Record the trade
                    trade_info = {
                        'id': trade_id,
                        'currency_set': currency_set,
                        'path': path,
                        'start_time': time.time(),
                        'status': 'executing',
                        'expected_profit': current_profit,
                        'exchange_id': opportunity.get('exchange_id', self.exchange_id)
                    }
                    
                    with self.lock:
                        self.active_trades[trade_id] = trade_info
                    
                    # Simulate execution delay
                    time.sleep(1)
                    
                    # Add some randomization to the actual profit (to simulate slippage)
                    import random
                    actual_profit = current_profit * random.uniform(0.7, 1.0)
                    
                    # Complete the trade
                    with self.lock:
                        if trade_id in self.active_trades:
                            trade_info = self.active_trades[trade_id]
                            trade_info['status'] = 'completed'
                            trade_info['end_time'] = time.time()
                            trade_info['actual_profit'] = actual_profit
                            
                            # Move to history
                            self.trade_history.append(trade_info)
                            del self.active_trades[trade_id]
                            
                            # Update stats
                            self.successful_trades += 1
                            self.total_profit += actual_profit
                            
                            logger.info(f"[SIMULATION] Completed arbitrage trade: {trade_id}, profit: {actual_profit:.2%}")
                
                elif self.execution_mode == ExecutionMode.REAL:
                    # Execute real trades (placeholder implementation)
                    logger.warning(f"[REAL] Real trade execution not fully implemented yet")
                    
                    # TODO: Implement real trade execution
                    # 1. Place first order (base to quote or base to intermediate)
                    # 2. Wait for fill
                    # 3. Place second order
                    # 4. Wait for fill
                    # 5. Place third order
                    # 6. Calculate actual profit
                    
                    # For now, just log that we would have executed
                    logger.info(f"[REAL] Would execute: {currency_set}, path: {path}, expected profit: {current_profit:.2%}")
                
            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                # Update trade as failed
                with self.lock:
                    if trade_id in self.active_trades:
                        trade_info = self.active_trades[trade_id]
                        trade_info['status'] = 'failed'
                        trade_info['end_time'] = time.time()
                        trade_info['error'] = str(e)
                        
                        # Move to history
                        self.trade_history.append(trade_info)
                        del self.active_trades[trade_id]
                        
                        # Update stats
                        self.failed_trades += 1
            
            # Sleep between executions
            time.sleep(self.execution_interval)
            
        except Exception as e:
            logger.error(f"Error in trade execution loop: {str(e)}")
            # Sleep longer on error
            time.sleep(self.execution_interval * 2)

def _calculate_triangular_arbitrage(self, exchange, base_currency, quote_currency, intermediate_currency):
    """
    Calculate potential profit from triangular arbitrage between three currencies.
    
    Path: base_currency -> quote_currency -> intermediate_currency -> base_currency
    
    Args:
        exchange: Exchange instance to get ticker data from
        base_currency: Starting currency (e.g., BTC)
        quote_currency: Second currency in the path (e.g., ETH)
        intermediate_currency: Third currency in the path (e.g., USDT)
        
    Returns:
        Profit percentage or None if calculation fails
    """
    try:
        # Define trading pairs needed
        pair1 = f"{base_currency}/{quote_currency}"
        pair2 = f"{quote_currency}/{intermediate_currency}"
        pair3 = f"{intermediate_currency}/{base_currency}"
        
        # Get ticker data for all pairs
        ticker1 = exchange.get_ticker(pair1)
        ticker2 = exchange.get_ticker(pair2)
        ticker3 = exchange.get_ticker(pair3)
        
        if not all([ticker1, ticker2, ticker3]):
            logger.debug(f"Missing ticker data for triangular arbitrage: {base_currency}/{quote_currency}/{intermediate_currency}")
            return None
        
        # Calculate conversion rates (accounting for fees)
        # For a->b we use the ask price (what we pay)
        # For b->a we would use the bid price (what we receive)
        
        # Starting with 1 unit of base_currency
        amount_base = 1.0
        
        # Convert base to quote (use appropriate ask/bid based on pair direction)
        rate1 = float(ticker1['ask']) if pair1.startswith(base_currency) else 1.0 / float(ticker1['bid'])
        amount_quote = amount_base * rate1 * (1 - float(self.fee_rate))
        
        # Convert quote to intermediate
        rate2 = float(ticker2['ask']) if pair2.startswith(quote_currency) else 1.0 / float(ticker2['bid'])
        amount_intermediate = amount_quote * rate2 * (1 - float(self.fee_rate))
        
        # Convert intermediate back to base
        rate3 = float(ticker3['ask']) if pair3.startswith(intermediate_currency) else 1.0 / float(ticker3['bid'])
        final_amount_base = amount_intermediate * rate3 * (1 - float(self.fee_rate))
        
        # Calculate profit percentage
        profit_pct = (final_amount_base / amount_base) - 1.0
        
        return profit_pct
        
    except Exception as e:
        logger.error(f"Error calculating triangular arbitrage: {str(e)}")
        return None
