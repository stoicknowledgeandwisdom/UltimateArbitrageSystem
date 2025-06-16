#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Trading Engine
=======================

Fully automated trading engine that executes buy/sell orders based on
arbitrage opportunities and AI-powered strategies. This engine handles:

- Automatic opportunity detection
- Real-time order execution
- Portfolio management
- Risk controls
- Profit tracking
- Test mode and live trading
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import threading
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading execution modes"""
    SIMULATION = "simulation"  # Simulated trades with mock data
    PAPER = "paper"           # Paper trading with real data
    LIVE = "live"             # Live trading with real money

class OrderType(Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingOpportunity:
    """Represents a trading opportunity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal('0')
    entry_price: Decimal = Decimal('0')
    target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    expected_profit: Decimal = Decimal('0')
    confidence: float = 0.0
    max_hold_time: int = 3600  # seconds
    priority: int = 1  # 1=highest, 5=lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class ExecutedTrade:
    """Represents an executed trade"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    opportunity_id: str = ""
    strategy: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal('0')
    executed_price: Decimal = Decimal('0')
    target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    actual_profit: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    execution_time: float = 0.0  # seconds
    status: str = "completed"  # pending, completed, failed, cancelled
    exchange: str = ""
    order_id: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AutomatedTradingEngine:
    """
    Fully automated trading engine that executes trades based on opportunities
    detected by various strategies and market analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the automated trading engine.
        
        Args:
            config: Configuration dictionary with trading parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Trading mode configuration
        self.trading_mode = TradingMode(config.get('trading_mode', 'simulation'))
        self.max_concurrent_trades = config.get('max_concurrent_trades', 5)
        self.max_daily_trades = config.get('max_daily_trades', 100)
        self.min_profit_threshold = Decimal(str(config.get('min_profit_threshold', '0.005')))  # 0.5%
        self.max_risk_per_trade = Decimal(str(config.get('max_risk_per_trade', '0.02')))  # 2%
        self.base_trade_size = Decimal(str(config.get('base_trade_size', '100')))  # $100
        
        # Execution parameters
        self.execution_delay = config.get('execution_delay', 0.1)  # seconds
        self.opportunity_timeout = config.get('opportunity_timeout', 60)  # seconds
        self.max_slippage = Decimal(str(config.get('max_slippage', '0.001')))  # 0.1%
        self.enable_stop_loss = config.get('enable_stop_loss', True)
        self.enable_take_profit = config.get('enable_take_profit', True)
        
        # Portfolio management
        self.initial_capital = Decimal(str(config.get('initial_capital', '10000')))
        self.available_capital = self.initial_capital
        self.allocated_capital = Decimal('0')
        self.total_profit = Decimal('0')
        self.daily_trades_count = 0
        self.daily_profit = Decimal('0')
        self.last_reset_date = datetime.now().date()
        
        # State management
        self.is_running = False
        self.opportunities = asyncio.Queue()
        self.active_trades: Dict[str, ExecutedTrade] = {}
        self.completed_trades: List[ExecutedTrade] = []
        self.trade_history: List[ExecutedTrade] = []
        
        # Performance tracking
        self.stats = {
            'total_opportunities': 0,
            'opportunities_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': Decimal('0'),
            'total_fees': Decimal('0'),
            'average_profit': Decimal('0'),
            'win_rate': 0.0,
            'average_execution_time': 0.0,
            'best_trade': Decimal('0'),
            'worst_trade': Decimal('0'),
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': Decimal('0'),
            'current_drawdown': Decimal('0')
        }
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.trading_task = None
        self.monitoring_task = None
        self.cleanup_task = None
        
        # Risk management
        self.circuit_breaker_triggered = False
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', '0.05')))  # 5%
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.consecutive_losses = 0
        
        self.logger.info(f"Automated Trading Engine initialized in {self.trading_mode.value} mode")
        self.logger.info(f"Max concurrent trades: {self.max_concurrent_trades}")
        self.logger.info(f"Min profit threshold: {self.min_profit_threshold:.2%}")
    
    async def start(self) -> bool:
        """
        Start the automated trading engine.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return True
        
        try:
            self.logger.info("Starting automated trading engine...")
            self.is_running = True
            
            # Reset daily counters if needed
            self._reset_daily_counters_if_needed()
            
            # Start background tasks
            self.trading_task = asyncio.create_task(self._trading_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Automated trading engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {str(e)}")
            self.is_running = False
            return False
    
    async def stop(self) -> bool:
        """
        Stop the automated trading engine.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Trading engine is not running")
            return True
        
        try:
            self.logger.info("Stopping automated trading engine...")
            self.is_running = False
            
            # Cancel all background tasks
            tasks = [self.trading_task, self.monitoring_task, self.cleanup_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            for task in tasks:
                if task:
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            
            # Close any remaining active trades
            await self._close_all_active_trades("Engine shutdown")
            
            # Generate final performance report
            self._generate_performance_report()
            
            self.logger.info("Automated trading engine stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {str(e)}")
            return False
    
    async def submit_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """
        Submit a trading opportunity for automated execution.
        
        Args:
            opportunity: Trading opportunity to execute
            
        Returns:
            bool: True if opportunity was queued, False otherwise
        """
        try:
            # Validate opportunity
            if not self._validate_opportunity(opportunity):
                return False
            
            # Check if we can accept more trades
            if not self._can_accept_new_trade():
                self.logger.debug(f"Cannot accept new trade: limits reached")
                return False
            
            # Add to opportunity queue
            await self.opportunities.put(opportunity)
            self.stats['total_opportunities'] += 1
            
            self.logger.info(f"Opportunity queued: {opportunity.strategy} {opportunity.symbol} {opportunity.side.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting opportunity: {str(e)}")
            return False
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status and statistics.
        
        Returns:
            dict: Portfolio status information
        """
        # Calculate current portfolio value
        portfolio_value = self.available_capital + self.allocated_capital + self.total_profit
        
        # Calculate ROI
        roi = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        return {
            'trading_mode': self.trading_mode.value,
            'is_running': self.is_running,
            'portfolio_value': float(portfolio_value),
            'available_capital': float(self.available_capital),
            'allocated_capital': float(self.allocated_capital),
            'total_profit': float(self.total_profit),
            'daily_profit': float(self.daily_profit),
            'roi_percentage': float(roi),
            'active_trades': len(self.active_trades),
            'daily_trades_count': self.daily_trades_count,
            'opportunities_in_queue': self.opportunities.qsize(),
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'consecutive_losses': self.consecutive_losses,
            'stats': {
                k: float(v) if isinstance(v, Decimal) else v 
                for k, v in self.stats.items()
            }
        }
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active trades.
        
        Returns:
            list: List of active trade dictionaries
        """
        return [
            {
                'id': trade.id,
                'strategy': trade.strategy,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': float(trade.quantity),
                'executed_price': float(trade.executed_price),
                'current_profit': float(trade.actual_profit),
                'execution_time': trade.execution_time,
                'status': trade.status,
                'executed_at': trade.executed_at.isoformat(),
                'hold_time': (datetime.now() - trade.executed_at).total_seconds()
            }
            for trade in self.active_trades.values()
        ]
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of recent completed trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            list: List of recent trade dictionaries
        """
        recent_trades = sorted(
            self.completed_trades, 
            key=lambda t: t.executed_at, 
            reverse=True
        )[:limit]
        
        return [
            {
                'id': trade.id,
                'strategy': trade.strategy,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': float(trade.quantity),
                'executed_price': float(trade.executed_price),
                'actual_profit': float(trade.actual_profit),
                'fees': float(trade.fees),
                'execution_time': trade.execution_time,
                'status': trade.status,
                'executed_at': trade.executed_at.isoformat(),
                'closed_at': trade.closed_at.isoformat() if trade.closed_at else None
            }
            for trade in recent_trades
        ]
    
    async def _trading_loop(self):
        """
        Main trading loop that processes opportunities and executes trades.
        """
        self.logger.info("Starting automated trading loop")
        
        while self.is_running:
            try:
                # Check circuit breaker
                if self.circuit_breaker_triggered:
                    self.logger.warning("Circuit breaker triggered, pausing trading")
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Get next opportunity (with timeout)
                try:
                    opportunity = await asyncio.wait_for(
                        self.opportunities.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if opportunity is still valid
                if not self._is_opportunity_valid(opportunity):
                    self.logger.debug(f"Opportunity expired: {opportunity.id}")
                    continue
                
                # Execute the trade
                await self._execute_trade(opportunity)
                
                # Small delay between executions
                await asyncio.sleep(self.execution_delay)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _execute_trade(self, opportunity: TradingOpportunity):
        """
        Execute a trading opportunity.
        
        Args:
            opportunity: Trading opportunity to execute
        """
        start_time = time.time()
        
        try:
            # Calculate position size
            position_size = self._calculate_position_size(opportunity)
            if position_size <= 0:
                self.logger.warning(f"Invalid position size for {opportunity.symbol}")
                return
            
            # Create trade record
            trade = ExecutedTrade(
                opportunity_id=opportunity.id,
                strategy=opportunity.strategy,
                symbol=opportunity.symbol,
                side=opportunity.side,
                quantity=position_size,
                executed_price=opportunity.entry_price,
                target_price=opportunity.target_price,
                stop_loss_price=opportunity.stop_loss_price,
                status="pending"
            )
            
            # Execute based on trading mode
            if self.trading_mode == TradingMode.SIMULATION:
                success = await self._execute_simulated_trade(trade, opportunity)
            elif self.trading_mode == TradingMode.PAPER:
                success = await self._execute_paper_trade(trade, opportunity)
            elif self.trading_mode == TradingMode.LIVE:
                success = await self._execute_live_trade(trade, opportunity)
            else:
                success = False
            
            # Record execution time
            trade.execution_time = time.time() - start_time
            
            if success:
                # Add to active trades
                self.active_trades[trade.id] = trade
                
                # Update capital allocation
                trade_value = trade.quantity * trade.executed_price
                self.allocated_capital += trade_value
                self.available_capital -= trade_value
                
                # Update statistics
                self.stats['opportunities_executed'] += 1
                self.daily_trades_count += 1
                
                self.logger.info(
                    f"Trade executed: {trade.strategy} {trade.symbol} {trade.side.value} "
                    f"{trade.quantity} @ {trade.executed_price}"
                )
            else:
                trade.status = "failed"
                self.stats['failed_trades'] += 1
                self.consecutive_losses += 1
                
                self.logger.warning(f"Trade execution failed: {opportunity.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
    
    async def _execute_simulated_trade(self, trade: ExecutedTrade, opportunity: TradingOpportunity) -> bool:
        """
        Execute a simulated trade (for testing and development).
        
        Args:
            trade: Trade to execute
            opportunity: Original opportunity
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Simulate some market impact and slippage
            slippage_factor = np.random.uniform(0.998, 1.002)  # ±0.2% slippage
            actual_price = trade.executed_price * Decimal(str(slippage_factor))
            
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Update trade with actual execution details
            trade.executed_price = actual_price
            trade.status = "completed"
            trade.exchange = "simulation"
            trade.order_id = f"sim_{uuid.uuid4().hex[:8]}"
            
            # Calculate fees (0.1% default)
            trade_value = trade.quantity * trade.executed_price
            trade.fees = trade_value * Decimal('0.001')
            
            # Calculate profit based on expected profit with some randomization
            profit_variance = np.random.uniform(0.8, 1.2)  # ±20% variance
            expected_profit_value = trade_value * opportunity.expected_profit
            trade.actual_profit = expected_profit_value * Decimal(str(profit_variance)) - trade.fees
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in simulated trade execution: {str(e)}")
            return False
    
    async def _execute_paper_trade(self, trade: ExecutedTrade, opportunity: TradingOpportunity) -> bool:
        """
        Execute a paper trade (real market data, no actual money).
        
        Args:
            trade: Trade to execute
            opportunity: Original opportunity
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # In paper trading, we would get real market prices
            # For now, simulate with more realistic market behavior
            
            # Simulate order book impact
            market_impact = np.random.uniform(0.9995, 1.0005)  # ±0.05% impact
            actual_price = trade.executed_price * Decimal(str(market_impact))
            
            # Simulate variable execution delay
            execution_delay = np.random.uniform(0.05, 0.5)
            await asyncio.sleep(execution_delay)
            
            # Update trade details
            trade.executed_price = actual_price
            trade.status = "completed"
            trade.exchange = "paper"
            trade.order_id = f"paper_{uuid.uuid4().hex[:8]}"
            
            # More realistic fee calculation
            trade_value = trade.quantity * trade.executed_price
            trade.fees = trade_value * Decimal('0.001')  # 0.1% fee
            
            # More conservative profit calculation
            expected_profit_value = trade_value * opportunity.expected_profit
            profit_realization = np.random.uniform(0.7, 1.1)  # 70-110% of expected
            trade.actual_profit = expected_profit_value * Decimal(str(profit_realization)) - trade.fees
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in paper trade execution: {str(e)}")
            return False
    
    async def _execute_live_trade(self, trade: ExecutedTrade, opportunity: TradingOpportunity) -> bool:
        """
        Execute a live trade (real money, real market).
        
        Args:
            trade: Trade to execute
            opportunity: Original opportunity
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # WARNING: This is where real money would be traded
            # Implementation would involve:
            # 1. Connect to exchange API
            # 2. Place market/limit order
            # 3. Monitor order status
            # 4. Handle partial fills
            # 5. Update trade record with actual results
            
            self.logger.warning("Live trading execution not implemented - use simulation or paper mode")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in live trade execution: {str(e)}")
            return False
    
    async def _monitoring_loop(self):
        """
        Background loop that monitors active trades and manages positions.
        """
        self.logger.info("Starting trade monitoring loop")
        
        while self.is_running:
            try:
                # Monitor active trades
                await self._monitor_active_trades()
                
                # Update performance statistics
                self._update_performance_stats()
                
                # Check risk management rules
                self._check_risk_management()
                
                # Sleep for monitoring interval
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10.0)
    
    async def _monitor_active_trades(self):
        """
        Monitor active trades for exit conditions.
        """
        trades_to_close = []
        
        for trade_id, trade in self.active_trades.items():
            try:
                # Check if trade should be closed
                should_close, reason = self._should_close_trade(trade)
                
                if should_close:
                    trades_to_close.append((trade_id, reason))
                    
            except Exception as e:
                self.logger.error(f"Error monitoring trade {trade_id}: {str(e)}")
        
        # Close trades that meet exit conditions
        for trade_id, reason in trades_to_close:
            await self._close_trade(trade_id, reason)
    
    def _should_close_trade(self, trade: ExecutedTrade) -> Tuple[bool, str]:
        """
        Determine if a trade should be closed.
        
        Args:
            trade: Trade to evaluate
            
        Returns:
            tuple: (should_close, reason)
        """
        # Check maximum hold time
        hold_time = (datetime.now() - trade.executed_at).total_seconds()
        if hold_time > 3600:  # 1 hour maximum
            return True, "max_hold_time"
        
        # For simulation mode, simulate position closure
        if self.trading_mode == TradingMode.SIMULATION:
            # Randomly close some positions to simulate market movements
            if np.random.random() < 0.1:  # 10% chance per check
                return True, "simulated_closure"
        
        return False, ""
    
    async def _close_trade(self, trade_id: str, reason: str):
        """
        Close an active trade.
        
        Args:
            trade_id: ID of trade to close
            reason: Reason for closing
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            # Mark trade as closed
            trade.status = "completed"
            trade.closed_at = datetime.now()
            trade.metadata['close_reason'] = reason
            
            # Update capital allocation
            trade_value = trade.quantity * trade.executed_price
            self.allocated_capital -= trade_value
            self.available_capital += trade_value + trade.actual_profit
            
            # Update profit tracking
            self.total_profit += trade.actual_profit
            self.daily_profit += trade.actual_profit
            
            # Move to completed trades
            self.completed_trades.append(trade)
            self.trade_history.append(trade)
            del self.active_trades[trade_id]
            
            # Update statistics
            if trade.actual_profit > 0:
                self.stats['successful_trades'] += 1
                self.consecutive_losses = 0
            else:
                self.stats['failed_trades'] += 1
                self.consecutive_losses += 1
            
            self.logger.info(
                f"Trade closed: {trade.symbol} profit: {trade.actual_profit:.2f} reason: {reason}"
            )
            
        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {str(e)}")
    
    async def _close_all_active_trades(self, reason: str):
        """
        Close all active trades.
        
        Args:
            reason: Reason for closing all trades
        """
        trade_ids = list(self.active_trades.keys())
        for trade_id in trade_ids:
            await self._close_trade(trade_id, reason)
    
    async def _cleanup_loop(self):
        """
        Background loop for cleanup tasks.
        """
        while self.is_running:
            try:
                # Reset daily counters if needed
                self._reset_daily_counters_if_needed()
                
                # Cleanup old completed trades (keep last 1000)
                if len(self.completed_trades) > 1000:
                    self.completed_trades = self.completed_trades[-1000:]
                
                # Sleep for cleanup interval
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    def _validate_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """
        Validate a trading opportunity.
        
        Args:
            opportunity: Opportunity to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check minimum profit threshold
        if opportunity.expected_profit < self.min_profit_threshold:
            return False
        
        # Check required fields
        if not all([opportunity.strategy, opportunity.symbol, opportunity.quantity]):
            return False
        
        # Check quantity is positive
        if opportunity.quantity <= 0:
            return False
        
        # Check entry price is positive
        if opportunity.entry_price <= 0:
            return False
        
        return True
    
    def _can_accept_new_trade(self) -> bool:
        """
        Check if we can accept a new trade based on limits.
        
        Returns:
            bool: True if we can accept new trade, False otherwise
        """
        # Check if engine is running
        if not self.is_running:
            return False
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False
        
        # Check concurrent trade limit
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        
        # Check daily trade limit
        if self.daily_trades_count >= self.max_daily_trades:
            return False
        
        # Check available capital
        if self.available_capital <= self.base_trade_size:
            return False
        
        return True
    
    def _is_opportunity_valid(self, opportunity: TradingOpportunity) -> bool:
        """
        Check if an opportunity is still valid.
        
        Args:
            opportunity: Opportunity to check
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if opportunity has expired
        if opportunity.expires_at and datetime.now() > opportunity.expires_at:
            return False
        
        # Check opportunity age
        age = (datetime.now() - opportunity.created_at).total_seconds()
        if age > self.opportunity_timeout:
            return False
        
        return True
    
    def _calculate_position_size(self, opportunity: TradingOpportunity) -> Decimal:
        """
        Calculate the position size for a trade.
        
        Args:
            opportunity: Trading opportunity
            
        Returns:
            Decimal: Position size
        """
        # Calculate maximum position size based on risk limits
        max_risk_amount = self.available_capital * self.max_risk_per_trade
        max_position_value = max_risk_amount / self.max_slippage  # Conservative estimate
        
        # Use minimum of base trade size and risk-adjusted size
        trade_value = min(self.base_trade_size, max_position_value)
        
        # Calculate quantity
        position_size = trade_value / opportunity.entry_price
        
        # Round down to avoid over-allocation
        return position_size.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
    
    def _update_performance_stats(self):
        """
        Update performance statistics.
        """
        try:
            if not self.completed_trades:
                return
            
            # Calculate basic stats
            profits = [float(trade.actual_profit) for trade in self.completed_trades]
            
            self.stats['total_profit'] = sum(profits)
            self.stats['average_profit'] = np.mean(profits) if profits else 0
            self.stats['win_rate'] = len([p for p in profits if p > 0]) / len(profits) * 100
            
            # Calculate execution time stats
            execution_times = [trade.execution_time for trade in self.completed_trades if trade.execution_time > 0]
            self.stats['average_execution_time'] = np.mean(execution_times) if execution_times else 0
            
            # Calculate best/worst trades
            if profits:
                self.stats['best_trade'] = max(profits)
                self.stats['worst_trade'] = min(profits)
            
            # Calculate profit factor
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [abs(p) for p in profits if p < 0]
            
            if losing_trades:
                self.stats['profit_factor'] = sum(winning_trades) / sum(losing_trades)
            else:
                self.stats['profit_factor'] = float('inf') if winning_trades else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(profits) > 1:
                self.stats['sharpe_ratio'] = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {str(e)}")
    
    def _check_risk_management(self):
        """
        Check risk management rules and trigger circuit breaker if needed.
        """
        try:
            # Check daily loss limit
            if self.daily_profit < -self.max_daily_loss * self.initial_capital:
                self.circuit_breaker_triggered = True
                self.logger.warning("Circuit breaker triggered: Daily loss limit exceeded")
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.circuit_breaker_triggered = True
                self.logger.warning("Circuit breaker triggered: Too many consecutive losses")
            
            # Reset circuit breaker if conditions improve
            if (self.circuit_breaker_triggered and 
                self.daily_profit > -self.max_daily_loss * self.initial_capital * 0.5 and
                self.consecutive_losses < self.max_consecutive_losses * 0.8):
                
                self.circuit_breaker_triggered = False
                self.logger.info("Circuit breaker reset: Conditions improved")
            
        except Exception as e:
            self.logger.error(f"Error in risk management check: {str(e)}")
    
    def _reset_daily_counters_if_needed(self):
        """
        Reset daily counters if it's a new day.
        """
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_profit = Decimal('0')
            self.last_reset_date = current_date
            self.circuit_breaker_triggered = False  # Reset daily circuit breaker
            self.consecutive_losses = 0  # Reset at start of new day
            self.logger.info("Daily counters reset for new trading day")
    
    def _generate_performance_report(self):
        """
        Generate a performance report.
        """
        try:
            report = {
                'trading_mode': self.trading_mode.value,
                'total_trades': len(self.completed_trades),
                'active_trades': len(self.active_trades),
                'total_profit': float(self.total_profit),
                'daily_profit': float(self.daily_profit),
                'portfolio_value': float(self.available_capital + self.allocated_capital + self.total_profit),
                'roi_percentage': float((self.total_profit / self.initial_capital) * 100) if self.initial_capital > 0 else 0,
                'statistics': self.stats.copy()
            }
            
            # Save report to file
            report_file = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")

