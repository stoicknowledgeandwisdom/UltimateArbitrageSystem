#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Broker Interface
====================

Abstract base class for all broker integrations with:
- Standardized order types and status
- Position and trade tracking
- Real-time updates
- Error handling and retry logic
- Performance monitoring
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from decimal import Decimal
import uuid

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Standard order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    MOO = "moo"  # Market-On-Open
    MOC = "moc"  # Market-On-Close
    LOO = "loo"  # Limit-On-Open
    LOC = "loc"  # Limit-On-Close

class OrderStatus(Enum):
    """Standard order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"

class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date

@dataclass
class Order:
    """Standardized order representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal('0')
    
    # Price parameters
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trail_amount: Optional[Decimal] = None
    trail_percent: Optional[float] = None
    
    # Execution parameters
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    
    # Bracket order parameters
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    
    # Metadata
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    legs: List['Order'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                              OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'broker_id': self.broker_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.type.value,
            'quantity': float(self.quantity),
            'limit_price': float(self.limit_price) if self.limit_price else None,
            'stop_price': float(self.stop_price) if self.stop_price else None,
            'trail_amount': float(self.trail_amount) if self.trail_amount else None,
            'trail_percent': self.trail_percent,
            'time_in_force': self.time_in_force.value,
            'extended_hours': self.extended_hours,
            'status': self.status.value,
            'filled_quantity': float(self.filled_quantity),
            'remaining_quantity': float(self.remaining_quantity) if self.remaining_quantity else None,
            'average_fill_price': float(self.average_fill_price) if self.average_fill_price else None,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'canceled_at': self.canceled_at.isoformat() if self.canceled_at else None,
            'take_profit_price': float(self.take_profit_price) if self.take_profit_price else None,
            'stop_loss_price': float(self.stop_loss_price) if self.stop_loss_price else None,
            'client_order_id': self.client_order_id,
            'parent_order_id': self.parent_order_id,
            'metadata': self.metadata
        }

@dataclass
class Trade:
    """Trade execution record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    broker_trade_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal('0')
    price: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.now)
    commission: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including fees."""
        return self.notional_value + self.commission + self.fees
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'broker_trade_id': self.broker_trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'timestamp': self.timestamp.isoformat(),
            'commission': float(self.commission),
            'fees': float(self.fees),
            'notional_value': float(self.notional_value),
            'total_cost': float(self.total_cost),
            'metadata': self.metadata
        }

@dataclass
class Position:
    """Position tracking."""
    symbol: str
    quantity: Decimal = Decimal('0')
    average_cost: Decimal = Decimal('0')
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal('0')
    
    # Position details
    long_quantity: Decimal = Decimal('0')
    short_quantity: Decimal = Decimal('0')
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0
    
    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value."""
        if self.market_value is None:
            return None
        return abs(self.quantity) * self.market_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'average_cost': float(self.average_cost),
            'market_value': float(self.market_value) if self.market_value else None,
            'unrealized_pnl': float(self.unrealized_pnl) if self.unrealized_pnl else None,
            'realized_pnl': float(self.realized_pnl),
            'long_quantity': float(self.long_quantity),
            'short_quantity': float(self.short_quantity),
            'is_long': self.is_long,
            'is_short': self.is_short,
            'is_flat': self.is_flat,
            'notional_value': float(self.notional_value) if self.notional_value else None,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class BrokerStats:
    """Broker performance statistics."""
    broker_name: str
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_trades: int = 0
    total_volume: Decimal = Decimal('0')
    average_fill_time_ms: float = 0.0
    uptime_percentage: float = 100.0
    last_heartbeat: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate order success rate."""
        if self.total_orders == 0:
            return 100.0
        return (self.successful_orders / self.total_orders) * 100.0

class BaseBroker(ABC):
    """Abstract base class for all broker integrations."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.stats = BrokerStats(broker_name=name)
        
        # Order and position tracking
        self.orders: Dict[str, Order] = {}
        self.trades: Dict[str, Trade] = {}
        self.positions: Dict[str, Position] = {}
        
        # Event callbacks
        self.order_update_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.position_update_callbacks: List[Callable] = []
        
        # Rate limiting
        self.rate_limiter = None
        self._setup_rate_limiter()
        
    def _setup_rate_limiter(self):
        """Setup rate limiting based on broker limits."""
        requests_per_minute = self.config.get('requests_per_minute', 200)
        self.request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def add_order_update_callback(self, callback: Callable[[Order], None]):
        """Add callback for order updates."""
        self.order_update_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]):
        """Add callback for trade notifications."""
        self.trade_callbacks.append(callback)
    
    def add_position_update_callback(self, callback: Callable[[Position], None]):
        """Add callback for position updates."""
        self.position_update_callbacks.append(callback)
    
    async def _notify_order_update(self, order: Order):
        """Notify all order update callbacks."""
        for callback in self.order_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
    
    async def _notify_trade(self, trade: Trade):
        """Notify all trade callbacks."""
        for callback in self.trade_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trade)
                else:
                    callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    async def _notify_position_update(self, position: Position):
        """Notify all position update callbacks."""
        for callback in self.position_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(position)
                else:
                    callback(position)
            except Exception as e:
                logger.error(f"Error in position update callback: {e}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submit an order to the broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """Modify an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass
    
    @abstractmethod
    async def stream_order_updates(self) -> AsyncGenerator[Order, None]:
        """Stream real-time order updates."""
        pass
    
    @abstractmethod
    async def stream_trade_updates(self) -> AsyncGenerator[Trade, None]:
        """Stream real-time trade updates."""
        pass
    
    @abstractmethod
    def get_supported_order_types(self) -> List[OrderType]:
        """Get list of supported order types."""
        pass
    
    @abstractmethod
    def get_supported_time_in_force(self) -> List[TimeInForce]:
        """Get list of supported time in force options."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the broker connection."""
        try:
            start_time = time.time()
            
            # Try to get account info as a basic health check
            account_info = await self.get_account_info()
            
            latency = (time.time() - start_time) * 1000
            
            return {
                'broker': self.name,
                'status': 'healthy' if account_info else 'degraded',
                'connected': self.is_connected,
                'latency_ms': latency,
                'stats': {
                    'success_rate': self.stats.success_rate,
                    'total_orders': self.stats.total_orders,
                    'total_trades': self.stats.total_trades,
                    'uptime': self.stats.uptime_percentage
                }
            }
        except Exception as e:
            return {
                'broker': self.name,
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    def get_stats(self) -> BrokerStats:
        """Get broker performance statistics."""
        return self.stats
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID."""
        return self.trades.get(trade_id)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [order for order in self.orders.values() if order.is_active]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [order for order in self.orders.values() if order.symbol == symbol]

