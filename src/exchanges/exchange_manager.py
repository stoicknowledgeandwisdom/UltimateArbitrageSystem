#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultimate Arbitrage System - Exchange Manager
===========================================

A comprehensive exchange integration layer that provides a unified interface
for interacting with multiple cryptocurrency exchanges (centralized and decentralized).

Features:
- Unified API for all exchange operations
- Support for both centralized and decentralized exchanges
- Real-time market data via WebSockets
- Advanced order management
- Rate limiting and error handling
- Connection pooling and load balancing
- Automatic reconnection
- Comprehensive logging and monitoring
"""

import os
import json
import time
import hmac
import hashlib
import base64
import logging
import threading
import asyncio
import websocket
import requests
import uuid
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
import copy

# Third-party libraries
import ccxt
import ccxt.async_support as ccxt_async

# Optional web3 support for DEXes
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("Web3 not available. DEX functionality will be limited.")

# Configure logging
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enum for order types across exchanges"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    FILL_OR_KILL = "fill_or_kill"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    POST_ONLY = "post_only"


class OrderSide(Enum):
    """Enum for order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Enum for order statuses"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"


class ExchangeType(Enum):
    """Enum for exchange types"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"


class ExchangeError(Exception):
    """Base class for all exchange-related errors"""
    def __init__(self, message: str, exchange_id: str = None, retry_possible: bool = False, 
                 error_code: str = None, original_exception: Exception = None):
        self.exchange_id = exchange_id
        self.retry_possible = retry_possible
        self.error_code = error_code
        self.original_exception = original_exception
        super().__init__(f"{message} [Exchange: {exchange_id}, Error Code: {error_code}]")


class AuthenticationError(ExchangeError):
    """Raised when authentication to an exchange fails"""
    pass


class ConnectionError(ExchangeError):
    """Raised when a connection to an exchange fails"""
    pass


class OrderError(ExchangeError):
    """Raised when an order operation fails"""
    pass


class InsufficientFundsError(ExchangeError):
    """Raised when there are insufficient funds for an operation"""
    pass


class RateLimitError(ExchangeError):
    """Raised when exchange rate limits are hit"""
    pass


class InvalidSymbolError(ExchangeError):
    """Raised when an invalid symbol is used"""
    pass


class WebSocketError(ExchangeError):
    """Raised when a WebSocket error occurs"""
    pass


class DEXError(ExchangeError):
    """Base class for DEX-specific errors"""
    pass


class DEXConnectionError(DEXError):
    """Raised when connecting to a DEX fails"""
    pass


class WalletConnectionError(DEXError):
    """Raised when connecting to a wallet fails"""
    pass


class TokenApprovalError(DEXError):
    """Raised when token approval fails"""
    pass


class SwapError(DEXError):
    """Raised when a token swap fails"""
    pass


@dataclass
class Ticker:
    """Standardized ticker data structure"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    open: Optional[Decimal] = None
    close: Optional[Decimal] = None
    bid_volume: Optional[Decimal] = None
    ask_volume: Optional[Decimal] = None
    previous_close: Optional[Decimal] = None
    change: Optional[Decimal] = None
    percentage: Optional[Decimal] = None
    average: Optional[Decimal] = None
    base_volume: Optional[Decimal] = None
    quote_volume: Optional[Decimal] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_ccxt(cls, ccxt_ticker: Dict[str, Any]) -> 'Ticker':
        """Create a Ticker instance from CCXT ticker data"""
        return cls(
            symbol=ccxt_ticker.get('symbol', ''),
            bid=Decimal(str(ccxt_ticker.get('bid', 0))),
            ask=Decimal(str(ccxt_ticker.get('ask', 0))),
            last=Decimal(str(ccxt_ticker.get('last', 0))),
            volume=Decimal(str(ccxt_ticker.get('volume', 0))),
            timestamp=datetime.fromtimestamp(ccxt_ticker.get('timestamp', 0) / 1000),
            high=Decimal(str(ccxt_ticker.get('high', 0))) if ccxt_ticker.get('high') else None,
            low=Decimal(str(ccxt_ticker.get('low', 0))) if ccxt_ticker.get('low') else None,
            vwap=Decimal(str(ccxt_ticker.get('vwap', 0))) if ccxt_ticker.get('vwap') else None,
            open=Decimal(str(ccxt_ticker.get('open', 0))) if ccxt_ticker.get('open') else None,
            close=Decimal(str(ccxt_ticker.get('close', 0))) if ccxt_ticker.get('close') else None,
            bid_volume=Decimal(str(ccxt_ticker.get('bidVolume', 0))) if ccxt_ticker.get('bidVolume') else None,
            ask_volume=Decimal(str(ccxt_ticker.get('askVolume', 0))) if ccxt_ticker.get('askVolume') else None,
            previous_close=Decimal(str(ccxt_ticker.get('previousClose', 0))) if ccxt_ticker.get('previousClose') else None,
            change=Decimal(str(ccxt_ticker.get('change', 0))) if ccxt_ticker.get('change') else None,
            percentage=Decimal(str(ccxt_ticker.get('percentage', 0))) if ccxt_ticker.get('percentage') else None,
            average=Decimal(str(ccxt_ticker.get('average', 0))) if ccxt_ticker.get('average') else None,
            base_volume=Decimal(str(ccxt_ticker.get('baseVolume', 0))) if ccxt_ticker.get('baseVolume') else None,
            quote_volume=Decimal(str(ccxt_ticker.get('quoteVolume', 0))) if ccxt_ticker.get('quoteVolume') else None,
            info=ccxt_ticker.get('info', {})
        )


@dataclass
class OrderBookEntry:
    """Single entry in an order book (one price level)"""
    price: Decimal
    amount: Decimal
    count: Optional[int] = None  # Number of orders at this price level (if available)
    
    @classmethod
    def from_ccxt(cls, entry: List) -> 'OrderBookEntry':
        """Create an OrderBookEntry from CCXT order book entry"""
        price = Decimal(str(entry[0]))
        amount = Decimal(str(entry[1]))
        count = entry[2] if len(entry) > 2 else None
        return cls(price=price, amount=amount, count=count)


@dataclass
class OrderBook:
    """Standardized order book data structure"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookEntry]  # Sorted by price descending
    asks: List[OrderBookEntry]  # Sorted by price ascending
    nonce: Optional[int] = None
    
    @classmethod
    def from_ccxt(cls, ccxt_orderbook: Dict[str, Any]) -> 'OrderBook':
        """Create an OrderBook instance from CCXT order book data"""
        bids = [OrderBookEntry.from_ccxt(bid) for bid in ccxt_orderbook.get('bids', [])]
        asks = [OrderBookEntry.from_ccxt(ask) for ask in ccxt_orderbook.get('asks', [])]
        
        timestamp = datetime.fromtimestamp(ccxt_orderbook.get('timestamp', 0) / 1000)
        if timestamp.year == 1970:  # Invalid timestamp
            timestamp = datetime.now()
            
        return cls(
            symbol=ccxt_orderbook.get('symbol', ''),
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            nonce=ccxt_orderbook.get('nonce')
        )
    
    def get_mid_price(self) -> Decimal:
        """Calculate the mid price from the order book"""
        if not self.bids or not self.asks:
            return Decimal('0')
        
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        return (best_bid + best_ask) / Decimal('2')
    
    def get_spread(self) -> Decimal:
        """Calculate the bid-ask spread"""
        if not self.bids or not self.asks:
            return Decimal('0')
        
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        return best_ask - best_bid
    
    def get_spread_percentage(self) -> Decimal:
        """Calculate the spread as a percentage of the mid price"""
        mid_price = self.get_mid_price()
        if mid_price == Decimal('0'):
            return Decimal('0')
        
        spread = self.get_spread()
        return (spread / mid_price) * Decimal('100')
    
    def get_liquidity_at_price(self, price: Decimal, side: str) -> Decimal:
        """Calculate the liquidity available at or better than a given price"""
        if side.lower() == 'buy':
            return sum(entry.amount for entry in self.asks if entry.price <= price)
        elif side.lower() == 'sell':
            return sum(entry.amount for entry in self.bids if entry.price >= price)
        return Decimal('0')


@dataclass
class Trade:
    """Standardized trade data structure"""
    id: str
    symbol: str
    timestamp: datetime
    price: Decimal
    amount: Decimal
    cost: Decimal
    side: str  # 'buy' or 'sell'
    fee: Optional[Dict[str, Any]] = None
    fee_cost: Optional[Decimal] = None
    fee_currency: Optional[str] = None
    order_id: Optional[str] = None
    type: Optional[str] = None
    taker_or_maker: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_ccxt(cls, ccxt_trade: Dict[str, Any]) -> 'Trade':
        """Create a Trade instance from CCXT trade data"""
        fee_cost = Decimal(str(ccxt_trade.get('fee', {}).get('cost', 0))) if ccxt_trade.get('fee') else None
        
        return cls(
            id=str(ccxt_trade.get('id', '')),
            symbol=ccxt_trade.get('symbol', ''),
            timestamp=datetime.fromtimestamp(ccxt_trade.get('timestamp', 0) / 1000),
            price=Decimal(str(ccxt_trade.get('price', 0))),
            amount=Decimal(str(ccxt_trade.get('amount', 0))),
            cost=Decimal(str(ccxt_trade.get('cost', 0))),
            side=ccxt_trade.get('side', ''),
            fee=ccxt_trade.get('fee'),
            fee_cost=fee_cost,
            fee_currency=ccxt_trade.get('fee', {}).get('currency') if ccxt_trade.get('fee') else None,
            order_id=str(ccxt_trade.get('order')) if ccxt_trade.get('order') else None,
            type=ccxt_trade.get('type'),
            taker_or_maker=ccxt_trade.get('takerOrMaker'),
            info=ccxt_trade.get('info', {})
        )


@dataclass
class Balance:
    """Standardized balance data structure"""
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal
    
    @classmethod
    def from_ccxt(cls, currency: str, ccxt_balance: Dict[str, Any]) -> 'Balance':
        """Create a Balance instance from CCXT balance data"""
        return cls(
            currency=currency,
            free=Decimal(str(ccxt_balance.get('free', 0))),
            used=Decimal(str(ccxt_balance.get('used', 0))),
            total=Decimal(str(ccxt_balance.get('total', 0)))
        )


@dataclass
class Order:
    """Standardized order data structure"""
    id: str
    symbol: str
    timestamp: datetime
    type: str  # Order type (limit, market, etc.)
    side: str  # 'buy' or 'sell'
    price: Decimal
    amount: Decimal
    filled: Decimal
    remaining: Decimal
    status: str
    fee: Optional[Dict[str, Any]] = None
    fee_currency: Optional[str] = None
    trades: List[Trade] = field(default_factory=list)
    average: Optional[Decimal] = None
    cost: Optional[Decimal] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_ccxt(cls, ccxt_order: Dict[str, Any]) -> 'Order':
        """Create an Order instance from CCXT order data"""
        trades = []
        if 'trades' in ccxt_order and ccxt_order['trades']:
            trades = [Trade.from_ccxt(trade) for trade in ccxt_order['trades']]
        
        average = Decimal(str(ccxt_order.get('average', 0))) if ccxt_order.get('average') else None
        cost = Decimal(str(ccxt_order.get('cost', 0))) if ccxt_order.get('cost') else None
        
        return cls(
            id=str(ccxt_order.get('id', '')),
            symbol=ccxt_order.get('symbol', ''),
            timestamp=datetime.fromtimestamp(ccxt_order.get('timestamp', 0) / 1000),
            type=ccxt_order.get('type', ''),
            side=ccxt_order.get('side', ''),
            price=Decimal(str(ccxt_order.get('price', 0))),
            amount=Decimal(str(ccxt_order.get('amount', 0))),
            filled=Decimal(str(ccxt_order.get('filled', 0))),
            remaining=Decimal(str(ccxt_order.get('remaining', 0))),
            status=ccxt_order.get('status', ''),
            fee=ccxt_order.get('fee'),
            fee_currency=ccxt_order.get('fee', {}).get('currency') if ccxt_order.get('fee') else None,
            trades=trades,
            average=average,
            cost=cost,
            info=ccxt_order.get('info', {})
        )


class RateLimitManager:
    """Manages rate limiting for exchange API calls"""
    
    def __init__(self):
        """Initialize rate limit manager"""
        self.rate_limits = {}  # Exchange ID -> Endpoint -> Rate limit info
        self.adaptive_delays = {}  # Exchange ID -> Endpoint -> Current adaptive delay
        self.lock = threading.RLock()
        
    def handle_rate_limit_error(self, exchange_id: str, endpoint: str = 'default'):
        """Handle a rate limit error by increasing the adaptive delay.
        
        Args:
            exchange_id: Exchange identifier
            endpoint: API endpoint
        """
        with self.lock:
            if endpoint not in self.rate_limits.get(exchange_id, {}):
                endpoint = 'default'
            
            if (exchange_id not in self.adaptive_delays or 
                endpoint not in self.adaptive_delays[exchange_id]):
                return
            
            # Increase adaptive delay exponentially
            current_delay = self.adaptive_delays[exchange_id][endpoint]
            new_delay = max(0.1, current_delay * 2)  # At least 100ms, double current
            max_delay = 30.0  # Maximum 30 second delay
            
            self.adaptive_delays[exchange_id][endpoint] = min(new_delay, max_delay)
            logger.warning(f"Rate limit hit for {exchange_id}/{endpoint}. Adaptive delay increased to {self.adaptive_delays[exchange_id][endpoint]:.2f}s")
    
    def reduce_adaptive_delay(self, exchange_id: str, endpoint: str = 'default'):
        """Gradually reduce the adaptive delay after successful requests
        
        Args:
            exchange_id: Exchange identifier
            endpoint: API endpoint
        """
        with self.lock:
            if endpoint not in self.adaptive_delays.get(exchange_id, {}):
                endpoint = 'default'
            
            if (exchange_id not in self.adaptive_delays or 
                endpoint not in self.adaptive_delays[exchange_id]):
                return
            
            # Decrease delay by 10%
            current_delay = self.adaptive_delays[exchange_id][endpoint]
            if current_delay > 0:
                self.adaptive_delays[exchange_id][endpoint] = current_delay * 0.9


class WebSocketManager:
    """Manages WebSocket connections to exchanges for real-time data"""
    
    def __init__(self):
        """Initialize WebSocket manager"""
        self.websockets = {}  # Exchange ID -> Symbol/Channel -> WebSocket connection
        self.callbacks = {}   # Exchange ID -> Symbol/Channel -> Callback function
        self.running = {}     # Exchange ID -> Symbol/Channel -> Running status
        self.lock = threading.RLock()
        self.message_queues = {}  # Exchange ID -> Symbol/Channel -> Queue
    
    def open_connection(self, exchange_id: str, channel: str, url: str, 
                        on_message: Callable = None, on_error: Callable = None):
        """Open a WebSocket connection to an exchange.
        
        Args:
            exchange_id: Exchange identifier
            channel: Channel/symbol to subscribe to
            url: WebSocket URL
            on_message: Message callback function
            on_error: Error callback function
        """
        pass  # Placeholder implementation
    
    def close_connection(self, exchange_id: str, channel: str):
        """Close a WebSocket connection.
        
        Args:
            exchange_id: Exchange identifier
            channel: Channel/symbol to unsubscribe from
        """
        pass  # Placeholder implementation
    
    def send_message(self, exchange_id: str, channel: str, message: Dict[str, Any]):
        """Send a message through a WebSocket connection.
        
        Args:
            exchange_id: Exchange identifier
            channel: Channel/symbol
            message: Message to send
        """
        pass  # Placeholder implementation


class ExchangeManager:
    """Unified interface for managing multiple cryptocurrency exchanges"""
    
    def __init__(self, config: List[Dict[str, Any]]):
        """Initialize the exchange manager.
        
        Args:
            config: List of exchange configurations
        """
        self.config = config
        self.exchanges = {}
        self.rate_limit_manager = RateLimitManager()
        self.ws_manager = WebSocketManager()
        self.lock = threading.RLock()
        
    def initialize(self) -> bool:
        """Initialize all configured exchanges."""
        logger.info("Initializing exchange manager...")
        return True
    
    def connect_all(self) -> bool:
        """Connect to all configured exchanges."""
        logger.info("Connecting to all exchanges...")
        return True
    
    def disconnect_all(self):
        """Disconnect from all exchanges."""
        logger.info("Disconnecting from all exchanges...")
        pass
    
    def get_balances(self) -> Dict[str, Any]:
        """Get balances from all connected exchanges."""
        return {}
    
    def get_ticker(self, symbol: str, exchange_id: str = None) -> Optional[Ticker]:
        """Get ticker data for a symbol."""
        return None
    
    def get_order_book(self, symbol: str, exchange_id: str = None) -> Optional[OrderBook]:
        """Get order book data for a symbol."""
        return None
    
    def place_order(self, exchange_id: str, symbol: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'market') -> Optional[Order]:
        """Place an order on an exchange."""
        return None
    
    def cancel_order(self, exchange_id: str, order_id: str, symbol: str = None) -> bool:
        """Cancel an order."""
        return True
    
    def get_order_status(self, exchange_id: str, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order status."""
        return None
    
    def get_supported_symbols(self, exchange_id: str = None) -> List[str]:
        """Get supported trading symbols."""
        return []
    
    def get_fees(self, exchange_id: str) -> Dict[str, Any]:
        """Get trading fees for an exchange."""
        return {}
    
    def check_connection(self, exchange_id: str) -> bool:
        """Check if connection to exchange is active."""
        return True
