#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Data Connector
==================

Abstract base class for all market data connectors with:
- Standardized data formats
- Quality assessment framework
- Real-time and historical data support
- Error handling and retry mechanisms
- Performance monitoring
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import numpy as np
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality assessment levels."""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    UNUSABLE = 1

class MarketDataType(Enum):
    """Types of market data."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"
    OPTION = "option"
    FUTURE = "future"

@dataclass
class MarketDataPoint:
    """Standardized market data point."""
    symbol: str
    timestamp: datetime
    data_type: MarketDataType
    
    # Price data
    price: Optional[Decimal] = None
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    
    # Volume and trading data
    volume: Optional[int] = None
    trade_count: Optional[int] = None
    vwap: Optional[Decimal] = None
    
    # Market data
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    spread: Optional[Decimal] = None
    
    # Derived metrics
    returns: Optional[float] = None
    volatility: Optional[float] = None
    rsi: Optional[float] = None
    moving_avg_20: Optional[Decimal] = None
    moving_avg_50: Optional[Decimal] = None
    
    # Metadata
    source: str = "unknown"
    quality: DataQuality = DataQuality.FAIR
    latency_ms: Optional[float] = None
    exchange: Optional[str] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type.value,
            'price': float(self.price) if self.price else None,
            'open_price': float(self.open_price) if self.open_price else None,
            'high_price': float(self.high_price) if self.high_price else None,
            'low_price': float(self.low_price) if self.low_price else None,
            'close_price': float(self.close_price) if self.close_price else None,
            'volume': self.volume,
            'trade_count': self.trade_count,
            'vwap': float(self.vwap) if self.vwap else None,
            'bid_price': float(self.bid_price) if self.bid_price else None,
            'ask_price': float(self.ask_price) if self.ask_price else None,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'spread': float(self.spread) if self.spread else None,
            'returns': self.returns,
            'volatility': self.volatility,
            'rsi': self.rsi,
            'moving_avg_20': float(self.moving_avg_20) if self.moving_avg_20 else None,
            'moving_avg_50': float(self.moving_avg_50) if self.moving_avg_50 else None,
            'source': self.source,
            'quality': self.quality.value,
            'latency_ms': self.latency_ms,
            'exchange': self.exchange,
            'metadata': self.metadata
        }

@dataclass
class DataConnectorStats:
    """Performance statistics for data connectors."""
    connector_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    uptime_percentage: float = 100.0
    data_quality_avg: float = 4.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

class BaseDataConnector(ABC):
    """Abstract base class for all market data connectors."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.stats = DataConnectorStats(connector_name=name)
        self.rate_limiter = None
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes default
        self._setup_rate_limiter()
        
    def _setup_rate_limiter(self):
        """Setup rate limiting based on API limits."""
        requests_per_minute = self.config.get('requests_per_minute', 60)
        self.request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _cache_key(self, symbol: str, data_type: str, period: str = 'current') -> str:
        """Generate cache key for data."""
        return f"{self.name}:{symbol}:{data_type}:{period}"
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = cache_entry['timestamp']
        current_time = time.time()
        
        return (current_time - cache_time) < self.cache_ttl
    
    def _assess_data_quality(self, data_point: MarketDataPoint) -> DataQuality:
        """Assess the quality of a data point."""
        quality_score = 5.0  # Start with excellent
        
        # Check for missing critical data
        if data_point.price is None and data_point.close_price is None:
            quality_score -= 2.0
        
        # Check timestamp freshness
        if data_point.timestamp:
            age_minutes = (datetime.now() - data_point.timestamp).total_seconds() / 60
            if age_minutes > 60:  # More than 1 hour old
                quality_score -= 1.0
            elif age_minutes > 15:  # More than 15 minutes old
                quality_score -= 0.5
        
        # Check for reasonable price values
        if data_point.price and (data_point.price <= 0 or data_point.price > Decimal('1000000')):
            quality_score -= 1.0
        
        # Check latency
        if data_point.latency_ms and data_point.latency_ms > 5000:  # > 5 seconds
            quality_score -= 0.5
        
        # Convert to enum
        quality_score = max(1.0, min(5.0, quality_score))
        return DataQuality(int(quality_score))
    
    def _update_stats(self, success: bool, latency_ms: float, quality: DataQuality):
        """Update connector performance statistics."""
        self.stats.total_requests += 1
        self.stats.last_request_time = datetime.now()
        
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # Update average latency
        if self.stats.total_requests == 1:
            self.stats.average_latency_ms = latency_ms
        else:
            self.stats.average_latency_ms = (
                (self.stats.average_latency_ms * (self.stats.total_requests - 1) + latency_ms) /
                self.stats.total_requests
            )
        
        # Update average quality
        if success:
            if self.stats.successful_requests == 1:
                self.stats.data_quality_avg = quality.value
            else:
                self.stats.data_quality_avg = (
                    (self.stats.data_quality_avg * (self.stats.successful_requests - 1) + quality.value) /
                    self.stats.successful_requests
                )
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time data for a symbol."""
        pass
    
    @abstractmethod
    async def get_historical_data(self, 
                                symbol: str, 
                                start_date: datetime, 
                                end_date: datetime,
                                interval: str = '1d') -> List[MarketDataPoint]:
        """Get historical data for a symbol."""
        pass
    
    @abstractmethod
    async def stream_real_time_data(self, symbols: List[str]) -> AsyncGenerator[MarketDataPoint, None]:
        """Stream real-time data for multiple symbols."""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        pass
    
    @abstractmethod
    def get_supported_data_types(self) -> List[MarketDataType]:
        """Get list of supported data types."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the connector."""
        try:
            start_time = time.time()
            
            # Try to get data for a test symbol
            test_symbols = ['SPY', 'AAPL', 'BTC-USD']  # Common symbols
            test_result = None
            
            for symbol in test_symbols:
                try:
                    test_result = await self.get_real_time_data(symbol)
                    if test_result:
                        break
                except Exception:
                    continue
            
            latency = (time.time() - start_time) * 1000
            
            return {
                'connector': self.name,
                'status': 'healthy' if test_result else 'degraded',
                'connected': self.is_connected,
                'latency_ms': latency,
                'last_data': test_result.to_dict() if test_result else None,
                'stats': {
                    'success_rate': self.stats.success_rate,
                    'average_latency': self.stats.average_latency_ms,
                    'total_requests': self.stats.total_requests,
                    'uptime': self.stats.uptime_percentage
                }
            }
        except Exception as e:
            return {
                'connector': self.name,
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }
    
    def get_stats(self) -> DataConnectorStats:
        """Get connector performance statistics."""
        return self.stats

