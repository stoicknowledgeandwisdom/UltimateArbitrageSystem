#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Data Manager
====================

Centralized data management system with:
- Multi-source data aggregation
- Real-time data fusion
- Quality-based source selection
- Intelligent caching strategies
- Failover and redundancy
- Performance optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .base_connector import (
    BaseDataConnector, MarketDataPoint, DataQuality, 
    MarketDataType, DataConnectorStats
)
from .yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration and status."""
    name: str
    connector: BaseDataConnector
    priority: int = 1  # Higher = better
    is_active: bool = True
    reliability_score: float = 1.0
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

@dataclass
class DataRequest:
    """Data request specification."""
    symbol: str
    data_type: str = 'realtime'  # realtime, historical, stream
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = '1d'
    max_age_seconds: int = 300  # Maximum acceptable data age
    min_quality: DataQuality = DataQuality.FAIR
    preferred_sources: List[str] = field(default_factory=list)
    fallback_allowed: bool = True

@dataclass
class AggregatedDataPoint:
    """Aggregated data from multiple sources."""
    primary_data: MarketDataPoint
    source_data: Dict[str, MarketDataPoint] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_consensus: float = 1.0
    aggregation_timestamp: datetime = field(default_factory=datetime.now)
    
class AdvancedDataManager:
    """Advanced multi-source data management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources: Dict[str, DataSource] = {}
        self.cache = {}  # In-memory cache
        self.redis_client = None
        self.performance_tracker = defaultdict(deque)
        self.health_check_interval = config.get('health_check_interval', 300)  # 5 minutes
        self.consensus_threshold = config.get('consensus_threshold', 0.8)
        self.max_source_latency = config.get('max_source_latency', 5000)  # ms
        
        # Thread safety
        self.lock = threading.RLock()
        self.is_running = False
        
        # Performance metrics
        self.request_count = 0
        self.cache_hits = 0
        self.source_failures = defaultdict(int)
        
        # Initialize Redis if available
        self._setup_redis()
        
        # Initialize data sources
        self._initialize_sources()
        
        logger.info("AdvancedDataManager initialized")
    
    def _setup_redis(self):
        """Setup Redis connection for distributed caching."""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available - using in-memory cache only")
            return
        
        redis_config = self.config.get('redis', {})
        if redis_config.get('enabled', False):
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.redis_client = None
    
    def _initialize_sources(self):
        """Initialize data sources based on configuration."""
        sources_config = self.config.get('sources', {})
        
        # Yahoo Finance (always available)
        if sources_config.get('yahoo_finance', {}).get('enabled', True):
            yahoo_config = sources_config.get('yahoo_finance', {})
            yahoo_connector = YahooFinanceConnector(yahoo_config)
            self.add_source('yahoo_finance', yahoo_connector, priority=2)
        
        # Additional sources would be added here
        # self.add_source('alpha_vantage', AlphaVantageConnector(av_config), priority=3)
        # self.add_source('polygon', PolygonConnector(poly_config), priority=4)
        
        logger.info(f"Initialized {len(self.sources)} data sources")
    
    def add_source(self, name: str, connector: BaseDataConnector, priority: int = 1):
        """Add a data source to the manager."""
        with self.lock:
            source = DataSource(
                name=name,
                connector=connector,
                priority=priority
            )
            self.sources[name] = source
            logger.info(f"Added data source: {name} (priority: {priority})")
    
    def remove_source(self, name: str):
        """Remove a data source from the manager."""
        with self.lock:
            if name in self.sources:
                del self.sources[name]
                logger.info(f"Removed data source: {name}")
    
    async def start(self):
        """Start the data manager and all sources."""
        self.is_running = True
        
        # Connect all sources
        connection_tasks = []
        for source in self.sources.values():
            connection_tasks.append(source.connector.connect())
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Update source status based on connection results
        for i, (source_name, source) in enumerate(self.sources.items()):
            if isinstance(results[i], bool) and results[i]:
                source.is_active = True
                logger.info(f"Successfully connected to {source_name}")
            else:
                source.is_active = False
                logger.warning(f"Failed to connect to {source_name}: {results[i]}")
        
        # Start health check task
        asyncio.create_task(self._health_check_loop())
        
        logger.info("AdvancedDataManager started")
    
    async def stop(self):
        """Stop the data manager and all sources."""
        self.is_running = False
        
        # Disconnect all sources
        disconnect_tasks = []
        for source in self.sources.values():
            disconnect_tasks.append(source.connector.disconnect())
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info("AdvancedDataManager stopped")
    
    async def get_data(self, request: DataRequest) -> Optional[AggregatedDataPoint]:
        """Get data with intelligent source selection and aggregation."""
        self.request_count += 1
        
        # Check cache first
        cached_data = await self._get_cached_data(request)
        if cached_data:
            self.cache_hits += 1
            return cached_data
        
        # Get active sources sorted by reliability and priority
        available_sources = self._get_available_sources(request)
        
        if not available_sources:
            logger.warning(f"No available sources for request: {request.symbol}")
            return None
        
        # Try to get data from multiple sources for consensus
        data_results = await self._fetch_from_sources(request, available_sources)
        
        if not data_results:
            logger.warning(f"No data available from any source for {request.symbol}")
            return None
        
        # Aggregate and validate data
        aggregated = await self._aggregate_data(data_results, request)
        
        # Cache the result
        if aggregated:
            await self._cache_data(request, aggregated)
        
        return aggregated
    
    async def get_real_time_data(self, symbol: str, **kwargs) -> Optional[AggregatedDataPoint]:
        """Convenience method for real-time data."""
        request = DataRequest(symbol=symbol, data_type='realtime', **kwargs)
        return await self.get_data(request)
    
    async def get_historical_data(self, 
                                symbol: str, 
                                start_date: datetime, 
                                end_date: datetime,
                                interval: str = '1d',
                                **kwargs) -> List[MarketDataPoint]:
        """Get historical data with source aggregation."""
        request = DataRequest(
            symbol=symbol,
            data_type='historical',
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            **kwargs
        )
        
        # For historical data, we typically use the best available source
        # rather than aggregating (due to volume of data)
        available_sources = self._get_available_sources(request)
        
        for source in available_sources:
            try:
                data = await source.connector.get_historical_data(
                    symbol, start_date, end_date, interval
                )
                if data:
                    source.success_count += 1
                    logger.info(f"Retrieved {len(data)} historical points from {source.name}")
                    return data
            except Exception as e:
                source.failure_count += 1
                self.source_failures[source.name] += 1
                logger.warning(f"Failed to get historical data from {source.name}: {e}")
                continue
        
        return []
    
    async def stream_real_time_data(self, symbols: List[str]) -> AsyncGenerator[AggregatedDataPoint, None]:
        """Stream real-time data with automatic failover."""
        if not symbols:
            return
        
        logger.info(f"Starting real-time stream for {len(symbols)} symbols")
        
        # Select best streaming source
        streaming_sources = [s for s in self.sources.values() 
                           if s.is_active and hasattr(s.connector, 'stream_real_time_data')]
        
        if not streaming_sources:
            logger.error("No streaming sources available")
            return
        
        # Use highest priority source for streaming
        primary_source = max(streaming_sources, key=lambda s: s.priority)
        
        try:
            async for data_point in primary_source.connector.stream_real_time_data(symbols):
                # Wrap in aggregated format
                aggregated = AggregatedDataPoint(
                    primary_data=data_point,
                    source_data={primary_source.name: data_point},
                    confidence_score=0.8,  # Single source confidence
                    source_consensus=1.0
                )
                yield aggregated
                
        except Exception as e:
            logger.error(f"Streaming failed from {primary_source.name}: {e}")
            primary_source.failure_count += 1
    
    def _get_available_sources(self, request: DataRequest) -> List[DataSource]:
        """Get available sources sorted by suitability for the request."""
        available = []
        
        for source in self.sources.values():
            if not source.is_active:
                continue
            
            # Check if source supports the requested data type
            if request.data_type == 'historical' and not hasattr(source.connector, 'get_historical_data'):
                continue
            
            # Check preferred sources
            if request.preferred_sources and source.name not in request.preferred_sources:
                continue
            
            available.append(source)
        
        # Sort by composite score (priority * reliability)
        available.sort(
            key=lambda s: s.priority * s.reliability_score * s.success_rate,
            reverse=True
        )
        
        return available
    
    async def _fetch_from_sources(self, 
                                request: DataRequest, 
                                sources: List[DataSource]) -> Dict[str, MarketDataPoint]:
        """Fetch data from multiple sources concurrently."""
        tasks = []
        source_names = []
        
        # Limit concurrent requests to avoid overwhelming APIs
        max_concurrent = min(len(sources), 3)
        
        for source in sources[:max_concurrent]:
            if request.data_type == 'realtime':
                task = source.connector.get_real_time_data(request.symbol)
            else:
                continue  # Other types handled separately
            
            tasks.append(task)
            source_names.append(source.name)
        
        if not tasks:
            return {}
        
        # Execute requests with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.get('fetch_timeout', 10)
            )
        except asyncio.TimeoutError:
            logger.warning("Data fetch timeout")
            return {}
        
        # Process results
        data_results = {}
        
        for i, (source_name, result) in enumerate(zip(source_names, results)):
            source = self.sources[source_name]
            
            if isinstance(result, MarketDataPoint):
                # Check data quality and age
                if (result.quality.value >= request.min_quality.value and
                    self._is_data_fresh(result, request.max_age_seconds)):
                    
                    data_results[source_name] = result
                    source.success_count += 1
                else:
                    logger.debug(f"Data from {source_name} rejected due to quality/age")
                    source.failure_count += 1
            else:
                logger.warning(f"Failed to get data from {source_name}: {result}")
                source.failure_count += 1
                self.source_failures[source_name] += 1
        
        return data_results
    
    async def _aggregate_data(self, 
                            data_results: Dict[str, MarketDataPoint], 
                            request: DataRequest) -> Optional[AggregatedDataPoint]:
        """Aggregate data from multiple sources."""
        if not data_results:
            return None
        
        # If only one source, use it directly
        if len(data_results) == 1:
            source_name, data_point = next(iter(data_results.items()))
            return AggregatedDataPoint(
                primary_data=data_point,
                source_data=data_results,
                confidence_score=0.8,  # Single source confidence
                source_consensus=1.0
            )
        
        # Multi-source aggregation
        primary_source = self._select_primary_source(data_results)
        primary_data = data_results[primary_source]
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(data_results)
        
        # Calculate confidence based on consensus and source quality
        confidence_score = min(1.0, (
            consensus_score * 0.6 +
            primary_data.quality.value / 5.0 * 0.4
        ))
        
        return AggregatedDataPoint(
            primary_data=primary_data,
            source_data=data_results,
            confidence_score=confidence_score,
            source_consensus=consensus_score
        )
    
    def _select_primary_source(self, data_results: Dict[str, MarketDataPoint]) -> str:
        """Select the primary source based on quality and source reliability."""
        scores = {}
        
        for source_name, data_point in data_results.items():
            source = self.sources[source_name]
            
            # Composite score: source priority + data quality + reliability
            score = (
                source.priority * 0.4 +
                data_point.quality.value * 0.3 +
                source.reliability_score * 0.3
            )
            
            scores[source_name] = score
        
        return max(scores, key=scores.get)
    
    def _calculate_consensus(self, data_results: Dict[str, MarketDataPoint]) -> float:
        """Calculate consensus score based on price agreement."""
        if len(data_results) < 2:
            return 1.0
        
        prices = [float(dp.price) for dp in data_results.values() if dp.price]
        
        if len(prices) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_price = statistics.mean(prices)
        if mean_price == 0:
            return 0.0
        
        std_price = statistics.stdev(prices)
        cv = std_price / mean_price
        
        # Convert to consensus score (lower variation = higher consensus)
        consensus = max(0.0, 1.0 - cv * 10)  # Scale factor
        
        return min(1.0, consensus)
    
    def _is_data_fresh(self, data_point: MarketDataPoint, max_age_seconds: int) -> bool:
        """Check if data is fresh enough."""
        if not data_point.timestamp:
            return False
        
        age = (datetime.now() - data_point.timestamp).total_seconds()
        return age <= max_age_seconds
    
    async def _get_cached_data(self, request: DataRequest) -> Optional[AggregatedDataPoint]:
        """Get data from cache if available and fresh."""
        cache_key = self._generate_cache_key(request)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_json = self.redis_client.get(cache_key)
                if cached_json:
                    # In a full implementation, deserialize the JSON
                    # For now, skip Redis caching of complex objects
                    pass
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Try memory cache
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            
            # Check if cache entry is still fresh
            cache_age = (datetime.now() - cached_entry['timestamp']).total_seconds()
            if cache_age <= request.max_age_seconds:
                return cached_entry['data']
            else:
                # Remove stale entry
                del self.cache[cache_key]
        
        return None
    
    async def _cache_data(self, request: DataRequest, data: AggregatedDataPoint):
        """Cache data for future requests."""
        cache_key = self._generate_cache_key(request)
        
        # Memory cache
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Redis cache (if available)
        if self.redis_client:
            try:
                # In a full implementation, serialize to JSON
                # For now, skip Redis caching of complex objects
                pass
            except Exception as e:
                logger.warning(f"Failed to cache to Redis: {e}")
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for a request."""
        return f"dm:{request.symbol}:{request.data_type}:{request.interval}"
    
    async def _health_check_loop(self):
        """Periodic health check for all sources."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Brief pause on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all sources."""
        health_tasks = []
        
        for source in self.sources.values():
            health_tasks.append(source.connector.health_check())
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for i, (source_name, source) in enumerate(self.sources.items()):
            result = results[i]
            source.last_health_check = datetime.now()
            
            if isinstance(result, dict) and result.get('status') == 'healthy':
                if not source.is_active:
                    source.is_active = True
                    logger.info(f"Source {source_name} is back online")
                
                # Update reliability score based on performance
                latency = result.get('latency_ms', 0)
                if latency < self.max_source_latency:
                    source.reliability_score = min(1.0, source.reliability_score + 0.01)
                else:
                    source.reliability_score = max(0.1, source.reliability_score - 0.05)
                    
            else:
                if source.is_active:
                    source.is_active = False
                    logger.warning(f"Source {source_name} failed health check: {result}")
                
                source.reliability_score = max(0.1, source.reliability_score - 0.1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0
        
        source_stats = {}
        for name, source in self.sources.items():
            source_stats[name] = {
                'is_active': source.is_active,
                'priority': source.priority,
                'reliability_score': source.reliability_score,
                'success_rate': source.success_rate,
                'success_count': source.success_count,
                'failure_count': source.failure_count,
                'last_health_check': source.last_health_check.isoformat() if source.last_health_check else None
            }
        
        return {
            'total_requests': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'active_sources': len([s for s in self.sources.values() if s.is_active]),
            'total_sources': len(self.sources),
            'source_failures': dict(self.source_failures),
            'sources': source_stats
        }

