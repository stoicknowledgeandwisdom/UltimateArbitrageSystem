#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UltimateArbitrageSystem: Market Data Provider
==============================================

Quantum-enhanced, ultra-fast system for retrieving, caching, and processing market data
from multiple exchanges with advanced multi-dimensional analysis capabilities.

Features:
- Hyper-efficient multi-layered caching with predictive prefetching
- Real-time streaming data with microsecond precision
- Advanced technical indicators and statistical arbitrage models
- Cross-exchange correlation and cointegration analysis
- Machine learning-based opportunity detection
- Order book imbalance analysis
- Market microstructure modeling
- Ultra-low latency websocket connections
- Self-optimizing data refresh rates
- Anomaly detection for market inefficiencies
- Multithreaded and distributed processing capabilities
"""

import logging
import time
import threading
import asyncio
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from datetime import datetime, timedelta
import requests
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import zlib
import gzip
import lz4.frame
from collections import deque, defaultdict, OrderedDict
import hashlib
import tempfile
import mmap
import shutil
import heapq
import uuid
import redis
import joblib
import warnings
import traceback
import statistics
import math
from functools import lru_cache, partial
from itertools import combinations, product

# Scientific and ML libraries
from scipy import stats, signal
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Technical indicators
import talib
from ta import momentum, trend, volatility, volume

# Visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("MarketDataProvider")

class HyperCache:
    """Advanced multi-layered caching system with intelligent prefetching and compression."""
    
    COMPRESSION_NONE = 0
    COMPRESSION_ZLIB = 1
    COMPRESSION_GZIP = 2
    COMPRESSION_LZ4 = 3
    
    def __init__(self, cache_dir=None, redis_url=None, memory_limit=1024*1024*1024, 
                 ttl_settings=None, compression_level=COMPRESSION_LZ4):
        """
        Initialize the hyperefficient caching system.
        
        Args:
            cache_dir: Directory for persistent cache storage
            redis_url: Optional Redis URL for distributed caching
            memory_limit: Memory limit for in-memory cache (default: 1GB)
            ttl_settings: TTL for different data types
            compression_level: Compression algorithm to use
        """
        # Cache storage locations
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "market_data_cache")
        self.memory_limit = memory_limit
        self.compression_level = compression_level
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default TTL settings for different data types (in seconds)
        self.ttl_settings = ttl_settings or {
            "ticker": 5,            # Very short lived
            "orderbook": 5,         # Very short lived
            "trades": 300,          # 5 minutes
            "historical_1m": 600,   # 10 minutes
            "historical_5m": 1800,  # 30 minutes
            "historical_15m": 3600, # 1 hour
            "historical_1h": 7200,  # 2 hours
            "historical_4h": 14400, # 4 hours
            "historical_1d": 86400, # 1 day
            "indicators": 1800,     # 30 minutes
            "metadata": 86400       # 1 day
        }
        
        # Memory cache for ultra-fast access
        self.memory_cache = OrderedDict()
        self.memory_usage = 0
        
        # Access statistics for smart eviction
        self.access_frequency = defaultdict(int)
        self.access_recency = {}
        self.predictive_score = {}
        
        # Redis connection for distributed cache
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis cache at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to local cache.")
        
        # Mutex for thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": defaultdict(int),
            "misses": defaultdict(int),
            "evictions": defaultdict(int),
            "total_bytes_cached": 0,
            "total_bytes_saved": 0
        }
        
        # Prefetch queue
        self.prefetch_queue = []
        self.prefetch_lock = threading.RLock()
        
        # Background prefetch thread
        self.prefetch_thread = None
        self.running = False
        
        logger.info(f"HyperCache initialized with {self.memory_limit/1024/1024:.1f}MB memory limit")
    
    def start(self):
        """Start background operations like prefetching."""
        if self.running:
            return
            
        self.running = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.debug("Cache prefetch worker started")
    
    def stop(self):
        """Stop background operations."""
        self.running = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)
            self.prefetch_thread = None
    
    def _compress_data(self, data):
        """Compress data using the configured compression algorithm."""
        try:
            serialized = pickle.dumps(data)
            
            if self.compression_level == self.COMPRESSION_NONE:
                return serialized, len(serialized)
            elif self.compression_level == self.COMPRESSION_ZLIB:
                compressed = zlib.compress(serialized)
                return compressed, len(serialized)
            elif self.compression_level == self.COMPRESSION_GZIP:
                compressed = gzip.compress(serialized)
                return compressed, len(serialized)
            elif self.compression_level == self.COMPRESSION_LZ4:
                compressed = lz4.frame.compress(serialized)
                return compressed, len(serialized)
            else:
                return serialized, len(serialized)
        except Exception as e:
            logger.warning(f"Compression error: {e}. Falling back to uncompressed.")
            return pickle.dumps(data), 0
    
    def _decompress_data(self, data, compression_level):
        """Decompress data using the specified compression algorithm."""
        try:
            if compression_level == self.COMPRESSION_NONE:
                return pickle.loads(data)
            elif compression_level == self.COMPRESSION_ZLIB:
                return pickle.loads(zlib.decompress(data))
            elif compression_level == self.COMPRESSION_GZIP:
                return pickle.loads(gzip.decompress(data))
            elif compression_level == self.COMPRESSION_LZ4:
                return pickle.loads(lz4.frame.decompress(data))
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Decompression error: {e}")
            return None
    
    def _generate_key(self, data_type, exchange_id, symbol, timeframe=None, extra=None):
        """Generate a unique cache key for the data."""
        key_parts = [data_type, exchange_id, symbol]
        
        if timeframe:
            key_parts.append(timeframe)
        
        if extra:
            if isinstance(extra, dict):
                # Sort dict items for consistent keys
                key_parts.append(json.dumps(extra, sort_keys=True))
            else:
                key_parts.append(str(extra))
        
        key = "_".join(key_parts)
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _get_ttl(self, data_type, timeframe=None):
        """Get the appropriate TTL for a data type."""
        if timeframe and f"{data_type}_{timeframe}" in self.ttl_settings:
            return self.ttl_settings[f"{data_type}_{timeframe}"]
        return self.ttl_settings.get(data_type, 300)  # Default: 5 minutes
    
    def get(self, data_type, exchange_id, symbol, timeframe=None, extra=None, default=None):
        """
        Retrieve data from cache using multi-tiered approach.
        
        Args:
            data_type: Type of data (ticker, orderbook, etc.)
            exchange_id: Exchange identifier
            symbol: Trading symbol
            timeframe: Optional timeframe for historical data
            extra: Additional parameters to include in the cache key
            default: Default value if not found
            
        Returns:
            The cached data or default if not found
        """
        key = self._generate_key(data_type, exchange_id, symbol, timeframe, extra)
        ttl = self._get_ttl(data_type, timeframe)
        
        # Update access metrics for smart eviction
        with self.lock:
            self.access_frequency[key] += 1
            self.access_recency[key] = time.time()
            
            # Check memory cache first (Level 1)
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                current_time = time.time()
                
                # Check if the entry is still valid
                if current_time - entry['timestamp'] <= ttl:
                    # Move to the end of OrderedDict to mark as recently used
                    self.memory_cache.move_to_end(key)
                    
                    # Update statistics
                    self.stats["hits"]["memory"] += 1
                    
                    # Update predictive score
                    self._update_predictive_score(key, hit=True)
                    
                    return entry['data']
                else:
                    # Expired entry - remove from memory cache
                    self._remove_from_memory_cache(key)
        
        # Check Redis cache (Level 2)
        if self.redis_client:
            try:
                redis_key = f"market_data:{key}"
                cached_data = self.redis_client.get(redis_key)
                
                if cached_data:
                    # Data found in Redis
                    try:
                        cache_entry = json.loads(cached_data)
                        
                        if time.time() - cache_entry['timestamp'] <= ttl:
                            # Valid data - decompress and deserialize
                            data = self._decompress_data(
                                cache_entry['data'], 
                                cache_entry.get('compression', self.COMPRESSION_NONE)
                            )
                            
                            # Store in memory cache for faster future access
                            self._store_in_memory(key, data)
                            
                            # Update statistics
                            self.stats["hits"]["redis"] += 1
                            
                            # Update predictive score
                            self._update_predictive_score(key, hit=True)
                            
                            return data
                    except Exception as e:
                        logger.warning(f"Redis data parsing error: {e}")
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Check disk cache (Level 3)
        file_path = os.path.join(self.cache_dir, f"{key}.cache")
        if os.path.exists(file_path):
            try:
                file_time = os.path.getmtime(file_path)
                
                # Check if file is within TTL
                if time.time() - file_time <= ttl:
                    with open(file_path, 'rb') as f:
                        # Read the header to get compression info
                        header = f.read(8)
                        compression = int.from_bytes(header[:4], byteorder='little')
                        original_size = int.from_bytes(header[4:8], byteorder='little')
                        
                        # Read the actual data
                        data_bytes = f.read()
                        
                        # Decompress and deserialize
                        data = self._decompress_data(data_bytes, compression)
                        
                        if data is not None:
                            # Store in memory cache for faster future access
                            self._store_in_memory(key, data)
                            
                            # Update statistics
                            self.stats["hits"]["disk"] += 1
                            
                            # Update predictive score for future prefetching
                            self._update_predictive_score(key, hit=True)
                            
                            return data
            except Exception as e:
                logger.warning(f"Disk cache error for {key}: {e}")
        
        # Data not found in any cache level
        with self.lock:
            self.stats["misses"][data_type] += 1
            self._update_predictive_score(key, hit=False)
        
        return default
    
    def set(self, data_type, exchange_id, symbol, data, timeframe=None, extra=None):
        """
        Store data in all cache layers.
        
        Args:
            data_type: Type of data (ticker, orderbook, etc.)
            exchange_id: Exchange identifier
            symbol: Trading symbol
            data: Data to cache
            timeframe: Optional timeframe for historical data
            extra: Additional parameters to include in the cache key
            
        Returns:
            bool: Success status
        """
        if data is None:
            return False
            
        key = self._generate_key(data_type, exchange_id, symbol, timeframe, extra)
        ttl = self._get_ttl(data_type, timeframe)
        timestamp = time.time()
        
        # Compress the data
        compressed_data, original_size = self._compress_data(data)
        
        # Store in memory cache (Level 1)
        with self.lock:
            self._store_in_memory(key, data)
            self.stats["total_bytes_cached"] += original_size
            if original_size > 0:
                self.stats["total_bytes_saved"] += (original_size - len(compressed_data))
        
        # Store in Redis if available (Level 2)
        # Store in Redis if available (Level 2)
        if self.redis_client:
            try:
                redis_key = f"market_data:{key}"
                cache_entry = {
                    'timestamp': timestamp,
                    'compression': self.compression_level,
                    'data': compressed_data
                }
                # Store with TTL (add 10% buffer to TTL)
                self.redis_client.setex(
                    redis_key,
                    int(ttl * 1.1),
                    json.dumps(cache_entry)
                )
            except Exception as e:
                logger.warning(f"Redis cache storage error: {e}")
        
        # Store in disk cache (Level 3)
        try:
            file_path = os.path.join(self.cache_dir, f"{key}.cache")
            with open(file_path, 'wb') as f:
                # Write a header with compression info and original size
                f.write(self.compression_level.to_bytes(4, byteorder='little'))
                f.write(original_size.to_bytes(4, byteorder='little'))
                
                # Write the compressed data
                f.write(compressed_data)
        except Exception as e:
            logger.warning(f"Disk cache storage error for {key}: {e}")
            return False
            
        return True
    
    def invalidate(self, data_type, exchange_id, symbol, timeframe=None, extra=None):
        """Invalidate cached data."""
        key = self._generate_key(data_type, exchange_id, symbol, timeframe, extra)
        
        with self.lock:
            # Remove from memory cache
            if key in self.memory_cache:
                self._remove_from_memory_cache(key)
        
        # Remove from Redis
        if self.redis_client:
            try:
                redis_key = f"market_data:{key}"
                self.redis_client.delete(redis_key)
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
        
        # Remove from disk
        try:
            file_path = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Disk cache invalidation error: {e}")
    
    def _store_in_memory(self, key, data):
        """Store data in memory cache with eviction policy."""
        # Estimate the data size
        data_size = len(pickle.dumps(data))
        
        # Check if we need to evict entries to make space
        if self.memory_usage + data_size > self.memory_limit:
            self._evict_entries(data_size)
        
        # Store the data
        self.memory_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'size': data_size
        }
        self.memory_usage += data_size
        
        # Move to the end of OrderedDict to mark as recently used
        self.memory_cache.move_to_end(key)
    
    def _remove_from_memory_cache(self, key):
        """Remove an item from the memory cache."""
        if key in self.memory_cache:
            data_size = self.memory_cache[key]['size']
            del self.memory_cache[key]
            self.memory_usage -= data_size
            self.stats["evictions"]["memory"] += 1
    
    def _evict_entries(self, required_space):
        """Evict entries to free up required space using smart eviction policy."""
        # Calculate predictive scores for all entries
        for key in list(self.memory_cache.keys()):
            self._update_predictive_score(key, update_only=True)
        
        # Sort keys by predictive score (lower is more evictable)
        eviction_candidates = sorted(
            self.memory_cache.keys(),
            key=lambda k: self.predictive_score.get(k, 0)
        )
        
        # Evict entries until we free up enough space
        space_freed = 0
        for key in eviction_candidates:
            if key in self.memory_cache:  # May have been removed in previous iteration
                data_size = self.memory_cache[key]['size']
                space_freed += data_size
                self._remove_from_memory_cache(key)
                logger.debug(f"Evicted cache entry {key}, freed {data_size} bytes")
                
                if space_freed >= required_space:
                    break
    
    def _update_predictive_score(self, key, hit=None, update_only=False):
        """Update the predictive score for an entry to guide eviction decisions."""
        # Factors to consider:
        # 1. Access frequency (higher is better)
        # 2. Access recency (more recent is better)
        # 3. Hit/miss pattern (more hits is better)
        
        if not update_only and hit is not None:
            # Record hit/miss
            if key not in self.predictive_score:
                self.predictive_score[key] = 0.5  # Initialize
                
            # Update score with exponential moving average
            current_score = self.predictive_score[key]
            hit_value = 1.0 if hit else 0.0
            self.predictive_score[key] = 0.8 * current_score + 0.2 * hit_value
        
        # Always update based on recency and frequency
        frequency_factor = min(1.0, math.log1p(self.access_frequency.get(key, 0)) / 10)
        
        recency_factor = 0.0
        if key in self.access_recency:
            # Normalize recency to a 0-1 scale where 1 is most recent
            age_seconds = time.time() - self.access_recency[key]
            recency_factor = max(0.0, min(1.0, 1.0 - (age_seconds / 3600)))  # 1-hour scale
        
        # Final score combines all factors
        final_score = (
            0.4 * self.predictive_score.get(key, 0.0) +  # Hit/miss history
            0.3 * frequency_factor +                     # Access frequency
            0.3 * recency_factor                         # Recency
        )
        
        self.predictive_score[key] = final_score
    
    def _prefetch_worker(self):
        """Background worker for predictive prefetching."""
        while self.running:
            try:
                # Sleep to avoid CPU hogging
                time.sleep(0.1)
                
                # Process prefetch queue
                with self.prefetch_lock:
                    if not self.prefetch_queue:
                        continue
                        
                    # Get the highest priority item
                    priority, prefetch_fn = heapq.heappop(self.prefetch_queue)
                    
                # Execute the prefetch function
                prefetch_fn()
                    
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                
    def prefetch(self, data_type, exchange_id, symbol, timeframe=None, extra=None, priority=0):
        """
        Queue a prefetch operation with the given priority.
        
        Args:
            data_type: Type of data to prefetch
            exchange_id: Exchange identifier
            symbol: Trading symbol
            timeframe: Optional timeframe
            extra: Additional key parameters
            priority: Priority (lower number = higher priority)
        """
        # Create a prefetch function to be executed by the worker
        def _prefetch_fn():
            try:
                # Generate the key
                key = self._generate_key(data_type, exchange_id, symbol, timeframe, extra)
                
                # Check if already in memory cache
                with self.lock:
                    if key in self.memory_cache:
                        return
                
                # Try Redis and disk in sequence
                if self.redis_client:
                    try:
                        redis_key = f"market_data:{key}"
                        cached_data = self.redis_client.get(redis_key)
                        
                        if cached_data:
                            try:
                                cache_entry = json.loads(cached_data)
                                ttl = self._get_ttl(data_type, timeframe)
                                
                                if time.time() - cache_entry['timestamp'] <= ttl:
                                    # Valid data - decompress and store in memory
                                    data = self._decompress_data(
                                        cache_entry['data'], 
                                        cache_entry.get('compression', self.COMPRESSION_NONE)
                                    )
                                    
                                    if data is not None:
                                        # Store in memory cache for future access
                                        with self.lock:
                                            self._store_in_memory(key, data)
                                            logger.debug(f"Prefetched {key} from Redis")
                                            return
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # Try disk cache
                file_path = os.path.join(self.cache_dir, f"{key}.cache")
                if os.path.exists(file_path):
                    try:
                        file_time = os.path.getmtime(file_path)
                        ttl = self._get_ttl(data_type, timeframe)
                        
                        # Check if file is within TTL
                        if time.time() - file_time <= ttl:
                            with open(file_path, 'rb') as f:
                                # Read the header
                                header = f.read(8)
                                compression = int.from_bytes(header[:4], byteorder='little')
                                
                                # Read the data
                                data_bytes = f.read()
                                
                                # Decompress and store in memory
                                data = self._decompress_data(data_bytes, compression)
                                
                                if data is not None:
                                    with self.lock:
                                        self._store_in_memory(key, data)
                                        logger.debug(f"Prefetched {key} from disk")
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error in prefetch operation: {e}")
        
        # Add to prefetch queue with priority
        with self.prefetch_lock:
            heapq.heappush(self.prefetch_queue, (priority, _prefetch_fn))
    
    def get_stats(self):
        """Get cache performance statistics."""
        with self.lock:
            total_hits = sum(self.stats["hits"].values())
            total_misses = sum(self.stats["misses"].values())
            total_requests = total_hits + total_misses
            hit_rate = total_hits / max(1, total_requests) * 100
            
            compression_ratio = 0
            if self.stats["total_bytes_cached"] > 0:
                bytes_after_compression = self.stats["total_bytes_cached"] - self.stats["total_bytes_saved"]
                compression_ratio = bytes_after_compression / self.stats["total_bytes_cached"]
            
            return {
                "memory_usage": self.memory_usage,
                "memory_limit": self.memory_limit,
                "memory_utilization": self.memory_usage / max(1, self.memory_limit) * 100,
                "items_in_memory": len(self.memory_cache),
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate_percent": hit_rate,
                "total_bytes_cached": self.stats["total_bytes_cached"],
                "total_bytes_saved": self.stats["total_bytes_saved"],
                "compression_ratio": compression_ratio,
                "hits_by_level": dict(self.stats["hits"]),
                "misses_by_type": dict(self.stats["misses"]),
                "evictions": dict(self.stats["evictions"])
            }


class WebSocketManager:
    """Manages real-time WebSocket connections to exchanges."""
    
    def __init__(self, connection_limit=100, reconnect_interval=5, ping_interval=30):
        """
        Initialize the WebSocket manager.
        
        Args:
            connection_limit: Maximum concurrent WebSocket connections
            reconnect_interval: Time to wait before reconnecting (seconds)
            ping_interval: Interval for sending keep-alive pings (seconds)
        """
        self.connection_limit = connection_limit
        self.reconnect_interval = reconnect_interval
        self.ping_interval = ping_interval
        
        # Active connections
        self.connections = {}  # key -> connection
        self.connection_status = {}  # key -> status
        self.connection_stats = {}  # key -> stats
        self.message_handlers = {}  # key -> handler function
        self.error_handlers = {}  # key -> error handler function
        
        # Connection threads
        self.connection_threads = {}  # key -> thread
        
        # Message queues for each connection
        self.message_queues = {}  # key -> queue
        
        # Connection lock
        self.lock = threading.RLock()
        
        # Background tasks
        self.running = False
        self.monitor_thread = None
