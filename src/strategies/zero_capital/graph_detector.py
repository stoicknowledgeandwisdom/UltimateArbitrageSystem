#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graph-Based Opportunity Detection Engine
=======================================

An advanced graph-based detection system for multi-dimensional arbitrage opportunities
across multiple exchanges.

This module implements:
1. A weighted directed graph representing all market pairs across exchanges
2. Bellman-Ford algorithm for negative cycle detection to find arbitrage opportunities
3. Support for detecting triangular, quadrangular, and higher-order arbitrage cycles
4. Real-time graph updating with nanosecond precision timestamps
5. Opportunity scoring and prioritization based on profitability, liquidity, and fees
6. Support for both single-exchange and cross-exchange arbitrage

The system can detect complex arbitrage paths that would be impossible to identify
with traditional methods.
"""

import logging
import time
import threading
import math
import heapq
import uuid
import json
from typing import Dict, List, Tuple, Set, Optional, Any, Union, NamedTuple, Callable
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from collections import defaultdict, deque
import itertools
import concurrent.futures
from dataclasses import dataclass, field
import copy

# Set higher precision for Decimal calculations
getcontext().prec = 28

# Optional dependencies with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Using built-in graph implementation.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Some optimization features will be disabled.")

# Configure logging
logger = logging.getLogger("GraphDetector")


@dataclass
class MarketInfo:
    """Information about a market (trading pair) in the graph."""
    exchange_id: str
    symbol: str
    base: str
    quote: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    base_volume: Decimal = Decimal('0')
    quote_volume: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.now)
    depth_bid: Dict[Decimal, Decimal] = field(default_factory=dict)  # Price -> Volume
    depth_ask: Dict[Decimal, Decimal] = field(default_factory=dict)  # Price -> Volume
    fee_rate: Decimal = Decimal('0.001')  # Default 0.1% fee
    min_order_size: Decimal = Decimal('0')
    max_order_size: Decimal = Decimal('0')
    price_precision: int = 8
    amount_precision: int = 8
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    id: str
    path: List[Tuple[str, str, str]]  # [(exchange_id, base, quote), ...]
    profit_pct: Decimal
    volume_limit: Decimal
    fee_adjusted_profit: Decimal
    timestamp: datetime
    exchanges_involved: List[str]
    currencies_involved: List[str]
    path_length: int
    execution_difficulty: int  # Scale of 1-10, 10 being most difficult
    confidence_score: Decimal  # 0-1 scale
    estimated_execution_time_ms: int
    slippage_estimate_pct: Decimal
    profit_with_slippage: Decimal
    # Additional data for execution
    edge_data: List[Dict[str, Any]] = field(default_factory=list)
    orderbook_snapshot: Dict[str, Any] = field(default_factory=dict)
    # Market data at detection time
    market_rates: Dict[str, Decimal] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary for serialization."""
        result = {
            "id": self.id,
            "path": [(exc, base, quote) for exc, base, quote in self.path],
            "profit_pct": float(self.profit_pct),
            "volume_limit": float(self.volume_limit),
            "fee_adjusted_profit": float(self.fee_adjusted_profit),
            "timestamp": self.timestamp.isoformat(),
            "exchanges_involved": self.exchanges_involved,
            "currencies_involved": self.currencies_involved,
            "path_length": self.path_length,
            "execution_difficulty": self.execution_difficulty,
            "confidence_score": float(self.confidence_score),
            "estimated_execution_time_ms": self.estimated_execution_time_ms,
            "slippage_estimate_pct": float(self.slippage_estimate_pct),
            "profit_with_slippage": float(self.profit_with_slippage),
            "market_rates": {k: float(v) for k, v in self.market_rates.items()}
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArbitrageOpportunity':
        """Create an opportunity from a dictionary."""
        # Convert back to appropriate types
        data["profit_pct"] = Decimal(str(data["profit_pct"]))
        data["volume_limit"] = Decimal(str(data["volume_limit"]))
        data["fee_adjusted_profit"] = Decimal(str(data["fee_adjusted_profit"]))
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["confidence_score"] = Decimal(str(data["confidence_score"]))
        data["slippage_estimate_pct"] = Decimal(str(data["slippage_estimate_pct"]))
        data["profit_with_slippage"] = Decimal(str(data["profit_with_slippage"]))
        data["market_rates"] = {k: Decimal(str(v)) for k, v in data["market_rates"].items()}
        return cls(**data)


class MarketGraph:
    """
    A graph-based representation of cryptocurrency markets across multiple exchanges.
    
    This class maintains a weighted directed graph where:
    - Nodes are (exchange, currency) pairs
    - Edges represent trading pairs with conversion rates as weights
    - Negative cycles in the graph represent arbitrage opportunities
    
    The graph is continuously updated with real-time market data to identify
    emerging opportunities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market graph.
        
        Args:
            config: Configuration parameters for the graph
        """
        self.config = config or {}
        # Default configuration
        self.max_path_length = self.config.get("max_path_length", 5)
        self.min_profit_threshold = Decimal(str(self.config.get("min_profit_threshold", 0.002)))  # 0.2%
        self.max_detection_time = self.config.get("max_detection_time", 0.5)  # seconds
        self.include_cross_exchange = self.config.get("include_cross_exchange", True)
        self.update_interval = self.config.get("update_interval", 1.0)  # seconds
        self.opportunity_expiry = self.config.get("opportunity_expiry", 60)  # seconds
        self.market_data_timeout = self.config.get("market_data_timeout", 60)  # seconds
        self.max_concurrent_detections = self.config.get("max_concurrent_detections", 10)
        self.min_volume_threshold = Decimal(str(self.config.get("min_volume_threshold", 100.0)))  # in USD equivalent
        self.liquidity_preference = Decimal(str(self.config.get("liquidity_preference", 0.7)))  # 0-1, higher values prioritize liquidity
        
        # Use NetworkX if available, otherwise use custom implementation
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            self.use_networkx = True
        else:
            # Custom graph implementation
            self.graph = {
                "nodes": set(),
                "edges": {},
                "predecessors": defaultdict(set),
                "successors": defaultdict(set)
            }
            self.use_networkx = False
        
        # Tracking collections
        self.markets = {}  # (exchange_id, symbol) -> MarketInfo
        self.exchange_currencies = defaultdict(set)  # exchange_id -> set of currencies
        self.cross_exchange_bridges = defaultdict(set)  # currency -> set of exchanges
        self.opportunities = []  # List of detected opportunities
        self.opportunity_index = {}  # id -> opportunity for quick lookups
        self.recently_processed_paths = set()  # To avoid redundant processing
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.detection_times = deque(maxlen=100)
        self.update_times = deque(maxlen=100)
        self.last_full_update = datetime.now()
        self.last_opportunity_detection = datetime.now()
        self.total_detections = 0
        self.valid_opportunities = 0
        
        # Real-time tracking
        self.is_running = False
        self.update_thread = None
        self.detection_thread = None
        self.detection_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent_detections
        )
        
        # Initialize graph memory
        self._clear_graph()
        
        logger.info(f"MarketGraph initialized with {self.max_path_length} max path length")
        
    def _clear_graph(self):
        """Reset the graph to an empty state."""
        if self.use_networkx:
            self.graph.clear()
        else:
            self.graph["nodes"] = set()
            self.graph["edges"] = {}
            self.graph["predecessors"] = defaultdict(set)
            self.graph["successors"] = defaultdict(set)
    
    def start(self) -> bool:
        """
        Start the graph monitoring and detection threads.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Market graph is already running")
            return False
        
        logger.info("Starting market graph monitoring")
        self.is_running = True
        
        try:
            # Start update thread
            self.update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True
            )
            self.update_thread.start()
            
            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True
            )
            self.detection_thread.start()
            
            logger.info("Market graph monitoring started")
            return True
        
        except Exception as e:
            logger.error(f"Error starting market graph: {str(e)}")
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the graph monitoring and detection threads.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("Market graph is already stopped")
            return True
        
        logger.info("Stopping market graph monitoring")
        self.is_running = False
        
        try:
            # Wait for threads to terminate
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5.0)
            
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=5.0)
            
            # Shut down thread pool
            self.detection_pool.shutdown(wait=False)
            
            logger.info("Market graph monitoring stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping market graph: {str(e)}")
            return False
    
    def update_market(self, market_info: MarketInfo) -> bool:
        """
        Update a market in the graph with new market data.
        
        Args:
            market_info: Market information to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        market_key = (market_info.exchange_id, market_info.symbol)
        
        try:
            start_time = time.time()
            
            with self.lock:
                # Store the updated market info
                self.markets[market_key] = market_info
                
                # Update the exchange-currency mappings
                self.exchange_currencies[market_info.exchange_id].add(market_info.base)
                self.exchange_currencies[market_info.exchange_id].add(market_info.quote)
                
                # Update cross-exchange bridges
                self.cross_exchange_bridges[market_info.base].add(market_info.exchange_id)
                self.cross_exchange_bridges[market_info.quote].add(market_info.exchange_id)
                
                # Update the graph
                self._update_graph_edge(market_info)
            
            update_time = time.time() - start_time
            self.update_times.append(update_time)
            
            # Log periodically
            if len(self.update_times) % 100 == 0:
                avg_update_time = sum(self.update_times) / len(self.update_times)
                logger.debug(f"Average market update time: {avg_update_time:.6f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating market {market_key}: {str(e)}")
            return False
    
    def update_markets_batch(self, market_info_batch: List[MarketInfo]) -> int:
        """
        Update multiple markets in the graph in a single batch.
        
        Args:
            market_info_batch: List of market information to update
            
        Returns:
            int: Number of markets successfully updated
        """
        if not market_info_batch:
            return 0
        
        successful_updates = 0
        start_time = time.time()
        
        try:
            with self.lock:
                for market_info in market_info_batch:
                    try:
                        # Update the market
                        market_key = (market_info.exchange_id, market_info.symbol)
                        self.markets[market_key] = market_info
                        
                        # Update the exchange-currency mappings
                        self.exchange_currencies[market_info.exchange_id].add(market_info.base)
                        self.exchange_currencies[market_info.exchange_id].add(market_info.quote)
                        
                        # Update cross-exchange bridges
                        self.cross_exchange_bridges[market_info.base].add(market_info.exchange_id)
                        self.cross_exchange_bridges[market_info.quote].add(

