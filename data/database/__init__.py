#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UltimateArbitrageSystem: Database Module
========================================

Advanced database system for the UltimateArbitrageSystem with specialized storage
for high-frequency data, system state, and real-time caching.

Features:
- TimescaleDB integration for time-series market data
- PostgreSQL for system state, configuration, and trading history
- Redis for real-time data caching and pub/sub messaging
- Automatic migration and schema management
- Optimized query engines for high-performance data retrieval
- Fault-tolerant connection management
- Concurrent access handling
- Data compression and optimization
"""

from data.database.connection_manager import ConnectionManager
from data.database.timescale_manager import TimescaleManager
from data.database.postgres_manager import PostgresManager
from data.database.redis_manager import RedisManager
from data.database.models import (
    Exchange, Symbol, MarketData, OrderBook, Trade, 
    Position, Strategy, Performance, SystemConfig
)

__all__ = [
    'ConnectionManager',
    'TimescaleManager',
    'PostgresManager',
    'RedisManager',
    'Exchange',
    'Symbol',
    'MarketData',
    'OrderBook',
    'Trade',
    'Position',
    'Strategy',
    'Performance',
    'SystemConfig',
]

