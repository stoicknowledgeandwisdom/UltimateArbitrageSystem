#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Connection Manager for Database Systems
======================================

Manages connections to multiple database systems with advanced features:
- Connection pooling and load balancing
- Automatic reconnection
- Circuit breaker patterns
- Query profiling and optimization
- Read/write splitting for high-throughput systems
"""

import os
import logging
import time
import threading
import contextlib
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from dataclasses import dataclass
from enum import Enum
import random
import asyncio
from urllib.parse import urlparse

# Optional imports with fallbacks
try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logging.warning("psycopg2 not available. PostgreSQL support will be disabled.")

try:
    import redis
    from redis.sentinel import Sentinel
    from redis.cluster import RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis-py not available. Redis support will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Types of database connections supported by the system."""
    POSTGRES = "postgres"
    TIMESCALE = "timescaledb"
    REDIS = "redis"
    SQLITE = "sqlite"


@dataclass
class ConnectionConfig:
    """Configuration for a database connection."""
    type: ConnectionType
    host: str = "localhost"
    port: int = 0  # Default will be set based on type
    username: str = ""
    password: str = ""
    database: str = ""
    schema: str = "public"
    ssl_mode: str = "prefer"
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_recycle: int = 600  # 10 minutes
    connect_timeout: int = 10
    max_retries: int = 3
    retry_delay: int = 2
    application_name: str = "UltimateArbitrageSystem"
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default ports based on connection type."""
        if self.port == 0:
            if self.type == ConnectionType.POSTGRES or self.type == ConnectionType.TIMESCALE:
                self.port = 5432
            elif self.type == ConnectionType.REDIS:
                self.port = 6379
        
        if self.extra_params is None:
            self.extra_params = {}
    
    @classmethod
    def from_url(cls, url: str) -> 'ConnectionConfig':
        """Create a connection config from a URL string."""
        parsed = urlparse(url)
        
        # Determine connection type from scheme
        if parsed.scheme in ['postgres', 'postgresql']:
            conn_type = ConnectionType.POSTGRES
        elif parsed.scheme in ['timescaledb', 'timescale']:
            conn_type = ConnectionType.TIMESCALE
        elif parsed.scheme == 'redis':
            conn_type = ConnectionType.REDIS
        elif parsed.scheme == 'sqlite':
            conn_type = ConnectionType.SQLITE
        else:
            raise ValueError(f"Unsupported connection scheme: {parsed.scheme}")
        
        # Extract username, password, host, port
        username = parsed.username or ""
        password = parsed.password or ""
        host = parsed.hostname or "localhost"
        port = parsed.port or 0  # Will be set to default in __post_init__
        
        # Extract database name (path without leading slash)
        database = parsed.path.lstrip('/') if parsed.path else ""
        
        # Parse query params for extra options
        extra_params = {}
        if parsed.query:
            import urllib.parse
            extra_params = dict(urllib.parse.parse_qsl(parsed.query))
        
        return cls(
            type=conn_type,
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            extra_params=extra_params
        )
    
    def to_url(self, include_password: bool = False) -> str:
        """Convert connection config to URL string."""
        scheme = ""
        if self.type == ConnectionType.POSTGRES:
            scheme = "postgresql"
        elif self.type == ConnectionType.TIMESCALE:
            scheme = "timescaledb+postgresql"
        elif self.type == ConnectionType.REDIS:
            scheme = "redis"
        elif self.type == ConnectionType.SQLITE:
            return f"sqlite:///{self.database}"
        
        auth = ""
        if self.username:
            if include_password and self.password:
                auth = f"{self.username}:{self.password}@"
            else:
                auth = f"{self.username}@"
        
        port_str = f":{self.port}" if self.port > 0 else ""
        
        return f"{scheme}://{auth}{self.host}{port_str}/{self.database}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert connection config to dictionary."""
        return {
            "type": self.type.value,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "********" if self.password else "",
            "database": self.database,
            "schema": self.schema,
            "ssl_mode": self.ssl_mode,
            "pool_min_size": self.pool_min_size,
            "pool_max_size": self.pool_max_size,
            "pool_recycle": self.pool_recycle,
            "connect_timeout": self.connect_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "application_name": self.application_name,
            "extra_params": self.extra_params
        }


class ConnectionHealthStatus(Enum):
    """Health status of a database connection."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ConnectionManager:
    """
    Primary manager for all database connections in the system.
    
    This class handles:
    - Connection initialization and management
    - Connection pooling and recycling
    - Health checks and automatic recovery
    - Statistics and profiling
    """
    
    def __init__(self, config_path: Optional[str] = None, configs: Optional[Dict[str, ConnectionConfig]] = None):
        """
        Initialize the connection manager.
        
        Args:
            config_path: Path to the JSON configuration file
            configs: Dict of connection configurations
        """
        self.connections = {}  # name -> connection object
        self.connection_pools = {}  # name -> connection pool
        self.configs = {}  # name -> ConnectionConfig
        self.health_status = {}  # name -> ConnectionHealthStatus
        self.last_health_check = {}  # name -> timestamp
        
        self.lock = threading.RLock()
        self.running = False
        self.health_check_thread = None
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        elif configs:
            self.configs = configs
        else:
            logger.warning("No database configuration provided. Please add configuration before use.")
        
        # Statistics for profiling
        self.stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "connection_errors": 0,
            "queries_executed": 0,
            "queries_failed": 0,
            "total_query_time": 0.0,
            "connection_errors_by_name": {}
        }
    
    def _load_config(self, config_path: str) -> None:
        """
        Load connection configurations from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for name, conn_config in config_data.items():
                if isinstance(conn_config, str):
                    # Parse connection URL
                    self.configs[name] = ConnectionConfig.from_url(conn_config)
                else:
                    # Parse connection dict
                    conn_type = ConnectionType(conn_config.get("type", "postgres"))
                    self.configs[name] = ConnectionConfig(
                        type=conn_type,
                        host=conn_config.get("host", "localhost"),
                        port=conn_config.get("port", 0),
                        username=conn_config.get("username", ""),
                        password=conn_config.get("password", ""),
                        database=conn_config.get("database", ""),
                        schema=conn_config.get("schema", "public"),
                        ssl_mode=conn_config.get("ssl_mode", "prefer"),
                        pool_min_size=conn_config.get("pool_min_size", 5),
                        pool_max_size=conn_config.get("pool_max_size", 20),
                        pool_recycle=conn_config.get("pool_recycle", 600),
                        connect_timeout=conn_config.get("connect_timeout", 10),
                        max_retries=conn_config.get("max_retries", 3),
                        retry_delay=conn_config.get("retry_delay", 2),
                        application_name=conn_config.get("application_name", "UltimateArbitrageSystem"),
                        extra_params=conn_config.get("extra_params", {})
                    )
            
            logger.info(f"Loaded {len(self.configs)} database configurations from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading database configuration: {e}")
            raise
    
    def initialize(self, validate: bool = True) -> bool:
        """
        Initialize all configured connections.
        
        Args:
            validate: Whether to validate connections by performing test queries
            
        Returns:
            bool: True if all connections initialized successfully, False otherwise
        """
        success = True
        with self.lock:
            for name, config in self.configs.items():
                logger.info(f"Initializing database connection: {name}")
                try:
                    if config.type in (ConnectionType.POSTGRES, ConnectionType.TIMESCALE):
                        if not POSTGRES_AVAILABLE:
                            logger.error(f"Cannot initialize PostgreSQL connection '{name}': psycopg2 not available")
                            success = False
                            continue
                        self._init_postgres_pool(name, config)
                    elif config.type == ConnectionType.REDIS:
                        if not REDIS_AVAILABLE:
                            logger.error(f"Cannot initialize Redis connection '{name}': redis-py not available")
                            success = False
                            continue
                        self._init_redis_connection(name, config)
                    elif config.type == ConnectionType.SQLITE:
                        self._init_sqlite_connection(name, config)
                    
                    if validate:
                        if not self._validate_connection(name):
                            success = False
                
                except Exception as e:
                    logger.error(f"Error initializing connection '{name}': {e}")
                    self.stats["connection_errors"] += 1
                    if name not in self.stats["connection_errors_by_name"]:
                        self.stats["connection_errors_by_name"][name] = 0
                    self.stats["connection_errors_by_name"][name] += 1
                    success = False
        
        # Start health check thread
        self.running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        return success
    
    def _init_postgres_pool(self, name: str, config: ConnectionConfig) -> None:
        """
        Initialize a PostgreSQL connection pool.
        
        Args:
            name: The connection name
            config: Connection configuration
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL connections")
        
        # Connection string parameters
        conn_params = {
            "host": config.host,
            "port": config.port,
            "dbname": config.database,
            "user": config.username,
            "password": config.password,
            "application_name": config.application_name,
            "sslmode": config.ssl_mode,
            "connect_timeout": config.connect_timeout
        }
        
        # Add any extra parameters
        if config.extra_params:
            conn_params.update(config.extra_params)
        
        # Create connection pool
        self.connection_pools[name] = pg_pool.ThreadedConnectionPool(
            minconn=config.pool_min_size,
            maxconn=config.pool_max_size,
            **conn_params
        )
        
        self.health_status[name] = ConnectionHealthStatus.UNKNOWN
        self.last_health_check[name] = 0
        logger.info(f"Initialized PostgreSQL connection pool for '{name}' with {config.pool_min_size}-{config.pool_max_size} connections")
    
    def _init_redis_connection(self, name: str, config: ConnectionConfig) -> None:
        """
        Initialize a Redis connection.
        
        Args:
            name: The connection name
            config: Connection configuration
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis-py is required for Redis connections")
        
        # Check if using Sentinel for high-availability
        if "sentinel" in config.extra_params and config.extra_params

