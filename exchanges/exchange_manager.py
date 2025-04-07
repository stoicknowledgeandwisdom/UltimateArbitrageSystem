#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exchange Manager Module

This module handles connections to multiple cryptocurrency exchanges (both centralized
and decentralized), provides a unified interface for market data retrieval, order execution,
and balance tracking, all while implementing robust error handling and rate limit management.

Features:
- Multi-exchange support with unified API
- Secure API key management with encryption
- Real-time market data access via REST and WebSockets
- Order placement and management with advanced order types
- Balance tracking across exchanges with automatic reconciliation
- Comprehensive error handling with retry mechanisms
- Adaptive rate limit protection
- Support for both centralized and decentralized exchanges
- Performance monitoring and metrics collection
- High-availability design with failover capabilities
"""

import os
import json
import time
import hmac
import base64
import hashlib
import logging
import threading
import queue
import websocket
import requests
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import ccxt
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Set up logging
logger = logging.getLogger("ExchangeManager")

# Load environment variables
load_dotenv()

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

class TimeInForce(Enum):
    """Time in force options for orders"""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date

class ExchangeFeature(Enum):
    """Features that may be supported by exchanges"""
    MARGIN_TRADING = "margin_trading"
    FUTURES_TRADING = "futures_trading"
    SPOT_TRADING = "spot_trading"
    WEBSOCKET_SUPPORT = "websocket_support"
    LENDING = "lending"
    STAKING = "staking"
    MULTIPLE_ORDER_TYPES = "multiple_order_types"
    REST_API = "rest_api"
    PORTFOLIO_MARGIN = "portfolio_margin"

class ExchangeError(Exception):
    """Base class for exchange-related exceptions"""
    def __init__(self, message, exchange_id=None, retry_possible=False, error_code=None):
        self.exchange_id = exchange_id
        self.retry_possible = retry_possible
        self.error_code = error_code
        super().__init__(f"{message} [Exchange: {exchange_id}, Error Code: {error_code}]")

class ConnectionError(ExchangeError):
    """Raised when a connection to an exchange fails"""
    pass

class AuthenticationError(ExchangeError):
    """Raised when authentication to an exchange fails"""
    pass

class OrderError(ExchangeError):
    """Raised when an order fails"""
    pass

class RateLimitError(ExchangeError):
    """Raised when rate limits are exceeded"""
    pass

class BalanceError(ExchangeError):
    """Raised when there's an issue with balances"""
    pass

class MarketDataError(ExchangeError):
    """Raised when market data cannot be retrieved"""
    pass

class WebSocketError(ExchangeError):
    """Raised when there's an issue with WebSocket connection"""
    pass

class APIKeyManager:
    """Secure management of API keys with encryption"""
    
    def __init__(self, master_password=None):
        """Initialize API key manager
        
        Args:
            master_password: Master password for encryption/decryption. If None, will use environment variable.
        """
        self.keys_file = Path("config/api_keys.encrypted")
        self.master_password = master_password or os.environ.get("API_MASTER_PASSWORD")
        
        if not self.master_password:
            self.master_password = os.urandom(16).hex()
            logger.warning(f"No master password provided. Generated temporary password: {self.master_password}")
            logger.warning("This is insecure. Please set API_MASTER_PASSWORD environment variable.")
        
        # Initialize encryption key
        self._init_encryption()
        
        # Load existing keys if available
        self.api_keys = self._load_keys()
    
    def _init_encryption(self):
        """Initialize encryption using master password"""
        # Convert master password to encryption key using PBKDF2
        salt = b'ExchangeManagerSalt'  # In production, this should be stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        self.cipher = Fernet(key)
    
    def _load_keys(self) -> Dict[str, Dict[str, str]]:
        """Load and decrypt API keys from the encrypted file"""
        if not self.keys_file.exists():
            return {}
        
        try:
            encrypted_data = self.keys_file.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
            return {}
    
    def _save_keys(self):
        """Encrypt and save API keys to the file"""
        try:
            # Create directory if it doesn't exist
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Encrypt and save
            encrypted_data = self.cipher.encrypt(json.dumps(self.api_keys).encode())
            self.keys_file.write_bytes(encrypted_data)
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")
    
    def add_api_key(self, exchange_id: str, api_key: str, api_secret: str, 
                    additional_params: Dict[str, str] = None) -> bool:
        """Add an API key for an exchange
        
        Args:
            exchange_id: Unique identifier for the exchange
            api_key: API key
            api_secret: API secret
            additional_params: Additional parameters like passphrase for some exchanges
            
        Returns:
            bool: Success status
        """
        try:
            key_data = {
                "api_key": api_key,
                "api_secret": api_secret,
                "timestamp": datetime.now().isoformat()
            }
            
            if additional_params:
                key_data.update(additional_params)
            
            self.api_keys[exchange_id] = key_data
            self._save_keys()
            logger.info(f"Added API key for {exchange_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding API key for {exchange_id}: {str(e)}")
            return False
    
    def get_api_key(self, exchange_id: str) -> Dict[str, str]:
        """Get API key data for an exchange
        
        Args:
            exchange_id: Unique identifier for the exchange
            
        Returns:
            dict: API key data including key, secret, and any additional params
        """
        return self.api_keys.get(exchange_id, {})
    
    def remove_api_key(self, exchange_id: str) -> bool:
        """Remove API key for an exchange
        
        Args:
            exchange_id: Unique identifier for the exchange
            
        Returns:
            bool: Success status
        """
        if exchange_id in self.api_keys:
            del self.api_keys[exchange_id]
            self._save_keys()
            logger.info(f"Removed API key for {exchange_id}")
            return True
        return False
    
    def list_exchanges(self) -> List[str]:
        """List all exchanges with stored API keys
        
        Returns:
            list: List of exchange IDs
        """
        return list(self.api_keys.keys())


class RateLimitManager:
    """Manages rate limits for exchange API calls"""
    
    def __init__(self):
        """Initialize rate limit manager"""
        # Exchange ID -> Endpoint -> Rate Limit Info
        self.rate_limits = {}
        self.last_request_times = {}
        self.request_counts = {}
        self.adaptive_delays = {}
        self.lock = threading.RLock()
    
    def set_rate_limit(self, exchange_id: str, endpoint: str, 
                       requests_per_second: float, burst_limit: int = None):
        """Set rate limit for an exchange endpoint
        
        Args:
            exchange_id: Exchange identifier
            endpoint: API endpoint or 'default' for all endpoints
            requests_per_second: Max requests per second
            burst_limit: Number of requests allowed in a burst
        """
        with self.lock:
            if exchange_id not in self.rate_limits:
                self.rate_limits[exchange_id] = {}
                self.last_request_times[exchange_id] = {}
                self.request_counts[exchange_id] = {}
                self.adaptive_delays[exchange_id] = {}
            
            self.rate_limits[exchange_id][endpoint] = {
                'requests_per_second': requests_per_second,
                'min_interval': 1.0 / requests_per_second,
                'burst_limit': burst_limit
            }
            
            self.last_request_times[exchange_id][endpoint] = 0
            self.request_counts[exchange_id][endpoint] = 0
            self.adaptive_delays[exchange_id][endpoint] = 0
    
    def apply_rate_limit(self, exchange_id: str, endpoint: str = 'default'):
        """Apply rate limiting by adding appropriate delay
        
        Args:
            exchange_id: Exchange identifier
            endpoint: API endpoint or 'default' for default rate limit
            
        Returns:
            float: Time slept in seconds
        """
        with self.lock:
            # Use default endpoint if specific one not defined
            if endpoint not in self.rate_limits.get(exchange_id, {}):
                endpoint = 'default'
            
            if (exchange_id not in self.rate_limits or 
                endpoint not in self.rate_limits[exchange_id]):
                return 0
            
            current_time = time.time()
            rate_limit_info = self.rate_limits[exchange_id][endpoint]
            last_request_time = self.last_request_times[exchange_id][endpoint]
            min_interval = rate_limit_info['min_interval']
            
            # Calculate required delay including adaptive component
            elapsed = current_time - last_request_time
            adaptive_delay = self.adaptive_delays[exchange_id][endpoint]
            total_delay = max(0, min_interval - elapsed) + adaptive_delay
            
            if total_delay > 0:
                time.sleep(total_delay)
            
            # Update request counts and timing
            self.last_request_times[exchange_id][endpoint] = time.time()
            self.request_counts[exchange_id][endpoint] += 1
            
            return total_delay
    
    def handle_rate_limit_error(self, exchange_id: str, endpoint: str = 'default'):
        """Handle a rate limit error by increasing the adaptive delay
        
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
        """Open a WebSocket connection to an exchange
        

