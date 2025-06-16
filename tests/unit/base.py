"""Base classes and utilities for unit testing"""

import pytest
import asyncio
import unittest.mock as mock
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock
import tempfile
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class BaseUnitTest:
    """Base class for all unit tests providing common utilities"""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup test logging"""
        logging.getLogger().setLevel(logging.DEBUG)
    
    @pytest.fixture
    def mock_exchange_api(self):
        """Mock exchange API responses"""
        mock_api = MagicMock()
        mock_api.get_orderbook = AsyncMock(return_value={
            'bids': [[100.0, 1.5], [99.9, 2.0]],
            'asks': [[100.1, 1.0], [100.2, 0.5]],
            'timestamp': datetime.now().timestamp()
        })
        mock_api.place_order = AsyncMock(return_value={
            'id': '12345',
            'status': 'open',
            'symbol': 'BTC/USDT',
            'amount': 1.0,
            'price': 100.0
        })
        return mock_api
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)  # For reproducible tests
        
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        volumes = np.random.exponential(1000, 1000)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.rand(1000) * 0.01),
            'low': prices * (1 - np.random.rand(1000) * 0.01),
            'close': prices,
            'volume': volumes
        })
    
    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio data"""
        return {
            'BTC': {'balance': 1.5, 'locked': 0.0},
            'ETH': {'balance': 10.0, 'locked': 2.0},
            'USDT': {'balance': 50000.0, 'locked': 1000.0}
        }
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'exchanges': {
                'binance': {
                    'api_key': 'test_key',
                    'secret': 'test_secret',
                    'sandbox': True
                }
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_percentage': 0.02,
                'take_profit_percentage': 0.03
            },
            'strategy': {
                'name': 'test_strategy',
                'parameters': {
                    'timeframe': '1m',
                    'lookback': 100
                }
            }
        }
    
    def assert_float_equal(self, a: float, b: float, tolerance: float = 1e-6):
        """Assert two floats are equal within tolerance"""
        assert abs(a - b) < tolerance, f"Expected {a} ≈ {b} (tolerance: {tolerance})"
    
    def assert_portfolio_valid(self, portfolio: Dict[str, Any]):
        """Assert portfolio structure is valid"""
        assert isinstance(portfolio, dict)
        for asset, balance_info in portfolio.items():
            assert isinstance(asset, str)
            assert 'balance' in balance_info
            assert 'locked' in balance_info
            assert balance_info['balance'] >= 0
            assert balance_info['locked'] >= 0
    
    def create_mock_order(self, 
                         symbol: str = 'BTC/USDT',
                         side: str = 'buy',
                         amount: float = 1.0,
                         price: float = 100.0,
                         order_type: str = 'limit') -> Dict[str, Any]:
        """Create a mock order for testing"""
        return {
            'id': f'test_order_{np.random.randint(10000, 99999)}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'type': order_type,
            'status': 'open',
            'timestamp': datetime.now().timestamp(),
            'filled': 0.0,
            'remaining': amount,
            'fee': 0.001 * amount * price
        }
    
    def create_mock_trade(self,
                         symbol: str = 'BTC/USDT',
                         side: str = 'buy',
                         amount: float = 1.0,
                         price: float = 100.0) -> Dict[str, Any]:
        """Create a mock trade for testing"""
        return {
            'id': f'test_trade_{np.random.randint(10000, 99999)}',
            'order_id': f'test_order_{np.random.randint(10000, 99999)}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now().timestamp(),
            'fee': 0.001 * amount * price,
            'fee_currency': 'USDT'
        }


class AsyncTestMixin:
    """Mixin for testing async functions"""
    
    @pytest.fixture
    def event_loop(self):
        """Create an instance of the default event loop for the test session."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    async def run_async_test(self, coro):
        """Helper to run async test functions"""
        return await coro


class MockDataGenerator:
    """Generate mock data for testing"""
    
    @staticmethod
    def generate_ohlcv_data(symbol: str = 'BTC/USDT',
                           periods: int = 1000,
                           base_price: float = 100.0,
                           volatility: float = 0.02) -> pd.DataFrame:
        """Generate OHLCV data for testing"""
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=periods, freq='1min')
        returns = np.random.normal(0, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        volumes = np.random.exponential(1000, periods)
        
        # Generate OHLC from prices
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'close': prices,
            'volume': volumes
        })
        
        # Add realistic high/low
        df['high'] = df['close'] * (1 + np.random.rand(periods) * 0.01)
        df['low'] = df['close'] * (1 - np.random.rand(periods) * 0.01)
        
        return df
    
    @staticmethod
    def generate_orderbook_data(symbol: str = 'BTC/USDT',
                               levels: int = 20,
                               mid_price: float = 100.0,
                               spread: float = 0.01) -> Dict[str, Any]:
        """Generate orderbook data for testing"""
        half_spread = spread / 2
        
        # Generate bids (descending prices)
        bid_prices = np.linspace(mid_price - half_spread, 
                                mid_price - half_spread - 1.0, levels)
        bid_amounts = np.random.exponential(1.0, levels)
        bids = [[float(p), float(a)] for p, a in zip(bid_prices, bid_amounts)]
        
        # Generate asks (ascending prices)
        ask_prices = np.linspace(mid_price + half_spread,
                                mid_price + half_spread + 1.0, levels)
        ask_amounts = np.random.exponential(1.0, levels)
        asks = [[float(p), float(a)] for p, a in zip(ask_prices, ask_amounts)]
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().timestamp(),
            'datetime': datetime.now().isoformat(),
            'nonce': None
        }
    
    @staticmethod
    def generate_trade_history(symbol: str = 'BTC/USDT',
                              count: int = 100) -> List[Dict[str, Any]]:
        """Generate trade history for testing"""
        trades = []
        base_time = datetime.now().timestamp()
        
        for i in range(count):
            trades.append({
                'id': f'trade_{i}',
                'symbol': symbol,
                'side': np.random.choice(['buy', 'sell']),
                'amount': np.random.exponential(1.0),
                'price': 100.0 + np.random.normal(0, 1.0),
                'timestamp': base_time - (count - i) * 60,  # 1 minute intervals
                'fee': np.random.uniform(0.001, 0.01)
            })
        
        return trades

