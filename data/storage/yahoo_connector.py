#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yahoo Finance Data Connector
===========================

Advanced Yahoo Finance connector with:
- Real-time and historical data
- Multiple asset class support
- Technical indicators calculation
- WebSocket streaming simulation
- Comprehensive error handling
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from decimal import Decimal
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from .base_connector import (
    BaseDataConnector, MarketDataPoint, DataQuality, 
    MarketDataType, DataConnectorStats
)

logger = logging.getLogger(__name__)

class YahooFinanceConnector(BaseDataConnector):
    """Yahoo Finance data connector with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("YahooFinance", config)
        self.session = None
        self.supported_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        self.cache_historical = {}  # Separate cache for historical data
        
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available - Yahoo Finance connector will have limited functionality")
    
    async def connect(self) -> bool:
        """Establish connection to Yahoo Finance."""
        try:
            if not YFINANCE_AVAILABLE:
                logger.error("yfinance library not available")
                return False
            
            # Test connection with a simple request
            test_ticker = yf.Ticker("SPY")
            test_info = test_ticker.info
            
            if test_info and 'symbol' in test_info:
                self.is_connected = True
                logger.info("Successfully connected to Yahoo Finance")
                return True
            else:
                logger.error("Failed to get test data from Yahoo Finance")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Yahoo Finance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Yahoo Finance."""
        self.is_connected = False
        self.session = None
        logger.info("Disconnected from Yahoo Finance")
    
    async def get_real_time_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time data for a symbol."""
        if not self.is_connected or not YFINANCE_AVAILABLE:
            return None
        
        await self._enforce_rate_limit()
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._cache_key(symbol, 'realtime')
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                cached_data = self.cache[cache_key]['data']
                logger.debug(f"Returning cached real-time data for {symbol}")
                return cached_data
            
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get recent 2-day history for current price
            hist = ticker.history(period="2d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            # Get most recent data point
            latest = hist.iloc[-1]
            
            # Determine data type based on symbol
            data_type = self._determine_data_type(symbol, info)
            
            # Create market data point
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                data_type=data_type,
                price=Decimal(str(latest['Close'])),
                open_price=Decimal(str(latest['Open'])),
                high_price=Decimal(str(latest['High'])),
                low_price=Decimal(str(latest['Low'])),
                close_price=Decimal(str(latest['Close'])),
                volume=int(latest['Volume']) if not pd.isna(latest['Volume']) else None,
                source=self.name,
                exchange=info.get('exchange', 'Unknown'),
                metadata={
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
            )
            
            # Calculate technical indicators
            await self._add_technical_indicators(data_point, hist)
            
            # Assess data quality
            latency_ms = (time.time() - start_time) * 1000
            data_point.latency_ms = latency_ms
            data_point.quality = self._assess_data_quality(data_point)
            
            # Update statistics
            self._update_stats(True, latency_ms, data_point.quality)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': data_point,
                'timestamp': time.time()
            }
            
            logger.debug(f"Retrieved real-time data for {symbol} in {latency_ms:.2f}ms")
            return data_point
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms, DataQuality.UNUSABLE)
            logger.error(f"Failed to get real-time data for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, 
                                symbol: str, 
                                start_date: datetime, 
                                end_date: datetime,
                                interval: str = '1d') -> List[MarketDataPoint]:
        """Get historical data for a symbol."""
        if not self.is_connected or not YFINANCE_AVAILABLE:
            return []
        
        await self._enforce_rate_limit()
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{symbol}:{start_date.date()}:{end_date.date()}:{interval}"
            if cache_key in self.cache_historical and self._is_cache_valid(self.cache_historical[cache_key]):
                cached_data = self.cache_historical[cache_key]['data']
                logger.debug(f"Returning cached historical data for {symbol}")
                return cached_data
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            data_type = self._determine_data_type(symbol, info)
            
            # Get historical data
            hist = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return []
            
            data_points = []
            
            for idx, row in hist.iterrows():
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime(),
                    data_type=data_type,
                    open_price=Decimal(str(row['Open'])),
                    high_price=Decimal(str(row['High'])),
                    low_price=Decimal(str(row['Low'])),
                    close_price=Decimal(str(row['Close'])),
                    price=Decimal(str(row['Close'])),
                    volume=int(row['Volume']) if not pd.isna(row['Volume']) else None,
                    source=self.name,
                    quality=DataQuality.GOOD,
                    exchange=info.get('exchange', 'Unknown')
                )
                
                data_points.append(data_point)
            
            # Calculate returns and technical indicators for the series
            await self._add_series_indicators(data_points)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(True, latency_ms, DataQuality.GOOD)
            
            # Cache the result
            self.cache_historical[cache_key] = {
                'data': data_points,
                'timestamp': time.time()
            }
            
            logger.info(f"Retrieved {len(data_points)} historical data points for {symbol} in {latency_ms:.2f}ms")
            return data_points
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms, DataQuality.UNUSABLE)
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def stream_real_time_data(self, symbols: List[str]) -> AsyncGenerator[MarketDataPoint, None]:
        """Stream real-time data for multiple symbols (simulated)."""
        if not self.is_connected:
            return
        
        logger.info(f"Starting real-time data stream for {len(symbols)} symbols")
        
        # Since Yahoo Finance doesn't have true real-time streaming,
        # we simulate it by periodically fetching data
        while self.is_connected:
            try:
                # Fetch data for all symbols
                tasks = [self.get_real_time_data(symbol) for symbol in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Yield successful results
                for result in results:
                    if isinstance(result, MarketDataPoint):
                        yield result
                
                # Wait before next update (configurable)
                update_interval = self.config.get('stream_interval', 60)  # 60 seconds default
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time stream: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols (major indices and stocks)."""
        return [
            # Major indices
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VXUS',
            # Major stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
            # Crypto (through Yahoo)
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD',
            # Forex
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
            # Commodities
            'GC=F', 'SI=F', 'CL=F', 'NG=F'
        ]
    
    def get_supported_data_types(self) -> List[MarketDataType]:
        """Get list of supported data types."""
        return [
            MarketDataType.EQUITY,
            MarketDataType.INDEX,
            MarketDataType.CRYPTO,
            MarketDataType.FOREX,
            MarketDataType.COMMODITY,
            MarketDataType.FUTURE
        ]
    
    def _determine_data_type(self, symbol: str, info: Dict[str, Any]) -> MarketDataType:
        """Determine the data type based on symbol and info."""
        symbol_upper = symbol.upper()
        
        # Crypto patterns
        if '-USD' in symbol_upper or symbol_upper.endswith('-USD'):
            return MarketDataType.CRYPTO
        
        # Forex patterns
        if '=X' in symbol_upper:
            return MarketDataType.FOREX
        
        # Futures patterns
        if '=F' in symbol_upper:
            if symbol_upper.startswith(('GC', 'SI', 'CL', 'NG')):
                return MarketDataType.COMMODITY
            return MarketDataType.FUTURE
        
        # Index patterns
        if symbol_upper in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VXUS']:
            return MarketDataType.INDEX
        
        # Use info to determine type
        quote_type = info.get('quoteType', '').lower()
        if quote_type == 'etf' or quote_type == 'index':
            return MarketDataType.INDEX
        elif quote_type == 'cryptocurrency':
            return MarketDataType.CRYPTO
        
        # Default to equity
        return MarketDataType.EQUITY
    
    async def _add_technical_indicators(self, data_point: MarketDataPoint, hist_data: pd.DataFrame):
        """Add technical indicators to a data point."""
        try:
            if len(hist_data) < 50:
                return  # Not enough data for meaningful indicators
            
            closes = hist_data['Close']
            
            # Moving averages
            if len(closes) >= 20:
                data_point.moving_avg_20 = Decimal(str(closes.rolling(20).mean().iloc[-1]))
            if len(closes) >= 50:
                data_point.moving_avg_50 = Decimal(str(closes.rolling(50).mean().iloc[-1]))
            
            # RSI calculation
            if len(closes) >= 14:
                data_point.rsi = self._calculate_rsi(closes, 14)
            
            # Volatility (20-day)
            if len(closes) >= 20:
                returns = closes.pct_change().dropna()
                if len(returns) >= 20:
                    data_point.volatility = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252))
            
            # VWAP if volume data available
            if 'Volume' in hist_data.columns and not hist_data['Volume'].isna().all():
                data_point.vwap = self._calculate_vwap(hist_data)
                
        except Exception as e:
            logger.warning(f"Failed to calculate technical indicators: {e}")
    
    async def _add_series_indicators(self, data_points: List[MarketDataPoint]):
        """Add technical indicators to a series of data points."""
        try:
            if len(data_points) < 20:
                return
            
            # Extract close prices
            closes = [float(dp.close_price) for dp in data_points]
            close_series = pd.Series(closes)
            
            # Calculate returns
            returns = close_series.pct_change()
            
            # Add returns to data points
            for i, data_point in enumerate(data_points):
                if i > 0:  # Skip first point (no return)
                    data_point.returns = float(returns.iloc[i])
            
            # Moving averages
            ma_20 = close_series.rolling(20).mean()
            ma_50 = close_series.rolling(50).mean()
            
            # RSI
            rsi_values = []
            for i in range(len(closes)):
                if i >= 13:  # Need 14 points for RSI
                    rsi = self._calculate_rsi(close_series.iloc[max(0, i-13):i+1], 14)
                    rsi_values.append(rsi)
                else:
                    rsi_values.append(None)
            
            # Volatility (rolling 20-day)
            volatility = returns.rolling(20).std() * np.sqrt(252)
            
            # Apply indicators to data points
            for i, data_point in enumerate(data_points):
                if i >= 19:  # 20-day MA
                    data_point.moving_avg_20 = Decimal(str(ma_20.iloc[i]))
                if i >= 49:  # 50-day MA
                    data_point.moving_avg_50 = Decimal(str(ma_50.iloc[i]))
                if rsi_values[i] is not None:
                    data_point.rsi = rsi_values[i]
                if i >= 19 and not pd.isna(volatility.iloc[i]):
                    data_point.volatility = float(volatility.iloc[i])
                    
        except Exception as e:
            logger.warning(f"Failed to calculate series indicators: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        except Exception:
            return None
    
    def _calculate_vwap(self, hist_data: pd.DataFrame) -> Optional[Decimal]:
        """Calculate Volume Weighted Average Price."""
        try:
            typical_price = (hist_data['High'] + hist_data['Low'] + hist_data['Close']) / 3
            vwap = (typical_price * hist_data['Volume']).sum() / hist_data['Volume'].sum()
            return Decimal(str(vwap)) if not pd.isna(vwap) else None
        except Exception:
            return None
    
    def get_market_hours(self, symbol: str) -> Dict[str, Any]:
        """Get market hours for a symbol."""
        # Simplified market hours (could be enhanced with actual exchange data)
        symbol_upper = symbol.upper()
        
        if '-USD' in symbol_upper:  # Crypto
            return {
                'market': 'crypto',
                'is_open': True,  # Crypto markets never close
                'next_open': None,
                'next_close': None
            }
        elif '=X' in symbol_upper:  # Forex
            return {
                'market': 'forex',
                'is_open': True,  # Simplified - forex is mostly always open
                'next_open': None,
                'next_close': None
            }
        else:  # Equity/Index
            # This is simplified - in production, use actual market calendar
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_open = (now.weekday() < 5 and  # Monday to Friday
                      market_open <= now <= market_close)
            
            return {
                'market': 'equity',
                'is_open': is_open,
                'market_open': market_open.isoformat(),
                'market_close': market_close.isoformat()
            }

