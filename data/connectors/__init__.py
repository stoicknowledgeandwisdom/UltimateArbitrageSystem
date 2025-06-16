#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Market Data Connectors
==============================

Multi-source, real-time market data integration framework with:
- Real-time WebSocket streaming
- Multiple data provider support
- Advanced caching and quality control
- Cryptocurrency and traditional market support
- News and sentiment data integration
"""

from .base_connector import BaseDataConnector, MarketDataPoint, DataQuality
from .yahoo_connector import YahooFinanceConnector
from .alpha_vantage_connector import AlphaVantageConnector
from .crypto_connector import CryptoDataConnector
from .news_connector import NewsDataConnector
from .data_manager import AdvancedDataManager

__all__ = [
    'BaseDataConnector',
    'MarketDataPoint',
    'DataQuality',
    'YahooFinanceConnector',
    'AlphaVantageConnector', 
    'CryptoDataConnector',
    'NewsDataConnector',
    'AdvancedDataManager'
]

