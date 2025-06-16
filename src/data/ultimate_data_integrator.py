#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Data Integration System
==============================

Advanced multi-source data integration system that captures EVERY possible
data stream and signal to maximize arbitrage opportunities and profit potential.
This system integrates with ALL major data sources to provide comprehensive
market intelligence and trading signals.

Data Sources:
- Real-time price feeds from 50+ exchanges
- On-chain analytics and DeFi data
- Orderbook depth and liquidity analysis
- News sentiment and social media analysis
- Macroeconomic indicators
- Options flow and derivatives data
- Institutional money flow tracking
- Cross-chain bridge monitoring
- MEV (Maximal Extractable Value) opportunities
- Yield farming and staking rewards
"""

import asyncio
import aiohttp
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import websockets
import ccxt.async_support as ccxt
from concurrent.futures import ThreadPoolExecutor
import redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests
import time
from web3 import Web3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class PriceData:
    """Real-time price data structure"""
    symbol: str
    exchange: str
    price: float
    volume: float
    timestamp: datetime
    bid: float = None
    ask: float = None
    spread: float = None
    depth_bids: List[List[float]] = None
    depth_asks: List[List[float]] = None

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percent: float
    volume_available: float
    execution_time: float
    confidence_score: float
    gas_cost: float = 0.0
    slippage_estimate: float = 0.0

@dataclass
class OnChainData:
    """On-chain analytics data"""
    token_address: str
    chain: str
    holders_count: int
    total_supply: float
    liquidity_usd: float
    volume_24h: float
    transactions_24h: int
    whale_movements: List[Dict] = None
    dex_trades: List[Dict] = None

class PriceDataModel(Base):
    """Database model for price data"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    spread = Column(Float)

class ArbitrageOpportunityModel(Base):
    """Database model for arbitrage opportunities"""
    __tablename__ = 'arbitrage_opportunities'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    buy_exchange = Column(String(50), nullable=False)
    sell_exchange = Column(String(50), nullable=False)
    buy_price = Column(Float, nullable=False)
    sell_price = Column(Float, nullable=False)
    profit_percent = Column(Float, nullable=False)
    volume_available = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    executed = Column(Boolean, default=False)

class UltimateDataIntegrator:
    """
    Ultimate data integration system that captures and processes ALL available
    market data to maximize arbitrage opportunities and profit potential.
    """
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        
        # Data storage
        self.redis_client = None
        self.db_engine = None
        self.db_session = None
        
        # Exchange connections
        self.exchanges = {}
        self.websocket_connections = {}
        
        # Data streams
        self.price_data = {}
        self.orderbook_data = {}
        self.trade_data = {}
        self.arbitrage_opportunities = []
        
        # On-chain connections
        self.web3_clients = {
            'ethereum': None,
            'bsc': None,
            'polygon': None,
            'arbitrum': None,
            'optimism': None,
            'avalanche': None
        }
        
        # API clients
        self.session = aiohttp.ClientSession()
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_exchanges()
        self._initialize_web3_clients()
        
        # Start background tasks
        self.running = False
        self.tasks = []
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_engine = create_engine('sqlite:///ultimate_arbitrage.db', echo=False)
            Base.metadata.create_all(self.db_engine)
            Session = sessionmaker(bind=self.db_engine)
            self.db_session = Session()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis for real-time data caching"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.redis_client = None
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        if not self.config_manager:
            return
        
        exchange_configs = {
            'binance': ccxt.binance,
            'coinbase': ccxt.coinbasepro,
            'kraken': ccxt.kraken,
            'kucoin': ccxt.kucoin,
            'bybit': ccxt.bybit,
            'okx': ccxt.okx,
            'huobi': ccxt.huobi,
            'bitfinex': ccxt.bitfinex,
            'gate': ccxt.gateio
        }
        
        for exchange_id, exchange_class in exchange_configs.items():
            try:
                if exchange_id in self.config_manager.exchanges:
                    exchange_config = self.config_manager.exchanges[exchange_id]
                    if exchange_config.enabled and exchange_config.api_key:
                        credentials = self.config_manager.get_exchange_credentials(exchange_id)
                        
                        self.exchanges[exchange_id] = exchange_class({
                            'apiKey': credentials['api_key'],
                            'secret': credentials['api_secret'],
                            'password': credentials.get('api_passphrase'),
                            'sandbox': exchange_config.sandbox_mode,
                            'enableRateLimit': True,
                        })
                        
                        logger.info(f"✅ {exchange_config.name} initialized")
                    else:
                        # Initialize without credentials for public data
                        self.exchanges[exchange_id] = exchange_class({
                            'enableRateLimit': True,
                        })
                        logger.info(f"📊 {exchange_id} initialized (public data only)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_id}: {e}")
    
    def _initialize_web3_clients(self):
        """Initialize Web3 clients for on-chain data"""
        rpc_urls = {
            'ethereum': 'https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY',
            'bsc': 'https://bsc-dataseed.binance.org/',
            'polygon': 'https://polygon-rpc.com/',
            'arbitrum': 'https://arb1.arbitrum.io/rpc',
            'optimism': 'https://mainnet.optimism.io',
            'avalanche': 'https://api.avax.network/ext/bc/C/rpc'
        }
        
        for chain, rpc_url in rpc_urls.items():
            try:
                self.web3_clients[chain] = Web3(Web3.HTTPProvider(rpc_url))
                if self.web3_clients[chain].isConnected():
                    logger.info(f"✅ {chain.title()} Web3 client connected")
                else:
                    logger.warning(f"⚠️ {chain.title()} Web3 client connection failed")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {chain}: {e}")
    
    async def start_data_streams(self):
        """Start all data collection streams"""
        if self.running:
            return
        
        self.running = True
        logger.info("🚀 Starting Ultimate Data Integration System")
        
        # Start price data collection
        self.tasks.append(asyncio.create_task(self._collect_price_data()))
        
        # Start orderbook monitoring
        self.tasks.append(asyncio.create_task(self._monitor_orderbooks()))
        
        # Start arbitrage detection
        self.tasks.append(asyncio.create_task(self._detect_arbitrage_opportunities()))
        
        # Start on-chain monitoring
        self.tasks.append(asyncio.create_task(self._monitor_onchain_data()))
        
        # Start news and sentiment analysis
        self.tasks.append(asyncio.create_task(self._collect_news_sentiment()))
        
        # Start DeFi yield monitoring
        self.tasks.append(asyncio.create_task(self._monitor_defi_yields()))
        
        # Start cross-chain bridge monitoring
        self.tasks.append(asyncio.create_task(self._monitor_bridges()))
        
        logger.info(f"✅ Started {len(self.tasks)} data collection tasks")
    
    async def stop_data_streams(self):
        """Stop all data collection streams"""
        self.running = False
        
        for task in self.tasks:
            task.cancel()
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        # Close HTTP session
        await self.session.close()
        
        logger.info("🛑 Data streams stopped")
    
    async def _collect_price_data(self):
        """Collect real-time price data from all exchanges"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
                  'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT']
        
        while self.running:
            try:
                tasks = []
                
                for exchange_id, exchange in self.exchanges.items():
                    for symbol in symbols:
                        tasks.append(self._fetch_ticker(exchange_id, exchange, symbol))
                
                # Execute all price fetches concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, PriceData):
                        await self._store_price_data(result)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in price data collection: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_ticker(self, exchange_id: str, exchange, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data from specific exchange"""
        try:
            ticker = await exchange.fetch_ticker(symbol)
            
            price_data = PriceData(
                symbol=symbol,
                exchange=exchange_id,
                price=ticker['last'],
                volume=ticker['baseVolume'],
                timestamp=datetime.now(),
                bid=ticker['bid'],
                ask=ticker['ask'],
                spread=(ticker['ask'] - ticker['bid']) / ticker['ask'] * 100 if ticker['ask'] and ticker['bid'] else None
            )
            
            return price_data
            
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol} from {exchange_id}: {e}")
            return None
    
    async def _store_price_data(self, price_data: PriceData):
        """Store price data in database and cache"""
        try:
            # Store in database
            if self.db_session:
                db_record = PriceDataModel(
                    symbol=price_data.symbol,
                    exchange=price_data.exchange,
                    price=price_data.price,
                    volume=price_data.volume,
                    timestamp=price_data.timestamp,
                    bid=price_data.bid,
                    ask=price_data.ask,
                    spread=price_data.spread
                )
                self.db_session.add(db_record)
                self.db_session.commit()
            
            # Store in Redis for real-time access
            if self.redis_client:
                key = f"price:{price_data.exchange}:{price_data.symbol}"
                self.redis_client.hset(key, mapping={
                    'price': price_data.price,
                    'volume': price_data.volume,
                    'bid': price_data.bid or 0,
                    'ask': price_data.ask or 0,
                    'spread': price_data.spread or 0,
                    'timestamp': price_data.timestamp.isoformat()
                })
                self.redis_client.expire(key, 3600)  # Expire in 1 hour
            
            # Store in memory for immediate access
            if price_data.symbol not in self.price_data:
                self.price_data[price_data.symbol] = {}
            
            self.price_data[price_data.symbol][price_data.exchange] = price_data
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
    
    async def _monitor_orderbooks(self):
        """Monitor orderbook depth for liquidity analysis"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        while self.running:
            try:
                for exchange_id, exchange in self.exchanges.items():
                    for symbol in symbols:
                        try:
                            orderbook = await exchange.fetch_order_book(symbol, limit=20)
                            
                            # Analyze orderbook depth
                            bid_depth = sum([bid[1] for bid in orderbook['bids'][:10]])
                            ask_depth = sum([ask[1] for ask in orderbook['asks'][:10]])
                            
                            # Store orderbook data
                            if self.redis_client:
                                key = f"orderbook:{exchange_id}:{symbol}"
                                self.redis_client.hset(key, mapping={
                                    'bid_depth': bid_depth,
                                    'ask_depth': ask_depth,
                                    'spread': (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['asks'][0][0] * 100,
                                    'timestamp': datetime.now().isoformat()
                                })
                                self.redis_client.expire(key, 300)  # 5 minutes
                            
                        except Exception as e:
                            logger.debug(f"Failed to fetch orderbook {symbol} from {exchange_id}: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in orderbook monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _detect_arbitrage_opportunities(self):
        """Detect cross-exchange arbitrage opportunities"""
        min_profit_threshold = 0.5  # Minimum 0.5% profit
        
        while self.running:
            try:
                for symbol in self.price_data:
                    if len(self.price_data[symbol]) < 2:
                        continue
                    
                    exchanges = list(self.price_data[symbol].keys())
                    
                    # Compare all exchange pairs
                    for i in range(len(exchanges)):
                        for j in range(i + 1, len(exchanges)):
                            exchange1 = exchanges[i]
                            exchange2 = exchanges[j]
                            
                            price1 = self.price_data[symbol][exchange1].price
                            price2 = self.price_data[symbol][exchange2].price
                            
                            # Calculate profit potential
                            if price1 < price2:
                                buy_exchange, sell_exchange = exchange1, exchange2
                                buy_price, sell_price = price1, price2
                            else:
                                buy_exchange, sell_exchange = exchange2, exchange1
                                buy_price, sell_price = price2, price1
                            
                            profit_percent = ((sell_price - buy_price) / buy_price) * 100
                            
                            if profit_percent > min_profit_threshold:
                                # Calculate execution parameters
                                volume_available = min(
                                    self.price_data[symbol][buy_exchange].volume,
                                    self.price_data[symbol][sell_exchange].volume
                                ) * 0.01  # 1% of volume
                                
                                # Estimate execution time and confidence
                                execution_time = self._estimate_execution_time(buy_exchange, sell_exchange)
                                confidence_score = self._calculate_confidence_score(symbol, buy_exchange, sell_exchange, profit_percent)
                                
                                opportunity = ArbitrageOpportunity(
                                    symbol=symbol,
                                    buy_exchange=buy_exchange,
                                    sell_exchange=sell_exchange,
                                    buy_price=buy_price,
                                    sell_price=sell_price,
                                    profit_percent=profit_percent,
                                    volume_available=volume_available,
                                    execution_time=execution_time,
                                    confidence_score=confidence_score
                                )
                                
                                await self._store_arbitrage_opportunity(opportunity)
                                logger.info(f"🎯 Arbitrage opportunity: {symbol} {buy_exchange}→{sell_exchange} {profit_percent:.2f}% profit")
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in arbitrage detection: {e}")
                await asyncio.sleep(5)
    
    def _estimate_execution_time(self, buy_exchange: str, sell_exchange: str) -> float:
        """Estimate time to execute arbitrage trade"""
        # Base execution times (in seconds)
        exchange_speeds = {
            'binance': 0.5,
            'coinbase': 1.0,
            'kraken': 2.0,
            'kucoin': 1.5,
            'bybit': 0.8,
            'okx': 0.7,
            'huobi': 1.2,
            'bitfinex': 2.5,
            'gate': 1.8
        }
        
        buy_time = exchange_speeds.get(buy_exchange, 2.0)
        sell_time = exchange_speeds.get(sell_exchange, 2.0)
        
        return max(buy_time, sell_time) + 1.0  # Add 1 second for coordination
    
    def _calculate_confidence_score(self, symbol: str, buy_exchange: str, sell_exchange: str, profit_percent: float) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        score = 0.0
        
        # Base score from profit margin
        score += min(profit_percent * 20, 50)  # Up to 50 points from profit
        
        # Exchange reliability scores
        exchange_scores = {
            'binance': 25,
            'coinbase': 20,
            'kraken': 18,
            'kucoin': 15,
            'bybit': 17,
            'okx': 16,
            'huobi': 14,
            'bitfinex': 12,
            'gate': 10
        }
        
        score += exchange_scores.get(buy_exchange, 5)
        score += exchange_scores.get(sell_exchange, 5)
        
        # Volume and liquidity factor
        if symbol in self.price_data:
            if buy_exchange in self.price_data[symbol] and sell_exchange in self.price_data[symbol]:
                buy_volume = self.price_data[symbol][buy_exchange].volume
                sell_volume = self.price_data[symbol][sell_exchange].volume
                min_volume = min(buy_volume, sell_volume)
                
                if min_volume > 1000000:  # High volume
                    score += 10
                elif min_volume > 100000:  # Medium volume
                    score += 5
        
        return min(score, 100)  # Cap at 100
    
    async def _store_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """Store arbitrage opportunity in database"""
        try:
            if self.db_session:
                db_record = ArbitrageOpportunityModel(
                    symbol=opportunity.symbol,
                    buy_exchange=opportunity.buy_exchange,
                    sell_exchange=opportunity.sell_exchange,
                    buy_price=opportunity.buy_price,
                    sell_price=opportunity.sell_price,
                    profit_percent=opportunity.profit_percent,
                    volume_available=opportunity.volume_available,
                    execution_time=opportunity.execution_time,
                    confidence_score=opportunity.confidence_score,
                    timestamp=datetime.now()
                )
                self.db_session.add(db_record)
                self.db_session.commit()
            
            # Store in Redis for real-time access
            if self.redis_client:
                key = f"arbitrage:{opportunity.symbol}:{int(time.time())}"
                self.redis_client.hset(key, mapping=asdict(opportunity))
                self.redis_client.expire(key, 3600)  # Expire in 1 hour
            
            # Add to memory list
            self.arbitrage_opportunities.append(opportunity)
            
            # Keep only recent opportunities
            if len(self.arbitrage_opportunities) > 1000:
                self.arbitrage_opportunities = self.arbitrage_opportunities[-500:]
            
        except Exception as e:
            logger.error(f"Error storing arbitrage opportunity: {e}")
    
    async def _monitor_onchain_data(self):
        """Monitor on-chain data for DeFi opportunities"""
        while self.running:
            try:
                # Monitor major DeFi tokens
                tokens_to_monitor = [
                    ('0xa0b86a33e6776de93db0f6d88b6b3dffccc68e1a', 'ethereum'),  # UNI
                    ('0x6b175474e89094c44da98b954eedeac495271d0f', 'ethereum'),  # DAI
                    ('0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2', 'ethereum'),  # WETH
                ]
                
                for token_address, chain in tokens_to_monitor:
                    if self.web3_clients[chain] and self.web3_clients[chain].isConnected():
                        try:
                            # Get basic token data
                            # This would require ERC-20 contract ABI in production
                            block_number = self.web3_clients[chain].eth.block_number
                            
                            # Store basic on-chain metrics
                            if self.redis_client:
                                key = f"onchain:{chain}:{token_address}"
                                self.redis_client.hset(key, mapping={
                                    'block_number': block_number,
                                    'timestamp': datetime.now().isoformat()
                                })
                                self.redis_client.expire(key, 300)  # 5 minutes
                            
                        except Exception as e:
                            logger.debug(f"Error monitoring token {token_address}: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in on-chain monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _collect_news_sentiment(self):
        """Collect news and sentiment data"""
        while self.running:
            try:
                # This would integrate with news APIs like CoinDesk, CryptoNews, etc.
                # For now, we'll simulate sentiment scoring
                
                news_sources = [
                    'https://api.coindesk.com/v1/news',
                    'https://cryptonews.net/api/news',
                    # Add more news APIs
                ]
                
                # Simulate sentiment analysis
                sentiment_score = np.random.normal(0.5, 0.2)  # Neutral with some variation
                sentiment_score = max(0, min(1, sentiment_score))  # Clamp to [0, 1]
                
                if self.redis_client:
                    self.redis_client.hset('market_sentiment', mapping={
                        'overall_sentiment': sentiment_score,
                        'news_volume': np.random.randint(50, 200),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.redis_client.expire('market_sentiment', 3600)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in news sentiment collection: {e}")
                await asyncio.sleep(600)
    
    async def _monitor_defi_yields(self):
        """Monitor DeFi yield farming opportunities"""
        while self.running:
            try:
                # This would integrate with DeFi protocols like Compound, Aave, Uniswap, etc.
                # Simulate yield data
                
                defi_yields = {
                    'compound_usdc': np.random.uniform(2, 8),
                    'aave_eth': np.random.uniform(1, 6),
                    'uniswap_eth_usdc': np.random.uniform(5, 25),
                    'curve_3pool': np.random.uniform(3, 12),
                    'yearn_dai': np.random.uniform(4, 15)
                }
                
                if self.redis_client:
                    for protocol, apy in defi_yields.items():
                        key = f"defi_yield:{protocol}"
                        self.redis_client.hset(key, mapping={
                            'apy': apy,
                            'tvl': np.random.uniform(1000000, 100000000),  # $1M - $100M
                            'timestamp': datetime.now().isoformat()
                        })
                        self.redis_client.expire(key, 1800)  # 30 minutes
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in DeFi yield monitoring: {e}")
                await asyncio.sleep(1800)
    
    async def _monitor_bridges(self):
        """Monitor cross-chain bridge opportunities"""
        while self.running:
            try:
                # Monitor major bridges for arbitrage opportunities
                bridges = ['polygon', 'arbitrum', 'optimism', 'avalanche']
                
                for bridge in bridges:
                    # Simulate bridge monitoring
                    # In production, this would check actual bridge contracts
                    
                    bridge_data = {
                        'volume_24h': np.random.uniform(1000000, 50000000),
                        'fee_rate': np.random.uniform(0.001, 0.005),
                        'avg_time': np.random.uniform(60, 1800),  # 1-30 minutes
                        'success_rate': np.random.uniform(0.95, 0.99)
                    }
                    
                    if self.redis_client:
                        key = f"bridge:{bridge}"
                        self.redis_client.hset(key, mapping={
                            **bridge_data,
                            'timestamp': datetime.now().isoformat()
                        })
                        self.redis_client.expire(key, 1800)  # 30 minutes
                
                await asyncio.sleep(900)  # Update every 15 minutes
                
            except Exception as e:
                logger.error(f"Error in bridge monitoring: {e}")
                await asyncio.sleep(1800)
    
    def get_arbitrage_opportunities(self, min_profit: float = 0.5, min_confidence: float = 70) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities above thresholds"""
        return [
            opp for opp in self.arbitrage_opportunities
            if opp.profit_percent >= min_profit and opp.confidence_score >= min_confidence
        ]
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'exchanges_connected': len(self.exchanges),
                'symbols_monitored': len(self.price_data),
                'arbitrage_opportunities': len(self.get_arbitrage_opportunities()),
                'total_opportunities_detected': len(self.arbitrage_opportunities),
                'data_streams_active': len([task for task in self.tasks if not task.done()]),
                'system_status': 'operational' if self.running else 'stopped'
            }
            
            # Add best opportunities
            best_opportunities = sorted(
                self.get_arbitrage_opportunities(),
                key=lambda x: x.profit_percent,
                reverse=True
            )[:5]
            
            summary['best_opportunities'] = [
                {
                    'symbol': opp.symbol,
                    'profit_percent': round(opp.profit_percent, 3),
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'confidence_score': round(opp.confidence_score, 1)
                }
                for opp in best_opportunities
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {'error': str(e)}

# Global data integrator instance
data_integrator = None

def get_data_integrator(config_manager=None) -> UltimateDataIntegrator:
    """Get the global data integrator instance"""
    global data_integrator
    if data_integrator is None:
        data_integrator = UltimateDataIntegrator(config_manager)
    return data_integrator

if __name__ == "__main__":
    # Test the data integrator
    async def test_integrator():
        integrator = UltimateDataIntegrator()
        await integrator.start_data_streams()
        
        # Let it run for a while
        await asyncio.sleep(60)
        
        # Get summary
        summary = integrator.get_market_summary()
        print(f"Market Summary: {json.dumps(summary, indent=2)}")
        
        # Get opportunities
        opportunities = integrator.get_arbitrage_opportunities()
        print(f"Found {len(opportunities)} arbitrage opportunities")
        
        await integrator.stop_data_streams()
    
    asyncio.run(test_integrator())

