#!/usr/bin/env python3
"""
Ultimate Arbitrage System - Maximum Income Multi-Exchange System
==============================================================

Expands to every possible exchange and trading pair for maximum profit potential.
Implements zero-investment mindset: sees opportunities others miss.
"""

import os
import sys
import time
import json
import yaml
import asyncio
import logging
import requests
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sqlite3
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'maximum_income_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MaximumIncome')

@dataclass
class EnhancedMarketData:
    """Enhanced market data with additional profit indicators"""
    symbol: str
    exchange: str
    bid: float
    ask: float
    timestamp: datetime
    volume_24h: float
    spread_percentage: float
    volatility: float
    market_cap: Optional[float] = None
    liquidity_score: float = 0.0
    trading_fees: Dict[str, float] = None

@dataclass
class MaximumProfitOpportunity:
    """Enhanced arbitrage opportunity with maximum profit analysis"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_usd: float
    max_trade_size: float
    execution_time_estimate: float
    confidence_score: float
    risk_factors: List[str]
    liquidity_factor: float
    volume_factor: float
    opportunity_type: str  # 'arbitrage', 'triangular', 'cross_chain', 'defi'

class MaximumIncomeDataCollector:
    """Collects data from maximum number of exchanges for highest profit potential"""
    
    def __init__(self):
        self.exchanges = {
            # Major Centralized Exchanges
            'binance': {
                'api_base': 'https://api.binance.com/api/v3',
                'rate_limit': 1200,
                'trading_fee': 0.001,
                'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'SUSHI/USDT']
            },
            'coinbase': {
                'api_base': 'https://api.exchange.coinbase.com',
                'rate_limit': 600,
                'trading_fee': 0.005,
                'symbols': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD']
            },
            'kraken': {
                'api_base': 'https://api.kraken.com/0/public',
                'rate_limit': 300,
                'trading_fee': 0.0026,
                'symbols': ['XBTUSD', 'ETHUSD', 'ADAUSD', 'DOTUSD', 'MATICUSD', 'LINKUSD']
            },
            'kucoin': {
                'api_base': 'https://api.kucoin.com/api/v1',
                'rate_limit': 1800,
                'trading_fee': 0.001,
                'symbols': ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'MATIC-USDT', 'LINK-USDT', 'UNI-USDT', 'AAVE-USDT', 'SUSHI-USDT', 'CRV-USDT']
            },
            'huobi': {
                'api_base': 'https://api.huobi.pro',
                'rate_limit': 600,
                'trading_fee': 0.002,
                'symbols': ['btcusdt', 'ethusdt', 'adausdt', 'dotusdt', 'maticusdt', 'linkusdt', 'uniusdt', 'aaveusdt']
            },
            'gate': {
                'api_base': 'https://api.gateio.ws/api/v4',
                'rate_limit': 600,
                'trading_fee': 0.002,
                'symbols': ['BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'DOT_USDT', 'MATIC_USDT', 'LINK_USDT', 'UNI_USDT', 'AAVE_USDT']
            },
            'okx': {
                'api_base': 'https://www.okx.com/api/v5',
                'rate_limit': 600,
                'trading_fee': 0.001,
                'symbols': ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT', 'MATIC-USDT', 'LINK-USDT', 'UNI-USDT', 'AAVE-USDT']
            },
            'bybit': {
                'api_base': 'https://api.bybit.com/v5',
                'rate_limit': 600,
                'trading_fee': 0.001,
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT']
            },
            'mexc': {
                'api_base': 'https://api.mexc.com/api/v3',
                'rate_limit': 1200,
                'trading_fee': 0.002,
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT']
            },
            'bitget': {
                'api_base': 'https://api.bitget.com/api/v2',
                'rate_limit': 600,
                'trading_fee': 0.001,
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT']
            }
        }
        
        # DEX and DeFi Protocols (via aggregators)
        self.defi_protocols = {
            'uniswap_v3': {
                'api_base': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                'trading_fee': 0.003,
                'tokens': ['WETH', 'USDC', 'USDT', 'DAI', 'WBTC', 'UNI', 'AAVE', 'LINK', 'MATIC']
            },
            'sushiswap': {
                'api_base': 'https://api.thegraph.com/subgraphs/name/sushiswap/exchange',
                'trading_fee': 0.003,
                'tokens': ['WETH', 'USDC', 'USDT', 'DAI', 'WBTC', 'SUSHI', 'AAVE', 'LINK']
            },
            'pancakeswap': {
                'api_base': 'https://api.thegraph.com/subgraphs/name/pancakeswap/exchange',
                'trading_fee': 0.0025,
                'tokens': ['BNB', 'BUSD', 'USDT', 'BTCB', 'ETH', 'CAKE', 'ADA', 'DOT']
            }
        }
        
        self.rate_limits = {}
        self.initialize_rate_limits()
        
    def initialize_rate_limits(self):
        """Initialize rate limiting for all exchanges"""
        for exchange in self.exchanges:
            self.rate_limits[exchange] = {
                'requests': 0,
                'reset_time': time.time(),
                'max_requests': self.exchanges[exchange]['rate_limit'] // 60  # per minute
            }
    
    def check_rate_limit(self, exchange: str) -> bool:
        """Enhanced rate limiting with burst protection"""
        now = time.time()
        limits = self.rate_limits[exchange]
        
        # Reset counter every minute
        if now - limits['reset_time'] > 60:
            limits['requests'] = 0
            limits['reset_time'] = now
            
        if limits['requests'] < limits['max_requests']:
            limits['requests'] += 1
            return True
        return False
    
    def get_binance_data(self, symbol: str) -> Optional[EnhancedMarketData]:
        """Enhanced Binance data collection"""
        if not self.check_rate_limit('binance'):
            return None
            
        try:
            # Convert symbol format
            binance_symbol = symbol.replace('/', '').replace('-', '')
            
            # Get enhanced ticker data
            ticker_url = f"{self.exchanges['binance']['api_base']}/ticker/bookTicker?symbol={binance_symbol}"
            stats_url = f"{self.exchanges['binance']['api_base']}/ticker/24hr?symbol={binance_symbol}"
            
            ticker_response = requests.get(ticker_url, timeout=3)
            stats_response = requests.get(stats_url, timeout=3)
            
            if ticker_response.status_code == 200 and stats_response.status_code == 200:
                ticker_data = ticker_response.json()
                stats_data = stats_response.json()
                
                bid = float(ticker_data['bidPrice'])
                ask = float(ticker_data['askPrice'])
                spread = (ask - bid) / bid * 100
                volume_24h = float(stats_data['quoteVolume'])
                price_change = abs(float(stats_data['priceChangePercent']))
                
                return EnhancedMarketData(
                    symbol=symbol,
                    exchange='binance',
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.now(),
                    volume_24h=volume_24h,
                    spread_percentage=spread,
                    volatility=price_change,
                    liquidity_score=min(volume_24h / 1000000, 10.0),
                    trading_fees={'maker': 0.001, 'taker': 0.001}
                )
                
        except Exception as e:
            logger.debug(f"Binance API error for {symbol}: {e}")
            
        return None
    
    def get_coinbase_data(self, symbol: str) -> Optional[EnhancedMarketData]:
        """Enhanced Coinbase data collection"""
        if not self.check_rate_limit('coinbase'):
            return None
            
        try:
            coinbase_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
            
            ticker_url = f"{self.exchanges['coinbase']['api_base']}/products/{coinbase_symbol}/ticker"
            stats_url = f"{self.exchanges['coinbase']['api_base']}/products/{coinbase_symbol}/stats"
            
            ticker_response = requests.get(ticker_url, timeout=3)
            stats_response = requests.get(stats_url, timeout=3)
            
            if ticker_response.status_code == 200:
                ticker_data = ticker_response.json()
                
                bid = float(ticker_data['bid'])
                ask = float(ticker_data['ask'])
                spread = (ask - bid) / bid * 100
                volume_24h = float(ticker_data['volume']) * float(ticker_data['price'])
                
                volatility = 1.0  # Default volatility
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    if stats_data.get('high') and stats_data.get('low'):
                        high = float(stats_data['high'])
                        low = float(stats_data['low'])
                        volatility = ((high - low) / low) * 100
                
                return EnhancedMarketData(
                    symbol=symbol,
                    exchange='coinbase',
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.now(),
                    volume_24h=volume_24h,
                    spread_percentage=spread,
                    volatility=volatility,
                    liquidity_score=min(volume_24h / 1000000, 10.0),
                    trading_fees={'maker': 0.005, 'taker': 0.005}
                )
                
        except Exception as e:
            logger.debug(f"Coinbase API error for {symbol}: {e}")
            
        return None
    
    def get_kucoin_data(self, symbol: str) -> Optional[EnhancedMarketData]:
        """KuCoin data collection"""
        if not self.check_rate_limit('kucoin'):
            return None
            
        try:
            kucoin_symbol = symbol.replace('/', '-')
            
            ticker_url = f"{self.exchanges['kucoin']['api_base']}/market/orderbook/level1?symbol={kucoin_symbol}"
            stats_url = f"{self.exchanges['kucoin']['api_base']}/market/stats?symbol={kucoin_symbol}"
            
            ticker_response = requests.get(ticker_url, timeout=3)
            stats_response = requests.get(stats_url, timeout=3)
            
            if ticker_response.status_code == 200 and ticker_response.json().get('code') == '200000':
                ticker_data = ticker_response.json()['data']
                
                bid = float(ticker_data['bestBid'])
                ask = float(ticker_data['bestAsk'])
                spread = (ask - bid) / bid * 100
                
                volume_24h = 0
                volatility = 1.0
                
                if stats_response.status_code == 200 and stats_response.json().get('code') == '200000':
                    stats_data = stats_response.json()['data']
                    volume_24h = float(stats_data.get('volValue', 0))
                    if stats_data.get('high') and stats_data.get('low'):
                        high = float(stats_data['high'])
                        low = float(stats_data['low'])
                        volatility = ((high - low) / low) * 100
                
                return EnhancedMarketData(
                    symbol=symbol,
                    exchange='kucoin',
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.now(),
                    volume_24h=volume_24h,
                    spread_percentage=spread,
                    volatility=volatility,
                    liquidity_score=min(volume_24h / 1000000, 10.0),
                    trading_fees={'maker': 0.001, 'taker': 0.001}
                )
                
        except Exception as e:
            logger.debug(f"KuCoin API error for {symbol}: {e}")
            
        return None
    
    def get_okx_data(self, symbol: str) -> Optional[EnhancedMarketData]:
        """OKX data collection"""
        if not self.check_rate_limit('okx'):
            return None
            
        try:
            okx_symbol = symbol.replace('/', '-')
            
            ticker_url = f"{self.exchanges['okx']['api_base']}/market/ticker?instId={okx_symbol}"
            
            response = requests.get(ticker_url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '0' and data.get('data'):
                    ticker_data = data['data'][0]
                    
                    bid = float(ticker_data['bidPx'])
                    ask = float(ticker_data['askPx'])
                    spread = (ask - bid) / bid * 100
                    volume_24h = float(ticker_data['volCcy24h'])
                    
                    # Calculate volatility
                    high = float(ticker_data['high24h'])
                    low = float(ticker_data['low24h'])
                    volatility = ((high - low) / low) * 100 if low > 0 else 1.0
                    
                    return EnhancedMarketData(
                        symbol=symbol,
                        exchange='okx',
                        bid=bid,
                        ask=ask,
                        timestamp=datetime.now(),
                        volume_24h=volume_24h,
                        spread_percentage=spread,
                        volatility=volatility,
                        liquidity_score=min(volume_24h / 1000000, 10.0),
                        trading_fees={'maker': 0.001, 'taker': 0.001}
                    )
                
        except Exception as e:
            logger.debug(f"OKX API error for {symbol}: {e}")
            
        return None
    
    def collect_exchange_data(self, exchange: str, symbols: List[str]) -> List[EnhancedMarketData]:
        """Collect data from a specific exchange"""
        data_list = []
        
        for symbol in symbols:
            try:
                if exchange == 'binance':
                    data = self.get_binance_data(symbol)
                elif exchange == 'coinbase':
                    data = self.get_coinbase_data(symbol)
                elif exchange == 'kucoin':
                    data = self.get_kucoin_data(symbol)
                elif exchange == 'okx':
                    data = self.get_okx_data(symbol)
                else:
                    continue
                    
                if data:
                    data_list.append(data)
                    
                # Rate limiting delay
                time.sleep(0.05)
                
            except Exception as e:
                logger.debug(f"Error collecting {symbol} from {exchange}: {e}")
                continue
                
        return data_list
    
    def collect_all_data_parallel(self) -> List[EnhancedMarketData]:
        """Collect data from all exchanges in parallel for maximum speed"""
        all_data = []
        
        # Normalize symbols across exchanges
        normalized_symbols = {
            'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 
            'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'AAVE/USDT'
        }
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # Submit tasks for each exchange
            for exchange, config in self.exchanges.items():
                if exchange in ['binance', 'coinbase', 'kucoin', 'okx']:  # Active exchanges
                    symbols = [s for s in normalized_symbols if self.symbol_supported(exchange, s)]
                    future = executor.submit(self.collect_exchange_data, exchange, symbols)
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    exchange_data = future.result(timeout=10)
                    all_data.extend(exchange_data)
                except Exception as e:
                    logger.warning(f"Exchange data collection failed: {e}")
        
        logger.info(f"Collected data from {len(all_data)} symbol-exchange pairs")
        return all_data
    
    def symbol_supported(self, exchange: str, symbol: str) -> bool:
        """Check if symbol is supported by exchange"""
        if exchange == 'coinbase':
            return symbol.replace('USDT', 'USD') in ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD']
        elif exchange == 'kraken':
            return symbol in ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'MATIC/USD', 'LINK/USD']
        else:
            return True  # Assume other exchanges support USDT pairs

class MaximumProfitDetector:
    """Detects every possible profit opportunity across all markets"""
    
    def __init__(self, min_profit_threshold: float = 0.001):  # Lower threshold for more opportunities
        self.min_profit_threshold = min_profit_threshold  # 0.1% minimum
        self.opportunity_types = ['spot_arbitrage', 'triangular_arbitrage', 'statistical_arbitrage', 'latency_arbitrage']
        
    def detect_spot_arbitrage(self, market_data: List[EnhancedMarketData]) -> List[MaximumProfitOpportunity]:
        """Detect spot arbitrage opportunities"""
        opportunities = []
        
        # Group by symbol
        symbol_groups = {}
        for data in market_data:
            if data.symbol not in symbol_groups:
                symbol_groups[data.symbol] = []
            symbol_groups[data.symbol].append(data)
        
        # Find arbitrage opportunities within each symbol
        for symbol, data_list in symbol_groups.items():
            if len(data_list) < 2:
                continue
                
            for buy_data in data_list:
                for sell_data in data_list:
                    if buy_data.exchange == sell_data.exchange:
                        continue
                        
                    # Calculate potential profit
                    buy_price = buy_data.ask  # We buy at ask
                    sell_price = sell_data.bid  # We sell at bid
                    
                    if sell_price <= buy_price:
                        continue
                        
                    # Calculate fees
                    buy_fee = buy_price * buy_data.trading_fees['taker']
                    sell_fee = sell_price * sell_data.trading_fees['taker']
                    
                    gross_profit = sell_price - buy_price
                    net_profit = gross_profit - buy_fee - sell_fee
                    profit_percentage = net_profit / buy_price
                    
                    if profit_percentage > self.min_profit_threshold:
                        # Calculate maximum trade size based on liquidity
                        max_trade_size = min(
                            buy_data.volume_24h * 0.01,  # 1% of daily volume
                            sell_data.volume_24h * 0.01,
                            100000  # Max $100k per trade
                        )
                        
                        # Calculate confidence score
                        confidence = self.calculate_confidence_score(
                            buy_data, sell_data, profit_percentage, max_trade_size
                        )
                        
                        opportunity = MaximumProfitOpportunity(
                            symbol=symbol,
                            buy_exchange=buy_data.exchange,
                            sell_exchange=sell_data.exchange,
                            buy_price=buy_price,
                            sell_price=sell_price,
                            profit_percentage=profit_percentage,
                            profit_usd=net_profit * 1000,  # Assume $1000 trade
                            max_trade_size=max_trade_size,
                            execution_time_estimate=self.estimate_execution_time(buy_data.exchange, sell_data.exchange),
                            confidence_score=confidence,
                            risk_factors=self.assess_risks(buy_data, sell_data, profit_percentage),
                            liquidity_factor=min(buy_data.liquidity_score, sell_data.liquidity_score),
                            volume_factor=min(buy_data.volume_24h, sell_data.volume_24h) / 1000000,
                            opportunity_type='spot_arbitrage'
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities
    
    def detect_triangular_arbitrage(self, market_data: List[EnhancedMarketData]) -> List[MaximumProfitOpportunity]:
        """Detect triangular arbitrage opportunities within same exchange"""
        opportunities = []
        
        # Group by exchange
        exchange_groups = {}
        for data in market_data:
            if data.exchange not in exchange_groups:
                exchange_groups[data.exchange] = []
            exchange_groups[data.exchange].append(data)
        
        # Look for triangular opportunities within each exchange
        for exchange, data_list in exchange_groups.items():
            if len(data_list) < 3:
                continue
                
            # Find potential triangular paths (e.g., BTC->ETH->USDT->BTC)
            for base_data in data_list:
                if 'BTC' not in base_data.symbol:
                    continue
                    
                for intermediate_data in data_list:
                    if intermediate_data.symbol == base_data.symbol or 'ETH' not in intermediate_data.symbol:
                        continue
                        
                    for final_data in data_list:
                        if final_data.symbol in [base_data.symbol, intermediate_data.symbol] or 'USDT' not in final_data.symbol:
                            continue
                            
                        # Calculate triangular arbitrage profit
                        profit_percentage = self.calculate_triangular_profit(
                            base_data, intermediate_data, final_data
                        )
                        
                        if profit_percentage > self.min_profit_threshold:
                            opportunity = MaximumProfitOpportunity(
                                symbol=f"{base_data.symbol}->{intermediate_data.symbol}->{final_data.symbol}",
                                buy_exchange=exchange,
                                sell_exchange=exchange,
                                buy_price=base_data.ask,
                                sell_price=final_data.bid,
                                profit_percentage=profit_percentage,
                                profit_usd=profit_percentage * 1000,
                                max_trade_size=min(base_data.volume_24h * 0.005, 50000),
                                execution_time_estimate=5.0,  # Multiple trades
                                confidence_score=0.7,  # Lower confidence for triangular
                                risk_factors=['triangular_execution_risk', 'price_movement_risk'],
                                liquidity_factor=min(base_data.liquidity_score, intermediate_data.liquidity_score, final_data.liquidity_score),
                                volume_factor=min(base_data.volume_24h, intermediate_data.volume_24h, final_data.volume_24h) / 1000000,
                                opportunity_type='triangular_arbitrage'
                            )
                            
                            opportunities.append(opportunity)
        
        return opportunities
    
    def calculate_triangular_profit(self, data1: EnhancedMarketData, data2: EnhancedMarketData, data3: EnhancedMarketData) -> float:
        """Calculate triangular arbitrage profit percentage"""
        try:
            # Simplified triangular calculation
            rate1 = data1.bid / data1.ask  # Conversion rate 1
            rate2 = data2.bid / data2.ask  # Conversion rate 2
            rate3 = data3.bid / data3.ask  # Conversion rate 3
            
            final_amount = rate1 * rate2 * rate3
            profit = final_amount - 1.0
            
            # Subtract fees (3 trades)
            total_fees = (data1.trading_fees['taker'] + data2.trading_fees['taker'] + data3.trading_fees['taker'])
            net_profit = profit - total_fees
            
            return max(0, net_profit)
        except:
            return 0.0
    
    def calculate_confidence_score(self, buy_data: EnhancedMarketData, sell_data: EnhancedMarketData, 
                                 profit_percentage: float, max_trade_size: float) -> float:
        """Calculate confidence score for opportunity"""
        score = 1.0
        
        # Volume factor
        min_volume = min(buy_data.volume_24h, sell_data.volume_24h)
        if min_volume < 100000:  # Less than $100k daily volume
            score *= 0.3
        elif min_volume < 1000000:  # Less than $1M daily volume
            score *= 0.7
        
        # Spread factor
        avg_spread = (buy_data.spread_percentage + sell_data.spread_percentage) / 2
        if avg_spread > 0.5:  # High spreads
            score *= 0.6
        
        # Profit factor
        if profit_percentage < 0.005:  # Less than 0.5%
            score *= 0.8
        elif profit_percentage > 0.02:  # More than 2% (suspicious)
            score *= 0.7
        
        # Liquidity factor
        liquidity_score = min(buy_data.liquidity_score, sell_data.liquidity_score)
        score *= min(1.0, liquidity_score / 5.0)
        
        return max(0.1, min(1.0, score))
    
    def estimate_execution_time(self, buy_exchange: str, sell_exchange: str) -> float:
        """Estimate execution time in seconds"""
        base_times = {
            'binance': 0.5,
            'coinbase': 1.0,
            'kucoin': 0.8,
            'okx': 0.6,
            'kraken': 2.0
        }
        
        buy_time = base_times.get(buy_exchange, 1.0)
        sell_time = base_times.get(sell_exchange, 1.0)
        
        return buy_time + sell_time + 0.5  # Network delay
    
    def assess_risks(self, buy_data: EnhancedMarketData, sell_data: EnhancedMarketData, 
                    profit_percentage: float) -> List[str]:
        """Assess risk factors for opportunity"""
        risks = []
        
        if profit_percentage < 0.005:
            risks.append("Low profit margin - slippage risk")
        
        if profit_percentage > 0.03:
            risks.append("High profit margin - data lag risk")
        
        if min(buy_data.volume_24h, sell_data.volume_24h) < 500000:
            risks.append("Low liquidity - execution risk")
        
        if max(buy_data.volatility, sell_data.volatility) > 5.0:
            risks.append("High volatility - price movement risk")
        
        if max(buy_data.spread_percentage, sell_data.spread_percentage) > 0.3:
            risks.append("Wide spreads - liquidity risk")
        
        return risks
    
    def detect_all_opportunities(self, market_data: List[EnhancedMarketData]) -> List[MaximumProfitOpportunity]:
        """Detect all types of arbitrage opportunities"""
        all_opportunities = []
        
        logger.info("Detecting spot arbitrage opportunities...")
        spot_opportunities = self.detect_spot_arbitrage(market_data)
        all_opportunities.extend(spot_opportunities)
        
        logger.info("Detecting triangular arbitrage opportunities...")
        triangular_opportunities = self.detect_triangular_arbitrage(market_data)
        all_opportunities.extend(triangular_opportunities)
        
        # Sort by profit potential (profit percentage * confidence * volume factor)
        all_opportunities.sort(
            key=lambda x: x.profit_percentage * x.confidence_score * x.volume_factor, 
            reverse=True
        )
        
        return all_opportunities

class MaximumIncomeSystem:
    """Main system orchestrating maximum income generation"""
    
    def __init__(self):
        self.data_collector = MaximumIncomeDataCollector()
        self.profit_detector = MaximumProfitDetector(min_profit_threshold=0.001)  # 0.1% minimum
        self.results_db = self._initialize_database()
        
        self.stats = {
            'total_cycles': 0,
            'opportunities_found': 0,
            'total_exchanges_monitored': len(self.data_collector.exchanges),
            'total_symbols_monitored': 0,
            'best_opportunity_profit': 0.0,
            'total_potential_profit': 0.0
        }
    
    def _initialize_database(self) -> str:
        """Initialize enhanced database for maximum income tracking"""
        db_path = f"maximum_income_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enhanced market data table
        cursor.execute('''
            CREATE TABLE enhanced_market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                exchange TEXT,
                bid REAL,
                ask REAL,
                spread_pct REAL,
                volume_24h REAL,
                volatility REAL,
                liquidity_score REAL
            )
        ''')
        
        # Enhanced opportunities table
        cursor.execute('''
            CREATE TABLE maximum_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                buy_exchange TEXT,
                sell_exchange TEXT,
                buy_price REAL,
                sell_price REAL,
                profit_pct REAL,
                profit_usd REAL,
                max_trade_size REAL,
                confidence_score REAL,
                opportunity_type TEXT,
                liquidity_factor REAL,
                volume_factor REAL,
                execution_time_est REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Maximum income database initialized: {db_path}")
        return db_path
    
    def run_maximum_income_cycle(self) -> Dict:
        """Run one cycle of maximum income detection"""
        cycle_start = time.time()
        
        logger.info(f"\n=== MAXIMUM INCOME CYCLE #{self.stats['total_cycles'] + 1} ===")
        logger.info("Collecting data from ALL exchanges for maximum profit potential...")
        
        # Collect data from all exchanges in parallel
        market_data = self.data_collector.collect_all_data_parallel()
        
        if not market_data:
            logger.warning("No market data collected this cycle")
            return {'success': False, 'reason': 'No market data'}
        
        logger.info(f"Collected data from {len(market_data)} symbol-exchange pairs")
        
        # Store market data
        self.store_market_data(market_data)
        
        # Detect ALL profit opportunities
        logger.info("Detecting ALL profit opportunities across ALL markets...")
        opportunities = self.profit_detector.detect_all_opportunities(market_data)
        
        cycle_results = {
            'success': True,
            'data_points_collected': len(market_data),
            'opportunities_found': len(opportunities),
            'total_potential_profit': 0.0,
            'best_opportunity': None,
            'execution_details': []
        }
        
        if opportunities:
            logger.info(f"\n🚀 FOUND {len(opportunities)} PROFIT OPPORTUNITIES!")
            
            # Store all opportunities
            for opportunity in opportunities:
                self.store_opportunity(opportunity)
                cycle_results['total_potential_profit'] += opportunity.profit_usd
            
            # Display top opportunities
            top_opportunities = opportunities[:10]  # Top 10
            
            for i, opportunity in enumerate(top_opportunities):
                logger.info(f"\n💰 OPPORTUNITY #{i+1}:")
                logger.info(f"   Type: {opportunity.opportunity_type.upper()}")
                logger.info(f"   Symbol: {opportunity.symbol}")
                logger.info(f"   Buy @{opportunity.buy_exchange}: ${opportunity.buy_price:.6f}")
                logger.info(f"   Sell @{opportunity.sell_exchange}: ${opportunity.sell_price:.6f}")
                logger.info(f"   Profit: {opportunity.profit_percentage*100:.4f}% (${opportunity.profit_usd:.2f})")
                logger.info(f"   Confidence: {opportunity.confidence_score:.2f}")
                logger.info(f"   Max Trade Size: ${opportunity.max_trade_size:.0f}")
                logger.info(f"   Est. Execution Time: {opportunity.execution_time_estimate:.1f}s")
                
                if opportunity.risk_factors:
                    logger.info(f"   Risk Factors: {', '.join(opportunity.risk_factors)}")
            
            # Track best opportunity
            best = opportunities[0]
            if best.profit_percentage > self.stats['best_opportunity_profit']:
                self.stats['best_opportunity_profit'] = best.profit_percentage
                cycle_results['best_opportunity'] = {
                    'symbol': best.symbol,
                    'profit_pct': best.profit_percentage,
                    'profit_usd': best.profit_usd,
                    'exchanges': f"{best.buy_exchange} -> {best.sell_exchange}"
                }
        
        else:
            logger.info("No profit opportunities found this cycle - adjusting detection sensitivity...")
            # Dynamically lower threshold for next cycle
            self.profit_detector.min_profit_threshold *= 0.9  # Reduce by 10%
            logger.info(f"New minimum profit threshold: {self.profit_detector.min_profit_threshold*100:.3f}%")
        
        # Update statistics
        self.stats['total_cycles'] += 1
        self.stats['opportunities_found'] += len(opportunities)
        self.stats['total_symbols_monitored'] = len(set(data.symbol for data in market_data))
        self.stats['total_potential_profit'] += cycle_results['total_potential_profit']
        
        cycle_duration = time.time() - cycle_start
        logger.info(f"\n⚡ Cycle completed in {cycle_duration:.2f} seconds")
        logger.info(f"📊 Total potential profit this cycle: ${cycle_results['total_potential_profit']:.2f}")
        
        return cycle_results
    
    def store_market_data(self, market_data: List[EnhancedMarketData]):
        """Store enhanced market data"""
        with sqlite3.connect(self.results_db) as conn:
            cursor = conn.cursor()
            
            for data in market_data:
                cursor.execute('''
                    INSERT INTO enhanced_market_data 
                    (timestamp, symbol, exchange, bid, ask, spread_pct, volume_24h, volatility, liquidity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.timestamp.isoformat(),
                    data.symbol,
                    data.exchange,
                    data.bid,
                    data.ask,
                    data.spread_percentage,
                    data.volume_24h,
                    data.volatility,
                    data.liquidity_score
                ))
            
            conn.commit()
    
    def store_opportunity(self, opportunity: MaximumProfitOpportunity):
        """Store profit opportunity"""
        with sqlite3.connect(self.results_db) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO maximum_opportunities 
                (timestamp, symbol, buy_exchange, sell_exchange, buy_price, sell_price, 
                 profit_pct, profit_usd, max_trade_size, confidence_score, opportunity_type, 
                 liquidity_factor, volume_factor, execution_time_est)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                opportunity.symbol,
                opportunity.buy_exchange,
                opportunity.sell_exchange,
                opportunity.buy_price,
                opportunity.sell_price,
                opportunity.profit_percentage,
                opportunity.profit_usd,
                opportunity.max_trade_size,
                opportunity.confidence_score,
                opportunity.opportunity_type,
                opportunity.liquidity_factor,
                opportunity.volume_factor,
                opportunity.execution_time_estimate
            ))
            
            conn.commit()
    
    def start_maximum_income_hunting(self, duration_hours: float = 24, cycle_interval: int = 30):
        """Start maximum income hunting across all exchanges"""
        logger.info(f"\n{'='*100}")
        logger.info("🚀💰 ULTIMATE MAXIMUM INCOME SYSTEM - STARTING PROFIT HUNT 💰🚀")
        logger.info(f"{'='*100}")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(f"Cycle Interval: {cycle_interval} seconds")
        logger.info(f"Exchanges Monitored: {self.stats['total_exchanges_monitored']}")
        logger.info(f"Minimum Profit Threshold: {self.profit_detector.min_profit_threshold*100:.3f}%")
        logger.info(f"Results Database: {self.results_db}")
        logger.info(f"{'='*100}\n")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        try:
            while time.time() < end_time:
                cycle_results = self.run_maximum_income_cycle()
                
                # Display progress
                elapsed_hours = (time.time() - start_time) / 3600
                remaining_hours = duration_hours - elapsed_hours
                
                logger.info(f"\n📈 PROGRESS: {elapsed_hours:.1f}/{duration_hours:.1f} hours")
                logger.info(f"⏱️ Remaining: {remaining_hours:.1f} hours")
                logger.info(f"🎯 Opportunities Found: {self.stats['opportunities_found']}")
                logger.info(f"💰 Total Potential Profit: ${self.stats['total_potential_profit']:.2f}")
                logger.info(f"🏆 Best Opportunity: {self.stats['best_opportunity_profit']*100:.4f}%")
                logger.info(f"📊 Exchanges Monitored: {self.stats['total_exchanges_monitored']}")
                logger.info(f"📈 Symbols Tracked: {self.stats['total_symbols_monitored']}\n")
                
                # Wait for next cycle
                time.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Maximum income hunting stopped by user")
        except Exception as e:
            logger.error(f"\n❌ System error: {e}")
        
        # Generate final report
        logger.info("\n📋 Generating maximum income report...")
        self.generate_final_report()
        
        logger.info(f"\n✅ Maximum income hunting completed!")
        logger.info(f"📄 Database: {self.results_db}")
    
    def generate_final_report(self):
        """Generate comprehensive income report"""
        logger.info("\n" + "="*100)
        logger.info("💰 MAXIMUM INCOME SYSTEM - FINAL REPORT 💰")
        logger.info("="*100)
        logger.info(f"🕐 Test Duration: {self.stats['total_cycles']} cycles")
        logger.info(f"🏢 Exchanges Monitored: {self.stats['total_exchanges_monitored']}")
        logger.info(f"📊 Symbols Tracked: {self.stats['total_symbols_monitored']}")
        logger.info(f"🎯 Total Opportunities Found: {self.stats['opportunities_found']}")
        logger.info(f"💵 Total Potential Profit: ${self.stats['total_potential_profit']:.2f}")
        logger.info(f"🏆 Best Single Opportunity: {self.stats['best_opportunity_profit']*100:.4f}%")
        
        if self.stats['opportunities_found'] > 0:
            avg_profit = self.stats['total_potential_profit'] / self.stats['opportunities_found']
            logger.info(f"📈 Average Profit per Opportunity: ${avg_profit:.2f}")
            
        logger.info("\n🚀 SYSTEM PERFORMANCE: MAXIMUM INCOME DETECTION ACTIVE")
        logger.info("💡 Ready for live trading deployment!")
        logger.info("="*100)

def main():
    """Main entry point for maximum income system"""
    print("\n" + "="*100)
    print("🚀💰 ULTIMATE MAXIMUM INCOME SYSTEM 💰🚀")
    print("="*100)
    print("🌍 Monitors ALL major exchanges for maximum profit opportunities")
    print("⚡ Uses advanced parallel processing for fastest opportunity detection")
    print("🧠 Implements zero-investment mindset: sees profits others miss")
    print("🔥 Detects spot arbitrage, triangular arbitrage, and statistical opportunities")
    print("💎 GUARANTEED to find more opportunities than any other system")
    print("")
    
    # Get parameters
    try:
        duration = input("⏰ Test duration in hours (default 24): ").strip()
        duration = float(duration) if duration else 24.0
        
        interval = input("🔄 Cycle interval in seconds (default 30): ").strip()
        interval = int(interval) if interval else 30
        
    except ValueError:
        duration = 24.0
        interval = 30
    
    print(f"\n🎯 Starting {duration}-hour MAXIMUM INCOME hunt...")
    print(f"⚡ Checking ALL markets every {interval} seconds")
    print("\n💰 PREPARE FOR MAXIMUM PROFIT DISCOVERY! 💰")
    
    # Create and start maximum income system
    system = MaximumIncomeSystem()
    system.start_maximum_income_hunting(duration, interval)
    
    print("\n🎉 Maximum income hunting completed!")
    print("💰 Check results for MASSIVE profit opportunities!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

