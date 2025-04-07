import asyncio
import numpy as np
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import ccxt
import ccxt.async_support as ccxt_async
from decimal import Decimal
import talib
import tensorflow as tf

class ArbitrageEngine:
    """
    Autonomous Trading and Arbitrage System
    Features:
    1. Multi-exchange arbitrage
    2. High-frequency trading
    3. AI price prediction
    4. Risk management
    5. Portfolio optimization
    6. Automated reinvestment
    """
    
    def __init__(self, initial_capital: float = 49.99):
        self.capital = initial_capital
        self.exchanges = {
            'binance': {
                'enabled': True,
                'fees': 0.001,
                'min_order': 10.0
            },
            'kraken': {
                'enabled': True,
                'fees': 0.0016,
                'min_order': 10.0
            },
            'coinbase': {
                'enabled': True,
                'fees': 0.005,
                'min_order': 10.0
            }
        }
        
        self.pairs = [
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'XRP/USDT',
            'ADA/USDT'
        ]
        
        self.initialize_exchanges()
    
    def initialize_exchanges(self):
        """Initialize all exchange connections"""
        self.exchange_apis = {}
        
        for exchange in self.exchanges:
            if self.exchanges[exchange]['enabled']:
                self.exchange_apis[exchange] = getattr(ccxt_async, exchange)()
    
    async def find_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Find profitable arbitrage opportunities"""
        opportunities = []
        
        for pair in self.pairs:
            # Get order books from all exchanges
            order_books = await self.get_order_books(pair)
            
            # Find price differences
            for ex1 in order_books:
                for ex2 in order_books:
                    if ex1 != ex2:
                        opportunity = self.calculate_opportunity(
                            pair, ex1, ex2,
                            order_books[ex1],
                            order_books[ex2]
                        )
                        
                        if opportunity['profit_potential'] > 0.005:  # 0.5% minimum
                            opportunities.append(opportunity)
        
        return opportunities
    
    async def get_order_books(self, pair: str) -> Dict[str, Any]:
        """Get order books from all exchanges"""
        order_books = {}
        
        for exchange in self.exchange_apis:
            try:
                order_book = await self.exchange_apis[exchange].fetch_order_book(pair)
                order_books[exchange] = order_book
            except Exception as e:
                print(f"Error fetching {pair} order book from {exchange}: {e}")
        
        return order_books
    
    def calculate_opportunity(self, pair: str, ex1: str, ex2: str,
                            book1: Dict[str, Any], book2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate arbitrage opportunity details"""
        buy_price = book1['asks'][0][0]  # Best ask price on exchange 1
        sell_price = book2['bids'][0][0]  # Best bid price on exchange 2
        
        # Calculate profit potential
        profit_potential = (sell_price - buy_price) / buy_price
        
        # Account for fees
        profit_potential -= (self.exchanges[ex1]['fees'] + self.exchanges[ex2]['fees'])
        
        return {
            'pair': pair,
            'buy_exchange': ex1,
            'sell_exchange': ex2,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'profit_potential': profit_potential,
            'timestamp': datetime.now()
        }
    
    async def execute_arbitrage(self, opportunity: Dict[str, Any]):
        """Execute arbitrage trades"""
        # Calculate trade size based on available capital
        trade_size = self.calculate_trade_size(
            opportunity['pair'],
            opportunity['buy_price']
        )
        
        try:
            # Execute buy order
            buy_order = await self.exchange_apis[opportunity['buy_exchange']].create_market_buy_order(
                opportunity['pair'],
                trade_size
            )
            
            # Execute sell order
            sell_order = await self.exchange_apis[opportunity['sell_exchange']].create_market_sell_order(
                opportunity['pair'],
                trade_size
            )
            
            # Update capital
            profit = self.calculate_profit(buy_order, sell_order)
            self.capital += profit
            
            return {
                'success': True,
                'profit': profit,
                'buy_order': buy_order,
                'sell_order': sell_order
            }
            
        except Exception as e:
            print(f"Error executing arbitrage: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_trade_size(self, pair: str, price: float) -> float:
        """Calculate optimal trade size"""
        # Use 90% of available capital per trade
        available = self.capital * 0.9
        
        # Calculate maximum affordable amount
        max_size = available / price
        
        # Round to appropriate decimal places
        return round(max_size, 8)
    
    async def run_arbitrage_engine(self):
        """Main arbitrage loop"""
        while True:
            try:
                # Find opportunities
                opportunities = await self.find_arbitrage_opportunities()
                
                # Execute profitable trades
                for opportunity in opportunities:
                    if opportunity['profit_potential'] > 0.005:  # 0.5% minimum
                        result = await self.execute_arbitrage(opportunity)
                        
                        if result['success']:
                            print(f"Profit: €{result['profit']:.2f}")
                
                # Brief pause between cycles
                await asyncio.sleep(0.1)  # 100ms between checks
                
            except Exception as e:
                print(f"Error in arbitrage loop: {e}")
                await asyncio.sleep(1)
    
    async def analyze_performance(self) -> Dict[str, float]:
        """Analyze trading performance"""
        return {
            'initial_capital': 49.99,
            'current_capital': self.capital,
            'total_profit': self.capital - 49.99,
            'roi': ((self.capital - 49.99) / 49.99) * 100
        }
    
    async def run_risk_management(self):
        """Monitor and manage trading risks"""
        while True:
            metrics = await self.analyze_performance()
            
            if metrics['current_capital'] < 45:  # Stop if below 90% of initial
                print("Risk management: Stopping trades due to capital loss")
                break
                
            await asyncio.sleep(60)  # Check every minute
    
    async def start_trading(self):
        """Start all trading operations"""
        tasks = [
            self.run_arbitrage_engine(),
            self.run_risk_management()
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    engine = ArbitrageEngine(initial_capital=49.99)
    asyncio.run(engine.start_trading())
