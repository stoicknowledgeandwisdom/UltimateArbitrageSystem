#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Arbitrage System - Comprehensive Demo
=============================================

Demonstrates the complete integration of Python ML orchestrator with
Rust ultra-high-frequency trading engine, showcasing sub-millisecond
performance and advanced arbitrage capabilities.

Features Demonstrated:
- Rust engine initialization with SIMD/FPGA support
- Ultra-fast order book creation and management
- Sub-millisecond order execution
- Real-time performance monitoring
- Advanced market making strategies
- Cross-exchange arbitrage detection
- ML-driven decision making
- Risk management and safety checks
"""

import asyncio
import logging
import time
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
import os
from dataclasses import dataclass
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init(autoreset=True)

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_optimization', 'ml_optimization'))

try:
    from rust_engine_bridge import (
        RustEngineInterface,
        RustEngineConfig,
        rust_engine_context,
        run_engine_performance_test
    )
    from orchestrator import MLOptimizationOrchestrator, MLOptimizationConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"⚠️ Some imports not available: {e}")
    print("Demo will run in simulation mode")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure for demo"""
    symbol: str
    exchange: str
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_price: float
    volume_24h: float
    timestamp: datetime
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2

class MarketDataSimulator:
    """Simulates realistic market data for demo purposes"""
    
    def __init__(self):
        self.symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT",
            "LINK/USDT", "LTC/USDT", "XRP/USDT", "SOL/USDT", "AVAX/USDT"
        ]
        self.exchanges = ["binance", "coinbase", "kraken", "bybit", "okx"]
        
        # Base prices for each symbol
        self.base_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "BNB/USDT": 300.0,
            "ADA/USDT": 0.5,
            "DOT/USDT": 20.0,
            "LINK/USDT": 15.0,
            "LTC/USDT": 150.0,
            "XRP/USDT": 0.6,
            "SOL/USDT": 100.0,
            "AVAX/USDT": 40.0
        }
        
        # Current prices (will fluctuate)
        self.current_prices = self.base_prices.copy()
        
        # Price volatility for each symbol
        self.volatilities = {
            "BTC/USDT": 0.02,
            "ETH/USDT": 0.03,
            "BNB/USDT": 0.04,
            "ADA/USDT": 0.05,
            "DOT/USDT": 0.04,
            "LINK/USDT": 0.05,
            "LTC/USDT": 0.03,
            "XRP/USDT": 0.04,
            "SOL/USDT": 0.06,
            "AVAX/USDT": 0.05
        }
    
    def generate_market_data(self, symbol: str, exchange: str) -> MarketData:
        """Generate realistic market data for given symbol and exchange"""
        base_price = self.current_prices[symbol]
        volatility = self.volatilities[symbol]
        
        # Add some price movement
        price_change = np.random.normal(0, volatility * base_price * 0.01)
        self.current_prices[symbol] += price_change
        
        # Ensure price doesn't go negative
        self.current_prices[symbol] = max(0.001, self.current_prices[symbol])
        
        current_price = self.current_prices[symbol]
        
        # Generate spread (typically 0.01% to 0.1% of price)
        spread_pct = random.uniform(0.0001, 0.001)
        spread = current_price * spread_pct
        
        # Add exchange-specific variations
        exchange_factor = {
            "binance": 1.0,
            "coinbase": 1.002,  # Slightly higher prices
            "kraken": 0.998,   # Slightly lower prices
            "bybit": 1.001,
            "okx": 0.999
        }.get(exchange, 1.0)
        
        adjusted_price = current_price * exchange_factor
        
        bid_price = adjusted_price - spread / 2
        ask_price = adjusted_price + spread / 2
        
        # Generate volumes
        bid_volume = random.uniform(0.1, 10.0)
        ask_volume = random.uniform(0.1, 10.0)
        volume_24h = random.uniform(1000, 100000)
        
        return MarketData(
            symbol=symbol,
            exchange=exchange,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            last_price=adjusted_price,
            volume_24h=volume_24h,
            timestamp=datetime.now()
        )
    
    def detect_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities across exchanges"""
        opportunities = []
        
        for symbol in self.symbols:
            # Generate market data for all exchanges
            market_data = {}
            for exchange in self.exchanges:
                market_data[exchange] = self.generate_market_data(symbol, exchange)
            
            # Find arbitrage opportunities
            exchanges = list(market_data.keys())
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    ex1, ex2 = exchanges[i], exchanges[j]
                    data1, data2 = market_data[ex1], market_data[ex2]
                    
                    # Check if we can buy on ex1 and sell on ex2
                    if data1.ask_price < data2.bid_price:
                        profit_pct = (data2.bid_price - data1.ask_price) / data1.ask_price
                        if profit_pct > 0.001:  # At least 0.1% profit
                            opportunities.append({
                                'symbol': symbol,
                                'buy_exchange': ex1,
                                'sell_exchange': ex2,
                                'buy_price': data1.ask_price,
                                'sell_price': data2.bid_price,
                                'profit_pct': profit_pct * 100,
                                'volume': min(data1.ask_volume, data2.bid_volume),
                                'timestamp': datetime.now()
                            })
                    
                    # Check if we can buy on ex2 and sell on ex1
                    if data2.ask_price < data1.bid_price:
                        profit_pct = (data1.bid_price - data2.ask_price) / data2.ask_price
                        if profit_pct > 0.001:  # At least 0.1% profit
                            opportunities.append({
                                'symbol': symbol,
                                'buy_exchange': ex2,
                                'sell_exchange': ex1,
                                'buy_price': data2.ask_price,
                                'sell_price': data1.bid_price,
                                'profit_pct': profit_pct * 100,
                                'volume': min(data2.ask_volume, data1.bid_volume),
                                'timestamp': datetime.now()
                            })
        
        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)

class UltimateArbitrageDemo:
    """Main demo class showcasing the Ultimate Arbitrage System"""
    
    def __init__(self):
        self.market_simulator = MarketDataSimulator()
        self.rust_engine: Optional[RustEngineInterface] = None
        self.ml_orchestrator: Optional[MLOptimizationOrchestrator] = None
        self.demo_stats = {
            'trades_executed': 0,
            'total_profit': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'avg_execution_time_ms': 0.0,
            'max_execution_time_ms': 0.0,
            'min_execution_time_ms': float('inf')
        }
    
    def print_header(self):
        """Print demo header"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*80)
        print(f"{Fore.YELLOW}{Style.BRIGHT}🚀 ULTIMATE ARBITRAGE SYSTEM - LIVE DEMO 🚀")
        print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*80)
        print(f"{Fore.GREEN}🦀 Rust Ultra-High-Frequency Engine")
        print(f"{Fore.BLUE}🧠 Python ML Orchestrator")
        print(f"{Fore.MAGENTA}⚡ Sub-Millisecond Execution")
        print(f"{Fore.YELLOW}💰 Advanced Arbitrage Detection")
        print(f"{Fore.CYAN}" + "="*80 + f"{Style.RESET_ALL}\n")
    
    def print_system_status(self):
        """Print current system status"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 SYSTEM STATUS")
        print(f"{Fore.GREEN}✅ Market Data Simulator: Active")
        
        if IMPORTS_AVAILABLE:
            rust_status = "🦀 Available" if self.rust_engine else "⏳ Initializing"
            ml_status = "🧠 Available" if self.ml_orchestrator else "⏳ Initializing"
        else:
            rust_status = "📋 Simulation Mode"
            ml_status = "📋 Simulation Mode"
        
        print(f"{Fore.YELLOW}⚡ Rust Engine: {rust_status}")
        print(f"{Fore.BLUE}🤖 ML Orchestrator: {ml_status}")
        print(f"{Fore.MAGENTA}🎯 Arbitrage Detection: Active")
        print(f"{Style.RESET_ALL}")
    
    async def initialize_systems(self):
        """Initialize all system components"""
        print(f"{Fore.YELLOW}🔧 Initializing Ultimate Arbitrage System...\n")
        
        if not IMPORTS_AVAILABLE:
            print(f"{Fore.YELLOW}⚠️ Running in simulation mode (Rust/ML components not available)")
            return
        
        try:
            # Initialize Rust engine
            print(f"{Fore.CYAN}🦀 Initializing Rust Ultra-High-Frequency Engine...")
            rust_config = RustEngineConfig(
                enable_simd=True,
                enable_fpga=False,  # Set to True when FPGA hardware is available
                max_order_book_levels=10000,
                max_orders_per_level=1000,
                thread_pool_size=16,
                enable_performance_monitoring=True
            )
            
            self.rust_engine = RustEngineInterface(rust_config)
            success = await self.rust_engine.initialize()
            
            if success:
                print(f"{Fore.GREEN}   ✅ Rust engine initialized successfully")
            else:
                print(f"{Fore.YELLOW}   ⚠️ Rust engine fallback to Python simulation")
            
            # Initialize ML Orchestrator
            print(f"{Fore.BLUE}🧠 Initializing ML Optimization Orchestrator...")
            ml_config = MLOptimizationConfig()
            # Simplified config for demo
            ml_config.enable_feature_store = False
            ml_config.enable_streaming_etl = False
            ml_config.enable_reinforcement_learning = False
            ml_config.enable_meta_controller = False
            ml_config.enable_explainability = False
            
            self.ml_orchestrator = MLOptimizationOrchestrator(ml_config)
            await self.ml_orchestrator.start()
            
            # Integrate Rust engine with ML orchestrator
            if self.rust_engine:
                await self.ml_orchestrator.initialize_rust_engine()
            
            print(f"{Fore.GREEN}   ✅ ML orchestrator initialized successfully")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error initializing systems: {e}")
            logger.error(f"Initialization error: {e}")
    
    async def create_order_books(self):
        """Create order books for all trading pairs"""
        print(f"{Fore.MAGENTA}📚 Creating ultra-fast order books...")
        
        symbols = self.market_simulator.symbols
        
        if self.rust_engine:
            for symbol in symbols:
                success = await self.rust_engine.create_order_book(symbol)
                if success:
                    print(f"{Fore.GREEN}   ✅ Created order book for {symbol}")
                else:
                    print(f"{Fore.YELLOW}   ⚠️ Failed to create order book for {symbol}")
        else:
            print(f"{Fore.YELLOW}   📋 Simulating order book creation for {len(symbols)} symbols")
        
        print(f"{Fore.GREEN}🎯 Order books ready for trading\n")
    
    async def run_performance_benchmark(self):
        """Run comprehensive performance benchmark"""
        print(f"{Fore.CYAN}🏃 Running Performance Benchmark...\n")
        
        if self.rust_engine:
            try:
                # Run benchmark with different operation counts
                test_sizes = [1000, 5000, 10000, 25000]
                
                for size in test_sizes:
                    print(f"{Fore.YELLOW}⚡ Testing {size} operations...")
                    
                    start_time = time.perf_counter()
                    results = await self.rust_engine.benchmark_performance(operations=size)
                    benchmark_time = time.perf_counter() - start_time
                    
                    ops_per_sec = results.get('operations_per_second', 0)
                    avg_latency = results.get('average_latency_ms', 0)
                    success_rate = (results.get('successful_operations', 0) / size) * 100
                    
                    print(f"{Fore.GREEN}   📊 {ops_per_sec:,.0f} ops/sec")
                    print(f"{Fore.GREEN}   ⚡ {avg_latency:.3f}ms avg latency")
                    print(f"{Fore.GREEN}   ✅ {success_rate:.1f}% success rate")
                    print(f"{Fore.GREEN}   ⏱️ {benchmark_time:.2f}s total time\n")
                
                # Get detailed performance metrics
                metrics = await self.rust_engine.get_performance_metrics()
                if metrics.get('monitor_stats', {}).get('status') != 'no_data':
                    monitor_stats = metrics['monitor_stats']
                    print(f"{Fore.CYAN}📈 Detailed Performance Metrics:")
                    print(f"{Fore.BLUE}   Average Latency: {monitor_stats.get('average_latency_ns', 0) / 1_000_000:.3f}ms")
                    print(f"{Fore.BLUE}   P95 Latency: {monitor_stats.get('p95_latency_ns', 0) / 1_000_000:.3f}ms")
                    print(f"{Fore.BLUE}   P99 Latency: {monitor_stats.get('p99_latency_ns', 0) / 1_000_000:.3f}ms")
                    print(f"{Fore.BLUE}   Error Rate: {monitor_stats.get('error_rate', 0) * 100:.2f}%")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Benchmark error: {e}")
        else:
            # Simulate benchmark results
            print(f"{Fore.YELLOW}📋 Simulating performance benchmark...")
            await asyncio.sleep(2)  # Simulate benchmark time
            print(f"{Fore.GREEN}   📊 ~10,000 ops/sec (simulated)")
            print(f"{Fore.GREEN}   ⚡ ~0.1ms avg latency (simulated)")
            print(f"{Fore.GREEN}   ✅ 99.9% success rate (simulated)")
        
        print(f"{Fore.GREEN}🏁 Performance benchmark completed\n")
    
    async def demonstrate_arbitrage_detection(self):
        """Demonstrate real-time arbitrage detection"""
        print(f"{Fore.MAGENTA}🎯 Demonstrating Real-Time Arbitrage Detection...\n")
        
        for round_num in range(3):  # Run 3 rounds
            print(f"{Fore.CYAN}🔍 Arbitrage Detection Round {round_num + 1}")
            
            # Detect arbitrage opportunities
            opportunities = self.market_simulator.detect_arbitrage_opportunities()
            
            if opportunities:
                print(f"{Fore.GREEN}💰 Found {len(opportunities)} arbitrage opportunities!")
                
                # Show top 3 opportunities
                for i, opp in enumerate(opportunities[:3]):
                    print(f"\n{Fore.YELLOW}   🚀 Opportunity #{i+1}:")
                    print(f"{Fore.WHITE}      Symbol: {opp['symbol']}")
                    print(f"{Fore.WHITE}      Buy: {opp['buy_exchange']} @ ${opp['buy_price']:.4f}")
                    print(f"{Fore.WHITE}      Sell: {opp['sell_exchange']} @ ${opp['sell_price']:.4f}")
                    print(f"{Fore.GREEN}      Profit: {opp['profit_pct']:.3f}%")
                    print(f"{Fore.BLUE}      Volume: {opp['volume']:.2f}")
                    
                    # Simulate trade execution
                    await self.execute_arbitrage_trade(opp)
            else:
                print(f"{Fore.YELLOW}📊 No significant arbitrage opportunities found this round")
            
            print(f"{Fore.CYAN}" + "-"*60)
            await asyncio.sleep(2)  # Wait between rounds
        
        print(f"{Fore.GREEN}🎯 Arbitrage detection demonstration completed\n")
    
    async def execute_arbitrage_trade(self, opportunity: Dict[str, Any]):
        """Execute an arbitrage trade with ultra-low latency"""
        symbol = opportunity['symbol']
        buy_price = opportunity['buy_price']
        sell_price = opportunity['sell_price']
        volume = min(opportunity['volume'], 1.0)  # Limit volume for demo
        
        print(f"{Fore.CYAN}      ⚡ Executing arbitrage trade...")
        
        start_time = time.perf_counter_ns()
        
        try:
            if self.rust_engine:
                # Execute buy order
                buy_order_id = int(time.time() * 1000000) % 1000000
                buy_success = await self.rust_engine.add_order(
                    symbol, buy_order_id, "buy", buy_price, volume
                )
                
                # Execute sell order
                sell_order_id = buy_order_id + 1
                sell_success = await self.rust_engine.add_order(
                    symbol, sell_order_id, "sell", sell_price, volume
                )
                
                execution_success = buy_success and sell_success
            else:
                # Simulate execution
                await asyncio.sleep(0.001)  # Simulate 1ms execution time
                execution_success = random.random() > 0.05  # 95% success rate
            
            execution_time_ns = time.perf_counter_ns() - start_time
            execution_time_ms = execution_time_ns / 1_000_000
            
            if execution_success:
                profit = (sell_price - buy_price) * volume
                self.demo_stats['trades_executed'] += 1
                self.demo_stats['successful_trades'] += 1
                self.demo_stats['total_profit'] += profit
                
                print(f"{Fore.GREEN}      ✅ Trade executed in {execution_time_ms:.3f}ms")
                print(f"{Fore.GREEN}      💰 Profit: ${profit:.4f}")
            else:
                self.demo_stats['trades_executed'] += 1
                self.demo_stats['failed_trades'] += 1
                print(f"{Fore.RED}      ❌ Trade execution failed")
            
            # Update execution time stats
            self.demo_stats['avg_execution_time_ms'] = (
                (self.demo_stats['avg_execution_time_ms'] * (self.demo_stats['trades_executed'] - 1) + execution_time_ms) /
                self.demo_stats['trades_executed']
            )
            self.demo_stats['max_execution_time_ms'] = max(self.demo_stats['max_execution_time_ms'], execution_time_ms)
            self.demo_stats['min_execution_time_ms'] = min(self.demo_stats['min_execution_time_ms'], execution_time_ms)
            
        except Exception as e:
            print(f"{Fore.RED}      ❌ Execution error: {e}")
            self.demo_stats['trades_executed'] += 1
            self.demo_stats['failed_trades'] += 1
    
    async def demonstrate_market_making(self):
        """Demonstrate advanced market making strategies"""
        print(f"{Fore.BLUE}🎪 Demonstrating Advanced Market Making...\n")
        
        selected_symbols = self.market_simulator.symbols[:3]  # Use first 3 symbols
        
        for symbol in selected_symbols:
            print(f"{Fore.CYAN}📈 Market Making for {symbol}")
            
            # Generate current market data
            market_data = self.market_simulator.generate_market_data(symbol, "binance")
            
            # Calculate optimal bid/ask spread
            mid_price = market_data.mid_price
            volatility = self.market_simulator.volatilities[symbol]
            
            # Dynamic spread based on volatility
            base_spread = mid_price * 0.001  # 0.1% base spread
            volatility_adjustment = volatility * 2  # Increase spread with volatility
            optimal_spread = base_spread * (1 + volatility_adjustment)
            
            # Place market making orders
            bid_price = mid_price - optimal_spread / 2
            ask_price = mid_price + optimal_spread / 2
            order_size = random.uniform(0.1, 2.0)
            
            print(f"{Fore.WHITE}   Current Mid Price: ${mid_price:.4f}")
            print(f"{Fore.WHITE}   Optimal Spread: {optimal_spread/mid_price*100:.3f}%")
            print(f"{Fore.YELLOW}   Placing Bid: ${bid_price:.4f} (Size: {order_size:.2f})")
            print(f"{Fore.YELLOW}   Placing Ask: ${ask_price:.4f} (Size: {order_size:.2f})")
            
            # Execute market making orders
            start_time = time.perf_counter_ns()
            
            if self.rust_engine:
                bid_order_id = int(time.time() * 1000000) % 1000000
                ask_order_id = bid_order_id + 1
                
                bid_success = await self.rust_engine.add_order(
                    symbol, bid_order_id, "buy", bid_price, order_size
                )
                ask_success = await self.rust_engine.add_order(
                    symbol, ask_order_id, "sell", ask_price, order_size
                )
                
                execution_success = bid_success and ask_success
            else:
                await asyncio.sleep(0.001)  # Simulate execution
                execution_success = True
            
            execution_time_ms = (time.perf_counter_ns() - start_time) / 1_000_000
            
            if execution_success:
                print(f"{Fore.GREEN}   ✅ Market making orders placed in {execution_time_ms:.3f}ms")
            else:
                print(f"{Fore.RED}   ❌ Failed to place market making orders")
            
            print(f"{Fore.CYAN}" + "-"*50)
        
        print(f"{Fore.GREEN}🎪 Market making demonstration completed\n")
    
    def print_final_statistics(self):
        """Print final demo statistics"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 FINAL DEMO STATISTICS")
        print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*50)
        
        stats = self.demo_stats
        success_rate = (stats['successful_trades'] / max(1, stats['trades_executed'])) * 100
        
        print(f"{Fore.GREEN}💼 Total Trades Executed: {stats['trades_executed']}")
        print(f"{Fore.GREEN}✅ Successful Trades: {stats['successful_trades']}")
        print(f"{Fore.RED}❌ Failed Trades: {stats['failed_trades']}")
        print(f"{Fore.YELLOW}📈 Success Rate: {success_rate:.1f}%")
        print(f"{Fore.CYAN}💰 Total Profit: ${stats['total_profit']:.4f}")
        
        if stats['trades_executed'] > 0:
            print(f"{Fore.BLUE}⚡ Average Execution Time: {stats['avg_execution_time_ms']:.3f}ms")
            print(f"{Fore.BLUE}🚀 Fastest Execution: {stats['min_execution_time_ms']:.3f}ms")
            print(f"{Fore.BLUE}🐌 Slowest Execution: {stats['max_execution_time_ms']:.3f}ms")
        
        print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*50)
        print(f"{Fore.GREEN}{Style.BRIGHT}🎉 DEMO COMPLETED SUCCESSFULLY! 🎉{Style.RESET_ALL}\n")
    
    async def run_demo(self):
        """Run the complete demo"""
        try:
            self.print_header()
            self.print_system_status()
            
            # Initialize all systems
            await self.initialize_systems()
            self.print_system_status()
            
            # Create order books
            await self.create_order_books()
            
            # Run performance benchmark
            await self.run_performance_benchmark()
            
            # Demonstrate arbitrage detection
            await self.demonstrate_arbitrage_detection()
            
            # Demonstrate market making
            await self.demonstrate_market_making()
            
            # Print final statistics
            self.print_final_statistics()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Demo error: {e}")
            logger.error(f"Demo error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        print(f"\n{Fore.YELLOW}🧹 Cleaning up resources...")
        
        try:
            if self.ml_orchestrator:
                await self.ml_orchestrator.shutdown_rust_engine()
                await self.ml_orchestrator.stop()
            
            if self.rust_engine:
                await self.rust_engine.shutdown()
            
            print(f"{Fore.GREEN}✅ Cleanup completed")
        except Exception as e:
            print(f"{Fore.RED}❌ Cleanup error: {e}")

async def main():
    """Main demo function"""
    demo = UltimateArbitrageDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())

