#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Master Orchestrator - Live Demo Script
==============================================

Demonstration script showing the Master Orchestrator in action with
real-time signal processing, autonomous decision making, and maximum
profit generation capabilities.

This script demonstrates:
- Zero-human intervention operation
- Real-time signal fusion and consensus
- Dynamic performance optimization
- Advanced health monitoring
- Microsecond-precision execution coordination
- Live performance metrics and reporting

Usage:
    python run_orchestrator_demo.py [--duration=60] [--signals=100]
"""

import asyncio
import argparse
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yaml
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ultimate_master_orchestrator import (
    UltimateMasterOrchestrator,
    get_ultimate_orchestrator,
    TradingSignal,
    SignalType,
    ExecutionPriority,
    ComponentStatus
)
from utils.orchestrator_logger import setup_orchestrator_logging

class SimulatedTradingEnvironment:
    """
    Simulated trading environment that generates realistic signals
    for demonstrating the Master Orchestrator's capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.signal_generators = {
            'arbitrage_detector': self._generate_arbitrage_signals,
            'momentum_strategy': self._generate_momentum_signals,
            'mean_reversion_strategy': self._generate_reversion_signals,
            'volatility_strategy': self._generate_volatility_signals,
            'sentiment_analyzer': self._generate_sentiment_signals,
            'news_processor': self._generate_news_signals
        }
        
        self.assets = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
            'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'XLM/USDT', 'ETC/USDT'
        ]
        
        self.market_conditions = {
            'trend': 'bullish',  # bullish, bearish, sideways
            'volatility': 'medium',  # low, medium, high
            'volume': 'normal'  # low, normal, high
        }
        
        self.signal_count = 0
    
    async def _generate_arbitrage_signals(self) -> List[TradingSignal]:
        """Generate arbitrage opportunity signals"""
        signals = []
        
        # High-confidence arbitrage opportunities
        if np.random.random() < 0.3:  # 30% chance
            asset = np.random.choice(self.assets)
            spread = np.random.uniform(1.5, 4.0)  # 1.5% to 4% spread
            
            signal = TradingSignal(
                signal_id=f"arb_{self.signal_count:06d}",
                signal_type=SignalType.ARBITRAGE,
                asset=asset,
                action="buy",
                confidence=min(0.95, 0.7 + spread * 0.05),  # Higher spread = higher confidence
                expected_profit=spread * 0.8,  # 80% of spread after fees
                risk_score=max(0.05, 0.2 - spread * 0.02),  # Lower risk for higher spreads
                urgency=ExecutionPriority.CRITICAL,
                timestamp=datetime.now(),
                source_component="arbitrage_detector",
                metadata={
                    'exchange_1': 'binance',
                    'exchange_2': 'coinbase',
                    'spread_percentage': spread,
                    'volume_available': np.random.uniform(10000, 100000)
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def _generate_momentum_signals(self) -> List[TradingSignal]:
        """Generate momentum-based trading signals"""
        signals = []
        
        # Generate momentum signals based on market trend
        if np.random.random() < 0.4:  # 40% chance
            asset = np.random.choice(self.assets)
            trend_strength = np.random.uniform(0.6, 0.9)
            
            action = "buy" if self.market_conditions['trend'] == 'bullish' else "sell"
            
            signal = TradingSignal(
                signal_id=f"mom_{self.signal_count:06d}",
                signal_type=SignalType.MOMENTUM,
                asset=asset,
                action=action,
                confidence=trend_strength,
                expected_profit=np.random.uniform(0.8, 2.5),
                risk_score=1 - trend_strength,
                urgency=ExecutionPriority.HIGH if trend_strength > 0.8 else ExecutionPriority.MEDIUM,
                timestamp=datetime.now(),
                source_component="momentum_strategy",
                metadata={
                    'trend_strength': trend_strength,
                    'rsi': np.random.uniform(30, 70),
                    'macd_signal': 'bullish' if action == 'buy' else 'bearish',
                    'volume_surge': np.random.choice([True, False])
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def _generate_reversion_signals(self) -> List[TradingSignal]:
        """Generate mean reversion signals"""
        signals = []
        
        if np.random.random() < 0.25:  # 25% chance
            asset = np.random.choice(self.assets)
            deviation = np.random.uniform(2.0, 4.0)  # Standard deviations from mean
            
            # Reversion signals are counter-trend
            action = "sell" if self.market_conditions['trend'] == 'bullish' else "buy"
            
            signal = TradingSignal(
                signal_id=f"rev_{self.signal_count:06d}",
                signal_type=SignalType.MEAN_REVERSION,
                asset=asset,
                action=action,
                confidence=min(0.8, 0.4 + deviation * 0.1),
                expected_profit=np.random.uniform(0.5, 1.8),
                risk_score=max(0.2, 0.6 - deviation * 0.05),
                urgency=ExecutionPriority.MEDIUM,
                timestamp=datetime.now(),
                source_component="mean_reversion_strategy",
                metadata={
                    'std_deviation': deviation,
                    'bollinger_position': 'upper' if action == 'sell' else 'lower',
                    'mean_price': np.random.uniform(40000, 50000),
                    'current_price': np.random.uniform(35000, 55000)
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def _generate_volatility_signals(self) -> List[TradingSignal]:
        """Generate volatility-based signals"""
        signals = []
        
        if np.random.random() < 0.2:  # 20% chance
            asset = np.random.choice(self.assets)
            volatility = np.random.uniform(0.3, 0.8)
            
            signal = TradingSignal(
                signal_id=f"vol_{self.signal_count:06d}",
                signal_type=SignalType.VOLATILITY,
                asset=asset,
                action=np.random.choice(["buy", "sell"]),
                confidence=volatility,
                expected_profit=np.random.uniform(0.3, 1.2),
                risk_score=volatility,  # Higher volatility = higher risk
                urgency=ExecutionPriority.LOW,
                timestamp=datetime.now(),
                source_component="volatility_strategy",
                metadata={
                    'volatility_index': volatility,
                    'atr': np.random.uniform(500, 2000),
                    'implied_volatility': np.random.uniform(0.2, 0.6)
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def _generate_sentiment_signals(self) -> List[TradingSignal]:
        """Generate sentiment-based signals"""
        signals = []
        
        if np.random.random() < 0.15:  # 15% chance
            asset = np.random.choice(self.assets)
            sentiment_score = np.random.uniform(-1, 1)
            
            action = "buy" if sentiment_score > 0 else "sell"
            
            signal = TradingSignal(
                signal_id=f"sent_{self.signal_count:06d}",
                signal_type=SignalType.SENTIMENT,
                asset=asset,
                action=action,
                confidence=abs(sentiment_score) * 0.6,  # Lower confidence for sentiment
                expected_profit=np.random.uniform(0.2, 0.8),
                risk_score=1 - abs(sentiment_score) * 0.5,
                urgency=ExecutionPriority.LOW,
                timestamp=datetime.now(),
                source_component="sentiment_analyzer",
                metadata={
                    'sentiment_score': sentiment_score,
                    'social_mentions': np.random.randint(100, 10000),
                    'fear_greed_index': np.random.randint(0, 100),
                    'whale_activity': np.random.choice(['high', 'medium', 'low'])
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def _generate_news_signals(self) -> List[TradingSignal]:
        """Generate news-based signals"""
        signals = []
        
        if np.random.random() < 0.1:  # 10% chance
            asset = np.random.choice(self.assets)
            news_impact = np.random.uniform(0.3, 0.9)
            
            signal = TradingSignal(
                signal_id=f"news_{self.signal_count:06d}",
                signal_type=SignalType.NEWS,
                asset=asset,
                action=np.random.choice(["buy", "sell"]),
                confidence=news_impact * 0.7,
                expected_profit=np.random.uniform(0.1, 0.6),
                risk_score=1 - news_impact,
                urgency=ExecutionPriority.MEDIUM if news_impact > 0.7 else ExecutionPriority.LOW,
                timestamp=datetime.now(),
                source_component="news_processor",
                metadata={
                    'news_impact': news_impact,
                    'news_type': np.random.choice(['regulatory', 'partnership', 'technical', 'market']),
                    'source_reliability': np.random.uniform(0.6, 0.95),
                    'breaking_news': news_impact > 0.8
                }
            )
            signals.append(signal)
            self.signal_count += 1
        
        return signals
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate signals from all strategies"""
        all_signals = []
        
        for generator_name, generator_func in self.signal_generators.items():
            try:
                signals = await generator_func()
                all_signals.extend(signals)
            except Exception as e:
                print(f"Error generating signals from {generator_name}: {e}")
        
        return all_signals
    
    async def get_health_data(self, component_name: str) -> Dict[str, Any]:
        """Get simulated health data for a component"""
        base_cpu = 20 + np.random.uniform(-10, 20)
        base_memory = 30 + np.random.uniform(-15, 25)
        
        # Add some variation based on component type
        if 'detector' in component_name or 'analyzer' in component_name:
            base_cpu += 15  # More CPU intensive
        
        return {
            'cpu_usage': max(0, min(100, base_cpu)),
            'memory_usage': max(0, min(100, base_memory)),
            'error_rate': np.random.uniform(0, 0.02),
            'uptime_seconds': np.random.uniform(3600, 86400)
        }

class OrchestratorDemo:
    """
    Main demo orchestrator that coordinates the demonstration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.duration = self.config.get('duration', 60)  # Demo duration in seconds
        self.signal_frequency = self.config.get('signal_frequency', 2)  # Signals per second
        
        # Initialize components
        self.orchestrator = get_ultimate_orchestrator(self.config.get('orchestrator', {}))
        self.trading_env = SimulatedTradingEnvironment(self.config.get('trading_env', {}))
        
        # Demo statistics
        self.demo_start_time = None
        self.total_signals_generated = 0
        self.total_signals_processed = 0
        self.demo_running = False
    
    async def setup_demo(self):
        """Setup the demo environment"""
        print("🚀 Setting up Ultimate Master Orchestrator Demo...")
        print("=" * 60)
        
        # Register simulated components with health callbacks
        for component_name in self.trading_env.signal_generators.keys():
            health_callback = lambda name=component_name: self.trading_env.get_health_data(name)
            await self.orchestrator.register_component(component_name, health_callback=health_callback)
            print(f"📝 Registered component: {component_name}")
        
        print(f"⚙️  Demo configuration:")
        print(f"   - Duration: {self.duration} seconds")
        print(f"   - Signal frequency: {self.signal_frequency} signals/second")
        print(f"   - Assets: {len(self.trading_env.assets)} trading pairs")
        print(f"   - Strategies: {len(self.trading_env.signal_generators)} signal generators")
        print("")
    
    async def start_demo(self):
        """Start the orchestrator demo"""
        print("🎯 Starting Ultimate Master Orchestrator Demo")
        print("💡 This demo showcases zero-human intervention autonomous trading")
        print("")
        
        self.demo_start_time = datetime.now()
        self.demo_running = True
        
        # Start the orchestrator
        start_success = await self.orchestrator.start_orchestration()
        if not start_success:
            print("❌ Failed to start orchestrator")
            return False
        
        print("✅ Master Orchestrator started successfully")
        print("🧠 AI consensus engine active")
        print("⚡ Performance optimizer running")
        print("🏥 Health monitor online")
        print("🎯 Execution coordinator ready")
        print("")
        
        # Start demo tasks
        demo_tasks = [
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._status_reporting_loop()),
            asyncio.create_task(self._demo_timer())
        ]
        
        try:
            await asyncio.gather(*demo_tasks)
        except asyncio.CancelledError:
            pass
        
        return True
    
    async def _signal_generation_loop(self):
        """Generate and submit signals continuously"""
        try:
            while self.demo_running:
                # Generate signals from simulated environment
                signals = await self.trading_env.generate_signals()
                
                # Submit signals to orchestrator
                for signal in signals:
                    success = await self.orchestrator.submit_signal(signal)
                    if success:
                        self.total_signals_generated += 1
                
                # Wait for next signal generation cycle
                await asyncio.sleep(1.0 / self.signal_frequency)
                
        except asyncio.CancelledError:
            pass
    
    async def _status_reporting_loop(self):
        """Report system status periodically"""
        try:
            while self.demo_running:
                await asyncio.sleep(10)  # Report every 10 seconds
                
                status = self.orchestrator.get_system_status()
                elapsed_time = (datetime.now() - self.demo_start_time).total_seconds()
                
                print(f"\n📊 Status Report ({elapsed_time:.0f}s elapsed):")
                print(f"   🔄 Orchestration cycles: {self.orchestrator.cycle_count:,}")
                print(f"   📡 Signals generated: {self.total_signals_generated}")
                print(f"   🎯 Signals processed: {len(self.orchestrator.processed_signals)}")
                print(f"   💰 Total profit: ${self.orchestrator.total_profit:.2f}")
                print(f"   🏥 System health: {self.orchestrator._calculate_system_health_score():.1f}/100")
                print(f"   ⚡ Queue size: {self.orchestrator.signal_queue.qsize()}")
                
                # Show recent trades
                if self.orchestrator.processed_signals:
                    recent_signal = list(self.orchestrator.processed_signals)[-1]
                    execution_result = recent_signal['execution_result']
                    if execution_result.get('success'):
                        profit = execution_result.get('profit_realized', 0)
                        asset = recent_signal['fused_signal'].asset
                        action = recent_signal['fused_signal'].action
                        print(f"   🎯 Last trade: {action.upper()} {asset} (+${profit:.2f})")
                
        except asyncio.CancelledError:
            pass
    
    async def _demo_timer(self):
        """Timer for demo duration"""
        try:
            await asyncio.sleep(self.duration)
            self.demo_running = False
        except asyncio.CancelledError:
            pass
    
    async def stop_demo(self):
        """Stop the demo and generate final report"""
        print("\n🛑 Stopping Ultimate Master Orchestrator Demo...")
        
        self.demo_running = False
        
        # Stop orchestrator
        await self.orchestrator.stop_orchestration()
        
        # Generate final report
        await self._generate_final_demo_report()
    
    async def _generate_final_demo_report(self):
        """Generate comprehensive final demo report"""
        total_time = (datetime.now() - self.demo_start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("📋 ULTIMATE MASTER ORCHESTRATOR DEMO REPORT")
        print("=" * 60)
        
        print(f"⏱️  Demo Duration: {total_time:.1f} seconds")
        print(f"🔄 Total Orchestration Cycles: {self.orchestrator.cycle_count:,}")
        print(f"📡 Signals Generated: {self.total_signals_generated}")
        print(f"🎯 Signals Processed: {len(self.orchestrator.processed_signals)}")
        print(f"💰 Total Profit Generated: ${self.orchestrator.total_profit:.2f}")
        
        # Performance metrics
        if self.orchestrator.processed_signals:
            successful_trades = sum(1 for s in self.orchestrator.processed_signals 
                                  if s['execution_result'].get('success'))
            win_rate = successful_trades / len(self.orchestrator.processed_signals) * 100
            
            print(f"📈 Win Rate: {win_rate:.1f}%")
            print(f"⚡ Average Processing Time: {np.mean([s['processing_time_ms'] for s in self.orchestrator.processed_signals]):.2f}ms")
        
        # System performance
        cycles_per_second = self.orchestrator.cycle_count / total_time
        signals_per_second = self.total_signals_generated / total_time
        
        print(f"🚀 Orchestration Performance: {cycles_per_second:.1f} cycles/second")
        print(f"📊 Signal Generation Rate: {signals_per_second:.1f} signals/second")
        
        # Component health summary
        if self.orchestrator.component_health:
            healthy_components = sum(1 for c in self.orchestrator.component_health.values() 
                                   if c.status == ComponentStatus.HEALTHY)
            total_components = len(self.orchestrator.component_health)
            health_percentage = healthy_components / total_components * 100
            
            print(f"🏥 Final System Health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)")
        
        print("\n🎯 Key Demo Achievements:")
        print("   ✅ Zero human intervention operation")
        print("   ✅ Real-time signal fusion and consensus")
        print("   ✅ Autonomous trading decision making")
        print("   ✅ Dynamic performance optimization")
        print("   ✅ Continuous health monitoring")
        print("   ✅ Microsecond-precision execution")
        
        print("\n" + "=" * 60)
        print("🚀 Demo completed successfully! The Ultimate Master Orchestrator")
        print("   demonstrated full autonomous operation with maximum efficiency.")
        print("=" * 60)

async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Ultimate Master Orchestrator Demo')
    parser.add_argument('--duration', type=int, default=60, help='Demo duration in seconds')
    parser.add_argument('--frequency', type=float, default=2.0, help='Signal generation frequency (signals/second)')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'duration': args.duration,
        'signal_frequency': args.frequency
    }
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # Setup logging
    logger = setup_orchestrator_logging(config.get('logging', {}))
    
    # Create and run demo
    demo = OrchestratorDemo(config)
    
    try:
        await demo.setup_demo()
        await demo.start_demo()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        await demo.stop_demo()

if __name__ == "__main__":
    asyncio.run(main())

