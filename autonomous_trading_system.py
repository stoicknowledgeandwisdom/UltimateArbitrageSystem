#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Trading System - Ultimate Profit Maximization
=======================================================

A fully autonomous trading system that operates 24/7 without human intervention,
designed to maximize profits through aggressive opportunity capture and execution.

Features:
- 24/7 autonomous operation
- Real-time market monitoring
- Instant opportunity execution
- Dynamic position sizing
- Risk management automation
- Multi-strategy execution
- Performance optimization
- Emergency stop mechanisms
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import time
import json
import sqlite3
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

from ultra_high_frequency_engine import UltraHighFrequencyEngine, UltraArbitrageOpportunity
from maximum_income_optimizer import MaximumIncomeOptimizer

logger = logging.getLogger(__name__)

@dataclass
class TradingPosition:
    """Represents an active trading position"""
    position_id: str
    strategy_type: str
    symbols: List[str]
    exchanges: List[str]
    entry_prices: List[float]
    position_sizes: List[float]
    target_prices: List[float]
    stop_loss_prices: List[float]
    expected_profit: float
    current_profit: float
    confidence_score: float
    risk_score: float
    entry_time: datetime
    status: str  # 'open', 'closed', 'partial'
    execution_time_ms: float

@dataclass
class AutoTradingConfig:
    """Configuration for autonomous trading"""
    max_concurrent_positions: int = 50
    max_position_size_usd: float = 5000.0
    min_profit_threshold: float = 10.0  # Minimum $10 profit per trade
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    portfolio_risk_limit: float = 0.10  # 10% total portfolio risk
    execution_delay_ms: float = 50.0  # Maximum execution delay
    profit_take_percentage: float = 0.8  # Take 80% profit at target
    stop_loss_percentage: float = 0.02  # 2% stop loss
    rebalance_frequency_minutes: int = 5  # Rebalance every 5 minutes
    emergency_stop_loss: float = 0.05  # 5% emergency stop
    enable_flash_loans: bool = True
    enable_mev_extraction: bool = True
    enable_cross_chain: bool = True
    max_daily_trades: int = 1000
    target_daily_profit: float = 10000.0  # Target $10k daily profit

class AutonomousTradingSystem:
    """Fully autonomous trading system for maximum profit generation"""
    
    def __init__(self, config: AutoTradingConfig = None):
        self.config = config or AutoTradingConfig()
        self.ultra_engine = UltraHighFrequencyEngine()
        self.optimizer = MaximumIncomeOptimizer()
        
        # Trading state
        self.active_positions: Dict[str, TradingPosition] = {}
        self.portfolio_balance = 100000.0  # Starting with $100k
        self.daily_profit = 0.0
        self.total_trades_today = 0
        self.is_running = False
        self.emergency_stop = False
        
        # Performance tracking
        self.performance_stats = {
            'total_profit': 0.0,
            'daily_profit': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'recovery_factor': 0.0
        }
        
        # Database for persistence
        self.db_path = Path("autonomous_trading.db")
        self.setup_database()
        
        # Threading for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
    
    def setup_database(self):
        """Setup database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    strategy_type TEXT,
                    symbols TEXT,
                    exchanges TEXT,
                    entry_prices TEXT,
                    position_sizes TEXT,
                    expected_profit REAL,
                    current_profit REAL,
                    confidence_score REAL,
                    risk_score REAL,
                    entry_time TIMESTAMP,
                    status TEXT,
                    execution_time_ms REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT,
                    timestamp TIMESTAMP,
                    action TEXT,
                    profit_loss REAL,
                    portfolio_balance REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_daily (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    total_profit REAL,
                    portfolio_balance REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL
                )
            """)
    
    async def start_autonomous_trading(self):
        """Start the autonomous trading system"""
        logger.info("🚀 Starting Autonomous Trading System")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start multiple concurrent tasks
        tasks = [
            asyncio.create_task(self.opportunity_scanner()),
            asyncio.create_task(self.position_manager()),
            asyncio.create_task(self.risk_manager()),
            asyncio.create_task(self.performance_monitor()),
            asyncio.create_task(self.portfolio_rebalancer()),
            asyncio.create_task(self.emergency_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Critical error in autonomous trading: {e}")
            await self.emergency_shutdown()
    
    async def opportunity_scanner(self):
        """Continuously scan for trading opportunities"""
        logger.info("🔍 Starting opportunity scanner")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Generate realistic market data
                market_data = self._generate_market_data()
                
                # Detect ultra-high-frequency opportunities
                opportunities = await self.ultra_engine.detect_ultra_opportunities(market_data)
                
                # Filter and rank opportunities
                filtered_opps = self._filter_opportunities(opportunities)
                
                # Execute top opportunities
                for opp in filtered_opps[:10]:  # Top 10 opportunities
                    if self._can_execute_trade(opp):
                        await self._execute_opportunity(opp)
                
                # Short delay to prevent overwhelming the system
                await asyncio.sleep(0.1)  # 100ms scan frequency
                
            except Exception as e:
                logger.error(f"Error in opportunity scanner: {e}")
                await asyncio.sleep(1.0)
    
    async def position_manager(self):
        """Manage active positions and execute exits"""
        logger.info("📊 Starting position manager")
        
        while self.is_running and not self.emergency_stop:
            try:
                positions_to_close = []
                
                for position_id, position in self.active_positions.items():
                    # Update position profit/loss
                    current_profit = self._calculate_position_pnl(position)
                    position.current_profit = current_profit
                    
                    # Check exit conditions
                    if self._should_close_position(position):
                        positions_to_close.append(position_id)
                
                # Close positions that meet exit criteria
                for position_id in positions_to_close:
                    await self._close_position(position_id)
                
                await asyncio.sleep(0.5)  # Check positions every 500ms
                
            except Exception as e:
                logger.error(f"Error in position manager: {e}")
                await asyncio.sleep(1.0)
    
    async def risk_manager(self):
        """Monitor and manage risk levels"""
        logger.info("🛡️ Starting risk manager")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Calculate current portfolio risk
                total_risk = self._calculate_portfolio_risk()
                
                # Check risk limits
                if total_risk > self.config.portfolio_risk_limit:
                    logger.warning(f"Portfolio risk too high: {total_risk:.2%}")
                    await self._reduce_risk()
                
                # Check for emergency stop conditions
                if self._check_emergency_conditions():
                    logger.critical("EMERGENCY STOP CONDITIONS MET")
                    await self.emergency_shutdown()
                
                await asyncio.sleep(2.0)  # Risk check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in risk manager: {e}")
                await asyncio.sleep(1.0)
    
    async def performance_monitor(self):
        """Monitor and log performance metrics"""
        logger.info("📈 Starting performance monitor")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Update performance statistics
                self._update_performance_stats()
                
                # Log current performance
                if self.total_trades_today % 100 == 0 and self.total_trades_today > 0:
                    self._log_performance_summary()
                
                # Save performance to database
                await self._save_performance_data()
                
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(1.0)
    
    async def portfolio_rebalancer(self):
        """Rebalance portfolio for optimal allocation"""
        logger.info("⚖️ Starting portfolio rebalancer")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Rebalance if needed
                if self._needs_rebalancing():
                    await self._rebalance_portfolio()
                
                await asyncio.sleep(self.config.rebalance_frequency_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in portfolio rebalancer: {e}")
                await asyncio.sleep(60.0)
    
    async def emergency_monitor(self):
        """Monitor for emergency conditions"""
        logger.info("🚨 Starting emergency monitor")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Check daily profit target
                if self.daily_profit >= self.config.target_daily_profit:
                    logger.info(f"🎯 Daily profit target reached: ${self.daily_profit:.2f}")
                    # Could implement profit-taking logic here
                
                # Check maximum daily trades
                if self.total_trades_today >= self.config.max_daily_trades:
                    logger.warning("Maximum daily trades reached, pausing trading")
                    await asyncio.sleep(3600)  # Pause for 1 hour
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in emergency monitor: {e}")
                await asyncio.sleep(1.0)
    
    def _generate_market_data(self) -> Dict[str, Any]:
        """Generate realistic market data for testing"""
        # In production, this would connect to real exchanges
        return {
            'binance': {
                'BTC/USDT': {'price': 45000 + np.random.normal(0, 100), 'volume': 1000},
                'ETH/USDT': {'price': 3000 + np.random.normal(0, 50), 'volume': 500}
            },
            'coinbase': {
                'BTC/USDT': {'price': 45000 + np.random.normal(0, 100), 'volume': 800},
                'ETH/USDT': {'price': 3000 + np.random.normal(0, 50), 'volume': 600}
            },
            'timestamp': datetime.now()
        }
    
    def _filter_opportunities(self, opportunities: List[UltraArbitrageOpportunity]) -> List[UltraArbitrageOpportunity]:
        """Filter opportunities based on current trading conditions"""
        filtered = []
        
        for opp in opportunities:
            # Only consider high-quality opportunities
            if (opp.confidence_score > 0.85 and 
                opp.profit_per_1000_usd > self.config.min_profit_threshold and
                opp.risk_score < 0.2):
                
                # Check if we have capacity for this trade
                if len(self.active_positions) < self.config.max_concurrent_positions:
                    filtered.append(opp)
        
        # Sort by profit potential and urgency
        filtered.sort(key=lambda x: (x.urgency_level, x.profit_per_1000_usd), reverse=True)
        return filtered
    
    def _can_execute_trade(self, opportunity: UltraArbitrageOpportunity) -> bool:
        """Check if we can execute this trade"""
        # Calculate position size
        position_size = min(
            self.config.max_position_size_usd,
            self.portfolio_balance * 0.05  # Max 5% of portfolio per trade
        )
        
        # Check portfolio limits
        if position_size < 100:  # Minimum $100 position
            return False
        
        # Check risk limits
        trade_risk = position_size * opportunity.risk_score
        current_risk = self._calculate_portfolio_risk()
        
        if current_risk + trade_risk > self.config.portfolio_risk_limit:
            return False
        
        # Check daily trade limit
        if self.total_trades_today >= self.config.max_daily_trades:
            return False
        
        return True
    
    async def _execute_opportunity(self, opportunity: UltraArbitrageOpportunity):
        """Execute a trading opportunity"""
        try:
            # Calculate position size
            position_size = min(
                self.config.max_position_size_usd,
                self.portfolio_balance * 0.05
            )
            
            # Create position
            position_id = f"{opportunity.strategy_type}_{int(time.time()*1000)}"
            
            position = TradingPosition(
                position_id=position_id,
                strategy_type=opportunity.strategy_type,
                symbols=opportunity.symbols,
                exchanges=opportunity.exchanges,
                entry_prices=opportunity.entry_prices,
                position_sizes=[position_size / len(opportunity.symbols)],
                target_prices=[p * (1 + opportunity.profit_percentage) for p in opportunity.entry_prices],
                stop_loss_prices=[p * (1 - self.config.stop_loss_percentage) for p in opportunity.entry_prices],
                expected_profit=opportunity.profit_per_1000_usd * (position_size / 1000),
                current_profit=0.0,
                confidence_score=opportunity.confidence_score,
                risk_score=opportunity.risk_score,
                entry_time=datetime.now(),
                status='open',
                execution_time_ms=opportunity.execution_time_ms
            )
            
            # Add to active positions
            self.active_positions[position_id] = position
            
            # Update portfolio balance (simulate execution)
            self.portfolio_balance -= position_size * 0.001  # 0.1% execution cost
            
            # Increment trade counter
            self.total_trades_today += 1
            
            # Log execution
            logger.info(f"🔥 Executed {opportunity.strategy_type}: "
                       f"${position.expected_profit:.2f} expected profit, "
                       f"{opportunity.confidence_score:.1%} confidence")
            
            # Save to database
            await self._save_position(position)
            
        except Exception as e:
            logger.error(f"Error executing opportunity: {e}")
    
    def _calculate_position_pnl(self, position: TradingPosition) -> float:
        """Calculate current profit/loss for a position"""
        # Simulate P&L calculation based on time and market movement
        time_factor = (datetime.now() - position.entry_time).total_seconds() / 3600  # Hours
        
        # Simulate price movement (in production, get real prices)
        price_movement = np.random.normal(0, 0.01)  # 1% volatility per hour
        
        # Calculate P&L based on confidence and time
        if position.confidence_score > 0.9:
            # High confidence trades tend to be profitable
            base_return = position.expected_profit * (0.5 + time_factor * 0.1)
            noise = base_return * np.random.normal(0, 0.1)  # 10% noise
            return base_return + noise
        else:
            # Lower confidence trades are more volatile
            base_return = position.expected_profit * np.random.normal(0.2, 0.3)
            return base_return
    
    def _should_close_position(self, position: TradingPosition) -> bool:
        """Determine if a position should be closed"""
        # Time-based closure (max 1 hour for ultra-HF strategies)
        time_elapsed = (datetime.now() - position.entry_time).total_seconds() / 3600
        
        if position.strategy_type in ['micro_arbitrage', 'mev_extraction', 'latency_ultra']:
            max_time = 0.1  # 6 minutes for ultra-fast strategies
        else:
            max_time = 1.0  # 1 hour for other strategies
        
        if time_elapsed > max_time:
            return True
        
        # Profit target reached
        if position.current_profit >= position.expected_profit * self.config.profit_take_percentage:
            return True
        
        # Stop loss triggered
        if position.current_profit <= -position.expected_profit * self.config.stop_loss_percentage:
            return True
        
        return False
    
    async def _close_position(self, position_id: str):
        """Close a trading position"""
        try:
            position = self.active_positions[position_id]
            
            # Calculate final profit/loss
            final_pnl = position.current_profit
            
            # Update portfolio balance
            self.portfolio_balance += sum(position.position_sizes) + final_pnl
            self.daily_profit += final_pnl
            
            # Update statistics
            if final_pnl > 0:
                self.performance_stats['winning_trades'] += 1
                if final_pnl > self.performance_stats['largest_win']:
                    self.performance_stats['largest_win'] = final_pnl
            else:
                self.performance_stats['losing_trades'] += 1
                if final_pnl < self.performance_stats['largest_loss']:
                    self.performance_stats['largest_loss'] = final_pnl
            
            # Log closure
            profit_text = "PROFIT" if final_pnl > 0 else "LOSS"
            logger.info(f"💰 Closed {position.strategy_type}: "
                       f"${final_pnl:.2f} {profit_text} "
                       f"(Target: ${position.expected_profit:.2f})")
            
            # Remove from active positions
            position.status = 'closed'
            del self.active_positions[position_id]
            
            # Save trade record
            await self._save_trade_record(position_id, final_pnl)
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        if not self.active_positions:
            return 0.0
        
        total_risk = sum(
            pos.risk_score * sum(pos.position_sizes) / self.portfolio_balance
            for pos in self.active_positions.values()
        )
        
        return total_risk
    
    def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met"""
        # Emergency stop if daily loss exceeds limit
        if self.daily_profit < -self.portfolio_balance * self.config.emergency_stop_loss:
            return True
        
        # Emergency stop if too many consecutive losses
        if self.performance_stats['consecutive_losses'] > 10:
            return True
        
        return False
    
    async def _reduce_risk(self):
        """Reduce portfolio risk by closing positions"""
        # Sort positions by risk and close the riskiest ones
        risky_positions = sorted(
            self.active_positions.items(),
            key=lambda x: x[1].risk_score,
            reverse=True
        )
        
        positions_to_close = min(5, len(risky_positions) // 2)
        
        for i in range(positions_to_close):
            position_id = risky_positions[i][0]
            await self._close_position(position_id)
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        total_trades = self.performance_stats['winning_trades'] + self.performance_stats['losing_trades']
        
        if total_trades > 0:
            self.performance_stats['win_rate'] = self.performance_stats['winning_trades'] / total_trades
            self.performance_stats['average_profit_per_trade'] = self.daily_profit / total_trades
        
        self.performance_stats['total_profit'] = self.daily_profit
        self.performance_stats['daily_profit'] = self.daily_profit
        
        # Calculate profit factor
        total_wins = self.performance_stats['winning_trades'] * abs(self.performance_stats['largest_win'])
        total_losses = self.performance_stats['losing_trades'] * abs(self.performance_stats['largest_loss'])
        
        if total_losses > 0:
            self.performance_stats['profit_factor'] = total_wins / total_losses
    
    def _log_performance_summary(self):
        """Log performance summary"""
        logger.info(f"📊 PERFORMANCE SUMMARY (Last 100 trades)")
        logger.info(f"Portfolio Balance: ${self.portfolio_balance:.2f}")
        logger.info(f"Daily Profit: ${self.daily_profit:.2f}")
        logger.info(f"Win Rate: {self.performance_stats['win_rate']:.1%}")
        logger.info(f"Average Profit/Trade: ${self.performance_stats['average_profit_per_trade']:.2f}")
        logger.info(f"Active Positions: {len(self.active_positions)}")
        logger.info(f"Total Trades Today: {self.total_trades_today}")
    
    async def _save_position(self, position: TradingPosition):
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.position_id,
                    position.strategy_type,
                    json.dumps(position.symbols),
                    json.dumps(position.exchanges),
                    json.dumps(position.entry_prices),
                    json.dumps(position.position_sizes),
                    position.expected_profit,
                    position.current_profit,
                    position.confidence_score,
                    position.risk_score,
                    position.entry_time,
                    position.status,
                    position.execution_time_ms
                ))
        except Exception as e:
            logger.error(f"Error saving position: {e}")
    
    async def _save_trade_record(self, position_id: str, pnl: float):
        """Save trade record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (position_id, timestamp, action, profit_loss, portfolio_balance)
                    VALUES (?, ?, ?, ?, ?)
                """, (position_id, datetime.now(), 'close', pnl, self.portfolio_balance))
        except Exception as e:
            logger.error(f"Error saving trade record: {e}")
    
    async def _save_performance_data(self):
        """Save daily performance data"""
        try:
            today = datetime.now().date().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO performance_daily 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    today,
                    self.total_trades_today,
                    self.performance_stats['winning_trades'],
                    self.daily_profit,
                    self.portfolio_balance,
                    self.performance_stats['max_drawdown'],
                    self.performance_stats['sharpe_ratio']
                ))
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing"""
        # Simple rebalancing logic - could be more sophisticated
        return len(self.active_positions) > self.config.max_concurrent_positions * 0.8
    
    async def _rebalance_portfolio(self):
        """Rebalance portfolio allocation"""
        logger.info("⚖️ Rebalancing portfolio")
        
        # Close least profitable positions
        positions_by_profit = sorted(
            self.active_positions.items(),
            key=lambda x: x[1].current_profit
        )
        
        positions_to_close = len(positions_by_profit) // 4  # Close bottom 25%
        
        for i in range(positions_to_close):
            position_id = positions_by_profit[i][0]
            await self._close_position(position_id)
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        self.emergency_stop = True
        
        # Close all positions immediately
        position_ids = list(self.active_positions.keys())
        for position_id in position_ids:
            await self._close_position(position_id)
        
        # Save final state
        await self._save_performance_data()
        
        logger.critical(f"Emergency shutdown complete. Final balance: ${self.portfolio_balance:.2f}")
    
    def graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown signals"""
        logger.info("Received shutdown signal, initiating graceful shutdown...")
        self.is_running = False
        asyncio.create_task(self.emergency_shutdown())

# Configuration for maximum profit
ULTRA_AGGRESSIVE_CONFIG = AutoTradingConfig(
    max_concurrent_positions=100,  # Increased capacity
    max_position_size_usd=10000.0,  # Larger positions
    min_profit_threshold=5.0,  # Lower threshold for more opportunities
    max_risk_per_trade=0.03,  # 3% risk per trade
    portfolio_risk_limit=0.15,  # 15% total portfolio risk
    execution_delay_ms=25.0,  # Faster execution
    profit_take_percentage=0.7,  # Take 70% profit quickly
    stop_loss_percentage=0.015,  # Tighter stop loss
    rebalance_frequency_minutes=3,  # More frequent rebalancing
    emergency_stop_loss=0.08,  # 8% emergency stop
    enable_flash_loans=True,
    enable_mev_extraction=True,
    enable_cross_chain=True,
    max_daily_trades=2000,  # More trades
    target_daily_profit=25000.0  # Higher target: $25k daily
)

async def main():
    """Run the autonomous trading system"""
    # Use ultra-aggressive configuration
    trading_system = AutonomousTradingSystem(ULTRA_AGGRESSIVE_CONFIG)
    
    print("\n🚀 ULTIMATE AUTONOMOUS TRADING SYSTEM 🚀")
    print("=" * 60)
    print(f"💰 Starting Portfolio: ${trading_system.portfolio_balance:.2f}")
    print(f"🎯 Daily Profit Target: ${trading_system.config.target_daily_profit:.2f}")
    print(f"📊 Max Concurrent Positions: {trading_system.config.max_concurrent_positions}")
    print(f"⚡ Max Daily Trades: {trading_system.config.max_daily_trades}")
    print(f"🛡️ Portfolio Risk Limit: {trading_system.config.portfolio_risk_limit:.1%}")
    print("=" * 60)
    print("System running... Press Ctrl+C to stop")
    
    try:
        await trading_system.start_autonomous_trading()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        await trading_system.emergency_shutdown()
    except Exception as e:
        print(f"\n❌ System error: {e}")
        await trading_system.emergency_shutdown()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('autonomous_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run the autonomous trading system
    asyncio.run(main())

