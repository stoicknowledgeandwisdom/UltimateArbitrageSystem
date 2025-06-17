#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Arbitrage Orchestrator - Ultimate Profit Extraction Command Center
=========================================================================

Central orchestration system that coordinates multiple arbitrage engines
to maximize profit extraction across all opportunities simultaneously.

Features:
- 🎯 Multi-Engine Coordination
- 🌐 Cross-Chain + Intra-Chain Arbitrage
- ⚡ Real-time Opportunity Prioritization
- 💰 Capital Allocation Optimization
- 🛡️ Risk Management & Monitoring
- 📊 Comprehensive Performance Analytics
- 🔄 Dynamic Strategy Adaptation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import json

# Import our arbitrage engines
from cross_chain_arbitrage_engine import CrossChainArbitrageEngine
from multi_dex_arbitrage_engine import MultiDEXArbitrageEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OpportunityMetrics:
    """Unified opportunity metrics across all engines"""
    opportunity_id: str
    engine_type: str
    estimated_profit: float
    confidence_score: float
    risk_score: float
    execution_time_estimate: int
    capital_required: float
    profit_per_second: float
    priority_score: float
    timestamp: datetime

@dataclass
class EnginePerformance:
    """Performance metrics for each engine"""
    engine_name: str
    total_profit: float
    successful_trades: int
    failed_trades: int
    success_rate: float
    avg_profit_per_trade: float
    active_opportunities: int
    last_execution: Optional[datetime]
    performance_score: float

class MasterArbitrageOrchestrator:
    """Master orchestrator for all arbitrage strategies"""
    
    def __init__(self):
        self.is_running = False
        self.engines: Dict[str, Any] = {}
        self.opportunity_queue: List[OpportunityMetrics] = []
        self.execution_history: List[Dict] = []
        
        # Performance tracking
        self.total_system_profit = 0.0
        self.total_opportunities_detected = 0
        self.total_opportunities_executed = 0
        self.engine_performances: Dict[str, EnginePerformance] = {}
        
        # Risk management
        self.max_concurrent_trades = 5
        self.max_capital_per_trade = 100000.0
        self.active_trades = 0
        self.daily_profit_target = 10000.0
        self.daily_loss_limit = 2000.0
        self.current_daily_pnl = 0.0
        
        # Strategy parameters
        self.profit_threshold = 10.0  # Minimum $10 profit
        self.max_risk_score = 0.7
        self.min_confidence = 0.6
        
        logger.info("🎯 Master Arbitrage Orchestrator initialized")
    
    async def initialize_engines(self):
        """Initialize all arbitrage engines"""
        try:
            # Initialize Cross-Chain Arbitrage Engine
            self.engines['cross_chain'] = CrossChainArbitrageEngine()
            logger.info("🌐 Cross-Chain Engine initialized")
            
            # Initialize Multi-DEX Arbitrage Engine for Ethereum
            self.engines['dex_ethereum'] = MultiDEXArbitrageEngine("ethereum")
            logger.info("⚡ DEX Ethereum Engine initialized")
            
            # Initialize additional DEX engines for other chains
            self.engines['dex_polygon'] = MultiDEXArbitrageEngine("polygon")
            self.engines['dex_arbitrum'] = MultiDEXArbitrageEngine("arbitrum")
            logger.info("⚡ Additional DEX Engines initialized")
            
            # Initialize engine performance tracking
            for engine_name in self.engines.keys():
                self.engine_performances[engine_name] = EnginePerformance(
                    engine_name=engine_name,
                    total_profit=0.0,
                    successful_trades=0,
                    failed_trades=0,
                    success_rate=0.0,
                    avg_profit_per_trade=0.0,
                    active_opportunities=0,
                    last_execution=None,
                    performance_score=1.0
                )
            
            logger.info("🎯 All engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing engines: {e}")
            raise
    
    async def start_orchestration(self):
        """Start the master orchestration system"""
        if self.is_running:
            logger.warning("Orchestration already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Master Arbitrage Orchestration...")
        
        # Start all engines concurrently
        engine_tasks = []
        for engine_name, engine in self.engines.items():
            if hasattr(engine, 'start_cross_chain_hunting'):
                task = asyncio.create_task(engine.start_cross_chain_hunting())
            elif hasattr(engine, 'start_dex_hunting'):
                task = asyncio.create_task(engine.start_dex_hunting())
            else:
                continue
            engine_tasks.append(task)
        
        # Start orchestration tasks
        orchestration_tasks = [
            asyncio.create_task(self._opportunity_aggregation_loop()),
            asyncio.create_task(self._priority_execution_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._risk_management_loop()),
            asyncio.create_task(self._strategy_optimization_loop())
        ]
        
        all_tasks = engine_tasks + orchestration_tasks
        
        try:
            await asyncio.gather(*all_tasks)
        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
        finally:
            self.is_running = False
    
    async def _opportunity_aggregation_loop(self):
        """Aggregate opportunities from all engines"""
        while self.is_running:
            try:
                aggregated_opportunities = []
                
                # Collect opportunities from all engines
                for engine_name, engine in self.engines.items():
                    if hasattr(engine, 'active_opportunities'):
                        for opp in engine.active_opportunities:
                            unified_opp = self._convert_to_unified_opportunity(opp, engine_name)
                            if unified_opp:
                                aggregated_opportunities.append(unified_opp)
                
                # Update opportunity queue with prioritization
                self.opportunity_queue = self._prioritize_opportunities(aggregated_opportunities)
                self.total_opportunities_detected = len(self.opportunity_queue)
                
                await asyncio.sleep(1)  # Aggregate every second
                
            except Exception as e:
                logger.error(f"Error in opportunity aggregation: {e}")
                await asyncio.sleep(2)
    
    async def _priority_execution_loop(self):
        """Execute opportunities based on priority"""
        while self.is_running:
            try:
                if (self.opportunity_queue and 
                    self.active_trades < self.max_concurrent_trades and
                    self._can_execute_trade()):
                    
                    # Get highest priority opportunity
                    best_opportunity = self.opportunity_queue[0]
                    
                    if self._should_execute_opportunity(best_opportunity):
                        # Execute the opportunity
                        execution_result = await self._execute_opportunity(best_opportunity)
                        
                        # Update metrics
                        self._update_execution_metrics(execution_result)
                        
                        # Remove executed opportunity
                        self.opportunity_queue.remove(best_opportunity)
                        self.total_opportunities_executed += 1
                
                await asyncio.sleep(0.1)  # High-frequency execution check
                
            except Exception as e:
                logger.error(f"Error in priority execution: {e}")
                await asyncio.sleep(1)
    
    async def _performance_monitoring_loop(self):
        """Monitor performance of all engines"""
        while self.is_running:
            try:
                # Update engine performances
                for engine_name, engine in self.engines.items():
                    if hasattr(engine, 'get_cross_chain_performance_metrics'):
                        metrics = engine.get_cross_chain_performance_metrics()
                    elif hasattr(engine, 'get_dex_performance_metrics'):
                        metrics = engine.get_dex_performance_metrics()
                    else:
                        continue
                    
                    # Update performance tracking
                    perf = self.engine_performances[engine_name]
                    perf.total_profit = metrics.get('total_cross_chain_profit_usd', 0) or metrics.get('total_dex_profit_usd', 0)
                    perf.successful_trades = metrics.get('successful_arbitrages', 0) or metrics.get('successful_dex_arbitrages', 0)
                    perf.failed_trades = metrics.get('failed_arbitrages', 0) or metrics.get('failed_dex_arbitrages', 0)
                    perf.success_rate = metrics.get('success_rate', 0)
                    perf.active_opportunities = metrics.get('active_opportunities', 0)
                    
                    if perf.successful_trades > 0:
                        perf.avg_profit_per_trade = perf.total_profit / perf.successful_trades
                    
                    # Calculate performance score
                    perf.performance_score = self._calculate_engine_performance_score(perf)
                
                # Update total system profit
                self.total_system_profit = sum(perf.total_profit for perf in self.engine_performances.values())
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _risk_management_loop(self):
        """Manage risk across all strategies"""
        while self.is_running:
            try:
                # Check daily PnL limits
                if self.current_daily_pnl >= self.daily_profit_target:
                    logger.info(f"🎯 Daily profit target reached: ${self.current_daily_pnl:.2f}")
                    # Could reduce aggressiveness or take profits
                
                if self.current_daily_pnl <= -self.daily_loss_limit:
                    logger.warning(f"⚠️ Daily loss limit hit: ${self.current_daily_pnl:.2f}")
                    # Could pause trading or reduce position sizes
                    self._reduce_risk_exposure()
                
                # Monitor for unusual market conditions
                market_volatility = self._assess_market_volatility()
                if market_volatility > 0.8:
                    logger.warning("🌊 High market volatility detected - adjusting thresholds")
                    self._adjust_risk_parameters(market_volatility)
                
                await asyncio.sleep(30)  # Risk check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in risk management: {e}")
                await asyncio.sleep(30)
    
    async def _strategy_optimization_loop(self):
        """Optimize strategies based on performance"""
        while self.is_running:
            try:
                # Analyze engine performance and adjust allocations
                best_performers = sorted(
                    self.engine_performances.values(),
                    key=lambda x: x.performance_score,
                    reverse=True
                )
                
                # Adjust thresholds based on performance
                for perf in best_performers:
                    if perf.performance_score > 1.5:
                        # Increase allocation to high-performing engines
                        logger.info(f"📈 Boosting allocation to {perf.engine_name}")
                    elif perf.performance_score < 0.5:
                        # Reduce allocation to underperforming engines
                        logger.info(f"📉 Reducing allocation to {perf.engine_name}")
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Error in strategy optimization: {e}")
                await asyncio.sleep(60)
    
    def _convert_to_unified_opportunity(self, opportunity: Any, engine_type: str) -> Optional[OpportunityMetrics]:
        """Convert engine-specific opportunity to unified format"""
        try:
            # Extract common fields regardless of opportunity type
            opp_id = getattr(opportunity, 'opportunity_id', f"{engine_type}_{int(time.time())}")
            estimated_profit = getattr(opportunity, 'estimated_profit', 0)
            confidence_score = getattr(opportunity, 'confidence_score', 0)
            risk_score = getattr(opportunity, 'risk_score', 0.5)
            execution_time = getattr(opportunity, 'execution_time_estimate', 1000)
            
            # Calculate capital required (simplified)
            capital_required = getattr(opportunity, 'max_trade_size', estimated_profit * 10)
            if capital_required == 0:
                capital_required = estimated_profit * 20  # Estimate if not available
            
            # Calculate profit per second
            profit_per_second = estimated_profit / max(1, execution_time / 1000)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(
                estimated_profit, confidence_score, risk_score, profit_per_second
            )
            
            return OpportunityMetrics(
                opportunity_id=opp_id,
                engine_type=engine_type,
                estimated_profit=estimated_profit,
                confidence_score=confidence_score,
                risk_score=risk_score,
                execution_time_estimate=execution_time,
                capital_required=capital_required,
                profit_per_second=profit_per_second,
                priority_score=priority_score,
                timestamp=getattr(opportunity, 'timestamp', datetime.now())
            )
        
        except Exception as e:
            logger.error(f"Error converting opportunity: {e}")
            return None
    
    def _prioritize_opportunities(self, opportunities: List[OpportunityMetrics]) -> List[OpportunityMetrics]:
        """Prioritize opportunities based on multiple factors"""
        # Filter opportunities based on criteria
        filtered_opportunities = [
            opp for opp in opportunities
            if (opp.estimated_profit >= self.profit_threshold and
                opp.confidence_score >= self.min_confidence and
                opp.risk_score <= self.max_risk_score and
                opp.capital_required <= self.max_capital_per_trade)
        ]
        
        # Sort by priority score (higher is better)
        filtered_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return filtered_opportunities[:20]  # Keep top 20 opportunities
    
    def _calculate_priority_score(self, profit: float, confidence: float, risk: float, profit_per_sec: float) -> float:
        """Calculate priority score for opportunity ranking"""
        # Weighted scoring algorithm
        profit_score = min(1.0, profit / 100.0)  # Normalize to $100 max
        speed_score = min(1.0, profit_per_sec / 10.0)  # Normalize to $10/sec max
        risk_adjusted_confidence = confidence * (1 - risk)
        
        priority_score = (profit_score * 0.4 + 
                         speed_score * 0.3 + 
                         risk_adjusted_confidence * 0.3)
        
        return priority_score
    
    def _should_execute_opportunity(self, opportunity: OpportunityMetrics) -> bool:
        """Determine if opportunity should be executed"""
        # Check if opportunity is still fresh
        age_seconds = (datetime.now() - opportunity.timestamp).total_seconds()
        max_age = 30 if opportunity.engine_type == 'cross_chain' else 5  # Cross-chain has longer window
        
        if age_seconds > max_age:
            return False
        
        # Check capital requirements
        if opportunity.capital_required > self.max_capital_per_trade:
            return False
        
        # Check risk limits
        if opportunity.risk_score > self.max_risk_score:
            return False
        
        return True
    
    def _can_execute_trade(self) -> bool:
        """Check if we can execute more trades"""
        if self.active_trades >= self.max_concurrent_trades:
            return False
        
        if self.current_daily_pnl <= -self.daily_loss_limit:
            return False
        
        return True
    
    async def _execute_opportunity(self, opportunity: OpportunityMetrics) -> Dict:
        """Execute an opportunity through the appropriate engine"""
        try:
            self.active_trades += 1
            execution_start = time.time()
            
            # Get the appropriate engine
            engine = self.engines.get(opportunity.engine_type)
            if not engine:
                return {
                    'success': False,
                    'error': f'Engine {opportunity.engine_type} not found',
                    'opportunity_id': opportunity.opportunity_id
                }
            
            # Find the original opportunity in the engine
            original_opp = None
            if hasattr(engine, 'active_opportunities'):
                original_opp = next(
                    (opp for opp in engine.active_opportunities 
                     if getattr(opp, 'opportunity_id', '') == opportunity.opportunity_id),
                    None
                )
            
            if not original_opp:
                return {
                    'success': False,
                    'error': 'Original opportunity not found in engine',
                    'opportunity_id': opportunity.opportunity_id
                }
            
            # Execute through the appropriate engine method
            if hasattr(engine, '_execute_cross_chain_arbitrage'):
                result = await engine._execute_cross_chain_arbitrage(original_opp)
            elif hasattr(engine, '_execute_dex_arbitrage'):
                result = await engine._execute_dex_arbitrage(original_opp)
            else:
                return {
                    'success': False,
                    'error': 'No execution method found in engine',
                    'opportunity_id': opportunity.opportunity_id
                }
            
            execution_time = (time.time() - execution_start) * 1000
            
            # Add orchestrator metadata
            result['orchestrator_metadata'] = {
                'engine_type': opportunity.engine_type,
                'priority_score': opportunity.priority_score,
                'orchestrator_execution_time_ms': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing opportunity: {e}")
            return {
                'success': False,
                'error': str(e),
                'opportunity_id': opportunity.opportunity_id
            }
        finally:
            self.active_trades = max(0, self.active_trades - 1)
    
    def _update_execution_metrics(self, result: Dict):
        """Update execution metrics based on result"""
        try:
            engine_type = result.get('orchestrator_metadata', {}).get('engine_type', 'unknown')
            
            if result.get('success'):
                profit = result.get('actual_profit', 0)
                self.current_daily_pnl += profit
                logger.info(f"✅ Successful execution: ${profit:.2f} via {engine_type}")
            else:
                logger.warning(f"❌ Failed execution via {engine_type}: {result.get('error', 'Unknown error')}")
            
            # Store execution history
            self.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'daily_pnl': self.current_daily_pnl
            })
            
            # Keep only last 1000 executions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
        
        except Exception as e:
            logger.error(f"Error updating execution metrics: {e}")
    
    def _calculate_engine_performance_score(self, perf: EnginePerformance) -> float:
        """Calculate performance score for an engine"""
        if perf.successful_trades == 0:
            return 1.0  # Default score for new engines
        
        # Factors: success rate, profit per trade, recent activity
        success_factor = perf.success_rate
        profit_factor = min(1.0, perf.avg_profit_per_trade / 50.0)  # Normalize to $50
        activity_factor = min(1.0, perf.active_opportunities / 10.0)  # Normalize to 10 opportunities
        
        performance_score = (success_factor * 0.5 + 
                           profit_factor * 0.3 + 
                           activity_factor * 0.2)
        
        return performance_score
    
    def _assess_market_volatility(self) -> float:
        """Assess current market volatility"""
        # Simulate volatility assessment based on recent execution results
        if len(self.execution_history) < 10:
            return 0.3  # Low volatility assumption
        
        recent_results = self.execution_history[-10:]
        success_count = sum(1 for r in recent_results if r['result'].get('success'))
        volatility = 1.0 - (success_count / len(recent_results))
        
        return volatility
    
    def _adjust_risk_parameters(self, volatility: float):
        """Adjust risk parameters based on market conditions"""
        if volatility > 0.7:
            self.profit_threshold *= 1.2
            self.min_confidence *= 1.1
            self.max_risk_score *= 0.9
            logger.info("🎛️ Risk parameters tightened due to high volatility")
    
    def _reduce_risk_exposure(self):
        """Reduce risk exposure when hitting limits"""
        self.max_concurrent_trades = max(1, self.max_concurrent_trades - 1)
        self.max_capital_per_trade *= 0.8
        logger.warning("🔒 Risk exposure reduced due to losses")
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics"""
        total_executions = self.total_opportunities_executed
        execution_rate = total_executions / max(1, self.total_opportunities_detected)
        
        # Calculate hourly profit rate
        uptime_hours = 1  # Simplified - would track actual uptime
        hourly_profit_rate = self.total_system_profit / max(1, uptime_hours)
        
        return {
            'total_system_profit_usd': self.total_system_profit,
            'current_daily_pnl': self.current_daily_pnl,
            'total_opportunities_detected': self.total_opportunities_detected,
            'total_opportunities_executed': self.total_opportunities_executed,
            'execution_rate': execution_rate,
            'active_opportunities': len(self.opportunity_queue),
            'active_trades': self.active_trades,
            'hourly_profit_rate': hourly_profit_rate,
            'engine_performances': {
                name: {
                    'profit': perf.total_profit,
                    'success_rate': perf.success_rate,
                    'performance_score': perf.performance_score
                }
                for name, perf in self.engine_performances.items()
            },
            'risk_parameters': {
                'profit_threshold': self.profit_threshold,
                'min_confidence': self.min_confidence,
                'max_risk_score': self.max_risk_score,
                'max_concurrent_trades': self.max_concurrent_trades
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def stop_orchestration(self):
        """Stop all engines and orchestration"""
        self.is_running = False
        
        # Stop all engines
        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'stop_cross_chain_hunting'):
                    await engine.stop_cross_chain_hunting()
                elif hasattr(engine, 'stop_dex_hunting'):
                    await engine.stop_dex_hunting()
            except Exception as e:
                logger.error(f"Error stopping {engine_name}: {e}")
        
        logger.info("🛑 Master Arbitrage Orchestration stopped")

# Example usage
async def main():
    """Demonstrate master arbitrage orchestrator"""
    orchestrator = MasterArbitrageOrchestrator()
    
    try:
        # Initialize all engines
        await orchestrator.initialize_engines()
        
        # Run orchestration for demo
        await asyncio.wait_for(orchestrator.start_orchestration(), timeout=120.0)
    except asyncio.TimeoutError:
        await orchestrator.stop_orchestration()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        await orchestrator.stop_orchestration()
    
    # Display final metrics
    metrics = orchestrator.get_orchestrator_metrics()
    print("\n🎯 MASTER ARBITRAGE ORCHESTRATOR METRICS 🎯")
    print("=" * 60)
    print(f"Total System Profit: ${metrics['total_system_profit_usd']:.2f}")
    print(f"Current Daily PnL: ${metrics['current_daily_pnl']:.2f}")
    print(f"Opportunities Detected: {metrics['total_opportunities_detected']}")
    print(f"Opportunities Executed: {metrics['total_opportunities_executed']}")
    print(f"Execution Rate: {metrics['execution_rate']:.1%}")
    print(f"Hourly Profit Rate: ${metrics['hourly_profit_rate']:.2f}/hour")
    print("\nEngine Performance:")
    for engine, perf in metrics['engine_performances'].items():
        print(f"  {engine}: ${perf['profit']:.2f} (Success: {perf['success_rate']:.1%}, Score: {perf['performance_score']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())

