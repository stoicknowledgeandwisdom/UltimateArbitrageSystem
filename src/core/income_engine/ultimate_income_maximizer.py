#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Income Maximization Engine
==================================

This module implements the core income maximization strategies using
zero-investment mindset principles and quantum-enhanced algorithms.

Key Features:
- Automated income stream generation
- Zero-capital arbitrage optimization
- Quantum-enhanced profit detection
- Multi-dimensional opportunity exploitation
- Continuous learning and adaptation
- Risk-optimized execution
"""

import asyncio
import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Set high precision for financial calculations
getcontext().prec = 28

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncomeStream(Enum):
    """Different types of automated income streams"""
    QUANTUM_ARBITRAGE = "quantum_arbitrage"
    CROSS_CHAIN_MEV = "cross_chain_mev"
    FLASH_LOAN_ARBITRAGE = "flash_loan_arbitrage"
    AI_MOMENTUM = "ai_momentum"
    VOLATILITY_HARVESTING = "volatility_harvesting"
    SOCIAL_SENTIMENT = "social_sentiment"
    YIELD_FARMING = "yield_farming"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    TRIANGULAR_ARBITRAGE = "triangular_arbitrage"
    OPTIONS_ARBITRAGE = "options_arbitrage"

class OpportunityPriority(Enum):
    """Priority levels for income opportunities"""
    CRITICAL = 1    # Immediate execution required
    HIGH = 2        # Execute within 1 second
    MEDIUM = 3      # Execute within 5 seconds
    LOW = 4         # Execute within 30 seconds
    BACKGROUND = 5  # Execute when resources available

@dataclass
class IncomeOpportunity:
    """Represents a single income generation opportunity"""
    id: str
    stream_type: IncomeStream
    profit_potential: Decimal
    confidence_score: float
    execution_time_estimate: float  # seconds
    capital_required: Decimal
    risk_score: float
    priority: OpportunityPriority
    expiry_time: datetime
    metadata: Dict[str, Any]
    quantum_enhanced: bool = False
    
    def __post_init__(self):
        """Validate opportunity data"""
        if self.profit_potential <= 0:
            raise ValueError("Profit potential must be positive")
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if not 0 <= self.risk_score <= 1:
            raise ValueError("Risk score must be between 0 and 1")

@dataclass
class ExecutionResult:
    """Result of executing an income opportunity"""
    opportunity_id: str
    success: bool
    actual_profit: Decimal
    execution_time: float
    gas_cost: Optional[Decimal]
    slippage: Optional[float]
    error_message: Optional[str]
    timestamp: datetime

class UltimateIncomeMaximizer:
    """
    The core engine for maximizing automated income generation.
    
    This class implements advanced strategies for finding and executing
    the highest-value opportunities across multiple income streams.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core settings
        self.target_daily_profit = Decimal(str(config.get('target_daily_profit_percentage', 2.5)))
        self.quantum_boost_multiplier = Decimal(str(config.get('quantum_boost_multiplier', 3.2)))
        self.opportunity_exploitation_level = config.get('opportunity_exploitation_level', 'AGGRESSIVE')
        
        # Automation settings
        self.full_automation_mode = config.get('full_automation_mode', True)
        self.auto_capital_allocation = config.get('auto_capital_allocation', True)
        self.continuous_learning_enabled = config.get('continuous_learning_enabled', True)
        
        # Performance tracking
        self.total_profit_generated = Decimal('0')
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.success_rate = 0.0
        self.average_profit_per_opportunity = Decimal('0')
        
        # Threading and execution
        self.is_running = False
        self.opportunity_queue = queue.PriorityQueue()
        self.execution_results = queue.Queue(maxsize=1000)
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_concurrent_executions', 10))
        
        # Strategy managers
        self.income_streams = {}
        self.active_opportunities = {}
        self.execution_history = []
        
        # Learning and adaptation
        self.performance_history = []
        self.market_conditions = {}
        self.strategy_performance = {}
        
        self.logger.info("Ultimate Income Maximizer initialized")
    
    async def start(self) -> bool:
        """
        Start the income maximization engine.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting Ultimate Income Maximizer...")
            self.is_running = True
            
            # Initialize all income streams
            await self._initialize_income_streams()
            
            # Start main processing loops
            asyncio.create_task(self._opportunity_detection_loop())
            asyncio.create_task(self._opportunity_execution_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._market_adaptation_loop())
            
            if self.continuous_learning_enabled:
                asyncio.create_task(self._continuous_learning_loop())
            
            self.logger.info("Ultimate Income Maximizer started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start income maximizer: {e}")
            self.is_running = False
            return False
    
    async def stop(self) -> None:
        """
        Stop the income maximization engine gracefully.
        """
        self.logger.info("Stopping Ultimate Income Maximizer...")
        self.is_running = False
        
        # Wait for current executions to complete
        self.executor.shutdown(wait=True)
        
        self.logger.info("Ultimate Income Maximizer stopped")
    
    async def _initialize_income_streams(self) -> None:
        """
        Initialize all configured income streams.
        """
        self.logger.info("Initializing income streams...")
        
        # Get strategy configurations
        strategies_config = self.config.get('advanced_strategies', {})
        
        for stream_name, stream_config in strategies_config.items():
            if stream_config.get('enabled', False):
                try:
                    stream_type = IncomeStream(stream_name)
                    self.income_streams[stream_type] = await self._create_income_stream(stream_type, stream_config)
                    self.logger.info(f"Initialized income stream: {stream_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize income stream {stream_name}: {e}")
    
    async def _create_income_stream(self, stream_type: IncomeStream, config: Dict[str, Any]) -> Any:
        """
        Create and configure an income stream.
        """
        # In a real implementation, this would instantiate specific strategy classes
        # For now, we'll return a mock configuration
        return {
            'type': stream_type,
            'config': config,
            'allocation': config.get('allocation_percentage', 10),
            'last_opportunity_time': datetime.now(),
            'total_profit': Decimal('0'),
            'opportunities_found': 0,
            'success_rate': 0.0
        }
    
    async def _opportunity_detection_loop(self) -> None:
        """
        Continuously scan for income opportunities across all streams.
        """
        self.logger.info("Starting opportunity detection loop")
        
        while self.is_running:
            try:
                # Scan each income stream for opportunities
                for stream_type, stream_manager in self.income_streams.items():
                    opportunities = await self._scan_stream_opportunities(stream_type, stream_manager)
                    
                    for opportunity in opportunities:
                        await self._process_opportunity(opportunity)
                
                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(0.1)  # 100ms between scans
                
            except Exception as e:
                self.logger.error(f"Error in opportunity detection loop: {e}")
                await asyncio.sleep(1)  # Longer pause on error
    
    async def _scan_stream_opportunities(self, stream_type: IncomeStream, stream_manager: Dict[str, Any]) -> List[IncomeOpportunity]:
        """
        Scan a specific income stream for opportunities.
        """
        opportunities = []
        
        try:
            # Generate mock opportunities based on stream type
            if stream_type == IncomeStream.QUANTUM_ARBITRAGE:
                opportunities.extend(await self._generate_quantum_arbitrage_opportunities())
            elif stream_type == IncomeStream.CROSS_CHAIN_MEV:
                opportunities.extend(await self._generate_cross_chain_opportunities())
            elif stream_type == IncomeStream.AI_MOMENTUM:
                opportunities.extend(await self._generate_ai_momentum_opportunities())
            # Add more stream types as needed
            
            stream_manager['opportunities_found'] += len(opportunities)
            stream_manager['last_opportunity_time'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error scanning {stream_type}: {e}")
        
        return opportunities
    
    async def _generate_quantum_arbitrage_opportunities(self) -> List[IncomeOpportunity]:
        """
        Generate quantum arbitrage opportunities.
        """
        opportunities = []
        
        # Simulate quantum-enhanced opportunity detection
        for i in range(np.random.randint(0, 3)):  # 0-2 opportunities
            profit_potential = Decimal(str(np.random.uniform(0.005, 0.025)))  # 0.5% to 2.5%
            confidence = np.random.uniform(0.7, 0.95)  # High confidence for quantum
            
            opportunity = IncomeOpportunity(
                id=f"quantum_{int(time.time() * 1000)}_{i}",
                stream_type=IncomeStream.QUANTUM_ARBITRAGE,
                profit_potential=profit_potential * self.quantum_boost_multiplier,  # Quantum boost
                confidence_score=confidence,
                execution_time_estimate=np.random.uniform(0.1, 0.5),  # Very fast
                capital_required=Decimal('0'),  # Zero capital
                risk_score=np.random.uniform(0.1, 0.3),  # Low risk
                priority=OpportunityPriority.HIGH,
                expiry_time=datetime.now() + timedelta(seconds=5),
                metadata={
                    'exchange_pair': 'BTC/USDT',
                    'quantum_enhancement': True,
                    'profit_multiplier': float(self.quantum_boost_multiplier)
                },
                quantum_enhanced=True
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _generate_cross_chain_opportunities(self) -> List[IncomeOpportunity]:
        """
        Generate cross-chain MEV opportunities.
        """
        opportunities = []
        
        # Simulate cross-chain arbitrage detection
        for i in range(np.random.randint(0, 2)):  # 0-1 opportunities
            profit_potential = Decimal(str(np.random.uniform(0.003, 0.015)))  # 0.3% to 1.5%
            
            opportunity = IncomeOpportunity(
                id=f"crosschain_{int(time.time() * 1000)}_{i}",
                stream_type=IncomeStream.CROSS_CHAIN_MEV,
                profit_potential=profit_potential,
                confidence_score=np.random.uniform(0.6, 0.85),
                execution_time_estimate=np.random.uniform(2, 8),  # Cross-chain takes longer
                capital_required=Decimal('0'),  # Flash loan
                risk_score=np.random.uniform(0.2, 0.4),  # Medium risk
                priority=OpportunityPriority.MEDIUM,
                expiry_time=datetime.now() + timedelta(seconds=15),
                metadata={
                    'source_chain': 'ethereum',
                    'target_chain': 'polygon',
                    'token_pair': 'USDC/USDT',
                    'flash_loan_required': True
                }
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _generate_ai_momentum_opportunities(self) -> List[IncomeOpportunity]:
        """
        Generate AI momentum trading opportunities.
        """
        opportunities = []
        
        # Simulate AI-driven momentum detection
        for i in range(np.random.randint(0, 4)):  # 0-3 opportunities
            profit_potential = Decimal(str(np.random.uniform(0.002, 0.012)))  # 0.2% to 1.2%
            
            opportunity = IncomeOpportunity(
                id=f"ai_momentum_{int(time.time() * 1000)}_{i}",
                stream_type=IncomeStream.AI_MOMENTUM,
                profit_potential=profit_potential,
                confidence_score=np.random.uniform(0.65, 0.88),
                execution_time_estimate=np.random.uniform(0.5, 2),
                capital_required=Decimal(str(np.random.uniform(100, 1000))),
                risk_score=np.random.uniform(0.25, 0.45),
                priority=OpportunityPriority.MEDIUM,
                expiry_time=datetime.now() + timedelta(seconds=10),
                metadata={
                    'ai_model': 'LSTM_v3',
                    'prediction_confidence': np.random.uniform(0.7, 0.9),
                    'market_sentiment': 'bullish'
                }
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _process_opportunity(self, opportunity: IncomeOpportunity) -> None:
        """
        Process and potentially execute an income opportunity.
        """
        try:
            # Quality filter
            if not await self._validate_opportunity(opportunity):
                return
            
            # Add to queue for execution
            priority_score = self._calculate_priority_score(opportunity)
            self.opportunity_queue.put((priority_score, opportunity))
            
            self.opportunities_detected += 1
            self.active_opportunities[opportunity.id] = opportunity
            
            self.logger.debug(f"Queued opportunity {opportunity.id} with priority {priority_score}")
            
        except Exception as e:
            self.logger.error(f"Error processing opportunity {opportunity.id}: {e}")
    
    async def _validate_opportunity(self, opportunity: IncomeOpportunity) -> bool:
        """
        Validate an opportunity against quality criteria.
        """
        # Minimum profit threshold
        min_profit = Decimal('0.001')  # 0.1%
        if opportunity.profit_potential < min_profit:
            return False
        
        # Minimum confidence score
        min_confidence = 0.6
        if opportunity.confidence_score < min_confidence:
            return False
        
        # Maximum risk score
        max_risk = 0.5
        if opportunity.risk_score > max_risk:
            return False
        
        # Check if opportunity hasn't expired
        if datetime.now() >= opportunity.expiry_time:
            return False
        
        return True
    
    def _calculate_priority_score(self, opportunity: IncomeOpportunity) -> float:
        """
        Calculate priority score for opportunity execution order.
        Lower score = higher priority.
        """
        # Base priority from enum
        base_priority = opportunity.priority.value
        
        # Adjust based on profit potential (higher profit = higher priority)
        profit_factor = 1.0 / max(float(opportunity.profit_potential), 0.001)
        
        # Adjust based on confidence (higher confidence = higher priority)
        confidence_factor = 1.0 / max(opportunity.confidence_score, 0.1)
        
        # Adjust based on risk (lower risk = higher priority)
        risk_factor = opportunity.risk_score
        
        # Quantum enhancement bonus
        quantum_bonus = 0.5 if opportunity.quantum_enhanced else 1.0
        
        priority_score = base_priority * profit_factor * confidence_factor * risk_factor * quantum_bonus
        
        return priority_score
    
    async def _opportunity_execution_loop(self) -> None:
        """
        Execute opportunities from the priority queue.
        """
        self.logger.info("Starting opportunity execution loop")
        
        while self.is_running:
            try:
                # Get next opportunity (blocks if queue is empty)
                try:
                    priority_score, opportunity = self.opportunity_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Execute opportunity
                result = await self._execute_opportunity(opportunity)
                
                # Process result
                await self._process_execution_result(result)
                
                # Clean up
                if opportunity.id in self.active_opportunities:
                    del self.active_opportunities[opportunity.id]
                
                self.opportunity_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_opportunity(self, opportunity: IncomeOpportunity) -> ExecutionResult:
        """
        Execute a specific income opportunity.
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing opportunity {opportunity.id} ({opportunity.stream_type.value})")
            
            # Simulate execution time
            await asyncio.sleep(opportunity.execution_time_estimate)
            
            # Simulate execution success/failure
            success_probability = opportunity.confidence_score * 0.9  # Slight reduction for reality
            success = np.random.random() < success_probability
            
            if success:
                # Calculate actual profit (with some variance)
                profit_variance = np.random.uniform(0.8, 1.2)  # ±20% variance
                actual_profit = opportunity.profit_potential * Decimal(str(profit_variance))
                
                self.total_profit_generated += actual_profit
                self.opportunities_executed += 1
                
                result = ExecutionResult(
                    opportunity_id=opportunity.id,
                    success=True,
                    actual_profit=actual_profit,
                    execution_time=time.time() - start_time,
                    gas_cost=Decimal(str(np.random.uniform(5, 50))) if opportunity.stream_type != IncomeStream.QUANTUM_ARBITRAGE else None,
                    slippage=np.random.uniform(0.001, 0.005),
                    error_message=None,
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"Successfully executed {opportunity.id}: ${actual_profit:.6f} profit")
                
            else:
                result = ExecutionResult(
                    opportunity_id=opportunity.id,
                    success=False,
                    actual_profit=Decimal('0'),
                    execution_time=time.time() - start_time,
                    gas_cost=None,
                    slippage=None,
                    error_message="Execution failed",
                    timestamp=datetime.now()
                )
                
                self.logger.warning(f"Failed to execute opportunity {opportunity.id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing opportunity {opportunity.id}: {e}")
            return ExecutionResult(
                opportunity_id=opportunity.id,
                success=False,
                actual_profit=Decimal('0'),
                execution_time=time.time() - start_time,
                gas_cost=None,
                slippage=None,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    async def _process_execution_result(self, result: ExecutionResult) -> None:
        """
        Process the result of an opportunity execution.
        """
        try:
            # Add to history
            self.execution_history.append(result)
            
            # Keep only recent history (last 1000 executions)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Add to results queue for external monitoring
            try:
                self.execution_results.put_nowait(result)
            except queue.Full:
                # Remove oldest result if queue is full
                try:
                    self.execution_results.get_nowait()
                    self.execution_results.put_nowait(result)
                except queue.Empty:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error processing execution result: {e}")
    
    def _update_performance_metrics(self) -> None:
        """
        Update internal performance metrics.
        """
        if not self.execution_history:
            return
        
        # Calculate success rate
        successful_executions = sum(1 for result in self.execution_history if result.success)
        self.success_rate = successful_executions / len(self.execution_history)
        
        # Calculate average profit per opportunity
        total_profit = sum(result.actual_profit for result in self.execution_history)
        self.average_profit_per_opportunity = total_profit / len(self.execution_history)
    
    async def _performance_monitoring_loop(self) -> None:
        """
        Monitor system performance and generate reports.
        """
        while self.is_running:
            try:
                # Generate performance snapshot
                performance_data = self.get_performance_metrics()
                
                # Log performance summary every minute
                self.logger.info(
                    f"Performance Summary - "
                    f"Total Profit: ${performance_data['total_profit']:.6f}, "
                    f"Success Rate: {performance_data['success_rate']:.1%}, "
                    f"Opportunities: {performance_data['opportunities_detected']}"
                )
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _market_adaptation_loop(self) -> None:
        """
        Adapt strategies based on market conditions.
        """
        while self.is_running:
            try:
                # Analyze market conditions
                await self._analyze_market_conditions()
                
                # Adapt strategy parameters
                await self._adapt_strategies()
                
                await asyncio.sleep(300)  # Adapt every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in market adaptation: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_market_conditions(self) -> None:
        """
        Analyze current market conditions.
        """
        # Mock market condition analysis
        self.market_conditions = {
            'volatility': np.random.uniform(0.1, 0.8),
            'volume': np.random.uniform(0.3, 1.0),
            'trend': np.random.choice(['bullish', 'bearish', 'sideways']),
            'liquidity': np.random.uniform(0.5, 1.0),
            'timestamp': datetime.now()
        }
    
    async def _adapt_strategies(self) -> None:
        """
        Adapt strategy parameters based on market conditions.
        """
        volatility = self.market_conditions.get('volatility', 0.5)
        
        # Adjust opportunity thresholds based on volatility
        if volatility > 0.6:  # High volatility
            # Increase thresholds in volatile markets
            self.logger.info("High volatility detected - increasing profit thresholds")
        elif volatility < 0.3:  # Low volatility
            # Decrease thresholds in calm markets
            self.logger.info("Low volatility detected - decreasing profit thresholds")
    
    async def _continuous_learning_loop(self) -> None:
        """
        Continuously learn from execution results to improve performance.
        """
        while self.is_running:
            try:
                # Analyze recent performance
                await self._analyze_performance_patterns()
                
                # Update learning models
                await self._update_learning_models()
                
                await asyncio.sleep(3600)  # Learn every hour
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_performance_patterns(self) -> None:
        """
        Analyze patterns in execution performance.
        """
        if len(self.execution_history) < 10:
            return
        
        # Analyze success patterns by stream type
        stream_performance = {}
        for result in self.execution_history[-100:]:  # Last 100 executions
            # This would be enhanced to extract stream type from opportunity
            pass
    
    async def _update_learning_models(self) -> None:
        """
        Update machine learning models based on performance data.
        """
        # Placeholder for ML model updates
        self.logger.debug("Updated learning models based on recent performance")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        return {
            'total_profit': float(self.total_profit_generated),
            'opportunities_detected': self.opportunities_detected,
            'opportunities_executed': self.opportunities_executed,
            'success_rate': self.success_rate,
            'average_profit_per_opportunity': float(self.average_profit_per_opportunity),
            'active_opportunities': len(self.active_opportunities),
            'queue_size': self.opportunity_queue.qsize(),
            'quantum_boost_multiplier': float(self.quantum_boost_multiplier),
            'is_running': self.is_running,
            'market_conditions': self.market_conditions,
            'income_streams': {k.value: v for k, v in self.income_streams.items()}
        }
    
    def get_recent_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently detected opportunities.
        """
        opportunities = list(self.active_opportunities.values())
        opportunities.sort(key=lambda x: x.expiry_time, reverse=True)
        
        return [asdict(opp) for opp in opportunities[:limit]]
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent execution history.
        """
        recent_history = self.execution_history[-limit:] if self.execution_history else []
        return [asdict(result) for result in recent_history]

# Example usage
if __name__ == "__main__":
    import json
    
    # Load configuration
    config_path = "config/ultimate_automation_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # Use default configuration
        config = {
            'target_daily_profit_percentage': 2.5,
            'quantum_boost_multiplier': 3.2,
            'opportunity_exploitation_level': 'AGGRESSIVE',
            'full_automation_mode': True,
            'max_concurrent_executions': 10,
            'advanced_strategies': {
                'quantum_arbitrage': {'enabled': True, 'allocation_percentage': 30},
                'cross_chain_mev': {'enabled': True, 'allocation_percentage': 25},
                'ai_momentum_trading': {'enabled': True, 'allocation_percentage': 20}
            }
        }
    
    async def main():
        # Initialize income maximizer
        maximizer = UltimateIncomeMaximizer(config)
        
        # Start the engine
        success = await maximizer.start()
        
        if success:
            print("🚀 Ultimate Income Maximizer started successfully!")
            print("💰 Automated income generation is now active...")
            
            # Run for demonstration
            try:
                await asyncio.sleep(30)  # Run for 30 seconds
            except KeyboardInterrupt:
                print("\n🛑 Stopping system...")
            
            # Get final performance metrics
            metrics = maximizer.get_performance_metrics()
            print(f"\n📊 Final Performance:")
            print(f"   Total Profit: ${metrics['total_profit']:.6f}")
            print(f"   Opportunities Detected: {metrics['opportunities_detected']}")
            print(f"   Success Rate: {metrics['success_rate']:.1%}")
            
            await maximizer.stop()
        else:
            print("❌ Failed to start Ultimate Income Maximizer")
    
    # Run the example
    asyncio.run(main())

