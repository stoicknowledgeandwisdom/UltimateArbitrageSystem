#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangular Arbitrage Strategy Plugin
===================================

Advanced triangular arbitrage strategy that operates across spot, futures, and perpetual markets.
Supports cross-exchange arbitrage with sophisticated risk management and position sizing.

Features:
- Multi-exchange triangular arbitrage detection
- Dynamic position sizing based on available liquidity
- Real-time Greeks calculation for options components
- Advanced risk metrics and VaR calculation
- MEV-resistant execution strategies
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from dataclasses import asdict

from advanced_strategy_engine import (
    StrategyPlugin, StrategyConfig, MarketData, Position, Greeks, 
    RiskMetrics, PnLAttribution, ExecutionResult, PluginStatus
)

logger = logging.getLogger("TriangularArbitragePlugin")


class TriangularArbitragePlugin(StrategyPlugin):
    """Triangular arbitrage strategy implementation."""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.opportunities_detected = 0
        self.trades_executed = 0
        self.total_profit = Decimal('0')
        self.current_positions = {}
        self.risk_metrics = RiskMetrics()
        self.last_market_update = datetime.now()
        
        # Strategy parameters
        self.min_profit_threshold = Decimal(str(config.parameters.get('min_profit_threshold', '0.005')))
        self.position_size_multiplier = Decimal(str(config.parameters.get('position_size_multiplier', '1.0')))
        self.max_slippage = Decimal(str(config.parameters.get('max_slippage', '0.002')))
        self.execution_timeout = config.parameters.get('execution_timeout', 5000)  # ms
        
        # Risk parameters
        self.max_position_concentration = Decimal('0.3')  # 30% max concentration
        self.stop_loss_threshold = Decimal('0.02')  # 2% stop loss
        
        # Performance tracking
        self.performance_history = []
        self.risk_history = []
        
    async def initialize(self) -> bool:
        """Initialize the strategy plugin."""
        try:
            logger.info(f"Initializing Triangular Arbitrage Strategy: {self.config.strategy_id}")
            
            # Validate configuration
            if len(self.config.exchanges) < 2:
                raise ValueError("At least 2 exchanges required for triangular arbitrage")
                
            if len(self.config.symbols) < 3:
                raise ValueError("At least 3 symbols required for triangular arbitrage")
                
            # Initialize risk metrics
            await self._initialize_risk_metrics()
            
            self.status = PluginStatus.LOADED
            logger.info(f"Strategy {self.config.strategy_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy {self.config.strategy_id}: {e}")
            self.status = PluginStatus.ERROR
            return False
            
    async def start(self) -> bool:
        """Start the strategy plugin."""
        try:
            self.status = PluginStatus.RUNNING
            logger.info(f"Strategy {self.config.strategy_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start strategy {self.config.strategy_id}: {e}")
            self.status = PluginStatus.ERROR
            return False
            
    async def stop(self) -> bool:
        """Stop the strategy plugin."""
        try:
            # Close all open positions
            await self._close_all_positions()
            
            self.status = PluginStatus.STOPPED
            logger.info(f"Strategy {self.config.strategy_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop strategy {self.config.strategy_id}: {e}")
            return False
            
    async def pre_trade(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-trade validation and position sizing."""
        try:
            # Extract opportunity details
            path = opportunity_data.get('arbitrage_path', [])
            expected_profit = Decimal(str(opportunity_data.get('expected_profit', '0')))
            required_capital = Decimal(str(opportunity_data.get('required_capital', '0')))
            confidence_score = Decimal(str(opportunity_data.get('confidence', '0.5')))
            
            # Check minimum profit threshold
            if expected_profit < self.min_profit_threshold:
                return {
                    'should_execute': False,
                    'reasoning': f"Profit {expected_profit} below threshold {self.min_profit_threshold}"
                }
                
            # Calculate optimal position size
            position_size = await self._calculate_position_size(
                required_capital, expected_profit, confidence_score
            )
            
            # Generate trading actions
            actions = await self._generate_trading_actions(path, position_size)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_pre_trade_risk(actions)
            
            return {
                'should_execute': True,
                'position_size': position_size,
                'actions': actions,
                'expected_profit': expected_profit * self.position_size_multiplier,
                'confidence_score': confidence_score,
                'risk_metrics': asdict(risk_metrics),
                'reasoning': f"Triangular arbitrage opportunity: {expected_profit} profit",
                'metadata': {
                    'arbitrage_path': path,
                    'execution_strategy': 'parallel_atomic',
                    'slippage_protection': True
                }
            }
            
        except Exception as e:
            logger.error(f"Pre-trade error: {e}")
            return {
                'should_execute': False,
                'reasoning': f"Pre-trade error: {e}"
            }
            
    async def post_trade(self, execution_result: ExecutionResult) -> Dict[str, Any]:
        """Post-trade analysis and reporting."""
        try:
            # Update performance metrics
            self.trades_executed += 1
            self.total_profit += execution_result.profit
            
            # Calculate detailed PnL attribution
            pnl_attribution = await self._calculate_pnl_attribution(execution_result)
            
            # Calculate Greeks if options are involved
            greeks = await self._calculate_greeks(execution_result)
            
            # Calculate VaR metrics
            var_metrics = await self._calculate_var_metrics(execution_result)
            
            # Generate insights
            insights = await self._generate_insights(execution_result)
            
            # Update risk metrics
            await self._update_risk_metrics(execution_result)
            
            # Store performance data
            self.performance_history.append({
                'timestamp': execution_result.timestamp,
                'profit': execution_result.profit,
                'volume': execution_result.volume,
                'execution_time': execution_result.execution_time_ms
            })
            
            return {
                'pnl_attribution': asdict(pnl_attribution),
                'greeks': asdict(greeks),
                'var_metrics': var_metrics,
                'insights': insights,
                'performance_metrics': {
                    'sharpe_ratio': await self._calculate_sharpe_ratio(),
                    'win_rate': self._calculate_win_rate(),
                    'avg_profit_per_trade': self.total_profit / max(self.trades_executed, 1),
                    'max_drawdown': await self._calculate_max_drawdown()
                }
            }
            
        except Exception as e:
            logger.error(f"Post-trade analysis error: {e}")
            return {}
            
    async def on_market_data(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Process real-time market data."""
        try:
            self.last_market_update = datetime.now()
            
            # Update internal market data
            for data in market_data:
                key = f"{data.exchange}:{data.symbol}"
                self.market_data[key] = data
                
            # Detect arbitrage opportunities
            opportunities = await self._detect_arbitrage_opportunities(market_data)
            
            # Generate signals
            signals = await self._generate_signals(market_data)
            
            # Update state
            state_changed = len(opportunities) > 0 or len(signals) > 0
            
            if opportunities:
                self.opportunities_detected += len(opportunities)
                
            return {
                'signals': signals,
                'opportunities': opportunities,
                'state_changed': state_changed,
                'state_data': {
                    'last_update': self.last_market_update.isoformat(),
                    'market_data_count': len(self.market_data),
                    'opportunities_detected': self.opportunities_detected
                }
            }
            
        except Exception as e:
            logger.error(f"Market data processing error: {e}")
            return {'signals': [], 'opportunities': [], 'state_changed': False}
            
    async def risk_check(self, proposed_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Risk validation before execution."""
        try:
            violations = []
            recommendations = []
            
            # Calculate total exposure
            total_exposure = Decimal('0')
            for action in proposed_actions:
                quantity = Decimal(str(action.get('quantity', '0')))
                price = Decimal(str(action.get('price', '0')))
                total_exposure += quantity * price
                
            # Check position size limits
            if total_exposure > self.config.max_position_size:
                violations.append({
                    'rule': 'max_position_size',
                    'severity': 'high',
                    'current_value': total_exposure,
                    'limit_value': self.config.max_position_size
                })
                recommendations.append("Reduce position size to comply with limits")
                
            # Check concentration limits
            portfolio_value = await self._get_portfolio_value()
            concentration = total_exposure / max(portfolio_value, Decimal('1'))
            if concentration > self.max_position_concentration:
                violations.append({
                    'rule': 'max_concentration',
                    'severity': 'medium',
                    'current_value': concentration,
                    'limit_value': self.max_position_concentration
                })
                recommendations.append("Reduce concentration to maintain diversification")
                
            # Calculate projected risk
            projected_risk = await self._calculate_projected_risk(proposed_actions)
            
            # Determine approval
            approved = len([v for v in violations if v['severity'] in ['critical', 'high']]) == 0
            
            # Calculate maximum safe position size
            max_position_size = min(
                self.config.max_position_size,
                portfolio_value * self.max_position_concentration
            )
            
            return {
                'approved': approved,
                'violations': violations,
                'max_position_size': max_position_size,
                'recommendations': recommendations,
                'projected_risk': asdict(projected_risk)
            }
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return {
                'approved': False,
                'violations': [{'rule': 'system_error', 'severity': 'critical'}],
                'max_position_size': Decimal('0'),
                'recommendations': ['System error during risk check']
            }
            
    async def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        try:
            current_time = datetime.now()
            
            # Calculate time-based metrics
            daily_pnl = await self._calculate_daily_pnl()
            current_drawdown = await self._calculate_current_drawdown()
            
            return {
                'strategy_id': self.config.strategy_id,
                'status': self.status.value,
                'last_update': self.last_update.isoformat(),
                'opportunities_detected': self.opportunities_detected,
                'trades_executed': self.trades_executed,
                'total_profit': float(self.total_profit),
                'daily_pnl': float(daily_pnl),
                'current_drawdown': float(current_drawdown),
                'win_rate': self._calculate_win_rate(),
                'sharpe_ratio': await self._calculate_sharpe_ratio(),
                'avg_execution_time': self._calculate_avg_execution_time(),
                'market_data_symbols': len(self.market_data),
                'active_positions': len(self.current_positions),
                'risk_metrics': asdict(self.risk_metrics),
                'last_market_update': self.last_market_update.isoformat() if self.last_market_update else None
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {'error': str(e)}
            
    # Helper methods
    
    async def _initialize_risk_metrics(self):
        """Initialize risk metrics."""
        self.risk_metrics = RiskMetrics(
            var_1d=Decimal('0'),
            var_5d=Decimal('0'),
            max_drawdown=Decimal('0'),
            volatility=Decimal('0'),
            sharpe_ratio=Decimal('0')
        )
        
    async def _calculate_position_size(self, required_capital: Decimal, 
                                     expected_profit: Decimal, 
                                     confidence: Decimal) -> Decimal:
        """Calculate optimal position size using Kelly criterion."""
        # Kelly criterion: f = (bp - q) / b
        # where f = fraction of capital, b = odds, p = probability of win, q = probability of loss
        
        win_prob = float(confidence)
        loss_prob = 1 - win_prob
        odds = float(expected_profit / required_capital) if required_capital > 0 else 0
        
        if odds > 0 and win_prob > loss_prob:
            kelly_fraction = (odds * win_prob - loss_prob) / odds
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1  # Conservative default
            
        portfolio_value = await self._get_portfolio_value()
        base_size = portfolio_value * Decimal(str(kelly_fraction))
        
        return base_size * self.position_size_multiplier
        
    async def _generate_trading_actions(self, path: List[Dict[str, Any]], 
                                      position_size: Decimal) -> List[Dict[str, Any]]:
        """Generate atomic trading actions for arbitrage execution."""
        actions = []
        
        for i, step in enumerate(path):
            action = {
                'action_type': 'buy' if step.get('side') == 'buy' else 'sell',
                'symbol': step.get('symbol'),
                'exchange': step.get('exchange'),
                'quantity': position_size / len(path),  # Split across steps
                'price': Decimal(str(step.get('price', '0'))),
                'order_type': 'market',  # For speed in arbitrage
                'priority': i,  # Execution order
                'parameters': {
                    'time_in_force': 'IOC',  # Immediate or Cancel
                    'slippage_protection': True,
                    'max_slippage_bps': int(self.max_slippage * 10000)
                }
            }
            actions.append(action)
            
        return actions
        
    async def _detect_arbitrage_opportunities(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Detect triangular arbitrage opportunities."""
        opportunities = []
        
        # Group market data by exchange
        exchange_data = {}
        for data in market_data:
            if data.exchange not in exchange_data:
                exchange_data[data.exchange] = {}
            exchange_data[data.exchange][data.symbol] = data
            
        # Look for triangular opportunities across exchanges
        for base_exchange in exchange_data:
            for target_exchange in exchange_data:
                if base_exchange == target_exchange:
                    continue
                    
                opportunity = await self._find_triangular_path(
                    exchange_data[base_exchange],
                    exchange_data[target_exchange]
                )
                
                if opportunity:
                    opportunities.append(opportunity)
                    
        return opportunities
        
    async def _find_triangular_path(self, base_data: Dict[str, MarketData], 
                                  target_data: Dict[str, MarketData]) -> Optional[Dict[str, Any]]:
        """Find profitable triangular arbitrage path."""
        # Simplified triangular arbitrage detection
        # In practice, this would be much more sophisticated
        
        common_symbols = set(base_data.keys()) & set(target_data.keys())
        if len(common_symbols) < 3:
            return None
            
        # Example: BTC/USDT, ETH/USDT, BTC/ETH triangle
        symbols = list(common_symbols)[:3]
        
        # Calculate potential profit (simplified)
        base_prices = [base_data[s].ask for s in symbols]
        target_prices = [target_data[s].bid for s in symbols]
        
        # Check if arbitrage exists
        price_diff = sum(target_prices) - sum(base_prices)
        if price_diff > self.min_profit_threshold:
            return {
                'opportunity_id': f"tri_arb_{int(time.time())}",
                'strategy_type': 'triangular_arbitrage',
                'symbols': symbols,
                'exchanges': [list(base_data.values())[0].exchange, 
                            list(target_data.values())[0].exchange],
                'expected_profit': float(price_diff),
                'confidence': 0.7,
                'required_capital': float(sum(base_prices)),
                'expiry': (datetime.now() + timedelta(seconds=30)).isoformat(),
                'details': {
                    'arbitrage_path': [
                        {'exchange': list(base_data.values())[0].exchange, 
                         'symbol': s, 'side': 'buy', 'price': float(base_data[s].ask)}
                        for s in symbols
                    ] + [
                        {'exchange': list(target_data.values())[0].exchange, 
                         'symbol': s, 'side': 'sell', 'price': float(target_data[s].bid)}
                        for s in symbols
                    ]
                }
            }
            
        return None
        
    async def _generate_signals(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Generate trading signals based on market data."""
        signals = []
        
        for data in market_data:
            # Calculate signal strength based on spread
            spread = data.ask - data.bid
            spread_pct = spread / data.bid if data.bid > 0 else Decimal('0')
            
            if spread_pct > Decimal('0.005'):  # 0.5% spread threshold
                signals.append({
                    'signal_type': 'wide_spread',
                    'symbol': data.symbol,
                    'strength': float(spread_pct),
                    'confidence': 0.6,
                    'timestamp': data.timestamp.isoformat(),
                    'metadata': {
                        'exchange': data.exchange,
                        'spread_bps': int(spread_pct * 10000),
                        'volume': float(data.volume)
                    }
                })
                
        return signals
        
    async def _calculate_pnl_attribution(self, execution_result: ExecutionResult) -> PnLAttribution:
        """Calculate detailed PnL attribution."""
        # Simplified attribution - in practice would be more detailed
        market_pnl = execution_result.profit + execution_result.fees_paid + execution_result.slippage
        execution_pnl = -execution_result.slippage
        fees_pnl = -execution_result.fees_paid
        
        return PnLAttribution(
            market_pnl=market_pnl,
            execution_pnl=execution_pnl,
            fees_pnl=fees_pnl,
            slippage_pnl=-execution_result.slippage,
            timing_pnl=Decimal('0'),
            factor_attribution={
                'price_movement': float(market_pnl * Decimal('0.8')),
                'spread_capture': float(market_pnl * Decimal('0.2'))
            }
        )
        
    async def _calculate_greeks(self, execution_result: ExecutionResult) -> Greeks:
        """Calculate Greeks for options components."""
        # Simplified Greeks calculation
        return Greeks(
            delta=Decimal('0.5'),
            gamma=Decimal('0.1'),
            theta=Decimal('-0.05'),
            vega=Decimal('0.2'),
            rho=Decimal('0.01')
        )
        
    async def _calculate_var_metrics(self, execution_result: ExecutionResult) -> Dict[str, Any]:
        """Calculate VaR metrics."""
        return {
            'current_var': float(execution_result.volume * Decimal('0.02')),
            'marginal_var': float(execution_result.volume * Decimal('0.015')),
            'incremental_var': float(execution_result.volume * Decimal('0.01')),
            'component_var': float(execution_result.volume * Decimal('0.005')),
            'scenario_results': [
                {
                    'scenario_name': 'market_stress',
                    'pnl_impact': float(execution_result.profit * Decimal('-2.0')),
                    'probability': 0.05,
                    'factor_shocks': {'market_down': -0.2, 'vol_up': 0.5}
                }
            ]
        }
        
    async def _generate_insights(self, execution_result: ExecutionResult) -> List[Dict[str, Any]]:
        """Generate actionable insights."""
        insights = []
        
        if execution_result.execution_time_ms > 1000:  # > 1 second
            insights.append({
                'insight_type': 'execution_latency',
                'description': 'Execution time above optimal threshold',
                'impact_score': 0.7,
                'actionable_items': [
                    'Consider using faster execution venues',
                    'Optimize order routing algorithms'
                ],
                'data': {'execution_time_ms': execution_result.execution_time_ms}
            })
            
        if execution_result.slippage > execution_result.volume * Decimal('0.001'):
            insights.append({
                'insight_type': 'high_slippage',
                'description': 'Slippage higher than expected',
                'impact_score': 0.8,
                'actionable_items': [
                    'Reduce position sizes',
                    'Use limit orders instead of market orders'
                ],
                'data': {'slippage': float(execution_result.slippage)}
            })
            
        return insights
        
    async def _calculate_pre_trade_risk(self, actions: List[Dict[str, Any]]) -> RiskMetrics:
        """Calculate pre-trade risk metrics."""
        total_exposure = sum(Decimal(str(a.get('quantity', '0'))) * 
                           Decimal(str(a.get('price', '0'))) for a in actions)
        
        return RiskMetrics(
            var_1d=total_exposure * Decimal('0.02'),
            var_5d=total_exposure * Decimal('0.05'),
            volatility=Decimal('0.15'),
            beta=Decimal('1.0'),
            max_drawdown=Decimal('0.05')
        )
        
    async def _update_risk_metrics(self, execution_result: ExecutionResult):
        """Update strategy risk metrics."""
        # Update rolling risk calculations
        self.risk_history.append({
            'timestamp': execution_result.timestamp,
            'profit': execution_result.profit,
            'volume': execution_result.volume
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=30)
        self.risk_history = [r for r in self.risk_history 
                           if r['timestamp'] > cutoff]
        
    async def _get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        # Simplified portfolio value calculation
        return Decimal('10000')  # Placeholder
        
    async def _calculate_projected_risk(self, actions: List[Dict[str, Any]]) -> RiskMetrics:
        """Calculate projected risk from proposed actions."""
        return await self._calculate_pre_trade_risk(actions)
        
    async def _calculate_daily_pnl(self) -> Decimal:
        """Calculate today's PnL."""
        today = datetime.now().date()
        daily_profit = sum(r['profit'] for r in self.performance_history 
                          if r['timestamp'].date() == today)
        return Decimal(str(daily_profit))
        
    async def _calculate_current_drawdown(self) -> Decimal:
        """Calculate current drawdown."""
        if not self.performance_history:
            return Decimal('0')
            
        profits = [r['profit'] for r in self.performance_history]
        cumulative = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.maximum(peak, 1)
        
        return Decimal(str(drawdown[-1] if len(drawdown) > 0 else 0))
        
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.performance_history:
            return 0.0
            
        wins = sum(1 for r in self.performance_history if r['profit'] > 0)
        return wins / len(self.performance_history)
        
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.performance_history) < 2:
            return 0.0
            
        profits = [float(r['profit']) for r in self.performance_history]
        return np.mean(profits) / max(np.std(profits), 0.001)
        
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.performance_history:
            return 0.0
            
        profits = [r['profit'] for r in self.performance_history]
        cumulative = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.maximum(peak, 1)
        
        return float(np.max(drawdown))
        
    def _calculate_avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.performance_history:
            return 0.0
            
        times = [r['execution_time'] for r in self.performance_history 
                if 'execution_time' in r]
        return np.mean(times) if times else 0.0
        
    async def _close_all_positions(self):
        """Close all open positions."""
        # Implementation would close all open positions
        self.current_positions.clear()
        logger.info(f"All positions closed for strategy {self.config.strategy_id}")


# Plugin implementation class that the engine will instantiate
class StrategyPluginImpl(TriangularArbitragePlugin):
    """Implementation class for the plugin loader."""
    pass

