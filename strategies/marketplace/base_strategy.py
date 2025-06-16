#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Strategy Interface
======================

Advanced base class for all trading strategies with:
- Standardized signal generation
- Performance tracking and attribution
- Risk management integration
- Real-time adaptation capabilities
- Multi-timeframe analysis
- Advanced statistical measures
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
import uuid
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    REBALANCE = "rebalance"
    HEDGE = "hedge"
    CLOSE = "close"

class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class StrategySignal:
    """Standardized strategy signal."""
    strategy_id: str
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    target_weight: Optional[float] = None
    price_target: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    
    # Signal metadata
    timestamp: datetime = field(default_factory=datetime.now)
    timeframe: str = "1d"  # e.g., 1m, 5m, 1h, 1d
    expiry: Optional[datetime] = None
    
    # Strategy-specific data
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Risk metrics
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    max_drawdown_risk: Optional[float] = None
    var_contribution: Optional[float] = None
    
    @property
    def signal_score(self) -> float:
        """Calculate composite signal score."""
        base_score = self.strength.value / 5.0  # Normalize to 0-1
        confidence_weight = self.confidence
        
        # Adjust for signal type
        type_multiplier = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.8,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -0.8,
            SignalType.STRONG_SELL: -1.0,
            SignalType.REBALANCE: 0.5,
            SignalType.HEDGE: 0.3,
            SignalType.CLOSE: -0.5
        }.get(self.signal_type, 0.0)
        
        return base_score * confidence_weight * type_multiplier
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.confidence >= 0.6 and
            self.strength.value >= 3 and
            self.signal_type != SignalType.HOLD and
            (self.expiry is None or datetime.now() < self.expiry)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'target_weight': self.target_weight,
            'price_target': float(self.price_target) if self.price_target else None,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'features': self.features,
            'metadata': self.metadata,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'max_drawdown_risk': self.max_drawdown_risk,
            'var_contribution': self.var_contribution,
            'signal_score': self.signal_score,
            'is_actionable': self.is_actionable
        }

@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics."""
    strategy_id: str
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0  # vs benchmark
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_trade: float = 0.0
    trade_count: int = 0
    turnover: float = 0.0
    
    # Advanced metrics
    alpha: float = 0.0
    beta: float = 1.0
    tracking_error: float = 0.0
    downside_deviation: float = 0.0
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    
    # Strategy-specific metrics
    signal_accuracy: float = 0.0
    signal_coverage: float = 0.0  # % of time with active signals
    adaptation_speed: float = 0.0  # How quickly strategy adapts
    regime_consistency: float = 0.0  # Performance across regimes
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    inception_date: Optional[datetime] = None
    
    def calculate_composite_score(self) -> float:
        """Calculate composite strategy performance score."""
        # Weighted combination of key metrics
        return (
            self.sharpe_ratio * 0.3 +
            self.calmar_ratio * 0.2 +
            self.information_ratio * 0.2 +
            self.win_rate * 0.1 +
            self.signal_accuracy * 0.1 +
            self.regime_consistency * 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_id': self.strategy_id,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'excess_return': self.excess_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'treynor_ratio': self.treynor_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_trade': self.average_trade,
            'trade_count': self.trade_count,
            'turnover': self.turnover,
            'alpha': self.alpha,
            'beta': self.beta,
            'tracking_error': self.tracking_error,
            'downside_deviation': self.downside_deviation,
            'upside_capture': self.upside_capture,
            'downside_capture': self.downside_capture,
            'signal_accuracy': self.signal_accuracy,
            'signal_coverage': self.signal_coverage,
            'adaptation_speed': self.adaptation_speed,
            'regime_consistency': self.regime_consistency,
            'composite_score': self.calculate_composite_score(),
            'last_updated': self.last_updated.isoformat(),
            'inception_date': self.inception_date.isoformat() if self.inception_date else None
        }

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, 
                 strategy_id: str,
                 name: str,
                 config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.name = name
        self.config = config
        
        # Strategy state
        self.is_active = True
        self.is_trained = False
        self.last_signal_time = None
        
        # Performance tracking
        self.metrics = StrategyMetrics(strategy_id=strategy_id)
        self.signal_history = deque(maxlen=1000)
        self.returns_history = deque(maxlen=252)  # One year
        self.trades_history = []
        
        # Market regime tracking
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.adaptive_params = {}
        self.learning_rate = config.get('learning_rate', 0.01)
        self.adaptation_window = config.get('adaptation_window', 50)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 0.05)
        self.take_profit_threshold = config.get('take_profit_threshold', 0.15)
        
        logger.info(f"Initialized strategy: {self.name} ({self.strategy_id})")
    
    @abstractmethod
    async def generate_signal(self, 
                            market_data: Dict[str, Any],
                            portfolio_state: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate trading signal based on market data and portfolio state."""
        pass
    
    @abstractmethod
    async def update_parameters(self, 
                              market_data: Dict[str, Any],
                              performance_feedback: Dict[str, Any]):
        """Update strategy parameters based on market conditions and performance."""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Get list of required data fields for this strategy."""
        pass
    
    @abstractmethod
    def get_supported_assets(self) -> List[str]:
        """Get list of supported asset classes."""
        pass
    
    async def preprocess_data(self, 
                            raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw market data for strategy consumption."""
        # Default implementation - can be overridden
        processed_data = {}
        
        for symbol, data in raw_data.items():
            if isinstance(data, pd.DataFrame):
                # Calculate common technical indicators
                processed_data[symbol] = self._calculate_technical_indicators(data)
            else:
                processed_data[symbol] = data
        
        return processed_data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators."""
        result = df.copy()
        
        # Moving averages
        result['sma_20'] = df['close'].rolling(20).mean()
        result['sma_50'] = df['close'].rolling(50).mean()
        result['ema_12'] = df['close'].ewm(span=12).mean()
        result['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
        result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # Volatility
        result['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Volume indicators
        if 'volume' in df.columns:
            result['volume_sma'] = df['volume'].rolling(20).mean()
            result['volume_ratio'] = df['volume'] / result['volume_sma']
        
        return result
    
    async def detect_market_regime(self, 
                                 market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime."""
        # Simplified regime detection - can be enhanced
        try:
            # Get market proxy (e.g., SPY)
            market_proxy = market_data.get('SPY', market_data.get('QQQ'))
            if market_proxy is None:
                return MarketRegime.SIDEWAYS
            
            if isinstance(market_proxy, pd.DataFrame) and len(market_proxy) > 50:
                # Calculate regime indicators
                recent_returns = market_proxy['close'].pct_change().tail(20)
                volatility = recent_returns.std() * np.sqrt(252)
                trend = (market_proxy['close'].iloc[-1] / market_proxy['close'].iloc[-50] - 1)
                
                # Determine regime
                if volatility > 0.3:  # High volatility
                    regime = MarketRegime.VOLATILE
                elif volatility < 0.1:  # Low volatility
                    regime = MarketRegime.LOW_VOLATILITY
                elif trend > 0.1:  # Strong uptrend
                    regime = MarketRegime.BULL
                elif trend < -0.1:  # Strong downtrend
                    regime = MarketRegime.BEAR
                else:
                    regime = MarketRegime.SIDEWAYS
                
                self.current_regime = regime
                self.regime_history.append((datetime.now(), regime))
                
                return regime
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
        
        return MarketRegime.SIDEWAYS
    
    def update_metrics(self, 
                      signal: StrategySignal,
                      actual_return: Optional[float] = None,
                      trade_result: Optional[Dict[str, Any]] = None):
        """Update strategy performance metrics."""
        self.signal_history.append(signal)
        
        if actual_return is not None:
            self.returns_history.append(actual_return)
            
            # Update return metrics
            if len(self.returns_history) > 1:
                returns_array = np.array(self.returns_history)
                
                self.metrics.total_return = np.prod(1 + returns_array) - 1
                self.metrics.annualized_return = np.mean(returns_array) * 252
                self.metrics.volatility = np.std(returns_array) * np.sqrt(252)
                
                if self.metrics.volatility > 0:
                    self.metrics.sharpe_ratio = self.metrics.annualized_return / self.metrics.volatility
                
                # Calculate max drawdown
                cumulative_returns = np.cumprod(1 + returns_array)
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak
                self.metrics.max_drawdown = np.min(drawdown)
                
                if self.metrics.max_drawdown != 0:
                    self.metrics.calmar_ratio = self.metrics.annualized_return / abs(self.metrics.max_drawdown)
        
        if trade_result is not None:
            self.trades_history.append(trade_result)
            self._update_trading_metrics()
        
        # Update signal accuracy
        self._update_signal_metrics()
        
        self.metrics.last_updated = datetime.now()
    
    def _update_trading_metrics(self):
        """Update trading-specific metrics."""
        if not self.trades_history:
            return
        
        winning_trades = [t for t in self.trades_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades_history if t.get('pnl', 0) < 0]
        
        self.metrics.trade_count = len(self.trades_history)
        self.metrics.win_rate = len(winning_trades) / len(self.trades_history)
        
        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
        else:
            avg_win = 0
            
        if losing_trades:
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
        else:
            avg_loss = 1  # Avoid division by zero
            
        self.metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        self.metrics.average_trade = np.mean([t.get('pnl', 0) for t in self.trades_history])
    
    def _update_signal_metrics(self):
        """Update signal-specific metrics."""
        if len(self.signal_history) < 10:
            return
        
        # Calculate signal accuracy (simplified)
        recent_signals = list(self.signal_history)[-50:]  # Last 50 signals
        actionable_signals = [s for s in recent_signals if s.is_actionable]
        
        if actionable_signals:
            self.metrics.signal_coverage = len(actionable_signals) / len(recent_signals)
            # Signal accuracy would require correlation with actual returns
            # This is a simplified placeholder
            self.metrics.signal_accuracy = np.mean([s.confidence for s in actionable_signals])
    
    def get_current_allocation(self, symbol: str) -> float:
        """Get current allocation recommendation for a symbol."""
        recent_signals = [s for s in self.signal_history if s.symbol == symbol]
        
        if not recent_signals:
            return 0.0
        
        latest_signal = recent_signals[-1]
        
        if latest_signal.target_weight is not None:
            return latest_signal.target_weight
        
        # Convert signal to allocation
        base_allocation = self.config.get('base_allocation', 0.05)
        
        if latest_signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            return base_allocation * latest_signal.confidence * (latest_signal.strength.value / 5)
        elif latest_signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            return -base_allocation * latest_signal.confidence * (latest_signal.strength.value / 5)
        else:
            return 0.0
    
    def should_rebalance(self, current_positions: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced."""
        rebalance_threshold = self.config.get('rebalance_threshold', 0.05)
        
        for symbol, current_weight in current_positions.items():
            target_weight = self.get_current_allocation(symbol)
            
            if abs(current_weight - target_weight) > rebalance_threshold:
                return True
        
        return False
    
    async def optimize_parameters(self, 
                                historical_data: Dict[str, Any],
                                optimization_period: int = 252) -> Dict[str, Any]:
        """Optimize strategy parameters using historical data."""
        # This is a placeholder for parameter optimization
        # In practice, this would use techniques like:
        # - Grid search
        # - Bayesian optimization
        # - Genetic algorithms
        # - Reinforcement learning
        
        logger.info(f"Optimizing parameters for strategy {self.name}")
        
        # Return current parameters as optimized (placeholder)
        return self.config.copy()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'is_active': self.is_active,
            'is_trained': self.is_trained,
            'current_regime': self.current_regime.value,
            'config': self.config,
            'metrics': self.metrics.to_dict(),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'signal_count': len(self.signal_history),
            'trade_count': len(self.trades_history),
            'supported_assets': self.get_supported_assets(),
            'required_data': self.get_required_data()
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save strategy state for persistence."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'config': self.config,
            'metrics': self.metrics.to_dict(),
            'adaptive_params': self.adaptive_params,
            'is_active': self.is_active,
            'is_trained': self.is_trained,
            'current_regime': self.current_regime.value
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load strategy state from persistence."""
        self.config.update(state.get('config', {}))
        self.adaptive_params = state.get('adaptive_params', {})
        self.is_active = state.get('is_active', True)
        self.is_trained = state.get('is_trained', False)
        
        regime_value = state.get('current_regime', 'sideways')
        try:
            self.current_regime = MarketRegime(regime_value)
        except ValueError:
            self.current_regime = MarketRegime.SIDEWAYS
        
        logger.info(f"Loaded state for strategy {self.name}")

