"""Contextual Bandit controller for strategy selection."""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio
from enum import Enum
import json


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class TradingStrategy(Enum):
    """Available trading strategies."""
    ARBITRAGE_PURE = "arbitrage_pure"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"
    RISK_OFF = "risk_off"


@dataclass
class ContextualBanditConfig:
    """Configuration for contextual bandit controller."""
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    window_size: int = 1000
    confidence_threshold: float = 0.7
    regime_history_size: int = 100
    strategy_cooldown_minutes: int = 15
    max_concurrent_strategies: int = 3
    performance_lookback_hours: int = 24


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy: TradingStrategy
    regime: MarketRegime
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = None
    confidence_score: float = 0.0


class LinUCB:
    """Linear Upper Confidence Bound algorithm for contextual bandits."""
    
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize parameters for each arm (strategy)
        self.A = {}  # A_a matrices
        self.b = {}  # b_a vectors
        self.theta = {}  # theta_a parameters
        
    def add_arm(self, arm_id: str):
        """Add a new arm (strategy)."""
        self.A[arm_id] = np.identity(self.n_features)
        self.b[arm_id] = np.zeros(self.n_features)
        self.theta[arm_id] = np.zeros(self.n_features)
    
    def select_arm(self, context: np.ndarray, available_arms: List[str]) -> Tuple[str, float]:
        """Select the best arm given context."""
        best_arm = None
        best_value = -float('inf')
        confidence_scores = {}
        
        for arm_id in available_arms:
            if arm_id not in self.A:
                self.add_arm(arm_id)
            
            # Calculate confidence bound
            A_inv = np.linalg.inv(self.A[arm_id])
            theta = A_inv @ self.b[arm_id]
            
            # Expected reward
            expected_reward = theta @ context
            
            # Confidence interval
            confidence_width = self.alpha * np.sqrt(context @ A_inv @ context)
            
            # Upper confidence bound
            ucb_value = expected_reward + confidence_width
            confidence_scores[arm_id] = float(confidence_width)
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_arm = arm_id
        
        return best_arm, confidence_scores.get(best_arm, 0.0)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update the model with observed reward."""
        if arm_id not in self.A:
            self.add_arm(arm_id)
        
        # Update A and b
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context
        
        # Update theta
        A_inv = np.linalg.inv(self.A[arm_id])
        self.theta[arm_id] = A_inv @ self.b[arm_id]
    
    def get_arm_stats(self, arm_id: str) -> Dict[str, Any]:
        """Get statistics for an arm."""
        if arm_id not in self.A:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "theta": self.theta[arm_id].tolist(),
            "determinant": float(np.linalg.det(self.A[arm_id])),
            "condition_number": float(np.linalg.cond(self.A[arm_id]))
        }


class ContextualBanditController:
    """Contextual bandit controller for intelligent strategy selection."""
    
    def __init__(self, config: ContextualBanditConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize contextual bandit
        self.n_features = 15  # Number of context features
        self.bandit = LinUCB(self.n_features, alpha=1.0)
        
        # Strategy management
        self.active_strategies = set()
        self.strategy_cooldowns = {}
        self.strategy_performances = defaultdict(lambda: defaultdict(StrategyPerformance))
        
        # Context and regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = deque(maxlen=config.regime_history_size)
        self.feature_history = deque(maxlen=config.window_size)
        
        # Performance tracking
        self.trade_history = deque(maxlen=10000)
        self.pnl_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Exploration parameters
        self.exploration_rate = config.exploration_rate
        
        # Initialize all strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available strategies in the bandit."""
        for strategy in TradingStrategy:
            self.bandit.add_arm(strategy.value)
    
    def extract_context_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract context features from market data."""
        features = np.zeros(self.n_features)
        
        try:
            # Price-based features (0-4)
            features[0] = market_data.get('price_volatility_1m', 0.0)
            features[1] = market_data.get('price_volatility_5m', 0.0)
            features[2] = market_data.get('price_change_1m', 0.0)
            features[3] = market_data.get('price_change_5m', 0.0)
            features[4] = market_data.get('spread_pct', 0.0)
            
            # Volume-based features (5-8)
            features[5] = market_data.get('volume_ratio_1m', 1.0)
            features[6] = market_data.get('volume_spike_indicator', 0.0)
            features[7] = market_data.get('volume_imbalance', 0.0)
            features[8] = market_data.get('liquidity_score', 0.0)
            
            # Market regime features (9-11)
            features[9] = 1.0 if self.current_regime == MarketRegime.TRENDING_UP else 0.0
            features[10] = 1.0 if self.current_regime == MarketRegime.HIGH_VOLATILITY else 0.0
            features[11] = 1.0 if self.current_regime == MarketRegime.SIDEWAYS else 0.0
            
            # Time-based features (12-14)
            hour = datetime.now().hour
            features[12] = 1.0 if 9 <= hour <= 16 else 0.0  # Market hours
            features[13] = 1.0 if hour <= 12 else 0.0  # Morning session
            features[14] = market_data.get('arbitrage_opportunity_score', 0.0)
            
            # Normalize features
            features = np.clip(features, -3.0, 3.0)
            
        except Exception as e:
            self.logger.error(f"Error extracting context features: {e}")
            features = np.zeros(self.n_features)
        
        return features
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime based on market data."""
        try:
            volatility_1m = market_data.get('price_volatility_1m', 0.0)
            volatility_5m = market_data.get('price_volatility_5m', 0.0)
            price_change_1m = market_data.get('price_change_1m', 0.0)
            price_change_5m = market_data.get('price_change_5m', 0.0)
            volume_ratio = market_data.get('volume_ratio_1m', 1.0)
            volume_spike = market_data.get('volume_spike_indicator', 0.0)
            
            # High volatility regime
            if volatility_1m > 0.02 or volatility_5m > 0.015:
                if volume_spike > 0.5:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regime
            if volatility_1m < 0.005 and volatility_5m < 0.003:
                return MarketRegime.LOW_VOLATILITY
            
            # Trending regimes
            if abs(price_change_5m) > 0.01:
                if price_change_5m > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            
            # Reversal detection
            if abs(price_change_1m) > 0.005 and np.sign(price_change_1m) != np.sign(price_change_5m):
                return MarketRegime.REVERSAL
            
            # Default to sideways
            return MarketRegime.SIDEWAYS
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies (not in cooldown)."""
        current_time = datetime.now()
        available = []
        
        for strategy in TradingStrategy:
            strategy_name = strategy.value
            
            # Check cooldown
            if strategy_name in self.strategy_cooldowns:
                if current_time < self.strategy_cooldowns[strategy_name]:
                    continue
            
            # Check if we're at max concurrent strategies
            if len(self.active_strategies) >= self.config.max_concurrent_strategies:
                if strategy_name not in self.active_strategies:
                    continue
            
            available.append(strategy_name)
        
        return available
    
    async def select_strategy(self, market_data: Dict[str, Any]) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Select the best strategy based on current context."""
        with self._lock:
            try:
                # Update market regime
                self.current_regime = self.detect_market_regime(market_data)
                self.regime_history.append((datetime.now(), self.current_regime))
                
                # Extract context features
                context = self.extract_context_features(market_data)
                self.feature_history.append((datetime.now(), context))
                
                # Get available strategies
                available_strategies = self.get_available_strategies()
                
                if not available_strategies:
                    return None, 0.0, {"reason": "No strategies available"}
                
                # Exploration vs exploitation
                if np.random.random() < self.exploration_rate:
                    # Exploration: random strategy
                    selected_strategy = np.random.choice(available_strategies)
                    confidence = 0.0
                    selection_method = "exploration"
                else:
                    # Exploitation: use bandit
                    selected_strategy, confidence = self.bandit.select_arm(context, available_strategies)
                    selection_method = "exploitation"
                
                # Decay exploration rate
                self.exploration_rate = max(
                    self.config.min_exploration_rate,
                    self.exploration_rate * self.config.exploration_decay
                )
                
                # Add to active strategies
                self.active_strategies.add(selected_strategy)
                
                selection_info = {
                    "strategy": selected_strategy,
                    "confidence": confidence,
                    "regime": self.current_regime.value,
                    "selection_method": selection_method,
                    "exploration_rate": self.exploration_rate,
                    "available_strategies": available_strategies,
                    "active_strategies": list(self.active_strategies)
                }
                
                self.logger.info(f"Selected strategy: {selected_strategy} (confidence: {confidence:.3f}, regime: {self.current_regime.value})")
                
                return selected_strategy, confidence, selection_info
                
            except Exception as e:
                self.logger.error(f"Error selecting strategy: {e}")
                return None, 0.0, {"error": str(e)}
    
    async def update_strategy_performance(self, strategy: str, trade_result: Dict[str, Any]):
        """Update strategy performance based on trade results."""
        with self._lock:
            try:
                pnl = trade_result.get('pnl', 0.0)
                duration = trade_result.get('duration_minutes', 0.0)
                success = trade_result.get('success', False)
                
                # Get context features at time of trade
                context = self.extract_context_features(trade_result.get('market_data', {}))
                
                # Calculate reward (normalized PnL)
                reward = self._calculate_reward(pnl, duration, success)
                
                # Update bandit
                self.bandit.update(strategy, context, reward)
                
                # Update performance metrics
                regime = trade_result.get('regime', self.current_regime)
                perf = self.strategy_performances[strategy][regime]
                
                perf.total_trades += 1
                perf.total_pnl += pnl
                if success:
                    perf.winning_trades += 1
                
                perf.success_rate = perf.winning_trades / perf.total_trades
                perf.last_updated = datetime.now()
                
                # Update PnL history for Sharpe ratio calculation
                self.pnl_history[strategy].append(pnl)
                if len(self.pnl_history[strategy]) > 10:
                    pnl_array = np.array(list(self.pnl_history[strategy]))
                    perf.sharpe_ratio = self._calculate_sharpe_ratio(pnl_array)
                
                # Store trade in history
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'strategy': strategy,
                    'regime': regime,
                    'pnl': pnl,
                    'duration': duration,
                    'success': success,
                    'reward': reward
                })
                
                self.logger.info(f"Updated performance for {strategy}: PnL={pnl:.4f}, Success={success}, Reward={reward:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error updating strategy performance: {e}")
    
    def _calculate_reward(self, pnl: float, duration: float, success: bool) -> float:
        """Calculate reward signal for the bandit."""
        # Base reward from PnL (normalized)
        base_reward = np.tanh(pnl * 100)  # Scale and bound
        
        # Time penalty for long-duration trades
        time_penalty = max(0, (duration - 60) / 60) * 0.1  # Penalty after 1 hour
        
        # Success bonus
        success_bonus = 0.2 if success else -0.1
        
        # Combined reward
        reward = base_reward - time_penalty + success_bonus
        
        return np.clip(reward, -1.0, 1.0)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        return (mean_return / std_return) * np.sqrt(252)
    
    async def deactivate_strategy(self, strategy: str, cooldown_minutes: Optional[int] = None):
        """Deactivate a strategy and put it in cooldown."""
        with self._lock:
            try:
                # Remove from active strategies
                self.active_strategies.discard(strategy)
                
                # Set cooldown
                cooldown = cooldown_minutes or self.config.strategy_cooldown_minutes
                cooldown_until = datetime.now() + timedelta(minutes=cooldown)
                self.strategy_cooldowns[strategy] = cooldown_until
                
                self.logger.info(f"Deactivated strategy {strategy}, cooldown until {cooldown_until}")
                
            except Exception as e:
                self.logger.error(f"Error deactivating strategy: {e}")
    
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by performance."""
        rankings = []
        
        for strategy_name in [s.value for s in TradingStrategy]:
            total_pnl = 0.0
            total_trades = 0
            avg_sharpe = 0.0
            avg_success_rate = 0.0
            
            # Aggregate across all regimes
            regime_performances = self.strategy_performances[strategy_name]
            for regime_perf in regime_performances.values():
                total_pnl += regime_perf.total_pnl
                total_trades += regime_perf.total_trades
                avg_sharpe += regime_perf.sharpe_ratio
                avg_success_rate += regime_perf.success_rate
            
            num_regimes = len(regime_performances) or 1
            avg_sharpe /= num_regimes
            avg_success_rate /= num_regimes
            
            # Get bandit stats
            arm_stats = self.bandit.get_arm_stats(strategy_name)
            
            rankings.append({
                'strategy': strategy_name,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_success_rate': avg_success_rate,
                'is_active': strategy_name in self.active_strategies,
                'in_cooldown': strategy_name in self.strategy_cooldowns and 
                             datetime.now() < self.strategy_cooldowns[strategy_name],
                'bandit_stats': arm_stats
            })
        
        # Sort by total PnL
        rankings.sort(key=lambda x: x['total_pnl'], reverse=True)
        return rankings
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get analysis of market regime patterns."""
        if not self.regime_history:
            return {"error": "No regime history available"}
        
        # Count regime occurrences
        regime_counts = defaultdict(int)
        regime_durations = defaultdict(list)
        
        current_regime = None
        regime_start = None
        
        for timestamp, regime in self.regime_history:
            regime_counts[regime.value] += 1
            
            if current_regime != regime:
                if current_regime is not None and regime_start is not None:
                    duration = (timestamp - regime_start).total_seconds() / 60  # minutes
                    regime_durations[current_regime.value].append(duration)
                
                current_regime = regime
                regime_start = timestamp
        
        # Calculate statistics
        regime_stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                regime_stats[regime] = {
                    'count': len(durations),
                    'avg_duration_minutes': np.mean(durations),
                    'total_duration_minutes': sum(durations)
                }
        
        return {
            'current_regime': self.current_regime.value,
            'regime_counts': dict(regime_counts),
            'regime_stats': regime_stats,
            'total_regimes_tracked': len(self.regime_history)
        }
    
    def get_controller_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            'exploration_rate': self.exploration_rate,
            'active_strategies': list(self.active_strategies),
            'strategies_in_cooldown': len([s for s, cooldown in self.strategy_cooldowns.items() 
                                         if datetime.now() < cooldown]),
            'total_trades_processed': len(self.trade_history),
            'current_regime': self.current_regime.value,
            'bandit_arms': len([s.value for s in TradingStrategy]),
            'feature_history_size': len(self.feature_history)
        }

