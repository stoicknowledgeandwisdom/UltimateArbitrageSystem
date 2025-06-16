#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Momentum Strategies
===========================

Quantum-enhanced and AI-driven momentum strategies with:
- Quantum momentum detection algorithms
- Cross-asset momentum analysis
- Adaptive parameter optimization
- Multi-timeframe momentum signals
- Risk-adjusted momentum scoring
- Regime-aware momentum strategies
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.algorithms import VQE
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .base_strategy import (
    BaseStrategy, StrategySignal, SignalType, SignalStrength, 
    MarketRegime, StrategyMetrics
)

logger = logging.getLogger(__name__)

class QuantumMomentumStrategy(BaseStrategy):
    """
    Quantum-enhanced momentum strategy using quantum algorithms for 
    pattern recognition and momentum detection.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        super().__init__(strategy_id, "QuantumMomentum", config)
        
        # Quantum parameters
        self.use_quantum = config.get('use_quantum', QISKIT_AVAILABLE)
        self.quantum_depth = config.get('quantum_depth', 3)
        self.quantum_shots = config.get('quantum_shots', 1024)
        
        # Momentum parameters
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        
        # Risk management
        self.momentum_decay_factor = config.get('momentum_decay_factor', 0.95)
        self.max_momentum_exposure = config.get('max_momentum_exposure', 0.3)
        
        # Quantum circuit cache
        self.quantum_circuits = {}
        
        # Momentum tracking
        self.momentum_scores = {}
        self.momentum_history = deque(maxlen=100)
        
        logger.info(f"Initialized QuantumMomentumStrategy with quantum={self.use_quantum}")
    
    async def generate_signal(self, 
                            market_data: Dict[str, Any],
                            portfolio_state: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate momentum signal using quantum-enhanced analysis."""
        try:
            # Detect market regime first
            regime = await self.detect_market_regime(market_data)
            
            # Select best symbol for momentum trading
            best_symbol, momentum_score = await self._find_best_momentum_opportunity(market_data)
            
            if best_symbol is None or momentum_score < self.momentum_threshold:
                return None
            
            # Generate quantum-enhanced signal
            if self.use_quantum:
                signal_strength, confidence = await self._quantum_momentum_analysis(
                    market_data[best_symbol], momentum_score
                )
            else:
                signal_strength, confidence = await self._classical_momentum_analysis(
                    market_data[best_symbol], momentum_score
                )
            
            # Determine signal type based on momentum direction and strength
            if momentum_score > self.momentum_threshold * 2:
                signal_type = SignalType.STRONG_BUY
            elif momentum_score > self.momentum_threshold:
                signal_type = SignalType.BUY
            elif momentum_score < -self.momentum_threshold * 2:
                signal_type = SignalType.STRONG_SELL
            elif momentum_score < -self.momentum_threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Calculate position sizing based on momentum strength and risk
            target_weight = self._calculate_momentum_position_size(
                momentum_score, confidence, regime
            )
            
            # Risk management levels
            current_price = self._get_current_price(market_data[best_symbol])
            stop_loss = current_price * (1 - self.stop_loss_threshold) if momentum_score > 0 else current_price * (1 + self.stop_loss_threshold)
            take_profit = current_price * (1 + self.take_profit_threshold) if momentum_score > 0 else current_price * (1 - self.take_profit_threshold)
            
            signal = StrategySignal(
                strategy_id=self.strategy_id,
                symbol=best_symbol,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=confidence,
                target_weight=target_weight,
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                timeframe="1d",
                features={
                    'momentum_score': momentum_score,
                    'quantum_advantage': 1.0 if self.use_quantum else 0.0,
                    'regime': regime.value,
                    'volatility_adj': self.volatility_adjustment
                },
                expected_return=momentum_score * 0.5,  # Expected return based on momentum
                expected_volatility=self._estimate_volatility(market_data[best_symbol])
            )
            
            self.last_signal_time = datetime.now()
            self.momentum_history.append((datetime.now(), best_symbol, momentum_score))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating quantum momentum signal: {e}")
            return None
    
    async def _find_best_momentum_opportunity(self, 
                                            market_data: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Find the best momentum opportunity across all assets."""
        momentum_candidates = []
        
        for symbol, data in market_data.items():
            if isinstance(data, pd.DataFrame) and len(data) >= max(self.lookback_periods):
                momentum_score = await self._calculate_momentum_score(symbol, data)
                
                if abs(momentum_score) > self.momentum_threshold * 0.5:  # Pre-filter
                    momentum_candidates.append((symbol, momentum_score))
        
        if not momentum_candidates:
            return None, 0.0
        
        # Sort by absolute momentum score and return the best
        momentum_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        
        best_symbol, best_score = momentum_candidates[0]
        return best_symbol, best_score
    
    async def _calculate_momentum_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate comprehensive momentum score."""
        try:
            scores = []
            weights = []
            
            for period in self.lookback_periods:
                if len(data) >= period:
                    # Price momentum
                    price_momentum = (data['close'].iloc[-1] / data['close'].iloc[-period] - 1)
                    
                    # Volume-adjusted momentum
                    if 'volume' in data.columns:
                        avg_volume = data['volume'].rolling(period).mean().iloc[-1]
                        recent_volume = data['volume'].iloc[-5:].mean()
                        volume_factor = min(2.0, recent_volume / avg_volume) if avg_volume > 0 else 1.0
                        volume_adj_momentum = price_momentum * volume_factor
                    else:
                        volume_adj_momentum = price_momentum
                    
                    # Volatility adjustment
                    if self.volatility_adjustment:
                        volatility = data['close'].pct_change().rolling(period).std().iloc[-1]
                        vol_adj_momentum = volume_adj_momentum / (volatility + 1e-8)
                    else:
                        vol_adj_momentum = volume_adj_momentum
                    
                    scores.append(vol_adj_momentum)
                    weights.append(1.0 / period)  # Weight shorter periods more heavily
            
            if not scores:
                return 0.0
            
            # Weighted average of momentum scores
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            momentum_score = np.average(scores, weights=weights)
            
            # Apply momentum decay for stability
            if symbol in self.momentum_scores:
                prev_score = self.momentum_scores[symbol]
                momentum_score = (momentum_score * (1 - self.momentum_decay_factor) + 
                                prev_score * self.momentum_decay_factor)
            
            self.momentum_scores[symbol] = momentum_score
            
            return momentum_score
            
        except Exception as e:
            logger.warning(f"Error calculating momentum for {symbol}: {e}")
            return 0.0
    
    async def _quantum_momentum_analysis(self, 
                                       data: pd.DataFrame, 
                                       momentum_score: float) -> Tuple[SignalStrength, float]:
        """Quantum-enhanced momentum analysis."""
        if not QISKIT_AVAILABLE:
            return await self._classical_momentum_analysis(data, momentum_score)
        
        try:
            # Create quantum circuit for momentum pattern recognition
            qc = self._create_momentum_quantum_circuit(data)
            
            # Simulate quantum computation
            # In practice, this would run on actual quantum hardware
            quantum_result = self._simulate_quantum_momentum(qc, data)
            
            # Convert quantum result to signal strength and confidence
            strength_value = min(5, max(1, int(abs(momentum_score) * 20) + quantum_result))
            confidence = min(1.0, abs(momentum_score) * 5 + quantum_result * 0.2)
            
            signal_strength = SignalStrength(strength_value)
            
            return signal_strength, confidence
            
        except Exception as e:
            logger.warning(f"Quantum momentum analysis failed, falling back to classical: {e}")
            return await self._classical_momentum_analysis(data, momentum_score)
    
    def _create_momentum_quantum_circuit(self, data: pd.DataFrame) -> 'QuantumCircuit':
        """Create quantum circuit for momentum pattern analysis."""
        if not QISKIT_AVAILABLE:
            return None
        
        # Create quantum circuit with qubits for different momentum timeframes
        n_qubits = min(4, len(self.lookback_periods))
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Encode momentum data into quantum states
        recent_returns = data['close'].pct_change().tail(10).fillna(0)
        
        for i in range(n_qubits):
            # Encode return information as rotation angles
            if i < len(recent_returns):
                angle = recent_returns.iloc[-(i+1)] * np.pi  # Scale to rotation angle
                qc.ry(angle, i)
            
            # Add entanglement between momentum timeframes
            if i > 0:
                qc.cx(i-1, i)
        
        # Add measurement
        qc.measure_all()
        
        return qc
    
    def _simulate_quantum_momentum(self, qc: 'QuantumCircuit', data: pd.DataFrame) -> float:
        """Simulate quantum momentum computation."""
        # Simplified quantum simulation
        # In practice, this would use actual quantum simulators/hardware
        
        try:
            # Generate pseudo-quantum result based on circuit structure
            recent_volatility = data['close'].pct_change().tail(10).std()
            momentum_strength = abs(data['close'].iloc[-1] / data['close'].iloc[-10] - 1)
            
            # Quantum advantage simulation
            quantum_enhancement = np.sin(momentum_strength * np.pi) * np.cos(recent_volatility * np.pi)
            
            return max(0, min(1, quantum_enhancement))
            
        except Exception as e:
            logger.warning(f"Quantum simulation error: {e}")
            return 0.0
    
    async def _classical_momentum_analysis(self, 
                                         data: pd.DataFrame, 
                                         momentum_score: float) -> Tuple[SignalStrength, float]:
        """Classical momentum analysis as fallback."""
        try:
            # Calculate signal strength based on momentum magnitude
            abs_momentum = abs(momentum_score)
            
            if abs_momentum > 0.1:  # Very strong momentum
                strength = SignalStrength.VERY_STRONG
            elif abs_momentum > 0.05:
                strength = SignalStrength.STRONG
            elif abs_momentum > 0.03:
                strength = SignalStrength.MODERATE
            elif abs_momentum > 0.015:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            # Calculate confidence based on consistency and volume
            consistency_score = self._calculate_momentum_consistency(data)
            volume_score = self._calculate_volume_confidence(data)
            
            confidence = min(1.0, (abs_momentum * 10 + consistency_score + volume_score) / 3)
            
            return strength, confidence
            
        except Exception as e:
            logger.warning(f"Classical momentum analysis error: {e}")
            return SignalStrength.WEAK, 0.5
    
    def _calculate_momentum_consistency(self, data: pd.DataFrame) -> float:
        """Calculate momentum consistency across different timeframes."""
        try:
            consistencies = []
            
            for period in self.lookback_periods:
                if len(data) >= period:
                    returns = data['close'].pct_change().tail(period)
                    positive_days = (returns > 0).sum()
                    consistency = positive_days / len(returns)
                    
                    # Adjust for momentum direction
                    momentum = data['close'].iloc[-1] / data['close'].iloc[-period] - 1
                    if momentum > 0:
                        consistencies.append(consistency)
                    else:
                        consistencies.append(1 - consistency)
            
            return np.mean(consistencies) if consistencies else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_volume_confidence(self, data: pd.DataFrame) -> float:
        """Calculate volume-based confidence."""
        try:
            if 'volume' not in data.columns:
                return 0.5
            
            recent_volume = data['volume'].tail(5).mean()
            avg_volume = data['volume'].tail(20).mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Higher recent volume increases confidence
            return min(1.0, volume_ratio / 2)
            
        except Exception:
            return 0.5
    
    def _calculate_momentum_position_size(self, 
                                        momentum_score: float, 
                                        confidence: float,
                                        regime: MarketRegime) -> float:
        """Calculate position size based on momentum strength and market regime."""
        try:
            # Base position size
            base_size = self.config.get('base_allocation', 0.05)
            
            # Adjust for momentum strength
            momentum_multiplier = min(3.0, abs(momentum_score) * 20)
            
            # Adjust for confidence
            confidence_multiplier = confidence
            
            # Regime adjustment
            regime_multiplier = {
                MarketRegime.BULL: 1.2,
                MarketRegime.BEAR: 0.6,
                MarketRegime.SIDEWAYS: 0.8,
                MarketRegime.VOLATILE: 0.7,
                MarketRegime.LOW_VOLATILITY: 1.1,
                MarketRegime.CRISIS: 0.3,
                MarketRegime.RECOVERY: 1.0
            }.get(regime, 1.0)
            
            position_size = (base_size * momentum_multiplier * 
                           confidence_multiplier * regime_multiplier)
            
            # Apply maximum exposure limit
            max_exposure = self.max_momentum_exposure
            position_size = min(position_size, max_exposure)
            
            return position_size if momentum_score > 0 else -position_size
            
        except Exception as e:
            logger.warning(f"Error calculating position size: {e}")
            return 0.0
    
    def _get_current_price(self, data: pd.DataFrame) -> float:
        """Get current price from market data."""
        try:
            return float(data['close'].iloc[-1])
        except Exception:
            return 0.0
    
    def _estimate_volatility(self, data: pd.DataFrame) -> float:
        """Estimate expected volatility."""
        try:
            returns = data['close'].pct_change().tail(20)
            return float(returns.std() * np.sqrt(252))  # Annualized volatility
        except Exception:
            return 0.2  # Default volatility estimate
    
    async def update_parameters(self, 
                              market_data: Dict[str, Any],
                              performance_feedback: Dict[str, Any]):
        """Update strategy parameters based on performance."""
        try:
            # Adaptive momentum threshold based on recent performance
            recent_signals = list(self.signal_history)[-10:]
            if recent_signals:
                avg_confidence = np.mean([s.confidence for s in recent_signals])
                
                if avg_confidence < 0.6:  # Low confidence signals
                    self.momentum_threshold *= 1.05  # Increase threshold
                elif avg_confidence > 0.8:  # High confidence signals
                    self.momentum_threshold *= 0.98  # Decrease threshold
            
            # Adjust quantum parameters if using quantum
            if self.use_quantum and performance_feedback.get('quantum_advantage', 0) < 1.1:
                self.quantum_depth = min(5, self.quantum_depth + 1)
            
            logger.debug(f"Updated momentum threshold to {self.momentum_threshold:.4f}")
            
        except Exception as e:
            logger.warning(f"Error updating momentum strategy parameters: {e}")
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields."""
        return ['close', 'volume', 'high', 'low', 'open']
    
    def get_supported_assets(self) -> List[str]:
        """Get list of supported asset classes."""
        return ['equity', 'etf', 'crypto', 'forex', 'commodity']

class AdaptiveMomentumStrategy(BaseStrategy):
    """
    Adaptive momentum strategy that learns and adjusts parameters 
    based on market conditions and performance.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        super().__init__(strategy_id, "AdaptiveMomentum", config)
        
        # Adaptive learning parameters
        self.adaptation_frequency = config.get('adaptation_frequency', 20)  # trades
        self.learning_window = config.get('learning_window', 100)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        
        # Multi-timeframe momentum
        self.timeframes = config.get('timeframes', ['5min', '1h', '1d'])
        self.timeframe_weights = config.get('timeframe_weights', [0.2, 0.3, 0.5])
        
        # Neural network for adaptive learning (if available)
        self.use_neural_net = config.get('use_neural_net', TORCH_AVAILABLE)
        self.neural_net = None
        
        if self.use_neural_net and TORCH_AVAILABLE:
            self._initialize_neural_network()
        
        # Performance tracking for adaptation
        self.adaptation_counter = 0
        self.performance_window = deque(maxlen=self.learning_window)
        
        logger.info(f"Initialized AdaptiveMomentumStrategy with neural_net={self.use_neural_net}")
    
    def _initialize_neural_network(self):
        """Initialize neural network for adaptive parameter learning."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Simple feedforward network for parameter adaptation
            input_size = 10  # Market features
            hidden_size = 20
            output_size = 3  # Momentum threshold, lookback, confidence
            
            self.neural_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid()  # Normalize outputs
            )
            
            self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
            
            logger.info("Neural network initialized for adaptive learning")
            
        except Exception as e:
            logger.warning(f"Failed to initialize neural network: {e}")
            self.use_neural_net = False
    
    async def generate_signal(self, 
                            market_data: Dict[str, Any],
                            portfolio_state: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate adaptive momentum signal."""
        try:
            # Adapt parameters if needed
            if self.adaptation_counter >= self.adaptation_frequency:
                await self._adapt_parameters(market_data)
                self.adaptation_counter = 0
            
            # Multi-timeframe momentum analysis
            momentum_signals = await self._multi_timeframe_analysis(market_data)
            
            if not momentum_signals:
                return None
            
            # Combine signals from different timeframes
            combined_signal = self._combine_timeframe_signals(momentum_signals)
            
            if combined_signal['confidence'] < self.min_confidence_threshold:
                return None
            
            self.adaptation_counter += 1
            self.last_signal_time = datetime.now()
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error generating adaptive momentum signal: {e}")
            return None
    
    async def _multi_timeframe_analysis(self, 
                                      market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform momentum analysis across multiple timeframes."""
        signals = []
        
        for symbol, data in market_data.items():
            if not isinstance(data, pd.DataFrame) or len(data) < 50:
                continue
            
            timeframe_signals = []
            
            for i, timeframe in enumerate(self.timeframes):
                # Resample data for different timeframes
                resampled_data = self._resample_data(data, timeframe)
                
                if len(resampled_data) < 20:
                    continue
                
                # Calculate momentum for this timeframe
                momentum_score = await self._calculate_timeframe_momentum(
                    symbol, resampled_data, timeframe
                )
                
                if abs(momentum_score) > 0.01:  # Minimum threshold
                    timeframe_signals.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'momentum_score': momentum_score,
                        'weight': self.timeframe_weights[i],
                        'data': resampled_data
                    })
            
            if timeframe_signals:
                signals.extend(timeframe_signals)
        
        return signals
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe."""
        try:
            # This is a simplified resampling - in practice, you'd use proper OHLCV resampling
            if timeframe == '5min':
                # Use recent high-frequency data if available
                return data.tail(100)  # Last 100 data points
            elif timeframe == '1h':
                return data.tail(200)  # More data for hourly
            else:  # '1d'
                return data  # Use all available daily data
                
        except Exception as e:
            logger.warning(f"Error resampling data for {timeframe}: {e}")
            return data
    
    async def _calculate_timeframe_momentum(self, 
                                          symbol: str, 
                                          data: pd.DataFrame,
                                          timeframe: str) -> float:
        """Calculate momentum for specific timeframe."""
        try:
            # Adjust lookback based on timeframe
            lookback_map = {
                '5min': 12,   # 1 hour of 5-min bars
                '1h': 24,     # 1 day of hourly bars
                '1d': 20      # 20 days
            }
            
            lookback = lookback_map.get(timeframe, 20)
            
            if len(data) < lookback:
                return 0.0
            
            # Price momentum
            price_momentum = (data['close'].iloc[-1] / data['close'].iloc[-lookback] - 1)
            
            # Volatility adjustment
            volatility = data['close'].pct_change().rolling(lookback//2).std().iloc[-1]
            risk_adj_momentum = price_momentum / (volatility + 1e-8)
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(data, lookback)
            
            # Combined momentum score
            momentum_score = risk_adj_momentum * trend_strength
            
            return momentum_score
            
        except Exception as e:
            logger.warning(f"Error calculating timeframe momentum: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame, lookback: int) -> float:
        """Calculate trend strength using linear regression slope."""
        try:
            recent_prices = data['close'].tail(lookback).values
            x = np.arange(len(recent_prices))
            
            # Linear regression slope as trend strength
            slope = np.polyfit(x, recent_prices, 1)[0]
            
            # Normalize by average price
            avg_price = np.mean(recent_prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0
            
            return normalized_slope * lookback  # Scale by lookback period
            
        except Exception:
            return 0.0
    
    def _combine_timeframe_signals(self, signals: List[Dict[str, Any]]) -> Optional[StrategySignal]:
        """Combine signals from multiple timeframes."""
        if not signals:
            return None
        
        try:
            # Group signals by symbol
            symbol_signals = {}
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in symbol_signals:
                    symbol_signals[symbol] = []
                symbol_signals[symbol].append(signal)
            
            # Find best combined signal
            best_symbol = None
            best_combined_score = 0
            best_signal_data = None
            
            for symbol, symbol_sigs in symbol_signals.items():
                # Weighted combination of momentum scores
                total_weight = sum(s['weight'] for s in symbol_sigs)
                if total_weight == 0:
                    continue
                
                combined_momentum = sum(
                    s['momentum_score'] * s['weight'] for s in symbol_sigs
                ) / total_weight
                
                if abs(combined_momentum) > abs(best_combined_score):
                    best_combined_score = combined_momentum
                    best_symbol = symbol
                    best_signal_data = symbol_sigs
            
            if best_symbol is None or abs(best_combined_score) < 0.02:
                return None
            
            # Create combined signal
            signal_type = self._determine_signal_type(best_combined_score)
            signal_strength = self._determine_signal_strength(best_combined_score)
            confidence = self._calculate_combined_confidence(best_signal_data)
            
            # Position sizing
            target_weight = self._calculate_adaptive_position_size(
                best_combined_score, confidence
            )
            
            # Risk management
            latest_data = best_signal_data[0]['data']  # Use daily data for price
            current_price = float(latest_data['close'].iloc[-1])
            
            return StrategySignal(
                strategy_id=self.strategy_id,
                symbol=best_symbol,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=confidence,
                target_weight=target_weight,
                stop_loss=Decimal(str(current_price * 0.95)) if best_combined_score > 0 else Decimal(str(current_price * 1.05)),
                take_profit=Decimal(str(current_price * 1.10)) if best_combined_score > 0 else Decimal(str(current_price * 0.90)),
                timeframe="multi",
                features={
                    'combined_momentum': best_combined_score,
                    'timeframe_count': len(best_signal_data),
                    'adaptive_learning': self.use_neural_net
                },
                expected_return=best_combined_score * 0.7,
                expected_volatility=self._estimate_combined_volatility(best_signal_data)
            )
            
        except Exception as e:
            logger.error(f"Error combining timeframe signals: {e}")
            return None
    
    def _determine_signal_type(self, momentum_score: float) -> SignalType:
        """Determine signal type based on momentum score."""
        if momentum_score > 0.05:
            return SignalType.STRONG_BUY
        elif momentum_score > 0.02:
            return SignalType.BUY
        elif momentum_score < -0.05:
            return SignalType.STRONG_SELL
        elif momentum_score < -0.02:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _determine_signal_strength(self, momentum_score: float) -> SignalStrength:
        """Determine signal strength based on momentum score."""
        abs_momentum = abs(momentum_score)
        
        if abs_momentum > 0.08:
            return SignalStrength.VERY_STRONG
        elif abs_momentum > 0.05:
            return SignalStrength.STRONG
        elif abs_momentum > 0.03:
            return SignalStrength.MODERATE
        elif abs_momentum > 0.015:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_combined_confidence(self, signal_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on timeframe agreement."""
        try:
            if len(signal_data) == 1:
                return 0.7  # Lower confidence for single timeframe
            
            # Check momentum direction agreement
            positive_momentum = sum(1 for s in signal_data if s['momentum_score'] > 0)
            negative_momentum = sum(1 for s in signal_data if s['momentum_score'] < 0)
            
            agreement_ratio = max(positive_momentum, negative_momentum) / len(signal_data)
            
            # Base confidence on agreement and momentum strength
            avg_momentum_strength = np.mean([abs(s['momentum_score']) for s in signal_data])
            
            confidence = min(1.0, agreement_ratio * 0.6 + avg_momentum_strength * 10 * 0.4)
            
            return confidence
            
        except Exception:
            return 0.5
    
    def _calculate_adaptive_position_size(self, momentum_score: float, confidence: float) -> float:
        """Calculate position size with adaptive learning."""
        try:
            base_size = self.config.get('base_allocation', 0.05)
            
            # Adaptive multiplier based on recent performance
            if len(self.performance_window) > 10:
                recent_performance = np.mean(list(self.performance_window)[-10:])
                adaptive_multiplier = 1.0 + max(-0.5, min(0.5, recent_performance))
            else:
                adaptive_multiplier = 1.0
            
            position_size = (base_size * abs(momentum_score) * 20 * 
                           confidence * adaptive_multiplier)
            
            # Cap position size
            max_position = self.config.get('max_position_size', 0.15)
            position_size = min(position_size, max_position)
            
            return position_size if momentum_score > 0 else -position_size
            
        except Exception:
            return 0.0
    
    def _estimate_combined_volatility(self, signal_data: List[Dict[str, Any]]) -> float:
        """Estimate volatility from combined timeframe data."""
        try:
            volatilities = []
            
            for signal in signal_data:
                data = signal['data']
                returns = data['close'].pct_change().tail(20)
                vol = returns.std() * np.sqrt(252)
                volatilities.append(vol)
            
            return np.mean(volatilities) if volatilities else 0.2
            
        except Exception:
            return 0.2
    
    async def _adapt_parameters(self, market_data: Dict[str, Any]):
        """Adapt strategy parameters based on performance and market conditions."""
        try:
            if self.use_neural_net and len(self.performance_window) > 20:
                await self._neural_adaptation(market_data)
            else:
                await self._heuristic_adaptation()
            
            logger.debug("Adapted momentum strategy parameters")
            
        except Exception as e:
            logger.warning(f"Error adapting parameters: {e}")
    
    async def _neural_adaptation(self, market_data: Dict[str, Any]):
        """Use neural network for parameter adaptation."""
        if not self.use_neural_net or not self.neural_net:
            return
        
        try:
            # Prepare features for neural network
            features = self._extract_market_features(market_data)
            
            if features is None:
                return
            
            # Get current parameters as target
            current_params = torch.tensor([
                self.min_confidence_threshold,
                np.mean(self.timeframe_weights),
                self.config.get('base_allocation', 0.05)
            ], dtype=torch.float32)
            
            # Forward pass
            predicted_params = self.neural_net(features)
            
            # Calculate loss based on recent performance
            recent_performance = torch.tensor(
                np.mean(list(self.performance_window)[-10:]), 
                dtype=torch.float32
            )
            
            # Simple loss function - can be enhanced
            loss = self.loss_fn(predicted_params, current_params) - recent_performance
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update parameters based on neural network output
            with torch.no_grad():
                new_params = predicted_params.numpy()
                self.min_confidence_threshold = max(0.3, min(0.9, new_params[0]))
                # Update other parameters as needed
            
        except Exception as e:
            logger.warning(f"Neural adaptation failed: {e}")
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract market features for neural network."""
        try:
            features = []
            
            for symbol, data in list(market_data.items())[:3]:  # Use first 3 symbols
                if isinstance(data, pd.DataFrame) and len(data) >= 20:
                    # Price features
                    returns = data['close'].pct_change().tail(10)
                    features.extend([
                        returns.mean(),
                        returns.std(),
                        (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)  # 20-day momentum
                    ])
            
            # Pad or truncate to fixed size
            while len(features) < 10:
                features.append(0.0)
            features = features[:10]
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception:
            return None
    
    async def _heuristic_adaptation(self):
        """Simple heuristic-based parameter adaptation."""
        if len(self.performance_window) < 10:
            return
        
        try:
            # Calculate recent performance metrics
            recent_returns = list(self.performance_window)[-10:]
            avg_return = np.mean(recent_returns)
            return_volatility = np.std(recent_returns)
            
            # Adjust confidence threshold based on performance
            if avg_return < -0.01:  # Poor performance
                self.min_confidence_threshold = min(0.8, self.min_confidence_threshold + 0.05)
            elif avg_return > 0.01:  # Good performance
                self.min_confidence_threshold = max(0.4, self.min_confidence_threshold - 0.02)
            
            # Adjust timeframe weights based on volatility
            if return_volatility > 0.05:  # High volatility
                # Favor longer timeframes
                self.timeframe_weights = [0.1, 0.3, 0.6]
            else:  # Low volatility
                # More balanced across timeframes
                self.timeframe_weights = [0.2, 0.4, 0.4]
            
        except Exception as e:
            logger.warning(f"Heuristic adaptation failed: {e}")
    
    async def update_parameters(self, 
                              market_data: Dict[str, Any],
                              performance_feedback: Dict[str, Any]):
        """Update strategy parameters with performance feedback."""
        try:
            # Store performance for adaptation
            if 'return' in performance_feedback:
                self.performance_window.append(performance_feedback['return'])
            
            # Update metrics
            if 'signal' in performance_feedback:
                signal = performance_feedback['signal']
                actual_return = performance_feedback.get('return', 0.0)
                self.update_metrics(signal, actual_return)
            
        except Exception as e:
            logger.warning(f"Error updating adaptive momentum parameters: {e}")
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields."""
        return ['close', 'volume', 'high', 'low', 'open']
    
    def get_supported_assets(self) -> List[str]:
        """Get list of supported asset classes."""
        return ['equity', 'etf', 'crypto', 'forex']

class CrossAssetMomentumStrategy(BaseStrategy):
    """
    Cross-asset momentum strategy that analyzes momentum 
    across different asset classes and markets.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        super().__init__(strategy_id, "CrossAssetMomentum", config)
        
        # Asset class configuration
        self.asset_classes = config.get('asset_classes', {
            'equity': ['SPY', 'QQQ', 'IWM'],
            'bond': ['TLT', 'IEF', 'SHY'],
            'commodity': ['GLD', 'SLV', 'DBA'],
            'crypto': ['BTC-USD', 'ETH-USD'],
            'forex': ['UUP', 'FXE']
        })
        
        # Cross-asset parameters
        self.correlation_window = config.get('correlation_window', 60)
        self.momentum_lookback = config.get('momentum_lookback', 20)
        self.diversification_bonus = config.get('diversification_bonus', 0.1)
        
        # Momentum ranking
        self.top_n_assets = config.get('top_n_assets', 5)
        self.rebalance_frequency = config.get('rebalance_frequency', 5)  # days
        
        # Cross-asset momentum tracking
        self.asset_momentum_scores = {}
        self.correlation_matrix = None
        self.last_rebalance = None
        
        logger.info(f"Initialized CrossAssetMomentumStrategy with {len(self.asset_classes)} asset classes")
    
    async def generate_signal(self, 
                            market_data: Dict[str, Any],
                            portfolio_state: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate cross-asset momentum signal."""
        try:
            # Check if rebalancing is needed
            if not self._should_rebalance():
                return None
            
            # Calculate momentum scores for all assets
            await self._calculate_cross_asset_momentum(market_data)
            
            # Update correlation matrix
            self._update_correlation_matrix(market_data)
            
            # Rank assets by momentum
            ranked_assets = self._rank_assets_by_momentum()
            
            if not ranked_assets:
                return None
            
            # Select best diversified portfolio
            selected_assets = self._select_diversified_portfolio(ranked_assets)
            
            if not selected_assets:
                return None
            
            # Generate rebalancing signal
            signal = self._create_rebalancing_signal(selected_assets, market_data)
            
            self.last_rebalance = datetime.now()
            self.last_signal_time = datetime.now()
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating cross-asset momentum signal: {e}")
            return None
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.last_rebalance is None:
            return True
        
        days_since_rebalance = (datetime.now() - self.last_rebalance).days
        return days_since_rebalance >= self.rebalance_frequency
    
    async def _calculate_cross_asset_momentum(self, market_data: Dict[str, Any]):
        """Calculate momentum scores for all asset classes."""
        momentum_scores = {}
        
        for asset_class, symbols in self.asset_classes.items():
            class_scores = []
            
            for symbol in symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    if isinstance(data, pd.DataFrame) and len(data) >= self.momentum_lookback:
                        # Calculate momentum score
                        momentum = await self._calculate_asset_momentum(symbol, data)
                        if momentum is not None:
                            class_scores.append((symbol, momentum))
            
            # Average momentum for asset class
            if class_scores:
                avg_momentum = np.mean([score for _, score in class_scores])
                momentum_scores[asset_class] = {
                    'average_momentum': avg_momentum,
                    'assets': class_scores
                }
        
        self.asset_momentum_scores = momentum_scores
    
    async def _calculate_asset_momentum(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """Calculate momentum score for individual asset."""
        try:
            if len(data) < self.momentum_lookback:
                return None
            
            # Multiple momentum timeframes
            short_momentum = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1)  # 5-day
            medium_momentum = (data['close'].iloc[-1] / data['close'].iloc[-10] - 1)  # 10-day
            long_momentum = (data['close'].iloc[-1] / data['close'].iloc[-self.momentum_lookback] - 1)
            
            # Weighted combination
            combined_momentum = (short_momentum * 0.2 + 
                               medium_momentum * 0.3 + 
                               long_momentum * 0.5)
            
            # Risk adjustment
            returns = data['close'].pct_change().tail(self.momentum_lookback)
            volatility = returns.std()
            
            risk_adjusted_momentum = combined_momentum / (volatility + 1e-8)
            
            return risk_adjusted_momentum
            
        except Exception as e:
            logger.warning(f"Error calculating momentum for {symbol}: {e}")
            return None
    
    def _update_correlation_matrix(self, market_data: Dict[str, Any]):
        """Update correlation matrix for diversification analysis."""
        try:
            # Collect returns for all assets
            returns_data = {}
            
            for asset_class, symbols in self.asset_classes.items():
                for symbol in symbols:
                    if symbol in market_data:
                        data = market_data[symbol]
                        if isinstance(data, pd.DataFrame) and len(data) >= self.correlation_window:
                            returns = data['close'].pct_change().tail(self.correlation_window)
                            returns_data[symbol] = returns
            
            if len(returns_data) >= 2:
                # Create DataFrame of returns
                returns_df = pd.DataFrame(returns_data)
                returns_df = returns_df.dropna()
                
                if len(returns_df) >= 10:  # Minimum data for correlation
                    self.correlation_matrix = returns_df.corr()
            
        except Exception as e:
            logger.warning(f"Error updating correlation matrix: {e}")
    
    def _rank_assets_by_momentum(self) -> List[Tuple[str, float, str]]:
        """Rank all assets by momentum score."""
        ranked_assets = []
        
        for asset_class, class_data in self.asset_momentum_scores.items():
            for symbol, momentum in class_data['assets']:
                ranked_assets.append((symbol, momentum, asset_class))
        
        # Sort by momentum score (descending)
        ranked_assets.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_assets
    
    def _select_diversified_portfolio(self, 
                                    ranked_assets: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Select diversified portfolio from ranked assets."""
        try:
            selected_assets = []
            selected_classes = set()
            
            # First pass: select top asset from each class
            for symbol, momentum, asset_class in ranked_assets:
                if (asset_class not in selected_classes and 
                    momentum > 0 and 
                    len(selected_assets) < self.top_n_assets):
                    
                    selected_assets.append((symbol, momentum, asset_class))
                    selected_classes.add(asset_class)
            
            # Second pass: fill remaining slots with best assets
            remaining_slots = self.top_n_assets - len(selected_assets)
            
            for symbol, momentum, asset_class in ranked_assets:
                if (remaining_slots > 0 and 
                    momentum > 0 and 
                    symbol not in [s[0] for s in selected_assets]):
                    
                    # Check diversification benefit
                    if self._check_diversification_benefit(symbol, selected_assets):
                        selected_assets.append((symbol, momentum, asset_class))
                        remaining_slots -= 1
            
            return selected_assets
            
        except Exception as e:
            logger.warning(f"Error selecting diversified portfolio: {e}")
            return ranked_assets[:self.top_n_assets]  # Fallback
    
    def _check_diversification_benefit(self, 
                                     new_symbol: str, 
                                     existing_assets: List[Tuple[str, float, str]]) -> bool:
        """Check if adding new symbol improves diversification."""
        try:
            if self.correlation_matrix is None or new_symbol not in self.correlation_matrix.index:
                return True  # Default to allowing if no correlation data
            
            existing_symbols = [asset[0] for asset in existing_assets]
            
            # Calculate average correlation with existing assets
            correlations = []
            for existing_symbol in existing_symbols:
                if existing_symbol in self.correlation_matrix.index:
                    corr = self.correlation_matrix.loc[new_symbol, existing_symbol]
                    if not pd.isna(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                return avg_correlation < 0.7  # Low correlation threshold
            
            return True
            
        except Exception:
            return True  # Default to allowing
    
    def _create_rebalancing_signal(self, 
                                 selected_assets: List[Tuple[str, float, str]],
                                 market_data: Dict[str, Any]) -> StrategySignal:
        """Create rebalancing signal for selected assets."""
        try:
            if not selected_assets:
                return None
            
            # Calculate equal weights with momentum adjustment
            total_momentum = sum(abs(momentum) for _, momentum, _ in selected_assets)
            
            weights = {}
            for symbol, momentum, asset_class in selected_assets:
                # Base equal weight with momentum adjustment
                base_weight = 1.0 / len(selected_assets)
                momentum_adjustment = abs(momentum) / total_momentum if total_momentum > 0 else 0
                
                # Blend base weight with momentum weight
                final_weight = base_weight * 0.7 + momentum_adjustment * 0.3
                weights[symbol] = final_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Use the top asset for the main signal
            top_symbol, top_momentum, top_class = selected_assets[0]
            
            # Determine signal strength based on momentum strength
            if top_momentum > 0.1:
                strength = SignalStrength.VERY_STRONG
            elif top_momentum > 0.05:
                strength = SignalStrength.STRONG
            elif top_momentum > 0.03:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calculate confidence based on momentum consistency
            momentum_values = [momentum for _, momentum, _ in selected_assets]
            momentum_consistency = len([m for m in momentum_values if m > 0]) / len(momentum_values)
            
            # Add diversification bonus
            unique_classes = len(set(asset_class for _, _, asset_class in selected_assets))
            diversification_factor = 1.0 + (unique_classes - 1) * self.diversification_bonus
            
            confidence = min(1.0, momentum_consistency * diversification_factor)
            
            return StrategySignal(
                strategy_id=self.strategy_id,
                symbol=top_symbol,
                signal_type=SignalType.REBALANCE,
                strength=strength,
                confidence=confidence,
                target_weight=weights.get(top_symbol, 0.0),
                timeframe="multi_day",
                features={
                    'cross_asset_momentum': top_momentum,
                    'portfolio_weights': weights,
                    'asset_classes': unique_classes,
                    'diversification_score': diversification_factor
                },
                metadata={
                    'selected_assets': selected_assets,
                    'rebalance_type': 'cross_asset_momentum'
                },
                expected_return=np.mean(momentum_values) * 0.8,
                expected_volatility=self._estimate_portfolio_volatility(selected_assets, market_data)
            )
            
        except Exception as e:
            logger.error(f"Error creating rebalancing signal: {e}")
            return None
    
    def _estimate_portfolio_volatility(self, 
                                     selected_assets: List[Tuple[str, float, str]],
                                     market_data: Dict[str, Any]) -> float:
        """Estimate portfolio volatility."""
        try:
            volatilities = []
            
            for symbol, _, _ in selected_assets:
                if symbol in market_data:
                    data = market_data[symbol]
                    if isinstance(data, pd.DataFrame) and len(data) >= 20:
                        returns = data['close'].pct_change().tail(20)
                        vol = returns.std() * np.sqrt(252)
                        volatilities.append(vol)
            
            if volatilities:
                # Simplified portfolio volatility (assumes low correlation)
                avg_volatility = np.mean(volatilities)
                diversification_factor = np.sqrt(len(volatilities))  # Diversification benefit
                return avg_volatility / diversification_factor
            
            return 0.15  # Default volatility
            
        except Exception:
            return 0.15
    
    async def update_parameters(self, 
                              market_data: Dict[str, Any],
                              performance_feedback: Dict[str, Any]):
        """Update cross-asset momentum parameters."""
        try:
            # Adjust rebalancing frequency based on performance
            if len(self.signal_history) >= 5:
                recent_signals = list(self.signal_history)[-5:]
                avg_confidence = np.mean([s.confidence for s in recent_signals])
                
                if avg_confidence < 0.6:  # Low confidence
                    self.rebalance_frequency = min(10, self.rebalance_frequency + 1)
                elif avg_confidence > 0.8:  # High confidence
                    self.rebalance_frequency = max(3, self.rebalance_frequency - 1)
            
            # Adjust number of assets based on diversification success
            if 'diversification_score' in performance_feedback:
                div_score = performance_feedback['diversification_score']
                if div_score > 1.2:  # Good diversification
                    self.top_n_assets = min(8, self.top_n_assets + 1)
                elif div_score < 1.05:  # Poor diversification
                    self.top_n_assets = max(3, self.top_n_assets - 1)
            
            logger.debug(f"Updated cross-asset parameters: rebalance_freq={self.rebalance_frequency}, top_n={self.top_n_assets}")
            
        except Exception as e:
            logger.warning(f"Error updating cross-asset momentum parameters: {e}")
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields."""
        return ['close', 'volume', 'high', 'low']
    
    def get_supported_assets(self) -> List[str]:
        """Get list of supported asset classes."""
        return ['equity', 'etf', 'bond', 'commodity', 'crypto', 'forex']

