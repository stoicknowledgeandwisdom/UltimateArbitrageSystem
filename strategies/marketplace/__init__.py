#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Strategy Marketplace
============================

A comprehensive ecosystem for AI-driven trading strategies with:
- Dynamic strategy loading and execution
- Performance-based strategy selection
- Real-time strategy adaptation
- Multi-asset class strategy support
- Quantum-enhanced strategy optimization
- Advanced risk-adjusted performance metrics
- Strategy combination and ensemble methods
"""

from .base_strategy import BaseStrategy, StrategySignal, StrategyMetrics
from .strategy_manager import AdvancedStrategyManager, StrategyConfig
from .momentum_strategies import (
    QuantumMomentumStrategy,
    AdaptiveMomentumStrategy,
    CrossAssetMomentumStrategy
)
from .mean_reversion_strategies import (
    AIEnhancedMeanReversionStrategy,
    StatisticalArbitrageStrategy,
    PairsTrading2Strategy
)
from .volatility_strategies import (
    VolatilityTargetingStrategy,
    VIXBasedStrategy,
    GARCHVolatilityStrategy
)
from .arbitrage_strategies import (
    CrossExchangeArbitrageStrategy,
    IndexArbitrageStrategy,
    CalendarSpreadStrategy
)
from .ml_strategies import (
    DeepReinforcementLearningStrategy,
    EnsembleMachineLearningStrategy,
    NeuralNetworkStrategy
)
from .strategy_optimizer import StrategyOptimizer, OptimizationResult
from .performance_attribution import PerformanceAttributor

__all__ = [
    'BaseStrategy',
    'StrategySignal',
    'StrategyMetrics',
    'AdvancedStrategyManager',
    'StrategyConfig',
    'QuantumMomentumStrategy',
    'AdaptiveMomentumStrategy',
    'CrossAssetMomentumStrategy',
    'AIEnhancedMeanReversionStrategy',
    'StatisticalArbitrageStrategy',
    'PairsTrading2Strategy',
    'VolatilityTargetingStrategy',
    'VIXBasedStrategy',
    'GARCHVolatilityStrategy',
    'CrossExchangeArbitrageStrategy',
    'IndexArbitrageStrategy',
    'CalendarSpreadStrategy',
    'DeepReinforcementLearningStrategy',
    'EnsembleMachineLearningStrategy',
    'NeuralNetworkStrategy',
    'StrategyOptimizer',
    'OptimizationResult',
    'PerformanceAttributor'
]

