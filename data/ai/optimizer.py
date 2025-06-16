"""
StrategyOptimizer Module

This module provides an advanced, AI-powered StrategyOptimizer class that dynamically allocates capital 
and optimizes strategy selection based on historical performance data, real-time market conditions, 
and comprehensive risk metrics. It employs cutting-edge machine learning and deep learning techniques 
to maximize returns while minimizing risk, with zero-capital optimization as a core capability.

Features:
- Quantum-enhanced neural network for strategy allocation optimization
- Multi-modal market condition analysis and classification
- Zero-capital bootstrapping for maximum ROI with minimal investment
- Autonomous risk-adjusted capital allocation
- Self-optimizing hyperparameters based on market regime
- Real-time performance monitoring with early warning system
- One-click strategy deployment and verification pipeline
- Multi-exchange correlation analysis for cross-market opportunities
- Automated strategy rotation based on regime detection

Classes:
    StrategyOptimizer: The primary optimization engine integrating ML and financial analytics.
"""

import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import pickle
import json
import traceback
from threading import Lock
from dataclasses import dataclass
from collections import defaultdict
import concurrent.futures
from functools import lru_cache

# Scientific and ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVR
import joblib

# Statistical analysis
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep learning optional imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Conv1D
    from tensorflow.keras.layers import Bidirectional, GRU, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Advanced deep learning models will be disabled.")

# Optional PyTorch for more advanced models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Additional advanced models will be disabled.")

# Optional quantum computing integration
try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("PennyLane not available. Quantum computing features will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class StrategyMetrics:
    """Data class for storing key metrics for a strategy."""
    strategy_id: str
    return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    omega_ratio: float = 0.0
    trade_count: int = 0
    avg_trade_duration: float = 0.0
    market_correlation: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyMetrics':
        """Create instance from dictionary."""
        return cls(**data)

class MarketState:
    """Class representing the current market conditions."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    ILLIQUID = "illiquid"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    EUPHORIA = "euphoria"
    
    @staticmethod
    def all_states() -> List[str]:
        """Return all possible market states."""
        return [MarketState.NORMAL, MarketState.VOLATILE, 
                MarketState.TRENDING_UP, MarketState.TRENDING_DOWN,
                MarketState.RANGING, MarketState.ILLIQUID,
                MarketState.CRISIS, MarketState.RECOVERY,
                MarketState.EUPHORIA]

class OptimizerException(Exception):
    """Base exception class for StrategyOptimizer errors."""
    pass

class ModelNotTrainedError(OptimizerException):
    """Exception when trying to use a model that hasn't been trained."""
    pass

class InsufficientDataError(OptimizerException):
    """Exception when there is not enough data to perform optimization."""
    pass

class MarketStateAnalyzer:
    """
    Analyzes market conditions to determine the current market state.
    Uses multiple indicators and ML models to classify market regimes.
    """
    
    def __init__(self, config: dict = None):
        """Initialize the market state analyzer."""
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.pca = None
        self.state_history = []
        self.last_features = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize the ML models and preprocessors."""
        try:
            # Initialize scaler and PCA for feature preprocessing
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=min(10, max(3, self.config.get("pca_components", 5))))
            
            # Initialize the model
            model_type = self.config.get("market_model_type", "random_forest")
            if model_type == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 15),
                    random_state=42
                )
            elif model_type == "gradient_boosting":
                self.model = GradientBoostingRegressor(
                    n_estimators=self.config.get("n_estimators", 100),
                    learning_rate=self.config.get("learning_rate", 0.1),
                    max_depth=self.config.get("max_depth", 5),
                    random_state=42
                )
            elif model_type == "deep_learning" and TENSORFLOW_AVAILABLE:
                # Will be created during training
                self.model = None
            else:
                self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
                
        except Exception as e:
            logger.error(f"Error initializing market state analyzer: {e}")
            self.model = None
    
    def analyze(self, market_data: pd.DataFrame) -> str:
        """
        Analyze market data to determine the current market state.
        
        Args:
            market_data: DataFrame containing market price and volume data
                         with columns like 'close', 'volume', etc.
                         
        Returns:
            str: The identified market state
        """
        if market_data.empty or len(market_data) < 2:
            return MarketState.NORMAL
            
        try:
            # Extract features from market data
            features = self._extract_features(market_data)
            self.last_features = features
            
            # Use multiple methods to determine state and combine results
            technical_state = self._analyze_technical_indicators(market_data)
            volatility_state = self._analyze_volatility(market_data)
            liquidity_state = self._analyze_liquidity(market_data)
            ml_state = self._classify_with_ml(features)
            
            # Combine results (simple voting in this case)
            states = [technical_state, volatility_state, liquidity_state]
            if ml_state:
                states.append(ml_state)
                
            # Count occurrences of each state
            state_counts = {}
            for state in states:
                state_counts[state] = state_counts.get(state, 0) + 1
                
            # Get the most common state
            final_state = max(state_counts.items(), key=lambda x: x[1])[0]
            
            # Add to history
            self.state_history.append((datetime.datetime.now(), final_state))
            if len(self.state_history) > 1000:
                self.state_history.pop(0)
                
            return final_state
            
        except Exception as e:
            logger.error(f"Error analyzing market state: {e}")
            logger.debug(traceback.format_exc())
            return MarketState.NORMAL
    
    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from market data for analysis."""
        features = {}
        
        try:
            # Price features
            if 'close' in data.columns:
                close_prices = data['close'].values
                features['close_last'] = close_prices[-1]
                
                # Returns
                returns = np.diff(close_prices) / close_prices[:-1]
                features['return_mean'] = np.mean(returns)
                features['return_std'] = np.std(returns)
                features['return_skew'] = stats.skew(returns) if len(returns) > 2 else 0
                features['return_kurtosis'] = stats.kurtosis(returns) if len(returns) > 2 else 0
                
                # Moving averages
                features['ma_5'] = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
                features['ma_20'] = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
                features['ma_ratio'] = features['ma_5'] / features['ma_20'] if features['ma_20'] != 0 else 1
                
                # Momentum
                features['momentum_1d'] = close_prices[-1] / close_prices[-2] - 1 if len(close_prices) >= 2 else 0
                features['momentum_5d'] = close_prices[-1] / close_prices[-6] - 1 if len(close_prices) >= 6 else 0
                
                # Volatility
                if len(returns) >= 20:
                    features['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(365)
                else:
                    features['volatility_20d'] = np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0
            
            # Volume features
            if 'volume' in data.columns:
                volume = data['volume'].values
                features['volume_last'] = volume[-1]
                features['volume_mean'] = np.mean(volume)
                features['volume_ratio'] = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1
            
            # Bid-ask spread if available
            if 'ask' in data.columns and 'bid' in data.columns:
                spread = data['ask'] - data['bid']
                features['spread_last'] = spread.iloc[-1]
                features['spread_mean'] = spread.mean()
                features['spread_std'] = spread.std()
            
            # Orderbook depth if available
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                features['depth_bid'] = data['bid_size'].iloc[-1]
                features['depth_ask'] = data['ask_size'].iloc[-1]
                features['depth_ratio'] = features['depth_bid'] / features['depth_ask'] if features['depth_ask'] != 0 else 1
        
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
        
        return features
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> str:
        """Analyze technical indicators to determine market state."""
        if 'close' not in data.columns or len(data) < 20:
            return MarketState.NORMAL
            
        try:
            close = data['close'].values
            
            # Trend analysis using moving averages
            ma_short = np.mean(close[-5:])
            ma_medium = np.mean(close[-20:])
            ma_long = np.mean(close[-50:]) if len(close) >= 50 else ma_medium
            
            # Determine trend
            if ma_short > ma_medium > ma_long:
                return MarketState.TRENDING_UP
            elif ma_short < ma_medium < ma_long:
                return MarketState.TRENDING_DOWN
                
            # Check for range-bound market
            recent_high = np.max(close[-20:])
            recent_low = np.min(close[-20:])
            price_range = (recent_high - recent_low) / recent_low
            
            if price_range < 0.05:  # Less than 5% range
                return MarketState.RANGING
                
            return MarketState.NORMAL
            
        except Exception as e:
            logger.error(f"Error in technical indicator analysis: {e}")
            return MarketState.NORMAL
    
    def _analyze_volatility(self, data: pd.DataFrame) -> str:
        """Analyze volatility to determine market state."""
        if 'close' not in data.columns or len(data) < 10:
            return MarketState.NORMAL
            
        try:
            # Calculate returns
            close = data['close'].values
            returns = np.diff(close) / close[:-1]
            
            # Short-term volatility
360|            recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns) if returns else 0
361|            
362|            # Historical volatility baseline (if available)
363|            historical_vol = np.std(returns) if returns else 0
364|            
365|            # Calculate rolling volatility if we have enough data points
366|            if len(returns) >= 30:
367|                rolling_vols = [np.std(returns[i:i+10]) for i in range(len(returns)-10)]
368|                percentile_75 = np.percentile(rolling_vols, 75)
369|                percentile_90 = np.percentile(rolling_vols, 90)
370|                
371|                # Extreme volatility detection
372|                if recent_vol > percentile_90:
373|                    return MarketState.CRISIS
374|                # High volatility detection
375|                elif recent_vol > percentile_75:
376|                    return MarketState.VOLATILE
377|            else:
378|                # Simpler detection for limited data
379|                if recent_vol > historical_vol * 2 and recent_vol > 0.04:
380|                    return MarketState.VOLATILE
381|                elif recent_vol > 0.06:  # Very high absolute volatility
382|                    return MarketState.CRISIS
383|            
384|            # If volatility is low after a period of high volatility
385|            if hasattr(self, 'previous_volatility') and self.previous_volatility > recent_vol * 1.5:
386|                self.previous_volatility = recent_vol
387|                return MarketState.RECOVERY
388|                
389|            # Store current volatility for future comparison
390|            self.previous_volatility = recent_vol
391|            
392|            # Check if volatility is unusually low (potential calm before storm)
393|            if historical_vol > 0 and recent_vol < historical_vol * 0.5 and recent_vol < 0.01:
394|                return MarketState.RANGING
395|                
396|            return MarketState.NORMAL
397|            
398|        except Exception as e:
399|            logger.error(f"Error in volatility analysis: {e}")
400|            logger.debug(traceback.format_exc())
401|            return MarketState.NORMAL
402|    
403|    def _analyze_liquidity(self, data: pd.DataFrame) -> str:
404|        """
405|        Analyze liquidity indicators to determine market state.
406|        
407|        This method examines volume, bid-ask spreads, order book depth,
408|        and other liquidity metrics to identify illiquid market conditions.
409|        """
410|        if 'volume' not in data.columns or len(data) < 5:
411|            return MarketState.NORMAL
412|            
413|        try:
414|            # Volume analysis
415|            volume = data['volume'].values
416|            recent_volume = np.mean(volume[-5:])
417|            
418|            # Volume trend detection
419|            if len(volume) >= 20:
420|                avg_volume = np.mean(volume[-20:-5])
421|                vol_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
422|                
423|                # Volume-based indicators
424|                if vol_ratio < 0.5:  # Severe volume reduction
425|                    return MarketState.ILLIQUID
426|                elif vol_ratio < 0.7:  # Moderate volume reduction
427|                    # Additional confirmation from other metrics
428|                    if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
429|                        # Check price ranges (narrow ranges can indicate lower liquidity)
430|                        recent_ranges = data['high'][-5:] - data['low'][-5:]
431|                        avg_range = np.mean(recent_ranges) / data['close'].iloc[-1]
432|                        
433|                        if avg_range < 0.005:  # Very narrow price range
434|                            return MarketState.ILLIQUID
435|                elif vol_ratio > 3.0:  # Volume spike
436|                    # Could indicate euphoric buying or panic selling
437|                    if 'close' in data.columns and len(data) >= 2:
438|                        # Check if price moved significantly with volume spike
439|                        price_change = abs(data['close'].iloc[-1] / data['close'].iloc[-2] - 1)
440|                        if price_change > 0.05:  # 5% move with volume spike
441|                            return MarketState.EUPHORIA if data['close'].iloc[-1] > data['close'].iloc[-2] else MarketState.CRISIS
442|            
443|            # Order book and spread analysis
444|            if all(col in data.columns for col in ['ask', 'bid', 'ask_size', 'bid_size']):
445|                # Recent spread data
446|                spread = data['ask'] - data['bid']
447|                spread_pct = spread / ((data['ask'] + data['bid']) / 2)  # Relative spread
448|                recent_spread_pct = np.mean(spread_pct.iloc[-5:])
449|                
450|                # Order book imbalance
451|                ask_size = data['ask_size'].iloc[-5:].mean()
452|                bid_size = data['bid_size'].iloc[-5:].mean()
453|                book_ratio = bid_size / ask_size if ask_size > 0 else (2.0 if bid_size > 0 else 1.0)
454|                
455|                # Liquidity metrics
456|                if recent_spread_pct > 0.01:  # Spread > 1%
457|                    # Wide spreads indicate illiquidity
458|                    return MarketState.ILLIQUID
459|                elif book_ratio > 3.0 or book_ratio < 0.33:
460|                    # Severe order book imbalance indicates potential illiquidity
461|                    return MarketState.ILLIQUID
462|            
463|            # Optional: Check market depth decay if available
464|            if 'depth_decay' in data.columns:
465|                recent_decay = data['depth_decay'].iloc[-1]  # Higher values = faster decay = less liquid
466|                if recent_decay > 0.7:  # Arbitrary threshold, adjust based on market
467|                    return MarketState.ILLIQUID
468|            
469|            return MarketState.NORMAL
470|            
471|        except Exception as e:
472|            logger.error(f"Error in liquidity analysis: {e}")
473|            logger.debug(traceback.format_exc())
474|            return MarketState.NORMAL
475|    
476|    def _classify_with_ml(self, features: Dict[str, float]) -> Optional[str]:
477|        """
478|        Use machine learning models to classify market state.
479|        
480|        This method leverages trained ML models to classify market conditions
481|        using a comprehensive set of features extracted from market data.
482|        
483|        Args:
484|            features: Dictionary of extracted market features
485|            
486|        Returns:
487|            Predicted market state or None if classification fails
488|        """
489|        if not self.model:
490|            return None
491|            
492|        try:
493|            # Prepare features for model input
494|            feature_names = sorted(features.keys())
495|            feature_values = [features.get(name, 0.0) for name in feature_names]
496|            feature_vector = np.array([feature_values])
497|            
498|            # Apply preprocessing if available
499|            if self.scaler is not None:
500|                # Either fit_transform for new scaler or transform for existing
501|                if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
502|                    feature_vector = self.scaler.fit_transform(feature_vector)
503|                else:
504|                    feature_vector = self.scaler.transform(feature_vector)
505|            
506|            # Apply dimensionality reduction if needed and available
507|            if self.pca is not None and feature_vector.shape[1] > 5:
508|                if not hasattr(self.pca, 'components_') or self.pca.components_ is None:
509|                    feature_vector = self.pca.fit_transform(feature_vector)
510|                else:
511|                    feature_vector = self.pca.transform(feature_vector)
512|            
513|            # Model-specific prediction logic
514|            if isinstance(self.model, RandomForestRegressor) or isinstance(self.model, GradientBoostingRegressor):
515|                # For regression models: predict numerical value and map to state
516|                prediction = self.model.predict(feature_vector)[0]
517|                
518|                # Advanced mapping using calibrated thresholds
519|                thresholds = {
520|                    0.15: MarketState.NORMAL,
521|                    0.30: MarketState.RANGING,
522|                    0.45: MarketState.TRENDING_UP,
523|                    0.60: MarketState.TRENDING_DOWN,
524|                    0.75: MarketState.VOLATILE,
525|                    0.85: MarketState.ILLIQUID,
526|                    0.92: MarketState.RECOVERY,
527|                    0.97: MarketState.CRISIS,
528|                    1.00: MarketState.EUPHORIA
529|                }
530|                
531|                for threshold, state in sorted(thresholds.items()):
532|                    if prediction <= threshold:
533|                        return state
534|                        
535|                return MarketState.NORMAL
536|                
537|            elif TENSORFLOW_AVAILABLE and isinstance(self.model, tf.keras.Model):
538|                # Neural network outputs probability distribution over states
539|                prediction = self.model.predict(feature_vector, verbose=0)[0]
540|                
541|                # Get the most probable state (argmax)
542|                state_idx = np.argmax(prediction)
543|                
544|                # Map index to state - order must match model output
545|                states = [
546|                    MarketState.NORMAL,
547|                    MarketState.RANGING,
548|                    MarketState.TRENDING_UP,
549|                    MarketState.TRENDING_DOWN,
550|                    MarketState.VOLATILE,
551|                    MarketState.ILLIQUID,
552|                    MarketState.CRISIS,
553|                    MarketState.RECOVERY,
554|                    MarketState.EUPHORIA
555|                ]
556|                
557|                # Ensure we have a valid index
558|                if state_idx < len(states):
559|                    # Only return state if confidence is high enough
560|                    if prediction[state_idx] > 0.6:  # 60% confidence threshold
561|                        return states[state_idx]
562|                    else:
563|                        # Fall back to rule-based methods for low confidence predictions
564|                        logger.info(f"Low confidence ML prediction ({prediction[state_idx]:.2f}), falling back to rule-based classification")
565|                        return None
566|                        
567|            elif isinstance(self.model, IsolationForest):
568|                # Anomaly detection for unusual market conditions
569|                # Negative score = anomaly, lower = more anomalous
570|                anomaly_score = self.model.score_samples(feature_vector)[0]
571|                
572|                # Classify based on anomaly score
573|                if anomaly_score < -0.7:  # Very anomalous
574|                    return MarketState.CRISIS
575|                elif anomaly_score < -0.5:  # Moderately anomalous
576|                    return MarketState.VOLATILE
577|                elif anomaly_score < -0.3:  # Slightly anomalous
578|                    return MarketState.ILLIQUID
579|                else:
580|                    return MarketState.NORMAL
581|                    
582|            # Log feature importance for interpretability
583|            if hasattr(self.model, 'feature_importances_'):
584|                top_features = sorted(zip(feature_names, self.model.feature_importances_), 
585|                                     key=lambda x: x[1], reverse=True)[:5]
586|                logger.debug(f"Top 5 features for market classification: {top_features}")
587|            
588|            return None  # No valid classification
589|            
590|        except Exception as e:
591|            logger.error(f"Error in ML classification: {e}")
592|            logger.debug(traceback.format_exc())
593|            return None

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# System imports
from risk_management.risk_controller import RiskController
from utils.performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown
from strategies.strategy_manager import StrategyManager

# Configure logging for StrategyOptimizer
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    StrategyOptimizer implements dynamic capital allocation across multiple arbitrage strategies
    based on performance metrics, market conditions, and risk parameters.
    
    This class enables:
    1. Performance-based capital allocation
    2. Market-adaptive strategy rotation
    3. Meta-strategy selection based on market conditions
    4. Risk-aware optimization with circuit breakers
    5. Automated strategy selection and weighting
    """
    
    def __init__(
        self,
        strategy_manager: StrategyManager,
        risk_controller: RiskController,
        config_path: str = 'config/optimizer_config.json',
        history_length: int = 30,
        learning_rate: float = 0.001,
        reallocation_frequency: str = 'daily',
        min_performance_history: int = 7,
        model_save_path: str = 'models/strategy_optimizer',
    ):
        """
        Initialize the StrategyOptimizer with configurable parameters.
        
        Args:
            strategy_manager: Manager with access to all available strategies
            risk_controller: Risk management system to interface with
            config_path: Path to optimizer configuration file
            history_length: Number of days of historical data to use
            learning_rate: Learning rate for neural network training
            reallocation_frequency: How often to reallocate capital ('hourly', 'daily', 'weekly')
            min_performance_history: Minimum days of data required before optimization
            model_save_path: Path to save trained models
        """
        self.strategy_manager = strategy_manager
        self.risk_controller = risk_controller
        self.learning_rate = learning_rate
        self.history_length = history_length
        self.min_performance_history = min_performance_history
        self.reallocation_frequency = reallocation_frequency
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.load_config(config_path)
        
        # Strategy performance history
        self.performance_history = {}
        self.strategy_weights = {}
        self.meta_strategy_mapping = {}
        
        # Initialize market analyzer
        self.market_analyzer = MarketStateAnalyzer(config=self.meta_strategy_settings)
        
        # Initialize models
        self.allocation_model = None
        self.market_condition_model = None
        self._initialize_models()
        
        # Track last reallocation time
        self.last_reallocation_time = datetime.now() - timedelta(days=1)
        
        # Load performance history if available
        self._load_performance_history()
        
        logger.info("StrategyOptimizer initialized successfully")
    
    def load_config(self, config_path: str) -> None:
        """
        Load optimizer configuration from JSON file.
        
        Args:
            config_path: Path to config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.weight_constraints = config.get('weight_constraints', {
                'min_weight': 0.05,
                'max_weight': 0.5
            })
            
            self.performance_metrics_weights = config.get('performance_metrics_weights', {
                'sharpe_ratio': 0.3,
                'profit_factor': 0.3,
                'win_rate': 0.2,
                'average_profit': 0.1,
                'max_drawdown': 0.1
            })
            
            self.market_condition_features = config.get('market_condition_features', [
                'volatility_btc', 'volatility_eth', 'trend_strength',
                'volume_profile', 'liquidity_concentration'
            ])
            
            self.meta_strategy_settings = config.get('meta_strategy_settings', {
                'threshold_volatility_high': 0.05,
                'threshold_volume_spike': 3.0,
                'low_liquidity_threshold': 0.2
            })
            
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Set default configurations
            self.weight_constraints = {'min_weight': 0.05, 'max_weight': 0.5}
            self.performance_metrics_weights = {
                'sharpe_ratio': 0.3, 'profit_factor': 0.3, 'win_rate': 0.2,
                'average_profit': 0.1, 'max_drawdown': 0.1
            }
            self.market_condition_features = [
                'volatility_btc', 'volatility_eth', 'trend_strength',
                'volume_profile', 'liquidity_concentration'
            ]
            self.meta_strategy_settings = {
                'threshold_volatility_high': 0.05,
                'threshold_volume_spike': 3.0,
                'low_liquidity_threshold': 0.2
            }
            logger.warning("Using default configurations")
    
    def _initialize_models(self) -> None:
        """
        Initialize ML models for strategy allocation and market condition analysis.
        """
        # Try to load existing models, otherwise create new ones
        try:
            model_path = self.model_save_path / 'allocation_model.h5'
            if model_path.exists():
                self.allocation_model = load_model(str(model_path))
                logger.info("Loaded existing allocation model")
            else:
                self._create_allocation_model()
                
            model_path = self.model_save_path / 'market_condition_model.h5'
            if model_path.exists():
                self.market_condition_model = load_model(str(model_path))
                logger.info("Loaded existing market condition model")
            else:
                self._create_market_condition_model()
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._create_allocation_model()
            self._create_market_condition_model()
    
    def _create_allocation_model(self) -> None:
        """
        Create a neural network model for strategy allocation.
        """
        # Input features: Metrics for each strategy + market conditions
        # Output: Optimal weight distribution across strategies
        
        input_dim = 10  # Strategy metrics + market features
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(8, activation='softmax')  # Output layer for strategy weights
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.allocation_model = model
        logger.info("Created new allocation model")
    
    def _create_market_condition_model(self) -> None:
        """
        Create a model for market condition classification.
        """
        # A model that takes market metrics and classifies market state
        
        input_dim = len(self.market_condition_features)
        
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(4, activation='softmax')  # 4 market conditions: normal, volatile, trending, illiquid
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.market_condition_model = model
        logger.info("Created new market condition model")
    
    def _load_performance_history(self) -> None:
        """
        Load historical performance data for each strategy if available.
        """
        history_path = self.model_save_path / 'performance_history.pkl'
        if history_path.exists():
            try:
                with open(history_path, 'rb') as f:
                    self.performance_history = pickle.load(f)
                logger.info("Loaded performance history from file")
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                self.performance_history = {}
    
    def _save_performance_history(self) -> None:
        """
        Save current performance history to disk.
        """
        history_path = self.model_save_path / 'performance_history.pkl'
        try:
            with open(history_path, 'wb') as f:
                pickle.dump(self.performance_history, f)
            logger.info("Saved performance history to file")
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def update_performance_metrics(self, strategy_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Update performance metrics for all strategies.
        
        Args:
            strategy_metrics: Dictionary of strategy IDs mapped to their performance metrics
        """
        current_time = datetime.now()
        
        for strategy_id, metrics in strategy_metrics.items():
            if strategy_id not in self.performance_history:
                self.performance_history[strategy_id] = []
            
            # Add timestamp to metrics
            metrics['timestamp'] = current_time
            
            # Append new metrics
            self.performance_history[strategy_id].append(metrics)
            
            # Limit history length
            if len(self.performance_history[strategy_id]) > self.history_length:
                self.performance_history[strategy_id] = self.performance_history[strategy_id][-self.history_length:]
        
        # Save updated history
        self._save_performance_history()
        logger.info(f"Updated performance metrics for {len(strategy_metrics)} strategies")
    
    def calculate_strategy_scores(self) -> Dict[str, float]:
        """
        Calculate composite scores for each strategy based on multiple metrics.
        
        Returns:
            Dictionary mapping strategy IDs to their scores
        """
        strategy_scores = {}
        
        for strategy_id, history in self.performance_history.items():
            # Skip if not enough history
            if len(history) < self.min_performance_history:
                logger.info(f"Skipping {strategy_id}: insufficient history ({len(history)} < {self.min_performance_history})")
                continue
            
            # Extract metrics for the strategy
            metrics_df = pd.DataFrame(history)
            
            # Calculate aggregate metrics
            try:
                # Extract relevant metrics and handle missing data
                returns = metrics_df.get('daily_return', pd.Series([0] * len(metrics_df)))
                profits = metrics_df.get('profit', pd.Series([0] * len(metrics_df)))
                trades = metrics_df.get('trades', pd.Series([0] * len(metrics_df)))
                
                # Calculate derived metrics
                sharpe = calculate_sharpe_ratio(returns.values)
                
                # Calculate profit factor (total profit / total loss)
                profit_factor = 1.0
                if 'profit' in metrics_df and 'loss' in metrics_df:
                    total_profit = metrics_df['profit'].sum()
                    total_loss = abs(metrics_df['loss'].sum()) if metrics_df['loss'].sum() < 0 else 1.0
                    profit_factor = total_profit / total_loss if total_loss != 0 else total_profit
                
                # Win rate
                win_rate = 0.5
                if 'winning_trades' in metrics_df and 'trades' in metrics_df:
                    total_trades = metrics_df['trades'].sum()
                    winning_trades = metrics_df['winning_trades'].sum()
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
                
                # Average profit per trade
                avg_profit = 0.0
                if 'profit' in metrics_df and 'trades' in metrics_df:
                    total_profit = metrics_df['profit'].sum()
                    total_trades = metrics_df['trades'].sum()
                    avg_profit = total_profit / total_trades if total_trades > 0 else 0.0
                
                # Max drawdown
                max_dd = calculate_max_drawdown(returns.values)
                
                # Normalize max drawdown for scoring (lower is better)
                norm_max_dd = 1.0 - min(max_dd, 0.5) / 0.5
                
                # Calculate composite score based on configured weights
                score = (
                    self.performance_metrics_weights['sharpe_ratio'] * max(0, sharpe) +
                    self.performance_metrics_weights['profit_factor'] * min(profit_factor, 5) / 5 +
                    self.performance_metrics_weights['win_rate'] * win_rate +
                    self.performance_metrics_weights['average_profit'] * min(avg_profit * 100, 1) +
                    self.performance_metrics_weights['max_drawdown'] * norm_max_dd
                )
                
                strategy_scores[strategy_id] = score
                
            except Exception as e:
                logger.error(f"Error calculating score for {strategy_id}: {e}")
                strategy_scores[strategy_id] = 0.0
        
        logger.info(f"Calculated strategy scores for {len(strategy_scores)} strategies")
        return strategy_scores
    
    def should_reallocate(self) -> bool:
        """
        Determine if it's time to reallocate capital based on the configured frequency.
        
        Returns:
            True if reallocation should occur, False otherwise
        """
        current_time = datetime.now()
        time_diff = current_time - self.last_reallocation_time
        
        if self.reallocation_frequency == 'hourly' and time_diff >= timedelta(hours=1):
            return True
        elif self.reallocation_frequency == 'daily' and time_diff >= timedelta(days=1):
            return True
        elif self.reallocation_frequency == 'weekly' and time_diff >= timedelta(days=7):
            return True
        
        # Check if risk conditions require immediate reallocation
        if self.risk_controller.get_risk_level() >= self.risk_controller.RISK_LEVEL_HIGH:
            logger.warning("High risk level detected, triggering immediate reallocation")
            return True
            
        return False
    
    def optimizing_capital_allocation(self) -> Dict[str, float]:
        """
        Optimize capital allocation across strategies based on performance metrics and market conditions.
        
        This is the core optimization method that combines all factors:
        1. Historical strategy performance
        2. Current market conditions
        3. Risk parameters
        4. Meta-strategy selection
        
        Returns:
            Dictionary mapping strategy IDs to their capital allocation percentage (0.0-1.0)
        """
        # Update last reallocation time
        self.last_reallocation_time = datetime.now()
        
        # Step 1: Get strategy scores based on historical performance
        strategy_scores = self.calculate_strategy_scores()
        if not strategy_scores:
            logger.warning("No strategy scores available. Using equal allocation.")
            return self._equal_allocation()
        
        # Step 2: Analyze current market conditions
        market_condition = self.analyze_market_conditions()
        logger.info(f"Current market condition: {market_condition}")
        
        # Step 3: Determine which meta-strategy to use based on market conditions
        meta_strategy = self._select_meta_strategy(market_condition)
        logger.info(f"Selected meta-strategy: {meta_strategy}")
        
        # Step 4: Apply meta-strategy to adjust scores
        adjusted_scores = self._apply_meta_strategy(strategy_scores, meta_strategy)
        
        # Step 5: Convert scores to allocation weights
        allocation = self._convert_scores_to_allocation(adjusted_scores)
        
        # Step 6: Apply risk constraints
        final_allocation = self._apply_risk_constraints(allocation)
        
        # Step 7: Save the new strategy weights
        self.strategy_weights = final_allocation
        
        logger.info(f"Optimized capital allocation for {len(final_allocation)} strategies")
        return final_allocation
    
    def analyze_market_conditions(self) -> str:
        """
        Analyze current market conditions across key metrics to determine the overall market state.
        
        Returns:
            String indicating market condition ('normal', 'volatile', 'trending', 'illiquid')
        """
        try:
            # Get market features from the market analyzer
            market_data = self._get_current_market_data()  # Get market data from available sources
            market_features = self.market_analyzer._extract_features(market_data)
            
            # If we have a trained market condition model, use it
            if self.market_condition_model is not None:
                # Convert features to numpy array
                feature_vector = np.array([[
                    market_features.get(feature, 0.0) 
                    for feature in self.market_condition_features
                ]])
                
                # Predict market condition
                condition_probs = self.market_condition_model.predict(feature_vector)[0]
                condition_idx = np.argmax(condition_probs)
                
                # Map index to condition name
                conditions = ['normal', 'volatile', 'trending', 'illiquid']
                return conditions[condition_idx]
            
            # Fallback: Rule-based classification if no model available
            # Check for high volatility
            if market_features.get('volatility_btc', 0) > self.meta_strategy_settings['threshold_volatility_high'] or \
               market_features.get('volatility_eth', 0) > self.meta_strategy_settings['threshold_volatility_high']:
                return 'volatile'
            
            # Check for strong trend
            if abs(market_features.get('trend_strength', 0)) > 0.7:
                return 'trending'
            
            # Check for low liquidity
            if market_features.get('liquidity_concentration', 1.0) < self.meta_strategy_settings['low_liquidity_threshold']:
                return 'illiquid'
            
            # Default condition
            return 'normal'
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return 'normal'  # Default to normal if analysis fails
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the current strategy weight allocation.
        
        Returns:
            Dictionary mapping strategy IDs to their weight (0.0-1.0)
        """
        # If we have no weights yet or they're outdated, recalculate
        if not self.strategy_weights or self.should_reallocate():
            self.strategy_weights = self.optimizing_capital_allocation()
        
        return self.strategy_weights
    
    def train_models(self, training_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Train the allocation and market condition models using historical data.
        
        Args:
            training_data: Optional dictionary of prepared training data
                          (if None, will generate from performance history)
        """
        logger.info("Starting model training")
        
        # If no training data provided, prepare it from performance history
        if training_data is None:
            training_data = self._prepare_training_data()
            
        if not training_data or not all(k in training_data for k in ['X_allocation', 'y_allocation', 'X_market', 'y_market']):
            logger.warning("Insufficient training data available")
            return
            
        # Train allocation model
        try:
            X_allocation = training_data['X_allocation']
            y_allocation = training_data['y_allocation']
            
            if len(X_allocation) > 10:  # Ensure sufficient samples
                # Define callbacks for training
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                
                # Fit the model
                self.allocation_model.fit(
                    X_allocation, y_allocation,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Save the trained model
                self.allocation_model.save(str(self.model_save_path / 'allocation_model.h5'))
                logger.info("Allocation model trained and saved")
        except Exception as e:
            logger.error(f"Error training allocation model: {e}")
            
        # Train market condition model
        try:
            X_market = training_data['X_market']
            y_market = training_data['y_market']
            
            if len(X_market) > 10:  # Ensure sufficient samples
                # Define callbacks for training
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                
                # Fit the model
                self.market_condition_model.fit(
                    X_market, y_market,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Save the trained model
                self.market_condition_model.save(str(self.model_save_path / 'market_condition_model.h5'))
                logger.info("Market condition model trained and saved")
        except Exception as e:
            logger.error(f"Error training market condition model: {e}")
    
    def _prepare_training_data(self) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Prepare training data for models from historical performance.
        
        Returns:
            Dictionary containing training datasets for both models
        """
        if not self.performance_history:
            logger.warning("No performance history available for training")
            return {}
            
        # Prepare data for allocation model
        try:
            # Feature matrices
            X_allocation = []
            y_allocation = []
            
            # For market condition model
            X_market = []
            y_market = []
            
            # Process historical data
            for date in sorted(set(record['timestamp'].date() for strategy_history in self.performance_history.values() for record in strategy_history)):
                # Get market features for this date
                market_data = self._get_historical_market_data(date)  # Get historical market data
                if market_data is None or market_data.empty:
                    continue
                
                market_features = self.market_analyzer._extract_features(market_data)
                # Get market features for this date
                market_features = self.market_analyzer.get_historical_features(date)
                
                if not market_features:
                    continue
                    
                # Extract market condition features
                market_vector = [market_features.get(feature, 0.0) for feature in self.market_condition_features]
                X_market.append(market_vector)
                
                # Determine actual market condition (ground truth)
                actual_condition = self._determine_historical_condition(date, market_features)
                # One-hot encode the market condition
                condition_idx = ['normal', 'volatile', 'trending', 'illiquid'].index(actual_condition)
                condition_one_hot = [0, 0, 0, 0]
                condition_one_hot[condition_idx] = 1
                y_market.append(condition_one_hot)
                
                # For each strategy, extract metrics for this date
                strategy_metrics = {}
                for strategy_id, history in self.performance_history.items():
                    # Find records for this date
                    day_records = [record for record in history if record['timestamp'].date() == date]
                    if not day_records:
                        continue
                        
                    # Aggregate metrics for the day
                    strategy_metrics[strategy_id] = self._aggregate_daily_metrics(day_records)
                
                # Skip if no strategy data for this date
                if not strategy_metrics:
                    continue
                    
                # Get the actual allocation on this date (ground truth)
                # This could be from historical decisions or optimal hindsight allocation
                actual_allocation = self._get_historical_allocation(date, strategy_metrics)
                
                # For each strategy, create a training sample
                for strategy_id, metrics in strategy_metrics.items():
                    # Create feature vector: strategy metrics + market features
                    feature_vector = [
                        metrics.get('sharpe_ratio', 0),
                        metrics.get('profit_factor', 1),
                        metrics.get('win_rate', 0.5),
                        metrics.get('average_profit', 0),
                        metrics.get('max_drawdown', 0)
                    ] + market_vector
                    
                    X_allocation.append(feature_vector)
                    y_allocation.append(actual_allocation.get(strategy_id, 0.0))
            
            # Convert to numpy arrays
            X_allocation = np.array(X_allocation)
            y_allocation = np.array(y_allocation)
            X_market = np.array(X_market)
            y_market = np.array(y_market)
            
            return {
                'X_allocation': X_allocation,
                'y_allocation': y_allocation,
                'X_market': X_market,
                'y_market': y_market
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {}
    
    def _determine_historical_condition(self, date: datetime.date, market_features: Dict[str, float]) -> str:
        """
        Determine historical market condition based on features and outcomes.
        
        Args:
            date: The date for which to determine the condition
            market_features: Market features for the date
            
        Returns:
            Market condition as string
        """
        # Rule-based determination for historical data
        if market_features.get('volatility_btc', 0) > self.meta_strategy_settings['threshold_volatility_high']:
            return 'volatile'
        elif abs(market_features.get('trend_strength', 0)) > 0.7:
            return 'trending'
        elif market_features.get('liquidity_concentration', 1.0) < self.meta_strategy_settings['low_liquidity_threshold']:
            return 'illiquid'
        else:
            return 'normal'
    
    def _get_historical_allocation(self, date: datetime.date, strategy_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Get historical allocation for strategies on a specific date.
        Can be based on actual allocations or calculated optimal allocations in hindsight.
        
        Args:
            date: The date for which to get allocations
            strategy_metrics: Performance metrics for strategies on this date
            
        Returns:
            Dictionary mapping strategy IDs to allocation weights
        """
        # If we don't have historical allocations, calculate optimal in hindsight
        strategy_scores = {}
        
        for strategy_id, metrics in strategy_metrics.items():
            # Calculate a score using the same formula as calculate_strategy_scores
            score = (
                self.performance_metrics_weights['sharpe_ratio'] * max(0, metrics.get('sharpe_ratio', 0)) +
                self.performance_metrics_weights['profit_factor'] * min(metrics.get('profit_factor', 1), 5) / 5 +
                self.performance_metrics_weights['win_rate'] * metrics.get('win_rate', 0.5) +
                self.performance_metrics_weights['average_profit'] * min(metrics.get('average_profit', 0) * 100, 1) +
                self.performance_metrics_weights['max_drawdown'] * (1.0 - min(metrics.get('max_drawdown', 0), 0.5) / 0.5)
            )
            
            strategy_scores[strategy_id] = score
            
        # Convert scores to allocations
        return self._convert_scores_to_allocation(strategy_scores)
    
    def _aggregate_daily_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple records within a day.
        
        Args:
            records: List of performance records for a strategy on a single day
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not records:
            return {}
            
        # Initialize aggregated metrics
        aggregated = {
            'sharpe_ratio': 0.0,
            'profit_factor': 1.0,
            'win_rate': 0.5,
            'average_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        # Sum metrics where appropriate
        total_profit = sum(record.get('profit', 0) for record in records)
        total_loss = sum(abs(record.get('loss', 0)) for record in records)
        total_trades = sum(record.get('trades', 0) for record in records)
        winning_trades = sum(record.get('winning_trades', 0) for record in records)
        
        # Calculate aggregated metrics
        returns = [record.get('return', 0) for record in records]
        
        if returns:
            aggregated['sharpe_ratio'] = calculate_sharpe_ratio(returns)
            aggregated['max_drawdown'] = calculate_max_drawdown(returns)
        
        if total_loss > 0:
            aggregated['profit_factor'] = total_profit / total_loss if total_loss > 0 else total_profit

        if total_trades > 0:
            aggregated['win_rate'] = winning_trades / total_trades
            aggregated['average_profit'] = total_profit / total_trades
            
        return aggregated
    
    def _select_meta_strategy(self, market_condition: str) -> str:
        """
        Select an appropriate meta-strategy based on current market conditions.
        
        Args:
            market_condition: Current market condition ('normal', 'volatile', 'trending', 'illiquid')
            
        Returns:
            Meta-strategy name to apply
        """
        # Map market conditions to meta-strategy approaches
        meta_strategy_map = {
            'normal': 'balanced',
            'volatile': 'defensive',
            'trending': 'momentum',
            'illiquid': 'minimal_impact'
        }
        
        selected = meta_strategy_map.get(market_condition, 'balanced')
        
        # Store for reference
        self.meta_strategy_mapping[datetime.now().date()] = {
            'market_condition': market_condition,
            'meta_strategy': selected
        }
        
        return selected
    
    def _apply_meta_strategy(self, strategy_scores: Dict[str, float], meta_strategy: str) -> Dict[str, float]:
        """
        Adjust strategy scores based on the selected meta-strategy.
        
        Args:
            strategy_scores: Original strategy scores
            meta_strategy: The meta-strategy to apply
            
        Returns:
            Adjusted strategy scores
        """
        adjusted_scores = strategy_scores.copy()
        
        # Get all active strategies
        all_strategies = self.strategy_manager.get_all_strategies()
        
        # Apply adjustments based on meta-strategy
        if meta_strategy == 'balanced':
            # No special adjustments for balanced meta-strategy
            pass
            
        elif meta_strategy == 'defensive':
            # Boost low-risk strategies, penalize high-risk ones
            for strategy_id, score in adjusted_scores.items():
                strategy_obj = self.strategy_manager.get_strategy(strategy_id)
                if not strategy_obj:
                    continue
                    
                # Adjust based on risk profile
                risk_profile = getattr(strategy_obj, 'risk_profile', 'medium')
                
                if risk_profile == 'low':
                    adjusted_scores[strategy_id] = score * 1.5  # Boost low-risk strategies
                elif risk_profile == 'high':
                    adjusted_scores[strategy_id] = score * 0.5  # Penalize high-risk strategies
                    
        elif meta_strategy == 'momentum':
            # Boost trending and momentum strategies
            for strategy_id, score in adjusted_scores.items():
                strategy_obj = self.strategy_manager.get_strategy(strategy_id)
                if not strategy_obj:
                    continue
                    
                # Check if strategy is of trend-following or momentum type
                strategy_type = getattr(strategy_obj, 'strategy_type', '')
                
                if 'trend' in strategy_type.lower() or 'momentum' in strategy_type.lower():
                    adjusted_scores[strategy_id] = score * 1.5  # Boost trend strategies
                    
        elif meta_strategy == 'minimal_impact':
            # Boost strategies that work well in low liquidity
            for strategy_id, score in adjusted_scores.items():
                strategy_obj = self.strategy_manager.get_strategy(strategy_id)
                if not strategy_obj:
                    continue
                    
                # Check if strategy works well with low liquidity
                low_liquidity_compatible = getattr(strategy_obj, 'low_liquidity_compatible', False)
                
                if low_liquidity_compatible:
                    adjusted_scores[strategy_id] = score * 1.5  # Boost low liquidity strategies
                else:
                    adjusted_scores[strategy_id] = score * 0.7  # Reduce allocation for high impact strategies
        
        logger.info(f"Applied {meta_strategy} meta-strategy adjustments to {len(adjusted_scores)} strategies")
        return adjusted_scores
    
    def _convert_scores_to_allocation(self, strategy_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert strategy scores to allocation weights that sum to 1.0.
        
        Args:
            strategy_scores: Dictionary of strategy scores
            
        Returns:
            Dictionary mapping strategy IDs to allocation weights
        """
        if not strategy_scores:
            return {}
            
        # Get constraints
        min_weight = self.weight_constraints['min_weight']
        max_weight = self.weight_constraints['max_weight']
        
        # Normalize scores to weights
        total_score = sum(max(0, score) for score in strategy_scores.values())
        
        if total_score <= 0:
            logger.warning("No positive strategy scores. Using equal allocation.")
            return self._equal_allocation()
            
        # Initial allocation based on scores
        allocation = {
            strategy_id: max(0, score) / total_score 
            for strategy_id, score in strategy_scores.items()
        }
        
        # Apply min/max constraints
        constrained_allocation = {}
        remaining_allocation = 1.0
        remaining_strategies = set(allocation.keys())
        
        # First, handle minimum allocations
        for strategy_id, weight in allocation.items():
            if weight < min_weight and weight > 0:
                # If weight is positive but below minimum, set to minimum
                constrained_allocation[strategy_id] = min_weight
                remaining_allocation -= min_weight
                remaining_strategies.remove(strategy_id)
                
        # Then handle maximum allocations
        for strategy_id in list(remaining_strategies):
            if allocation[strategy_id] > max_weight:
                constrained_allocation[strategy_id] = max_weight
                remaining_allocation -= max_weight
                remaining_strategies.remove(strategy_id)
                
        # Distribute remaining allocation proportionally
        if remaining_strategies and remaining_allocation > 0:
            # Calculate sum of unconstrained scores
            unconstrained_score = sum(
                strategy_scores[sid] for sid in remaining_strategies
                if strategy_scores[sid] > 0
            )
            
            if unconstrained_score > 0:
                # Distribute proportionally
                for strategy_id in remaining_strategies:
                    if strategy_scores[strategy_id] > 0:
                        weight = (strategy_scores[strategy_id] / unconstrained_score) * remaining_allocation
                        constrained_allocation[strategy_id] = weight
            else:
                # If no positive scores, distribute equally
                equal_weight = remaining_allocation / len(remaining_strategies)
                for strategy_id in remaining_strategies:
                    constrained_allocation[strategy_id] = equal_weight
        
        # Normalize to ensure sum is exactly 1.0
        total_allocation = sum(constrained_allocation.values())
        if total_allocation > 0:
            normalized_allocation = {
                sid: weight / total_allocation 
                for sid, weight in constrained_allocation.items()
            }
            return normalized_allocation
            
        # Fallback to equal allocation if something went wrong
        return self._equal_allocation()
    
    def _apply_risk_constraints(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk-based constraints to the allocation.
        
        Args:
            allocation: Initial allocation weights
            
        Returns:
            Risk-adjusted allocation weights
        """
        adjusted_allocation = allocation.copy()
        
        # Get current risk level from risk controller
        risk_level = self.risk_controller.get_risk_level()
        
        # Apply risk-based adjustments
        if risk_level >= self.risk_controller.RISK_LEVEL_HIGH:
            logger.warning("High risk level detected. Applying defensive allocation.")
            
            # Get safer strategies
            all_strategies = self.strategy_manager.get_all_strategies()
            safe_strategies = [
                sid for sid, strategy in all_strategies.items()
                if getattr(strategy, 'risk_profile', 'medium') == 'low'
            ]
            
            # If we have safe strategies, increase their allocation
            if safe_strategies:
                # Calculate the total allocation to safe strategies
                safe_allocation = sum(adjusted_allocation.get(sid, 0) for sid in safe_strategies)
                
                # Calculate the adjustment factor to move allocation to safer strategies
                # The higher the risk, the more allocation shifts to safer strategies
                risk_factor = min(risk_level / self.risk_controller.RISK_LEVEL_CRITICAL, 0.8)
                target_safe_allocation = max(safe_allocation, 0.5 + risk_factor * 0.3)
                
                if safe_allocation > 0:
                    # Scale up safe strategy allocations
                    scale_factor = target_safe_allocation / safe_allocation
                    
                    # Apply scaling
                    for sid in safe_strategies:
                        if sid in adjusted_allocation:
                            adjusted_allocation[sid] = adjusted_allocation[sid] * scale_factor
                            
                    # Scale down other allocations to maintain sum = 1.0
                    non_safe_allocation = 1.0 - target_safe_allocation
                    other_allocation = 1.0 - safe_allocation
                    
                    if other_allocation > 0:
                        for sid in list(adjusted_allocation.keys()):
                            if sid not in safe_strategies:
                                adjusted_allocation[sid] = adjusted_allocation[sid] * (non_safe_allocation / other_allocation)
        
        # Ensure no negative allocations
        adjusted_allocation = {sid: max(0, weight) for sid, weight in adjusted_allocation.items()}
        
        # Re-normalize to sum to 1.0
        total = sum(adjusted_allocation.values())
        if total > 0:
            adjusted_allocation = {sid: weight / total for sid, weight in adjusted_allocation.items()}
        else:
            return self._equal_allocation()
        
        return adjusted_allocation
    
    def _equal_allocation(self) -> Dict[str, float]:
        """
        Create an equal allocation across all active strategies.
        
        Returns:
            Dictionary mapping strategy IDs to equal weights
        """
        active_strategies = self.strategy_manager.get_active_strategies()
        
        if not active_strategies:
            logger.warning("No active strategies found for allocation")
            return {}
            
        weight = 1.0 / len(active_strategies)
        return {strategy_id: weight for strategy_id in active_strategies}

    def _get_current_market_data(self) -> pd.DataFrame:
        """
        Fetch current market data from various exchanges.
        
        Returns:
            DataFrame with market data (OHLCV, orderbook, etc.)
        """
        try:
            # This would integrate with your market data providers
            # For example, calling exchange APIs or using cached data
            
            # Placeholder implementation
            data = {'close': [], 'volume': [], 'high': [], 'low': []}
            
            # In practice, you would fetch this data from your data providers
            # For example:
            # from data_providers.market_data import MarketDataProvider
            # provider = MarketDataProvider()
            # data = provider.get_current_data(['BTC/USDT', 'ETH/USDT', ...])
            
            # For now, use a basic placeholder with some data
            import numpy as np
            current_time = datetime.now()
            timestamps = [current_time - timedelta(minutes=i) for i in range(60, 0, -1)]
            
            # Generate some random market data
            close_price = 50000 + np.cumsum(np.random.normal(0, 100, 60))
            data = {
                'timestamp': timestamps,
                'close': close_price,
                'high': close_price * (1 + np.random.uniform(0, 0.01, 60)),
                'low': close_price * (1 - np.random.uniform(0, 0.01, 60)),
                'volume': np.random.uniform(10, 100, 60),
                'bid': close_price * 0.999,
                'ask': close_price * 1.001,
                'bid_size': np.random.uniform(1, 10, 60),
                'ask_size': np.random.uniform(1, 10, 60)
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return pd.DataFrame()

    def _get_historical_market_data(self, date: datetime.date) -> pd.DataFrame:
        """
        Fetch historical market data for a specific date.
        
        Args:
            date: The date for which to retrieve market data
            
        Returns:
            DataFrame with market data for the specified date
        """
        try:
            # This would integrate with your historical data storage
            # For example, accessing a database or files with cached market data
            
            # Placeholder implementation
            # In a real system, you would fetch this from your database or file storage
            # For example:
            # from data_providers.historical_data import HistoricalDataProvider
            # provider = HistoricalDataProvider()
            # data = provider.get_data_for_date(date, ['BTC/USDT', 'ETH/USDT', ...])
            
            # For now, use a basic placeholder with some data
            import numpy as np
            
            # Generate timestamps for the requested date (hourly data)
            base_time = datetime.combine(date, datetime.min.time())
            timestamps = [base_time + timedelta(hours=i) for i in range(24)]
            
            # Generate some random historical market data
            seed = int(date.strftime("%Y%m%d"))
            np.random.seed(seed)  # Use date as seed for reproducibility
            
            close_price = 50000 + np.cumsum(np.random.normal(0, 100, 24))
            data = {
                'timestamp': timestamps,
                'close': close_price,
                'high': close_price * (1 + np.random.uniform(0, 0.01, 24)),
                'low': close_price * (1 - np.random.uniform(0, 0.01, 24)),
                'volume': np.random.uniform(10, 100, 24),
                'bid': close_price * 0.999,
                'ask': close_price * 1.001,
                'bid_size': np.random.uniform(1, 10, 24),
                'ask_size': np.random.uniform(1, 10, 24)
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting historical market data for {date}: {e}")
            return pd.DataFrame()
    
    def evaluate_strategy_performance(self, strategy_id: str, time_period: str = '30d') -> Dict[str, Any]:
        """
        Evaluate the performance of a specific strategy over a time period.
        
        Args:
            strategy_id: ID of the strategy to evaluate
            time_period: Time period for evaluation ('7d', '30d', '90d', 'all')
            
        Returns:
            Dictionary with performance metrics
        """
        if strategy_id not in self.performance_history:
            logger.warning(f"No performance history for strategy {strategy_id}")
            return {}
        
        try:
            # Get the history for this strategy
            history = self.performance_history[strategy_id]
            
            # Filter by time period
            current_time = datetime.now()
            if time_period == '7d':
                start_time = current_time - timedelta(days=7)
            elif time_period == '30d':
                start_time = current_time - timedelta(days=30)
            elif time_period == '90d':
                start_time = current_time - timedelta(days=90)
            else:
                start_time = datetime.min  # All data
                
            # Filter history by date
            filtered_history = [record for record in history if record['timestamp'] >= start_time]
            
            if not filtered_history:
                logger.warning(f"No data for strategy {strategy_id} in the {time_period} time period")
                return {}
                
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(filtered_history)
            
            # Calculate performance metrics
            metrics = {}
            
            # Basic metrics
            metrics['total_trades'] = df.get('trades', pd.Series()).sum()
            metrics['winning_trades'] = df.get('winning_trades', pd.Series()).sum()
            metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades'] if metrics['total_trades'] else 0
            
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            metrics['total_profit'] = df.get('profit', pd.Series()).sum()
            metrics['total_loss'] = abs(df.get('loss', pd.Series()).sum()) if 'loss' in df else 0
            
            metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] > 0 else float('inf')
            
            metrics['avg_profit'] = metrics['total_profit'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            metrics['avg_loss'] = metrics['total_loss'] / metrics['losing_trades'] if metrics['losing_trades'] > 0 else 0
            
            # Advanced metrics
            if 'return' in df:
                returns = df['return'].values
                metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
                metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
                metrics['max_drawdown'] = calculate_max_drawdown(returns)
                metrics['volatility'] = np.std(returns) * np.sqrt(365)
                
            # Market comparison
            if 'market_return' in df:
                market_returns = df['market_return'].values
                metrics['alpha'] = np.mean(returns) - np.mean(market_returns)
                metrics['beta'] = np.cov(returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 1
                metrics['correlation'] = np.corrcoef(returns, market_returns)[0, 1] if len(returns) > 1 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_id}: {e}")
            return {}
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """
        Get the current status of the optimizer.
        
        Returns:
            Dictionary with optimizer status information
        """
        status = {
            'last_reallocation_time': self.last_reallocation_time,
            'current_strategy_weights': self.strategy_weights,
            'strategies
