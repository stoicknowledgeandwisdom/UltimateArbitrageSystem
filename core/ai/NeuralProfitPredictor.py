#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Profit Prediction System
===========================

Advanced neural network for profit prediction:
- Deep learning models
- Market pattern recognition
- Price movement prediction
- Volume analysis
- Sentiment integration
- Risk assessment
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Attention, MultiHeadAttention,
    LayerNormalization, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictorConfig:
    sequence_length: int = 100
    prediction_horizon: int = 10
    feature_dimension: int = 50
    lstm_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    dense_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    training_epochs: int = 100
    validation_split: float = 0.2
    attention_heads: int = 8

@dataclass
class PredictionResult:
    predicted_prices: np.ndarray
    confidence_scores: np.ndarray
    profit_probability: float
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime

class NeuralProfitPredictor:
    """Neural network based profit prediction system"""

    def __init__(self, config: PredictorConfig):
        self.config = config
        self.model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.attention_weights = {}
        
        # Initialize neural network
        self._build_neural_network()
    
    def _build_neural_network(self) -> None:
        """Build advanced neural network architecture"""
        try:
            # Input layers
            price_input = Input(shape=(self.config.sequence_length, 1))
            feature_input = Input(shape=(self.config.sequence_length, self.config.feature_dimension))
            
            # Price processing branch
            price_lstm = self._build_lstm_stack(price_input)
            
            # Feature processing branch
            feature_conv = self._build_conv_stack(feature_input)
            
            # Combine branches with attention
            combined = self._build_attention_layer(price_lstm, feature_conv)
            
            # Prediction head
            predictions = self._build_prediction_head(combined)
            
            # Create model
            self.model = Model(
                inputs=[price_input, feature_input],
                outputs=predictions
            )
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss=self._custom_loss_function,
                metrics=['mae', 'mse']
            )
            
            logger.info("Neural network built successfully")
            
        except Exception as e:
            logger.error(f"Error building neural network: {str(e)}")
            raise
    
    def _build_lstm_stack(self, input_layer: tf.Tensor) -> tf.Tensor:
        """Build LSTM layers for price processing"""
        x = input_layer
        
        for units in self.config.lstm_units:
            # Bidirectional LSTM layer
            x = tf.keras.layers.Bidirectional(
                LSTM(units, return_sequences=True)
            )(x)
            
            # Add normalization and dropout
            x = BatchNormalization()(x)
            x = Dropout(self.config.dropout_rate)(x)
        
        return x
    
    def _build_conv_stack(self, input_layer: tf.Tensor) -> tf.Tensor:
        """Build convolutional layers for feature processing"""
        x = input_layer
        
        # Multiple convolutional layers
        for i, units in enumerate([64, 128, 256]):
            x = Conv1D(
                filters=units,
                kernel_size=3,
                activation='relu',
                padding='same',
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            )(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
        
        return x
    
    def _build_attention_layer(self, price_features: tf.Tensor,
                            market_features: tf.Tensor) -> tf.Tensor:
        """Build multi-head attention layer"""
        # Self attention for price features
        price_attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=64
        )(price_features, price_features)
        price_attention = LayerNormalization()(price_attention + price_features)
        
        # Self attention for market features
        market_attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=64
        )(market_features, market_features)
        market_attention = LayerNormalization()(market_attention + market_features)
        
        # Cross attention
        cross_attention = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=64
        )(price_attention, market_attention)
        cross_attention = LayerNormalization()(cross_attention + price_attention)
        
        return cross_attention
    
    def _build_prediction_head(self, features: tf.Tensor) -> tf.Tensor:
        """Build prediction head for output"""
        x = features
        
        # Dense layers
        for units in self.config.dense_units:
            x = Dense(
                units,
                activation='relu',
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        predictions = Dense(
            self.config.prediction_horizon * 3,  # Price, Confidence, Risk
            activation='linear'
        )(x)
        
        return predictions
    
    def _custom_loss_function(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Custom loss function for training"""
        # Split predictions
        price_pred = y_pred[:, :self.config.prediction_horizon]
        confidence = y_pred[:, self.config.prediction_horizon:2*self.config.prediction_horizon]
        risk = y_pred[:, 2*self.config.prediction_horizon:]
        
        # True values
        price_true = y_true[:, :self.config.prediction_horizon]
        
        # Calculate losses
        price_loss = tf.keras.losses.mean_squared_error(price_true, price_pred)
        confidence_loss = -tf.reduce_mean(confidence * tf.abs(price_true - price_pred))
        risk_loss = tf.reduce_mean(risk * tf.square(price_true - price_pred))
        
        # Combine losses
        total_loss = price_loss + 0.2 * confidence_loss + 0.1 * risk_loss
        
        return total_loss
    
    async def predict_profit(self, market_data: Dict[str, Any]) -> PredictionResult:
        """Predict profit opportunities"""
        try:
            # Prepare input data
            price_data, feature_data = self._prepare_input_data(market_data)
            
            # Make prediction
            predictions = self.model.predict(
                [price_data, feature_data],
                batch_size=1
            )
            
            # Process predictions
            result = self._process_predictions(predictions[0])
            
            # Calculate profit probability
            profit_prob = self._calculate_profit_probability(result)
            
            # Assess risks
            risk_assessment = self._assess_prediction_risks(result)
            
            return PredictionResult(
                predicted_prices=result['prices'],
                confidence_scores=result['confidence'],
                profit_probability=profit_prob,
                risk_assessment=risk_assessment,
                metadata=self._generate_prediction_metadata(result),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting profit: {str(e)}")
            raise
    
    def _prepare_input_data(self, market_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input data for prediction"""
        try:
            # Extract price and feature data
            prices = np.array([d['price'] for d in market_data['history']])
            features = np.array([d['features'] for d in market_data['history']])
            
            # Scale data
            prices_scaled = self.price_scaler.fit_transform(prices.reshape(-1, 1))
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Reshape for model input
            price_input = prices_scaled[-self.config.sequence_length:].reshape(1, -1, 1)
            feature_input = features_scaled[-self.config.sequence_length:].reshape(
                1, -1, self.config.feature_dimension
            )
            
            return price_input, feature_input
            
        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise
    
    def _process_predictions(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Process model predictions"""
        try:
            # Split predictions
            prices = predictions[:self.config.prediction_horizon]
            confidence = predictions[self.config.prediction_horizon:2*self.config.prediction_horizon]
            risk = predictions[2*self.config.prediction_horizon:]
            
            # Inverse transform prices
            prices = self.price_scaler.inverse_transform(prices.reshape(-1, 1)).flatten()
            
            # Apply sigmoid to confidence and risk scores
            confidence = 1 / (1 + np.exp(-confidence))
            risk = 1 / (1 + np.exp(-risk))
            
            return {
                'prices': prices,
                'confidence': confidence,
                'risk': risk
            }
            
        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            raise
    
    def _calculate_profit_probability(self, prediction_result: Dict[str, np.ndarray]) -> float:
        """Calculate probability of profitable trade"""
        try:
            # Get predictions
            prices = prediction_result['prices']
            confidence = prediction_result['confidence']
            risk = prediction_result['risk']
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Weight by confidence and risk
            weighted_changes = price_changes * confidence[1:] * (1 - risk[1:])
            
            # Calculate probability
            positive_moves = np.sum(weighted_changes > 0)
            total_moves = len(weighted_changes)
            
            return positive_moves / total_moves if total_moves > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating profit probability: {str(e)}")
            return 0.0
    
    def _assess_prediction_risks(self, prediction_result: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Assess risks in predictions"""
        try:
            # Get predictions
            prices = prediction_result['prices']
            confidence = prediction_result['confidence']
            risk = prediction_result['risk']
            
            # Calculate metrics
            volatility = np.std(prices) / np.mean(prices)
            avg_confidence = np.mean(confidence)
            max_drawdown = self._calculate_max_drawdown(prices)
            trend_strength = self._calculate_trend_strength(prices)
            
            return {
                'volatility_risk': float(volatility),
                'confidence_risk': float(1 - avg_confidence),
                'drawdown_risk': float(max_drawdown),
                'trend_risk': float(1 - trend_strength),
                'overall_risk': float(np.mean(risk))
            }
            
        except Exception as e:
            logger.error(f"Error assessing prediction risks: {str(e)}")
            return {'overall_risk': 1.0}
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peaks = np.maximum.accumulate(prices)
        drawdowns = (peaks - prices) / peaks
        return np.max(drawdowns)
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength"""
        try:
            # Calculate price changes
            changes = np.diff(prices)
            positive_changes = np.sum(changes > 0)
            total_changes = len(changes)
            
            # Calculate trend consistency
            trend_strength = abs(positive_changes/total_changes - 0.5) * 2
            
            return trend_strength
            
        except Exception:
            return 0.0
    
    def _generate_prediction_metadata(self, prediction_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate metadata for prediction"""
        return {
            'sequence_length': self.config.sequence_length,
            'prediction_horizon': self.config.prediction_horizon,
            'mean_confidence': float(np.mean(prediction_result['confidence'])),
            'mean_risk': float(np.mean(prediction_result['risk'])),
            'price_volatility': float(np.std(prediction_result['prices'])),
            'timestamp': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = PredictorConfig(
        sequence_length=100,
        prediction_horizon=10,
        feature_dimension=50
    )
    
    # Initialize predictor
    predictor = NeuralProfitPredictor(config)
    
    # Example market data
    market_data = {
        'history': [
            {
                'price': 100 + i + np.random.normal(0, 1),
                'features': np.random.random(50)
            } for i in range(200)
        ]
    }
    
    async def test_predictor():
        # Make prediction
        result = await predictor.predict_profit(market_data)
        
        print("\nPrediction Result:")
        print(f"Profit Probability: {result.profit_probability:.2%}")
        print(f"Risk Assessment: {result.risk_assessment}")
        print(f"Confidence Scores: {result.confidence_scores}")
    
    # Run test
    import asyncio
    asyncio.run(test_predictor())

