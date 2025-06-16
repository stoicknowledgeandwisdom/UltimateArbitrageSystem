"""Explainability Dashboard with SHAP values for ML model interpretability."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Some explainability features will be limited.")

try:
    import lime
    import lime.lime_text
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Some explainability features will be limited.")


@dataclass
class ExplanationConfig:
    """Configuration for explainability dashboard."""
    feature_names: List[str] = None
    max_explanations_stored: int = 1000
    explanation_batch_size: int = 100
    shap_sample_size: int = 100
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    enable_real_time_explanation: bool = True
    explanation_threshold: float = 0.1  # Minimum importance for feature display
    dashboard_update_interval: float = 5.0  # seconds


@dataclass
class ModelExplanation:
    """Model explanation data structure."""
    timestamp: datetime
    model_id: str
    prediction: float
    features: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    explanation_method: str = "unknown"
    context: Optional[Dict[str, Any]] = None


class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model, feature_names: List[str], explainer_type: str = "auto"):
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer = None
        self.background_data = None
        self.logger = logging.getLogger(__name__)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAPExplainer")
    
    def initialize_explainer(self, background_data: np.ndarray):
        """Initialize SHAP explainer with background data."""
        try:
            self.background_data = background_data
            
            if self.explainer_type == "tree" or hasattr(self.model, 'predict_proba'):
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, background_data)
            elif self.explainer_type == "kernel":
                # General purpose but slower
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            elif self.explainer_type == "deep":
                # For deep learning models
                self.explainer = shap.DeepExplainer(self.model, background_data)
            else:
                # Auto-detect
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except:
                    try:
                        self.explainer = shap.LinearExplainer(self.model, background_data)
                    except:
                        self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            
            self.logger.info(f"Initialized SHAP explainer: {type(self.explainer).__name__}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain(self, features: np.ndarray, sample_size: int = 100) -> Dict[str, float]:
        """Generate SHAP explanations for features."""
        try:
            if self.explainer is None:
                raise ValueError("Explainer not initialized")
            
            # Limit sample size for performance
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class case, take first class or average
                if len(shap_values) > 1:
                    shap_values = np.mean(shap_values, axis=0)
                else:
                    shap_values = shap_values[0]
            
            # Convert to dictionary
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample if batch
            
            explanation = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values):
                    explanation[feature_name] = float(shap_values[i])
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanation: {e}")
            return {}


class LIMEExplainer:
    """LIME-based model explainer."""
    
    def __init__(self, model, feature_names: List[str], mode: str = "tabular"):
        self.model = model
        self.feature_names = feature_names
        self.mode = mode
        self.explainer = None
        self.logger = logging.getLogger(__name__)
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for LIMEExplainer")
    
    def initialize_explainer(self, training_data: np.ndarray):
        """Initialize LIME explainer with training data."""
        try:
            if self.mode == "tabular":
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data,
                    feature_names=self.feature_names,
                    mode='regression',  # or 'classification'
                    random_state=42
                )
            
            self.logger.info(f"Initialized LIME explainer in {self.mode} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LIME explainer: {e}")
            raise
    
    def explain(self, features: np.ndarray, num_features: int = 10, num_samples: int = 5000) -> Dict[str, float]:
        """Generate LIME explanations for features."""
        try:
            if self.explainer is None:
                raise ValueError("Explainer not initialized")
            
            if len(features.shape) > 1:
                features = features[0]  # Take first sample if batch
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                features,
                self.model.predict,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Convert to dictionary
            explanation_dict = {}
            for feature_idx, importance in explanation.as_list():
                if isinstance(feature_idx, int) and feature_idx < len(self.feature_names):
                    feature_name = self.feature_names[feature_idx]
                else:
                    feature_name = str(feature_idx)
                explanation_dict[feature_name] = float(importance)
            
            return explanation_dict
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanation: {e}")
            return {}


class ExplainabilityDashboard:
    """Comprehensive explainability dashboard for ML models."""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Explainers
        self.shap_explainer: Optional[SHAPExplainer] = None
        self.lime_explainer: Optional[LIMEExplainer] = None
        
        # Explanation storage
        self.explanations: deque = deque(maxlen=config.max_explanations_stored)
        self.feature_importance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Dashboard data
        self.dashboard_data: Dict[str, Any] = {
            'feature_importance': {},
            'prediction_timeline': [],
            'explanation_stats': {},
            'model_performance': {}
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Dashboard update
        self.dashboard_active = False
        self.dashboard_thread: Optional[threading.Thread] = None
        
        # Start dashboard updates if enabled
        if config.enable_real_time_explanation:
            self._start_dashboard_updates()
    
    def setup_shap_explainer(self, model, background_data: np.ndarray, explainer_type: str = "auto"):
        """Setup SHAP explainer."""
        try:
            feature_names = self.config.feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]
            self.shap_explainer = SHAPExplainer(model, feature_names, explainer_type)
            self.shap_explainer.initialize_explainer(background_data)
            self.logger.info("SHAP explainer setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup SHAP explainer: {e}")
    
    def setup_lime_explainer(self, model, training_data: np.ndarray, mode: str = "tabular"):
        """Setup LIME explainer."""
        try:
            feature_names = self.config.feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
            self.lime_explainer = LIMEExplainer(model, feature_names, mode)
            self.lime_explainer.initialize_explainer(training_data)
            self.logger.info("LIME explainer setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup LIME explainer: {e}")
    
    async def explain_prediction(self, model_id: str, features: np.ndarray, prediction: float, 
                               context: Optional[Dict[str, Any]] = None) -> ModelExplanation:
        """Generate comprehensive explanation for a prediction."""
        try:
            feature_dict = {}
            if self.config.feature_names:
                for i, name in enumerate(self.config.feature_names):
                    if i < len(features):
                        feature_dict[name] = float(features[i])
            
            explanation = ModelExplanation(
                timestamp=datetime.now(),
                model_id=model_id,
                prediction=prediction,
                features=feature_dict,
                context=context
            )
            
            # Generate SHAP explanation
            if self.shap_explainer:
                try:
                    shap_values = self.shap_explainer.explain(features, self.config.shap_sample_size)
                    explanation.shap_values = shap_values
                    explanation.explanation_method = "SHAP"
                except Exception as e:
                    self.logger.error(f"SHAP explanation failed: {e}")
            
            # Generate LIME explanation
            if self.lime_explainer:
                try:
                    lime_values = self.lime_explainer.explain(
                        features, 
                        self.config.lime_num_features, 
                        self.config.lime_num_samples
                    )
                    explanation.lime_explanation = lime_values
                    if not explanation.explanation_method or explanation.explanation_method == "unknown":
                        explanation.explanation_method = "LIME"
                except Exception as e:
                    self.logger.error(f"LIME explanation failed: {e}")
            
            # Calculate confidence based on explanation consistency
            explanation.confidence = self._calculate_explanation_confidence(explanation)
            
            # Store explanation
            with self._lock:
                self.explanations.append(explanation)
                
                # Update feature importance history
                if explanation.shap_values:
                    for feature, importance in explanation.shap_values.items():
                        self.feature_importance_history[feature].append(importance)
                
                # Update prediction history
                self.prediction_history.append({
                    'timestamp': explanation.timestamp,
                    'prediction': prediction,
                    'confidence': explanation.confidence
                })
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            # Return basic explanation
            return ModelExplanation(
                timestamp=datetime.now(),
                model_id=model_id,
                prediction=prediction,
                features={},
                context=context
            )
    
    def _calculate_explanation_confidence(self, explanation: ModelExplanation) -> float:
        """Calculate confidence score for explanation."""
        try:
            confidence = 0.5  # Base confidence
            
            # If we have SHAP values, calculate confidence based on magnitude
            if explanation.shap_values:
                shap_magnitudes = [abs(v) for v in explanation.shap_values.values()]
                if shap_magnitudes:
                    # Higher magnitude SHAP values indicate more confident explanations
                    avg_magnitude = np.mean(shap_magnitudes)
                    confidence += min(0.4, avg_magnitude * 2)  # Cap at 0.9 total
            
            # If we have both SHAP and LIME, check consistency
            if explanation.shap_values and explanation.lime_explanation:
                # Calculate correlation between SHAP and LIME values
                common_features = set(explanation.shap_values.keys()) & set(explanation.lime_explanation.keys())
                if len(common_features) > 2:
                    shap_vals = [explanation.shap_values[f] for f in common_features]
                    lime_vals = [explanation.lime_explanation[f] for f in common_features]
                    
                    correlation = np.corrcoef(shap_vals, lime_vals)[0, 1]
                    if not np.isnan(correlation):
                        confidence += max(0, correlation * 0.3)  # Bonus for consistency
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating explanation confidence: {e}")
            return 0.5
    
    def _start_dashboard_updates(self):
        """Start dashboard update thread."""
        if self.dashboard_active:
            return
        
        self.dashboard_active = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_update_loop, daemon=True)
        self.dashboard_thread.start()
        self.logger.info("Started dashboard updates")
    
    def _dashboard_update_loop(self):
        """Dashboard update loop."""
        while self.dashboard_active:
            try:
                self._update_dashboard_data()
                threading.Event().wait(self.config.dashboard_update_interval)
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                threading.Event().wait(5.0)
    
    def _update_dashboard_data(self):
        """Update dashboard data structures."""
        with self._lock:
            try:
                # Update feature importance summary
                current_importance = {}
                for feature, history in self.feature_importance_history.items():
                    if history:
                        current_importance[feature] = {
                            'current': history[-1],
                            'mean': np.mean(list(history)),
                            'std': np.std(list(history)),
                            'trend': self._calculate_trend(list(history))
                        }
                
                self.dashboard_data['feature_importance'] = current_importance
                
                # Update prediction timeline
                self.dashboard_data['prediction_timeline'] = list(self.prediction_history)[-100:]  # Last 100
                
                # Update explanation stats
                if self.explanations:
                    recent_explanations = list(self.explanations)[-50:]  # Last 50
                    
                    avg_confidence = np.mean([exp.confidence for exp in recent_explanations])
                    method_counts = defaultdict(int)
                    for exp in recent_explanations:
                        method_counts[exp.explanation_method] += 1
                    
                    self.dashboard_data['explanation_stats'] = {
                        'total_explanations': len(self.explanations),
                        'avg_confidence': avg_confidence,
                        'method_distribution': dict(method_counts),
                        'last_updated': datetime.now().isoformat()
                    }
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard data: {e}")
    
    def _calculate_trend(self, values: List[float], window: int = 20) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        recent_values = values[-window:] if len(values) > window else values
        if len(recent_values) < 2:
            return "stable"
        
        # Simple trend calculation
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_feature_importance_plot(self, num_features: int = 15) -> go.Figure:
        """Generate feature importance plot."""
        try:
            feature_data = self.dashboard_data.get('feature_importance', {})
            
            if not feature_data:
                return go.Figure().add_annotation(
                    text="No feature importance data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            # Sort features by absolute mean importance
            sorted_features = sorted(
                feature_data.items(),
                key=lambda x: abs(x[1]['mean']),
                reverse=True
            )[:num_features]
            
            features = [item[0] for item in sorted_features]
            means = [item[1]['mean'] for item in sorted_features]
            stds = [item[1]['std'] for item in sorted_features]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=features,
                y=means,
                error_y=dict(type='data', array=stds),
                name='Feature Importance',
                marker_color=['red' if v < 0 else 'blue' for v in means]
            ))
            
            fig.update_layout(
                title=f"Top {len(features)} Feature Importance (SHAP Values)",
                xaxis_title="Features",
                yaxis_title="Importance (SHAP Value)",
                xaxis_tickangle=-45,
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance plot: {e}")
            return go.Figure()
    
    def get_prediction_timeline_plot(self) -> go.Figure:
        """Generate prediction timeline plot."""
        try:
            timeline_data = self.dashboard_data.get('prediction_timeline', [])
            
            if not timeline_data:
                return go.Figure().add_annotation(
                    text="No prediction timeline data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            timestamps = [item['timestamp'] for item in timeline_data]
            predictions = [item['prediction'] for item in timeline_data]
            confidences = [item['confidence'] for item in timeline_data]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Predictions Over Time', 'Explanation Confidence'),
                vertical_spacing=0.1
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Confidence
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Prediction Timeline",
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Prediction Value", row=1, col=1)
            fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error generating prediction timeline plot: {e}")
            return go.Figure()
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        with self._lock:
            return {
                'explanation_stats': self.dashboard_data.get('explanation_stats', {}),
                'total_features_tracked': len(self.feature_importance_history),
                'explanations_generated': len(self.explanations),
                'dashboard_active': self.dashboard_active,
                'explainers_available': {
                    'shap': self.shap_explainer is not None,
                    'lime': self.lime_explainer is not None
                },
                'last_explanation': self.explanations[-1].timestamp.isoformat() if self.explanations else None
            }
    
    def get_recent_explanations(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent explanations in serializable format."""
        with self._lock:
            recent = list(self.explanations)[-count:] if self.explanations else []
            
            serializable = []
            for exp in recent:
                serializable.append({
                    'timestamp': exp.timestamp.isoformat(),
                    'model_id': exp.model_id,
                    'prediction': exp.prediction,
                    'confidence': exp.confidence,
                    'explanation_method': exp.explanation_method,
                    'top_features': self._get_top_features(exp, 5)
                })
            
            return serializable
    
    def _get_top_features(self, explanation: ModelExplanation, count: int = 5) -> List[Dict[str, float]]:
        """Get top contributing features from explanation."""
        features = []
        
        if explanation.shap_values:
            sorted_features = sorted(
                explanation.shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:count]
            
            features = [{'feature': f, 'importance': v} for f, v in sorted_features]
        
        return features
    
    def stop_dashboard(self):
        """Stop dashboard updates."""
        if self.dashboard_active:
            self.dashboard_active = False
            if self.dashboard_thread:
                self.dashboard_thread.join(timeout=5.0)
            self.logger.info("Stopped dashboard updates")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_dashboard()

