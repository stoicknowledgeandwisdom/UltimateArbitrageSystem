#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-Time Security Analytics System
==============================

Provides comprehensive security analytics:
- Real-time monitoring
- Threat detection
- Risk assessment
- Performance analysis
- Security metrics
- Automated responses
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    analysis_interval: int = 60  # seconds
    history_window: int = 24     # hours
    alert_threshold: float = 0.8
    risk_threshold: float = 0.7
    auto_response: bool = True
    ml_enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: [
        'transaction_volume',
        'error_rate',
        'response_time',
        'risk_score',
        'threat_level'
    ])

@dataclass
class SecurityMetrics:
    timestamp: datetime
    transaction_count: int
    error_count: int
    avg_response_time: float
    risk_score: float
    threat_level: float
    anomaly_score: float
    active_threats: int
    blocked_attempts: int
    metrics: Dict[str, float] = field(default_factory=dict)

class SecurityAnalytics:
    """Real-time security analytics implementation"""

    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.metrics_history: List[SecurityMetrics] = []
        self.current_metrics: Optional[SecurityMetrics] = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.alert_callbacks: List[callable] = []
        
        # Initialize analytics system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the analytics system"""
        logger.info("Initializing security analytics system...")
        
        try:
            # Initialize ML models if enabled
            if self.config.ml_enabled:
                self._initialize_ml_models()
            
            # Initialize metrics collection
            self._initialize_metrics()
            
            # Start analysis loop
            asyncio.create_task(self._analysis_loop())
            
            logger.info("Security analytics system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics system: {str(e)}")
            raise
    
    def _initialize_ml_models(self) -> None:
        """Initialize machine learning models"""
        try:
            self.anomaly_detector = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            self.config.ml_enabled = False
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics collection"""
        self.current_metrics = SecurityMetrics(
            timestamp=datetime.now(),
            transaction_count=0,
            error_count=0,
            avg_response_time=0.0,
            risk_score=0.0,
            threat_level=0.0,
            anomaly_score=0.0,
            active_threats=0,
            blocked_attempts=0
        )
    
    async def _analysis_loop(self) -> None:
        """Main analysis loop"""
        while True:
            try:
                # Collect and analyze metrics
                await self._collect_metrics()
                await self._analyze_metrics()
                
                # Store metrics history
                self.metrics_history.append(self.current_metrics)
                
                # Trim history to window size
                window_start = datetime.now() - timedelta(hours=self.config.history_window)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > window_start
                ]
                
                # Wait for next interval
                await asyncio.sleep(self.config.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _collect_metrics(self) -> None:
        """Collect current security metrics"""
        try:
            # Update timestamp
            self.current_metrics.timestamp = datetime.now()
            
            # Collect system metrics
            metrics = await self._get_system_metrics()
            
            # Update current metrics
            self.current_metrics = SecurityMetrics(
                timestamp=datetime.now(),
                transaction_count=metrics['transactions'],
                error_count=metrics['errors'],
                avg_response_time=metrics['response_time'],
                risk_score=metrics['risk_score'],
                threat_level=metrics['threat_level'],
                anomaly_score=0.0,  # Will be updated in analysis
                active_threats=metrics['active_threats'],
                blocked_attempts=metrics['blocked_attempts'],
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    async def _analyze_metrics(self) -> None:
        """Analyze collected metrics"""
        try:
            # Prepare data for analysis
            if len(self.metrics_history) > 0:
                # Extract features for anomaly detection
                features = np.array([
                    [
                        m.transaction_count,
                        m.error_count,
                        m.avg_response_time,
                        m.risk_score,
                        m.threat_level
                    ] for m in self.metrics_history
                ])
                
                if self.config.ml_enabled and len(features) > 10:
                    # Scale features
                    scaled_features = self.scaler.fit_transform(features)
                    
                    # Detect anomalies
                    anomaly_scores = self.anomaly_detector.fit_predict(scaled_features)
                    latest_score = anomaly_scores[-1]
                    
                    # Update anomaly score (-1 for anomaly, 1 for normal)
                    self.current_metrics.anomaly_score = \
                        0.0 if latest_score == 1 else 1.0
                    
                    # Check for alerts
                    if self.current_metrics.anomaly_score > self.config.alert_threshold:
                        await self._handle_anomaly()
            
            # Update risk assessment
            risk_level = await self._assess_risk()
            if risk_level > self.config.risk_threshold:
                await self._handle_high_risk()
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {str(e)}")
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Collect system-wide security metrics"""
        try:
            # These would be replaced with actual metric collection
            # from various system components
            return {
                'transactions': 1000,
                'errors': 5,
                'response_time': 0.1,
                'risk_score': 0.3,
                'threat_level': 0.2,
                'active_threats': 0,
                'blocked_attempts': 10
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return defaultdict(float)
    
    async def _assess_risk(self) -> float:
        """Assess current system risk level"""
        try:
            # Calculate weighted risk score
            weights = {
                'anomaly_score': 0.3,
                'threat_level': 0.3,
                'error_rate': 0.2,
                'response_time': 0.2
            }
            
            metrics = self.current_metrics
            error_rate = metrics.error_count / max(metrics.transaction_count, 1)
            
            risk_score = (
                weights['anomaly_score'] * metrics.anomaly_score +
                weights['threat_level'] * metrics.threat_level +
                weights['error_rate'] * error_rate +
                weights['response_time'] * min(metrics.avg_response_time / 5.0, 1.0)
            )
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return 0.0
    
    async def _handle_anomaly(self) -> None:
        """Handle detected anomalies"""
        try:
            # Log anomaly
            logger.warning(
                f"Security anomaly detected! Score: {self.current_metrics.anomaly_score}"
            )
            
            # Notify subscribers
            await self._notify_subscribers('anomaly', {
                'score': self.current_metrics.anomaly_score,
                'metrics': self.current_metrics.__dict__
            })
            
            # Automatic response if enabled
            if self.config.auto_response:
                await self._automated_response('anomaly')
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {str(e)}")
    
    async def _handle_high_risk(self) -> None:
        """Handle high risk situations"""
        try:
            # Log high risk
            logger.warning(
                f"High risk level detected! Score: {self.current_metrics.risk_score}"
            )
            
            # Notify subscribers
            await self._notify_subscribers('high_risk', {
                'risk_score': self.current_metrics.risk_score,
                'metrics': self.current_metrics.__dict__
            })
            
            # Automatic response if enabled
            if self.config.auto_response:
                await self._automated_response('high_risk')
            
        except Exception as e:
            logger.error(f"Error handling high risk: {str(e)}")
    
    async def _automated_response(self, trigger: str) -> None:
        """Execute automated response actions"""
        try:
            if trigger == 'anomaly':
                # Implement anomaly response actions
                pass
            elif trigger == 'high_risk':
                # Implement high risk response actions
                pass
            
        except Exception as e:
            logger.error(f"Error in automated response: {str(e)}")
    
    async def _notify_subscribers(self, event: str, data: Dict[str, Any]) -> None:
        """Notify subscribers of security events"""
        try:
            for callback in self.alert_callbacks:
                try:
                    await callback(event, data)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error notifying subscribers: {str(e)}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        return self.current_metrics
    
    def get_metrics_history(self) -> List[SecurityMetrics]:
        """Get historical security metrics"""
        return self.metrics_history
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of current security status"""
        try:
            metrics = self.current_metrics
            history = self.metrics_history[-100:]  # Last 100 metrics
            
            return {
                'current_status': {
                    'risk_level': metrics.risk_score,
                    'threat_level': metrics.threat_level,
                    'anomaly_score': metrics.anomaly_score,
                    'active_threats': metrics.active_threats
                },
                'metrics': {
                    'transactions': metrics.transaction_count,
                    'errors': metrics.error_count,
                    'response_time': metrics.avg_response_time
                },
                'trends': {
                    'risk_trend': self._calculate_trend([m.risk_score for m in history]),
                    'threat_trend': self._calculate_trend([m.threat_level for m in history]),
                    'performance_trend': self._calculate_trend([m.avg_response_time for m in history])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting security summary: {str(e)}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from historical values"""
        if len(values) < 2:
            return 'stable'
        
        avg_change = np.mean(np.diff(values))
        if abs(avg_change) < 0.01:
            return 'stable'
        return 'increasing' if avg_change > 0 else 'decreasing'

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = AnalyticsConfig(
        analysis_interval=60,
        history_window=24,
        auto_response=True,
        ml_enabled=True
    )
    
    async def test_analytics():
        # Initialize analytics
        analytics = SecurityAnalytics(config)
        
        # Add test alert callback
        async def alert_callback(event: str, data: Dict[str, Any]):
            print(f"\nAlert: {event}")
            print(json.dumps(data, indent=2))
        
        analytics.add_alert_callback(alert_callback)
        
        # Wait for some analysis cycles
        print("\nMonitoring security metrics...")
        await asyncio.sleep(180)
        
        # Get security summary
        summary = await analytics.get_security_summary()
        print("\nSecurity Summary:")
        print(json.dumps(summary, indent=2))
    
    # Run test
    asyncio.run(test_analytics())

