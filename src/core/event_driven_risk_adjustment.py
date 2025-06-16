#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-Driven Risk Adjustment Module
=================================

Real-time risk adjustment system that dynamically responds to world events,
market shocks, and environmental changes to protect capital and maximize
profit opportunities.

This module continuously monitors global events and automatically adjusts
risk parameters to maintain optimal performance under any conditions.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from textblob import TextBlob
import yfinance as yf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import tensorflow as tf
from transformers import pipeline
import torch
import re
from collections import defaultdict, deque
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskEvent:
    """Risk event classification"""
    event_id: str
    event_type: str  # geopolitical, economic, market, natural, cyber, health
    severity: float  # 0.0 to 1.0
    confidence: float
    impact_prediction: float
    affected_assets: List[str]
    time_horizon: str  # immediate, short, medium, long
    event_source: str
    timestamp: datetime
    description: str
    keywords: List[str]

@dataclass
class RiskAdjustment:
    """Risk adjustment parameters"""
    adjustment_id: str
    triggered_by: str  # event_id that triggered this adjustment
    position_size_multiplier: float
    stop_loss_adjustment: float
    volatility_threshold_adjustment: float
    correlation_limit_adjustment: float
    leverage_reduction: float
    emergency_exit_threshold: float
    duration_hours: int
    confidence: float
    timestamp: datetime

@dataclass
class MarketStressIndicator:
    """Market stress level indicator"""
    stress_level: float  # 0.0 to 1.0
    volatility_stress: float
    liquidity_stress: float
    correlation_stress: float
    sentiment_stress: float
    macro_stress: float
    overall_confidence: float
    timestamp: datetime

class EventDrivenRiskAdjustment:
    """
    Real-time event-driven risk adjustment system that automatically
    adapts to changing market conditions and world events.
    """
    
    def __init__(self, config_file: str = "config/risk_adjustment_config.json"):
        self.config = self._load_config(config_file)
        self.active_events = {}
        self.active_adjustments = {}
        self.risk_history = deque(maxlen=1000)
        self.stress_indicators = deque(maxlen=100)
        
        # Event monitoring
        self.event_sources = {
            'news_apis': [
                'https://newsapi.org/v2/everything',
                'https://api.marketaux.com/v1/news',
                'https://feeds.finance.yahoo.com/rss/2.0/headline'
            ],
            'social_media': [
                'https://api.reddit.com/r/wallstreetbets',
                'https://api.twitter.com/2/tweets/search/recent'
            ],
            'economic_data': [
                'https://api.stlouisfed.org/fred/series/observations',
                'https://api.census.gov/data',
                'https://api.bls.gov/publicAPI/v2/timeseries/data'
            ],
            'market_data': [
                'yahoo_finance',
                'alpha_vantage',
                'quandl'
            ]
        }
        
        # AI Models
        self.event_classifier = None
        self.impact_predictor = None
        self.sentiment_analyzer = None
        self.stress_detector = None
        
        # Risk thresholds
        self.base_risk_limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_total_exposure': 0.8,  # 80% of portfolio
            'max_correlation': 0.7,
            'min_liquidity_score': 0.3,
            'max_volatility': 0.4,
            'max_drawdown': 0.1
        }
        
        # Event weights for different types
        self.event_weights = {
            'geopolitical': 0.8,
            'economic': 0.9,
            'market': 1.0,
            'natural': 0.6,
            'cyber': 0.7,
            'health': 0.75
        }
        
        self.initialize_ai_models()
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load risk adjustment configuration"""
        default_config = {
            "monitoring_settings": {
                "scan_interval_seconds": 30,
                "event_retention_hours": 72,
                "adjustment_retention_hours": 24,
                "stress_threshold_high": 0.7,
                "stress_threshold_critical": 0.9,
                "auto_adjustment_enabled": True
            },
            "event_detection": {
                "sentiment_threshold": 0.6,
                "volatility_threshold": 0.3,
                "volume_spike_threshold": 2.0,
                "price_movement_threshold": 0.05,
                "correlation_break_threshold": 0.4
            },
            "risk_adjustments": {
                "conservative_multiplier": 0.5,
                "moderate_multiplier": 0.7,
                "aggressive_multiplier": 0.3,
                "emergency_multiplier": 0.1,
                "recovery_rate": 0.1  # Rate at which adjustments decay
            },
            "asset_monitoring": {
                "equity_indices": ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
                "currencies": ['DXY', 'EUR/USD', 'GBP/USD', 'USD/JPY'],
                "commodities": ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
                "bonds": ['TLT', 'IEF', 'SHY', 'HYG', 'EMB'],
                "crypto": ['BTC-USD', 'ETH-USD', 'BNB-USD'],
                "volatility": ['VIX', 'VXST', 'VXN']
            }
        }
        
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                logger.error(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def initialize_ai_models(self):
        """Initialize AI models for event detection and risk assessment"""
        try:
            logger.info("🤖 Initializing AI models for event-driven risk adjustment...")
            
            # Sentiment analyzer for news and social media
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Event classifier
            self.event_classifier = self._build_event_classifier()
            
            # Impact predictor
            self.impact_predictor = self._build_impact_predictor()
            
            # Stress detector
            self.stress_detector = self._build_stress_detector()
            
            logger.info("✅ AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing AI models: {e}")
            self._initialize_fallback_models()
    
    def _build_event_classifier(self) -> tf.keras.Model:
        """Build neural network for event classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')  # 6 event types
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_impact_predictor(self) -> tf.keras.Model:
        """Build neural network for impact prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Impact score 0-1
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_stress_detector(self) -> RandomForestClassifier:
        """Build ensemble model for market stress detection"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
    
    def _initialize_fallback_models(self):
        """Initialize fallback models if advanced models fail"""
        logger.info("🔄 Initializing fallback models...")
        
        # Simple sentiment analyzer
        self.sentiment_analyzer = lambda text: {
            'label': 'POSITIVE' if TextBlob(text).sentiment.polarity > 0 else 'NEGATIVE',
            'score': abs(TextBlob(text).sentiment.polarity)
        }
        
        # Simple classifiers
        self.event_classifier = RandomForestClassifier(n_estimators=50)
        self.impact_predictor = RandomForestClassifier(n_estimators=50)
        self.stress_detector = IsolationForest(contamination=0.1)
    
    async def start_monitoring(self):
        """Start continuous event monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("🔍 Starting event-driven risk monitoring...")
        
        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start async tasks
        await asyncio.gather(
            self._continuous_event_detection(),
            self._continuous_stress_monitoring(),
            self._continuous_risk_adjustment()
        )
    
    def stop_monitoring(self):
        """Stop event monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Stopping event monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        scan_interval = self.config['monitoring_settings']['scan_interval_seconds']
        
        while self.monitoring_active:
            try:
                # This would be called from async context in real implementation
                pass
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(scan_interval)
    
    async def _continuous_event_detection(self):
        """Continuously detect and classify events"""
        scan_interval = self.config['monitoring_settings']['scan_interval_seconds']
        
        while self.monitoring_active:
            try:
                # Collect event data
                event_data = await self._collect_event_data()
                
                # Detect new events
                new_events = await self._detect_events(event_data)
                
                # Process and classify events
                for event in new_events:
                    classified_event = await self._classify_event(event)
                    if classified_event:
                        await self._process_new_event(classified_event)
                
                # Clean up old events
                await self._cleanup_old_events()
                
            except Exception as e:
                logger.error(f"Error in event detection: {e}")
            
            await asyncio.sleep(scan_interval)
    
    async def _continuous_stress_monitoring(self):
        """Continuously monitor market stress levels"""
        scan_interval = self.config['monitoring_settings']['scan_interval_seconds'] * 2
        
        while self.monitoring_active:
            try:
                # Calculate current stress levels
                stress_indicator = await self._calculate_market_stress()
                
                # Add to history
                self.stress_indicators.append(stress_indicator)
                
                # Check for stress threshold breaches
                await self._check_stress_thresholds(stress_indicator)
                
            except Exception as e:
                logger.error(f"Error in stress monitoring: {e}")
            
            await asyncio.sleep(scan_interval)
    
    async def _continuous_risk_adjustment(self):
        """Continuously adjust risk parameters based on events and stress"""
        adjustment_interval = self.config['monitoring_settings']['scan_interval_seconds'] * 3
        
        while self.monitoring_active:
            try:
                # Calculate required risk adjustments
                adjustments = await self._calculate_risk_adjustments()
                
                # Apply adjustments
                for adjustment in adjustments:
                    await self._apply_risk_adjustment(adjustment)
                
                # Decay existing adjustments
                await self._decay_adjustments()
                
                # Clean up expired adjustments
                await self._cleanup_expired_adjustments()
                
            except Exception as e:
                logger.error(f"Error in risk adjustment: {e}")
            
            await asyncio.sleep(adjustment_interval)
    
    async def _collect_event_data(self) -> Dict[str, Any]:
        """Collect event data from multiple sources"""
        event_data = {
            'news': [],
            'social_media': [],
            'economic_releases': [],
            'market_data': {},
            'timestamp': datetime.now()
        }
        
        try:
            # Collect news data (mock implementation)
            event_data['news'] = await self._collect_news_data()
            
            # Collect social media sentiment (mock implementation)
            event_data['social_media'] = await self._collect_social_data()
            
            # Collect economic data (mock implementation)
            event_data['economic_releases'] = await self._collect_economic_data()
            
            # Collect market data
            event_data['market_data'] = await self._collect_market_data()
            
        except Exception as e:
            logger.error(f"Error collecting event data: {e}")
        
        return event_data
    
    async def _collect_news_data(self) -> List[Dict[str, Any]]:
        """Collect news data (mock implementation)"""
        # Mock news data - in production, integrate with real news APIs
        mock_news = [
            {
                'title': 'Federal Reserve signals potential rate changes',
                'content': 'The Federal Reserve indicated possible adjustments to interest rates in response to economic conditions.',
                'source': 'Financial Times',
                'timestamp': datetime.now(),
                'url': 'https://example.com/news1',
                'keywords': ['federal reserve', 'interest rates', 'monetary policy']
            },
            {
                'title': 'Geopolitical tensions escalate in Eastern Europe',
                'content': 'Rising tensions between nations could impact global markets and energy supplies.',
                'source': 'Reuters',
                'timestamp': datetime.now(),
                'url': 'https://example.com/news2',
                'keywords': ['geopolitical', 'tensions', 'energy', 'markets']
            },
            {
                'title': 'Technology sector shows strong earnings growth',
                'content': 'Major technology companies report better than expected quarterly earnings.',
                'source': 'Bloomberg',
                'timestamp': datetime.now(),
                'url': 'https://example.com/news3',
                'keywords': ['technology', 'earnings', 'growth', 'quarterly']
            }
        ]
        
        return mock_news
    
    async def _collect_social_data(self) -> List[Dict[str, Any]]:
        """Collect social media sentiment data (mock implementation)"""
        # Mock social media data
        mock_social = [
            {
                'platform': 'reddit',
                'subreddit': 'wallstreetbets',
                'sentiment': 'bullish',
                'confidence': 0.7,
                'mentions': ['SPY', 'TSLA', 'AAPL'],
                'timestamp': datetime.now()
            },
            {
                'platform': 'twitter',
                'hashtags': ['marketcrash', 'volatility'],
                'sentiment': 'bearish',
                'confidence': 0.8,
                'timestamp': datetime.now()
            }
        ]
        
        return mock_social
    
    async def _collect_economic_data(self) -> List[Dict[str, Any]]:
        """Collect economic release data (mock implementation)"""
        # Mock economic data
        mock_economic = [
            {
                'indicator': 'GDP',
                'value': 2.1,
                'previous': 2.0,
                'forecast': 2.0,
                'importance': 'high',
                'timestamp': datetime.now()
            },
            {
                'indicator': 'CPI',
                'value': 3.2,
                'previous': 3.1,
                'forecast': 3.0,
                'importance': 'high',
                'timestamp': datetime.now()
            }
        ]
        
        return mock_economic
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect real-time market data"""
        market_data = {}
        
        # Collect data for all monitored assets
        all_assets = []
        for asset_class, symbols in self.config['asset_monitoring'].items():
            all_assets.extend(symbols)
        
        for symbol in all_assets:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d', interval='5m')
                
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    market_data[symbol] = {
                        'price': hist['Close'].iloc[-1],
                        'change': returns.iloc[-1] if len(returns) > 0 else 0,
                        'volatility': returns.std() * np.sqrt(252),
                        'volume': hist['Volume'].iloc[-1],
                        'volume_ratio': hist['Volume'].iloc[-1] / hist['Volume'].mean() if len(hist) > 1 else 1
                    }
            except Exception as e:
                logger.warning(f"Failed to collect data for {symbol}: {e}")
        
        return market_data
    
    async def _detect_events(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect significant events from collected data"""
        events = []
        
        # Detect news-based events
        for news_item in event_data['news']:
            event = await self._analyze_news_item(news_item)
            if event:
                events.append(event)
        
        # Detect market-based events
        market_events = await self._detect_market_events(event_data['market_data'])
        events.extend(market_events)
        
        # Detect social sentiment events
        sentiment_events = await self._detect_sentiment_events(event_data['social_media'])
        events.extend(sentiment_events)
        
        # Detect economic events
        economic_events = await self._detect_economic_events(event_data['economic_releases'])
        events.extend(economic_events)
        
        return events
    
    async def _analyze_news_item(self, news_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze news item for potential risk events"""
        try:
            # Analyze sentiment
            if callable(self.sentiment_analyzer):
                sentiment = self.sentiment_analyzer(news_item['content'])
                sentiment_score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
            else:
                result = self.sentiment_analyzer(news_item['content'])
                sentiment_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
            
            # Check if sentiment is extreme enough to be considered an event
            if abs(sentiment_score) < self.config['event_detection']['sentiment_threshold']:
                return None
            
            # Extract keywords and classify event type
            event_type = self._classify_event_type(news_item['keywords'])
            
            # Determine affected assets
            affected_assets = self._determine_affected_assets(news_item['keywords'], event_type)
            
            event = {
                'event_id': f"news_{hash(news_item['title'])}_{int(time.time())}",
                'event_type': event_type,
                'severity': abs(sentiment_score),
                'confidence': 0.7,  # Base confidence for news events
                'source': 'news',
                'description': news_item['title'],
                'content': news_item['content'],
                'keywords': news_item['keywords'],
                'affected_assets': affected_assets,
                'timestamp': news_item['timestamp'],
                'sentiment_score': sentiment_score
            }
            
            return event
            
        except Exception as e:
            logger.error(f"Error analyzing news item: {e}")
            return None
    
    def _classify_event_type(self, keywords: List[str]) -> str:
        """Classify event type based on keywords"""
        keyword_classifications = {
            'geopolitical': ['war', 'conflict', 'sanctions', 'diplomatic', 'military', 'terrorism'],
            'economic': ['gdp', 'inflation', 'unemployment', 'fed', 'central bank', 'interest rates'],
            'market': ['earnings', 'ipo', 'merger', 'acquisition', 'bankruptcy', 'delisting'],
            'natural': ['hurricane', 'earthquake', 'flood', 'drought', 'wildfire', 'pandemic'],
            'cyber': ['cyberattack', 'data breach', 'hacking', 'ransomware', 'security'],
            'health': ['pandemic', 'outbreak', 'vaccine', 'health crisis', 'disease']
        }
        
        scores = defaultdict(int)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for event_type, type_keywords in keyword_classifications.items():
                for type_keyword in type_keywords:
                    if type_keyword in keyword_lower:
                        scores[event_type] += 1
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'market'  # Default classification
    
    def _determine_affected_assets(self, keywords: List[str], event_type: str) -> List[str]:
        """Determine which assets might be affected by the event"""
        affected_assets = []
        
        # Asset mapping based on keywords
        asset_keywords = {
            'SPY': ['s&p', 'spy', 'us market', 'american stocks'],
            'QQQ': ['nasdaq', 'qqq', 'technology', 'tech stocks'],
            'VIX': ['volatility', 'vix', 'fear index'],
            'GLD': ['gold', 'precious metals'],
            'USO': ['oil', 'crude', 'energy'],
            'TLT': ['bonds', 'treasury', 'rates'],
            'DXY': ['dollar', 'currency', 'usd'],
            'BTC-USD': ['bitcoin', 'cryptocurrency', 'crypto']
        }
        
        # Check keywords against asset mappings
        for asset, asset_keys in asset_keywords.items():
            for keyword in keywords:
                for asset_key in asset_keys:
                    if asset_key.lower() in keyword.lower():
                        affected_assets.append(asset)
                        break
        
        # Add default assets based on event type
        if event_type == 'geopolitical':
            affected_assets.extend(['VIX', 'GLD', 'DXY'])
        elif event_type == 'economic':
            affected_assets.extend(['SPY', 'TLT', 'DXY'])
        elif event_type == 'market':
            affected_assets.extend(['SPY', 'QQQ', 'VIX'])
        
        return list(set(affected_assets))  # Remove duplicates
    
    async def _detect_market_events(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect significant market events"""
        events = []
        
        volatility_threshold = self.config['event_detection']['volatility_threshold']
        volume_threshold = self.config['event_detection']['volume_spike_threshold']
        price_threshold = self.config['event_detection']['price_movement_threshold']
        
        for symbol, data in market_data.items():
            # Detect high volatility
            if data['volatility'] > volatility_threshold:
                events.append({
                    'event_id': f"volatility_{symbol}_{int(time.time())}",
                    'event_type': 'market',
                    'severity': min(1.0, data['volatility'] / volatility_threshold),
                    'confidence': 0.8,
                    'source': 'market_data',
                    'description': f"High volatility detected in {symbol}",
                    'affected_assets': [symbol],
                    'timestamp': datetime.now(),
                    'volatility': data['volatility']
                })
            
            # Detect volume spikes
            if data['volume_ratio'] > volume_threshold:
                events.append({
                    'event_id': f"volume_{symbol}_{int(time.time())}",
                    'event_type': 'market',
                    'severity': min(1.0, data['volume_ratio'] / volume_threshold / 2),
                    'confidence': 0.7,
                    'source': 'market_data',
                    'description': f"Volume spike detected in {symbol}",
                    'affected_assets': [symbol],
                    'timestamp': datetime.now(),
                    'volume_ratio': data['volume_ratio']
                })
            
            # Detect large price movements
            if abs(data['change']) > price_threshold:
                events.append({
                    'event_id': f"price_{symbol}_{int(time.time())}",
                    'event_type': 'market',
                    'severity': min(1.0, abs(data['change']) / price_threshold),
                    'confidence': 0.9,
                    'source': 'market_data',
                    'description': f"Large price movement in {symbol}: {data['change']:.2%}",
                    'affected_assets': [symbol],
                    'timestamp': datetime.now(),
                    'price_change': data['change']
                })
        
        return events
    
    async def _detect_sentiment_events(self, social_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect sentiment-based events from social media"""
        events = []
        
        for item in social_data:
            confidence = item.get('confidence', 0.5)
            
            # Only consider high-confidence sentiment
            if confidence > 0.7:
                sentiment_severity = confidence
                
                if item['sentiment'] in ['very_bearish', 'very_bullish']:
                    events.append({
                        'event_id': f"sentiment_{item['platform']}_{int(time.time())}",
                        'event_type': 'market',
                        'severity': sentiment_severity,
                        'confidence': confidence,
                        'source': 'social_media',
                        'description': f"{item['sentiment']} sentiment on {item['platform']}",
                        'affected_assets': item.get('mentions', ['SPY']),
                        'timestamp': item['timestamp'],
                        'sentiment': item['sentiment'],
                        'platform': item['platform']
                    })
        
        return events
    
    async def _detect_economic_events(self, economic_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant economic events"""
        events = []
        
        for release in economic_data:
            # Calculate surprise factor
            if 'forecast' in release and 'value' in release:
                surprise = abs(release['value'] - release['forecast']) / abs(release['forecast']) if release['forecast'] != 0 else 0
                
                # Only consider significant surprises
                if surprise > 0.1:  # 10% surprise threshold
                    events.append({
                        'event_id': f"economic_{release['indicator']}_{int(time.time())}",
                        'event_type': 'economic',
                        'severity': min(1.0, surprise),
                        'confidence': 0.9 if release['importance'] == 'high' else 0.7,
                        'source': 'economic_data',
                        'description': f"{release['indicator']} surprise: {release['value']} vs {release['forecast']} forecast",
                        'affected_assets': self._get_economic_affected_assets(release['indicator']),
                        'timestamp': release['timestamp'],
                        'indicator': release['indicator'],
                        'value': release['value'],
                        'forecast': release['forecast'],
                        'surprise': surprise
                    })
        
        return events
    
    def _get_economic_affected_assets(self, indicator: str) -> List[str]:
        """Get assets affected by economic indicator"""
        indicator_mappings = {
            'GDP': ['SPY', 'QQQ', 'DXY'],
            'CPI': ['TLT', 'GLD', 'DXY'],
            'Unemployment': ['SPY', 'QQQ'],
            'Fed Rate': ['TLT', 'DXY', 'SPY'],
            'Trade Balance': ['DXY'],
            'Retail Sales': ['SPY', 'QQQ']
        }
        
        return indicator_mappings.get(indicator, ['SPY'])
    
    async def _classify_event(self, event: Dict[str, Any]) -> Optional[RiskEvent]:
        """Classify and validate detected event"""
        try:
            # Predict impact using AI model
            impact_features = self._extract_impact_features(event)
            
            if hasattr(self.impact_predictor, 'predict'):
                impact_prediction = self.impact_predictor.predict([impact_features])[0]
            else:
                # Fallback impact calculation
                impact_prediction = event['severity'] * self.event_weights.get(event['event_type'], 0.5)
            
            # Determine time horizon
            time_horizon = self._determine_time_horizon(event)
            
            # Create RiskEvent object
            risk_event = RiskEvent(
                event_id=event['event_id'],
                event_type=event['event_type'],
                severity=event['severity'],
                confidence=event['confidence'],
                impact_prediction=float(impact_prediction),
                affected_assets=event['affected_assets'],
                time_horizon=time_horizon,
                event_source=event['source'],
                timestamp=event['timestamp'],
                description=event['description'],
                keywords=event.get('keywords', [])
            )
            
            return risk_event
            
        except Exception as e:
            logger.error(f"Error classifying event: {e}")
            return None
    
    def _extract_impact_features(self, event: Dict[str, Any]) -> List[float]:
        """Extract features for impact prediction"""
        features = [
            event['severity'],
            event['confidence'],
            len(event['affected_assets']),
            self.event_weights.get(event['event_type'], 0.5)
        ]
        
        # Add event type one-hot encoding
        event_types = ['geopolitical', 'economic', 'market', 'natural', 'cyber', 'health']
        for et in event_types:
            features.append(1.0 if event['event_type'] == et else 0.0)
        
        # Pad to fixed size
        target_size = 50
        while len(features) < target_size:
            features.append(0.0)
        
        return features[:target_size]
    
    def _determine_time_horizon(self, event: Dict[str, Any]) -> str:
        """Determine time horizon for event impact"""
        if event['event_type'] == 'market':
            if 'volatility' in event or 'volume' in event:
                return 'immediate'
            else:
                return 'short'
        elif event['event_type'] == 'economic':
            return 'medium'
        elif event['event_type'] == 'geopolitical':
            return 'long'
        else:
            return 'short'
    
    async def _process_new_event(self, risk_event: RiskEvent):
        """Process newly detected risk event"""
        try:
            # Add to active events
            self.active_events[risk_event.event_id] = risk_event
            
            # Log event
            logger.info(f"🚨 New risk event detected: {risk_event.event_type} - {risk_event.description} (severity: {risk_event.severity:.2f})")
            
            # Trigger immediate risk adjustment if severe
            if risk_event.severity > 0.8 or risk_event.impact_prediction > 0.7:
                await self._trigger_emergency_adjustment(risk_event)
            
            # Save event to history
            self.risk_history.append(asdict(risk_event))
            
        except Exception as e:
            logger.error(f"Error processing new event: {e}")
    
    async def _trigger_emergency_adjustment(self, risk_event: RiskEvent):
        """Trigger emergency risk adjustment for severe events"""
        try:
            # Calculate emergency adjustment parameters
            severity_multiplier = min(1.0, risk_event.severity * risk_event.impact_prediction)
            
            adjustment = RiskAdjustment(
                adjustment_id=f"emergency_{risk_event.event_id}",
                triggered_by=risk_event.event_id,
                position_size_multiplier=max(0.1, 1.0 - severity_multiplier * 0.8),
                stop_loss_adjustment=0.7,  # Tighter stops
                volatility_threshold_adjustment=0.5,  # Lower vol tolerance
                correlation_limit_adjustment=0.5,  # Lower correlation tolerance
                leverage_reduction=0.3,  # Reduce leverage
                emergency_exit_threshold=0.03,  # 3% emergency exit
                duration_hours=24,  # Emergency adjustments last 24 hours
                confidence=risk_event.confidence,
                timestamp=datetime.now()
            )
            
            await self._apply_risk_adjustment(adjustment)
            
            logger.warning(f"⚡ Emergency risk adjustment triggered for {risk_event.event_type} event")
            
        except Exception as e:
            logger.error(f"Error triggering emergency adjustment: {e}")
    
    async def _calculate_market_stress(self) -> MarketStressIndicator:
        """Calculate current market stress levels"""
        try:
            # Collect current market data
            market_data = await self._collect_market_data()
            
            # Calculate stress components
            volatility_stress = self._calculate_volatility_stress(market_data)
            liquidity_stress = self._calculate_liquidity_stress(market_data)
            correlation_stress = self._calculate_correlation_stress(market_data)
            sentiment_stress = self._calculate_sentiment_stress()
            macro_stress = self._calculate_macro_stress()
            
            # Calculate overall stress
            stress_components = [volatility_stress, liquidity_stress, correlation_stress, sentiment_stress, macro_stress]
            overall_stress = np.mean(stress_components)
            confidence = 1.0 - np.std(stress_components) / np.mean(stress_components) if np.mean(stress_components) > 0 else 0.5
            
            stress_indicator = MarketStressIndicator(
                stress_level=overall_stress,
                volatility_stress=volatility_stress,
                liquidity_stress=liquidity_stress,
                correlation_stress=correlation_stress,
                sentiment_stress=sentiment_stress,
                macro_stress=macro_stress,
                overall_confidence=confidence,
                timestamp=datetime.now()
            )
            
            return stress_indicator
            
        except Exception as e:
            logger.error(f"Error calculating market stress: {e}")
            # Return neutral stress indicator
            return MarketStressIndicator(
                stress_level=0.5, volatility_stress=0.5, liquidity_stress=0.5,
                correlation_stress=0.5, sentiment_stress=0.5, macro_stress=0.5,
                overall_confidence=0.3, timestamp=datetime.now()
            )
    
    def _calculate_volatility_stress(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based stress"""
        volatilities = [data.get('volatility', 0.2) for data in market_data.values()]
        if not volatilities:
            return 0.5
        
        avg_vol = np.mean(volatilities)
        normal_vol = 0.2  # Normal volatility baseline
        stress = min(1.0, avg_vol / normal_vol)
        return stress
    
    def _calculate_liquidity_stress(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity-based stress"""
        volume_ratios = [data.get('volume_ratio', 1.0) for data in market_data.values()]
        if not volume_ratios:
            return 0.5
        
        avg_volume_ratio = np.mean(volume_ratios)
        # Lower volume indicates higher stress
        stress = max(0.0, min(1.0, 2.0 - avg_volume_ratio))
        return stress
    
    def _calculate_correlation_stress(self, market_data: Dict[str, Any]) -> float:
        """Calculate correlation-based stress"""
        # Simple correlation stress based on price movements
        price_changes = [data.get('change', 0) for data in market_data.values() if 'change' in data]
        
        if len(price_changes) < 2:
            return 0.5
        
        # High correlation (all moving together) indicates stress
        correlation_matrix = np.corrcoef(price_changes)
        if correlation_matrix.size > 1:
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            stress = min(1.0, avg_correlation * 1.5)
        else:
            stress = 0.5
        
        return stress
    
    def _calculate_sentiment_stress(self) -> float:
        """Calculate sentiment-based stress"""
        # Count recent negative events
        recent_events = [event for event in self.active_events.values() 
                        if (datetime.now() - event.timestamp).total_seconds() < 3600]  # Last hour
        
        if not recent_events:
            return 0.3
        
        negative_events = sum(1 for event in recent_events if event.severity > 0.6)
        stress = min(1.0, negative_events / 5.0)  # Normalize by expected max events
        return stress
    
    def _calculate_macro_stress(self) -> float:
        """Calculate macro-economic stress"""
        # Simple macro stress calculation
        # In production, this would analyze economic indicators
        return 0.4  # Placeholder
    
    async def _check_stress_thresholds(self, stress_indicator: MarketStressIndicator):
        """Check if stress levels breach thresholds"""
        high_threshold = self.config['monitoring_settings']['stress_threshold_high']
        critical_threshold = self.config['monitoring_settings']['stress_threshold_critical']
        
        if stress_indicator.stress_level > critical_threshold:
            logger.critical(f"🔴 CRITICAL market stress detected: {stress_indicator.stress_level:.2f}")
            await self._trigger_critical_stress_response(stress_indicator)
        elif stress_indicator.stress_level > high_threshold:
            logger.warning(f"🟡 HIGH market stress detected: {stress_indicator.stress_level:.2f}")
            await self._trigger_high_stress_response(stress_indicator)
    
    async def _trigger_critical_stress_response(self, stress_indicator: MarketStressIndicator):
        """Trigger response to critical stress levels"""
        # Create emergency risk adjustment
        adjustment = RiskAdjustment(
            adjustment_id=f"critical_stress_{int(time.time())}",
            triggered_by="critical_market_stress",
            position_size_multiplier=0.2,  # Reduce positions to 20%
            stop_loss_adjustment=0.5,
            volatility_threshold_adjustment=0.3,
            correlation_limit_adjustment=0.4,
            leverage_reduction=0.1,  # Minimal leverage
            emergency_exit_threshold=0.02,  # 2% emergency exit
            duration_hours=12,
            confidence=stress_indicator.overall_confidence,
            timestamp=datetime.now()
        )
        
        await self._apply_risk_adjustment(adjustment)
    
    async def _trigger_high_stress_response(self, stress_indicator: MarketStressIndicator):
        """Trigger response to high stress levels"""
        adjustment = RiskAdjustment(
            adjustment_id=f"high_stress_{int(time.time())}",
            triggered_by="high_market_stress",
            position_size_multiplier=0.6,  # Reduce positions to 60%
            stop_loss_adjustment=0.8,
            volatility_threshold_adjustment=0.7,
            correlation_limit_adjustment=0.6,
            leverage_reduction=0.5,
            emergency_exit_threshold=0.03,
            duration_hours=6,
            confidence=stress_indicator.overall_confidence,
            timestamp=datetime.now()
        )
        
        await self._apply_risk_adjustment(adjustment)
    
    async def _calculate_risk_adjustments(self) -> List[RiskAdjustment]:
        """Calculate required risk adjustments based on active events and stress"""
        adjustments = []
        
        # Calculate adjustments for each active event
        for event in self.active_events.values():
            if event.event_id not in [adj.triggered_by for adj in self.active_adjustments.values()]:
                adjustment = self._calculate_event_adjustment(event)
                if adjustment:
                    adjustments.append(adjustment)
        
        return adjustments
    
    def _calculate_event_adjustment(self, event: RiskEvent) -> Optional[RiskAdjustment]:
        """Calculate risk adjustment for specific event"""
        try:
            # Base adjustment on event severity and impact
            impact_factor = event.severity * event.impact_prediction
            
            if impact_factor < 0.3:
                return None  # No adjustment needed for low impact events
            
            # Calculate adjustment parameters
            position_multiplier = max(0.3, 1.0 - impact_factor * 0.6)
            stop_loss_adj = max(0.5, 1.0 - impact_factor * 0.4)
            vol_threshold_adj = max(0.4, 1.0 - impact_factor * 0.5)
            correlation_limit_adj = max(0.5, 1.0 - impact_factor * 0.3)
            leverage_reduction = min(0.9, impact_factor)
            emergency_threshold = max(0.02, 0.05 - impact_factor * 0.02)
            
            # Duration based on time horizon
            duration_map = {'immediate': 2, 'short': 6, 'medium': 24, 'long': 72}
            duration = duration_map.get(event.time_horizon, 6)
            
            adjustment = RiskAdjustment(
                adjustment_id=f"event_{event.event_id}",
                triggered_by=event.event_id,
                position_size_multiplier=position_multiplier,
                stop_loss_adjustment=stop_loss_adj,
                volatility_threshold_adjustment=vol_threshold_adj,
                correlation_limit_adjustment=correlation_limit_adj,
                leverage_reduction=leverage_reduction,
                emergency_exit_threshold=emergency_threshold,
                duration_hours=duration,
                confidence=event.confidence,
                timestamp=datetime.now()
            )
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating event adjustment: {e}")
            return None
    
    async def _apply_risk_adjustment(self, adjustment: RiskAdjustment):
        """Apply risk adjustment to the system"""
        try:
            # Add to active adjustments
            self.active_adjustments[adjustment.adjustment_id] = adjustment
            
            # Log adjustment
            logger.info(f"🛡️ Applied risk adjustment: {adjustment.adjustment_id} (position size: {adjustment.position_size_multiplier:.2f}x)")
            
            # In a real system, this would update the trading system's risk parameters
            # For now, we'll just store the adjustment
            
        except Exception as e:
            logger.error(f"Error applying risk adjustment: {e}")
    
    async def _decay_adjustments(self):
        """Gradually decay risk adjustments over time"""
        decay_rate = self.config['risk_adjustments']['recovery_rate']
        
        for adjustment in list(self.active_adjustments.values()):
            # Calculate age in hours
            age_hours = (datetime.now() - adjustment.timestamp).total_seconds() / 3600
            
            if age_hours > 1:  # Start decaying after 1 hour
                # Apply exponential decay
                decay_factor = np.exp(-decay_rate * (age_hours - 1))
                
                # Move adjustment parameters back toward normal (1.0)
                adjustment.position_size_multiplier = 1.0 - (1.0 - adjustment.position_size_multiplier) * decay_factor
                adjustment.stop_loss_adjustment = 1.0 - (1.0 - adjustment.stop_loss_adjustment) * decay_factor
                adjustment.volatility_threshold_adjustment = 1.0 - (1.0 - adjustment.volatility_threshold_adjustment) * decay_factor
                adjustment.correlation_limit_adjustment = 1.0 - (1.0 - adjustment.correlation_limit_adjustment) * decay_factor
                adjustment.leverage_reduction = adjustment.leverage_reduction * decay_factor
    
    async def _cleanup_expired_adjustments(self):
        """Remove expired risk adjustments"""
        expired_adjustments = []
        
        for adj_id, adjustment in self.active_adjustments.items():
            age_hours = (datetime.now() - adjustment.timestamp).total_seconds() / 3600
            
            if age_hours > adjustment.duration_hours:
                expired_adjustments.append(adj_id)
        
        for adj_id in expired_adjustments:
            del self.active_adjustments[adj_id]
            logger.info(f"🔄 Expired risk adjustment removed: {adj_id}")
    
    async def _cleanup_old_events(self):
        """Remove old events from active tracking"""
        retention_hours = self.config['monitoring_settings']['event_retention_hours']
        expired_events = []
        
        for event_id, event in self.active_events.items():
            age_hours = (datetime.now() - event.timestamp).total_seconds() / 3600
            
            if age_hours > retention_hours:
                expired_events.append(event_id)
        
        for event_id in expired_events:
            del self.active_events[event_id]
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk adjustment status"""
        # Calculate combined risk adjustments
        if not self.active_adjustments:
            return {
                'active_adjustments': 0,
                'combined_position_multiplier': 1.0,
                'combined_stop_loss_adjustment': 1.0,
                'combined_volatility_adjustment': 1.0,
                'active_events': len(self.active_events),
                'current_stress_level': self.stress_indicators[-1].stress_level if self.stress_indicators else 0.5,
                'status': 'normal'
            }
        
        # Combine all active adjustments (take the most conservative)
        combined_position = min(adj.position_size_multiplier for adj in self.active_adjustments.values())
        combined_stop_loss = min(adj.stop_loss_adjustment for adj in self.active_adjustments.values())
        combined_volatility = min(adj.volatility_threshold_adjustment for adj in self.active_adjustments.values())
        
        # Determine overall status
        if combined_position < 0.3:
            status = 'critical'
        elif combined_position < 0.6:
            status = 'high_risk'
        elif combined_position < 0.8:
            status = 'elevated'
        else:
            status = 'normal'
        
        return {
            'active_adjustments': len(self.active_adjustments),
            'combined_position_multiplier': combined_position,
            'combined_stop_loss_adjustment': combined_stop_loss,
            'combined_volatility_adjustment': combined_volatility,
            'active_events': len(self.active_events),
            'current_stress_level': self.stress_indicators[-1].stress_level if self.stress_indicators else 0.5,
            'status': status,
            'last_update': datetime.now().isoformat()
        }
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get recent event history"""
        return list(self.risk_history)[-50:]  # Return last 50 events
    
    def get_stress_history(self) -> List[Dict[str, Any]]:
        """Get stress level history"""
        return [asdict(indicator) for indicator in self.stress_indicators]

# Factory function
async def create_risk_adjustment_system() -> EventDrivenRiskAdjustment:
    """Create and initialize the event-driven risk adjustment system"""
    system = EventDrivenRiskAdjustment()
    return system

if __name__ == "__main__":
    # Test the risk adjustment system
    async def test_risk_system():
        system = await create_risk_adjustment_system()
        await system.start_monitoring()
    
    asyncio.run(test_risk_system())

