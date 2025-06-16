import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Advanced ML imports with fallbacks
try:
    import tensorflow as tf
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    import torch.nn as nn
    HAS_ADVANCED_ML = True
except ImportError:
    HAS_ADVANCED_ML = False
    print("Advanced ML libraries not available. Using simplified models.")

# Market data and news APIs
try:
    import yfinance as yf
    import feedparser
    import tweepy
    HAS_DATA_SOURCES = True
except ImportError:
    HAS_DATA_SOURCES = False
    print("Data source libraries not available. Using mock data.")

logger = logging.getLogger(__name__)

class MarketSentiment(Enum):
    EXTREMELY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    EXTREMELY_BULLISH = 2

class MarketRegime(Enum):
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class MarketSignal:
    symbol: str
    signal_type: str
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: datetime
    timeframe: str
    description: str
    supporting_indicators: List[str]
    risk_level: str

@dataclass
class MarketIntelligence:
    timestamp: datetime
    overall_sentiment: MarketSentiment
    market_regime: MarketRegime
    volatility_forecast: float
    sector_rotation: Dict[str, float]
    economic_indicators: Dict[str, float]
    geopolitical_risk: float
    liquidity_conditions: str
    correlation_matrix: Dict[str, Dict[str, float]]
    opportunities: List[MarketSignal]
    risks: List[str]
    recommended_actions: List[str]

class AdvancedNewsAnalyzer:
    """Advanced NLP-powered news and sentiment analysis"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.initialized = False
        
        if HAS_ADVANCED_ML:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                self.initialized = True
                logger.info("Advanced NLP models initialized")
            except Exception as e:
                logger.warning(f"Failed to load advanced models: {e}")
                self.initialized = False
    
    async def analyze_news_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        """Analyze sentiment from news headlines using FinBERT"""
        if not self.initialized or not headlines:
            return self._fallback_sentiment_analysis(headlines)
        
        try:
            sentiments = []
            entities = []
            
            for headline in headlines:
                # Sentiment analysis
                sentiment_result = self.sentiment_pipeline(headline)
                sentiments.append(sentiment_result[0])
                
                # Named entity recognition
                ner_result = self.ner_pipeline(headline)
                entities.extend(ner_result)
            
            # Aggregate sentiment
            avg_sentiment = self._aggregate_sentiment(sentiments)
            
            # Extract important entities (companies, persons, locations)
            important_entities = self._extract_important_entities(entities)
            
            return {
                "overall_sentiment": avg_sentiment,
                "sentiment_distribution": self._calculate_sentiment_distribution(sentiments),
                "key_entities": important_entities,
                "confidence": np.mean([s["score"] for s in sentiments]),
                "headline_count": len(headlines)
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return self._fallback_sentiment_analysis(headlines)
    
    def _fallback_sentiment_analysis(self, headlines: List[str]) -> Dict[str, Any]:
        """Simple keyword-based sentiment analysis fallback"""
        positive_words = ['gain', 'rise', 'up', 'growth', 'positive', 'bull', 'high', 'strong']
        negative_words = ['loss', 'fall', 'down', 'decline', 'negative', 'bear', 'low', 'weak']
        
        total_score = 0
        for headline in headlines:
            headline_lower = headline.lower()
            pos_count = sum(1 for word in positive_words if word in headline_lower)
            neg_count = sum(1 for word in negative_words if word in headline_lower)
            total_score += (pos_count - neg_count)
        
        # Normalize to -1 to 1 range
        normalized_score = np.tanh(total_score / len(headlines)) if headlines else 0
        
        return {
            "overall_sentiment": normalized_score,
            "sentiment_distribution": {"positive": 0.5, "negative": 0.3, "neutral": 0.2},
            "key_entities": [],
            "confidence": 0.6,
            "headline_count": len(headlines)
        }
    
    def _aggregate_sentiment(self, sentiments: List[Dict]) -> float:
        """Aggregate individual sentiment scores"""
        if not sentiments:
            return 0.0
        
        total_score = 0
        for sentiment in sentiments:
            # Convert FinBERT labels to numerical scores
            if sentiment["label"] == "positive":
                total_score += sentiment["score"]
            elif sentiment["label"] == "negative":
                total_score -= sentiment["score"]
            # neutral contributes 0
        
        return total_score / len(sentiments)
    
    def _calculate_sentiment_distribution(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of sentiment categories"""
        distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        for sentiment in sentiments:
            distribution[sentiment["label"]] += 1
        
        total = len(sentiments)
        return {k: v/total for k, v in distribution.items()}
    
    def _extract_important_entities(self, entities: List[Dict]) -> List[Dict]:
        """Extract and rank important entities from NER results"""
        entity_counts = {}
        
        for entity in entities:
            key = (entity["word"], entity["entity_group"])
            if key not in entity_counts:
                entity_counts[key] = {"count": 0, "score": 0}
            entity_counts[key]["count"] += 1
            entity_counts[key]["score"] += entity["score"]
        
        # Sort by relevance (count * average score)
        sorted_entities = sorted(
            entity_counts.items(),
            key=lambda x: x[1]["count"] * (x[1]["score"] / x[1]["count"]),
            reverse=True
        )
        
        return [
            {
                "entity": key[0],
                "type": key[1],
                "relevance": value["count"] * (value["score"] / value["count"]),
                "mentions": value["count"]
            }
            for key, value in sorted_entities[:10]  # Top 10 entities
        ]

class PredictiveMarketModel:
    """Advanced ML model for market prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'price_change', 'volume_change', 'volatility', 'rsi', 'macd',
            'bb_position', 'sentiment_score', 'vix_level', 'treasury_yield',
            'dollar_index', 'oil_price', 'gold_price'
        ]
        self.sequence_length = 20  # Use last 20 periods for prediction
        
        if HAS_ADVANCED_ML:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM model for market prediction"""
        try:
            # Simple LSTM model for demonstration
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: bearish, neutral, bullish
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Predictive model built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build predictive model: {e}")
            self.model = None
    
    async def predict_market_direction(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict market direction using ML model"""
        if not HAS_ADVANCED_ML or self.model is None:
            return self._fallback_prediction(market_data)
        
        try:
            # Prepare features
            features = self._prepare_features(market_data)
            
            if len(features) < self.sequence_length:
                return self._fallback_prediction(market_data)
            
            # Make prediction
            prediction = self.model.predict(features[-1:])  # Predict on last sequence
            
            # Interpret results
            probabilities = prediction[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            direction_map = {0: "bearish", 1: "neutral", 2: "bullish"}
            
            return {
                "direction": direction_map[predicted_class],
                "confidence": float(confidence),
                "probabilities": {
                    "bearish": float(probabilities[0]),
                    "neutral": float(probabilities[1]),
                    "bullish": float(probabilities[2])
                },
                "model_type": "LSTM"
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(market_data)
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        # Calculate technical indicators
        features_df = pd.DataFrame()
        
        # Price and volume changes
        features_df['price_change'] = data['close'].pct_change()
        features_df['volume_change'] = data['volume'].pct_change()
        
        # Volatility (rolling standard deviation)
        features_df['volatility'] = data['close'].rolling(window=20).std()
        
        # RSI
        features_df['rsi'] = self._calculate_rsi(data['close'])
        
        # MACD
        features_df['macd'] = self._calculate_macd(data['close'])
        
        # Bollinger Bands position
        features_df['bb_position'] = self._calculate_bb_position(data['close'])
        
        # Mock additional features (in production, fetch from APIs)
        features_df['sentiment_score'] = np.random.normal(0, 0.1, len(data))
        features_df['vix_level'] = np.random.normal(20, 5, len(data))
        features_df['treasury_yield'] = np.random.normal(2.5, 0.5, len(data))
        features_df['dollar_index'] = np.random.normal(100, 5, len(data))
        features_df['oil_price'] = np.random.normal(70, 10, len(data))
        features_df['gold_price'] = np.random.normal(1800, 100, len(data))
        
        # Fill NaN values and normalize
        features_df = features_df.fillna(0)
        
        # Create sequences
        sequences = []
        for i in range(self.sequence_length, len(features_df)):
            sequences.append(features_df.iloc[i-self.sequence_length:i].values)
        
        return np.array(sequences)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _calculate_bb_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        # Position: 0 = lower band, 0.5 = middle, 1 = upper band
        return (prices - lower_band) / (upper_band - lower_band)
    
    def _fallback_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple technical analysis fallback"""
        if len(data) < 20:
            return {
                "direction": "neutral",
                "confidence": 0.5,
                "probabilities": {"bearish": 0.33, "neutral": 0.34, "bullish": 0.33},
                "model_type": "technical_fallback"
            }
        
        # Simple moving average strategy
        recent_price = data['close'].iloc[-1]
        sma_5 = data['close'].tail(5).mean()
        sma_20 = data['close'].tail(20).mean()
        
        if recent_price > sma_5 > sma_20:
            direction = "bullish"
            confidence = 0.7
        elif recent_price < sma_5 < sma_20:
            direction = "bearish"
            confidence = 0.7
        else:
            direction = "neutral"
            confidence = 0.6
        
        return {
            "direction": direction,
            "confidence": confidence,
            "probabilities": {"bearish": 0.3, "neutral": 0.4, "bullish": 0.3},
            "model_type": "technical_fallback"
        }

class MarketIntelligenceEngine:
    """Main engine for comprehensive market intelligence"""
    
    def __init__(self):
        self.news_analyzer = AdvancedNewsAnalyzer()
        self.predictive_model = PredictiveMarketModel()
        self.data_sources = {
            'news_feeds': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.bloomberg.com/markets/news.rss'
            ],
            'market_data_symbols': ['SPY', 'QQQ', 'VIX', 'TLT', 'GLD', 'USO']
        }
        self.cache = {}
        self.last_update = None
        self.update_interval = timedelta(minutes=5)
        
        # Start background data collection
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start the intelligence engine"""
        self.running = True
        logger.info("Market Intelligence Engine started")
        
        # Start background tasks
        asyncio.create_task(self._background_data_collection())
        asyncio.create_task(self._background_analysis())
    
    async def stop(self):
        """Stop the intelligence engine"""
        self.running = False
        self.executor.shutdown(wait=False)
        logger.info("Market Intelligence Engine stopped")
    
    async def get_market_intelligence(self) -> MarketIntelligence:
        """Get comprehensive market intelligence"""
        try:
            # Collect latest data
            market_data = await self._get_market_data()
            news_sentiment = await self._get_news_sentiment()
            economic_data = await self._get_economic_indicators()
            
            # Run predictive analysis
            market_prediction = await self.predictive_model.predict_market_direction(market_data)
            
            # Analyze market regime
            market_regime = self._determine_market_regime(market_data, news_sentiment)
            
            # Calculate sector rotation
            sector_rotation = await self._analyze_sector_rotation()
            
            # Identify opportunities and risks
            opportunities = await self._identify_opportunities(market_data, news_sentiment)
            risks = self._assess_risks(market_data, news_sentiment, economic_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                market_prediction, market_regime, opportunities, risks
            )
            
            return MarketIntelligence(
                timestamp=datetime.now(),
                overall_sentiment=self._convert_to_sentiment_enum(news_sentiment.get('overall_sentiment', 0)),
                market_regime=market_regime,
                volatility_forecast=self._forecast_volatility(market_data),
                sector_rotation=sector_rotation,
                economic_indicators=economic_data,
                geopolitical_risk=self._assess_geopolitical_risk(news_sentiment),
                liquidity_conditions=self._assess_liquidity(market_data),
                correlation_matrix=self._calculate_correlations(market_data),
                opportunities=opportunities,
                risks=risks,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to generate market intelligence: {e}")
            return self._get_fallback_intelligence()
    
    async def _background_data_collection(self):
        """Background task for continuous data collection"""
        while self.running:
            try:
                # Update market data cache
                self.cache['market_data'] = await self._fetch_market_data()
                self.cache['news_data'] = await self._fetch_news_data()
                self.cache['economic_data'] = await self._fetch_economic_data()
                
                self.last_update = datetime.now()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Background data collection error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _background_analysis(self):
        """Background task for continuous analysis"""
        while self.running:
            try:
                if 'market_data' in self.cache and 'news_data' in self.cache:
                    # Run analysis on cached data
                    analysis_result = await self._run_comprehensive_analysis()
                    self.cache['latest_analysis'] = analysis_result
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Background analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _get_market_data(self) -> pd.DataFrame:
        """Get market data from cache or fetch fresh"""
        if 'market_data' in self.cache:
            return self.cache['market_data']
        
        return await self._fetch_market_data()
    
    async def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch fresh market data"""
        if not HAS_DATA_SOURCES:
            return self._generate_mock_market_data()
        
        try:
            # Fetch data for primary market index (SPY)
            ticker = yf.Ticker('SPY')
            data = ticker.history(period='3mo', interval='1d')
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return self._generate_mock_market_data()
    
    def _generate_mock_market_data(self) -> pd.DataFrame:
        """Generate mock market data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(50000000, 200000000, len(dates))
        }, index=dates)
    
    async def _get_news_sentiment(self) -> Dict[str, Any]:
        """Get news sentiment from cache or fetch fresh"""
        if 'news_sentiment' in self.cache:
            return self.cache['news_sentiment']
        
        news_data = await self._fetch_news_data()
        sentiment = await self.news_analyzer.analyze_news_sentiment(news_data)
        self.cache['news_sentiment'] = sentiment
        
        return sentiment
    
    async def _fetch_news_data(self) -> List[str]:
        """Fetch latest financial news headlines"""
        headlines = []
        
        try:
            # Mock news headlines for demonstration
            mock_headlines = [
                "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
                "Tech stocks rally on strong quarterly earnings reports",
                "Global markets show resilience despite geopolitical tensions",
                "Cryptocurrency market experiences significant volatility",
                "Energy sector outperforms as oil prices stabilize",
                "Banking stocks under pressure from regulatory changes",
                "Consumer confidence index reaches new highs",
                "Inflation data shows continued moderation",
                "Emerging markets attract increased investor interest",
                "AI and technology companies drive market growth"
            ]
            
            headlines.extend(mock_headlines)
            
        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
        
        return headlines
    
    async def _get_economic_indicators(self) -> Dict[str, float]:
        """Get economic indicators"""
        # Mock economic indicators (in production, fetch from APIs)
        return {
            'unemployment_rate': 3.7,
            'inflation_rate': 2.4,
            'gdp_growth': 2.1,
            'consumer_confidence': 102.3,
            'manufacturing_pmi': 52.1,
            'services_pmi': 54.8,
            'treasury_10y': 4.2,
            'treasury_2y': 4.8,
            'dollar_index': 104.2,
            'oil_price': 72.5,
            'gold_price': 1985.3,
            'vix': 18.7
        }
    
    async def _fetch_economic_data(self) -> Dict[str, float]:
        """Fetch economic data from APIs"""
        return await self._get_economic_indicators()
    
    def _determine_market_regime(self, market_data: pd.DataFrame, sentiment: Dict[str, Any]) -> MarketRegime:
        """Determine current market regime"""
        if len(market_data) < 50:
            return MarketRegime.SIDEWAYS
        
        # Calculate recent performance
        recent_return = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-21] - 1)
        volatility = market_data['Close'].pct_change().tail(21).std() * np.sqrt(252)
        sentiment_score = sentiment.get('overall_sentiment', 0)
        
        # Regime classification logic
        if recent_return > 0.05 and sentiment_score > 0.2 and volatility < 0.25:
            return MarketRegime.BULL_MARKET
        elif recent_return < -0.05 and sentiment_score < -0.2:
            return MarketRegime.BEAR_MARKET
        elif volatility > 0.4 or sentiment_score < -0.5:
            return MarketRegime.CRISIS
        elif recent_return > 0 and sentiment_score > 0:
            return MarketRegime.RECOVERY
        else:
            return MarketRegime.SIDEWAYS
    
    async def _analyze_sector_rotation(self) -> Dict[str, float]:
        """Analyze sector rotation patterns"""
        # Mock sector performance (in production, fetch real sector ETF data)
        return {
            'Technology': 0.15,
            'Healthcare': 0.08,
            'Financial': 0.12,
            'Energy': 0.18,
            'Consumer_Discretionary': 0.06,
            'Consumer_Staples': 0.02,
            'Utilities': -0.03,
            'Real_Estate': 0.01,
            'Materials': 0.09,
            'Industrials': 0.11,
            'Communication': 0.07
        }
    
    async def _identify_opportunities(self, market_data: pd.DataFrame, sentiment: Dict[str, Any]) -> List[MarketSignal]:
        """Identify trading opportunities"""
        opportunities = []
        
        # Mock opportunity identification
        if sentiment.get('overall_sentiment', 0) > 0.3:
            opportunities.append(MarketSignal(
                symbol='SPY',
                signal_type='MOMENTUM_BREAKOUT',
                strength=0.8,
                confidence=0.75,
                timestamp=datetime.now(),
                timeframe='1D',
                description='Strong bullish sentiment with technical breakout pattern',
                supporting_indicators=['Moving Average', 'RSI', 'Volume'],
                risk_level='Medium'
            ))
        
        return opportunities
    
    def _assess_risks(self, market_data: pd.DataFrame, sentiment: Dict[str, Any], economic_data: Dict[str, float]) -> List[str]:
        """Assess current market risks"""
        risks = []
        
        # Volatility risk
        if len(market_data) >= 21:
            volatility = market_data['Close'].pct_change().tail(21).std() * np.sqrt(252)
            if volatility > 0.3:
                risks.append(f"High market volatility detected ({volatility:.1%})")
        
        # Sentiment risk
        if sentiment.get('overall_sentiment', 0) < -0.3:
            risks.append("Negative market sentiment prevailing")
        
        # Economic risks
        if economic_data.get('vix', 0) > 25:
            risks.append("Elevated fear index (VIX > 25)")
        
        if economic_data.get('treasury_10y', 0) > 5.0:
            risks.append("High interest rate environment")
        
        return risks
    
    def _generate_recommendations(self, prediction: Dict[str, Any], regime: MarketRegime, 
                                opportunities: List[MarketSignal], risks: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on prediction
        if prediction['confidence'] > 0.7:
            if prediction['direction'] == 'bullish':
                recommendations.append("Consider increasing equity exposure")
            elif prediction['direction'] == 'bearish':
                recommendations.append("Consider defensive positioning")
        
        # Based on regime
        if regime == MarketRegime.BULL_MARKET:
            recommendations.append("Favor growth strategies and momentum plays")
        elif regime == MarketRegime.BEAR_MARKET:
            recommendations.append("Focus on capital preservation and defensive assets")
        elif regime == MarketRegime.CRISIS:
            recommendations.append("Maintain high cash levels and quality positions")
        
        # Based on opportunities
        if opportunities:
            recommendations.append(f"Monitor {len(opportunities)} identified opportunities")
        
        # Risk management
        if risks:
            recommendations.append("Implement enhanced risk management protocols")
        
        return recommendations
    
    def _convert_to_sentiment_enum(self, sentiment_score: float) -> MarketSentiment:
        """Convert numerical sentiment to enum"""
        if sentiment_score >= 0.5:
            return MarketSentiment.EXTREMELY_BULLISH
        elif sentiment_score >= 0.2:
            return MarketSentiment.BULLISH
        elif sentiment_score <= -0.5:
            return MarketSentiment.EXTREMELY_BEARISH
        elif sentiment_score <= -0.2:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.NEUTRAL
    
    def _forecast_volatility(self, market_data: pd.DataFrame) -> float:
        """Forecast volatility using GARCH-like model"""
        if len(market_data) < 21:
            return 0.2  # Default volatility
        
        returns = market_data['Close'].pct_change().dropna()
        current_vol = returns.tail(21).std() * np.sqrt(252)
        
        # Simple volatility forecast (in production, use GARCH model)
        forecast_vol = current_vol * 0.94 + 0.2 * 0.06  # Mean reversion to 20%
        
        return float(forecast_vol)
    
    def _assess_geopolitical_risk(self, sentiment: Dict[str, Any]) -> float:
        """Assess geopolitical risk from news sentiment"""
        # Simple assessment based on negative sentiment intensity
        base_risk = 0.3  # Base geopolitical risk
        sentiment_impact = max(0, -sentiment.get('overall_sentiment', 0)) * 0.5
        
        return min(1.0, base_risk + sentiment_impact)
    
    def _assess_liquidity(self, market_data: pd.DataFrame) -> str:
        """Assess market liquidity conditions"""
        if len(market_data) < 21:
            return "normal"
        
        avg_volume = market_data['Volume'].tail(21).mean()
        recent_volume = market_data['Volume'].tail(5).mean()
        
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.5:
            return "high"
        elif volume_ratio < 0.7:
            return "low"
        else:
            return "normal"
    
    def _calculate_correlations(self, market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate asset correlations"""
        # Mock correlation matrix (in production, calculate from multiple assets)
        return {
            'SPY': {'QQQ': 0.85, 'VIX': -0.72, 'TLT': -0.45, 'GLD': -0.15},
            'QQQ': {'SPY': 0.85, 'VIX': -0.68, 'TLT': -0.52, 'GLD': -0.20},
            'VIX': {'SPY': -0.72, 'QQQ': -0.68, 'TLT': 0.35, 'GLD': 0.25},
            'TLT': {'SPY': -0.45, 'QQQ': -0.52, 'VIX': 0.35, 'GLD': 0.18},
            'GLD': {'SPY': -0.15, 'QQQ': -0.20, 'VIX': 0.25, 'TLT': 0.18}
        }
    
    async def _run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive market analysis"""
        try:
            market_data = self.cache.get('market_data')
            news_data = self.cache.get('news_data', [])
            
            if market_data is None:
                return {}
            
            # Run parallel analysis
            sentiment_task = self.news_analyzer.analyze_news_sentiment(news_data)
            prediction_task = self.predictive_model.predict_market_direction(market_data)
            
            sentiment_result = await sentiment_task
            prediction_result = await prediction_task
            
            return {
                'sentiment_analysis': sentiment_result,
                'market_prediction': prediction_result,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {}
    
    def _get_fallback_intelligence(self) -> MarketIntelligence:
        """Provide fallback intelligence when systems fail"""
        return MarketIntelligence(
            timestamp=datetime.now(),
            overall_sentiment=MarketSentiment.NEUTRAL,
            market_regime=MarketRegime.SIDEWAYS,
            volatility_forecast=0.2,
            sector_rotation={},
            economic_indicators={},
            geopolitical_risk=0.5,
            liquidity_conditions="normal",
            correlation_matrix={},
            opportunities=[],
            risks=["Intelligence system temporarily unavailable"],
            recommended_actions=["Maintain current positions until system recovery"]
        )

# Example usage and testing
if __name__ == "__main__":
    async def test_market_intelligence():
        engine = MarketIntelligenceEngine()
        await engine.start()
        
        try:
            intelligence = await engine.get_market_intelligence()
            
            print("\n=== MARKET INTELLIGENCE REPORT ===")
            print(f"Timestamp: {intelligence.timestamp}")
            print(f"Overall Sentiment: {intelligence.overall_sentiment.name}")
            print(f"Market Regime: {intelligence.market_regime.value}")
            print(f"Volatility Forecast: {intelligence.volatility_forecast:.2%}")
            print(f"Geopolitical Risk: {intelligence.geopolitical_risk:.2f}")
            print(f"Liquidity: {intelligence.liquidity_conditions}")
            
            print("\nOpportunities:")
            for opp in intelligence.opportunities:
                print(f"  {opp.symbol}: {opp.description} (Confidence: {opp.confidence:.2%})")
            
            print("\nRisks:")
            for risk in intelligence.risks:
                print(f"  - {risk}")
            
            print("\nRecommendations:")
            for rec in intelligence.recommended_actions:
                print(f"  - {rec}")
                
        finally:
            await engine.stop()
    
    # Run test
    asyncio.run(test_market_intelligence())

