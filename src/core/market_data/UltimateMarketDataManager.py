import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import statistics
import time

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    exchange: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None
    liquidity_score: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    
@dataclass
class OrderBookData:
    """Order book depth data"""
    symbol: str
    exchange: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    timestamp: datetime
    depth: int = 10
    spread: float = 0.0
    mid_price: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    imbalance: float = 0.0
    
@dataclass
class ArbitrageOpportunity:
    """Real-time arbitrage opportunity"""
    opportunity_id: str
    strategy_type: str
    symbol: str
    exchanges: List[str]
    profit_potential: float
    profit_percentage: float
    confidence_score: float
    execution_time_estimate: int  # seconds
    risk_score: float
    liquidity_requirement: float
    price_data: Dict[str, float]
    timestamp: datetime
    expiry_time: datetime
    status: str = "detected"  # detected, analyzing, ready, executing, expired
    
@dataclass
class MarketRegimeData:
    """Market regime analysis data"""
    timestamp: datetime
    volatility_regime: str  # low, medium, high
    trend_regime: str  # bull, bear, sideways
    liquidity_regime: str  # abundant, normal, scarce
    correlation_regime: str  # normal, breakdown, extreme
    risk_regime: str  # low, medium, high, extreme
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    stability_score: float = 0.0
    
class DataFeedInterface(ABC):
    """Abstract interface for market data feeds"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data feed"""
        pass
    
    @abstractmethod
    async def subscribe_ticks(self, symbols: List[str]) -> bool:
        """Subscribe to tick data"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 10) -> bool:
        """Subscribe to order book data"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from data feed"""
        pass

class UltimateMarketDataManager:
    """Advanced market data management with real-time analytics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Data storage
        self.tick_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.orderbook_data: Dict[str, OrderBookData] = {}
        self.opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.market_regimes: deque = deque(maxlen=1000)
        
        # Real-time analytics
        self.price_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.liquidity_scores: Dict[str, float] = {}
        
        # Data feeds
        self.data_feeds: Dict[str, DataFeedInterface] = {}
        self.feed_connections: Dict[str, bool] = {}
        
        # Processing queues
        self.tick_queue = asyncio.Queue(maxsize=10000)
        self.orderbook_queue = asyncio.Queue(maxsize=5000)
        self.analysis_queue = asyncio.Queue(maxsize=1000)
        
        # Callbacks and subscribers
        self.tick_callbacks: List[Callable] = []
        self.opportunity_callbacks: List[Callable] = []
        self.regime_callbacks: List[Callable] = []
        
        # Processing tasks
        self.processing_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Performance monitoring
        self.processing_stats = {
            'ticks_processed': 0,
            'opportunities_detected': 0,
            'last_tick_time': None,
            'processing_latency': deque(maxlen=1000),
            'queue_sizes': {},
            'error_count': 0
        }
        
        self.logger.info("Ultimate Market Data Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'max_tick_history': 10000,
            'max_orderbook_depth': 20,
            'opportunity_min_profit': 0.001,  # 0.1% minimum profit
            'opportunity_max_age': 30,  # 30 seconds max age
            'volatility_window': 100,
            'correlation_window': 200,
            'regime_analysis_interval': 60,  # seconds
            'cleanup_interval': 300,  # 5 minutes
            'processing_batch_size': 100,
            'max_processing_latency': 0.1,  # 100ms
            'error_threshold': 10,
            'reconnect_delay': 5,
            'data_quality_threshold': 0.95,
            'anomaly_detection_enabled': True,
            'quantum_analysis_enabled': True
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('UltimateMarketDataManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def start_data_processing(self) -> bool:
        """Start all data processing tasks"""
        try:
            if self.is_running:
                self.logger.warning("Data processing already running")
                return True
            
            self.logger.info("Starting Ultimate Market Data Processing...")
            
            # Start processing tasks
            self.processing_tasks = [
                asyncio.create_task(self._tick_processor()),
                asyncio.create_task(self._orderbook_processor()),
                asyncio.create_task(self._opportunity_analyzer()),
                asyncio.create_task(self._regime_analyzer()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._cleanup_task())
            ]
            
            self.is_running = True
            self.logger.info("Market data processing started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting data processing: {str(e)}")
            return False
    
    async def add_data_feed(self, feed_name: str, feed_instance: DataFeedInterface) -> bool:
        """Add a new data feed"""
        try:
            self.data_feeds[feed_name] = feed_instance
            
            # Attempt connection
            connected = await feed_instance.connect()
            self.feed_connections[feed_name] = connected
            
            if connected:
                self.logger.info(f"Data feed {feed_name} connected successfully")
            else:
                self.logger.warning(f"Failed to connect data feed {feed_name}")
            
            return connected
            
        except Exception as e:
            self.logger.error(f"Error adding data feed {feed_name}: {str(e)}")
            return False
    
    async def subscribe_to_symbols(self, symbols: List[str], feed_name: Optional[str] = None) -> bool:
        """Subscribe to market data for symbols"""
        try:
            feeds_to_use = [feed_name] if feed_name else list(self.data_feeds.keys())
            
            for feed_name in feeds_to_use:
                if feed_name in self.data_feeds and self.feed_connections.get(feed_name, False):
                    feed = self.data_feeds[feed_name]
                    
                    # Subscribe to ticks and order books
                    await feed.subscribe_ticks(symbols)
                    await feed.subscribe_orderbook(symbols, self.config['max_orderbook_depth'])
                    
                    self.logger.info(f"Subscribed to {len(symbols)} symbols on {feed_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to symbols: {str(e)}")
            return False
    
    async def process_tick(self, tick: MarketTick) -> None:
        """Process incoming tick data"""
        try:
            start_time = time.time()
            
            # Add to queue for processing
            await self.tick_queue.put(tick)
            
            # Update processing stats
            self.processing_stats['last_tick_time'] = tick.timestamp
            
            # Calculate processing latency
            latency = time.time() - start_time
            self.processing_stats['processing_latency'].append(latency)
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}")
            self.processing_stats['error_count'] += 1
    
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process order book data"""
        try:
            await self.orderbook_queue.put(orderbook)
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook: {str(e)}")
            self.processing_stats['error_count'] += 1
    
    async def _tick_processor(self) -> None:
        """Main tick data processing loop"""
        while self.is_running:
            try:
                batch = []
                
                # Collect batch of ticks
                for _ in range(self.config['processing_batch_size']):
                    try:
                        tick = await asyncio.wait_for(self.tick_queue.get(), timeout=0.1)
                        batch.append(tick)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_tick_batch(batch)
                
                # Update processing stats
                self.processing_stats['ticks_processed'] += len(batch)
                
            except Exception as e:
                self.logger.error(f"Error in tick processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_tick_batch(self, ticks: List[MarketTick]) -> None:
        """Process a batch of ticks"""
        try:
            for tick in ticks:
                # Store tick data
                key = f"{tick.symbol}_{tick.exchange}"
                self.tick_data[key].append(tick)
                
                # Update price cache
                self.price_cache[tick.symbol][tick.exchange] = tick.price
                
                # Calculate real-time metrics
                await self._calculate_tick_metrics(tick)
                
                # Notify callbacks
                for callback in self.tick_callbacks:
                    try:
                        await callback(tick)
                    except Exception as e:
                        self.logger.warning(f"Tick callback failed: {str(e)}")
                
                # Check for arbitrage opportunities
                await self._check_arbitrage_opportunities(tick)
                
        except Exception as e:
            self.logger.error(f"Error processing tick batch: {str(e)}")
    
    async def _calculate_tick_metrics(self, tick: MarketTick) -> None:
        """Calculate real-time metrics for tick"""
        try:
            key = f"{tick.symbol}_{tick.exchange}"
            tick_history = list(self.tick_data[key])
            
            if len(tick_history) < 2:
                return
            
            # Calculate volatility
            recent_prices = [t.price for t in tick_history[-self.config['volatility_window']:]]
            if len(recent_prices) > 1:
                returns = np.diff(np.log(recent_prices))
                volatility = np.std(returns) * np.sqrt(60)  # Annualized volatility
                self.volatility_cache[key] = volatility
                tick.volatility = volatility
            
            # Calculate momentum
            if len(recent_prices) >= 10:
                momentum = (recent_prices[-1] - recent_prices[-10]) / recent_prices[-10]
                tick.momentum = momentum
            
            # Calculate spread and liquidity
            if tick.bid and tick.ask:
                tick.spread = tick.ask - tick.bid
                
                # Simple liquidity score based on spread and volume
                if tick.spread > 0:
                    tick.liquidity_score = tick.volume / tick.spread
                    self.liquidity_scores[key] = tick.liquidity_score
                
        except Exception as e:
            self.logger.warning(f"Error calculating tick metrics: {str(e)}")
    
    async def _orderbook_processor(self) -> None:
        """Process order book data"""
        while self.is_running:
            try:
                orderbook = await asyncio.wait_for(self.orderbook_queue.get(), timeout=1.0)
                
                # Calculate order book metrics
                await self._calculate_orderbook_metrics(orderbook)
                
                # Store order book
                key = f"{orderbook.symbol}_{orderbook.exchange}"
                self.orderbook_data[key] = orderbook
                
                # Check for market microstructure opportunities
                await self._analyze_microstructure(orderbook)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in orderbook processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def _calculate_orderbook_metrics(self, orderbook: OrderBookData) -> None:
        """Calculate order book metrics"""
        try:
            if not orderbook.bids or not orderbook.asks:
                return
            
            # Calculate basic metrics
            best_bid = max(orderbook.bids, key=lambda x: x[0])[0]
            best_ask = min(orderbook.asks, key=lambda x: x[0])[0]
            
            orderbook.spread = best_ask - best_bid
            orderbook.mid_price = (best_bid + best_ask) / 2
            
            # Calculate volume metrics
            orderbook.total_bid_volume = sum(size for _, size in orderbook.bids)
            orderbook.total_ask_volume = sum(size for _, size in orderbook.asks)
            
            # Calculate imbalance
            total_volume = orderbook.total_bid_volume + orderbook.total_ask_volume
            if total_volume > 0:
                orderbook.imbalance = (orderbook.total_bid_volume - orderbook.total_ask_volume) / total_volume
            
        except Exception as e:
            self.logger.warning(f"Error calculating orderbook metrics: {str(e)}")
    
    async def _check_arbitrage_opportunities(self, tick: MarketTick) -> None:
        """Check for arbitrage opportunities"""
        try:
            symbol = tick.symbol
            
            # Get prices from all exchanges for this symbol
            exchange_prices = self.price_cache.get(symbol, {})
            
            if len(exchange_prices) < 2:
                return
            
            # Find min and max prices
            min_exchange = min(exchange_prices.keys(), key=lambda x: exchange_prices[x])
            max_exchange = max(exchange_prices.keys(), key=lambda x: exchange_prices[x])
            
            min_price = exchange_prices[min_exchange]
            max_price = exchange_prices[max_exchange]
            
            # Calculate profit potential
            profit_potential = max_price - min_price
            profit_percentage = profit_potential / min_price
            
            # Check if opportunity meets minimum criteria
            if profit_percentage >= self.config['opportunity_min_profit']:
                opportunity = await self._create_arbitrage_opportunity(
                    symbol, min_exchange, max_exchange, min_price, max_price, profit_percentage
                )
                
                if opportunity:
                    await self._process_opportunity(opportunity)
                    
        except Exception as e:
            self.logger.warning(f"Error checking arbitrage opportunities: {str(e)}")
    
    async def _create_arbitrage_opportunity(
        self, 
        symbol: str, 
        buy_exchange: str, 
        sell_exchange: str, 
        buy_price: float, 
        sell_price: float, 
        profit_percentage: float
    ) -> Optional[ArbitrageOpportunity]:
        """Create arbitrage opportunity object"""
        try:
            opportunity_id = f"{symbol}_{buy_exchange}_{sell_exchange}_{int(time.time())}"
            
            # Calculate confidence score based on multiple factors
            confidence_factors = []
            
            # Liquidity factor
            buy_key = f"{symbol}_{buy_exchange}"
            sell_key = f"{symbol}_{sell_exchange}"
            
            buy_liquidity = self.liquidity_scores.get(buy_key, 0)
            sell_liquidity = self.liquidity_scores.get(sell_key, 0)
            liquidity_score = min(buy_liquidity, sell_liquidity) / max(buy_liquidity, sell_liquidity, 1)
            confidence_factors.append(liquidity_score)
            
            # Volatility factor (lower volatility = higher confidence)
            buy_volatility = self.volatility_cache.get(buy_key, 0)
            sell_volatility = self.volatility_cache.get(sell_key, 0)
            avg_volatility = (buy_volatility + sell_volatility) / 2
            volatility_score = max(0, 1 - avg_volatility * 10)  # Scale volatility
            confidence_factors.append(volatility_score)
            
            # Profit magnitude factor
            profit_score = min(1.0, profit_percentage / 0.01)  # Normalize to 1% profit
            confidence_factors.append(profit_score)
            
            # Calculate overall confidence
            confidence_score = np.mean(confidence_factors)
            
            # Calculate risk score
            risk_score = avg_volatility + (1 - liquidity_score)
            
            # Estimate execution time and liquidity requirement
            execution_time = max(5, int(20 * (1 - liquidity_score)))  # 5-20 seconds
            liquidity_requirement = sell_price * 100  # Assume 100 units minimum
            
            opportunity = ArbitrageOpportunity(
                opportunity_id=opportunity_id,
                strategy_type="simple_arbitrage",
                symbol=symbol,
                exchanges=[buy_exchange, sell_exchange],
                profit_potential=sell_price - buy_price,
                profit_percentage=profit_percentage,
                confidence_score=confidence_score,
                execution_time_estimate=execution_time,
                risk_score=risk_score,
                liquidity_requirement=liquidity_requirement,
                price_data={
                    buy_exchange: buy_price,
                    sell_exchange: sell_price
                },
                timestamp=datetime.now(),
                expiry_time=datetime.now() + timedelta(seconds=self.config['opportunity_max_age'])
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error creating arbitrage opportunity: {str(e)}")
            return None
    
    async def _process_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        """Process detected arbitrage opportunity"""
        try:
            # Store opportunity
            self.opportunities[opportunity.opportunity_id] = opportunity
            
            # Update stats
            self.processing_stats['opportunities_detected'] += 1
            
            # Notify callbacks
            for callback in self.opportunity_callbacks:
                try:
                    await callback(opportunity)
                except Exception as e:
                    self.logger.warning(f"Opportunity callback failed: {str(e)}")
            
            self.logger.info(
                f"Arbitrage opportunity detected: {opportunity.symbol} "
                f"{opportunity.profit_percentage:.4f}% profit, confidence: {opportunity.confidence_score:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing opportunity: {str(e)}")
    
    async def _opportunity_analyzer(self) -> None:
        """Analyze and refine arbitrage opportunities"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Clean up expired opportunities
                expired_ids = [
                    opp_id for opp_id, opp in self.opportunities.items()
                    if opp.expiry_time < current_time
                ]
                
                for opp_id in expired_ids:
                    del self.opportunities[opp_id]
                
                # Analyze active opportunities
                for opportunity in list(self.opportunities.values()):
                    await self._analyze_opportunity(opportunity)
                
                await asyncio.sleep(5)  # Analyze every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in opportunity analyzer: {str(e)}")
                await asyncio.sleep(10)
    
    async def _analyze_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        """Analyze individual opportunity"""
        try:
            # Re-calculate profit based on current prices
            current_prices = self.price_cache.get(opportunity.symbol, {})
            
            if not all(exchange in current_prices for exchange in opportunity.exchanges):
                return
            
            buy_exchange, sell_exchange = opportunity.exchanges
            current_buy_price = current_prices[buy_exchange]
            current_sell_price = current_prices[sell_exchange]
            
            # Update opportunity data
            opportunity.price_data[buy_exchange] = current_buy_price
            opportunity.price_data[sell_exchange] = current_sell_price
            opportunity.profit_potential = current_sell_price - current_buy_price
            opportunity.profit_percentage = opportunity.profit_potential / current_buy_price
            
            # Update status based on profitability
            if opportunity.profit_percentage < self.config['opportunity_min_profit']:
                opportunity.status = "expired"
            elif opportunity.profit_percentage > 0.005:  # 0.5% profit
                opportunity.status = "ready"
            else:
                opportunity.status = "analyzing"
                
        except Exception as e:
            self.logger.warning(f"Error analyzing opportunity: {str(e)}")
    
    async def _regime_analyzer(self) -> None:
        """Analyze market regimes"""
        while self.is_running:
            try:
                regime_data = await self._calculate_market_regime()
                
                if regime_data:
                    self.market_regimes.append(regime_data)
                    
                    # Notify callbacks
                    for callback in self.regime_callbacks:
                        try:
                            await callback(regime_data)
                        except Exception as e:
                            self.logger.warning(f"Regime callback failed: {str(e)}")
                
                await asyncio.sleep(self.config['regime_analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in regime analyzer: {str(e)}")
                await asyncio.sleep(60)
    
    async def _calculate_market_regime(self) -> Optional[MarketRegimeData]:
        """Calculate current market regime"""
        try:
            if not self.tick_data:
                return None
            
            # Aggregate data across all symbols and exchanges
            all_prices = []
            all_volatilities = []
            all_volumes = []
            
            for key, ticks in self.tick_data.items():
                if len(ticks) >= 100:  # Minimum data requirement
                    recent_ticks = list(ticks)[-100:]
                    prices = [t.price for t in recent_ticks]
                    volumes = [t.volume for t in recent_ticks]
                    
                    all_prices.extend(prices)
                    all_volumes.extend(volumes)
                    
                    # Calculate volatility for this instrument
                    if len(prices) > 1:
                        returns = np.diff(np.log(prices))
                        volatility = np.std(returns)
                        all_volatilities.append(volatility)
            
            if not all_prices:
                return None
            
            # Calculate regime indicators
            avg_volatility = np.mean(all_volatilities) if all_volatilities else 0
            price_trend = (all_prices[-1] - all_prices[0]) / all_prices[0] if len(all_prices) > 1 else 0
            avg_volume = np.mean(all_volumes) if all_volumes else 0
            
            # Determine regimes
            volatility_regime = self._classify_volatility_regime(avg_volatility)
            trend_regime = self._classify_trend_regime(price_trend)
            liquidity_regime = self._classify_liquidity_regime(avg_volume)
            
            # Calculate correlation regime (simplified)
            correlation_regime = "normal"  # Would need more sophisticated analysis
            
            # Calculate overall risk regime
            risk_regime = self._calculate_risk_regime(volatility_regime, trend_regime, liquidity_regime)
            
            regime_data = MarketRegimeData(
                timestamp=datetime.now(),
                volatility_regime=volatility_regime,
                trend_regime=trend_regime,
                liquidity_regime=liquidity_regime,
                correlation_regime=correlation_regime,
                risk_regime=risk_regime,
                confidence_scores={
                    'volatility': 0.8,
                    'trend': 0.7,
                    'liquidity': 0.6,
                    'correlation': 0.5
                },
                stability_score=0.7
            )
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime: {str(e)}")
            return None
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.01:
            return "low"
        elif volatility < 0.03:
            return "medium"
        else:
            return "high"
    
    def _classify_trend_regime(self, trend: float) -> str:
        """Classify trend regime"""
        if trend > 0.02:
            return "bull"
        elif trend < -0.02:
            return "bear"
        else:
            return "sideways"
    
    def _classify_liquidity_regime(self, volume: float) -> str:
        """Classify liquidity regime"""
        # This is simplified - would need historical benchmarks
        if volume > 1000000:
            return "abundant"
        elif volume > 100000:
            return "normal"
        else:
            return "scarce"
    
    def _calculate_risk_regime(self, vol_regime: str, trend_regime: str, liquidity_regime: str) -> str:
        """Calculate overall risk regime"""
        risk_score = 0
        
        if vol_regime == "high":
            risk_score += 2
        elif vol_regime == "medium":
            risk_score += 1
        
        if trend_regime == "bear":
            risk_score += 1
        
        if liquidity_regime == "scarce":
            risk_score += 2
        elif liquidity_regime == "normal":
            risk_score += 1
        
        if risk_score >= 4:
            return "extreme"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    async def _performance_monitor(self) -> None:
        """Monitor system performance"""
        while self.is_running:
            try:
                # Update queue sizes
                self.processing_stats['queue_sizes'] = {
                    'tick_queue': self.tick_queue.qsize(),
                    'orderbook_queue': self.orderbook_queue.qsize(),
                    'analysis_queue': self.analysis_queue.qsize()
                }
                
                # Check processing latency
                if self.processing_stats['processing_latency']:
                    avg_latency = statistics.mean(self.processing_stats['processing_latency'])
                    
                    if avg_latency > self.config['max_processing_latency']:
                        self.logger.warning(f"High processing latency detected: {avg_latency:.4f}s")
                
                # Check error rate
                if self.processing_stats['error_count'] > self.config['error_threshold']:
                    self.logger.error(f"High error rate detected: {self.processing_stats['error_count']} errors")
                    
                    # Reset error count
                    self.processing_stats['error_count'] = 0
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self) -> None:
        """Periodic cleanup task"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config['cleanup_interval'])
                
                # Clean up old tick data
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=1)
                
                for key, ticks in self.tick_data.items():
                    # Remove old ticks
                    while ticks and ticks[0].timestamp < cutoff_time:
                        ticks.popleft()
                
                # Clean up expired opportunities
                expired_opportunities = [
                    opp_id for opp_id, opp in self.opportunities.items()
                    if opp.expiry_time < current_time
                ]
                
                for opp_id in expired_opportunities:
                    del self.opportunities[opp_id]
                
                self.logger.info(f"Cleanup completed. Removed {len(expired_opportunities)} expired opportunities")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
    
    def register_tick_callback(self, callback: Callable) -> None:
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
    
    def register_opportunity_callback(self, callback: Callable) -> None:
        """Register callback for arbitrage opportunities"""
        self.opportunity_callbacks.append(callback)
    
    def register_regime_callback(self, callback: Callable) -> None:
        """Register callback for market regime changes"""
        self.regime_callbacks.append(callback)
    
    def get_current_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities"""
        current_time = datetime.now()
        
        active_opportunities = [
            opp for opp in self.opportunities.values()
            if opp.expiry_time > current_time and opp.status in ['ready', 'analyzing']
        ]
        
        # Sort by profit potential
        return sorted(active_opportunities, key=lambda x: x.profit_percentage, reverse=True)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            summary = {
                'active_symbols': len(set(key.split('_')[0] for key in self.tick_data.keys())),
                'active_exchanges': len(set(key.split('_')[1] for key in self.tick_data.keys())),
                'total_ticks_processed': self.processing_stats['ticks_processed'],
                'active_opportunities': len(self.get_current_opportunities()),
                'total_opportunities_detected': self.processing_stats['opportunities_detected'],
                'average_processing_latency': statistics.mean(self.processing_stats['processing_latency']) if self.processing_stats['processing_latency'] else 0,
                'queue_sizes': self.processing_stats['queue_sizes'],
                'data_feeds_connected': sum(self.feed_connections.values()),
                'current_regime': self.market_regimes[-1] if self.market_regimes else None,
                'system_health': self._calculate_system_health()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health"""
        try:
            health_score = 0
            max_score = 5
            
            # Data feed health
            if sum(self.feed_connections.values()) > 0:
                health_score += 1
            
            # Processing latency health
            if self.processing_stats['processing_latency']:
                avg_latency = statistics.mean(self.processing_stats['processing_latency'])
                if avg_latency < self.config['max_processing_latency']:
                    health_score += 1
            
            # Error rate health
            if self.processing_stats['error_count'] < self.config['error_threshold']:
                health_score += 1
            
            # Queue health
            queue_sizes = self.processing_stats.get('queue_sizes', {})
            if all(size < 1000 for size in queue_sizes.values()):
                health_score += 1
            
            # Data freshness health
            if self.processing_stats['last_tick_time']:
                time_since_last_tick = (datetime.now() - self.processing_stats['last_tick_time']).total_seconds()
                if time_since_last_tick < 60:  # Less than 1 minute
                    health_score += 1
            
            health_ratio = health_score / max_score
            
            if health_ratio >= 0.8:
                return "excellent"
            elif health_ratio >= 0.6:
                return "good"
            elif health_ratio >= 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            self.logger.error(f"Error calculating system health: {str(e)}")
            return "unknown"
    
    async def stop_data_processing(self) -> None:
        """Stop all data processing"""
        try:
            self.is_running = False
            
            # Cancel all processing tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            # Disconnect data feeds
            for feed_name, feed in self.data_feeds.items():
                try:
                    await feed.disconnect()
                except Exception as e:
                    self.logger.warning(f"Error disconnecting feed {feed_name}: {str(e)}")
            
            self.logger.info("Market data processing stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping data processing: {str(e)}")

# Mock data feed for testing
class MockDataFeed(DataFeedInterface):
    """Mock data feed for testing"""
    
    def __init__(self, feed_name: str):
        self.feed_name = feed_name
        self.connected = False
        self.symbols = []
        self.data_manager = None
        self.simulation_task = None
    
    async def connect(self) -> bool:
        self.connected = True
        return True
    
    async def subscribe_ticks(self, symbols: List[str]) -> bool:
        self.symbols.extend(symbols)
        
        # Start simulation
        if not self.simulation_task:
            self.simulation_task = asyncio.create_task(self._simulate_data())
        
        return True
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 10) -> bool:
        return True
    
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Generate mock historical data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
        prices = 50000 + np.cumsum(np.random.randn(limit) * 10)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, limit)
        })
    
    async def _simulate_data(self):
        """Simulate market data"""
        base_prices = {symbol: 50000 + i * 1000 for i, symbol in enumerate(self.symbols)}
        
        while self.connected:
            try:
                for symbol in self.symbols:
                    # Generate mock tick
                    price_change = np.random.randn() * 10
                    base_prices[symbol] += price_change
                    
                    tick = MarketTick(
                        symbol=symbol,
                        exchange=self.feed_name,
                        price=base_prices[symbol],
                        volume=np.random.randint(1, 100),
                        timestamp=datetime.now(),
                        bid=base_prices[symbol] - 0.5,
                        ask=base_prices[symbol] + 0.5,
                        bid_size=np.random.randint(10, 100),
                        ask_size=np.random.randint(10, 100)
                    )
                    
                    if self.data_manager:
                        await self.data_manager.process_tick(tick)
                
                await asyncio.sleep(0.1)  # 10 ticks per second
                
            except Exception as e:
                print(f"Error in data simulation: {str(e)}")
                break
    
    async def disconnect(self) -> bool:
        self.connected = False
        if self.simulation_task:
            self.simulation_task.cancel()
        return True
    
    def set_data_manager(self, data_manager):
        """Set data manager for sending data"""
        self.data_manager = data_manager

# Example usage
if __name__ == "__main__":
    async def test_market_data_manager():
        """Test the market data manager"""
        
        # Initialize manager
        manager = UltimateMarketDataManager()
        
        # Create mock data feeds
        binance_feed = MockDataFeed("binance")
        coinbase_feed = MockDataFeed("coinbase")
        
        binance_feed.set_data_manager(manager)
        coinbase_feed.set_data_manager(manager)
        
        # Add data feeds
        await manager.add_data_feed("binance", binance_feed)
        await manager.add_data_feed("coinbase", coinbase_feed)
        
        # Start processing
        await manager.start_data_processing()
        
        # Subscribe to symbols
        await manager.subscribe_to_symbols(["BTC/USDT", "ETH/USDT"])
        
        # Register callbacks
        async def opportunity_callback(opportunity):
            print(f"Opportunity: {opportunity.symbol} {opportunity.profit_percentage:.4f}% profit")
        
        async def regime_callback(regime):
            print(f"Market Regime: {regime.volatility_regime} volatility, {regime.trend_regime} trend")
        
        manager.register_opportunity_callback(opportunity_callback)
        manager.register_regime_callback(regime_callback)
        
        # Run for a while
        await asyncio.sleep(30)
        
        # Get summary
        summary = manager.get_market_summary()
        print("\nMarket Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get opportunities
        opportunities = manager.get_current_opportunities()
        print(f"\nActive Opportunities: {len(opportunities)}")
        for opp in opportunities[:3]:  # Top 3
            print(f"  {opp.symbol}: {opp.profit_percentage:.4f}% ({opp.confidence_score:.3f} confidence)")
        
        # Stop processing
        await manager.stop_data_processing()
    
    # Run test
    asyncio.run(test_market_data_manager())

