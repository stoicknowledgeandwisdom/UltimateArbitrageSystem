"""Apache Flink-based ETL pipeline for real-time feature processing."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import websockets
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from collections import deque, defaultdict


@dataclass
class FlinkConfig:
    """Configuration for Flink ETL pipeline."""
    checkpoint_interval: int = 5000  # milliseconds
    parallelism: int = 4
    buffer_size: int = 10000
    window_size: int = 60  # seconds
    watermark_delay: int = 5  # seconds
    kafka_bootstrap_servers: str = "localhost:9092"
    output_topic: str = "processed_features"
    max_workers: int = 8


class StreamWindow:
    """Time-based sliding window for stream processing."""
    
    def __init__(self, window_size: int, slide_interval: int = 1):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.data = deque()
        self.timestamps = deque()
        self._lock = threading.Lock()
    
    def add(self, timestamp: float, data: Any):
        """Add data point to window."""
        with self._lock:
            self.data.append(data)
            self.timestamps.append(timestamp)
            
            # Remove old data outside window
            cutoff_time = timestamp - self.window_size
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.timestamps.popleft()
                self.data.popleft()
    
    def get_window_data(self) -> List[Any]:
        """Get current window data."""
        with self._lock:
            return list(self.data)
    
    def get_window_stats(self) -> Dict[str, float]:
        """Get statistical summary of window data."""
        with self._lock:
            if not self.data:
                return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
            # Assume data contains numeric values
            values = []
            for item in self.data:
                if isinstance(item, dict) and 'value' in item:
                    values.append(item['value'])
                elif isinstance(item, (int, float)):
                    values.append(item)
            
            if not values:
                return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            
            values_array = np.array(values)
            return {
                "count": len(values),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array))
            }


class FeatureAggregator:
    """Real-time feature aggregation engine."""
    
    def __init__(self):
        self.windows = {
            '1m': StreamWindow(window_size=60),
            '5m': StreamWindow(window_size=300),
            '15m': StreamWindow(window_size=900)
        }
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
    
    def process_price_data(self, symbol: str, exchange: str, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming price data and compute features."""
        timestamp = datetime.utcnow().timestamp()
        key = f"{symbol}:{exchange}"
        
        # Extract price information
        bid_price = price_data.get('bid_price', 0.0)
        ask_price = price_data.get('ask_price', 0.0)
        mid_price = (bid_price + ask_price) / 2.0 if bid_price > 0 and ask_price > 0 else 0.0
        spread = ask_price - bid_price if ask_price > bid_price else 0.0
        spread_pct = spread / mid_price if mid_price > 0 else 0.0
        
        # Store price history
        with self._lock:
            self.price_history[key].append({
                'timestamp': timestamp,
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct
            })
        
        # Add to time windows
        for window in self.windows.values():
            window.add(timestamp, {'value': mid_price, 'spread': spread})
        
        # Compute features
        features = {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct
        }
        
        # Add window-based features
        features.update(self._compute_price_features(key))
        
        return features
    
    def process_volume_data(self, symbol: str, exchange: str, volume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming volume data and compute features."""
        timestamp = datetime.utcnow().timestamp()
        key = f"{symbol}:{exchange}"
        
        volume = volume_data.get('volume', 0.0)
        bid_volume = volume_data.get('bid_volume', 0.0)
        ask_volume = volume_data.get('ask_volume', 0.0)
        
        # Store volume history
        with self._lock:
            self.volume_history[key].append({
                'timestamp': timestamp,
                'volume': volume,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            })
        
        # Add to time windows
        for window in self.windows.values():
            window.add(timestamp, {'value': volume})
        
        # Compute features
        features = {
            'volume': volume,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'volume_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0
        }
        
        # Add window-based features
        features.update(self._compute_volume_features(key))
        
        return features
    
    def _compute_price_features(self, key: str) -> Dict[str, Any]:
        """Compute price-based features from historical data."""
        features = {}
        
        with self._lock:
            if key not in self.price_history or len(self.price_history[key]) < 2:
                return {
                    'price_volatility_1m': 0.0,
                    'price_volatility_5m': 0.0,
                    'price_change_1m': 0.0,
                    'price_change_5m': 0.0
                }
            
            history = list(self.price_history[key])
            current_time = datetime.utcnow().timestamp()
            
            # Get prices within different time windows
            prices_1m = [h['mid_price'] for h in history if current_time - h['timestamp'] <= 60]
            prices_5m = [h['mid_price'] for h in history if current_time - h['timestamp'] <= 300]
            
            # Calculate volatility (standard deviation)
            if len(prices_1m) > 1:
                features['price_volatility_1m'] = float(np.std(prices_1m))
                features['price_change_1m'] = (prices_1m[-1] - prices_1m[0]) / prices_1m[0] if prices_1m[0] > 0 else 0.0
            else:
                features['price_volatility_1m'] = 0.0
                features['price_change_1m'] = 0.0
            
            if len(prices_5m) > 1:
                features['price_volatility_5m'] = float(np.std(prices_5m))
                features['price_change_5m'] = (prices_5m[-1] - prices_5m[0]) / prices_5m[0] if prices_5m[0] > 0 else 0.0
            else:
                features['price_volatility_5m'] = 0.0
                features['price_change_5m'] = 0.0
        
        return features
    
    def _compute_volume_features(self, key: str) -> Dict[str, Any]:
        """Compute volume-based features from historical data."""
        features = {}
        
        with self._lock:
            if key not in self.volume_history or len(self.volume_history[key]) < 2:
                return {
                    'volume_ma_1m': 0.0,
                    'volume_ma_5m': 0.0,
                    'volume_ratio_1m': 1.0,
                    'volume_spike_indicator': 0.0
                }
            
            history = list(self.volume_history[key])
            current_time = datetime.utcnow().timestamp()
            
            # Get volumes within different time windows
            volumes_1m = [h['volume'] for h in history if current_time - h['timestamp'] <= 60]
            volumes_5m = [h['volume'] for h in history if current_time - h['timestamp'] <= 300]
            
            # Calculate moving averages
            if volumes_1m:
                features['volume_ma_1m'] = float(np.mean(volumes_1m))
                current_volume = volumes_1m[-1] if volumes_1m else 0
                features['volume_ratio_1m'] = current_volume / features['volume_ma_1m'] if features['volume_ma_1m'] > 0 else 1.0
                
                # Volume spike detection (current volume > 2x average)
                features['volume_spike_indicator'] = 1.0 if features['volume_ratio_1m'] > 2.0 else 0.0
            else:
                features['volume_ma_1m'] = 0.0
                features['volume_ratio_1m'] = 1.0
                features['volume_spike_indicator'] = 0.0
            
            if volumes_5m:
                features['volume_ma_5m'] = float(np.mean(volumes_5m))
            else:
                features['volume_ma_5m'] = 0.0
        
        return features


class FlinkETLPipeline:
    """Apache Flink-inspired ETL pipeline for real-time feature processing."""
    
    def __init__(self, config: FlinkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.aggregator = FeatureAggregator()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.input_queue = queue.Queue(maxsize=config.buffer_size)
        self.output_queue = queue.Queue(maxsize=config.buffer_size)
        self.is_running = False
        self.worker_threads = []
        self.processors = []  # List of custom processors
    
    def add_processor(self, processor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add custom data processor to the pipeline."""
        self.processors.append(processor)
    
    async def start(self):
        """Start the ETL pipeline."""
        self.is_running = True
        self.logger.info("Starting Flink ETL Pipeline")
        
        # Start worker threads
        for i in range(self.config.parallelism):
            worker = threading.Thread(target=self._worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        
        # Start output processor
        output_thread = threading.Thread(target=self._output_processor)
        output_thread.daemon = True
        output_thread.start()
        self.worker_threads.append(output_thread)
        
        self.logger.info(f"ETL Pipeline started with {self.config.parallelism} workers")
    
    async def stop(self):
        """Stop the ETL pipeline."""
        self.logger.info("Stopping Flink ETL Pipeline")
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("ETL Pipeline stopped")
    
    async def process_message(self, message: Dict[str, Any]):
        """Add message to processing queue."""
        try:
            if not self.input_queue.full():
                self.input_queue.put(message, block=False)
            else:
                self.logger.warning("Input queue is full, dropping message")
        except Exception as e:
            self.logger.error(f"Failed to queue message: {e}")
    
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing messages."""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get message from queue with timeout
                message = self.input_queue.get(timeout=1.0)
                
                # Process message
                processed_data = self._process_single_message(message)
                
                if processed_data:
                    # Add to output queue
                    if not self.output_queue.full():
                        self.output_queue.put(processed_data, block=False)
                    else:
                        self.logger.warning(f"Worker {worker_id}: Output queue full")
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _process_single_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single message through the pipeline."""
        try:
            # Extract message details
            msg_type = message.get('type', 'unknown')
            symbol = message.get('symbol', '')
            exchange = message.get('exchange', '')
            data = message.get('data', {})
            timestamp = message.get('timestamp', datetime.utcnow().isoformat())
            
            processed_features = {}
            
            # Process based on message type
            if msg_type == 'price':
                processed_features = self.aggregator.process_price_data(symbol, exchange, data)
            elif msg_type == 'volume':
                processed_features = self.aggregator.process_volume_data(symbol, exchange, data)
            elif msg_type == 'trade':
                # Process trade data (extract both price and volume info)
                price_features = self.aggregator.process_price_data(symbol, exchange, {
                    'bid_price': data.get('price', 0) * 0.999,  # Approximate bid
                    'ask_price': data.get('price', 0) * 1.001,  # Approximate ask
                })
                volume_features = self.aggregator.process_volume_data(symbol, exchange, {
                    'volume': data.get('quantity', 0),
                    'bid_volume': data.get('quantity', 0) if data.get('side') == 'buy' else 0,
                    'ask_volume': data.get('quantity', 0) if data.get('side') == 'sell' else 0,
                })
                processed_features = {**price_features, **volume_features}
            
            # Apply custom processors
            for processor in self.processors:
                try:
                    additional_features = processor({
                        'symbol': symbol,
                        'exchange': exchange,
                        'type': msg_type,
                        'data': data,
                        'features': processed_features
                    })
                    if additional_features:
                        processed_features.update(additional_features)
                except Exception as e:
                    self.logger.error(f"Custom processor error: {e}")
            
            # Return processed result
            return {
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': timestamp,
                'features': processed_features,
                'processing_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return None
    
    def _output_processor(self):
        """Process output queue and send to downstream systems."""
        self.logger.info("Output processor started")
        
        while self.is_running:
            try:
                # Get processed data from output queue
                processed_data = self.output_queue.get(timeout=1.0)
                
                # Send to downstream systems (feature store, etc.)
                asyncio.create_task(self._send_to_downstream(processed_data))
                
                self.output_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Output processor error: {e}")
        
        self.logger.info("Output processor stopped")
    
    async def _send_to_downstream(self, processed_data: Dict[str, Any]):
        """Send processed data to downstream systems."""
        try:
            # This would typically send to Kafka, feature store, etc.
            # For now, just log the output
            self.logger.debug(f"Processed features for {processed_data['symbol']}:{processed_data['exchange']}")
            
            # You could add integrations here:
            # - Send to Kafka topic
            # - Push to feature store
            # - Send to monitoring systems
            
        except Exception as e:
            self.logger.error(f"Failed to send to downstream: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        return {
            'is_running': self.is_running,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'worker_count': len(self.worker_threads),
            'processor_count': len(self.processors)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pipeline."""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'stats': self.get_pipeline_stats(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.stop())

