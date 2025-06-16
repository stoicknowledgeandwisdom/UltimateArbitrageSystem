"""Historical Orderbook Replayer with Nanosecond Granularity"""

import asyncio
import gzip
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import heapq
from enum import Enum
import logging


class EventType(Enum):
    """Event types for orderbook replay"""
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE = "trade"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    MARKET_DATA_UPDATE = "market_data_update"


@dataclass
class OrderbookEvent:
    """Single orderbook event with nanosecond precision"""
    timestamp_ns: int  # Nanoseconds since epoch
    event_type: EventType
    symbol: str
    exchange: str
    data: Dict[str, Any]
    sequence_number: Optional[int] = None
    
    @property
    def timestamp_ms(self) -> int:
        return self.timestamp_ns // 1_000_000
    
    @property
    def timestamp_us(self) -> int:
        return self.timestamp_ns // 1_000
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ns / 1_000_000_000)


@dataclass
class OrderbookSnapshot:
    """Full orderbook snapshot"""
    timestamp_ns: int
    symbol: str
    exchange: str
    bids: List[List[float]]  # [[price, amount], ...]
    asks: List[List[float]]  # [[price, amount], ...]
    sequence_number: Optional[int] = None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price (best bid + best ask) / 2"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None
    
    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points"""
        spread = self.get_spread()
        mid = self.get_mid_price()
        if spread is not None and mid is not None and mid > 0:
            return (spread / mid) * 10000
        return None


class OrderbookReplayer:
    """High-performance orderbook replayer with nanosecond precision"""
    
    def __init__(self, 
                 data_path: Path,
                 buffer_size: int = 10000,
                 compression: bool = True):
        self.data_path = Path(data_path)
        self.buffer_size = buffer_size
        self.compression = compression
        self.logger = logging.getLogger(__name__)
        
        # Event queue (priority queue by timestamp)
        self.event_queue: List[OrderbookEvent] = []
        self.current_orderbooks: Dict[str, OrderbookSnapshot] = {}
        
        # Statistics
        self.events_processed = 0
        self.start_time = None
        self.end_time = None
        
        # Callbacks
        self.event_callbacks: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.latency_stats = {
            'min_ns': float('inf'),
            'max_ns': 0,
            'total_ns': 0,
            'count': 0
        }
    
    def register_callback(self, event_type: EventType, callback: Callable):
        """Register callback for specific event type"""
        self.event_callbacks[event_type].append(callback)
    
    async def load_historical_data(self, 
                                  symbol: str,
                                  exchange: str,
                                  start_time: datetime,
                                  end_time: datetime) -> None:
        """Load historical orderbook data from files"""
        self.logger.info(f"Loading historical data for {symbol} on {exchange}")
        
        # Generate file paths based on date range
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            file_path = self._get_data_file_path(symbol, exchange, current_date)
            if file_path.exists():
                await self._load_data_file(file_path, start_time, end_time)
            current_date += timedelta(days=1)
        
        # Sort events by timestamp
        heapq.heapify(self.event_queue)
        self.logger.info(f"Loaded {len(self.event_queue)} events")
    
    def _get_data_file_path(self, symbol: str, exchange: str, date) -> Path:
        """Get file path for specific symbol, exchange, and date"""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{exchange}_{symbol.replace('/', '_')}_{date_str}.jsonl"
        if self.compression:
            filename += ".gz"
        return self.data_path / exchange / filename
    
    async def _load_data_file(self, 
                             file_path: Path,
                             start_time: datetime,
                             end_time: datetime) -> None:
        """Load data from a single file"""
        self.logger.debug(f"Loading file: {file_path}")
        
        start_ns = int(start_time.timestamp() * 1_000_000_000)
        end_ns = int(end_time.timestamp() * 1_000_000_000)
        
        opener = gzip.open if self.compression else open
        mode = 'rt' if self.compression else 'r'
        
        with opener(file_path, mode) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    timestamp_ns = data['timestamp_ns']
                    
                    # Filter by time range
                    if start_ns <= timestamp_ns <= end_ns:
                        event = OrderbookEvent(
                            timestamp_ns=timestamp_ns,
                            event_type=EventType(data['event_type']),
                            symbol=data['symbol'],
                            exchange=data['exchange'],
                            data=data['data'],
                            sequence_number=data.get('sequence_number')
                        )
                        heapq.heappush(self.event_queue, 
                                     (timestamp_ns, len(self.event_queue), event))
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Skipping invalid line: {e}")
    
    async def replay(self, 
                    speed_multiplier: float = 1.0,
                    real_time: bool = True) -> AsyncGenerator[OrderbookEvent, None]:
        """Replay events with timing control"""
        if not self.event_queue:
            raise ValueError("No events loaded. Call load_historical_data first.")
        
        self.logger.info(f"Starting replay of {len(self.event_queue)} events")
        self.start_time = time.time_ns()
        
        first_event_time = None
        replay_start_time = time.time_ns()
        
        while self.event_queue:
            timestamp_ns, _, event = heapq.heappop(self.event_queue)
            
            if first_event_time is None:
                first_event_time = timestamp_ns
            
            # Calculate timing for real-time replay
            if real_time and speed_multiplier > 0:
                elapsed_historical_ns = timestamp_ns - first_event_time
                elapsed_replay_ns = time.time_ns() - replay_start_time
                target_elapsed_ns = elapsed_historical_ns / speed_multiplier
                
                if target_elapsed_ns > elapsed_replay_ns:
                    sleep_time = (target_elapsed_ns - elapsed_replay_ns) / 1_000_000_000
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Process event
            process_start = time.time_ns()
            await self._process_event(event)
            process_end = time.time_ns()
            
            # Update latency statistics
            latency_ns = process_end - process_start
            self._update_latency_stats(latency_ns)
            
            self.events_processed += 1
            yield event
        
        self.end_time = time.time_ns()
        self.logger.info(f"Replay completed. Processed {self.events_processed} events")
    
    async def _process_event(self, event: OrderbookEvent) -> None:
        """Process individual event and update internal state"""
        key = f"{event.exchange}_{event.symbol}"
        
        if event.event_type == EventType.ORDERBOOK_UPDATE:
            # Update orderbook snapshot
            if 'full_snapshot' in event.data:
                snapshot_data = event.data['full_snapshot']
                self.current_orderbooks[key] = OrderbookSnapshot(
                    timestamp_ns=event.timestamp_ns,
                    symbol=event.symbol,
                    exchange=event.exchange,
                    bids=snapshot_data['bids'],
                    asks=snapshot_data['asks'],
                    sequence_number=event.sequence_number
                )
            elif 'updates' in event.data:
                # Apply incremental updates
                if key in self.current_orderbooks:
                    self._apply_orderbook_updates(key, event.data['updates'])
        
        # Call registered callbacks
        for callback in self.event_callbacks[event.event_type]:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def _apply_orderbook_updates(self, key: str, updates: List[Dict]) -> None:
        """Apply incremental orderbook updates"""
        orderbook = self.current_orderbooks[key]
        
        for update in updates:
            side = update['side']  # 'bid' or 'ask'
            price = update['price']
            amount = update['amount']
            
            if side == 'bid':
                levels = orderbook.bids
            else:
                levels = orderbook.asks
            
            # Find and update level
            for i, (level_price, _) in enumerate(levels):
                if abs(level_price - price) < 1e-8:  # Price match
                    if amount == 0:
                        # Remove level
                        levels.pop(i)
                    else:
                        # Update amount
                        levels[i][1] = amount
                    break
            else:
                # Add new level if amount > 0
                if amount > 0:
                    levels.append([price, amount])
                    # Keep sorted (descending for bids, ascending for asks)
                    if side == 'bid':
                        levels.sort(key=lambda x: x[0], reverse=True)
                    else:
                        levels.sort(key=lambda x: x[0])
    
    def _update_latency_stats(self, latency_ns: int) -> None:
        """Update latency statistics"""
        stats = self.latency_stats
        stats['min_ns'] = min(stats['min_ns'], latency_ns)
        stats['max_ns'] = max(stats['max_ns'], latency_ns)
        stats['total_ns'] += latency_ns
        stats['count'] += 1
    
    def get_current_orderbook(self, symbol: str, exchange: str) -> Optional[OrderbookSnapshot]:
        """Get current orderbook state"""
        key = f"{exchange}_{symbol}"
        return self.current_orderbooks.get(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics"""
        stats = self.latency_stats
        avg_latency_ns = stats['total_ns'] / stats['count'] if stats['count'] > 0 else 0
        
        total_time_ns = (self.end_time or time.time_ns()) - (self.start_time or 0)
        events_per_second = self.events_processed / (total_time_ns / 1_000_000_000) if total_time_ns > 0 else 0
        
        return {
            'events_processed': self.events_processed,
            'total_time_seconds': total_time_ns / 1_000_000_000,
            'events_per_second': events_per_second,
            'latency_stats': {
                'min_us': stats['min_ns'] / 1000,
                'max_us': stats['max_ns'] / 1000,
                'avg_us': avg_latency_ns / 1000,
                'count': stats['count']
            },
            'orderbooks_tracked': len(self.current_orderbooks)
        }


class HistoricalDataCollector:
    """Collect and store historical orderbook data"""
    
    def __init__(self, output_path: Path, compression: bool = True):
        self.output_path = Path(output_path)
        self.compression = compression
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    async def collect_live_data(self,
                               symbol: str,
                               exchange: str,
                               duration_hours: int = 24) -> None:
        """Collect live orderbook data for specified duration"""
        end_time = datetime.now() + timedelta(hours=duration_hours)
        current_date = datetime.now().date()
        
        # Open output file
        file_path = self._get_output_file_path(symbol, exchange, current_date)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        opener = gzip.open if self.compression else open
        mode = 'wt' if self.compression else 'w'
        
        self.logger.info(f"Starting data collection for {symbol} on {exchange}")
        
        with opener(file_path, mode) as f:
            # Start collecting data (this would connect to real exchange APIs)
            async for event_data in self._collect_from_exchange(symbol, exchange, end_time):
                f.write(json.dumps(event_data) + '\n')
                f.flush()
        
        self.logger.info(f"Data collection completed. File: {file_path}")
    
    def _get_output_file_path(self, symbol: str, exchange: str, date) -> Path:
        """Get output file path"""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{exchange}_{symbol.replace('/', '_')}_{date_str}.jsonl"
        if self.compression:
            filename += ".gz"
        return self.output_path / exchange / filename
    
    async def _collect_from_exchange(self, 
                                   symbol: str, 
                                   exchange: str,
                                   end_time: datetime) -> AsyncGenerator[Dict, None]:
        """Collect data from exchange (mock implementation)"""
        # This would connect to real exchange WebSocket APIs
        # For now, generate synthetic data
        
        sequence_number = 0
        base_price = 50000.0
        
        while datetime.now() < end_time:
            # Generate synthetic orderbook update
            timestamp_ns = time.time_ns()
            
            # Generate realistic orderbook data
            bids = []
            asks = []
            
            for i in range(20):
                bid_price = base_price - (i + 1) * 0.01
                ask_price = base_price + (i + 1) * 0.01
                
                bid_amount = np.random.exponential(1.0)
                ask_amount = np.random.exponential(1.0)
                
                bids.append([bid_price, bid_amount])
                asks.append([ask_price, ask_amount])
            
            event_data = {
                'timestamp_ns': timestamp_ns,
                'event_type': EventType.ORDERBOOK_UPDATE.value,
                'symbol': symbol,
                'exchange': exchange,
                'sequence_number': sequence_number,
                'data': {
                    'full_snapshot': {
                        'bids': bids,
                        'asks': asks
                    }
                }
            }
            
            yield event_data
            
            sequence_number += 1
            base_price += np.random.normal(0, 0.1)  # Random walk
            
            # Wait for next update (100ms)
            await asyncio.sleep(0.1)

