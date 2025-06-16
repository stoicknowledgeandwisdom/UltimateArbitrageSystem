//! Lock-Free Disruptor Ring Buffer Implementation
//! 
//! High-performance, lock-free ring buffer for ultra-low latency market data ingestion
//! Based on the LMAX Disruptor pattern for 2M+ messages/sec throughput

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::mem::{MaybeUninit, size_of};
use std::ptr;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use tracing::{debug, warn, instrument};
use uuid::Uuid;

/// Market data event structure optimized for cache alignment
#[repr(C, align(64))]  // Cache line alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataEvent {
    pub event_id: u64,
    pub timestamp_ns: u64,
    pub symbol: String,
    pub event_type: MarketDataEventType,
    pub price: f64,
    pub quantity: f64,
    pub side: OrderSide,
    pub sequence: u64,
    pub exchange: String,
    pub checksum: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataEventType {
    Trade,
    BidUpdate,
    AskUpdate,
    OrderBookSnapshot,
    OrderBookDelta,
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
    Unknown,
}

/// Lock-free ring buffer slot
#[repr(C, align(64))]
struct RingBufferSlot<T> {
    sequence: AtomicU64,
    data: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T> Sync for RingBufferSlot<T> {}

impl<T> RingBufferSlot<T> {
    fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
    
    #[inline]
    unsafe fn write(&self, value: T, sequence: u64) {
        // Write data first
        ptr::write((*self.data.get()).as_mut_ptr(), value);
        
        // Memory barrier and publish sequence
        std::sync::atomic::fence(Ordering::Release);
        self.sequence.store(sequence, Ordering::Release);
    }
    
    #[inline]
    unsafe fn read(&self, expected_sequence: u64) -> Option<T> {
        // Check if sequence is available
        let current_sequence = self.sequence.load(Ordering::Acquire);
        if current_sequence != expected_sequence {
            return None;
        }
        
        // Read data
        let data = ptr::read((*self.data.get()).as_ptr());
        Some(data)
    }
}

/// Ultra-high performance lock-free ring buffer using Disruptor pattern
pub struct DisruptorRingBuffer<T> {
    buffer: Vec<RingBufferSlot<T>>,
    buffer_size: usize,
    mask: usize,
    
    // Producer sequence
    producer_sequence: AtomicU64,
    
    // Consumer sequence  
    consumer_sequence: AtomicU64,
    
    // Cached consumer sequence for producer
    cached_consumer_sequence: AtomicU64,
    
    // Performance metrics
    messages_published: AtomicU64,
    messages_consumed: AtomicU64,
    contentions: AtomicU64,
}

impl<T> DisruptorRingBuffer<T> {
    /// Create new disruptor ring buffer with specified size (must be power of 2)
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() {
            return Err(anyhow!("Buffer size must be power of 2, got: {}", size));
        }
        
        if size < 64 {
            return Err(anyhow!("Buffer size must be at least 64, got: {}", size));
        }
        
        let buffer = (0..size)
            .map(|_| RingBufferSlot::new())
            .collect();
        
        Ok(Self {
            buffer,
            buffer_size: size,
            mask: size - 1,
            producer_sequence: AtomicU64::new(0),
            consumer_sequence: AtomicU64::new(0),
            cached_consumer_sequence: AtomicU64::new(0),
            messages_published: AtomicU64::new(0),
            messages_consumed: AtomicU64::new(0),
            contentions: AtomicU64::new(0),
        })
    }
    
    /// Publish single event to ring buffer (lock-free)
    #[instrument(skip(self, event))]
    pub fn publish(&self, event: T) -> Result<u64> {
        // Get next producer sequence
        let next_sequence = self.producer_sequence.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Check if buffer is full (wait-free check)
        let cached_consumer = self.cached_consumer_sequence.load(Ordering::Acquire);
        if next_sequence > cached_consumer + self.buffer_size as u64 {
            // Update cached consumer sequence
            let current_consumer = self.consumer_sequence.load(Ordering::Acquire);
            self.cached_consumer_sequence.store(current_consumer, Ordering::Release);
            
            // Check again with updated consumer sequence
            if next_sequence > current_consumer + self.buffer_size as u64 {
                self.contentions.fetch_add(1, Ordering::Relaxed);
                return Err(anyhow!("Ring buffer full: producer at {}, consumer at {}", 
                    next_sequence, current_consumer));
            }
        }
        
        // Get buffer slot
        let slot_index = (next_sequence - 1) & self.mask as u64;
        let slot = &self.buffer[slot_index as usize];
        
        // Write event to slot
        unsafe {
            slot.write(event, next_sequence);
        }
        
        self.messages_published.fetch_add(1, Ordering::Relaxed);
        
        debug!("Published event at sequence: {}", next_sequence);
        
        Ok(next_sequence)
    }
    
    /// Try to publish event without blocking
    pub fn try_publish(&self, event: T) -> Result<Option<u64>> {
        match self.publish(event) {
            Ok(seq) => Ok(Some(seq)),
            Err(_) => Ok(None), // Buffer full, don't error
        }
    }
    
    /// Consume single event from ring buffer (lock-free)
    #[instrument(skip(self))]
    pub fn consume(&self) -> Result<Option<T>> {
        let current_sequence = self.consumer_sequence.load(Ordering::Relaxed);
        let next_sequence = current_sequence + 1;
        
        // Get buffer slot
        let slot_index = (current_sequence) & self.mask as u64;
        let slot = &self.buffer[slot_index as usize];
        
        // Try to read event
        let event = unsafe { slot.read(next_sequence) };
        
        if let Some(event) = event {
            // Update consumer sequence
            self.consumer_sequence.store(next_sequence, Ordering::Release);
            self.messages_consumed.fetch_add(1, Ordering::Relaxed);
            
            debug!("Consumed event at sequence: {}", next_sequence);
            
            Ok(Some(event))
        } else {
            Ok(None) // No new events available
        }
    }
    
    /// Try to consume batch of events for better throughput
    #[instrument(skip(self))]
    pub fn try_consume_batch(&self, max_batch_size: usize) -> Result<Vec<T>> {
        let mut events = Vec::with_capacity(max_batch_size);
        
        for _ in 0..max_batch_size {
            match self.consume()? {
                Some(event) => events.push(event),
                None => break, // No more events available
            }
        }
        
        Ok(events)
    }
    
    /// Get current buffer utilization (0.0 to 1.0)
    pub fn get_utilization(&self) -> f64 {
        let producer_seq = self.producer_sequence.load(Ordering::Relaxed);
        let consumer_seq = self.consumer_sequence.load(Ordering::Relaxed);
        
        if producer_seq >= consumer_seq {
            let used_slots = producer_seq - consumer_seq;
            used_slots as f64 / self.buffer_size as f64
        } else {
            0.0
        }
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> DisruptorStats {
        DisruptorStats {
            buffer_size: self.buffer_size,
            messages_published: self.messages_published.load(Ordering::Relaxed),
            messages_consumed: self.messages_consumed.load(Ordering::Relaxed),
            contentions: self.contentions.load(Ordering::Relaxed),
            utilization: self.get_utilization(),
            producer_sequence: self.producer_sequence.load(Ordering::Relaxed),
            consumer_sequence: self.consumer_sequence.load(Ordering::Relaxed),
        }
    }
    
    /// Reset all sequences (use with caution)
    pub fn reset(&self) {
        self.producer_sequence.store(0, Ordering::Release);
        self.consumer_sequence.store(0, Ordering::Release);
        self.cached_consumer_sequence.store(0, Ordering::Release);
        self.messages_published.store(0, Ordering::Release);
        self.messages_consumed.store(0, Ordering::Release);
        self.contentions.store(0, Ordering::Release);
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let producer_seq = self.producer_sequence.load(Ordering::Relaxed);
        let consumer_seq = self.consumer_sequence.load(Ordering::Relaxed);
        producer_seq == consumer_seq
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.get_utilization() >= 1.0
    }
    
    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        let utilization = self.get_utilization();
        ((1.0 - utilization) * self.buffer_size as f64) as usize
    }
}

/// Performance statistics for the disruptor
#[derive(Debug, Clone)]
pub struct DisruptorStats {
    pub buffer_size: usize,
    pub messages_published: u64,
    pub messages_consumed: u64,
    pub contentions: u64,
    pub utilization: f64,
    pub producer_sequence: u64,
    pub consumer_sequence: u64,
}

impl DisruptorStats {
    /// Get throughput in messages per second
    pub fn get_throughput(&self, duration_secs: f64) -> f64 {
        if duration_secs > 0.0 {
            self.messages_consumed as f64 / duration_secs
        } else {
            0.0
        }
    }
    
    /// Get contention rate
    pub fn get_contention_rate(&self) -> f64 {
        if self.messages_published > 0 {
            self.contentions as f64 / self.messages_published as f64
        } else {
            0.0
        }
    }
}

/// High-performance market data producer
pub struct MarketDataProducer {
    disruptor: Arc<DisruptorRingBuffer<MarketDataEvent>>,
    event_id_counter: AtomicU64,
}

impl MarketDataProducer {
    pub fn new(disruptor: Arc<DisruptorRingBuffer<MarketDataEvent>>) -> Self {
        Self {
            disruptor,
            event_id_counter: AtomicU64::new(0),
        }
    }
    
    /// Publish market data event with automatic ID generation
    pub fn publish_market_data(
        &self,
        symbol: String,
        event_type: MarketDataEventType,
        price: f64,
        quantity: f64,
        side: OrderSide,
        exchange: String,
    ) -> Result<u64> {
        let event_id = self.event_id_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let event = MarketDataEvent {
            event_id,
            timestamp_ns,
            symbol: symbol.clone(),
            event_type,
            price,
            quantity,
            side,
            sequence: event_id,
            exchange,
            checksum: Self::calculate_checksum(&symbol, price, quantity),
        };
        
        self.disruptor.publish(event)
    }
    
    /// Calculate simple checksum for data integrity
    fn calculate_checksum(symbol: &str, price: f64, quantity: f64) -> u32 {
        let mut hash = 0u32;
        for byte in symbol.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash = hash.wrapping_add((price * 10000.0) as u32);
        hash = hash.wrapping_add((quantity * 10000.0) as u32);
        hash
    }
    
    /// Batch publish multiple events for higher throughput
    pub fn publish_batch(
        &self,
        events: Vec<(String, MarketDataEventType, f64, f64, OrderSide, String)>
    ) -> Result<Vec<u64>> {
        let mut sequences = Vec::with_capacity(events.len());
        
        for (symbol, event_type, price, quantity, side, exchange) in events {
            let seq = self.publish_market_data(
                symbol, event_type, price, quantity, side, exchange
            )?;
            sequences.push(seq);
        }
        
        Ok(sequences)
    }
    
    /// Get disruptor statistics
    pub fn get_stats(&self) -> DisruptorStats {
        self.disruptor.get_stats()
    }
}

/// High-performance market data consumer
pub struct MarketDataConsumer {
    disruptor: Arc<DisruptorRingBuffer<MarketDataEvent>>,
    events_processed: AtomicU64,
    last_sequence: AtomicU64,
}

impl MarketDataConsumer {
    pub fn new(disruptor: Arc<DisruptorRingBuffer<MarketDataEvent>>) -> Self {
        Self {
            disruptor,
            events_processed: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
        }
    }
    
    /// Consume single event
    pub fn consume_event(&self) -> Result<Option<MarketDataEvent>> {
        let event = self.disruptor.consume()?;
        
        if event.is_some() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(event)
    }
    
    /// Consume batch of events for higher throughput
    pub fn consume_batch(&self, max_batch_size: usize) -> Result<Vec<MarketDataEvent>> {
        let events = self.disruptor.try_consume_batch(max_batch_size)?;
        
        if !events.is_empty() {
            self.events_processed.fetch_add(events.len() as u64, Ordering::Relaxed);
        }
        
        Ok(events)
    }
    
    /// Get number of events processed
    pub fn get_events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }
    
    /// Validate event integrity
    pub fn validate_event(&self, event: &MarketDataEvent) -> bool {
        let expected_checksum = MarketDataProducer::calculate_checksum(
            &event.symbol, event.price, event.quantity
        );
        event.checksum == expected_checksum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_disruptor_single_producer_consumer() {
        let disruptor = Arc::new(DisruptorRingBuffer::new(1024).unwrap());
        
        // Publish events
        for i in 0..100 {
            let event = MarketDataEvent {
                event_id: i,
                timestamp_ns: i * 1000,
                symbol: "BTCUSD".to_string(),
                event_type: MarketDataEventType::Trade,
                price: 50000.0 + i as f64,
                quantity: 1.0,
                side: OrderSide::Buy,
                sequence: i,
                exchange: "test".to_string(),
                checksum: 0,
            };
            
            disruptor.publish(event).unwrap();
        }
        
        // Consume events
        for i in 0..100 {
            let event = disruptor.consume().unwrap().unwrap();
            assert_eq!(event.event_id, i);
            assert_eq!(event.price, 50000.0 + i as f64);
        }
        
        // Should be empty now
        assert!(disruptor.consume().unwrap().is_none());
    }
    
    #[test]
    fn test_disruptor_batch_operations() {
        let disruptor = Arc::new(DisruptorRingBuffer::new(1024).unwrap());
        
        // Publish batch
        for i in 0..500 {
            let event = MarketDataEvent {
                event_id: i,
                timestamp_ns: i * 1000,
                symbol: "ETHUSD".to_string(),
                event_type: MarketDataEventType::BidUpdate,
                price: 3000.0 + i as f64,
                quantity: 2.0,
                side: OrderSide::Sell,
                sequence: i,
                exchange: "test".to_string(),
                checksum: 0,
            };
            
            disruptor.publish(event).unwrap();
        }
        
        // Consume in batches
        let mut total_consumed = 0;
        while total_consumed < 500 {
            let batch = disruptor.try_consume_batch(100).unwrap();
            total_consumed += batch.len();
            
            if batch.is_empty() {
                break;
            }
        }
        
        assert_eq!(total_consumed, 500);
    }
    
    #[test]
    fn test_disruptor_performance_stats() {
        let disruptor = Arc::new(DisruptorRingBuffer::new(256).unwrap());
        
        // Initial stats
        let stats = disruptor.get_stats();
        assert_eq!(stats.messages_published, 0);
        assert_eq!(stats.messages_consumed, 0);
        assert_eq!(stats.utilization, 0.0);
        
        // Publish some events
        for i in 0..100 {
            let event = MarketDataEvent {
                event_id: i,
                timestamp_ns: i * 1000,
                symbol: "TEST".to_string(),
                event_type: MarketDataEventType::Trade,
                price: 100.0,
                quantity: 1.0,
                side: OrderSide::Buy,
                sequence: i,
                exchange: "test".to_string(),
                checksum: 0,
            };
            
            disruptor.publish(event).unwrap();
        }
        
        let stats_after_publish = disruptor.get_stats();
        assert_eq!(stats_after_publish.messages_published, 100);
        assert!(stats_after_publish.utilization > 0.0);
        
        // Consume all events
        while disruptor.consume().unwrap().is_some() {}
        
        let final_stats = disruptor.get_stats();
        assert_eq!(final_stats.messages_consumed, 100);
        assert_eq!(final_stats.utilization, 0.0);
    }
}

