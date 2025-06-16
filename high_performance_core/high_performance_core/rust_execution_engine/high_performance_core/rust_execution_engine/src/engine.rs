//! Ultra-Low Latency Engine Core
//! 
//! Main engine implementation with CQRS, Event Sourcing, and lock-free operations

use std::sync::Arc;
use std::time::{Duration, Instant};
use atomic::{Atomic, Ordering};
use crossbeam_channel::{unbounded, Receiver, Sender};
use dashmap::DashMap;
use tokio::time::{interval, timeout};
use tracing::{info, debug, error, warn, instrument};
use anyhow::{Result, anyhow};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::config::EngineConfig;
use crate::disruptor::{DisruptorRingBuffer, MarketDataEvent};
use crate::event_store::{EventStore, TradingEvent};
use crate::memory_pool::MemoryPool;
use crate::order_book::{OrderBook, OrderBookUpdate};
use crate::market_data::{MarketDataManager, TickData};
use crate::simd_ops::SIMDProcessor;
use crate::metrics::{LatencyRecorder, ThroughputCounter};

/// Command types for CQRS pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingCommand {
    PlaceOrder {
        id: Uuid,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        price: Option<f64>,
        order_type: OrderType,
        timestamp: u64,
    },
    CancelOrder {
        id: Uuid,
        order_id: Uuid,
        timestamp: u64,
    },
    ModifyOrder {
        id: Uuid,
        order_id: Uuid,
        new_quantity: Option<f64>,
        new_price: Option<f64>,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Ultra-low latency execution statistics
#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub messages_processed: Atomic<u64>,
    pub orders_executed: Atomic<u64>,
    pub average_latency_ns: Atomic<u64>,
    pub p99_latency_ns: Atomic<u64>,
    pub throughput_per_sec: Atomic<u64>,
    pub memory_pool_hits: Atomic<u64>,
    pub simd_operations: Atomic<u64>,
}

/// Main ultra-low latency trading engine
pub struct UltraLowLatencyEngine {
    config: EngineConfig,
    
    // Lock-free market data ingestion (Disruptor pattern)
    market_data_disruptor: DisruptorRingBuffer<MarketDataEvent>,
    
    // CQRS Command/Query channels
    command_sender: Sender<TradingCommand>,
    command_receiver: Receiver<TradingCommand>,
    
    // Event sourcing store
    event_store: Arc<EventStore>,
    
    // Order books per symbol (lock-free)
    order_books: Arc<DashMap<String, Arc<OrderBook>>>,
    
    // Memory pool for zero-allocation operations
    memory_pool: Arc<MemoryPool>,
    
    // Market data manager
    market_data_manager: Arc<MarketDataManager>,
    
    // SIMD processor for high-performance calculations
    simd_processor: Arc<SIMDProcessor>,
    
    // Performance metrics
    stats: Arc<ExecutionStats>,
    latency_recorder: Arc<LatencyRecorder>,
    throughput_counter: Arc<ThroughputCounter>,
    
    // Engine state
    is_running: Atomic<bool>,
    shutdown_signal: tokio::sync::broadcast::Sender<()>,
}

impl UltraLowLatencyEngine {
    /// Create new ultra-low latency engine
    #[instrument(skip(config))]
    pub async fn new(config: EngineConfig) -> Result<Self> {
        info!("Initializing Ultra-Low Latency Engine");
        
        // Initialize lock-free command channels
        let (command_sender, command_receiver) = unbounded();
        
        // Initialize event store with NATS JetStream
        let event_store = Arc::new(EventStore::new(&config.nats_url).await?);
        
        // Initialize memory pool
        let memory_pool = Arc::new(MemoryPool::new(
            config.memory_pool_size,
            config.memory_pool_block_size,
        ));
        
        // Initialize market data disruptor
        let market_data_disruptor = DisruptorRingBuffer::new(
            config.disruptor_buffer_size,
        )?;
        
        // Initialize order books
        let order_books = Arc::new(DashMap::new());
        
        // Initialize market data manager
        let market_data_manager = Arc::new(
            MarketDataManager::new(config.clone()).await?
        );
        
        // Initialize SIMD processor
        let simd_processor = Arc::new(SIMDProcessor::new());
        
        // Initialize metrics
        let stats = Arc::new(ExecutionStats::default());
        let latency_recorder = Arc::new(LatencyRecorder::new());
        let throughput_counter = Arc::new(ThroughputCounter::new());
        
        // Create shutdown signal
        let (shutdown_signal, _) = tokio::sync::broadcast::channel(1);
        
        Ok(Self {
            config,
            market_data_disruptor,
            command_sender,
            command_receiver,
            event_store,
            order_books,
            memory_pool,
            market_data_manager,
            simd_processor,
            stats,
            latency_recorder,
            throughput_counter,
            is_running: Atomic::new(false),
            shutdown_signal,
        })
    }
    
    /// Run the engine with all subsystems
    #[instrument(skip(self))]
    pub async fn run(&self) -> Result<()> {
        info!("Starting Ultra-Low Latency Engine execution");
        
        self.is_running.store(true, Ordering::Release);
        
        // Start all subsystems concurrently
        let market_data_task = self.start_market_data_processing();
        let command_processing_task = self.start_command_processing();
        let order_book_maintenance_task = self.start_order_book_maintenance();
        let metrics_task = self.start_metrics_collection();
        let health_check_task = self.start_health_checks();
        
        // Wait for shutdown signal or any task completion
        tokio::select! {
            result = market_data_task => {
                error!("Market data processing completed: {:?}", result);
            }
            result = command_processing_task => {
                error!("Command processing completed: {:?}", result);
            }
            result = order_book_maintenance_task => {
                error!("Order book maintenance completed: {:?}", result);
            }
            result = metrics_task => {
                error!("Metrics collection completed: {:?}", result);
            }
            result = health_check_task => {
                error!("Health checks completed: {:?}", result);
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down gracefully");
                self.shutdown().await?;
            }
        }
        
        Ok(())
    }
    
    /// Start market data processing with Disruptor pattern
    async fn start_market_data_processing(&self) -> Result<()> {
        info!("Starting market data processing with Disruptor pattern");
        
        let mut shutdown_receiver = self.shutdown_signal.subscribe();
        
        loop {
            tokio::select! {
                _ = shutdown_receiver.recv() => {
                    info!("Market data processing shutdown requested");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_nanos(100)) => {
                    // Ultra-fast market data processing loop
                    self.process_market_data_batch().await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Process market data batch with <10μs latency target
    #[instrument(skip(self))]
    async fn process_market_data_batch(&self) -> Result<()> {
        let start_time = Instant::now();
        
        // Process available market data events from disruptor
        let events = self.market_data_disruptor.try_consume_batch(1000)?;
        
        if !events.is_empty() {
            // Use SIMD for batch processing
            let processed_count = self.simd_processor
                .process_market_data_batch(&events)
                .await?;
            
            // Update order books
            for event in events {
                self.update_order_book(event).await?;
            }
            
            // Record performance metrics
            let elapsed = start_time.elapsed();
            self.latency_recorder.record_latency(elapsed);
            self.throughput_counter.add_messages(processed_count as u64);
            
            // Ensure we're meeting <10μs target
            if elapsed.as_nanos() > 10_000 {
                warn!("Market data processing exceeded 10μs: {:?}", elapsed);
            }
        }
        
        Ok(())
    }
    
    /// Start command processing with CQRS pattern
    async fn start_command_processing(&self) -> Result<()> {
        info!("Starting CQRS command processing");
        
        let mut shutdown_receiver = self.shutdown_signal.subscribe();
        
        loop {
            tokio::select! {
                _ = shutdown_receiver.recv() => {
                    info!("Command processing shutdown requested");
                    break;
                }
                command = self.command_receiver.recv_async() => {
                    match command {
                        Ok(cmd) => {
                            if let Err(e) = self.execute_command(cmd).await {
                                error!("Command execution error: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Command receive error: {}", e);
                            break;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute trading command with ultra-low latency
    #[instrument(skip(self))]
    async fn execute_command(&self, command: TradingCommand) -> Result<()> {
        let start_time = Instant::now();
        
        match command {
            TradingCommand::PlaceOrder { 
                id, symbol, side, quantity, price, order_type, timestamp 
            } => {
                self.execute_place_order(
                    id, symbol, side, quantity, price, order_type, timestamp
                ).await?
            }
            TradingCommand::CancelOrder { id, order_id, timestamp } => {
                self.execute_cancel_order(id, order_id, timestamp).await?
            }
            TradingCommand::ModifyOrder { 
                id, order_id, new_quantity, new_price, timestamp 
            } => {
                self.execute_modify_order(
                    id, order_id, new_quantity, new_price, timestamp
                ).await?
            }
        }
        
        // Record execution latency
        let elapsed = start_time.elapsed();
        self.latency_recorder.record_execution_latency(elapsed);
        
        // Ensure sub-50μs end-to-end latency
        if elapsed.as_nanos() > 50_000 {
            warn!("Command execution exceeded 50μs: {:?}", elapsed);
        }
        
        self.stats.orders_executed.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Execute place order command
    async fn execute_place_order(
        &self,
        id: Uuid,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        price: Option<f64>,
        order_type: OrderType,
        timestamp: u64,
    ) -> Result<()> {
        // Get or create order book for symbol
        let order_book = self.order_books
            .entry(symbol.clone())
            .or_insert_with(|| Arc::new(OrderBook::new(symbol.clone())));
        
        // Execute order with memory pool allocation
        let order = self.memory_pool.allocate_order(
            id, symbol, side, quantity, price, order_type, timestamp
        )?;
        
        // Add to order book (lock-free operation)
        order_book.add_order(order).await?;
        
        // Store event for event sourcing
        let event = TradingEvent::OrderPlaced {
            order_id: id,
            symbol: symbol.clone(),
            timestamp,
        };
        
        self.event_store.append_event(event).await?;
        
        debug!("Order placed: {} {} {} @ {:?}", id, symbol, quantity, price);
        
        Ok(())
    }
    
    /// Execute cancel order command
    async fn execute_cancel_order(
        &self,
        id: Uuid,
        order_id: Uuid,
        timestamp: u64,
    ) -> Result<()> {
        // Find order in order books and cancel
        for entry in self.order_books.iter() {
            if entry.value().cancel_order(order_id).await? {
                // Store cancellation event
                let event = TradingEvent::OrderCancelled {
                    order_id,
                    timestamp,
                };
                
                self.event_store.append_event(event).await?;
                
                debug!("Order cancelled: {}", order_id);
                return Ok(());
            }
        }
        
        Err(anyhow!("Order not found: {}", order_id))
    }
    
    /// Execute modify order command
    async fn execute_modify_order(
        &self,
        id: Uuid,
        order_id: Uuid,
        new_quantity: Option<f64>,
        new_price: Option<f64>,
        timestamp: u64,
    ) -> Result<()> {
        // Find and modify order
        for entry in self.order_books.iter() {
            if entry.value().modify_order(order_id, new_quantity, new_price).await? {
                // Store modification event
                let event = TradingEvent::OrderModified {
                    order_id,
                    new_quantity,
                    new_price,
                    timestamp,
                };
                
                self.event_store.append_event(event).await?;
                
                debug!("Order modified: {}", order_id);
                return Ok(());
            }
        }
        
        Err(anyhow!("Order not found: {}", order_id))
    }
    
    /// Update order book from market data event
    async fn update_order_book(&self, event: MarketDataEvent) -> Result<()> {
        let order_book = self.order_books
            .entry(event.symbol.clone())
            .or_insert_with(|| Arc::new(OrderBook::new(event.symbol.clone())));
        
        order_book.apply_market_data(event).await?;
        
        Ok(())
    }
    
    /// Start order book maintenance
    async fn start_order_book_maintenance(&self) -> Result<()> {
        info!("Starting order book maintenance");
        
        let mut interval = interval(Duration::from_millis(100));
        let mut shutdown_receiver = self.shutdown_signal.subscribe();
        
        loop {
            tokio::select! {
                _ = shutdown_receiver.recv() => {
                    info!("Order book maintenance shutdown requested");
                    break;
                }
                _ = interval.tick() => {
                    self.maintain_order_books().await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Maintain order books (cleanup, validation)
    async fn maintain_order_books(&self) -> Result<()> {
        for entry in self.order_books.iter() {
            entry.value().cleanup_expired_orders().await?;
        }
        
        Ok(())
    }
    
    /// Start metrics collection
    async fn start_metrics_collection(&self) -> Result<()> {
        info!("Starting metrics collection");
        
        let mut interval = interval(Duration::from_secs(1));
        let mut shutdown_receiver = self.shutdown_signal.subscribe();
        
        loop {
            tokio::select! {
                _ = shutdown_receiver.recv() => {
                    info!("Metrics collection shutdown requested");
                    break;
                }
                _ = interval.tick() => {
                    self.collect_metrics().await;
                }
            }
        }
        
        Ok(())
    }
    
    /// Collect and update performance metrics
    async fn collect_metrics(&self) {
        let throughput = self.throughput_counter.get_throughput_per_second();
        let avg_latency = self.latency_recorder.get_average_latency_ns();
        let p99_latency = self.latency_recorder.get_p99_latency_ns();
        
        self.stats.throughput_per_sec.store(throughput, Ordering::Relaxed);
        self.stats.average_latency_ns.store(avg_latency, Ordering::Relaxed);
        self.stats.p99_latency_ns.store(p99_latency, Ordering::Relaxed);
        
        // Log performance metrics
        if throughput > 0 {
            info!(
                "Performance: {} msg/sec, avg: {}ns, p99: {}ns",
                throughput, avg_latency, p99_latency
            );
            
            // Check if we're meeting performance targets
            if throughput >= 2_000_000 {
                info!("✓ Throughput target met: {} msg/sec", throughput);
            }
            
            if avg_latency <= 10_000 {
                info!("✓ Latency target met: {} ns average", avg_latency);
            }
            
            if p99_latency <= 50_000 {
                info!("✓ P99 latency target met: {} ns", p99_latency);
            }
        }
    }
    
    /// Start health checks
    async fn start_health_checks(&self) -> Result<()> {
        info!("Starting health checks");
        
        let mut interval = interval(Duration::from_secs(10));
        let mut shutdown_receiver = self.shutdown_signal.subscribe();
        
        loop {
            tokio::select! {
                _ = shutdown_receiver.recv() => {
                    info!("Health checks shutdown requested");
                    break;
                }
                _ = interval.tick() => {
                    self.perform_health_check().await;
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform comprehensive health check
    async fn perform_health_check(&self) {
        let memory_usage = self.memory_pool.get_usage_stats();
        let order_book_count = self.order_books.len();
        let event_store_status = self.event_store.health_check().await;
        
        info!(
            "Health: {} order books, memory: {:.1}%, event store: {}",
            order_book_count,
            memory_usage.utilization_percent,
            if event_store_status { "OK" } else { "ERROR" }
        );
        
        // Check for performance degradation
        let current_throughput = self.stats.throughput_per_sec.load(Ordering::Relaxed);
        let current_latency = self.stats.p99_latency_ns.load(Ordering::Relaxed);
        
        if current_throughput < 1_000_000 && current_throughput > 0 {
            warn!("Throughput below target: {} msg/sec", current_throughput);
        }
        
        if current_latency > 100_000 {
            warn!("P99 latency above target: {} ns", current_latency);
        }
    }
    
    /// Submit trading command (public API)
    pub async fn submit_command(&self, command: TradingCommand) -> Result<()> {
        self.command_sender.send(command)
            .map_err(|e| anyhow!("Failed to submit command: {}", e))?;
        Ok(())
    }
    
    /// Get current performance statistics
    pub fn get_stats(&self) -> ExecutionStats {
        ExecutionStats {
            messages_processed: Atomic::new(
                self.stats.messages_processed.load(Ordering::Relaxed)
            ),
            orders_executed: Atomic::new(
                self.stats.orders_executed.load(Ordering::Relaxed)
            ),
            average_latency_ns: Atomic::new(
                self.stats.average_latency_ns.load(Ordering::Relaxed)
            ),
            p99_latency_ns: Atomic::new(
                self.stats.p99_latency_ns.load(Ordering::Relaxed)
            ),
            throughput_per_sec: Atomic::new(
                self.stats.throughput_per_sec.load(Ordering::Relaxed)
            ),
            memory_pool_hits: Atomic::new(
                self.stats.memory_pool_hits.load(Ordering::Relaxed)
            ),
            simd_operations: Atomic::new(
                self.stats.simd_operations.load(Ordering::Relaxed)
            ),
        }
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Ultra-Low Latency Engine");
        
        self.is_running.store(false, Ordering::Release);
        
        // Send shutdown signal to all tasks
        let _ = self.shutdown_signal.send(());
        
        // Wait for graceful shutdown
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Final metrics report
        let stats = self.get_stats();
        info!("Final stats: {} messages, {} orders, {} msg/sec throughput",
            stats.messages_processed.load(Ordering::Relaxed),
            stats.orders_executed.load(Ordering::Relaxed),
            stats.throughput_per_sec.load(Ordering::Relaxed)
        );
        
        info!("Ultra-Low Latency Engine shutdown complete");
        
        Ok(())
    }
}

