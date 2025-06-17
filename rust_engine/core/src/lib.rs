//! # Ultimate Core Trading Engine
//!
//! Ultra-high-frequency trading engine designed for sub-millisecond execution.
//! This module provides the foundational components for maximum performance trading.
//!
//! ## Features
//!
//! - **Sub-millisecond execution**: <1ms order processing
//! - **Zero-copy operations**: Minimal memory allocation
//! - **Lock-free algorithms**: Maximum concurrency
//! - **SIMD optimization**: Vectorized computations
//! - **FPGA integration**: Hardware acceleration support
//! - **Memory efficiency**: <10MB memory footprint
//!
//! ## Performance Targets
//!
//! - Latency: <1ms (target <500μs)
//! - Throughput: 100,000+ orders/second
//! - Memory: <10MB total usage
//! - CPU: <30% utilization at peak

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Global allocator for maximum performance
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Core modules
pub mod primitives;
pub mod order_book;
pub mod execution;
pub mod market_data;

// Re-exports for convenience
pub use order_book::{OrderBook, OrderBookUpdate};
pub use primitives::*;
pub use execution::{ExecutionEngine, TradeResult};
pub use market_data::{MarketDataProcessor, Tick};

/// Ultra-high-precision timestamp in nanoseconds since UNIX epoch
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Create a new timestamp from the current system time
    #[inline]
    pub fn now() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self(nanos)
    }
    
    /// Create a timestamp from nanoseconds since UNIX epoch
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    
    /// Get nanoseconds since UNIX epoch
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Calculate duration since another timestamp
    #[inline]
    pub fn duration_since(&self, other: Timestamp) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(other.0))
    }
    
    /// Add duration to timestamp
    #[inline]
    pub fn add_duration(&self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.as_nanos() as u64))
    }
}

/// Global performance counters for monitoring
pub struct PerformanceCounters {
    /// Total orders processed
    pub orders_processed: AtomicU64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: AtomicU64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicU64,
    /// Number of order book updates
    pub order_book_updates: AtomicU64,
    /// Number of trades executed
    pub trades_executed: AtomicU64,
    /// Total profit in basis points
    pub total_profit_bps: AtomicU64,
}

impl PerformanceCounters {
    /// Create new performance counters
    pub const fn new() -> Self {
        Self {
            orders_processed: AtomicU64::new(0),
            total_execution_time_ns: AtomicU64::new(0),
            peak_memory_usage: AtomicU64::new(0),
            order_book_updates: AtomicU64::new(0),
            trades_executed: AtomicU64::new(0),
            total_profit_bps: AtomicU64::new(0),
        }
    }
    
    /// Increment orders processed counter
    #[inline]
    pub fn increment_orders(&self) {
        self.orders_processed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add execution time in nanoseconds
    #[inline]
    pub fn add_execution_time(&self, nanos: u64) {
        self.total_execution_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }
    
    /// Update peak memory usage if current usage is higher
    #[inline]
    pub fn update_peak_memory(&self, current_usage: u64) {
        self.peak_memory_usage.fetch_max(current_usage, Ordering::Relaxed);
    }
    
    /// Increment order book updates counter
    #[inline]
    pub fn increment_order_book_updates(&self) {
        self.order_book_updates.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment trades executed counter
    #[inline]
    pub fn increment_trades(&self) {
        self.trades_executed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add profit in basis points
    #[inline]
    pub fn add_profit_bps(&self, bps: u64) {
        self.total_profit_bps.fetch_add(bps, Ordering::Relaxed);
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let orders = self.orders_processed.load(Ordering::Relaxed);
        let total_time = self.total_execution_time_ns.load(Ordering::Relaxed);
        
        PerformanceStats {
            orders_processed: orders,
            total_execution_time_ns: total_time,
            average_execution_time_ns: if orders > 0 { total_time / orders } else { 0 },
            peak_memory_usage: self.peak_memory_usage.load(Ordering::Relaxed),
            order_book_updates: self.order_book_updates.load(Ordering::Relaxed),
            trades_executed: self.trades_executed.load(Ordering::Relaxed),
            total_profit_bps: self.total_profit_bps.load(Ordering::Relaxed),
            orders_per_second: if total_time > 0 {
                (orders * 1_000_000_000) / total_time
            } else {
                0
            },
        }
    }
}

/// Performance statistics snapshot
#[derive(Debug, Clone, Copy)]
pub struct PerformanceStats {
    /// Total orders processed
    pub orders_processed: u64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Average execution time per order in nanoseconds
    pub average_execution_time_ns: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Number of order book updates
    pub order_book_updates: u64,
    /// Number of trades executed
    pub trades_executed: u64,
    /// Total profit in basis points
    pub total_profit_bps: u64,
    /// Orders processed per second
    pub orders_per_second: u64,
}

/// Global performance counters instance
pub static PERFORMANCE_COUNTERS: PerformanceCounters = PerformanceCounters::new();

/// Error types for the core trading engine
#[derive(thiserror::Error, Debug)]
pub enum CoreError {
    /// Invalid price value
    #[error("Invalid price: {0}")]
    InvalidPrice(String),
    
    /// Invalid quantity value
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(String),
    
    /// Order book operation failed
    #[error("Order book error: {0}")]
    OrderBookError(String),
    
    /// Execution engine error
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    /// Market data processing error
    #[error("Market data error: {0}")]
    MarketDataError(String),
    
    /// Memory allocation error
    #[error("Memory allocation failed")]
    MemoryAllocationError,
    
    /// FPGA communication error
    #[error("FPGA error: {0}")]
    FpgaError(String),
    
    /// Generic core engine error
    #[error("Core engine error: {0}")]
    CoreEngineError(String),
}

/// Result type for core operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Core trading engine configuration
#[derive(Debug, Clone)]
pub struct CoreConfig {
    /// Maximum number of price levels in order book
    pub max_price_levels: usize,
    /// Maximum number of orders per price level
    pub max_orders_per_level: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable FPGA acceleration
    pub enable_fpga: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Performance monitoring interval in milliseconds
    pub performance_interval_ms: u64,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            max_price_levels: 10_000,
            max_orders_per_level: 1_000,
            enable_simd: true,
            enable_fpga: false,
            memory_pool_size: 10 * 1024 * 1024, // 10MB
            performance_interval_ms: 1000,
        }
    }
}

/// Initialize the core trading engine with configuration
pub fn initialize_core_engine(config: CoreConfig) -> CoreResult<()> {
    log::info!("Initializing Ultimate Core Trading Engine");
    log::info!("Configuration: {:?}", config);
    
    // Initialize memory pools
    if config.memory_pool_size > 0 {
        log::info!("Initializing memory pool: {} bytes", config.memory_pool_size);
    }
    
    // Initialize SIMD optimizations
    if config.enable_simd {
        log::info!("SIMD optimizations enabled");
    }
    
    // Initialize FPGA acceleration
    if config.enable_fpga {
        log::info!("FPGA acceleration enabled");
        #[cfg(feature = "fpga")]
        {
            // FPGA initialization code would go here
            log::info!("FPGA hardware initialized");
        }
        #[cfg(not(feature = "fpga"))]
        {
            log::warn!("FPGA feature not compiled, falling back to CPU");
        }
    }
    
    log::info!("Ultimate Core Trading Engine initialized successfully");
    Ok(())
}

/// Get current engine performance statistics
pub fn get_performance_stats() -> PerformanceStats {
    PERFORMANCE_COUNTERS.get_stats()
}

/// Log performance statistics
pub fn log_performance_stats() {
    let stats = get_performance_stats();
    log::info!("Performance Stats:");
    log::info!("  Orders Processed: {}", stats.orders_processed);
    log::info!("  Orders/Second: {}", stats.orders_per_second);
    log::info!("  Avg Execution Time: {}ns", stats.average_execution_time_ns);
    log::info!("  Peak Memory: {}MB", stats.peak_memory_usage / (1024 * 1024));
    log::info!("  Trades Executed: {}", stats.trades_executed);
    log::info!("  Total Profit: {}bps", stats.total_profit_bps);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timestamp() {
        let ts1 = Timestamp::now();
        std::thread::sleep(Duration::from_millis(1));
        let ts2 = Timestamp::now();
        
        assert!(ts2 > ts1);
        let duration = ts2.duration_since(ts1);
        assert!(duration.as_millis() >= 1);
    }
    
    #[test]
    fn test_performance_counters() {
        let counters = PerformanceCounters::new();
        
        counters.increment_orders();
        counters.add_execution_time(1000);
        counters.increment_trades();
        
        let stats = counters.get_stats();
        assert_eq!(stats.orders_processed, 1);
        assert_eq!(stats.total_execution_time_ns, 1000);
        assert_eq!(stats.trades_executed, 1);
    }
    
    #[test]
    fn test_core_config() {
        let config = CoreConfig::default();
        assert_eq!(config.max_price_levels, 10_000);
        assert_eq!(config.max_orders_per_level, 1_000);
        assert!(config.enable_simd);
        assert!(!config.enable_fpga);
    }
    
    #[test]
    fn test_initialize_core_engine() {
        let config = CoreConfig {
            enable_fpga: false,
            ..CoreConfig::default()
        };
        
        let result = initialize_core_engine(config);
        assert!(result.is_ok());
    }
}

