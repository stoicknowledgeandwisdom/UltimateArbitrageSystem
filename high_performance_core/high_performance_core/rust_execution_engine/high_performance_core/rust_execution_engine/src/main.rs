//! Ultra-Low Latency Trading Execution Engine
//! 
//! This engine is designed for <10μs latency with 2M messages/sec throughput
//! Features:
//! - Lock-free ring buffers for market data ingestion (Disruptor pattern)
//! - CQRS + Event Sourcing with NATS JetStream
//! - SIMD/GPU acceleration for order book diffs
//! - Memory pooling with zero GC pauses
//! - p99 end-to-end latency < 50μs on commodity hardware

use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::{info, error, warn};
use anyhow::Result;

mod engine;
mod market_data;
mod order_book;
mod event_store;
mod memory_pool;
mod disruptor;
mod simd_ops;
mod metrics;
mod config;
mod exchange_adapters;

use engine::UltraLowLatencyEngine;
use config::EngineConfig;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for performance monitoring
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("Starting Ultra-Low Latency Trading Engine");
    info!("Target: <10μs execution latency, 2M msg/sec throughput");
    
    // Load configuration
    let config = EngineConfig::load().await?;
    info!("Configuration loaded: {:?}", config);
    
    // Create and start the engine
    let engine = UltraLowLatencyEngine::new(config).await?;
    
    // Start performance monitoring
    let metrics_handle = tokio::spawn(async move {
        metrics::start_prometheus_exporter().await
    });
    
    // Run the engine
    let engine_handle = tokio::spawn(async move {
        engine.run().await
    });
    
    // Wait for either task to complete
    tokio::select! {
        result = engine_handle => {
            match result {
                Ok(Ok(())) => info!("Engine completed successfully"),
                Ok(Err(e)) => error!("Engine error: {}", e),
                Err(e) => error!("Engine task error: {}", e),
            }
        }
        result = metrics_handle => {
            match result {
                Ok(Ok(())) => info!("Metrics exporter completed"),
                Ok(Err(e)) => error!("Metrics error: {}", e),
                Err(e) => error!("Metrics task error: {}", e),
            }
        }
    }
    
    Ok(())
}

