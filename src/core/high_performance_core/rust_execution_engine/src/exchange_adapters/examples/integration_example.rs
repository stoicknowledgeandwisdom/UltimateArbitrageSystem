//! Exchange & Liquidity Adapter Layer Integration Example
//!
//! This example demonstrates how to use the comprehensive exchange adapter system
//! with all its advanced features:
//! - Multi-exchange connectivity (Binance, Coinbase, Bybit, CME)
//! - Smart order routing with latency optimization
//! - Health monitoring with canary probes
//! - Adaptive rate limiting
//! - WebSocket failover management

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use tracing::{info, warn, error};

use crate::exchange_adapters::{
    core::*,
    centralized::binance::{BinanceAdapter, BinanceConfig},
    rate_limiter::{AdaptiveRateLimiter, RateLimitPolicy, default_policies},
    routing::{SmartOrderRouter, RoutingCriteria},
    canary::{CanaryProbeManager, ProbeConfig},
    websocket::{WebSocketManager, ConnectionConfig, ConnectionRole},
};

/// Complete exchange adapter system
pub struct ExchangeAdapterSystem {
    registry: Arc<ExchangeAdapterRegistry>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    smart_router: SmartOrderRouter,
    canary_manager: CanaryProbeManager,
    websocket_manager: WebSocketManager,
}

impl ExchangeAdapterSystem {
    /// Initialize the complete exchange adapter system
    pub async fn new() -> Result<Self> {
        info!("Initializing Exchange & Liquidity Adapter System");
        
        // Create exchange adapter registry
        let registry = Arc::new(ExchangeAdapterRegistry::new());
        
        // Create adaptive rate limiter
        let rate_limiter = Arc::new(AdaptiveRateLimiter::new());
        
        // Configure default rate limiting policies
        let policies = default_policies();
        for (exchange, policy) in policies {
            rate_limiter.configure_exchange(exchange, policy).await;
        }
        
        // Create smart order router
        let smart_router = SmartOrderRouter::new(registry.clone());
        
        // Create canary probe manager
        let canary_manager = CanaryProbeManager::new(
            registry.clone(),
            rate_limiter.clone(),
        );
        
        // Create WebSocket manager
        let websocket_manager = WebSocketManager::new();
        
        Ok(Self {
            registry,
            rate_limiter,
            smart_router,
            canary_manager,
            websocket_manager,
        })
    }
    
    /// Add Binance exchange adapter
    pub async fn add_binance(&self, config: BinanceConfig) -> Result<()> {
        info!("Adding Binance exchange adapter");
        
        // Create Binance adapter
        let adapter = Arc::new(BinanceAdapter::new(config.clone())?);
        
        // Register adapter
        self.registry.register_adapter(adapter.clone()).await;
        
        // Configure health monitoring
        self.canary_manager.configure_exchange(
            ExchangeType::Centralized(CentralizedExchange::Binance),
            ProbeConfig::default(),
        ).await;
        
        info!("Binance adapter registered successfully");
        Ok(())
    }
    
    /// Start all services
    pub async fn start(&self) -> Result<()> {
        info!("Starting exchange adapter system services");
        
        // Start health monitoring
        self.canary_manager.start().await?;
        
        info!("Exchange adapter system started successfully");
        Ok(())
    }
    
    /// Execute a smart order with full optimization
    pub async fn execute_smart_order(
        &self,
        pair: &TradingPair,
        side: OrderSide,
        quantity: f64,
    ) -> Result<uuid::Uuid> {
        info!("Executing smart order: {:?} {} {}", pair, quantity, format!("{:?}", side));
        
        // Define routing criteria for latency-optimized execution
        let criteria = RoutingCriteria {
            max_latency: Duration::from_millis(50),
            max_settlement_risk: 0.05,
            min_confidence_score: 0.8,
            max_price_impact: 0.001,
            max_fee_ratio: 0.01,
            prefer_settlement_speed: true,
            allow_partial_fills: true,
            max_hops: 3,
        };
        
        // Find optimal route
        let route = self.smart_router.find_optimal_route(
            pair,
            side,
            quantity,
            criteria,
        ).await?;
        
        info!(
            "Optimal route found: {} segments, estimated price: {:.4}, confidence: {:.3}",
            route.segments.len(),
            route.estimated_total_price,
            route.confidence_score
        );
        
        // Execute the route
        let execution_id = self.smart_router.execute_route(route).await?;
        
        info!("Order execution started: {}", execution_id);
        Ok(execution_id)
    }
    
    /// Get system health status
    pub async fn get_system_health(&self) -> SystemHealthStatus {
        let health_stats = self.canary_manager.get_health_statistics().await;
        let adapters = self.registry.get_all_adapters().await;
        
        let mut exchange_statuses = HashMap::new();
        
        for adapter in adapters {
            let exchange_type = adapter.exchange_type();
            let health_report = self.canary_manager.get_health_report(&exchange_type).await;
            let circuit_state = self.canary_manager.get_circuit_state(&exchange_type).await;
            
            exchange_statuses.insert(
                exchange_type,
                ExchangeStatus {
                    health_report,
                    circuit_state,
                    is_degraded: self.canary_manager.is_degraded(&exchange_type).await,
                },
            );
        }
        
        SystemHealthStatus {
            overall_health: if health_stats.healthy_count > health_stats.unhealthy_count {
                "Healthy".to_string()
            } else {
                "Degraded".to_string()
            },
            total_exchanges: health_stats.total_exchanges,
            healthy_exchanges: health_stats.healthy_count,
            degraded_exchanges: health_stats.degraded_count,
            unhealthy_exchanges: health_stats.unhealthy_count,
            average_health_score: health_stats.average_health_score,
            exchange_statuses,
        }
    }
    
    /// Monitor execution in real-time
    pub async fn monitor_execution(&self, execution_id: uuid::Uuid) -> Result<()> {
        info!("Monitoring execution: {}", execution_id);
        
        // Monitor execution status
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        
        loop {
            interval.tick().await;
            
            if let Some(execution) = self.smart_router.get_execution_status(execution_id).await {
                match execution.status {
                    crate::exchange_adapters::routing::ExecutionStatus::Completed => {
                        info!(
                            "Execution completed: filled {:.6}, avg price: {:.4}, fees: {:.6}",
                            execution.total_filled,
                            execution.average_price,
                            execution.total_fees
                        );
                        break;
                    },
                    crate::exchange_adapters::routing::ExecutionStatus::Failed => {
                        error!("Execution failed: {}", execution_id);
                        break;
                    },
                    crate::exchange_adapters::routing::ExecutionStatus::Cancelled => {
                        warn!("Execution cancelled: {}", execution_id);
                        break;
                    },
                    _ => {
                        // Continue monitoring
                        continue;
                    }
                }
            } else {
                warn!("Execution not found: {}", execution_id);
                break;
            }
        }
        
        Ok(())
    }
    
    /// Stop all services
    pub async fn stop(&self) {
        info!("Stopping exchange adapter system");
        self.canary_manager.stop().await;
        info!("Exchange adapter system stopped");
    }
}

/// System health status
#[derive(Debug)]
pub struct SystemHealthStatus {
    pub overall_health: String,
    pub total_exchanges: usize,
    pub healthy_exchanges: usize,
    pub degraded_exchanges: usize,
    pub unhealthy_exchanges: usize,
    pub average_health_score: f64,
    pub exchange_statuses: HashMap<ExchangeType, ExchangeStatus>,
}

/// Individual exchange status
#[derive(Debug)]
pub struct ExchangeStatus {
    pub health_report: Option<crate::exchange_adapters::canary::HealthReport>,
    pub circuit_state: Option<crate::exchange_adapters::canary::CircuitState>,
    pub is_degraded: bool,
}

/// Example usage function
pub async fn run_example() -> Result<()> {
    info!("Starting Exchange & Liquidity Adapter Layer Example");
    
    // Initialize the system
    let system = ExchangeAdapterSystem::new().await?;
    
    // Add Binance exchange
    let binance_config = BinanceConfig {
        api_key: "your_api_key".to_string(),
        secret_key: "your_secret_key".to_string(),
        testnet: true, // Use testnet for example
        ..Default::default()
    };
    
    system.add_binance(binance_config).await?;
    
    // Start all services
    system.start().await?;
    
    // Wait for system to initialize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Check system health
    let health = system.get_system_health().await;
    info!("System health: {:?}", health);
    
    // Execute a test order
    let pair = TradingPair {
        base: "BTC".to_string(),
        quote: "USDT".to_string(),
        exchange_symbol: "BTCUSDT".to_string(),
    };
    
    let execution_id = system.execute_smart_order(
        &pair,
        OrderSide::Buy,
        0.001, // 0.001 BTC
    ).await?;
    
    // Monitor the execution
    system.monitor_execution(execution_id).await?;
    
    // Clean up
    system.stop().await;
    
    info!("Example completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let system = ExchangeAdapterSystem::new().await.unwrap();
        let health = system.get_system_health().await;
        assert_eq!(health.total_exchanges, 0); // No exchanges added yet
    }
    
    #[tokio::test]
    async fn test_binance_adapter_registration() {
        let system = ExchangeAdapterSystem::new().await.unwrap();
        
        let config = BinanceConfig {
            api_key: "test_key".to_string(),
            secret_key: "test_secret".to_string(),
            testnet: true,
            ..Default::default()
        };
        
        system.add_binance(config).await.unwrap();
        
        let health = system.get_system_health().await;
        assert_eq!(health.total_exchanges, 1);
    }
}

