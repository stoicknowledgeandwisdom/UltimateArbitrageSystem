//! Core Exchange Adapter Types and Traits

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tokio::sync::RwLock;
use anyhow::Result;

/// Exchange type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExchangeType {
    /// Centralized exchanges
    Centralized(CentralizedExchange),
    /// Decentralized exchanges
    Decentralized(DecentralizedExchange),
    /// Cross-chain bridges
    Bridge(BridgeProtocol),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CentralizedExchange {
    Binance,
    Coinbase,
    Bybit,
    CME,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecentralizedExchange {
    UniswapV4,
    ZeroXRFQ,
    DydxV4,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgeProtocol {
    LayerZero,
    Wormhole,
    Custom(String),
}

/// Trading pair representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradingPair {
    pub base: String,
    pub quote: String,
    pub exchange_symbol: String, // Exchange-specific symbol
}

/// Order types supported across exchanges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
    IcebergOrder,
    PostOnly,
    ReduceOnly,
}

/// Order side
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order status across all exchanges
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Unified order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,
    pub exchange_id: String,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub status: OrderStatus,
    pub timestamp: Instant,
    pub filled_quantity: f64,
    pub average_price: Option<f64>,
    pub fees: f64,
    pub metadata: HashMap<String, String>,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    pub price: f64,
    pub quantity: f64,
    pub timestamp: Instant,
}

/// Market depth snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepth {
    pub pair: TradingPair,
    pub bids: Vec<OrderBookEntry>,
    pub asks: Vec<OrderBookEntry>,
    pub timestamp: Instant,
    pub sequence: u64,
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeExecution {
    pub order_id: Uuid,
    pub exchange: ExchangeType,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub fees: f64,
    pub timestamp: Instant,
    pub latency_microseconds: u64,
}

/// Exchange adapter health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdapterHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
}

/// Exchange adapter metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetrics {
    pub health: AdapterHealth,
    pub latency_microseconds: u64,
    pub success_rate: f64,
    pub rate_limit_remaining: u32,
    pub active_connections: u32,
    pub last_heartbeat: Instant,
    pub error_count: u64,
    pub total_requests: u64,
}

/// Core exchange adapter trait
#[async_trait]
pub trait ExchangeAdapter: Send + Sync {
    /// Get exchange type
    fn exchange_type(&self) -> ExchangeType;
    
    /// Get supported trading pairs
    async fn get_trading_pairs(&self) -> Result<Vec<TradingPair>>;
    
    /// Get current market depth
    async fn get_market_depth(&self, pair: &TradingPair, depth: u32) -> Result<MarketDepth>;
    
    /// Place an order
    async fn place_order(&self, order: Order) -> Result<Order>;
    
    /// Cancel an order
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    
    /// Get order status
    async fn get_order_status(&self, order_id: &str) -> Result<Order>;
    
    /// Get active orders
    async fn get_active_orders(&self) -> Result<Vec<Order>>;
    
    /// Get account balance
    async fn get_balance(&self, asset: &str) -> Result<f64>;
    
    /// Get adapter health and metrics
    async fn get_metrics(&self) -> Result<AdapterMetrics>;
    
    /// Start real-time data streams
    async fn start_streams(&self, pairs: Vec<TradingPair>) -> Result<()>;
    
    /// Stop real-time data streams
    async fn stop_streams(&self) -> Result<()>;
    
    /// Handle connection recovery
    async fn recover_connection(&self) -> Result<()>;
}

/// Liquidity source for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquiditySource {
    pub adapter: ExchangeType,
    pub pair: TradingPair,
    pub depth: MarketDepth,
    pub latency_estimate: Duration,
    pub settlement_risk: f64, // 0.0 = no risk, 1.0 = maximum risk
    pub fees: f64,
    pub available_liquidity: f64,
}

/// Route execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRoute {
    pub route_id: Uuid,
    pub sources: Vec<LiquiditySource>,
    pub total_quantity: f64,
    pub estimated_price: f64,
    pub estimated_fees: f64,
    pub estimated_latency: Duration,
    pub confidence_score: f64, // 0.0 = low confidence, 1.0 = high confidence
    pub created_at: Instant,
}

/// Exchange adapter registry
pub struct ExchangeAdapterRegistry {
    adapters: RwLock<HashMap<ExchangeType, Arc<dyn ExchangeAdapter>>>,
}

impl ExchangeAdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
        }
    }
    
    pub async fn register_adapter(&self, adapter: Arc<dyn ExchangeAdapter>) {
        let exchange_type = adapter.exchange_type();
        self.adapters.write().await.insert(exchange_type, adapter);
    }
    
    pub async fn get_adapter(&self, exchange_type: &ExchangeType) -> Option<Arc<dyn ExchangeAdapter>> {
        self.adapters.read().await.get(exchange_type).cloned()
    }
    
    pub async fn get_all_adapters(&self) -> Vec<Arc<dyn ExchangeAdapter>> {
        self.adapters.read().await.values().cloned().collect()
    }
    
    pub async fn remove_adapter(&self, exchange_type: &ExchangeType) -> Option<Arc<dyn ExchangeAdapter>> {
        self.adapters.write().await.remove(exchange_type)
    }
}

/// Configuration for exchange adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub api_key: String,
    pub secret_key: String,
    pub base_url: String,
    pub websocket_url: String,
    pub testnet: bool,
    pub rate_limit_per_second: u32,
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub request_timeout: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub heartbeat_interval: Duration,
}

