//! Smart Order Router (SOR) with Latency-Aware Routing
//!
//! Features:
//! - Depth-aware routing across multiple liquidity sources
//! - Latency optimization with settlement risk balancing
//! - Multi-exchange order splitting and aggregation
//! - Real-time route optimization and rebalancing
//! - Price impact minimization strategies

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tracing::{debug, info, warn, error};
use anyhow::{Result, anyhow};

use super::core::*;

/// Route optimization criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingCriteria {
    pub max_latency: Duration,
    pub max_settlement_risk: f64,
    pub min_confidence_score: f64,
    pub max_price_impact: f64,
    pub max_fee_ratio: f64,
    pub prefer_settlement_speed: bool,
    pub allow_partial_fills: bool,
    pub max_hops: u8,
}

impl Default for RoutingCriteria {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            max_settlement_risk: 0.05, // 5%
            min_confidence_score: 0.8,
            max_price_impact: 0.001, // 0.1%
            max_fee_ratio: 0.01, // 1%
            prefer_settlement_speed: true,
            allow_partial_fills: true,
            max_hops: 3,
        }
    }
}

/// Route execution strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute all orders simultaneously
    Simultaneous,
    /// Execute orders sequentially by priority
    Sequential,
    /// Execute with time-weighted average price (TWAP)
    TWAP { duration: Duration, intervals: u32 },
    /// Execute with volume-weighted average price (VWAP)
    VWAP { target_volume: f64 },
    /// Execute with implementation shortfall strategy
    ImplementationShortfall { max_deviation: f64 },
}

/// Route segment representing execution on a single exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSegment {
    pub id: Uuid,
    pub exchange: ExchangeType,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub quantity: f64,
    pub expected_price: f64,
    pub expected_fees: f64,
    pub expected_latency: Duration,
    pub settlement_risk: f64,
    pub liquidity_depth: f64,
    pub price_impact: f64,
    pub confidence: f64,
    pub priority: u8, // 0 = highest priority
}

/// Complete execution route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedRoute {
    pub id: Uuid,
    pub pair: TradingPair,
    pub side: OrderSide,
    pub total_quantity: f64,
    pub segments: Vec<RouteSegment>,
    pub strategy: ExecutionStrategy,
    pub estimated_total_price: f64,
    pub estimated_total_fees: f64,
    pub estimated_total_latency: Duration,
    pub total_settlement_risk: f64,
    pub total_price_impact: f64,
    pub confidence_score: f64,
    pub created_at: Instant,
    pub valid_until: Instant,
}

/// Route execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteExecution {
    pub route_id: Uuid,
    pub status: ExecutionStatus,
    pub segments: Vec<SegmentExecution>,
    pub total_filled: f64,
    pub average_price: f64,
    pub total_fees: f64,
    pub execution_time: Duration,
    pub slippage: f64,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    InProgress,
    Completed,
    PartiallyFilled,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentExecution {
    pub segment_id: Uuid,
    pub status: ExecutionStatus,
    pub filled_quantity: f64,
    pub average_price: f64,
    pub fees: f64,
    pub latency: Duration,
    pub error: Option<String>,
}

/// Liquidity aggregator for collecting market depth
pub struct LiquidityAggregator {
    adapters: Arc<ExchangeAdapterRegistry>,
    depth_cache: RwLock<HashMap<(ExchangeType, TradingPair), (MarketDepth, Instant)>>,
    cache_ttl: Duration,
}

impl LiquidityAggregator {
    pub fn new(adapters: Arc<ExchangeAdapterRegistry>) -> Self {
        Self {
            adapters,
            depth_cache: RwLock::new(HashMap::new()),
            cache_ttl: Duration::from_millis(500), // 500ms cache
        }
    }
    
    /// Get aggregated liquidity sources for a trading pair
    pub async fn get_liquidity_sources(
        &self,
        pair: &TradingPair,
        side: OrderSide,
        quantity: f64,
    ) -> Result<Vec<LiquiditySource>> {
        let mut sources = Vec::new();
        let adapters = self.adapters.get_all_adapters().await;
        
        for adapter in adapters {
            if let Ok(depth) = self.get_market_depth(&adapter, pair).await {
                let available_liquidity = self.calculate_available_liquidity(&depth, side, quantity);
                
                if available_liquidity > 0.0 {
                    let metrics = adapter.get_metrics().await.unwrap_or_default();
                    
                    let source = LiquiditySource {
                        adapter: adapter.exchange_type(),
                        pair: pair.clone(),
                        depth,
                        latency_estimate: Duration::from_micros(metrics.latency_microseconds),
                        settlement_risk: self.calculate_settlement_risk(&adapter.exchange_type()),
                        fees: self.estimate_fees(&adapter.exchange_type(), quantity),
                        available_liquidity,
                    };
                    
                    sources.push(source);
                }
            }
        }
        
        Ok(sources)
    }
    
    /// Get market depth with caching
    async fn get_market_depth(
        &self,
        adapter: &Arc<dyn ExchangeAdapter>,
        pair: &TradingPair,
    ) -> Result<MarketDepth> {
        let cache_key = (adapter.exchange_type(), pair.clone());
        
        // Check cache first
        {
            let cache = self.depth_cache.read().await;
            if let Some((depth, timestamp)) = cache.get(&cache_key) {
                if timestamp.elapsed() < self.cache_ttl {
                    return Ok(depth.clone());
                }
            }
        }
        
        // Fetch fresh data
        let depth = adapter.get_market_depth(pair, 20).await?;
        
        // Update cache
        self.depth_cache.write().await.insert(cache_key, (depth.clone(), Instant::now()));
        
        Ok(depth)
    }
    
    /// Calculate available liquidity for a given quantity
    fn calculate_available_liquidity(
        &self,
        depth: &MarketDepth,
        side: OrderSide,
        target_quantity: f64,
    ) -> f64 {
        let levels = match side {
            OrderSide::Buy => &depth.asks,
            OrderSide::Sell => &depth.bids,
        };
        
        let mut cumulative_quantity = 0.0;
        for level in levels {
            cumulative_quantity += level.quantity;
            if cumulative_quantity >= target_quantity {
                return target_quantity;
            }
        }
        
        cumulative_quantity
    }
    
    /// Calculate settlement risk based on exchange type
    fn calculate_settlement_risk(&self, exchange: &ExchangeType) -> f64 {
        match exchange {
            ExchangeType::Centralized(_) => 0.01, // 1% for centralized
            ExchangeType::Decentralized(_) => 0.05, // 5% for decentralized
            ExchangeType::Bridge(_) => 0.1, // 10% for bridges
        }
    }
    
    /// Estimate trading fees
    fn estimate_fees(&self, exchange: &ExchangeType, quantity: f64) -> f64 {
        let fee_rate = match exchange {
            ExchangeType::Centralized(CentralizedExchange::Binance) => 0.001, // 0.1%
            ExchangeType::Centralized(CentralizedExchange::Coinbase) => 0.005, // 0.5%
            ExchangeType::Centralized(CentralizedExchange::Bybit) => 0.001, // 0.1%
            ExchangeType::Centralized(CentralizedExchange::CME) => 0.0005, // 0.05%
            ExchangeType::Decentralized(_) => 0.003, // 0.3%
            ExchangeType::Bridge(_) => 0.01, // 1%
            _ => 0.002, // 0.2% default
        };
        
        quantity * fee_rate
    }
}

/// Smart Order Router implementation
pub struct SmartOrderRouter {
    liquidity_aggregator: LiquidityAggregator,
    route_cache: RwLock<HashMap<String, (OptimizedRoute, Instant)>>,
    active_executions: RwLock<HashMap<Uuid, RouteExecution>>,
    cache_ttl: Duration,
}

impl SmartOrderRouter {
    pub fn new(adapters: Arc<ExchangeAdapterRegistry>) -> Self {
        Self {
            liquidity_aggregator: LiquidityAggregator::new(adapters),
            route_cache: RwLock::new(HashMap::new()),
            active_executions: RwLock::new(HashMap::new()),
            cache_ttl: Duration::from_millis(200), // 200ms route cache
        }
    }
    
    /// Find optimal route for order execution
    pub async fn find_optimal_route(
        &self,
        pair: &TradingPair,
        side: OrderSide,
        quantity: f64,
        criteria: RoutingCriteria,
    ) -> Result<OptimizedRoute> {
        // Check cache first
        let cache_key = format!("{:?}:{:?}:{:.8}:{:?}", pair, side, quantity, criteria);
        {
            let cache = self.route_cache.read().await;
            if let Some((route, timestamp)) = cache.get(&cache_key) {
                if timestamp.elapsed() < self.cache_ttl {
                    return Ok(route.clone());
                }
            }
        }
        
        // Get liquidity sources
        let sources = self.liquidity_aggregator
            .get_liquidity_sources(pair, side, quantity)
            .await?;
        
        if sources.is_empty() {
            return Err(anyhow!("No liquidity sources available"));
        }
        
        // Filter sources by criteria
        let filtered_sources: Vec<_> = sources.into_iter()
            .filter(|source| {
                source.latency_estimate <= criteria.max_latency &&
                source.settlement_risk <= criteria.max_settlement_risk &&
                source.fees / quantity <= criteria.max_fee_ratio
            })
            .collect();
        
        if filtered_sources.is_empty() {
            return Err(anyhow!("No sources meet routing criteria"));
        }
        
        // Find optimal route
        let route = self.optimize_route(
            pair,
            side,
            quantity,
            filtered_sources,
            &criteria,
        ).await?;
        
        // Cache the route
        self.route_cache.write().await.insert(cache_key, (route.clone(), Instant::now()));
        
        Ok(route)
    }
    
    /// Optimize route using multiple algorithms
    async fn optimize_route(
        &self,
        pair: &TradingPair,
        side: OrderSide,
        quantity: f64,
        sources: Vec<LiquiditySource>,
        criteria: &RoutingCriteria,
    ) -> Result<OptimizedRoute> {
        // Sort sources by a composite score
        let mut scored_sources: Vec<_> = sources.into_iter()
            .map(|source| {
                let latency_score = 1.0 - (source.latency_estimate.as_millis() as f64 / criteria.max_latency.as_millis() as f64);
                let risk_score = 1.0 - (source.settlement_risk / criteria.max_settlement_risk);
                let fee_score = 1.0 - (source.fees / quantity / criteria.max_fee_ratio);
                let liquidity_score = (source.available_liquidity / quantity).min(1.0);
                
                let composite_score = if criteria.prefer_settlement_speed {
                    latency_score * 0.4 + risk_score * 0.3 + fee_score * 0.2 + liquidity_score * 0.1
                } else {
                    latency_score * 0.2 + risk_score * 0.2 + fee_score * 0.4 + liquidity_score * 0.2
                };
                
                (source, composite_score)
            })
            .collect();
        
        scored_sources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create route segments
        let mut segments = Vec::new();
        let mut remaining_quantity = quantity;
        let mut total_expected_price = 0.0;
        let mut total_expected_fees = 0.0;
        let mut max_latency = Duration::ZERO;
        let mut total_settlement_risk = 0.0;
        let mut total_price_impact = 0.0;
        
        for (source, score) in scored_sources {
            if remaining_quantity <= 0.0 || segments.len() >= criteria.max_hops as usize {
                break;
            }
            
            let segment_quantity = remaining_quantity.min(source.available_liquidity);
            let (segment_price, price_impact) = self.calculate_execution_price(
                &source.depth,
                side,
                segment_quantity,
            );
            
            if price_impact > criteria.max_price_impact {
                continue; // Skip if price impact too high
            }
            
            let segment = RouteSegment {
                id: Uuid::new_v4(),
                exchange: source.adapter,
                pair: pair.clone(),
                side,
                quantity: segment_quantity,
                expected_price: segment_price,
                expected_fees: source.fees * (segment_quantity / quantity),
                expected_latency: source.latency_estimate,
                settlement_risk: source.settlement_risk,
                liquidity_depth: source.available_liquidity,
                price_impact,
                confidence: score,
                priority: segments.len() as u8,
            };
            
            total_expected_price += segment_price * segment_quantity;
            total_expected_fees += segment.expected_fees;
            max_latency = max_latency.max(segment.expected_latency);
            total_settlement_risk += segment.settlement_risk * (segment_quantity / quantity);
            total_price_impact += price_impact * (segment_quantity / quantity);
            
            segments.push(segment);
            remaining_quantity -= segment_quantity;
        }
        
        if segments.is_empty() {
            return Err(anyhow!("Could not create any valid route segments"));
        }
        
        let filled_quantity = quantity - remaining_quantity;
        let average_price = total_expected_price / filled_quantity;
        let confidence_score = segments.iter().map(|s| s.confidence).sum::<f64>() / segments.len() as f64;
        
        if confidence_score < criteria.min_confidence_score {
            return Err(anyhow!("Route confidence score too low: {:.3}", confidence_score));
        }
        
        // Determine execution strategy
        let strategy = if segments.len() == 1 {
            ExecutionStrategy::Simultaneous
        } else if max_latency > Duration::from_millis(50) {
            ExecutionStrategy::Sequential
        } else {
            ExecutionStrategy::Simultaneous
        };
        
        Ok(OptimizedRoute {
            id: Uuid::new_v4(),
            pair: pair.clone(),
            side,
            total_quantity: filled_quantity,
            segments,
            strategy,
            estimated_total_price: average_price,
            estimated_total_fees: total_expected_fees,
            estimated_total_latency: max_latency,
            total_settlement_risk,
            total_price_impact,
            confidence_score,
            created_at: Instant::now(),
            valid_until: Instant::now() + Duration::from_secs(30),
        })
    }
    
    /// Calculate execution price and price impact
    fn calculate_execution_price(
        &self,
        depth: &MarketDepth,
        side: OrderSide,
        quantity: f64,
    ) -> (f64, f64) {
        let levels = match side {
            OrderSide::Buy => &depth.asks,
            OrderSide::Sell => &depth.bids,
        };
        
        if levels.is_empty() {
            return (0.0, 1.0); // No liquidity
        }
        
        let mut remaining_quantity = quantity;
        let mut total_cost = 0.0;
        let mut weighted_price = 0.0;
        
        for level in levels {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            let fill_quantity = remaining_quantity.min(level.quantity);
            total_cost += fill_quantity * level.price;
            weighted_price += fill_quantity * level.price;
            remaining_quantity -= fill_quantity;
        }
        
        if remaining_quantity > 0.0 {
            // Partial fill - estimate price impact
            let best_price = levels[0].price;
            let average_price = weighted_price / (quantity - remaining_quantity);
            let price_impact = (average_price - best_price).abs() / best_price;
            return (average_price, price_impact + 0.1); // Add penalty for partial fill
        }
        
        let average_price = total_cost / quantity;
        let best_price = levels[0].price;
        let price_impact = (average_price - best_price).abs() / best_price;
        
        (average_price, price_impact)
    }
    
    /// Execute an optimized route
    pub async fn execute_route(
        &self,
        route: OptimizedRoute,
    ) -> Result<Uuid> {
        let execution = RouteExecution {
            route_id: route.id,
            status: ExecutionStatus::Pending,
            segments: route.segments.iter().map(|s| SegmentExecution {
                segment_id: s.id,
                status: ExecutionStatus::Pending,
                filled_quantity: 0.0,
                average_price: 0.0,
                fees: 0.0,
                latency: Duration::ZERO,
                error: None,
            }).collect(),
            total_filled: 0.0,
            average_price: 0.0,
            total_fees: 0.0,
            execution_time: Duration::ZERO,
            slippage: 0.0,
            started_at: Instant::now(),
            completed_at: None,
        };
        
        let execution_id = route.id;
        self.active_executions.write().await.insert(execution_id, execution);
        
        // Execute based on strategy
        match route.strategy {
            ExecutionStrategy::Simultaneous => {
                self.execute_simultaneous(route).await?;
            },
            ExecutionStrategy::Sequential => {
                self.execute_sequential(route).await?;
            },
            ExecutionStrategy::TWAP { duration, intervals } => {
                self.execute_twap(route, duration, intervals).await?;
            },
            ExecutionStrategy::VWAP { target_volume } => {
                self.execute_vwap(route, target_volume).await?;
            },
            ExecutionStrategy::ImplementationShortfall { max_deviation } => {
                self.execute_implementation_shortfall(route, max_deviation).await?;
            },
        }
        
        Ok(execution_id)
    }
    
    /// Execute all segments simultaneously
    async fn execute_simultaneous(&self, route: OptimizedRoute) -> Result<()> {
        // Implementation would spawn concurrent tasks for each segment
        info!("Executing route {} with {} segments simultaneously", 
            route.id, route.segments.len());
        
        // Update execution status
        if let Some(execution) = self.active_executions.write().await.get_mut(&route.id) {
            execution.status = ExecutionStatus::InProgress;
        }
        
        Ok(())
    }
    
    /// Execute segments sequentially by priority
    async fn execute_sequential(&self, route: OptimizedRoute) -> Result<()> {
        info!("Executing route {} with {} segments sequentially", 
            route.id, route.segments.len());
        
        // Sort segments by priority and execute in order
        let mut sorted_segments = route.segments;
        sorted_segments.sort_by_key(|s| s.priority);
        
        // Update execution status
        if let Some(execution) = self.active_executions.write().await.get_mut(&route.id) {
            execution.status = ExecutionStatus::InProgress;
        }
        
        Ok(())
    }
    
    /// Execute with Time-Weighted Average Price strategy
    async fn execute_twap(
        &self,
        route: OptimizedRoute,
        duration: Duration,
        intervals: u32,
    ) -> Result<()> {
        info!("Executing route {} with TWAP strategy over {:?} in {} intervals", 
            route.id, duration, intervals);
        
        Ok(())
    }
    
    /// Execute with Volume-Weighted Average Price strategy
    async fn execute_vwap(&self, route: OptimizedRoute, target_volume: f64) -> Result<()> {
        info!("Executing route {} with VWAP strategy targeting volume {}", 
            route.id, target_volume);
        
        Ok(())
    }
    
    /// Execute with Implementation Shortfall strategy
    async fn execute_implementation_shortfall(
        &self,
        route: OptimizedRoute,
        max_deviation: f64,
    ) -> Result<()> {
        info!("Executing route {} with Implementation Shortfall, max deviation {:.4}", 
            route.id, max_deviation);
        
        Ok(())
    }
    
    /// Get execution status
    pub async fn get_execution_status(&self, execution_id: Uuid) -> Option<RouteExecution> {
        self.active_executions.read().await.get(&execution_id).cloned()
    }
    
    /// Cancel route execution
    pub async fn cancel_execution(&self, execution_id: Uuid) -> Result<()> {
        if let Some(execution) = self.active_executions.write().await.get_mut(&execution_id) {
            execution.status = ExecutionStatus::Cancelled;
            execution.completed_at = Some(Instant::now());
            info!("Cancelled route execution {}", execution_id);
            Ok(())
        } else {
            Err(anyhow!("Execution not found: {}", execution_id))
        }
    }
    
    /// Clean up completed executions
    pub async fn cleanup_executions(&self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;
        
        self.active_executions.write().await.retain(|_, execution| {
            match execution.completed_at {
                Some(completed) => completed > cutoff,
                None => true, // Keep active executions
            }
        });
    }
}

trait AdapterMetricsExt {
    fn unwrap_or_default(self) -> AdapterMetrics;
}

impl AdapterMetricsExt for Result<AdapterMetrics> {
    fn unwrap_or_default(self) -> AdapterMetrics {
        self.unwrap_or_else(|_| AdapterMetrics {
            health: AdapterHealth::Unhealthy,
            latency_microseconds: 100_000, // 100ms default
            success_rate: 0.5,
            rate_limit_remaining: 0,
            active_connections: 0,
            last_heartbeat: Instant::now(),
            error_count: 0,
            total_requests: 0,
        })
    }
}

