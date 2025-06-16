//! Canary Probe System for Exchange Health Monitoring
//!
//! Features:
//! - Real-time health monitoring with canary probes
//! - Automatic degraded-only mode on partial outages
//! - Circuit breaker pattern implementation
//! - Health score calculation and trending
//! - Automatic recovery detection and promotion

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast};
use tokio::time::{interval, timeout};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tracing::{debug, info, warn, error};
use anyhow::{Result, anyhow};

use super::core::*;
use super::rate_limiter::{AdaptiveRateLimiter, RequestContext, RequestPriority};

/// Health probe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub degraded_threshold: f64, // Health score below this triggers degraded mode
    pub recovery_threshold: f64, // Health score above this triggers recovery
    pub probe_endpoints: Vec<ProbeEndpoint>,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            degraded_threshold: 0.7,
            recovery_threshold: 0.85,
            probe_endpoints: vec![
                ProbeEndpoint {
                    name: "server_time".to_string(),
                    weight: 1.0,
                    critical: false,
                },
                ProbeEndpoint {
                    name: "exchange_info".to_string(),
                    weight: 0.8,
                    critical: false,
                },
                ProbeEndpoint {
                    name: "order_book".to_string(),
                    weight: 1.5,
                    critical: true,
                },
            ],
        }
    }
}

/// Probe endpoint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeEndpoint {
    pub name: String,
    pub weight: f64,
    pub critical: bool, // Critical endpoints must be healthy for overall health
}

/// Health status of an exchange
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Individual probe result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub endpoint: String,
    pub success: bool,
    pub latency: Duration,
    pub error: Option<String>,
    pub timestamp: Instant,
}

/// Exchange health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub exchange: ExchangeType,
    pub status: HealthStatus,
    pub health_score: f64, // 0.0 = unhealthy, 1.0 = perfectly healthy
    pub probe_results: Vec<ProbeResult>,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub last_healthy: Option<Instant>,
    pub last_unhealthy: Option<Instant>,
    pub total_probes: u64,
    pub successful_probes: u64,
    pub timestamp: Instant,
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Failing fast, not allowing requests
    HalfOpen,  // Testing if service recovered
}

/// Circuit breaker for exchange adapters
#[derive(Debug)]
struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure: Option<Instant>,
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, success_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure: None,
            failure_threshold,
            success_threshold,
            timeout,
        }
    }
    
    fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure {
                    if last_failure.elapsed() >= self.timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            CircuitState::HalfOpen => true,
        }
    }
    
    fn on_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            },
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            },
            CircuitState::Open => {},
        }
    }
    
    fn on_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.last_failure = Some(Instant::now());
                }
            },
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.last_failure = Some(Instant::now());
                self.success_count = 0;
            },
            CircuitState::Open => {
                self.last_failure = Some(Instant::now());
            },
        }
    }
}

/// Canary probe manager
pub struct CanaryProbeManager {
    adapters: Arc<ExchangeAdapterRegistry>,
    rate_limiter: Arc<AdaptiveRateLimiter>,
    health_reports: RwLock<HashMap<ExchangeType, HealthReport>>,
    circuit_breakers: RwLock<HashMap<ExchangeType, CircuitBreaker>>,
    probe_configs: RwLock<HashMap<ExchangeType, ProbeConfig>>,
    health_sender: broadcast::Sender<HealthReport>,
    running: RwLock<bool>,
}

impl CanaryProbeManager {
    pub fn new(
        adapters: Arc<ExchangeAdapterRegistry>,
        rate_limiter: Arc<AdaptiveRateLimiter>,
    ) -> Self {
        let (health_sender, _) = broadcast::channel(1000);
        
        Self {
            adapters,
            rate_limiter,
            health_reports: RwLock::new(HashMap::new()),
            circuit_breakers: RwLock::new(HashMap::new()),
            probe_configs: RwLock::new(HashMap::new()),
            health_sender,
            running: RwLock::new(false),
        }
    }
    
    /// Configure health probes for an exchange
    pub async fn configure_exchange(
        &self,
        exchange: ExchangeType,
        config: ProbeConfig,
    ) {
        // Initialize circuit breaker
        let breaker = CircuitBreaker::new(
            config.failure_threshold,
            config.success_threshold,
            Duration::from_secs(30), // Circuit timeout
        );
        
        // Initialize health report
        let report = HealthReport {
            exchange: exchange.clone(),
            status: HealthStatus::Unknown,
            health_score: 1.0,
            probe_results: Vec::new(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_healthy: None,
            last_unhealthy: None,
            total_probes: 0,
            successful_probes: 0,
            timestamp: Instant::now(),
        };
        
        self.probe_configs.write().await.insert(exchange.clone(), config);
        self.circuit_breakers.write().await.insert(exchange.clone(), breaker);
        self.health_reports.write().await.insert(exchange, report);
    }
    
    /// Start health monitoring
    pub async fn start(&self) -> Result<()> {
        *self.running.write().await = true;
        
        let adapters = self.adapters.clone();
        let rate_limiter = self.rate_limiter.clone();
        let health_reports = self.health_reports.clone();
        let circuit_breakers = self.circuit_breakers.clone();
        let probe_configs = self.probe_configs.clone();
        let health_sender = self.health_sender.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut probe_interval = interval(Duration::from_secs(10));
            
            while *running.read().await {
                probe_interval.tick().await;
                
                let exchanges: Vec<_> = probe_configs.read().await.keys().cloned().collect();
                
                for exchange in exchanges {
                    let adapters = adapters.clone();
                    let rate_limiter = rate_limiter.clone();
                    let health_reports = health_reports.clone();
                    let circuit_breakers = circuit_breakers.clone();
                    let probe_configs = probe_configs.clone();
                    let health_sender = health_sender.clone();
                    
                    // Spawn individual probe task for each exchange
                    tokio::spawn(async move {
                        if let Err(e) = Self::probe_exchange(
                            exchange,
                            adapters,
                            rate_limiter,
                            health_reports,
                            circuit_breakers,
                            probe_configs,
                            health_sender,
                        ).await {
                            error!("Probe failed for {:?}: {}", exchange, e);
                        }
                    });
                }
            }
        });
        
        info!("Canary probe manager started");
        Ok(())
    }
    
    /// Stop health monitoring
    pub async fn stop(&self) {
        *self.running.write().await = false;
        info!("Canary probe manager stopped");
    }
    
    /// Execute health probe for a specific exchange
    async fn probe_exchange(
        exchange: ExchangeType,
        adapters: Arc<ExchangeAdapterRegistry>,
        rate_limiter: Arc<AdaptiveRateLimiter>,
        health_reports: Arc<RwLock<HashMap<ExchangeType, HealthReport>>>,
        circuit_breakers: Arc<RwLock<HashMap<ExchangeType, CircuitBreaker>>>,
        probe_configs: Arc<RwLock<HashMap<ExchangeType, ProbeConfig>>>,
        health_sender: broadcast::Sender<HealthReport>,
    ) -> Result<()> {
        // Get adapter and config
        let adapter = adapters.get_adapter(&exchange).await
            .ok_or_else(|| anyhow!("Adapter not found for {:?}", exchange))?;
        
        let config = probe_configs.read().await.get(&exchange).cloned()
            .unwrap_or_default();
        
        // Check circuit breaker
        let can_probe = {
            let mut breakers = circuit_breakers.write().await;
            let breaker = breakers.get_mut(&exchange)
                .ok_or_else(|| anyhow!("Circuit breaker not found"))?;
            breaker.can_execute()
        };
        
        if !can_probe {
            debug!("Circuit breaker open for {:?}, skipping probe", exchange);
            return Ok(());
        }
        
        // Execute probes
        let mut probe_results = Vec::new();
        let mut critical_failure = false;
        
        for endpoint in &config.probe_endpoints {
            let result = Self::execute_probe(
                &adapter,
                &rate_limiter,
                &endpoint.name,
                config.timeout,
            ).await;
            
            if !result.success && endpoint.critical {
                critical_failure = true;
            }
            
            probe_results.push(result);
        }
        
        // Calculate health score
        let health_score = Self::calculate_health_score(&probe_results, &config.probe_endpoints);
        
        // Determine status
        let status = if critical_failure || health_score < config.degraded_threshold {
            if health_score < 0.3 {
                HealthStatus::Unhealthy
            } else {
                HealthStatus::Degraded
            }
        } else {
            HealthStatus::Healthy
        };
        
        // Update circuit breaker
        {
            let mut breakers = circuit_breakers.write().await;
            let breaker = breakers.get_mut(&exchange)
                .ok_or_else(|| anyhow!("Circuit breaker not found"))?;
            
            if status == HealthStatus::Healthy {
                breaker.on_success();
            } else {
                breaker.on_failure();
            }
        }
        
        // Update health report
        let mut report = {
            let mut reports = health_reports.write().await;
            let report = reports.get_mut(&exchange)
                .ok_or_else(|| anyhow!("Health report not found"))?;
            
            // Update counters
            report.total_probes += 1;
            if status == HealthStatus::Healthy {
                report.successful_probes += 1;
                report.consecutive_successes += 1;
                report.consecutive_failures = 0;
                report.last_healthy = Some(Instant::now());
            } else {
                report.consecutive_failures += 1;
                report.consecutive_successes = 0;
                report.last_unhealthy = Some(Instant::now());
            }
            
            // Update status and score
            report.status = status;
            report.health_score = health_score;
            report.probe_results = probe_results;
            report.timestamp = Instant::now();
            
            report.clone()
        };
        
        // Broadcast health update
        let _ = health_sender.send(report);
        
        info!(
            "Health probe for {:?}: {:?} (score: {:.3})",
            exchange, status, health_score
        );
        
        Ok(())
    }
    
    /// Execute a single health probe
    async fn execute_probe(
        adapter: &Arc<dyn ExchangeAdapter>,
        rate_limiter: &Arc<AdaptiveRateLimiter>,
        endpoint: &str,
        probe_timeout: Duration,
    ) -> ProbeResult {
        let start_time = Instant::now();
        
        // Acquire rate limit permit
        let permit_result = rate_limiter.acquire_permit(
            &adapter.exchange_type(),
            RequestContext {
                endpoint: endpoint.to_string(),
                priority: RequestPriority::Normal,
                weight: 1,
                timeout: probe_timeout,
                retry_count: 0,
            },
        ).await;
        
        let result = match permit_result {
            Ok(_permit) => {
                // Execute the actual probe based on endpoint type
                let probe_future = match endpoint {
                    "server_time" => Self::probe_server_time(adapter),
                    "exchange_info" => Self::probe_exchange_info(adapter),
                    "order_book" => Self::probe_order_book(adapter),
                    _ => Self::probe_generic(adapter),
                };
                
                match timeout(probe_timeout, probe_future).await {
                    Ok(Ok(())) => ProbeResult {
                        endpoint: endpoint.to_string(),
                        success: true,
                        latency: start_time.elapsed(),
                        error: None,
                        timestamp: Instant::now(),
                    },
                    Ok(Err(e)) => ProbeResult {
                        endpoint: endpoint.to_string(),
                        success: false,
                        latency: start_time.elapsed(),
                        error: Some(e.to_string()),
                        timestamp: Instant::now(),
                    },
                    Err(_) => ProbeResult {
                        endpoint: endpoint.to_string(),
                        success: false,
                        latency: probe_timeout,
                        error: Some("Timeout".to_string()),
                        timestamp: Instant::now(),
                    },
                }
            },
            Err(e) => ProbeResult {
                endpoint: endpoint.to_string(),
                success: false,
                latency: start_time.elapsed(),
                error: Some(format!("Rate limit error: {}", e)),
                timestamp: Instant::now(),
            },
        };
        
        result
    }
    
    /// Probe server time endpoint
    async fn probe_server_time(_adapter: &Arc<dyn ExchangeAdapter>) -> Result<()> {
        // Implementation would call exchange-specific server time endpoint
        Ok(())
    }
    
    /// Probe exchange info endpoint
    async fn probe_exchange_info(_adapter: &Arc<dyn ExchangeAdapter>) -> Result<()> {
        // Implementation would call exchange-specific info endpoint
        Ok(())
    }
    
    /// Probe order book endpoint
    async fn probe_order_book(adapter: &Arc<dyn ExchangeAdapter>) -> Result<()> {
        // Use a common trading pair for probing
        let pair = TradingPair {
            base: "BTC".to_string(),
            quote: "USDT".to_string(),
            exchange_symbol: "BTCUSDT".to_string(),
        };
        
        adapter.get_market_depth(&pair, 5).await?;
        Ok(())
    }
    
    /// Generic probe fallback
    async fn probe_generic(adapter: &Arc<dyn ExchangeAdapter>) -> Result<()> {
        adapter.get_metrics().await?;
        Ok(())
    }
    
    /// Calculate health score from probe results
    fn calculate_health_score(
        results: &[ProbeResult],
        endpoints: &[ProbeEndpoint],
    ) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        
        for result in results {
            if let Some(endpoint) = endpoints.iter().find(|e| e.name == result.endpoint) {
                let success_score = if result.success { 1.0 } else { 0.0 };
                
                // Apply latency penalty
                let latency_penalty = if result.success {
                    let latency_ms = result.latency.as_millis() as f64;
                    (latency_ms / 1000.0).min(0.5) // Max 0.5 penalty for latency
                } else {
                    0.0
                };
                
                let score = (success_score - latency_penalty).max(0.0);
                
                weighted_score += score * endpoint.weight;
                total_weight += endpoint.weight;
            }
        }
        
        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        }
    }
    
    /// Get current health report for an exchange
    pub async fn get_health_report(&self, exchange: &ExchangeType) -> Option<HealthReport> {
        self.health_reports.read().await.get(exchange).cloned()
    }
    
    /// Get all health reports
    pub async fn get_all_health_reports(&self) -> HashMap<ExchangeType, HealthReport> {
        self.health_reports.read().await.clone()
    }
    
    /// Subscribe to health updates
    pub fn subscribe_health_updates(&self) -> broadcast::Receiver<HealthReport> {
        self.health_sender.subscribe()
    }
    
    /// Check if exchange is in degraded mode
    pub async fn is_degraded(&self, exchange: &ExchangeType) -> bool {
        if let Some(report) = self.get_health_report(exchange).await {
            report.status == HealthStatus::Degraded
        } else {
            false
        }
    }
    
    /// Check if exchange is healthy
    pub async fn is_healthy(&self, exchange: &ExchangeType) -> bool {
        if let Some(report) = self.get_health_report(exchange).await {
            report.status == HealthStatus::Healthy
        } else {
            false
        }
    }
    
    /// Get circuit breaker state
    pub async fn get_circuit_state(&self, exchange: &ExchangeType) -> Option<CircuitState> {
        self.circuit_breakers.read().await.get(exchange).map(|b| b.state.clone())
    }
    
    /// Force circuit breaker state (for testing or emergency)
    pub async fn set_circuit_state(&self, exchange: &ExchangeType, state: CircuitState) {
        if let Some(breaker) = self.circuit_breakers.write().await.get_mut(exchange) {
            breaker.state = state;
            warn!("Manually set circuit breaker for {:?} to {:?}", exchange, state);
        }
    }
    
    /// Get health statistics
    pub async fn get_health_statistics(&self) -> HealthStatistics {
        let reports = self.health_reports.read().await;
        
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;
        let mut total_score = 0.0;
        
        for report in reports.values() {
            match report.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
                HealthStatus::Unknown => {},
            }
            total_score += report.health_score;
        }
        
        let total_exchanges = reports.len();
        let average_score = if total_exchanges > 0 {
            total_score / total_exchanges as f64
        } else {
            0.0
        };
        
        HealthStatistics {
            total_exchanges,
            healthy_count,
            degraded_count,
            unhealthy_count,
            average_health_score: average_score,
            last_updated: Instant::now(),
        }
    }
}

/// Overall health statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatistics {
    pub total_exchanges: usize,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unhealthy_count: usize,
    pub average_health_score: f64,
    pub last_updated: Instant,
}

