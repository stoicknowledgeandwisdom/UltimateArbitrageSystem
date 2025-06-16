//! Adaptive Rate Limiter with Exchange-Specific Policies
//!
//! Features:
//! - Per-exchange rate limiting with custom policies
//! - Token bucket algorithm with burst capacity
//! - Adaptive throttling based on server responses
//! - Rate limit recovery detection
//! - Priority-based request queuing

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::{sleep, timeout};
use tracing::{debug, warn, error};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use super::core::ExchangeType;

/// Rate limit policy for different exchanges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitPolicy {
    pub max_requests_per_second: u32,
    pub burst_capacity: u32,
    pub weight_multipliers: HashMap<String, u32>, // endpoint -> weight
    pub recovery_time: Duration,
    pub backoff_multiplier: f64,
    pub max_backoff: Duration,
    pub priority_levels: u8,
}

impl Default for RateLimitPolicy {
    fn default() -> Self {
        Self {
            max_requests_per_second: 100,
            burst_capacity: 200,
            weight_multipliers: HashMap::new(),
            recovery_time: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(300),
            priority_levels: 3,
        }
    }
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Critical = 0,  // Order cancellations, emergency stops
    High = 1,      // Order placements, balance queries
    Normal = 2,    // Market data, general queries
}

/// Rate limit violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitViolation {
    ExceededLimit,
    ServerThrottling,
    IPBanned,
    WeightExceeded,
}

/// Request context for rate limiting
#[derive(Debug)]
pub struct RequestContext {
    pub endpoint: String,
    pub priority: RequestPriority,
    pub weight: u32,
    pub timeout: Duration,
    pub retry_count: u32,
}

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u32, refill_rate: u32) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate: refill_rate as f64,
            last_refill: Instant::now(),
        }
    }
    
    fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();
        
        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }
    
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }
    
    fn time_until_tokens(&mut self, tokens: u32) -> Duration {
        self.refill();
        
        if self.tokens >= tokens as f64 {
            Duration::ZERO
        } else {
            let needed = tokens as f64 - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }
}

/// Adaptive rate limiter for exchanges
pub struct AdaptiveRateLimiter {
    policies: RwLock<HashMap<ExchangeType, RateLimitPolicy>>,
    buckets: RwLock<HashMap<ExchangeType, Arc<Mutex<TokenBucket>>>>,
    violations: RwLock<HashMap<ExchangeType, Vec<(Instant, RateLimitViolation)>>>,
    priority_semaphores: RwLock<HashMap<(ExchangeType, RequestPriority), Arc<Semaphore>>>,
    backoff_until: RwLock<HashMap<ExchangeType, Instant>>,
}

impl AdaptiveRateLimiter {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
            buckets: RwLock::new(HashMap::new()),
            violations: RwLock::new(HashMap::new()),
            priority_semaphores: RwLock::new(HashMap::new()),
            backoff_until: RwLock::new(HashMap::new()),
        }
    }
    
    /// Configure rate limiting policy for an exchange
    pub async fn configure_exchange(&self, exchange: ExchangeType, policy: RateLimitPolicy) {
        // Create token bucket
        let bucket = Arc::new(Mutex::new(TokenBucket::new(
            policy.burst_capacity,
            policy.max_requests_per_second,
        )));
        
        // Create priority semaphores
        let mut semaphores = HashMap::new();
        for priority_level in 0..policy.priority_levels {
            let priority = match priority_level {
                0 => RequestPriority::Critical,
                1 => RequestPriority::High,
                _ => RequestPriority::Normal,
            };
            
            let capacity = match priority {
                RequestPriority::Critical => policy.max_requests_per_second / 2,
                RequestPriority::High => policy.max_requests_per_second / 3,
                RequestPriority::Normal => policy.max_requests_per_second / 6,
            };
            
            semaphores.insert(
                (exchange.clone(), priority),
                Arc::new(Semaphore::new(capacity as usize)),
            );
        }
        
        // Store configurations
        self.policies.write().await.insert(exchange.clone(), policy);
        self.buckets.write().await.insert(exchange.clone(), bucket);
        
        let mut priority_sems = self.priority_semaphores.write().await;
        for ((ex, prio), sem) in semaphores {
            priority_sems.insert((ex, prio), sem);
        }
    }
    
    /// Acquire permission to make a request
    pub async fn acquire_permit(
        &self,
        exchange: &ExchangeType,
        context: RequestContext,
    ) -> Result<RateLimitPermit> {
        // Check if we're in backoff
        if let Some(backoff_until) = self.backoff_until.read().await.get(exchange) {
            if Instant::now() < *backoff_until {
                let wait_time = backoff_until.duration_since(Instant::now());
                debug!("Exchange {} in backoff, waiting {:?}", 
                    format!("{:?}", exchange), wait_time);
                sleep(wait_time).await;
            }
        }
        
        // Get policy and bucket
        let policy = self.policies.read().await.get(exchange).cloned()
            .unwrap_or_default();
        
        let bucket = self.buckets.read().await.get(exchange).cloned()
            .ok_or_else(|| anyhow!("Exchange not configured: {:?}", exchange))?;
        
        // Get priority semaphore
        let semaphore = self.priority_semaphores.read().await
            .get(&(exchange.clone(), context.priority)).cloned()
            .ok_or_else(|| anyhow!("Priority semaphore not found"))?;
        
        // Acquire priority permit
        let _priority_permit = timeout(context.timeout, semaphore.acquire()).await
            .map_err(|_| anyhow!("Priority permit acquisition timeout"))?
            .map_err(|_| anyhow!("Priority semaphore closed"))?;
        
        // Calculate request weight
        let weight = policy.weight_multipliers
            .get(&context.endpoint)
            .copied()
            .unwrap_or(context.weight);
        
        // Try to consume tokens
        let mut bucket_guard = bucket.lock().await;
        
        if !bucket_guard.try_consume(weight) {
            // Calculate wait time
            let wait_time = bucket_guard.time_until_tokens(weight);
            
            if wait_time > context.timeout {
                return Err(anyhow!("Rate limit timeout exceeded"));
            }
            
            debug!("Rate limited, waiting {:?}", wait_time);
            drop(bucket_guard);
            sleep(wait_time).await;
            
            // Try again
            bucket_guard = bucket.lock().await;
            if !bucket_guard.try_consume(weight) {
                return Err(anyhow!("Failed to acquire rate limit permit after wait"));
            }
        }
        
        Ok(RateLimitPermit {
            exchange: exchange.clone(),
            weight,
            acquired_at: Instant::now(),
            _priority_permit,
        })
    }
    
    /// Report a rate limit violation
    pub async fn report_violation(
        &self,
        exchange: &ExchangeType,
        violation: RateLimitViolation,
    ) {
        let now = Instant::now();
        
        // Record violation
        self.violations.write().await
            .entry(exchange.clone())
            .or_insert_with(Vec::new)
            .push((now, violation.clone()));
        
        // Apply adaptive backoff
        let policy = self.policies.read().await.get(exchange).cloned()
            .unwrap_or_default();
        
        let backoff_duration = match violation {
            RateLimitViolation::ExceededLimit => policy.recovery_time,
            RateLimitViolation::ServerThrottling => {
                Duration::from_secs_f64(
                    policy.recovery_time.as_secs_f64() * policy.backoff_multiplier
                ).min(policy.max_backoff)
            },
            RateLimitViolation::IPBanned => policy.max_backoff,
            RateLimitViolation::WeightExceeded => policy.recovery_time / 2,
        };
        
        self.backoff_until.write().await.insert(
            exchange.clone(),
            now + backoff_duration,
        );
        
        warn!(
            "Rate limit violation for {:?}: {:?}, backing off for {:?}",
            exchange, violation, backoff_duration
        );
    }
    
    /// Get rate limit statistics
    pub async fn get_statistics(&self, exchange: &ExchangeType) -> RateLimitStatistics {
        let violations = self.violations.read().await.get(exchange).cloned()
            .unwrap_or_default();
        
        let now = Instant::now();
        let recent_violations = violations.iter()
            .filter(|(timestamp, _)| now.duration_since(*timestamp) < Duration::from_hours(1))
            .count();
        
        let backoff_until = self.backoff_until.read().await.get(exchange).copied();
        let is_throttled = backoff_until.map_or(false, |until| now < until);
        
        RateLimitStatistics {
            total_violations: violations.len(),
            recent_violations,
            is_throttled,
            backoff_until,
            last_violation: violations.last().map(|(timestamp, violation)| 
                (timestamp.clone(), violation.clone())
            ),
        }
    }
    
    /// Clean up old violation records
    pub async fn cleanup_old_records(&self) {
        let cutoff = Instant::now() - Duration::from_days(1);
        
        let mut violations = self.violations.write().await;
        for records in violations.values_mut() {
            records.retain(|(timestamp, _)| *timestamp > cutoff);
        }
        
        // Remove expired backoffs
        let now = Instant::now();
        self.backoff_until.write().await.retain(|_, until| now < *until);
    }
}

/// Rate limit permit
pub struct RateLimitPermit {
    exchange: ExchangeType,
    weight: u32,
    acquired_at: Instant,
    _priority_permit: tokio::sync::SemaphorePermit<'static>,
}

impl RateLimitPermit {
    pub fn exchange(&self) -> &ExchangeType {
        &self.exchange
    }
    
    pub fn weight(&self) -> u32 {
        self.weight
    }
    
    pub fn acquired_at(&self) -> Instant {
        self.acquired_at
    }
}

/// Rate limit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatistics {
    pub total_violations: usize,
    pub recent_violations: usize,
    pub is_throttled: bool,
    pub backoff_until: Option<Instant>,
    pub last_violation: Option<(Instant, RateLimitViolation)>,
}

/// Default policies for known exchanges
pub fn default_policies() -> HashMap<ExchangeType, RateLimitPolicy> {
    let mut policies = HashMap::new();
    
    // Binance
    policies.insert(
        ExchangeType::Centralized(super::core::CentralizedExchange::Binance),
        RateLimitPolicy {
            max_requests_per_second: 20,
            burst_capacity: 100,
            weight_multipliers: {
                let mut weights = HashMap::new();
                weights.insert("/api/v3/order".to_string(), 10);
                weights.insert("/api/v3/order/test".to_string(), 1);
                weights.insert("/api/v3/openOrders".to_string(), 40);
                weights.insert("/api/v3/allOrders".to_string(), 10);
                weights.insert("/api/v3/depth".to_string(), 50);
                weights
            },
            recovery_time: Duration::from_secs(60),
            backoff_multiplier: 1.5,
            max_backoff: Duration::from_secs(300),
            priority_levels: 3,
        },
    );
    
    // Coinbase
    policies.insert(
        ExchangeType::Centralized(super::core::CentralizedExchange::Coinbase),
        RateLimitPolicy {
            max_requests_per_second: 10,
            burst_capacity: 50,
            weight_multipliers: HashMap::new(),
            recovery_time: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(600),
            priority_levels: 3,
        },
    );
    
    // Bybit
    policies.insert(
        ExchangeType::Centralized(super::core::CentralizedExchange::Bybit),
        RateLimitPolicy {
            max_requests_per_second: 100,
            burst_capacity: 200,
            weight_multipliers: HashMap::new(),
            recovery_time: Duration::from_secs(45),
            backoff_multiplier: 1.8,
            max_backoff: Duration::from_secs(240),
            priority_levels: 3,
        },
    );
    
    policies
}

trait DurationExt {
    fn from_days(days: u64) -> Duration;
    fn from_hours(hours: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_days(days: u64) -> Duration {
        Duration::from_secs(days * 24 * 60 * 60)
    }
    
    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 60 * 60)
    }
}

