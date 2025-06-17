//! # Ultra-High-Performance Market Data Processing
//!
//! High-frequency market data processing with microsecond precision.

use crate::primitives::*;
use crate::{Timestamp, CoreError, CoreResult};
use serde::{Deserialize, Serialize};

/// Market data tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    /// Trading symbol
    pub symbol: Symbol,
    /// Tick price
    pub price: Price,
    /// Tick volume
    pub volume: Quantity,
    /// Tick timestamp
    pub timestamp: Timestamp,
    /// Bid price
    pub bid: Option<Price>,
    /// Ask price
    pub ask: Option<Price>,
}

/// Market data processor
#[derive(Debug)]
pub struct MarketDataProcessor {
    /// Processor name
    name: String,
}

impl MarketDataProcessor {
    /// Create new market data processor
    pub fn new(name: String) -> Self {
        Self { name }
    }
    
    /// Process a market data tick
    pub fn process_tick(&self, tick: Tick) -> CoreResult<()> {
        // Ultra-fast tick processing logic would go here
        Ok(())
    }
    
    /// Get processor name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for MarketDataProcessor {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

