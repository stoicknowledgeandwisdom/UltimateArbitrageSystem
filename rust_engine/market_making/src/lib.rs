//! # Advanced Market Making Strategies
//!
//! Sophisticated market making algorithms for optimal liquidity provision.

pub mod spreads;
pub mod inventory;

use ultimate_core::*;
use thiserror::Error;

/// Market making errors
#[derive(Error, Debug)]
pub enum MarketMakingError {
    #[error("Invalid spread configuration")]
    InvalidSpread,
    
    #[error("Inventory limit exceeded")]
    InventoryLimitExceeded,
}

/// Market making result type
pub type MarketMakingResult<T> = Result<T, MarketMakingError>;

/// Market making engine
pub struct MarketMakingEngine {
    symbol: Symbol,
}

impl MarketMakingEngine {
    /// Create new market making engine
    pub fn new(symbol: Symbol) -> Self {
        Self { symbol }
    }
    
    /// Generate market making quotes
    pub fn generate_quotes(&self, mid_price: Price) -> MarketMakingResult<(Price, Price)> {
        let spread = Price::from_f64(0.001).unwrap(); // 0.1% spread
        let half_spread = Price::from_raw(spread.raw() / 2);
        
        let bid = mid_price.saturating_sub(half_spread);
        let ask = mid_price.saturating_add(half_spread);
        
        Ok((bid, ask))
    }
}

