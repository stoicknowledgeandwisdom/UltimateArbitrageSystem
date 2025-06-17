//! # Dynamic Spread Calculation
//!
//! Advanced spread calculation algorithms for market making.

use ultimate_core::*;
use crate::{MarketMakingError, MarketMakingResult};

/// Spread calculator
pub struct SpreadCalculator {
    base_spread: Price,
}

impl SpreadCalculator {
    /// Create new spread calculator
    pub fn new(base_spread: Price) -> Self {
        Self { base_spread }
    }
    
    /// Calculate dynamic spread based on market conditions
    pub fn calculate_spread(&self, volatility: f64) -> MarketMakingResult<Price> {
        let volatility_adjustment = volatility * 0.5; // 50% volatility adjustment
        let adjusted_spread = self.base_spread.to_f64() * (1.0 + volatility_adjustment);
        
        Price::from_f64(adjusted_spread).map_err(|_| MarketMakingError::InvalidSpread)
    }
}

impl Default for SpreadCalculator {
    fn default() -> Self {
        Self::new(Price::from_f64(0.001).unwrap()) // 0.1% default spread
    }
}

